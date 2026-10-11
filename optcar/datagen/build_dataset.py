"""Assemble selected real/synthetic raw bins using OptCar's existing FKD writer."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from optcar.dataset import DataConfig, DatasetWriter, session_split
from optcar.datagen.raw_bins import describe_bin, list_bins, load_bin, read_metadata
from optcar.utils.file_io import safe_name, write_json


def _manifest_episode(path, cache):
    for parent in path.parents:
        manifest = parent / "manifest.json"
        if not manifest.is_file():
            continue
        if manifest not in cache:
            doc = json.loads(manifest.read_text())
            index = {str((parent / entry["raw"]).resolve()): (name, entry)
                     for name, entry in doc.get("episodes", {}).items() if entry.get("raw")}
            cache[manifest] = doc, index
        doc, index = cache[manifest]
        return doc, index.get(str(path))
    return None, None


def inspect_sources(sources, data=None, seed=42, fractions=(.8, .1, .1)):
    """Read-only plan. Splits are assigned by session, never by overlapping window."""
    data = data or DataConfig()
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    session_split("validate", seed, fractions)
    if not sources:
        raise ValueError("Add at least one raw bin source")
    records, errors, seen, sessions, cache, synthetic_sessions = [], [], set(), {}, {}, set()
    for source_index, spec in enumerate(sources):
        path = Path(spec["path"]).expanduser().resolve()
        terrain, kind = safe_name(spec["terrain"]), spec["source"]
        if kind not in ("real", "sim"):
            raise ValueError("Source type must be real or sim")
        requested_split = spec.get("split", "auto")
        if requested_split not in ("auto", "train", "val", "test"):
            raise ValueError("Invalid source split")
        options = {key: spec[key] for key in ("dt", "columns", "num_envs") if spec.get(key) is not None}
        candidates = list_bins(path)
        selected = spec.get("selected_paths")
        if selected is not None:
            selected = {str(Path(file).expanduser().resolve()) for file in selected}
            outside = selected - {str(file) for file in candidates}
            if outside:
                raise ValueError("Selected bins must belong to their source folder")
            candidates = [file for file in candidates if str(file) in selected]
        for file in candidates:
            entry = {"path": str(file), "name": file.name,
                     "relative_path": str(file.relative_to(path.parent if path.is_file() else path)),
                     "source_index": source_index, "source": kind, "terrain": terrain}
            try:
                doc, known = _manifest_episode(file, cache)
                # Existing generated collections may include processed shards and raw copies.
                if doc and not known:
                    continue
                meta = read_metadata(file)
                if meta.get("is_auxiliary"):
                    continue
                file_options = dict(options)
                if known:
                    if "dt" in options and not math.isclose(options["dt"], doc["config"]["dt"], abs_tol=1e-9):
                        raise ValueError("The saved collection timestep cannot be overridden")
                    shape = known[1]["raw_shape"]
                    environments = shape[0] if len(shape) == 3 else 1
                    if options.get("columns", 15) != 15 or options.get("num_envs", environments) != environments:
                        raise ValueError(f"Saved layout is 15 columns and {environments} environment(s)")
                    file_options = {"dt": doc["config"]["dt"], "columns": 15, "num_envs": environments, **file_options}
                    if doc["source"] != kind or doc["terrain"] != terrain:
                        raise ValueError("Source type/terrain do not match the saved collection")
                description = describe_bin(file, **file_options)
                entry.update(description)
                entry["source"] = kind
                if not math.isclose(entry["dt"], data.dt, rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Timestep {entry['dt']}s differs from dataset timestep {data.dt}s")
                if file in seen:
                    raise ValueError("This bin is already selected in another source")
                seen.add(file)
                session = known[1]["session"] if known else "bag_" + hashlib.sha256(
                    str(Path(meta.get("source") or file).expanduser().resolve()).encode()).hexdigest()[:16]
                split = known[1]["split"] if known else (
                    session_split(session, seed, fractions) if requested_split == "auto" else requested_split)
                if known and requested_split != "auto" and requested_split != split:
                    raise ValueError("Saved episode splits are preserved; choose Automatic for this source")
                identity = (kind, session)
                if kind == "sim" and identity in synthetic_sessions:
                    raise ValueError("This synthetic episode is already selected in another source")
                if kind == "sim":
                    synthetic_sessions.add(identity)
                if identity in sessions and sessions[identity] != split:
                    raise ValueError("One recording/session cannot appear in different splits")
                sessions[identity] = split
                windows = max(0, (entry["samples"] - data.window_steps) // data.stride + 1)
                if not windows:
                    raise ValueError("Recording is shorter than history + future")
                calibration = doc.get("provenance", {}).get("calibration", {}) if doc else {}
                entry.update(session=session, split=split, estimated_windows=windows * entry["num_envs"],
                             options=file_options, calibration=calibration,
                             vehicle_id=int(spec["vehicle_id"]) if spec.get("vehicle_id") is not None else len(records),
                             size=file.stat().st_size, modified_ns=file.stat().st_mtime_ns)
            except (ValueError, OSError, KeyError, TypeError) as error:
                entry["error"] = str(error)
                errors.append(f"{file.name}: {error}")
            records.append(entry)
    for entry in records:
        calibration = entry.get("calibration", {})
        if calibration.get("session") and (calibration.get("split") != "train" or
                sessions.get(("real", calibration["session"]), "train") != "train"):
            entry["error"] = "The DBM preset was calibrated on a held-out recording"
            errors.append(entry["error"])
    valid = [entry for entry in records if not entry.get("error")]
    totals = {split: sum(e["estimated_windows"] for e in valid if e["split"] == split)
              for split in ("train", "val", "test")}
    return {"bins": records, "errors": errors, "files": len(valid), "windows": totals,
            "config": asdict(data), "seed": seed, "fractions": list(fractions)}


def build_dataset(output, sources, data=None, seed=42, fractions=(.8, .1, .1), progress=None, cancelled=None):
    data = data or DataConfig()
    plan = inspect_sources(sources, data, seed, fractions)
    if plan["errors"]:
        raise ValueError("Fix source errors before building: " + "; ".join(plan["errors"][:5]))
    if not plan["files"]:
        raise ValueError("No usable raw bins selected")
    root = Path(output).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("Dataset output must be a folder")
    for source in sources:
        source_path = Path(source["path"]).expanduser().resolve()
        base = source_path.parent if source_path.is_file() else source_path
        if root.is_relative_to(base) or base.is_relative_to(root):
            raise ValueError("Choose an output folder separate from the input folders")
    identity = {"config": plan["config"], "bins": plan["bins"], "seed": seed, "fractions": list(fractions)}
    index = root / "dataset_build.json"
    if root.exists() and not index.exists() and any(root.iterdir()):
        raise ValueError("Output is not an empty folder or an existing matching dataset build")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".builder.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another dataset builder is writing this folder") from error
        if index.exists() and json.loads(index.read_text()) != identity:
            raise ValueError("Inputs or settings changed; choose a new output folder")
        write_json(index, identity)
        grouped = defaultdict(list)
        for entry in plan["bins"]:
            grouped[entry["terrain"], entry["source"]].append(entry)
        completed, windows, started, skipped = 0, 0, time.monotonic(), []
        for (terrain, kind), entries in grouped.items():
            provenance = {"inputs": entries, "calibrations": [e["calibration"] for e in entries if e["calibration"]]}
            with DatasetWriter(root / terrain / kind, data, kind, terrain, provenance) as writer:
                for entry in entries:
                    if cancelled and cancelled():
                        return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
                    for environment in range(entry["num_envs"]):
                        rows, _ = load_bin(Path(entry["path"]), entry["columns"], entry["num_envs"], environment)
                        # WheeledLab processing_sim.py skips the entire environment
                        # trajectory if any sample has NaNs; do not invent gap segments.
                        if not np.isfinite(rows).all():
                            skipped.append(f"{entry['name']} environment {environment}: nonfinite trajectory")
                            continue
                        name = "bin_" + hashlib.sha256(entry["path"].encode()).hexdigest()[:16] + f"_e{environment}"
                        if writer.contains(name):
                            continue
                        windows += writer.write_episode(name, rows[:, :13], rows[:, 13:15],
                            entry["split"], session=entry["session"], vehicle_id=entry["vehicle_id"],
                            metadata={"input": entry["path"], "environment": environment})
                    writer.commit()
                    completed += 1
                    if progress:
                        progress({"completed": completed, "total": plan["files"], "new_windows": windows,
                                  "current": entry["name"], "elapsed_s": time.monotonic() - started})
        return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
