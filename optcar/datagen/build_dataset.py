"""Assemble selected real/synthetic raw bins using OptCar's existing FKD writer."""
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from optcar.dataset import DataConfig, DatasetWriter, chunk_view, pack_windows, session_split
from optcar.datagen.raw_bins import describe_bin, list_bins, load_bin, read_metadata
from optcar.utils.file_io import safe_name, write_json
from optcar.utils.devices import available_devices, resolve_device


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
    records, errors, seen, sessions, cache = [], [], set(), {}, {}
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
                # A build may live inside its inputs. Ignore its output bins on
                # subsequent scans, while retaining original inputs under it.
                if doc and "inputs" in doc.get("provenance", {}):
                    if any(item["path"] == str(file) for item in doc["provenance"]["inputs"]):
                        doc, known = None, None
                    else:
                        continue
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
                # Saved synthetic IDs derive from settings and run index, so
                # separate collection folders can repeat them. File paths above
                # identify duplicate inputs; retain saved session splits below.
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


def _episode_name(entry, environment):
    return "bin_" + hashlib.sha256(entry["path"].encode()).hexdigest()[:16] + f"_e{environment}"


def _check_output_files(root, grouped, data):
    """Check the files this build writes; unrelated files can share the folder."""
    for (terrain, kind), entries in grouped.items():
        target = root / terrain / kind
        manifest = target / "manifest.json"
        provenance = {"inputs": entries, "calibrations": [e["calibration"] for e in entries if e["calibration"]]}
        episodes = {}
        if manifest.exists():
            doc = json.loads(manifest.read_text())
            if (doc.get("config"), doc.get("source"), doc.get("terrain"), doc.get("provenance"),
                doc.get("process_windows", True)) != (asdict(data), kind, terrain, provenance, True):
                raise ValueError(f"Dataset files already exist at {target} with different inputs or settings")
            episodes = doc["episodes"]
        else:
            for relative in ("metadata/schema.json", "metadata/dataset_config.py",
                             "processed/metadata/schema.json", "processed/metadata/dataset_config.py"):
                path = target / relative
                if path.exists():
                    raise ValueError(f"Output would replace an existing file: {path}")
        for entry in entries:
            for environment in range(entry["num_envs"]):
                name = _episode_name(entry, environment)
                if name in episodes:
                    continue
                windows = entry["estimated_windows"] // entry["num_envs"]
                files = [target / "raw" / f"{name}.bin"]
                files += [target / "processed/data" / f"{name}_{offset:08d}.bin"
                          for offset in range(0, windows, data.shard_windows)]
                for path in files:
                    if path.exists() or path.with_suffix(".bin.tmp").exists():
                        raise ValueError(f"Output would replace an existing file: {path}")


def build_output(output, sources):
    root = Path(output).expanduser().resolve()
    for source in sources:
        path = Path(source["path"]).expanduser().resolve()
        base = path.parent if path.is_file() else path
        if base.is_relative_to(root):
            return root / "processed"
    return root


def load_files_parallel(entries, max_workers=16, cancelled=None):
    """Adapt WheeledLab RawDatasetProcessor.load_files_parallel's thread loading.

    Use the shared raw reader, materializing bytes in the loading threads.
    Keep at most max_workers files prefetched and retain source order/IDs rather
    than concatenating the entire corpus (sources can have different lengths).
    Frame conversion and writing remain in the existing GPU/writer path.
    """
    def load_single(entry):
        rows, _ = load_bin(Path(entry["path"]), entry["columns"], entry["num_envs"], None)
        return np.array(rows, dtype=np.float32, copy=True)

    remaining = iter(entries)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = deque()
        try:
            for _ in range(max_workers):
                if cancelled and cancelled():
                    break
                entry = next(remaining, None)
                if entry is None:
                    break
                pending.append((entry, executor.submit(load_single, entry)))
            while pending:
                if cancelled and cancelled():
                    break
                entry, future = pending.popleft()
                rows = future.result()
                following = next(remaining, None)
                if following is not None and not (cancelled and cancelled()):
                    pending.append((following, executor.submit(load_single, following)))
                yield entry, rows
        finally:
            for _, future in pending:
                future.cancel()


def build_dataset(output, sources, data=None, seed=42, fractions=(.8, .1, .1), progress=None, cancelled=None, device=None, workers=16):
    data = data or DataConfig()
    if type(workers) is not int or workers < 1:
        raise ValueError("File loading workers must be a positive integer")
    device = resolve_device(device if device is not None else available_devices()["default"])
    started = time.monotonic()
    if progress:
        progress({"completed": 0, "total": 0, "new_windows": 0, "current": "Inspecting selected bins", "device": device})
    plan = inspect_sources(sources, data, seed, fractions)
    if plan["errors"]:
        raise ValueError("Fix source errors before building: " + "; ".join(plan["errors"][:5]))
    if not plan["files"]:
        raise ValueError("No usable raw bins selected")
    root = build_output(output, sources)
    if root.exists() and not root.is_dir():
        raise ValueError("Dataset output must be a folder")
    identity = {"config": plan["config"], "bins": plan["bins"], "seed": seed, "fractions": list(fractions)}
    index = root / "dataset_build.json"
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".builder.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another dataset builder is writing this folder") from error
        if index.exists() and json.loads(index.read_text()) != identity:
            raise ValueError("Inputs or settings changed; choose a new output folder")
        grouped = defaultdict(list)
        for entry in plan["bins"]:
            grouped[entry["terrain"], entry["source"]].append(entry)
        _check_output_files(root, grouped, data)
        write_json(index, identity)
        completed, windows, skipped, done_windows = 0, 0, [], 0
        total_windows = sum(plan["windows"].values())
        def report(current, within=0):
            if progress:
                progress({"completed": done_windows + within, "total": total_windows, "new_windows": windows,
                          "files_completed": completed, "files_total": plan["files"], "current": current,
                          "device": device, "workers": workers, "elapsed_s": time.monotonic() - started})
        for (terrain, kind), entries in grouped.items():
            provenance = {"inputs": entries, "calibrations": [e["calibration"] for e in entries if e["calibration"]]}
            report(f"Loading bins · {workers} file loading workers")
            with DatasetWriter(root / terrain / kind, data, kind, terrain, provenance) as writer, \
                    closing(load_files_parallel(entries, workers, cancelled)) as loaded:
                for entry, rows in loaded:
                    if cancelled and cancelled():
                        return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
                    report(f"Loading {entry['name']}")
                    # Same whole-environment invalid-sample policy as WheeledLab.
                    valid = np.isfinite(rows).all(axis=(1, 2))
                    per_env = entry["estimated_windows"] // entry["num_envs"]
                    pending = []
                    for environment in range(entry["num_envs"]):
                        if not valid[environment]:
                            skipped.append(f"{entry['name']} environment {environment}: nonfinite trajectory")
                            done_windows += per_env
                        elif writer.contains(_episode_name(entry, environment)):
                            done_windows += per_env
                        else:
                            pending.append(environment)
                    # WheeledLab moves a batch to the device, unfolds windows there,
                    # and keeps frame conversion in Torch until the final disk write.
                    group_size = max(1, data.shard_windows // per_env)
                    for start in range(0, len(pending), group_size):
                        if cancelled and cancelled():
                            return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
                        group = pending[start:start + group_size]
                        report(f"Converting {entry['name']} · {len(group)} trajectories · {device}")
                        batch = torch.from_numpy(np.array(rows[group, ..., :15], copy=True)).to(device)
                        packed = None
                        if per_env <= data.shard_windows:
                            chunks = chunk_view(batch, data.window_steps, data.stride).contiguous().reshape(-1, data.window_steps, 15)
                            with torch.no_grad():
                                packed = pack_windows(chunks[..., :13], chunks[..., 13:15], data.history_steps,
                                                      entry["vehicle_id"]).reshape(len(group), per_env, data.window_steps, 42)
                            del chunks
                        for index_in_group, environment in enumerate(group):
                            def shard_progress(count, total, phase):
                                report(f"{phase} · {entry['name']} · trajectory {environment + 1}/{entry['num_envs']}", count)
                            windows += writer.write_episode(_episode_name(entry, environment), rows[environment, :, :13],
                                rows[environment, :, 13:15], entry["split"], session=entry["session"],
                                vehicle_id=entry["vehicle_id"], metadata={"input": entry["path"], "environment": environment},
                                device=device, raw_tensor=batch[index_in_group],
                                packed_windows=None if packed is None else packed[index_in_group], progress=shard_progress)
                            done_windows += per_env
                        del batch, packed
                    writer.commit()
                    completed += 1
                    report(f"Saved {entry['name']}")
        return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
