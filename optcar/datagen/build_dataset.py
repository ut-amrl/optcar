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


def builder_data_config(settings=None):
    """Shared GUI/CLI builder settings; output bins default to 10,000 windows."""
    if isinstance(settings, DataConfig):
        return settings
    return DataConfig(**{"shard_windows": 10000, **(settings or {})})


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
    data = builder_data_config(data)
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


class _BuildCancelled(Exception):
    pass


def _check_output_files(root, grouped, data):
    """Check the files this build writes; unrelated files can share the folder."""
    for (terrain, kind), entries in grouped.items():
        target = root / kind / terrain
        manifest = target / "manifest.json"
        provenance = {"inputs": entries, "calibrations": [e["calibration"] for e in entries if e["calibration"]]}
        written = set()
        counts = defaultdict(int)
        if manifest.exists():
            doc = json.loads(manifest.read_text())
            if (doc.get("config"), doc.get("source"), doc.get("terrain"), doc.get("provenance"),
                doc.get("process_windows", True), doc.get("flat_processed", False)) != (asdict(data), kind, terrain, provenance, True, True):
                raise ValueError(f"Inputs or settings changed for {target}; choose a new output folder for this source and terrain")
            written = {shard["path"] for shard in doc.get("processed_shards", [])}
            for shard in doc.get("processed_shards", []):
                counts[shard["split"]] += 1
        else:
            for relative in ("metadata/schema.json", "metadata/dataset_config.py"):
                path = target / relative
                if path.exists():
                    raise ValueError(f"Output would replace an existing file: {path}")
        for split in ("train", "val", "test"):
            windows = sum(entry["estimated_windows"] for entry in entries if entry["split"] == split)
            for index in range(counts[split] + math.ceil(windows / data.shard_windows)):
                relative = DatasetWriter.processed_filename(split, index, flat_processed=True)
                path = target / relative
                if relative not in written and (path.exists() or path.with_suffix(".bin.tmp").exists()):
                    raise ValueError(f"Output would replace an existing file: {path}")


def build_output(output, sources):
    """Always build <chosen folder>/processed/source/terrain/split."""
    return Path(output).expanduser().resolve() / "processed"


def load_files_parallel(entries, max_workers=16, cancelled=None):
    """Load bins concurrently while preserving source order.

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
    data = builder_data_config(data)
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
    identity = {"output_layout": "source_terrain_split_v1", "config": plan["config"],
                "bins": plan["bins"], "seed": seed, "fractions": list(fractions)}
    index = root / "dataset_build.json"
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".builder.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another dataset builder is writing this folder") from error
        if index.exists():
            previous = json.loads(index.read_text())
            if previous.get("output_layout") != identity["output_layout"]:
                raise ValueError("This folder contains the older nested output layout; use an empty output folder for the new layout")
            changed = [key for key in ("history_steps", "future_steps", "dt")
                       if previous["config"][key] != identity["config"][key]]
            if changed:
                raise ValueError(f"Training window settings changed ({', '.join(changed)}); choose a different output parent")
        grouped = defaultdict(list)
        for entry in plan["bins"]:
            grouped[entry["terrain"], entry["source"]].append(entry)
        _check_output_files(root, grouped, data)
        # Keep the first build record. Each source/terrain manifest validates
        # its own inputs and settings, allowing separate groups to share a root.
        if not index.exists():
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
            with DatasetWriter(root / kind / terrain, data, kind, terrain, provenance, flat_processed=True) as writer, \
                    closing(load_files_parallel(entries, workers, cancelled)) as loaded:
                for entry, rows in loaded:
                    if cancelled and cancelled():
                        return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
                    report(f"Loading {entry['name']}")
                    episode = _episode_name(entry, "batch")
                    if writer.contains(episode):
                        done_windows += entry["estimated_windows"]
                        completed += 1
                        report(f"Already saved {entry['name']}")
                        continue
                    # Reject an entire trajectory when any sample is invalid.
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
                    if pending and np.any(np.linalg.norm(rows[pending, :, 3:7], axis=-1) < 1e-6):
                        raise ValueError(f"Zero quaternion in {entry['name']}")

                    def processed_batches():
                        # Move all valid trajectories to the device, unfold windows,
                        # convert frames, and copy the processed batch to CPU.
                        if cancelled and cancelled():
                            raise _BuildCancelled()
                        if not pending:
                            return
                        report(f"Converting {entry['name']} · {len(pending)} trajectories · {device}")
                        batch = torch.from_numpy(np.array(rows[pending, ..., :15], copy=True)).to(device)
                        chunks = chunk_view(batch, data.window_steps, data.stride).contiguous().reshape(
                            -1, data.window_steps, 15)
                        with torch.no_grad():
                            packed = pack_windows(chunks[..., :13], chunks[..., 13:15], data.history_steps,
                                                  entry["vehicle_id"]).cpu()
                        del batch, chunks
                        # Drop windows with post-transform NaNs.
                        post_nan = torch.isnan(packed).any(dim=1).any(dim=1)
                        if post_nan.any():
                            packed = packed[~post_nan]
                        if cancelled and cancelled():
                            raise _BuildCancelled()
                        yield packed

                    def shard_progress(count, phase):
                        if cancelled and cancelled():
                            raise _BuildCancelled()
                        report(f"{phase} · {entry['name']}", count)
                    try:
                        saved = writer.write_window_batches(episode, processed_batches(), entry["split"],
                            entry["session"], entry["path"], list(rows.shape),
                            metadata={"input": entry["path"], "environments": pending}, progress=shard_progress)
                    except _BuildCancelled:
                        return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
                    windows += saved
                    done_windows += len(pending) * per_env
                    writer.commit()
                    completed += 1
                    report(f"Processed {entry['name']}")
                report("Writing final output bins")
                writer.flush_window_batches()
                writer.commit()
                report("Saved output bins")
        return {"dataset": str(root), "completed": completed, "total": plan["files"], "new_windows": windows, "skipped": skipped}
