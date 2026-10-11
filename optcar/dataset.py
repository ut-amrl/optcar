"""FKD binary format, episode writer, memory-mapped loader, and terrain sampler."""
from bisect import bisect_right
from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import operator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from optcar.utils.filter import quaternion_continuity_filter_batch
from optcar.utils.file_io import atomic_text, safe_name, write_json
from optcar.utils.frame_transforms import compute_delta_frame_representation, horizon_world_to_body_frame
from optcar.utils.math import qmultiply_torch, qrotate_torch


@dataclass
class DataConfig:
    history_steps: int = 250
    future_steps: int = 50
    dt: float = 0.02
    stride: int = 10
    shard_windows: int = 1024

    def __post_init__(self):
        if self.history_steps < 2 or self.future_steps < 1 or self.dt <= 0 or not math.isfinite(self.dt) or self.stride < 1 or self.shard_windows < 1:
            raise ValueError("Invalid history, future, timestep, stride, or shard size")
        if any(not isinstance(value, int) for value in (self.history_steps, self.future_steps, self.stride, self.shard_windows)):
            raise ValueError("History, future, stride, and shard size must be integers")

    @property
    def window_steps(self):
        return self.history_steps + self.future_steps


SCHEMA = {"version": 1, "dtype": "float32", "features": 42,
          "layout": ["world_state:13", "body_state:13", "delta_state:13", "commands:2", "vehicle_id:1"],
          "quaternion_order": "wxyz", "commands": ["velocity_m_s", "steering_rad"],
          "action_timing": "action[t] drives state[t] to state[t+1]",
          "future_steps": "number of transitions after current state"}

RAW_SCHEMA = {"version": 1, "dtype": "float32", "features": 15,
              "layout": ["world_state:13", "commands:2"], "quaternion_order": "wxyz",
              "action_timing": "action[t] drives state[t] to state[t+1]"}


def pack_windows(states, actions, history_steps, vehicle_id=0):
    """states/actions: [B,T,13/2]; body features anchored to current state."""
    # Same per-window continuity and frame conversion as WheeledLab's
    # DatasetProcessor._process_chunks_to_frames_torch (base.py).
    states = torch.as_tensor(states, dtype=torch.float32).clone()
    actions = torch.as_tensor(actions, dtype=torch.float32, device=states.device)
    states[:, :, 3:7] = quaternion_continuity_filter_batch(
        states[:, :, 3:7], batch_dim=0, time_dim=1, inplace=True)
    current = states[:, history_steps - 1]
    pos, quat, vel, ang = horizon_world_to_body_frame(states, current[:, :3], current[:, 3:7])
    quat = quaternion_continuity_filter_batch(quat, batch_dim=0, time_dim=1, inplace=True)
    body = torch.cat((pos, quat, vel, ang), -1)
    delta = compute_delta_frame_representation(states)
    car = states.new_full((*states.shape[:2], 1), float(vehicle_id))
    return torch.cat((states, body, delta, actions, car), -1)


def chunk_view(data, horizon, stride):
    """WheeledLab DatasetProcessor.generate_chunks_torch, before flattening.

    Keep the unfolded [environment, window, time, feature] view until a bounded
    processing batch is selected, so long recordings don't exhaust GPU memory.
    """
    return data.unfold(dimension=1, size=horizon, step=stride).permute(0, 1, 3, 2)


def model_sample(window, history_steps):
    """Use only historical states; future states are loss targets exclusively."""
    h = history_steps
    body = window[:, 13:26]
    actions = window[:, 39:41]
    # History contains observed transitions, ending at the current state.
    history = torch.cat((window[1:h, 26:39], actions[:h - 1]), -1)
    return {"history": history, "current": body[h - 1],
            "commands": actions[h - 1:-1], "target_delta": window[h:, 26:33],
            "target_pose": body[h:, :7], "origin": window[h - 1, :7]}


def integrate_deltas(deltas, origin=None):
    """Differentiable local-frame SE(3) integration, returning future poses."""
    b = deltas.shape[0]
    if origin is None:
        position = deltas.new_zeros((b, 3))
        quaternion = deltas.new_zeros((b, 4))
        quaternion[:, 0] = 1
    else:
        position, quaternion = origin[:, :3], origin[:, 3:7]
    poses = []
    for step in deltas.unbind(1):
        position = position + qrotate_torch(quaternion, step[:, :3])
        quaternion = F.normalize(qmultiply_torch(quaternion, step[:, 3:7]), dim=-1)
        poses.append(torch.cat((position, quaternion), -1))
    return torch.stack(poses, 1)


def session_split(session, seed=42, fractions=(0.8, 0.1, 0.1)):
    if len(fractions) != 3 or any(not math.isfinite(x) or x < 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Split fractions must be three nonnegative numbers summing to one")
    value = int(hashlib.sha256(f"{seed}:{session}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < fractions[0] else "val" if value < sum(fractions[:2]) else "test"


class DatasetWriter:
    """One active writer per dataset. Manifest commits make restarts idempotent."""
    def __init__(self, root, config, source, terrain, provenance=None, process_windows=True, flat_processed=False):
        self.root, self.config = Path(root), config
        self.process_windows = process_windows
        self.flat_processed = flat_processed
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".writer.lock").open("a")
        import fcntl
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError(f"Another writer is active in {self.root}")
        self.path = self.root / "manifest.json"
        try:
            if self.path.exists():
                self.manifest = json.loads(self.path.read_text())
                expected = (asdict(config), source, terrain, provenance or {})
                actual = tuple(self.manifest[k] for k in ("config", "source", "terrain", "provenance"))
                if actual != expected:
                    raise ValueError("Dataset configuration changed; choose a new output directory")
                if self.manifest.get("process_windows", True) != process_windows:
                    raise ValueError("Raw generation and dataset building need separate output directories")
                if self.manifest.get("flat_processed", False) != flat_processed:
                    raise ValueError("Dataset output layout changed; choose a new output directory")
            else:
                self.manifest = {"schema": SCHEMA if process_windows else RAW_SCHEMA,
                                 "process_windows": process_windows, "config": asdict(config), "source": source,
                                 "terrain": terrain, "provenance": provenance or {}, "episodes": {}}
                if flat_processed:
                    self.manifest["flat_processed"] = True
                write_json(self.path, self.manifest)
            self.session_splits = {e["session"]: e["split"] for e in self.manifest["episodes"].values()}
            self.dirty = False
            self.window_carryover = {split: None for split in ("train", "val", "test")}
            self.window_allocations = {split: deque() for split in self.window_carryover}
            self.batch_indices = {split: sum(s["split"] == split for s in self.manifest.get("processed_shards", []))
                                  for split in self.window_carryover}
        except Exception:
            self.lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.flush_window_batches()
            self.commit()
        finally:
            self.lock.close()

    def commit(self):
        """Commit completed episodes together, avoiding metadata rewrites per episode."""
        if self.dirty:
            write_json(self.path, self.manifest)
            self._metadata()
            self.dirty = False

    def contains(self, episode):
        entry = self.manifest["episodes"].get(episode)
        return entry is not None and entry.get("stored_windows", entry["windows"]) == entry["windows"]

    @staticmethod
    def processed_filename(split, index, flat_processed=False):
        folder = {"val": "valid"}.get(split, split) if flat_processed else f"processed/data/{split}"
        return f"{folder}/processed_{index:010d}.bin"

    def _save_window_batch(self, split, values, progress=None):
        """WheeledLab save_processed_batch plus OptCar's split/manifest metadata."""
        index = self.batch_indices[split]
        filename = self.processed_filename(split, index, self.flat_processed)
        path = self.root / filename
        if path.exists() or path.with_suffix(".bin.tmp").exists():
            raise ValueError(f"Output would replace an existing file: {path}")
        if progress:
            progress("Writing windows")
        try:
            self._binary(filename, values.numpy())
        except BaseException:
            path.unlink(missing_ok=True)
            path.with_suffix(".bin.tmp").unlink(missing_ok=True)
            raise
        self.manifest.setdefault("processed_shards", []).append(
            {"path": filename, "shape": list(values.shape), "split": split})
        self.batch_indices[split] += 1
        remaining = len(values)
        allocations = self.window_allocations[split]
        while remaining:
            episode, count = allocations[0]
            take = min(remaining, count)
            self.manifest["episodes"][episode]["stored_windows"] += take
            remaining -= take
            if take == count:
                allocations.popleft()
            else:
                allocations[0] = (episode, count - take)
        self.dirty = True
        if progress:
            progress("Saved windows")

    def flush_window_batches(self):
        """Only the last output bin of each terrain/source/split may be short."""
        for split, carryover in self.window_carryover.items():
            if carryover is not None:
                self._save_window_batch(split, carryover)
                self.window_carryover[split] = None
                self.commit()

    def write_window_batches(self, episode, batches, split, session, raw_path, raw_shape, metadata=None,
                             progress=None):
        """Carry input windows forward with WheeledLab's _ingest_processed logic.

        One carryover per split keeps recordings in their existing split. The
        original raw bin stays in place. Shared output shards are listed once;
        episode saved counts let interrupted builds skip their stored prefixes.
        """
        safe_name(episode)
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if session in self.session_splits and self.session_splits[session] != split:
            raise ValueError("A session cannot appear in multiple dataset splits")
        if self.contains(episode):
            return 0
        episodes = self.manifest["episodes"]
        previous = episodes.get(episode)
        stored = previous.get("stored_windows", 0) if previous else 0
        carryover = self.window_carryover[split]
        allocations = deque(self.window_allocations[split])
        stored_counts = {name: episodes[name]["stored_windows"] for name, _ in allocations}
        shard_start = len(self.manifest.get("processed_shards", []))
        batch_start = self.batch_indices[split]
        received, skip = 0, stored
        size = self.config.shard_windows
        episodes[episode] = {"split": split, "session": session,
            "raw": str(Path(raw_path).resolve()), "raw_shape": list(raw_shape), "shards": [],
            "windows": 0, "stored_windows": stored, "metadata": metadata or {}}

        def save(values):
            def report(phase):
                if progress:
                    progress(max(0, episodes[episode]["stored_windows"] - stored), phase)
            self._save_window_batch(split, values, report)

        try:
            for processed in batches:
                processed = processed.detach().to(device="cpu", dtype=torch.float32)
                if processed.ndim != 3 or tuple(processed.shape[1:]) != (self.config.window_steps, 42):
                    raise ValueError("Expected processed windows [B,history+future,42]")
                received += len(processed)
                episodes[episode]["windows"] = received
                skipped = min(skip, len(processed))
                processed = processed[skipped:]
                skip -= skipped
                if not len(processed):
                    continue
                self.window_allocations[split].append((episode, len(processed)))
                # Copied from WheeledLab RawDatasetProcessor._ingest_processed:
                # fill the previous remainder, save full batches, retain the tail.
                current = self.window_carryover[split]
                if current is not None:
                    needed = size - len(current)
                    if len(processed) < needed:
                        extended = torch.empty((len(current) + len(processed), *processed.shape[1:]),
                                               dtype=processed.dtype, device="cpu")
                        extended[:len(current)].copy_(current)
                        extended[len(current):].copy_(processed)
                        self.window_carryover[split] = extended
                        continue
                    to_save = torch.empty((size, *processed.shape[1:]), dtype=processed.dtype, device="cpu")
                    to_save[:len(current)].copy_(current)
                    to_save[len(current):].copy_(processed[:needed])
                    save(to_save)
                    processed = processed[needed:]
                    self.window_carryover[split] = None
                while len(processed) >= size:
                    save(processed[:size])
                    processed = processed[size:]
                self.window_carryover[split] = processed if len(processed) else None
            if skip:
                raise ValueError("Previously stored windows exceed this input's window count")
        except BaseException:
            # Roll back this input only. Earlier inputs' carried windows remain
            # available for the normal final flush on cancellation or failure.
            for shard in self.manifest.get("processed_shards", [])[shard_start:]:
                (self.root / shard["path"]).unlink(missing_ok=True)
            del self.manifest.get("processed_shards", [])[shard_start:]
            self.batch_indices[split] = batch_start
            self.window_carryover[split] = carryover
            self.window_allocations[split] = allocations
            for name, count in stored_counts.items():
                episodes[name]["stored_windows"] = count
            if previous is None:
                del episodes[episode]
            else:
                episodes[episode] = previous
            raise
        self.session_splits[session] = split
        self.dirty = True
        return received - stored

    def write_episode(self, episode, states, actions, split, session=None, vehicle_id=0, metadata=None,
                      device="cpu", packed_windows=None, raw_tensor=None, progress=None):
        safe_name(episode)
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        session = session or episode
        if session in self.session_splits and self.session_splits[session] != split:
            raise ValueError("A session cannot appear in multiple dataset splits")
        if self.contains(episode):
            return 0
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 13 or actions.shape not in ((len(states) - 1, 2), (len(states), 2)):
            raise ValueError("Expected states [T,13] and commands [T-1,2] or [T,2]")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite episode values")
        norms = np.linalg.norm(states[:, 3:7], axis=-1)
        if np.any(norms < 1e-6):
            raise ValueError("Zero quaternion in episode")
        states = states.copy()
        # Preserve recorded quaternion values, as WheeledLab processing does.
        if len(actions) == len(states) - 1:
            actions = np.concatenate((actions, np.zeros((1, 2), np.float32)))
        raw_file = f"raw/{episode}.bin"
        self._binary(raw_file, np.concatenate((states, actions), -1))
        shards, count = [], 0
        cfg = self.config
        starts = range(0, len(states) - cfg.window_steps + 1, cfg.stride) if self.process_windows else range(0)
        chunks = None
        if starts and packed_windows is None:
            rows = raw_tensor if raw_tensor is not None else torch.from_numpy(
                np.concatenate((states, actions), -1)).to(device)
            chunks = chunk_view(rows[None, ..., :15], cfg.window_steps, cfg.stride)[0]
        for offset in range(0, len(starts), cfg.shard_windows):
            if progress:
                progress(offset, len(starts), "Converting frames")
            if packed_windows is None:
                selected = chunks[offset:offset + cfg.shard_windows].contiguous()
                with torch.no_grad():
                    packed = pack_windows(selected[..., :13], selected[..., 13:15], cfg.history_steps, vehicle_id)
            else:
                packed = packed_windows[offset:offset + cfg.shard_windows]
            packed = packed.detach().cpu().numpy()
            if progress:
                progress(offset, len(starts), "Writing windows")
            folder = {"val": "valid"}.get(split, split) if self.flat_processed else "processed/data"
            filename = f"{folder}/{episode}_{offset:08d}.bin"
            self._binary(filename, packed)
            shards.append({"path": filename, "shape": list(packed.shape)})
            count += len(packed)
            if progress:
                progress(count, len(starts), "Saved windows")
        self.manifest["episodes"][episode] = {"split": split, "session": session,
            "raw": raw_file, "raw_shape": [len(states), 15], "shards": shards,
            "windows": count, "metadata": metadata or {}}
        self.session_splits[session] = split
        self.dirty = True
        return count

    def _binary(self, filename, values):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".bin.tmp")
        with temporary.open("wb") as stream:
            np.ascontiguousarray(values, dtype=np.float32).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _metadata(self):
        cfg = self.config
        if self.manifest["provenance"].get("pipeline") == "wheeledlab_batched_v1":
            return  # The copied DataCollector writes the full WheeledLab metadata.
        if not self.flat_processed:
            write_json(self.root / "metadata/schema.json", RAW_SCHEMA)
            atomic_text(self.root / "metadata/dataset_config.py",
                        f"class DatasetConfig:\n    control_dt = {cfg.dt!r}\n    feature_dim = 15\n    num_envs = 1\n")
        if not self.process_windows:
            return
        shards = [s for e in self.manifest["episodes"].values() for s in e["shards"]]
        shards += self.manifest.get("processed_shards", [])
        counts = [s["shape"][0] for s in shards]
        # FKD's future_horizon includes the current state; ours counts transitions.
        fields = {"history_horizon": cfg.history_steps, "future_horizon": cfg.future_steps + 1,
                  "total_horizon": cfg.window_steps, "chunk_horizon": cfg.window_steps,
                  "feature_dim": 42, "data_dtype": "float32", "control_dt": cfg.dt,
                  "sim_dt": cfg.dt, "decimation": 1, "stride": cfg.stride,
                  "num_files": len(counts), "total_chunks": sum(counts), "chunks_per_file_list": counts}
        text = '"""FKD-compatible dimensions; use manifest.json for session-disjoint splits."""\n'
        text += "class ProcessedDatasetConfig:\n" + "".join(f"    {k} = {v!r}\n" for k, v in fields.items())
        metadata = self.root / ("metadata" if self.flat_processed else "processed/metadata")
        atomic_text(metadata / "dataset_config.py", text)
        write_json(metadata / "schema.json", {**SCHEMA, **fields})


class FKDDataset(Dataset):
    def __init__(self, roots, config, split="train", sources=None, allow_wheeledlab=False):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if split not in ("train", "val", "test", "all"):
            raise ValueError("split must be train, val, test, or all")
        roots = [Path(root).expanduser().resolve() for root in roots]
        self.config, self.shards, self.ends = config, [], []
        self.cache = OrderedDict()
        self.dataset_paths = roots
        self.dataset_file_mapping, self.shard_splits = [], []
        self._seen_files = set()
        documents = {}
        sessions = {}
        calibrations = set()
        for dataset_idx, root in enumerate(roots):
            if not root.is_dir():
                raise ValueError(f"Dataset directory does not exist: {root}")
            manifests = [root / "manifest.json"] if (root / "manifest.json").exists() else sorted(root.rglob("manifest.json"))
            if not manifests:
                # A user can select processed/, data/, or one saved split.
                # Keep the ancestor's declared dimensions and recorded splits.
                for parent in root.parents:
                    candidate = parent / "manifest.json"
                    if not candidate.is_file():
                        continue
                    doc = json.loads(candidate.read_text())
                    if "episodes" in doc and doc.get("schema", {}).get("features") == 42:
                        documents[candidate] = doc
                        manifests = [candidate]
                        break
            for path in manifests:
                if path not in documents:
                    documents[path] = json.loads(path.read_text())
                doc = documents[path]
                if "schema" not in doc or "episodes" not in doc:
                    continue
                if sources and doc["source"] not in sources:
                    continue
                if allow_wheeledlab and not doc.get("process_windows", True):
                    continue
                if doc["schema"].get("features") != 42 or doc["schema"].get("dtype") != "float32":
                    raise ValueError(f"Unsupported feature layout in {path}")
                if doc["schema"].get("quaternion_order", "wxyz") != "wxyz":
                    raise ValueError(f"Unsupported quaternion order in {path}; expected wxyz")
                provenance = doc.get("provenance", {})
                for calibration in [provenance.get("calibration", {}), *provenance.get("calibrations", [])]:
                    if calibration.get("session"):
                        if calibration.get("split") != "train":
                            raise ValueError(f"Synthetic dataset was calibrated from held-out data: {path}")
                        calibrations.add(calibration["session"])
                for key in ("history_steps", "future_steps", "dt"):
                    if doc["config"][key] != getattr(config, key):
                        raise ValueError(f"{path}: {key} differs from training configuration")
                for episode in doc["episodes"].values():
                    identity = (doc["source"], episode["session"])
                    if identity in sessions and sessions[identity] != episode["split"]:
                        raise ValueError(f"Session leakage across splits: {identity}")
                    sessions[identity] = episode["split"]
                    if split != "all" and episode["split"] != split:
                        continue
                    for shard in episode["shards"]:
                        file = (path.parent / shard["path"]).resolve()
                        if file.is_relative_to(root):
                            self._append(file, tuple(shard["shape"]), doc["source"], doc["terrain"], dataset_idx, episode["split"])
                for shard in doc.get("processed_shards", []):
                    if shard["split"] not in ("train", "val", "test"):
                        raise ValueError(f"Invalid processed shard split in {path}")
                    file = (path.parent / shard["path"]).resolve()
                    if (split == "all" or shard["split"] == split) and file.is_relative_to(root):
                        self._append(file, tuple(shard["shape"]), doc["source"], doc["terrain"], dataset_idx, shard["split"])
            if not manifests:
                if allow_wheeledlab:
                    # WheeledLab FKDDataset._load_file_names uses entered path
                    # order, sorted data/*.bin, then direct *.bin as fallback.
                    folder = root / "data" if (root / "data").is_dir() else root
                    bins = sorted(folder.glob("*.bin"))
                    if bins:
                        if split != "all":
                            raise ValueError("WheeledLab bins have no recorded split; choose all to view them")
                        for file in bins:
                            count, remainder = divmod(file.stat().st_size, config.window_steps * 42 * 4)
                            if remainder:
                                raise ValueError(f"Invalid binary size: {file}")
                            self._append(file, (count, config.window_steps, 42), "wheeledlab", "unknown", dataset_idx, None)
                        continue
                # External generalist data must already be split by episode/session.
                # Its layout is declared with a schema.json, never inferred silently.
                metadata = root / "schema.json"
                bins = [(file, name) for name in (("train", "val", "test") if split == "all" else (split,))
                        for file in sorted((root / name).rglob("*.bin"))]
                if bins and not metadata.exists():
                    raise ValueError(f"External FKD data requires {metadata}; see README")
                if bins:
                    doc = json.loads(metadata.read_text())
                    for k, expected in {"features": 42, "history_steps": config.history_steps,
                                        "future_steps": config.future_steps, "dt": config.dt,
                                        "dtype": "float32", "session_disjoint_splits": True}.items():
                        if doc.get(k) != expected:
                            raise ValueError(f"{metadata}: expected {k}={expected}")
                    for path, saved_split in bins:
                        count, remainder = divmod(path.stat().st_size, config.window_steps * 42 * 4)
                        if remainder:
                            raise ValueError(f"Invalid binary size: {path}")
                        self._append(path, (count, config.window_steps, 42), "generalist", "generalist", dataset_idx, saved_split)
        for session in calibrations:
            if ("real", session) in sessions and sessions[("real", session)] != "train":
                raise ValueError(f"DBM calibration used held-out session {session}")
        if not self.shards:
            raise ValueError(f"No {split} windows found in {roots}. Supply data with complete-session splits.")

    def _append(self, path, shape, source, terrain, dataset_idx=0, split=None):
        path = Path(path).resolve()
        if path in self._seen_files:
            return
        if len(shape) != 3 or shape[1:] != (self.config.window_steps, 42):
            raise ValueError(f"Invalid shard shape: {path} {shape}")
        if path.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError(f"Incomplete or malformed shard: {path}")
        if shape[0]:
            self._seen_files.add(path)
            self.shards.append((path, shape, source, terrain))
            self.ends.append((self.ends[-1] if self.ends else 0) + shape[0])
            self.dataset_file_mapping.append(dataset_idx)
            self.shard_splits.append(split)

    def __len__(self):
        return self.ends[-1]

    def locate_window(self, index):
        """Locate a global trajectory index using WheeledLab's file ordering."""
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        path, shape, source, terrain = self.shards[shard]
        start = self.ends[shard - 1] if shard else 0
        return {"path": str(path), "file_index": shard, "window_index": index - start,
                "dataset_idx": self.dataset_file_mapping[shard], "source": source, "terrain": terrain,
                "split": self.shard_splits[shard]}

    def load_window(self, index):
        """Read one stored [time,42] window through the training loader's mmap."""
        location = self.locate_window(index)
        shard = location["file_index"]
        path, shape, _, _ = self.shards[shard]
        if shard not in self.cache:
            self.cache[shard] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return torch.from_numpy(np.array(self.cache[shard][location["window_index"]], copy=True))

    def get_fkd_sample(self, index):
        """WheeledLab FKDDataset.__getitem__ for its standard 42-column format.

        The current state is shared by history and future. Its future delta
        starts at identity, as in WheeledLab, without changing stored data.
        """
        data = self.load_window(index)
        current = self.config.history_steps - 1
        world, body, delta = data[:, :13], data[:, 13:26], data[:, 26:39]
        actions = data[:, 39:41]
        delta_future = delta[current:].clone()
        delta_future[0, :3] = 0.0
        delta_future[0, 3:7] = data.new_tensor([1.0, 0.0, 0.0, 0.0])
        return {"history": body[:current + 1].clone(), "future": body[current:].clone(),
                "future_actions": actions[current:], "history_actions": actions[:current + 1],
                "delta_history": delta[:current + 1], "delta_future": delta_future,
                "hmap": data.new_zeros((1, 1)), "vel_body": body[current:current + 1, 7:10],
                "ang_vel_body": body[current:current + 1, 10:13],
                "pos_world": world[current, :3], "quat_wrt_world": world[current, 3:7],
                "vel_wrt_world": world[current, 7:10], "angvel_wrt_world": world[current, 10:13],
                "dataset_idx": self.locate_window(index)["dataset_idx"], "car_number": data[0, 41].long(),
                "full_wf_states": world, "full_bf_states": body, "full_delta_states": delta, "full_data": data}

    def __getitem__(self, index):
        window = self.load_window(index)
        result = model_sample(window, self.config.history_steps)
        location = self.locate_window(index)
        result["terrain"], result["source"] = location["terrain"], location["source"]
        return result

    def __getstate__(self):
        return {**self.__dict__, "cache": OrderedDict()}


class TerrainSampler(Sampler):
    def __init__(self, dataset, samples, seed=42, real_fraction=0.5):
        self.samples, self.seed, self.epoch = samples, seed, 0
        grouped = defaultdict(list)
        previous = 0
        for shard, end in zip(dataset.shards, dataset.ends):
            grouped[(shard[2], shard[3])].append((previous, end - previous))
            previous = end
        if not 0 < real_fraction < 1:
            raise ValueError("real_fraction must lie strictly between zero and one")
        self.groups, weights = [], []
        for (source, terrain), chunks in sorted(grouped.items()):
            source_weight = real_fraction if source == "real" else 1 - real_fraction if source == "sim" else 1.0
            terrain_count = sum(s == source for s, _ in grouped)
            weights.append(source_weight / terrain_count)
            starts, lengths = np.asarray(chunks, dtype=np.int64).T
            self.groups.append((starts, np.cumsum(lengths)))
        self.probabilities = np.asarray(weights) / sum(weights)

    def __len__(self):
        return self.samples

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for offset in range(0, self.samples, 4096):
            groups = rng.choice(len(self.groups), min(4096, self.samples - offset), p=self.probabilities)
            for group in groups:
                starts, ends = self.groups[group]
                within = rng.integers(ends[-1])
                shard = np.searchsorted(ends, within, side="right")
                yield int(starts[shard] + within - (ends[shard - 1] if shard else 0))
