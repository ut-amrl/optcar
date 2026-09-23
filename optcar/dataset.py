"""FKD binary format, episode writer, memory-mapped loader, and terrain sampler."""
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

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


def pack_windows(states, actions, history_steps, vehicle_id=0):
    """states/actions: [B,T,13/2]; body features anchored to current state."""
    states = torch.as_tensor(states, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=states.device)
    current = states[:, history_steps - 1]
    body = torch.cat(horizon_world_to_body_frame(states, current[:, :3], current[:, 3:7]), -1)
    delta = compute_delta_frame_representation(states)
    car = states.new_full((*states.shape[:2], 1), float(vehicle_id))
    return torch.cat((states, body, delta, actions, car), -1)


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
    if len(fractions) != 3 or any(x < 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Split fractions must be three nonnegative numbers summing to one")
    value = int(hashlib.sha256(f"{seed}:{session}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < fractions[0] else "val" if value < sum(fractions[:2]) else "test"


class DatasetWriter:
    """One active writer per dataset. Manifest commits make restarts idempotent."""
    def __init__(self, root, config, source, terrain, provenance=None):
        self.root, self.config = Path(root), config
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
            else:
                self.manifest = {"schema": SCHEMA, "config": asdict(config), "source": source,
                                 "terrain": terrain, "provenance": provenance or {}, "episodes": {}}
                write_json(self.path, self.manifest)
            self.session_splits = {e["session"]: e["split"] for e in self.manifest["episodes"].values()}
            self.dirty = False
        except Exception:
            self.lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
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
        return episode in self.manifest["episodes"]

    def write_episode(self, episode, states, actions, split, session=None, vehicle_id=0, metadata=None):
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
        if states.ndim != 2 or states.shape[1] != 13 or actions.shape != (len(states) - 1, 2):
            raise ValueError("Expected T+1 states [13] and T commands [2]")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite episode values")
        norms = np.linalg.norm(states[:, 3:7], axis=-1)
        if np.any(norms < 1e-6):
            raise ValueError("Zero quaternion in episode")
        states = states.copy()
        states[:, 3:7] /= norms[:, None]
        adjacent_dot = np.sum(states[:-1, 3:7] * states[1:, 3:7], axis=-1)
        signs = np.r_[1.0, np.cumprod(np.where(adjacent_dot < 0, -1.0, 1.0))]
        states[:, 3:7] *= signs[:, None]
        actions = np.concatenate((actions, np.zeros((1, 2), np.float32)))
        raw_file = f"raw/{episode}.bin"
        self._binary(raw_file, np.concatenate((states, actions), -1))
        shards, count = [], 0
        cfg = self.config
        starts = range(0, len(states) - cfg.window_steps + 1, cfg.stride)
        for offset in range(0, len(starts), cfg.shard_windows):
            selected = starts[offset:offset + cfg.shard_windows]
            s = np.stack([states[i:i + cfg.window_steps] for i in selected])
            a = np.stack([actions[i:i + cfg.window_steps] for i in selected])
            packed = pack_windows(s, a, cfg.history_steps, vehicle_id).numpy()
            filename = f"processed/data/{episode}_{offset:08d}.bin"
            self._binary(filename, packed)
            shards.append({"path": filename, "shape": list(packed.shape)})
            count += len(packed)
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
        counts = [s["shape"][0] for e in self.manifest["episodes"].values() for s in e["shards"]]
        # FKD's future_horizon includes the current state; ours counts transitions.
        fields = {"history_horizon": cfg.history_steps, "future_horizon": cfg.future_steps + 1,
                  "total_horizon": cfg.window_steps, "chunk_horizon": cfg.window_steps,
                  "feature_dim": 42, "data_dtype": "float32", "control_dt": cfg.dt,
                  "sim_dt": cfg.dt, "decimation": 1, "stride": cfg.stride,
                  "num_files": len(counts), "total_chunks": sum(counts), "chunks_per_file_list": counts}
        text = '"""FKD-compatible dimensions; use manifest.json for session-disjoint splits."""\n'
        text += "class ProcessedDatasetConfig:\n" + "".join(f"    {k} = {v!r}\n" for k, v in fields.items())
        atomic_text(self.root / "processed/metadata/dataset_config.py", text)
        write_json(self.root / "processed/metadata/schema.json", {**SCHEMA, **fields})


class FKDDataset(Dataset):
    def __init__(self, roots, config, split="train", sources=None):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.config, self.shards, self.ends = config, [], []
        self.cache = OrderedDict()
        seen = set()
        sessions = {}
        calibrations = set()
        for root in roots:
            root = Path(root).expanduser().resolve()
            manifests = [root / "manifest.json"] if (root / "manifest.json").exists() else sorted(root.rglob("manifest.json"))
            for path in manifests:
                if path in seen:
                    continue
                seen.add(path)
                doc = json.loads(path.read_text())
                if "schema" not in doc or "episodes" not in doc:
                    continue
                if sources and doc["source"] not in sources:
                    continue
                if doc["schema"].get("features") != 42 or doc["schema"].get("dtype") != "float32":
                    raise ValueError(f"Unsupported feature layout in {path}")
                calibration = doc.get("provenance", {}).get("calibration", {})
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
                    if episode["split"] != split:
                        continue
                    for shard in episode["shards"]:
                        self._append(path.parent / shard["path"], tuple(shard["shape"]), doc["source"], doc["terrain"])
            if not manifests:
                # External generalist data must already be split by episode/session.
                # Its layout is declared with a schema.json, never inferred silently.
                metadata = root / "schema.json"
                bins = sorted((root / split).rglob("*.bin"))
                if bins and not metadata.exists():
                    raise ValueError(f"External FKD data requires {metadata}; see README")
                if bins:
                    doc = json.loads(metadata.read_text())
                    for k, expected in {"features": 42, "history_steps": config.history_steps,
                                        "future_steps": config.future_steps, "dt": config.dt,
                                        "dtype": "float32", "session_disjoint_splits": True}.items():
                        if doc.get(k) != expected:
                            raise ValueError(f"{metadata}: expected {k}={expected}")
                    for path in bins:
                        count, remainder = divmod(path.stat().st_size, config.window_steps * 42 * 4)
                        if remainder:
                            raise ValueError(f"Invalid binary size: {path}")
                        self._append(path, (count, config.window_steps, 42), "generalist", "generalist")
        for session in calibrations:
            if ("real", session) in sessions and sessions[("real", session)] != "train":
                raise ValueError(f"DBM calibration used held-out session {session}")
        if not self.shards:
            raise ValueError(f"No {split} windows found in {roots}. Supply data with complete-session splits.")

    def _append(self, path, shape, source, terrain):
        if len(shape) != 3 or shape[1:] != (self.config.window_steps, 42):
            raise ValueError(f"Invalid shard shape: {path} {shape}")
        if path.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError(f"Incomplete or malformed shard: {path}")
        if shape[0]:
            self.shards.append((path, shape, source, terrain))
            self.ends.append((self.ends[-1] if self.ends else 0) + shape[0])

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        path, shape, source, terrain = self.shards[shard]
        start = self.ends[shard - 1] if shard else 0
        if shard not in self.cache:
            self.cache[shard] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        window = torch.from_numpy(np.array(self.cache[shard][index - start], copy=True))
        result = model_sample(window, self.config.history_steps)
        result["terrain"], result["source"] = terrain, source
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
