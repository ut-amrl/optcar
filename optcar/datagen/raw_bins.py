"""Read raw float32 bins and their timestep, layout and auxiliary metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_POSE_COLS = slice(0, 7)
_VEL_COLS = slice(7, 13)
_ACTION_V = 13
_ACTION_S = 14
_CMD_THR = 15
_CMD_STEER = 16


def find_dataset_root(p: Path) -> Path:
    """Find metadata even when OptCar preserves deeply nested bag folders."""
    p = Path(p).expanduser().resolve()
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (candidate / "metadata").is_dir():
            return candidate
    return p


def list_bins(path: Path) -> List[Path]:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.suffix == ".bin":
        return [p]
    if not p.is_dir():
        raise ValueError("Enter an existing bin folder or .bin file")
    data = p / "data"
    return sorted((data if data.is_dir() else p).rglob("*.bin"))


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Invalid metadata: {path}")
    return value


def read_metadata(path: Path) -> dict:
    root = find_dataset_root(path)
    metadata = root / "metadata"
    conversion = _json(metadata / "conversion.json")
    schema = _json(metadata / "schema.json")
    metadata_info = _json(metadata / "metadata.json")
    relative = str(path.resolve().relative_to(root))
    record = conversion.get("bins", {}).get(relative, {})
    config = record.get("config", conversion.get("config", {}))
    text_path = metadata / "dataset_config.py"
    text = text_path.read_text() if text_path.exists() else ""
    # Read literal fields only; dataset_config.py is never executed.
    def literal(name):
        match = re.search(rf"\b{name}\s*(?::[^=\n]+)?=\s*([0-9.eE+-]+)", text)
        return float(match.group(1)) if match else None
    dt = config.get("dt", literal("control_dt") or metadata_info.get("control_dt", 0.02))
    cols = schema.get("features", literal("feature_dim"))
    yaml_path = metadata / "rosbag_conversion.yaml"
    if yaml_path.exists():
        import yaml
        yaml_config = yaml.safe_load(yaml_path.read_text()) or {}
        dt = yaml_config.get("dt", dt)
        if cols is None and "include_cmd_vel" in yaml_config:
            cols = 17 if yaml_config["include_cmd_vel"] else 15
    return {"dt": dt, "columns": cols, "num_envs": literal("num_envs") or metadata_info.get("num_envs", 1),
            "source": record.get("source"), "root": str(root), "aux": record.get("aux"),
            "is_auxiliary": (root / "metadata/aux_schema.json").exists() and Path(relative).parts[:1] == ("aux",)}


def read_dt(path: Path) -> float:
    return float(read_metadata(Path(path))["dt"])


def _read_expected_cols(bin_path: Path) -> int | None:
    value = read_metadata(bin_path)["columns"]
    return int(value) if value is not None else None


def _bin_layout_score(arr: np.ndarray) -> float:
    """Score how plausible [x,y,z,qw,qx,qy,qz,...] rows look."""
    if arr.shape[0] < 2:
        return 0.0

    pose = arr[:, _POSE_COLS].astype(np.float64)
    finite_penalty = 100.0 * (1.0 - float(np.isfinite(pose).mean()))

    quat_norm = np.linalg.norm(pose[:, 3:7], axis=1)
    quat_norm = quat_norm[np.isfinite(quat_norm)]
    if quat_norm.size == 0:
        quat_penalty = 100.0
    else:
        quat_err = np.abs(quat_norm - 1.0)
        quat_penalty = (
            12.0 * float(np.nanmedian(quat_err))
            + 4.0 * float(np.nanpercentile(quat_err, 90))
        )

    xy = pose[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.shape[0] < 2:
        jump_penalty = 100.0
        origin_penalty = 100.0
    else:
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        span = np.linalg.norm(np.nanmax(xy, axis=0) - np.nanmin(xy, axis=0))
        span = max(float(span), 1e-6)
        jump_penalty = float(np.nanpercentile(steps, 95)) / span
        origin_penalty = 2.0 * float(np.mean(np.linalg.norm(xy, axis=1) < 1e-6))

    return finite_penalty + quat_penalty + jump_penalty + origin_penalty


def load_bin(bin_path: Path, columns=None, num_envs=None, env_index=0) -> Tuple[np.ndarray, int]:
    """Read float32, preserving [environment, step, feature] order."""
    bin_path = Path(bin_path).expanduser().resolve()
    if not bin_path.is_file() or bin_path.suffix != ".bin":
        raise ValueError("Choose an existing raw .bin file")
    size = bin_path.stat().st_size
    if not size or size % 4:
        raise ValueError("Empty or incomplete float32 bin file")
    meta = read_metadata(bin_path)
    if meta["is_auxiliary"]:
        raise ValueError("Auxiliary twist bins are diagnostics; open their paired main bin in data/")
    environments = num_envs if num_envs is not None else meta["num_envs"]
    if int(environments) != float(environments) or not 1 <= int(environments) <= 100000:
        raise ValueError("Environment count must be a positive integer")
    environments = int(environments)
    if env_index is not None and (int(env_index) != float(env_index) or not 0 <= int(env_index) < environments):
        raise ValueError("Environment index is outside this bin")
    raw = np.memmap(bin_path, dtype="<f4", mode="r")
    expected = columns if columns is not None else meta["columns"]
    possible = [D for D in (15, 17) if raw.size % (D * environments) == 0]
    if expected is not None:
        if expected not in (15, 17) or expected not in possible:
            raise ValueError("Bin size does not match its raw layout (15 or 17 columns and environment count)")
        width = int(expected)
    elif possible:
        # Keep quaternion/trajectory plausibility scoring for headerless bins.
        _, width = min((_bin_layout_score(raw.reshape(-1, D)[:2048]), D) for D in possible)
    else:
        raise ValueError("Cannot infer raw bin columns; expected 15 or 17 float32 features")
    data = raw.reshape(environments, -1, width)
    return (data if env_index is None else data[int(env_index)]), width


def load_session(bin_path: Path) -> Dict[str, np.ndarray]:
    arr, D = load_bin(Path(bin_path))
    root = find_dataset_root(Path(bin_path))
    dt = read_dt(root)
    T = arr.shape[0]
    pose = arr[:, _POSE_COLS].astype(np.float64)
    vel = arr[:, _VEL_COLS].astype(np.float64)
    action_v = arr[:, _ACTION_V].astype(np.float64)
    action_s = arr[:, _ACTION_S].astype(np.float64)
    if D >= 17:
        cmd_thr = arr[:, _CMD_THR].astype(np.float64)
        cmd_steer = arr[:, _CMD_STEER].astype(np.float64)
    else:
        cmd_thr = action_v.copy()
        cmd_steer = action_s.copy()
    time = np.arange(T, dtype=np.float64) * dt
    return {
        "time": time,
        "pose": pose,
        "twist": vel,
        "action_v": action_v,
        "action_s": action_s,
        "cmd_throttle": cmd_thr,
        "cmd_steer": cmd_steer,
        "dt": float(dt),
        "n_frames": int(T),
        "n_cols": int(D),
    }


def yaw_from_quat(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def body_frame_window(
    pose: np.ndarray,
    frame_idx: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    end = min(frame_idx + horizon, pose.shape[0])
    seg = pose[frame_idx:end]
    x0, y0 = seg[0, 0], seg[0, 1]
    yaw0 = float(yaw_from_quat(seg[0, 3], seg[0, 4], seg[0, 5], seg[0, 6]))
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx = seg[:, 0] - x0
    dy = seg[:, 1] - y0
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    yaws = yaw_from_quat(seg[:, 3], seg[:, 4], seg[:, 5], seg[:, 6])
    byaw = np.unwrap(yaws) - yaw0
    return bx, by, byaw


COLUMNS = ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz", "command_velocity", "command_steering", "unshifted_velocity", "unshifted_steering"]


def describe_bin(path: Path, **options) -> dict:
    path = Path(path).expanduser().resolve()
    meta = read_metadata(path)
    arr, width = load_bin(path, options.get("columns"), options.get("num_envs"), options.get("env_index", 0))
    dt = float(options.get("dt") if options.get("dt") is not None else meta["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Timestep must be a positive finite number")
    return {"path": str(path), "name": path.name, "samples": len(arr), "columns": width,
            "num_envs": int(options.get("num_envs") or meta["num_envs"]), "dt": dt,
            "duration_s": (len(arr) - 1) * dt, "source": meta["source"]}


def scan_bins(path: str, **options) -> dict:
    root = Path(path).expanduser().resolve()
    base = root.parent if root.is_file() else root
    bins = []
    for file in list_bins(root):
        try:
            entry = describe_bin(file, **options)
        except (ValueError, OSError) as exc:
            entry = {"path": str(file), "name": file.name, "error": str(exc)}
        entry["relative_path"] = str(file.relative_to(base))
        bins.append(entry)
    return {"root": str(root), "bins": bins}


def _clean(values):
    """JSON null for missing/nonfinite signals, never NaN or Infinity."""
    array = np.asarray(values)
    return np.where(np.isfinite(array), array, None).tolist()


def forward_body_velocity(rows: np.ndarray) -> np.ndarray:
    """Rotate world velocity into each sample's own body frame (full quaternion)."""
    from scipy.spatial.transform import Rotation
    rows = np.atleast_2d(rows)
    velocity = np.full(len(rows), np.nan)
    valid = np.isfinite(rows[:, 3:10]).all(axis=1) & (np.linalg.norm(rows[:, 3:7], axis=1) > 1e-8)
    if valid.any():
        rotation = Rotation.from_quat(rows[valid][:, [4, 5, 6, 3]])
        velocity[valid] = rotation.inv().apply(rows[valid, 7:10])[:, 0]
    return velocity


def auxiliary_twist(path: Path, info: dict, env_index: int):
    """Only use companions explicitly linked in conversion metadata."""
    metadata = read_metadata(path)
    aux = metadata["aux"]
    if not aux or aux.get("frame") != "body":
        return None, "Derived from world velocity", None
    root = Path(metadata["root"])
    target = (root / aux["path"]).resolve()
    if not target.is_relative_to(root):
        return None, "Derived from world velocity", "Auxiliary path is outside the dataset; using derived body velocity."
    expected_bytes = info["num_envs"] * info["samples"] * 6 * 4
    if not target.is_file() or target.stat().st_size != expected_bytes:
        return None, "Derived from world velocity", "Auxiliary twist is missing or has a different row count; using derived body velocity."
    data = np.memmap(target, dtype="<f4", mode="r").reshape(info["num_envs"], info["samples"], 6)
    return data[env_index], f"{aux['topic']} · body x", None


def inspect_bin(path: str, **options) -> dict:
    info = describe_bin(Path(path), **options)
    arr, width = load_bin(Path(path), info["columns"], info["num_envs"], options.get("env_index", 0))
    indices = np.unique(np.linspace(0, len(arr) - 1, min(len(arr), 6000), dtype=int))
    aux, forward_source, aux_warning = auxiliary_twist(Path(path), info, int(options.get("env_index", 0)))
    forward = _clean(aux[indices, 0] if aux is not None else forward_body_velocity(arr[indices]))
    low = np.full(width, np.inf)
    high = np.full(width, -np.inf)
    missing = np.zeros(width, dtype=np.int64)
    forward_low, forward_high, forward_missing = np.inf, -np.inf, 0
    # Bound memory use for large simulated runs.
    for start in range(0, len(arr), 100000):
        chunk = arr[start:start + 100000]
        finite = np.isfinite(chunk)
        missing += (~finite).sum(axis=0)
        low = np.minimum(low, np.where(finite, chunk, np.inf).min(axis=0))
        high = np.maximum(high, np.where(finite, chunk, -np.inf).max(axis=0))
        forward_chunk = aux[start:start + len(chunk), 0] if aux is not None else forward_body_velocity(chunk)
        valid_forward = np.isfinite(forward_chunk)
        forward_missing += int((~valid_forward).sum())
        if valid_forward.any():
            forward_low = min(forward_low, float(forward_chunk[valid_forward].min()))
            forward_high = max(forward_high, float(forward_chunk[valid_forward].max()))
    return {**info, "env_index": int(options.get("env_index", 0)), "indices": indices.tolist(),
            "times": (indices * info["dt"]).tolist(), "rows": _clean(arr[indices]),
            "forward_velocity": forward, "forward_source": forward_source, "aux_warning": aux_warning,
            "forward_stats": {"min": _clean(forward_low), "max": _clean(forward_high), "missing": forward_missing},
            "forward_frame": _clean(aux[0, 0] if aux is not None else forward_body_velocity(arr[:1])[0]),
            "labels": COLUMNS[:width], "stats": [
                {"name": name, "min": _clean(low[i]), "max": _clean(high[i]), "missing": int(missing[i])}
                for i, name in enumerate(COLUMNS[:width])], "frame": _clean(arr[0])}


def bin_frame(path: str, index: int, **options) -> dict:
    info = describe_bin(Path(path), **options)
    arr, _ = load_bin(Path(path), info["columns"], info["num_envs"], options.get("env_index", 0))
    if int(index) != float(index) or not 0 <= int(index) < len(arr):
        raise ValueError("Sample is outside this bin")
    aux, _, _ = auxiliary_twist(Path(path), info, int(options.get("env_index", 0)))
    forward = aux[int(index), 0] if aux is not None else forward_body_velocity(arr[int(index)])[0]
    return {"index": int(index), "time": int(index) * info["dt"], "row": _clean(arr[int(index)]),
            "forward_velocity": _clean(forward)}


def action_window(path: str, index: int, before_s: float, after_s: float, **options) -> dict:
    """Read the action window around a sample, independently of overview sampling."""
    info = describe_bin(Path(path), **options)
    arr, _ = load_bin(Path(path), info["columns"], info["num_envs"], options.get("env_index", 0))
    if int(index) != float(index) or not 0 <= int(index) < len(arr):
        raise ValueError("Sample is outside this bin")
    before_s, after_s = float(before_s), float(after_s)
    if not np.isfinite([before_s, after_s]).all() or min(before_s, after_s) < 0 or before_s + after_s <= 0:
        raise ValueError("Before/after times must be nonnegative with a positive total window")
    current = int(index)
    # Clip to the recording, while the plot's relative time axis stays fixed.
    start = max(0, current - int(np.floor(min(before_s / info["dt"], len(arr)) + 1e-9)))
    end = min(len(arr) - 1, current + int(np.floor(min(after_s / info["dt"], len(arr)) + 1e-9)))
    indices = np.unique(np.append(np.linspace(start, end, min(end - start + 1, 6000), dtype=int), current))
    aux, _, _ = auxiliary_twist(Path(path), info, int(options.get("env_index", 0)))
    return {"current": current, "indices": indices.tolist(), "commands": _clean(arr[indices, 13:]),
            "forward_velocity": _clean(aux[indices, 0] if aux is not None else forward_body_velocity(arr[indices]))}
