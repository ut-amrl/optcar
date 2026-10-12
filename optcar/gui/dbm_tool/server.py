"""Manual DBM calibration with raw-bin browsing, rollouts and reference trajectories."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from flask import Blueprint, jsonify, render_template, request


from optcar.datagen.raw_bins import body_frame_window, find_dataset_root, scan_bins, load_session, read_dt, yaw_from_quat
from optcar.models.dynamic_bicycle import DBMParams, _rollout_torch as _batch_rollout_dynamic_bicycle_jit_ref
from optcar.utils.math import qinverse_torch, qrotate_torch
import torch

app = Blueprint("dbm_tool", __name__, template_folder="templates", static_folder="static")
_TORCH_OK = True


def list_bins(path):
    return [Path(item["path"]) for item in scan_bins(str(path))["bins"] if not item.get("error")]


@app.errorhandler(ValueError)
@app.errorhandler(TypeError)
@app.errorhandler(KeyError)
@app.errorhandler(FileNotFoundError)
def input_error(error):
    return jsonify({"error": str(error)}), 400


def window_options(data, payload):
    n = len(data["pose"])
    frame, horizon = int(payload["frame_idx"]), int(payload["horizon"])
    if frame != float(payload["frame_idx"]) or not 0 <= frame < n:
        raise ValueError("Choose a sample within this recording")
    if horizon != float(payload["horizon"]) or not 1 <= horizon <= 10000:
        raise ValueError("Horizon must be between 1 and 10,000 steps")
    shift = float(payload.get("cmd_shift_s", 0))
    if not np.isfinite(shift):
        raise ValueError("Command delay must be finite")
    return frame, min(horizon, n - frame - 1), shift


@app.route("/api/rollout", methods=["POST"])
def api_rollout():
    if not _TORCH_OK:
        return jsonify({"error": "torch unavailable on this server"}), 500

    p = request.get_json(force=True)
    sid = p["session_id"]
    frame_idx = int(p["frame_idx"])
    horizon = int(p["horizon"])
    cmd_shift_s = float(p.get("cmd_shift_s", 0.0))
    params = asdict(DBMParams(**p["params"]))

    data = _sessions.get(sid)
    if data is None:
        return jsonify({"error": "unknown session"}), 404

    frame_idx, horizon, cmd_shift_s = window_options(data, p)
    if horizon < 1:
        raise ValueError("No future samples after the selected sample")

    shift_steps = int(round(cmd_shift_s / max(data["dt"], 1e-9)))
    cmd_v_full = data["cmd_throttle"]
    cmd_s_full = data["cmd_steer"]
    L = cmd_v_full.shape[0]
    cmd_lo = frame_idx - shift_steps
    cmd_hi = frame_idx + horizon - shift_steps

    cmd_v_arr = np.full(horizon, np.nan, dtype=np.float32)
    cmd_s_arr = np.full(horizon, np.nan, dtype=np.float32)
    src_lo = max(cmd_lo, 0)
    src_hi = min(cmd_hi, L)
    if src_hi > src_lo:
        dst_lo = src_lo - cmd_lo
        dst_hi = dst_lo + (src_hi - src_lo)
        cmd_v_arr[dst_lo:dst_hi] = cmd_v_full[src_lo:src_hi].astype(np.float32)
        cmd_s_arr[dst_lo:dst_hi] = cmd_s_full[src_lo:src_hi].astype(np.float32)

    if not np.isfinite(cmd_v_arr).all() or not np.isfinite(cmd_s_arr).all():
        raise ValueError("Command delay moves this window outside recorded command coverage")

    twist_0 = data["twist"][frame_idx]  # world-frame [vx, vy, vz, wx, wy, wz]
    pose_0 = data["pose"][frame_idx]

    dtype = torch.float32
    q0 = torch.tensor([[pose_0[3], pose_0[4], pose_0[5], pose_0[6]]], dtype=dtype)
    vel_world_3d = torch.tensor([[twist_0[0], twist_0[1], twist_0[2]]], dtype=dtype)
    vel_body_3d = qrotate_torch(qinverse_torch(q0), vel_world_3d)
    v_long_0 = float(vel_body_3d[0, 0])
    v_lat_0  = float(vel_body_3d[0, 1])
    yaw_rate_0 = float(twist_0[5])

    pos = torch.zeros((1, 3), dtype=dtype)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    vel_body = torch.tensor([[v_long_0, v_lat_0, yaw_rate_0]], dtype=dtype)
    wheel_speed = torch.tensor([v_long_0], dtype=dtype)
    actions = torch.from_numpy(np.stack([cmd_v_arr, cmd_s_arr], axis=-1)[np.newaxis])  # [1, H, 2]

    def _t(v):
        return torch.tensor(float(v), dtype=dtype)

    dt = float(data["dt"])
    traj, _ = _batch_rollout_dynamic_bicycle_jit_ref(
        pos, quat, vel_body, wheel_speed, actions, dt,
        _t(params["len_f"]), _t(params["len_r"]),
        _t(params["mass"]), _t(params["inertia"]),
        _t(params["max_steering_angle"]),
        _t(params["C_tire_f"]), _t(params["C_tire_r"]),
        _t(params["C_throttle_a"]), _t(params["C_rollf"]),
        _t(params["C_drag"]), _t(params["wheel_time_constant"]),
        _t(params["min_wheel_velocity"]), _t(params["max_wheel_velocity"]),
    )
    traj_np = traj[0].detach().numpy()  # [H, 13]
    if not np.isfinite(traj_np).all():
        raise ValueError("These parameters produce an unstable rollout; increase wheel response time or reduce force coefficients")
    bx = [0.0] + traj_np[:, 0].tolist()
    by = [0.0] + traj_np[:, 1].tolist()
    return jsonify({"bx": bx, "by": by})


def _clean(arr) -> list:
    """numpy array -> list with NaN replaced by None (JSON-safe)."""
    a = np.asarray(arr, dtype=np.float64)
    out = a.tolist()
    return [None if not np.isfinite(v) else v for v in out]


_sessions: Dict[str, dict] = {}
_lock = threading.Lock()
_TRAJECTORY_PARAM_SKIP = {"self", "current_state", "device", "dtype"}
_TRAJECTORY_CLASS_SKIP = {"BaseTrajectory", "OpenLoopTrajectory"}
_TRAJECTORY_INT_PARAMS = {"num_points", "num_resample_points", "n_jobs"}


def _trajectory_module():
    from . import trajectories
    return trajectories


def _default_to_jsonable(value: Any) -> Any:
    if value is inspect._empty:
        return ""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _param_kind(name: str, default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int" if name in _TRAJECTORY_INT_PARAMS else "float"
    if isinstance(default, float):
        return "float"
    if isinstance(default, (list, tuple)) or hasattr(default, "tolist"):
        return "list"
    return "text"


def _coerce_trajectory_param(value: Any, kind: str) -> Any:
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if kind == "int":
        return int(float(value))
    if kind == "float":
        return float(value)
    if kind == "list":
        if isinstance(value, (list, tuple)):
            return list(value)
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [float(part) for part in text.replace(",", " ").split()]
        return parsed
    if value is None:
        return None
    text = str(value).strip()
    return None if text == "" else text


def _available_trajectory_classes():
    module = _trajectory_module()
    base = module.BaseTrajectory
    classes = []
    for name, cls in sorted(vars(module).items()):
        if (
            inspect.isclass(cls)
            and issubclass(cls, base)
            and cls is not base
            and name not in _TRAJECTORY_CLASS_SKIP
            and cls.__module__ == module.__name__
        ):
            classes.append((name, cls))
    return classes


def _trajectory_specs() -> list[dict]:
    specs = []
    for name, cls in _available_trajectory_classes():
        params = []
        sig = inspect.signature(cls.__init__)
        for pname, param in sig.parameters.items():
            if pname in _TRAJECTORY_PARAM_SKIP:
                continue
            default = _default_to_jsonable(param.default)
            if name == "DubinsTrajectory" and pname == "start_pose" and default is None:
                default = [0.0, 0.0, 0.0]
            params.append({
                "name": pname,
                "kind": _param_kind(pname, default),
                "default": default,
                "required": param.default is inspect._empty,
            })
        specs.append({"name": name, "params": params})
    return specs


def _reference_trajectory_xy(trajectory) -> np.ndarray:
    ref = trajectory.reference_trajectory
    if hasattr(ref, "detach"):
        ref = ref.detach().cpu().numpy()
    else:
        ref = np.asarray(ref)
    if ref.ndim != 2 or ref.shape[1] < 2:
        raise RuntimeError("trajectory reference_trajectory must have at least x/y columns")
    return ref[:, :2].astype(np.float64)


def _segment_yaml_path(bin_path: Path) -> Path:
    return Path(bin_path).expanduser().resolve().with_suffix(".yaml")


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _write_simple_yaml(path: Path, data: dict[str, Any]) -> None:
    lines = [f"{key}: {_yaml_scalar(value)}" for key, value in data.items()]
    path.write_text("\n".join(lines) + "\n")


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        data[key.strip()] = parsed
    return data


def _segment_from_yaml(yaml_path: Path) -> dict[str, Any]:
    yaml_path = Path(yaml_path).expanduser().resolve()
    info = _parse_simple_yaml(yaml_path)
    bin_path_value = info.get("bin_path")
    if bin_path_value:
        bin_path = Path(str(bin_path_value)).expanduser()
    else:
        bin_path = yaml_path.with_suffix(".bin")
    if not bin_path.is_absolute():
        bin_path = yaml_path.parent / bin_path
    if not bin_path.exists() and info.get("bin_name"):
        bin_path = yaml_path.parent / str(info["bin_name"])
    if not bin_path.exists():
        raise FileNotFoundError(f"Bin not found for segment YAML: {yaml_path}")
    return {
        "yaml_path": str(yaml_path),
        "bin_path": str(bin_path.resolve()),
        "bin_name": Path(bin_path).name,
        "start_idx": int(info.get("start_idx", 0)),
        "end_idx": int(info.get("end_idx", -1)),
    }


def _sample_pose_xy(data: dict, start_idx: int, end_idx: int, max_points: int) -> dict[str, Any]:
    pose = data["pose"]
    twist = data["twist"]
    n = int(pose.shape[0])
    if n == 0:
        raise RuntimeError("empty session")

    start_idx = max(0, min(int(start_idx), n - 1))
    if end_idx < 0:
        end_idx = n - 1
    end_idx = max(start_idx, min(int(end_idx), n - 1))
    step = max(1, int(np.ceil((end_idx - start_idx + 1) / max(int(max_points), 2))))
    sample_indices = np.arange(start_idx, end_idx + 1, step, dtype=int)
    if sample_indices.size == 0 or sample_indices[-1] != end_idx:
        sample_indices = np.append(sample_indices, end_idx)
    sample = pose[sample_indices]
    sample_twist = twist[sample_indices]
    yaws = yaw_from_quat(sample[:, 3], sample[:, 4], sample[:, 5], sample[:, 6])
    v_long_body = sample_twist[:, 0] * np.cos(yaws) + sample_twist[:, 1] * np.sin(yaws)

    return {
        "x": _clean(sample[:, 0].astype(np.float64)),
        "y": _clean(sample[:, 1].astype(np.float64)),
        "v_long": _clean(v_long_body.astype(np.float64)),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_frames": n,
        "dt": data["dt"],
        "t0": float(data["time"][start_idx]),
        "t1": float(data["time"][end_idx]),
        "points": int(sample.shape[0]),
        "sample_step": step,
    }


@app.route("/")
def index():
    return render_template("dbm_tool/index.html", initial_path=request.args.get("folder") or request.args.get("path", ""),
                           params=asdict(DBMParams()))


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    payload = request.get_json(force=True)
    p = payload.get("dataset_path", "").strip()
    path = Path(p).expanduser()
    if not path.exists():
        return jsonify({"error": f"Path not found: {p}"}), 400
    scan = scan_bins(str(path))
    if not scan["bins"]:
        return jsonify({"error": f"No .bin files at or under {path}"}), 400
    return jsonify({**scan, "dt": read_dt(path)})


@app.route("/api/load", methods=["POST"])
def api_load():
    payload = request.get_json(force=True)
    bin_path = Path(payload["bin_path"]).expanduser().resolve()
    try:
        data = load_session(bin_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    data["bin_path"] = str(bin_path)
    sid = uuid.uuid4().hex
    with _lock:
        if len(_sessions) >= 8:
            _sessions.pop(next(iter(_sessions)))
        _sessions[sid] = data
    segment_yaml = _segment_yaml_path(bin_path)
    saved_segment = None
    if segment_yaml.exists():
        try:
            saved_segment = _segment_from_yaml(segment_yaml)
        except Exception:
            saved_segment = None
    return jsonify({
        "session_id": sid,
        "n_frames": data["n_frames"],
        "dt": data["dt"],
        "n_cols": data["n_cols"],
        "duration": float((data["n_frames"] - 1) * data["dt"]),
        "bin_path": str(bin_path),
        "segment_yaml_path": str(segment_yaml),
        "saved_segment": saved_segment,
    })


@app.route("/api/reference_trajectories", methods=["GET"])
def api_reference_trajectories():
    try:
        return jsonify({"trajectories": _trajectory_specs()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reference_trajectory", methods=["POST"])
def api_reference_trajectory():
    payload = request.get_json(force=True)
    trajectory_name = payload.get("trajectory_name")
    params_in = payload.get("params", {})
    max_points = int(payload.get("max_points", 8000))

    specs = {spec["name"]: spec for spec in _trajectory_specs()}
    classes = dict(_available_trajectory_classes())
    if trajectory_name not in classes:
        return jsonify({"error": f"Unknown trajectory: {trajectory_name}"}), 400

    params = {}
    for param in specs[trajectory_name]["params"]:
        name = param["name"]
        if name not in params_in and not param["required"]:
            value = param["default"]
        else:
            value = params_in.get(name, "")
        try:
            coerced = _coerce_trajectory_param(value, param["kind"])
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid value for {name}: {value} ({e})"}), 400
        if param["required"] and coerced in ("", None, []):
            return jsonify({"error": f"Missing required parameter: {name}"}), 400
        if coerced is not None:
            params[name] = coerced

    try:
        trajectory = classes[trajectory_name](**params)
        xy = _reference_trajectory_xy(trajectory)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    step = max(1, int(np.ceil(xy.shape[0] / max(max_points, 2))))
    sample = xy[::step]
    if sample.shape[0] == 0 or not np.array_equal(sample[-1], xy[-1]):
        sample = np.vstack([sample, xy[-1]])
    reference_speed = getattr(trajectory, "reference_speed", None)
    try:
        reference_speed = float(reference_speed)
    except (TypeError, ValueError):
        reference_speed = None
    if reference_speed is not None and not np.isfinite(reference_speed):
        reference_speed = None

    return jsonify({
        "name": trajectory_name,
        "label": trajectory.get_capture_label() if hasattr(trajectory, "get_capture_label") else trajectory_name,
        "x": _clean(sample[:, 0]),
        "y": _clean(sample[:, 1]),
        "reference_speed": reference_speed,
        "points": int(sample.shape[0]),
        "sample_step": step,
    })


@app.route("/api/save_segment", methods=["POST"])
def api_save_segment():
    p = request.get_json(force=True)
    sid = p["session_id"]
    data = _sessions.get(sid)
    if data is None:
        return jsonify({"error": "unknown session"}), 404
    bin_path = Path(data["bin_path"])
    n = int(data["pose"].shape[0])
    start_idx = max(0, min(int(p.get("start_idx", 0)), n - 1))
    end_idx = max(start_idx, min(int(p.get("end_idx", n - 1)), n - 1))
    yaml_path = _segment_yaml_path(bin_path)
    info = {
        "bin_name": bin_path.name,
        "bin_path": str(bin_path),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "dt": data["dt"],
        "n_frames": n,
        "n_cols": data["n_cols"],
    }
    try:
        _write_simple_yaml(yaml_path, info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    info["yaml_path"] = str(yaml_path)
    return jsonify(info)


@app.route("/api/load_segment", methods=["POST"])
def api_load_segment():
    p = request.get_json(force=True)
    yaml_path = p.get("yaml_path")
    if not yaml_path:
        sid = p.get("session_id")
        data = _sessions.get(sid)
        if data is None:
            return jsonify({"error": "unknown session"}), 404
        yaml_path = _segment_yaml_path(Path(data["bin_path"]))
    try:
        segment = _segment_from_yaml(Path(yaml_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(segment)


@app.route("/api/trajectory", methods=["POST"])
def api_trajectory():
    p = request.get_json(force=True)
    sid = p["session_id"]
    start_idx = int(p.get("start_idx", 0))
    end_idx = int(p.get("end_idx", -1))
    max_points = int(p.get("max_points", 8000))
    data = _sessions.get(sid)
    if data is None:
        return jsonify({"error": "unknown session"}), 404

    try:
        return jsonify(_sample_pose_xy(data, start_idx, end_idx, max_points))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/frame", methods=["POST"])
def api_frame():
    p = request.get_json(force=True)
    sid = p["session_id"]
    frame_idx = int(p["frame_idx"])
    horizon = int(p["horizon"])
    cmd_shift_s = float(p.get("cmd_shift_s", 0.0))
    data = _sessions.get(sid)
    if data is None:
        return jsonify({"error": "unknown session"}), 404

    n = int(data["pose"].shape[0])
    frame_idx, horizon, cmd_shift_s = window_options(data, p)
    end = frame_idx + horizon + 1

    # Positive shift = cmd is delayed: at time t, show cmd that came in at t - shift.
    shift_steps = int(round(cmd_shift_s / max(data["dt"], 1e-9)))
    cmd_lo = frame_idx - shift_steps
    cmd_hi = end - shift_steps
    cmd_v_full = data["cmd_throttle"]
    cmd_s_full = data["cmd_steer"]
    L = cmd_v_full.shape[0]
    win_len = end - frame_idx
    cmd_v_arr = np.full(win_len, np.nan, dtype=np.float64)
    cmd_s_arr = np.full(win_len, np.nan, dtype=np.float64)
    src_lo = max(cmd_lo, 0)
    src_hi = min(cmd_hi, L)
    if src_hi > src_lo:
        dst_lo = src_lo - cmd_lo
        dst_hi = dst_lo + (src_hi - src_lo)
        cmd_v_arr[dst_lo:dst_hi] = cmd_v_full[src_lo:src_hi]
        cmd_s_arr[dst_lo:dst_hi] = cmd_s_full[src_lo:src_hi]

    bx, by, byaw = body_frame_window(data["pose"], frame_idx, horizon + 1)
    t_rel = data["time"][frame_idx:end] - data["time"][frame_idx]
    twist = data["twist"][frame_idx:end]
    pose_seg = data["pose"][frame_idx:end]
    yaws = yaw_from_quat(pose_seg[:, 3], pose_seg[:, 4], pose_seg[:, 5], pose_seg[:, 6])
    cos_y, sin_y = np.cos(yaws), np.sin(yaws)
    v_long_body = twist[:, 0] * cos_y + twist[:, 1] * sin_y
    v_lat_body  = -twist[:, 0] * sin_y + twist[:, 1] * cos_y

    return jsonify({
        "t": _clean(t_rel),
        "bx": _clean(bx),
        "by": _clean(by),
        "byaw": _clean(byaw),
        "cmd_v": _clean(cmd_v_arr),
        "cmd_s": _clean(cmd_s_arr),
        "v_long": _clean(v_long_body),
        "v_lat": _clean(v_lat_body),
        "yaw_rate": _clean(twist[:, 5]),
        "frame_idx": frame_idx,
        "n_frames": n,
        "end_idx": end,
    })
