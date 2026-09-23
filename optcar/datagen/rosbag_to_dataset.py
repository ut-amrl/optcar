"""ROS2 bag ingestion without requiring ROS during simulation or training."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from optcar.dataset import DatasetWriter
from optcar.utils.math import qrotate_torch


@dataclass
class BagConfig:
    pose_topic: str = "/odometry"
    twist_topic: str = "/odometry"
    command_topic: str = "/cmd_vel"
    twist_frame: str = "body"
    command_type: str = "twist_steering"
    command_shift_s: float = 0.0
    max_gap_s: float = 0.2
    timestamp_source: str = "bag"

    def __post_init__(self):
        if self.twist_frame not in ("body", "world"):
            raise ValueError("twist_frame must be body or world")
        if self.command_type not in ("twist_steering", "ackermann"):
            raise ValueError("command_type must be twist_steering or ackermann")
        if self.timestamp_source not in ("bag", "header") or self.max_gap_s <= 0 or not math.isfinite(self.max_gap_s) or not math.isfinite(self.command_shift_s):
            raise ValueError("Invalid timestamp source or maximum gap")


def _reader(path):
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    except ImportError as exc:
        raise ImportError("ROS2 bag loading requires: pip install -e '.[ros]'") from exc
    # Older ROS2 recordings do not embed definitions. Supply standard ROS2 types
    # plus the common Ackermann messages; embedded bag definitions take precedence.
    store = get_typestore(Stores.ROS2_HUMBLE)
    definitions = get_types_from_msg(
        "float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\nfloat32 acceleration\nfloat32 jerk\n",
        "ackermann_msgs/msg/AckermannDrive")
    definitions.update(get_types_from_msg(
        "std_msgs/Header header\nackermann_msgs/AckermannDrive drive\n", "ackermann_msgs/msg/AckermannDriveStamped"))
    store.register(definitions)
    return AnyReader([Path(path).expanduser().resolve()], default_typestore=store)


def topics(path):
    with _reader(path) as reader:
        return [{"topic": c.topic, "type": c.msgtype, "count": c.msgcount} for c in reader.connections]


def _xyz(value):
    return [value.x, value.y, value.z]


def _pose(message):
    value = message
    while hasattr(value, "pose"):
        value = value.pose
    q = value.orientation
    return _xyz(value.position) + [q.w, q.x, q.y, q.z]


def _twist(message):
    value = message
    while hasattr(value, "twist"):
        value = value.twist
    return _xyz(value.linear) + _xyz(value.angular)


def _command(message, kind):
    if kind == "ackermann":
        value = message.drive if hasattr(message, "drive") else message
        return [value.speed, value.steering_angle]
    value = message.twist if hasattr(message, "twist") else message
    # This repo uses angular.z as steering angle, not yaw-rate control.
    return [value.linear.x, value.angular.z]


def load_bag(path, config=None, dt=0.02):
    config = config or BagConfig()
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("dt must be positive")
    streams = {"pose": [], "twist": [], "command": []}
    selected = {"pose": config.pose_topic, "twist": config.twist_topic, "command": config.command_topic}
    with _reader(path) as reader:
        connections = [c for c in reader.connections if c.topic in selected.values()]
        missing = set(selected.values()) - {c.topic for c in connections}
        if missing:
            raise ValueError(f"Missing bag topics: {sorted(missing)}")
        for connection, stamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            if config.timestamp_source == "header" and hasattr(message, "header"):
                stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            time = stamp * 1e-9
            for key, topic in selected.items():
                if connection.topic == topic:
                    try:
                        value = _pose(message) if key == "pose" else _twist(message) if key == "twist" else _command(message, config.command_type)
                    except AttributeError as exc:
                        raise ValueError(f"Topic {topic} does not match configured {key} message semantics") from exc
                    streams[key].append((time + (config.command_shift_s if key == "command" else 0), value))
    arrays = {}
    for key, rows in streams.items():
        if len(rows) < 2:
            raise ValueError(f"Insufficient {key} messages")
        rows.sort(key=lambda row: row[0])
        stamps = np.asarray([r[0] for r in rows])
        values = np.asarray([r[1] for r in rows], dtype=np.float64)
        # Last observation wins at a duplicated timestamp.
        keep = np.r_[stamps[:-1] != stamps[1:], True]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {key} topic")
        if keep.sum() < 2:
            raise ValueError(f"{key} must contain at least two distinct timestamps")
        arrays[key] = (stamps[keep], values[keep])
    start = max(t[0] for t, _ in arrays.values())
    end = min(t[-1] for t, _ in arrays.values())
    if end - start < dt:
        raise ValueError("Topics have no overlapping time range")
    origin = start
    # Work in relative time to avoid unnecessary float loss on epoch timestamps.
    arrays = {k: (t - origin, v) for k, (t, v) in arrays.items()}
    grid = np.arange(0, end - start + 1e-8, dt)
    grid = grid[grid <= min(t[-1] for t, _ in arrays.values())]
    valid = np.ones(len(grid), bool)
    for key, (times, _) in arrays.items():
        left = np.searchsorted(times, grid, side="right") - 1
        right = np.minimum(left + 1, len(times) - 1)
        left = np.clip(left, 0, len(times) - 1)
        if key == "command":
            valid &= grid - times[left] <= config.max_gap_s
        else:
            valid &= (times[right] - times[left] <= config.max_gap_s) | np.isclose(grid, times[left], atol=1e-7, rtol=0)
    pose_time, pose = arrays["pose"]
    quat = pose[:, 3:7]
    if np.any(np.linalg.norm(quat, axis=1) < 1e-6):
        raise ValueError("Bag contains zero quaternions")
    rotation = Slerp(pose_time, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(grid)
    quaternion = rotation.as_quat()[:, [3, 0, 1, 2]]
    # Keep a continuous sign convention for derivatives and windows.
    for i in range(1, len(quaternion)):
        if np.dot(quaternion[i - 1], quaternion[i]) < 0:
            quaternion[i] *= -1
    position = np.stack([np.interp(grid, pose_time, pose[:, k]) for k in range(3)], -1)
    twist_time, twist_values = arrays["twist"]
    twist = np.stack([np.interp(grid, twist_time, twist_values[:, k]) for k in range(6)], -1)
    if config.twist_frame == "body":
        q = torch.tensor(quaternion, dtype=torch.float32)
        twist = np.concatenate([qrotate_torch(q, torch.tensor(twist[:, i:i + 3], dtype=torch.float32)).numpy()
                                for i in (0, 3)], -1)
    command_time, command_values = arrays["command"]
    command_idx = np.clip(np.searchsorted(command_time, grid, side="right") - 1, 0, len(command_time) - 1)
    commands = command_values[command_idx].astype(np.float32)
    states = np.concatenate((position, quaternion, twist), -1).astype(np.float32)
    intervals = []
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi - lo >= 2:
            intervals.append({"states": states[lo:hi], "actions": commands[lo:hi - 1], "times": grid[lo:hi]})
    if not intervals:
        raise ValueError("No continuous segments remain after timestamp/gap filtering")
    resolved = Path(path).expanduser().resolve()
    session = "bag_" + hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
    return {"session": session, "path": str(resolved), "dt": dt, "time_origin": origin,
            "config": asdict(config), "segments": intervals}


def export_bag(session, root, terrain, data, split="train", vehicle_id=0):
    if abs(session["dt"] - data.dt) > 1e-9:
        raise ValueError("Bag timestep does not match export configuration")
    total = 0
    signature = hashlib.sha256(str(sorted(session["config"].items())).encode()).hexdigest()[:10]
    with DatasetWriter(root, data, "real", terrain) as writer:
        # Prevent duplicate versions of one bag with different preprocessing.
        for entry in writer.manifest["episodes"].values():
            if entry["session"] == session["session"] and entry["metadata"].get("bag_config") != session["config"]:
                raise ValueError("This bag was exported with different settings; use a new dataset directory")
        for i, segment in enumerate(session["segments"]):
            total += writer.write_episode(f"{session['session']}_{signature}_{i:04d}",
                segment["states"], segment["actions"], split, session=session["session"], vehicle_id=vehicle_id,
                metadata={"path": session["path"], "bag_config": session["config"],
                          "start_s": float(segment["times"][0]), "end_s": float(segment["times"][-1])})
    return {"new_windows": total, "dataset": str(root), "split": split}
