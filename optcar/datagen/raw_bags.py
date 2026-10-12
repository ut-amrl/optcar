"""ROS2 bags to raw bins with uniform, last-observation timestamp alignment."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional
import fcntl
import json
import math
import os
import tempfile
import time

import numpy as np
import yaml


@dataclass
class RawBagConfig:
    dt: float = 0.02
    pose_topic: str = "/super_odometry/pose"
    twist_topic: str = "/super_odometry/twist_wf"
    command_topic: str = "/cmd_vel"
    twist_frame: str = "world"
    command_type: str = "twist_steering"
    action_shift: float = 0.0
    max_duration: float | None = None
    include_cmd_vel: bool = False
    aux_twist_topic: str | None = "/super_odometry/twist"

    def __post_init__(self):
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Timestep must be finite and positive")
        if not math.isfinite(self.action_shift):
            raise ValueError("Action shift must be finite")
        if self.max_duration is not None and (not math.isfinite(self.max_duration) or self.max_duration <= 0):
            raise ValueError("Maximum duration must be finite and positive")
        if self.twist_frame not in ("world", "body") or self.command_type not in ("twist_steering", "ackermann"):
            raise ValueError("Invalid velocity frame or command format")
        if any(not isinstance(t, str) or not t.strip() for t in (self.pose_topic, self.twist_topic, self.command_topic)):
            raise ValueError("Provide all three topic names")
        if self.aux_twist_topic is not None:
            if not isinstance(self.aux_twist_topic, str):
                raise ValueError("Auxiliary twist topic must be a name or null")
            self.aux_twist_topic = self.aux_twist_topic.strip() or None
        if not isinstance(self.include_cmd_vel, bool):
            raise ValueError("include_cmd_vel must be a boolean")


class ConversionCancelled(Exception):
    pass


def _check_cancel(cancelled):
    if cancelled():
        raise ConversionCancelled()


def default_output(root):
    return Path(root).expanduser().resolve().parent / "bins"


def bin_relative_path(root, bag):
    root, bag = Path(root).resolve(), Path(bag).resolve()
    relative = bag.relative_to(root)
    return Path(root.name) / relative.parent / (bag.name + ".bin") if relative != Path(".") else Path(root.name + ".bin")


def inspect_bag(path, root):
    path = Path(path).resolve()
    result = {"path": str(path), "relative_path": str(path.relative_to(root)),
              "output_relative": str(Path("data") / bin_relative_path(root, path)), "topics": [], "error": None}
    files = sorted(set(path.glob("*.db3")) | set(path.glob("*.mcap")))
    result["bytes"] = sum(f.stat().st_size for f in files)
    try:
        doc = yaml.safe_load((path / "metadata.yaml").read_text())
        info = doc["rosbag2_bagfile_information"]
        missing = [f for f in info["relative_file_paths"] if not (path / f).is_file()]
        if missing:
            raise ValueError(f"Missing storage files: {missing}")
        result.update(storage=info["storage_identifier"],
                      duration_s=info["duration"]["nanoseconds"] / 1e9,
                      start_ns=info["starting_time"]["nanoseconds_since_epoch"],
                      messages=info["message_count"], files=info["relative_file_paths"])
        for item in info["topics_with_message_count"]:
            topic = item["topic_metadata"]
            result["topics"].append({"topic": topic["name"], "type": topic["type"], "count": item["message_count"]})
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        result["error"] = f"Invalid or missing metadata.yaml: {exc}. Repair with ros2 bag reindex before conversion."
    return result


def discover_bags(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Bag folder does not exist: {root}")
    bags = []
    for directory, children, files in os.walk(root):
        children[:] = sorted(c for c in children if not c.startswith("."))
        if "metadata.yaml" in files or any(f.endswith((".db3", ".mcap")) for f in files):
            bags.append(inspect_bag(Path(directory), root))
            children[:] = []
    return {"root": str(root), "default_output": str(default_output(root)), "bags": sorted(bags, key=lambda b: b["relative_path"])}


def _typestore():
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    store = get_typestore(Stores.ROS2_HUMBLE)
    definitions = get_types_from_msg("float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\nfloat32 acceleration\nfloat32 jerk\n", "ackermann_msgs/msg/AckermannDrive")
    definitions.update(get_types_from_msg("std_msgs/Header header\nackermann_msgs/AckermannDrive drive\n", "ackermann_msgs/msg/AckermannDriveStamped"))
    store.register(definitions)
    return store


def _values(message, kind, command_type):
    if kind == "pose":
        while hasattr(message, "pose"):
            message = message.pose
        p, q = message.position, message.orientation
        return [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
    if kind == "command" and command_type == "ackermann":
        message = message.drive if hasattr(message, "drive") else message
        return [message.speed, message.steering_angle]
    while hasattr(message, "twist"):
        message = message.twist
    v, w = message.linear, message.angular
    return [v.x, w.z] if kind == "command" else [v.x, v.y, v.z, w.x, w.y, w.z]


@dataclass
class TopicConfig:
    pose_topic: str = "/super_odometry/pose"
    velocity_topic: str = "/super_odometry/twist_wf"
    action_topic: str = "/cmd_vel"
    cmd_vel_topic: str = "/cmd_vel"
    command_type: str = "twist_steering"

class ROS2BagAligner:
    """Align ROS2 bag topics to a uniform time grid."""

    def __init__(
        self,
        bag_path: Path,
        dt: float = 0.02,
        max_duration: Optional[float] = None,
        action_shift: float = 0.0,
        topic_config: Optional[TopicConfig] = None,
        include_cmd_vel: bool = True,
    ):
        self.bag_path = Path(bag_path)
        self.dt = float(dt)
        self.max_duration = None if max_duration is None else float(max_duration)
        self.action_shift = float(action_shift)
        self.topics = topic_config or TopicConfig()
        self.include_cmd_vel = include_cmd_vel
        if self.max_duration is not None and self.max_duration <= 0:
            raise ValueError("max_duration must be positive when provided.")

    # ------------------------------------------------------------------ #
    # rosbag helpers
    # ------------------------------------------------------------------ #
    def _get_reader(self):
        # Metadata is validated by OptCar before opening; do not repair sources.
        from rosbags.rosbag2 import Reader
        return Reader(self.bag_path)

    def _read_all_topics_once(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Read the relevant topics once, with the OptCar ROS-independent reader."""
        progress = getattr(self, "progress", lambda value: None)
        cancelled = getattr(self, "cancelled", lambda: False)
        selected = {"pose": self.topics.pose_topic, "twist": self.topics.velocity_topic,
                    "command": self.topics.action_topic}
        if self.include_cmd_vel:
            selected["cmd_vel"] = self.topics.cmd_vel_topic
        required_topics = set(selected.values())
        aux_topic = getattr(self, "aux_twist_topic", None)
        if aux_topic:
            selected["aux_twist"] = aux_topic
        streams = {key: {"times": [], "values": [], "frame_ids": set()} for key in selected}
        store = _typestore()
        import time
        with self._get_reader() as reader:
            connections = [c for c in reader.connections if c.topic in selected.values()]
            missing = required_topics - {c.topic for c in connections if c.msgcount > 0}
            if missing:
                raise ValueError(f"Missing or empty topics: {sorted(missing)}")
            total = sum(c.msgcount for c in connections)
            last_update = 0.0
            progress({"phase": "reading", "messages": 0, "messages_total": total, "bag_fraction": 0.0})
            for n, (connection, stamp, raw) in enumerate(reader.messages(connections=connections), 1):
                _check_cancel(cancelled)
                message = store.deserialize_cdr(raw, connection.msgtype)
                for key, topic in selected.items():
                    if connection.topic == topic:
                        kind = "command" if key == "cmd_vel" else key
                        try:
                            value = _values(message, kind, self.topics.command_type)
                        except AttributeError as exc:
                            raise ValueError(f"Topic {topic} does not match configured {kind} semantics") from exc
                        streams[key]["times"].append(stamp * 1e-9)
                        streams[key]["values"].append(value)
                        if key == "aux_twist" and hasattr(message, "header"):
                            streams[key]["frame_ids"].add(message.header.frame_id)
                now = time.monotonic()
                if now - last_update >= 0.1 or n == total:
                    progress({"phase": "reading", "messages": n, "messages_total": total,
                              "bag_fraction": 0.85 * n / max(1, total)})
                    last_update = now
        _check_cancel(cancelled)
        progress({"phase": "aligning", "messages": total, "messages_total": total, "bag_fraction": 0.87})
        for key, stream in streams.items():
            times = np.asarray(stream["times"])
            values = np.asarray(stream["values"]).reshape(-1, 6) if key == "aux_twist" else np.asarray(stream["values"])
            if key != "aux_twist" and not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values in {selected[key]}")
            order = np.argsort(times, kind="stable")
            stream["times"], stream["values"] = times[order], values[order]
        self.max_topic_gaps_s = {key: float(np.max(np.diff(s["times"]))) if len(s["times"]) > 1 else 0.0
                                 for key, s in streams.items()}
        data = {
            self.topics.pose_topic: {"times": streams["pose"]["times"], "values": streams["pose"]["values"]},
            self.topics.velocity_topic: {"times": streams["twist"]["times"], "values": streams["twist"]["values"]},
            self.topics.action_topic: {"times": streams["command"]["times"],
                                      "vel": streams["command"]["values"][:, :1],
                                      "steer": streams["command"]["values"][:, 1:]},
        }
        if self.include_cmd_vel:
            cmd = streams["cmd_vel"]
            data.setdefault(self.topics.cmd_vel_topic, {"times": cmd["times"]}).update(
                throttle=cmd["values"][:, :1], steering=cmd["values"][:, 1:])
        self._stream_data = streams
        return data


    @staticmethod
    def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
        """Normalize quaternions without flipping signs."""
        if quats.size == 0:
            return quats
        norm = np.linalg.norm(quats, axis=-1, keepdims=True)
        norm[norm == 0.0] = 1.0
        return quats / norm

    @staticmethod
    def _align_asof(
        source_t: np.ndarray, source_vals: np.ndarray, target_t: np.ndarray
    ) -> np.ndarray:
        """Align samples by forward-fill (asof) semantics."""
        if source_t.size == 0:
            dim = source_vals.shape[1] if source_vals.ndim > 1 else 1
            return np.full((len(target_t), dim), np.nan, dtype=float)

        dim = source_vals.shape[1] if source_vals.ndim > 1 else 1
        output = np.full((len(target_t), dim), np.nan, dtype=float)
        idx = 0
        for i, t in enumerate(target_t):
            while idx + 1 < len(source_t) and source_t[idx + 1] <= t:
                idx += 1
            if source_t[idx] <= t:
                output[i] = (
                    source_vals[idx]
                    if source_vals.ndim > 1
                    else np.array([source_vals[idx]])
                )
        return output

    def process(self) -> Dict[str, np.ndarray]:
        """Convert bag topics into aligned arrays."""
        all_data = self._read_all_topics_once()

        pose = self._stream_data["pose"]["values"]
        vel = self._stream_data["twist"]["values"]
        t_pose = self._stream_data["pose"]["times"]
        t_vel = self._stream_data["twist"]["times"]

        action_data = all_data[self.topics.action_topic]
        act_times = action_data["times"]
        act_vel = action_data["vel"]
        act_steer = action_data["steer"]

        required = [
            (self.topics.pose_topic, t_pose),
            (self.topics.velocity_topic, t_vel),
            (self.topics.action_topic, act_times),
        ]
        for topic_name, times in required:
            if times.size == 0:
                raise RuntimeError(f"No messages found for topic: {topic_name}")

        cmd_throttle = cmd_steer = t_cmd = np.array([])
        if self.include_cmd_vel and self.topics.cmd_vel_topic in all_data:
            cmd = all_data[self.topics.cmd_vel_topic]
            t_cmd = cmd["times"]
            cmd_throttle = cmd["throttle"]
            cmd_steer = cmd["steering"]

        # Apply optional timestamp shift to the action topic
        t_act_shift = act_times + self.action_shift if self.action_shift else act_times

        relevant_times = [
            arr
            for arr in [
                t_pose,
                t_vel,
                t_act_shift,
                t_cmd,
            ]
            if arr.size > 0
        ]
        if not relevant_times:
            raise RuntimeError("No valid timestamps found in bag.")

        t_start = max(np.min(arr) for arr in relevant_times)
        t_end = min(np.max(arr) for arr in relevant_times)
        if self.max_duration is not None:
            t_end = min(t_end, t_start + self.max_duration)
        if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
            raise RuntimeError(
                "Empty overlap between topics. Adjust action_shift or bag contents."
            )

        t_common = np.arange(t_start, t_end, self.dt)

        aligned_pose = self._align_asof(t_pose, pose, t_common)
        aligned_vel = self._align_asof(t_vel, vel, t_common)
        aligned_act = self._align_asof(t_act_shift, act_vel, t_common)
        aligned_steer = self._align_asof(t_act_shift, act_steer, t_common)

        aligned_cmd_throttle = (
            self._align_asof(t_cmd, cmd_throttle, t_common)
            if cmd_throttle.size > 0
            else np.full((len(t_common), 1), np.nan, dtype=float)
        )
        aligned_cmd_steer = (
            self._align_asof(t_cmd, cmd_steer, t_common)
            if cmd_steer.size > 0
            else np.full((len(t_common), 1), np.nan, dtype=float)
        )

        aligned_pose[:, 3:7] = self._normalize_quaternions(aligned_pose[:, 3:7])

        return {
            "time": t_common,
            "pose": aligned_pose,
            "velocity": aligned_vel,
            "action_velocity": aligned_act.squeeze(-1),
            "action_steering": aligned_steer.squeeze(-1),
            "cmd_vel_throttle": aligned_cmd_throttle.squeeze(-1),
            "cmd_vel_steering": aligned_cmd_steer.squeeze(-1),
        }

def _stack_rows(data: Dict[str, np.ndarray], include_cmd_vel: bool) -> np.ndarray:
    """Stack aligned arrays into a single [T, D] matrix."""
    fields = [
        data["pose"],  # 7
        data["velocity"],  # 6
        data["action_velocity"][:, None],
        data["action_steering"][:, None],
    ]
    if include_cmd_vel:
        fields.extend(
            [
                data["cmd_vel_throttle"][:, None],
                data["cmd_vel_steering"][:, None],
            ]
        )
    combined = np.concatenate(fields, axis=1)
    return combined

def read_raw_bag(path, config, progress=lambda value: None, cancelled=lambda: False, *, with_aux=False):
    topics = TopicConfig(pose_topic=config.pose_topic, velocity_topic=config.twist_topic,
                         action_topic=config.command_topic, cmd_vel_topic=config.command_topic,
                         command_type=config.command_type)
    aligner = ROS2BagAligner(Path(path), dt=config.dt, max_duration=config.max_duration,
                            action_shift=config.action_shift, topic_config=topics,
                            include_cmd_vel=config.include_cmd_vel)
    aligner.progress, aligner.cancelled = progress, cancelled
    aligner.aux_twist_topic = config.aux_twist_topic
    data = aligner.process()
    _check_cancel(cancelled)
    if np.any(np.linalg.norm(data["pose"][:, 3:7], axis=1) < 1e-8):
        raise ValueError("Pose topic contains zero quaternions")
    if config.twist_frame == "body":
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat(data["pose"][:, [4, 5, 6, 3]])
        twist = data["velocity"]
        data["velocity"] = np.concatenate([rotation.apply(twist[:, :3]), rotation.apply(twist[:, 3:])], axis=1)
    rows = _stack_rows(data, config.include_cmd_vel).astype("<f4")
    if len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError("Overlap must contain at least two finite float32 samples")
    progress({"phase": "writing", "bag_fraction": 0.95, "samples": len(rows)})
    info = {"samples": len(rows), "shape": list(rows.shape), "start_ns": round(data["time"][0] * 1e9),
            "duration_s": len(rows) * config.dt, "max_topic_gaps_s": aligner.max_topic_gaps_s}
    aux = None
    if config.aux_twist_topic:
        stream = aligner._stream_data["aux_twist"]
        if len(stream["times"]):
            # Use the main grid exactly; optional diagnostics never trim the main recording.
            aux = aligner._align_asof(stream["times"], stream["values"], data["time"]).astype("<f4")
            aux[data["time"] > stream["times"][-1]] = np.nan
            aux[~np.isfinite(aux)] = np.nan
            frames = sorted(stream["frame_ids"])
            frame = "world" if any(f in ("map", "world", "odom") for f in frames) else "body"
            info["aux"] = {"topic": config.aux_twist_topic, "frame": frame, "frame_ids": frames,
                           "shape": list(aux.shape), "missing_values": int((~np.isfinite(aux)).sum()),
                           "dtype": "float32", "alignment": "asof", "dt": config.dt, "start_ns": info["start_ns"]}
            if frame != "body":
                info["aux_warning"] = "Auxiliary twist is labelled as world-frame; viewer derives forward body velocity from the main bin."
        else:
            info["aux_warning"] = f"No auxiliary twist messages on {config.aux_twist_topic}; main bin saved without auxiliary data."
    return (rows, info, aux) if with_aux else (rows, info)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".optcar-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_bin(path, rows, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".optcar-")
    try:
        with os.fdopen(fd, "wb") as stream:
            rows.tofile(stream)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomically refuses an existing output
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def convert_raw_bags(root, bags, output=None, config=None, overwrite=False,
                     progress=lambda value: None, cancelled=lambda: False):
    config = config or RawBagConfig()
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else default_output(root)
    if not root.is_dir() or not bags:
        raise ValueError("Provide a valid source folder and select at least one bag")
    paths = list(dict.fromkeys(Path(p).expanduser().resolve() for p in bags))
    for p in paths:
        if not p.is_dir() or not p.is_relative_to(root):
            raise ValueError(f"Selected bag is outside the source folder or does not exist: {p}")
        if output == p or output.is_relative_to(p):
            raise ValueError("Choose an output folder outside the source bags")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".conversion.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another conversion is writing to this output folder") from exc
        return _convert_locked(root, paths, output, config, overwrite, progress, cancelled)


def _convert_locked(root, paths, output, config, overwrite, progress, cancelled):
    manifest_path = output / "metadata/conversion.json"
    if not manifest_path.exists() and any((output / "data").rglob("*.bin")):
        raise ValueError("Output already contains bins without OptCar conversion metadata; choose a new folder")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"config": asdict(config), "bins": {}}
    # Auxiliary diagnostics may be added to existing datasets without changing the main layout.
    def main_settings(settings):
        return {key: value for key, value in settings.items() if key != "aux_twist_topic"}
    if main_settings(manifest["config"]) != main_settings(asdict(config)):
        raise ValueError("Output folder already has different conversion settings; choose a new output folder")
    manifest["config"] = asdict(config)
    metadata = [inspect_bag(p, root) for p in paths]
    weights = [max(1, sum(t["count"] for t in b["topics"] if t["topic"] in (config.pose_topic, config.twist_topic, config.command_topic, config.aux_twist_topic))) for b in metadata]
    total_weight = sum(weights)
    done_weight = 0
    started = time.monotonic()
    results = []
    current = {}
    def emit(index, bag, value):
        if value.get("phase") == "opening":
            current.clear()
        current.update(value)
        value = dict(current)
        fraction = (done_weight + weights[index] * value.get("bag_fraction", 0)) / total_weight
        elapsed = time.monotonic() - started
        progress({**value, "bag": bag["relative_path"], "bag_index": index + 1, "total_bags": len(paths),
                  "completed_bags": len(results), "percent": 100 * fraction,
                  "elapsed_s": elapsed, "eta_s": elapsed * (1 - fraction) / fraction if fraction > 0.01 else None,
                  "output": str(output), "results": list(results)})
    for index, (path, bag) in enumerate(zip(paths, metadata)):
        if cancelled():
            break
        target = output / "data" / bin_relative_path(root, path)
        aux_target = output / "aux" / bin_relative_path(root, path)
        emit(index, bag, {"phase": "opening", "bag_fraction": 0})
        try:
            if bag["error"]:
                raise ValueError(bag["error"])
            if target.exists() and not overwrite:
                raise FileExistsError("Output exists; select overwrite or choose a new output folder")
            if aux_target.exists() and not overwrite:
                raise FileExistsError("Auxiliary output exists; select overwrite or choose a new output folder")
            rows, info, aux = read_raw_bag(path, config, lambda v: emit(index, bag, v), cancelled, with_aux=True)
            _check_cancel(cancelled)
            if aux is not None:
                _write_bin(aux_target, aux, overwrite)
                info["aux"]["path"] = str(aux_target.relative_to(output))
            _write_bin(target, rows, overwrite)
            if aux is None and overwrite:
                aux_target.unlink(missing_ok=True)
            record = {**info, "source": str(path), "output": str(target), "bytes": target.stat().st_size,
                      "topics": bag["topics"], "config": asdict(config)}
            manifest["bins"][str(target.relative_to(output))] = record
            _atomic_json(manifest_path, manifest)
            results.append({"bag": bag["relative_path"], "status": "complete", "output": str(target), "samples": len(rows), "warning": info.get("aux_warning")})
        except ConversionCancelled:
            break
        except Exception as exc:
            results.append({"bag": bag["relative_path"], "status": "failed", "error": str(exc)})
        emit(index, bag, {"phase": results[-1]["status"], "bag_fraction": 1})
        done_weight += weights[index]
    if manifest["bins"]:
        features = 17 if config.include_cmd_vel else 15
        _atomic_json(output / "metadata/schema.json", {"dtype": "float32", "byte_order": "little", "features": features,
                     "quaternion_order": "wxyz", "velocity_frame": "world", "alignment": "asof",
                     "columns": ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz", "command_velocity", "command_steering"] + (["cmd_vel_throttle", "cmd_vel_steering"] if features == 17 else [])})
        if any(b.get("aux") for b in manifest["bins"].values()):
            _atomic_json(output / "metadata/aux_schema.json", {"dtype": "float32", "byte_order": "little", "features": 6,
                         "columns": ["vx", "vy", "vz", "wx", "wy", "wz"], "alignment": "same rows and timestamps as main bin",
                         "frame": "as recorded; see each bin's aux.frame_ids", "purpose": "diagnostics only"})
        max_steps = max(b["samples"] for b in manifest["bins"].values())
        (output / "metadata/dataset_config.py").write_text(
            '"""Raw dataset configuration generated by OptCar."""\nfrom dataclasses import dataclass\nfrom typing import Optional\n\n@dataclass\nclass DatasetConfig:\n'
            f'    num_envs: int = 1\n    num_runs: int = {len(manifest["bins"])}\n    data_dtype: str = "float32"\n    control_dt: float = {config.dt!r}\n    max_steps_per_run: int = {max_steps}\n    sim_dt: Optional[float] = None\n    decimation: Optional[int] = None\n\nconfig = DatasetConfig()\n')
    result = {"output": str(output), "converted": sum(r["status"] == "complete" for r in results),
              "failed": sum(r["status"] == "failed" for r in results), "cancelled": cancelled(), "results": results,
              "elapsed_s": time.monotonic() - started}
    _atomic_json(output / "metadata/last_job.json", result)
    return result
