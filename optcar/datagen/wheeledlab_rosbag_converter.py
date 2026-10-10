"""Adapted from WheeledLab's wheeledlab_data/data_processing/rosbag_converter.py.

The alignment, row stacking, dataset configuration and legacy converter below
are copied from WheeledLab. Reader adaptation: rosbags replaces rosbag2_py;
progress/cancellation are optional callbacks. Source bags are never reindexed.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
from rosbags.rosbag2 import Reader

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
        return Reader(self.bag_path)

    def _read_all_topics_once(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Read the relevant topics once, with the OptCar ROS-independent reader."""
        from optcar.datagen.raw_bags import _check_cancel, _typestore, _values
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
    def _parse_pose(msg) -> np.ndarray:
        return np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=float,
        )

    @staticmethod
    def _parse_velocity(msg) -> np.ndarray:
        return np.array(
            [
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z,
                msg.twist.angular.x,
                msg.twist.angular.y,
                msg.twist.angular.z,
            ],
            dtype=float,
        )

    @staticmethod
    def _parse_velocity_twist(msg) -> np.ndarray:
        """Parse velocity from a plain Twist message (no .twist wrapper)."""
        return np.array(
            [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z,
            ],
            dtype=float,
        )

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


class ROSBagToRawConverter:
    """High-level helper that converts ROS2 bags into WheeledLab raw datasets."""

    def __init__(
        self,
        dt: float = 0.02,
        max_duration: Optional[float] = None,
        action_shift: float = 0.0,
        topic_config: Optional[TopicConfig] = None,
        include_cmd_vel: bool = True,
    ):
        self.dt = float(dt)
        self.max_duration = None if max_duration is None else float(max_duration)
        self.action_shift = float(action_shift)
        self.topic_config = topic_config or TopicConfig()
        self.include_cmd_vel = include_cmd_vel
        if self.max_duration is not None and self.max_duration <= 0:
            raise ValueError("max_duration must be positive when provided.")

    @staticmethod
    def find_ros2_bags(root_dirs: Sequence[str]) -> List[Path]:
        """Recursively find bag directories from a list of roots."""
        bag_dirs: List[Path] = []
        for root in root_dirs:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                print(f"[ROSBagToRawConverter] Skipping missing root: {root}")
                continue
            for db3 in root_path.rglob("*.db3"):
                bag_dirs.append(db3.parent)
        return sorted(set(bag_dirs))

    def convert_bags(
        self,
        bag_dirs: Sequence[Path],
        output_root: Path,
        overwrite: bool = False,
        bag_roots: Optional[Sequence[Path]] = None,
        preserve_bag_structure: bool = False,
    ) -> List[Path]:
        """Convert a list of bag directories into .bin files."""
        bag_dirs = [Path(b) for b in bag_dirs]
        bag_roots = [Path(r).expanduser().resolve() for r in (bag_roots or [])]
        output_root = Path(output_root)
        data_dir = output_root / "data"
        metadata_dir = output_root / "metadata"

        if data_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output data directory already exists: {data_dir}. "
                    "Use --overwrite to recreate."
                )
            for existing in data_dir.rglob("*.bin"):
                existing.unlink()
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        processed_bins: List[Path] = []
        source_bags: List[Path] = []
        seen_output_files = set()
        max_timesteps = 0

        for idx, bag in enumerate(bag_dirs):
            print(f"[ROSBagToRawConverter] Processing bag {idx+1}/{len(bag_dirs)}: {bag}")
            try:
                if preserve_bag_structure:
                    output_file = data_dir / self._structured_bin_relative_path(
                        bag=bag,
                        bag_roots=bag_roots,
                    )
                else:
                    output_file = data_dir / f"{idx:05d}.bin"
                output_file = output_file.resolve()
                if output_file in seen_output_files:
                    raise RuntimeError(
                        "Multiple bags map to the same output file: "
                        f"{output_file}. Use distinct bag root names or disable "
                        "--preserve-bag-structure."
                    )
                seen_output_files.add(output_file)

                aligner = ROS2BagAligner(
                    bag_path=bag,
                    dt=self.dt,
                    max_duration=self.max_duration,
                    action_shift=self.action_shift,
                    topic_config=self.topic_config,
                    include_cmd_vel=self.include_cmd_vel,
                )
                data = aligner.process()
                combined = self._stack_rows(data)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                combined.astype(np.float32).tofile(output_file)
                processed_bins.append(output_file)
                source_bags.append(bag)
                max_timesteps = max(max_timesteps, combined.shape[0])
                rel_output = output_file.relative_to(data_dir.resolve())
                print(f"  ✓ wrote {rel_output} ({combined.shape[0]} steps)")
            except Exception as exc:
                print(f"  ✗ failed to convert {bag}: {exc}")

        if not processed_bins:
            raise RuntimeError("No ROS bags were successfully converted.")

        self._write_dataset_config(
            metadata_dir=metadata_dir,
            num_runs=len(processed_bins),
            max_steps=max_timesteps,
        )
        self._write_conversion_metadata(
            metadata_dir,
            processed_bins,
            source_bags=source_bags,
            preserve_bag_structure=preserve_bag_structure,
        )
        return processed_bins

    @staticmethod
    def _structured_bin_relative_path(bag: Path, bag_roots: Sequence[Path]) -> Path:
        """Return data-relative output path preserving the matching bag root."""
        bag = Path(bag).expanduser().resolve()
        roots = sorted((Path(r).expanduser().resolve() for r in bag_roots), key=lambda p: len(p.parts), reverse=True)

        for root in roots:
            try:
                rel_to_root = bag.relative_to(root)
            except ValueError:
                continue

            if rel_to_root == Path("."):
                return Path(f"{root.name}.bin")

            parent = Path(root.name) / rel_to_root.parent
            return parent / f"{bag.name}.bin"

        return Path(f"{bag.name}.bin")

    def _stack_rows(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack aligned arrays into a single [T, D] matrix."""
        fields = [
            data["pose"],  # 7
            data["velocity"],  # 6
            data["action_velocity"][:, None],
            data["action_steering"][:, None],
        ]
        if self.include_cmd_vel:
            fields.extend(
                [
                    data["cmd_vel_throttle"][:, None],
                    data["cmd_vel_steering"][:, None],
                ]
            )
        combined = np.concatenate(fields, axis=1)
        return combined

    def _write_dataset_config(
        self,
        metadata_dir: Path,
        num_runs: int,
        max_steps: int,
    ) -> None:
        """Write dataset_config.py for downstream processing."""
        metadata_dir.mkdir(parents=True, exist_ok=True)
        config_file = metadata_dir / "dataset_config.py"
        config_content = f'''"""Dataset configuration generated from ROS bag conversion."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for real-world dataset."""

    num_envs: int = 1
    num_runs: int = {num_runs}
    data_dtype: str = "float32"
    control_dt: float = {self.dt}
    max_steps_per_run: int = {max_steps}
    sim_dt: Optional[float] = None
    decimation: Optional[int] = None


config = DatasetConfig()
'''
        config_file.write_text(config_content)
        print(f"[ROSBagToRawConverter] Wrote dataset_config: {config_file}")

    def _write_conversion_metadata(
        self,
        metadata_dir: Path,
        processed_bins: Sequence[Path],
        source_bags: Optional[Sequence[Path]] = None,
        preserve_bag_structure: bool = False,
    ) -> None:
        """Save auxiliary information about the conversion."""
        info = {
            "dt": self.dt,
            "max_duration": self.max_duration,
            "action_shift": self.action_shift,
            "include_cmd_vel": self.include_cmd_vel,
            "preserve_bag_structure": preserve_bag_structure,
            "num_bins": len(processed_bins),
            "bags": [str(p) for p in processed_bins],
        }
        if source_bags is not None:
            info["source_bags"] = [str(p) for p in source_bags]
        info_path = metadata_dir / "rosbag_conversion.yaml"
        import yaml  # Local import to avoid global dependency if unused

        with info_path.open("w") as f:
            yaml.safe_dump(info, f)
        print(f"[ROSBagToRawConverter] Wrote conversion metadata: {info_path}")


__all__ = [
    "ROS2BagAligner",
    "ROSBagToRawConverter",
    "TopicConfig",
]
