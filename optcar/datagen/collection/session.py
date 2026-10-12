"""Dataset collection utilities for runs."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch


# Fixed ordering of dynamic bicycle model parameters tiled into raw bins
# when --save-dbm-params is on. Keep in sync with process_dataset.py.
DBM_PARAM_NAMES: tuple[str, ...] = (
    "len_f",
    "len_r",
    "C_throttle_a",
    "C_rollf",
    "C_drag",
    "C_tire_f",
    "C_tire_r",
    "wheel_time_constant",
    "mass",
    "inertia",
    "dt",
    "throttle_multiplier",
    "steering_multiplier",
    "throttle_bias",
    "steering_bias",
)


@dataclass
class DataCollectionContext:
    """Metadata describing a data collection session."""

    project_name: str
    dataset_name: str
    device_label: str
    num_envs: int
    control_dt: float
    randomization_yaml: Optional[Path]
    env_file: Optional[Path]
    env_cfg_file: Optional[Path]
    missions_file: Optional[Path]
    num_runs: int = 1
    max_steps_per_run: int = 1000
    data_dtype: str = "float32"
    sim_dt: Optional[float] = None
    decimation: Optional[int] = None
    run_type: str = "runs"
    mode: str = "sim"
    date_tag: Optional[str] = None
    copy_context_to_root: bool = False
    command_line: Optional[str] = None


class DataCollector:
    """Manage batched state/action logging and artifact capture."""

    def __init__(
        self,
        output_root: Path,
        context: DataCollectionContext,
        file_namer: Optional[Callable[[int, str], str]] = None,
        file_prefix: str = "",
        save_dbm_params: bool = False,
        dataset_dir: Optional[Path] = None,
    ) -> None:
        self._context = context
        self._output_root = output_root.expanduser().resolve()
        self._actions_buffer: list[torch.Tensor] = []
        self._state_buffer: list[torch.Tensor] = []
        self._current_run_index: Optional[int] = None
        self._save_dbm_params = bool(save_dbm_params)
        self._current_dbm_vec: Optional[np.ndarray] = None

        self._file_namer = file_namer or _default_file_namer

        project_part = _sanitize_label(context.project_name)
        mode_part = _sanitize_label(context.mode)
        base_dir = self._output_root / context.run_type / project_part / mode_part
        if context.date_tag:
            base_dir = base_dir / _sanitize_label(context.date_tag)

        dataset_subpath = _sanitize_path(context.dataset_name)
        self._dataset_dir = (dataset_dir if dataset_dir is not None else base_dir / dataset_subpath).expanduser().resolve()
        self._data_dir = self._dataset_dir / "data"
        self._metadata_dir = self._dataset_dir / "metadata"
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._copy_context_to_root = context.copy_context_to_root

        self._device_label = _sanitize_label(context.device_label)
        self._next_file_index = _find_next_index(self._data_dir, prefix=file_prefix)

        self._write_context_files()
        self._write_metadata()
        self._write_dataset_config()
        if self._context.command_line:
            (self._dataset_dir / "run.txt").write_text(self._context.command_line + "\n")

    # ------------------------------------------------------------------
    # Lifecycle --------------------------------------------------------
    # ------------------------------------------------------------------
    def start_run(self, run_index: int) -> None:
        """Begin buffering data for a specific run."""
        self._current_run_index = run_index
        self._actions_buffer.clear()
        self._state_buffer.clear()
        self._current_dbm_vec = None

    def set_dbm_params(self, params: object) -> None:
        """Capture DBM params resolved for the current run (per-run scalar values)."""
        if not self._save_dbm_params or params is None:
            return

        # Aliases: some params dataclasses split steering into pos/neg variants.
        # Take the positive-side scalar as the canonical value (typical setup
        # samples both sides identically).
        aliases = {
            "steering_multiplier": ("steering_multiplier", "steer_pos_multiplier"),
            "steering_bias": ("steering_bias", "steer_pos_bias"),
        }

        def _lookup(name: str):
            for candidate in aliases.get(name, (name,)):
                val = getattr(params, candidate, None)
                if val is None and isinstance(params, dict):
                    val = params.get(candidate, None)
                if val is not None:
                    return val
            return None

        vec = np.zeros(len(DBM_PARAM_NAMES), dtype=np.float32)
        for i, name in enumerate(DBM_PARAM_NAMES):
            val = _lookup(name)
            if val is None:
                print(f"[DATA] Warning: DBM param '{name}' not on resolved params; saving 0.0")
                val = 0.0
            vec[i] = float(val)
        self._current_dbm_vec = vec

    def record_step(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor) -> None:
        """Append a single step of state/action data."""
        if self._current_run_index is None:
            raise RuntimeError("DataCollector.start_run must be called before recording data.")

        if state_tensor.ndim != 2:
            raise ValueError("state_tensor must have shape (num_envs, state_dim)")
        if action_tensor.ndim != 2:
            raise ValueError("action_tensor must have shape (num_envs, action_dim)")

        self._state_buffer.append(state_tensor.detach().cpu())
        self._actions_buffer.append(action_tensor.detach().cpu())

    def finish_run(self) -> Optional[Path]:
        """Persist the buffered data for the active run."""
        if self._current_run_index is None or not self._state_buffer:
            return None

        states = torch.stack(self._state_buffer, dim=0)  # (T, N, D_s)
        actions = torch.stack(self._actions_buffer, dim=0)  # (T, N, D_a)

        # Reorder to (N, T, D)
        states = states.permute(1, 0, 2)
        actions = actions.permute(1, 0, 2)

        combined = torch.cat([states, actions], dim=2)
        output = combined.detach().cpu().numpy().astype(np.float32)

        if self._save_dbm_params:
            if self._current_dbm_vec is None:
                raise RuntimeError(
                    "save_dbm_params=True but no DBM params were set for this run. "
                    "Call DataCollector.set_dbm_params() before finish_run()."
                )
            N, T, _ = output.shape
            dbm_tile = np.broadcast_to(self._current_dbm_vec, (N, T, len(DBM_PARAM_NAMES)))
            output = np.concatenate([output, dbm_tile.astype(np.float32, copy=False)], axis=2)

        file_index = self._next_file_index
        self._next_file_index += 1

        file_name = self._file_namer(file_index, self._device_label)
        file_path = self._data_dir / file_name  # Save to data/ subdirectory
        output.tofile(file_path)

        self._current_run_index = None
        self._actions_buffer.clear()
        self._state_buffer.clear()

        return file_path

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    def finalize(self, hydra_output_dir: Optional[Path] = None) -> None:
        """Save Hydra logs and finalize dataset.
        
        Args:
            hydra_output_dir: Optional path to Hydra output directory to copy logs from
        """
        if hydra_output_dir is not None and hydra_output_dir.exists():
            self._copy_hydra_logs(hydra_output_dir)
    
    def _copy_hydra_logs(self, hydra_output_dir: Path) -> None:
        """Copy Hydra logs to metadata directory."""
        try:
            hydra_dest = self._metadata_dir / "hydra_logs"
            shutil.copytree(hydra_output_dir, hydra_dest, dirs_exist_ok=True)
            print(f"[DATA] Hydra logs saved to {hydra_dest}")
        except Exception as exc:
            print(f"[DATA] Failed to copy Hydra logs: {exc}")

    # ------------------------------------------------------------------
    # Helpers ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _write_context_files(self) -> None:
        mapping: Sequence[tuple[Optional[Path], str]] = [
            (self._context.randomization_yaml, "randomization.yaml"),
            (self._context.missions_file, "custom_missions.py"),
            (self._context.env_file, "env.py"),
            (self._context.env_cfg_file, "env_config.py"),
        ]

        for source, target_name in mapping:
            if source is None:
                continue
            src_path = Path(source)
            if not src_path.exists():
                continue
            dest = self._metadata_dir / target_name
            try:
                shutil.copy2(src_path, dest)
                if self._copy_context_to_root:
                    shutil.copy2(src_path, self._dataset_dir / target_name)
            except shutil.SameFileError:
                continue

    def _write_metadata(self) -> None:
        metadata = {
            "project": self._context.project_name,
            "dataset": self._context.dataset_name,
            "device": self._device_label,
            "num_envs": self._context.num_envs,
            "control_dt": self._context.control_dt,
            "num_runs": self._context.num_runs,
            "max_steps_per_run": self._context.max_steps_per_run,
            "data_dtype": self._context.data_dtype,
            "sim_dt": self._context.sim_dt,
            "decimation": self._context.decimation,
            "run_type": self._context.run_type,
            "mode": self._context.mode,
            "date_tag": self._context.date_tag,
            "command_line": self._context.command_line,
            "save_dbm_params": self._save_dbm_params,
            "dbm_param_names": list(DBM_PARAM_NAMES) if self._save_dbm_params else [],
        }
        meta_path = self._metadata_dir / "metadata.json"  # Save to metadata/ subdirectory
        meta_path.write_text(json.dumps(metadata, indent=2))

    def _write_dataset_config(self) -> None:
        """Write dataset_config.py with dataclass containing dataset parameters."""
        sim_dt_line = f"sim_dt: float = {self._context.sim_dt}" if self._context.sim_dt is not None else "sim_dt: Optional[float] = None"
        decimation_line = f"decimation: int = {self._context.decimation}" if self._context.decimation is not None else "decimation: Optional[int] = None"
        
        config_content = f'''"""Dataset configuration generated from Hydra config."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters after Hydra overrides."""
    
    num_envs: int = {self._context.num_envs}
    num_runs: int = {self._context.num_runs}
    data_dtype: str = "{self._context.data_dtype}"
    control_dt: float = {self._context.control_dt}
    max_steps_per_run: int = {self._context.max_steps_per_run}
    {sim_dt_line}
    {decimation_line}
    
    @property
    def total_run_length(self) -> float:
        """Total time duration of a single run in seconds."""
        return self.max_steps_per_run * self.control_dt
    
    @property
    def total_timesteps(self) -> int:
        """Total number of timesteps across all runs and environments."""
        return self.num_runs * self.num_envs * self.max_steps_per_run


# Instantiate default config
config = DatasetConfig()
'''
        config_path = self._metadata_dir / "dataset_config.py"
        config_path.write_text(config_content)


# ----------------------------------------------------------------------
# Utility helpers ------------------------------------------------------
# ----------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    if not label:
        return "device"
    cleaned = re.sub(r"[^0-9a-zA-Z_=.-]+", "_", label)
    return cleaned.strip("_") or "device"


def _sanitize_path(path_str: str) -> Path:
    parts = Path(path_str).parts if path_str else ("dataset",)
    sanitized = [_sanitize_label(part) for part in parts if part not in ("", ".")]
    return Path(*sanitized) if sanitized else Path("dataset")


def _find_next_index(dataset_dir: Path, prefix: str = "") -> int:
    pattern = re.compile(r"(\d+)\.bin$")
    max_idx = -1
    for file in dataset_dir.glob("*.bin"):
        if prefix and not file.name.startswith(prefix):
            continue
        match = pattern.search(file.name)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)
    return max_idx + 1 if max_idx >= 0 else 0


def _default_file_namer(index: int, device_label: str) -> str:
    return f"{device_label}_{index:010d}.bin"
