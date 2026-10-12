"""Lightning backbone utilities for FKD models."""

from __future__ import annotations

import importlib
import logging
import threading
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib.colors import Normalize
from torch import nn
from torch import optim
from tqdm.auto import tqdm

try:  # optional dependency
    import wandb  # type: ignore
except:  # pragma: no cover
    wandb = None  # type: ignore

# try:  # prefer WheeledLab config if available
from .learning_config import SupervisedLearningConfig

from .config import FKD_model_config, resolve_dtype

from optcar.utils.filter import (
    quaternion_continuity_align,
    quaternion_continuity_align_torchscript,
)
from optcar.utils.math import qmultiply_torch, qrotate_torch
from optcar.utils.frame_transforms import reconstruct_absolute_from_deltas

# qmultiply_torch = getattr(_math_module, "qmultiply_torch")
# qrotate_torch = getattr(_math_module, "qrotate_torch")

warnings.filterwarnings("ignore", category=UserWarning)

norm_pos_cmap = Normalize(vmin=0, vmax=3)
norm_quat_cmap = Normalize(vmin=0, vmax=2)
cmap_bg = plt.cm.jet
# Supported normalization strategies. Data-driven strategies are populated by
# collect_normalization_stats() before training starts.
_NORMALIZATION_STRATEGIES = {"none", "mean_std", "min_max", "hard_min_max", "hard_mean_std", "fnorm"}
_STATE_COMPONENTS_13 = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "vel_x", "vel_y", "vel_z",
    "ang_x", "ang_y", "ang_z",
]
_STATE_COMPONENTS_7 = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
]
_ACTION_COMPONENTS_2 = ["throttle", "steer"]


def _default_learning_config() -> SimpleNamespace:
    return SimpleNamespace()


def _coerce_learning_config(cfg: Optional[Any]) -> Any:
    if cfg is None:
        if SupervisedLearningConfig is not None:
            return SupervisedLearningConfig()  # type: ignore[call-arg]
        return _default_learning_config()
    if isinstance(cfg, dict):
        if SupervisedLearningConfig is not None:
            return SupervisedLearningConfig(**cfg)  # type: ignore[arg-type]
        return SimpleNamespace(**cfg)  # type: ignore[arg-type]
    return cfg  # type: ignore[return-value]

# class TimerBlock:
#     def __init__(self, name: str):
#         # from timing_utils import ENABLE_TIMING  # Avoid circular import
#         self.name = name
#         self.enabled = ENABLE_TIMING
#         self.start_time = None

#     def __enter__(self):
#         if self.enabled:
#             torch.cuda.synchronize()
#             self.start_time = time.time()

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.enabled and self.start_time is not None:
#             torch.cuda.synchronize()
#             elapsed_sec = time.time() - self.start_time
#             if self.name not in TIMER_RESULTS:
#                 TIMER_RESULTS[self.name] = []
#             TIMER_RESULTS[self.name] = elapsed_sec

def set_axes_equal(ax):
    """
    Set equal scales for 3D plots so that x, y, and z axes have the same scale.
    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D plot axes.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    center = np.mean(limits, axis=1)
    max_range = np.max(limits[:, 1] - limits[:, 0]) / 2

    ax.set_xlim3d([center[0] - max_range, center[0] + max_range])
    ax.set_ylim3d([center[1] - max_range, center[1] + max_range])
    ax.set_zlim3d([-0.5, center[2] + max_range])


# ENABLE_TIMING = False
# TIMER_RESULTS: Dict[str, float] = {}


# class TimerBlock:
#     def __init__(self, name: str):
#         self.name = name
#         self.enabled = ENABLE_TIMING
#         self.start_time: Optional[float] = None

#     def __enter__(self):
#         if self.enabled:
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             self.start_time = time.time()

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.enabled and self.start_time is not None:
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             elapsed_sec = time.time() - self.start_time
#             TIMER_RESULTS[self.name] = elapsed_sec


def _component_names_for_field(field: str, dim: int) -> List[str]:
    """Return ordered component names given a tensor width."""
    if field in ("history", "future", "delta_history", "delta_future"):
        if dim == 13:
            return _STATE_COMPONENTS_13
        if dim == 7:
            return _STATE_COMPONENTS_7
    if field in ("history_actions", "future_actions") and dim == 2:
        return _ACTION_COMPONENTS_2
    return [f"dim{i}" for i in range(dim)]


class NormalizationGroup(nn.Module):
    """Applies a specific normalization strategy to a set of indices."""

    def __init__(
        self,
        name: str,
        strategy: str,
        indices: List[int],
        eps: float = 1e-6,
        fixed_min: Optional[torch.Tensor] = None,
        fixed_max: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.name = name
        self.strategy = strategy
        self.eps = eps
        self.register_buffer("_indices", torch.tensor(indices, dtype=torch.long), persistent=False)

        if strategy == "mean_std":
            # Mean/std will be populated from dataset statistics.
            self.register_buffer("mean", torch.zeros(len(indices)))
            self.register_buffer("std", torch.ones(len(indices)))
        elif strategy == "hard_mean_std":
            # Mean/std are provided explicitly (treated analogously to hard_min_max).
            if fixed_min is not None and fixed_max is not None:
                self.register_buffer("mean", fixed_min.clone().detach())
                self.register_buffer("std", fixed_max.clone().detach())
            else:
                self.register_buffer("mean", torch.zeros(len(indices)))
                self.register_buffer("std", torch.ones(len(indices)))
        elif strategy in ("min_max", "hard_min_max"):
            if fixed_min is not None and fixed_max is not None:
                self.register_buffer("min", fixed_min.clone().detach())
                self.register_buffer("max", fixed_max.clone().detach())
            else:
                self.register_buffer("min", torch.zeros(len(indices)))
                self.register_buffer("max", torch.ones(len(indices)))
        else:
            # Buffers kept for a consistent state dict even when no stats are needed
            self.register_buffer("dummy", torch.tensor(0.0), persistent=False)

    @property
    def indices(self) -> torch.Tensor:
        return self._indices

    def set_mean_std(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if hasattr(self, "mean"):
            self.mean.copy_(mean)
        if hasattr(self, "std"):
            self.std.copy_(std)

    def set_min_max(self, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        if hasattr(self, "min"):
            self.min.copy_(min_val)
        if hasattr(self, "max"):
            self.max.copy_(max_val)

    def apply(self, tensor: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Normalize or unnormalize in-place."""
        if self.strategy == "none":
            return tensor

        idx = self.indices
        if idx.device != tensor.device:
            idx = idx.to(tensor.device)
        stats_device = tensor.device
        stats_dtype = tensor.dtype
        eps_tensor = torch.as_tensor(self.eps, device=stats_device, dtype=stats_dtype)
        if self.strategy in ("mean_std", "hard_mean_std"):
            mean = self.mean.to(device=stats_device, dtype=stats_dtype)
            std = self.std.to(device=stats_device, dtype=stats_dtype)
            if inverse:
                tensor[..., idx] = tensor[..., idx] * std + mean
            else:
                tensor[..., idx] = (tensor[..., idx] - mean) / (std + eps_tensor)
        elif self.strategy == "min_max":
            min_val = self.min.to(device=stats_device, dtype=stats_dtype)
            max_val = self.max.to(device=stats_device, dtype=stats_dtype)
            rng = (max_val - min_val).clamp_min(eps_tensor)
            if inverse:
                tensor[..., idx] = tensor[..., idx] * rng + min_val
            else:
                tensor[..., idx] = (tensor[..., idx] - min_val) / rng
        elif self.strategy == "hard_min_max":
            min_val = self.min.to(device=stats_device, dtype=stats_dtype)
            max_val = self.max.to(device=stats_device, dtype=stats_dtype)
            rng = (max_val - min_val).clamp_min(eps_tensor)
            if inverse:
                tensor[..., idx] = (tensor[..., idx] + 1.0) * 0.5 * rng + min_val
            else:
                tensor[..., idx] = ((tensor[..., idx] - min_val) / rng) * 2.0 - 1.0
        elif self.strategy == "fnorm":
            # Normalize to unit norm; inverse simply re-normalizes.
            slice_tensor = tensor[..., idx]
            norm = torch.linalg.norm(slice_tensor, dim=-1, keepdim=True)
            norm = norm.to(dtype=stats_dtype).clamp_min(eps_tensor)
            tensor[..., idx] = slice_tensor / norm
        return tensor


class FieldNormalizer(nn.Module):
    """Field-level normalizer built from a config mapping."""

    def __init__(
        self,
        field: str,
        component_names: List[str],
        strategy_map: Dict[str, str],
        field_bounds: Optional[Dict[str, Any]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.field = field
        self.component_names = component_names
        self.eps = eps
        self.field_bounds = field_bounds or {}

        self.groups = nn.ModuleDict()
        self._comp_to_group: Dict[str, str] = {}
        self._group_needs_stats: Dict[str, bool] = {}
        self._accum: Dict[str, Dict[str, torch.Tensor]] = {}

        group_to_indices: Dict[str, List[int]] = {}
        group_to_strategy: Dict[str, str] = {}

        for idx, comp in enumerate(component_names):
            # Resolve strategy or alias (fallback to "none")
            root_group, strategy = self._resolve_strategy(comp, strategy_map)
            group_to_indices.setdefault(root_group, []).append(idx)
            group_to_strategy[root_group] = strategy
            self._comp_to_group[comp] = root_group

        for group_name, indices in group_to_indices.items():
            strategy = group_to_strategy[group_name]

            # Safety: fnorm should typically operate on a grouped vector
            # (e.g., quaternion w/x/y/z). Warn if configured on a single dim.
            if strategy == "fnorm" and len(indices) <= 1:
                logging.warning(
                    "FieldNormalizer for field '%s' has fnorm group '%s' with only %d component(s). "
                    "Check normalization config (consider aliasing related components to the same key).",
                    field,
                    group_name,
                    len(indices),
                )

            bounds = self._resolve_bounds(group_name)
            self.groups[group_name] = NormalizationGroup(
                name=f"{field}:{group_name}",
                strategy=strategy,
                indices=indices,
                eps=eps,
                fixed_min=bounds[0],
                fixed_max=bounds[1],
            )
            # Only strategies that estimate statistics from data need accumulation.
            self._group_needs_stats[group_name] = strategy in ("mean_std", "min_max")

        self.reset_accumulators()

    def _resolve_strategy(self, comp: str, strategy_map: Dict[str, str], visited: Optional[set] = None):
        if visited is None:
            visited = set()
        if comp in visited:
            raise ValueError(f"Circular normalization alias detected for component '{comp}' in field '{self.field}'.")
        visited.add(comp)

        raw = strategy_map.get(comp, "none")
        if raw in _NORMALIZATION_STRATEGIES:
            return comp, raw
        if raw not in strategy_map:
            # Unknown alias; treat as none for safety
            return comp, "none"
        return self._resolve_strategy(raw, strategy_map, visited)

    def _resolve_bounds(self, group: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return fixed min/max tensors for hard_min_max if provided."""
        if group not in self.field_bounds:
            return (None, None)
        bounds = self.field_bounds[group]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            return (None, None)
        min_val, max_val = bounds
        return (
            torch.as_tensor(min_val, dtype=torch.float32),
            torch.as_tensor(max_val, dtype=torch.float32),
        )

    def reset_accumulators(self):
        self._accum = {}
        for name, group in self.groups.items():
            if group.strategy not in ("mean_std", "min_max"):
                continue
            self._accum[name] = {
                "count": 0,
                "sum": None,
                "sumsq": None,
                "min": None,
                "max": None,
            }

    def accumulate(self, tensor: torch.Tensor):
        """Gather running statistics from an unnormalized tensor."""
        with torch.no_grad():
            for name, group in self.groups.items():
                if group.strategy not in ("mean_std", "min_max"):
                    continue
                idx = group.indices
                if idx.device != tensor.device:
                    idx = idx.to(tensor.device)
                slice_tensor = tensor[..., idx]
                flat = slice_tensor.reshape(-1, slice_tensor.shape[-1])
                if flat.numel() == 0:
                    continue

                acc = self._accum[name]
                acc["count"] += flat.shape[0]
                sum_now = flat.sum(dim=0)
                sumsq_now = (flat ** 2).sum(dim=0)
                acc["sum"] = sum_now if acc["sum"] is None else acc["sum"] + sum_now
                acc["sumsq"] = sumsq_now if acc["sumsq"] is None else acc["sumsq"] + sumsq_now

                min_now = flat.min(dim=0).values
                max_now = flat.max(dim=0).values
                acc["min"] = min_now if acc["min"] is None else torch.minimum(acc["min"], min_now)
                acc["max"] = max_now if acc["max"] is None else torch.maximum(acc["max"], max_now)

    def finalize(self):
        """Convert accumulators into buffers on each group."""
        for name, group in self.groups.items():
            if group.strategy not in ("mean_std", "min_max"):
                continue
            acc = self._accum.get(name)
            if acc is None or acc["count"] == 0:
                continue
            if group.strategy == "mean_std":
                count = float(acc["count"])
                mean = acc["sum"] / count
                var = acc["sumsq"] / count - mean * mean
                std = torch.sqrt(var.clamp_min(self.eps))
                group.set_mean_std(
                    mean.to(device=group.mean.device, dtype=group.mean.dtype),
                    std.to(device=group.std.device, dtype=group.std.dtype),
                )
            elif group.strategy == "min_max":
                group.set_min_max(
                    acc["min"].to(device=group.min.device, dtype=group.min.dtype),
                    acc["max"].to(device=group.max.device, dtype=group.max.dtype),
                )

    def apply(self, tensor: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        for group in self.groups.values():
            group.apply(tensor, inverse=inverse)
        return tensor

    def needs_statistics(self) -> bool:
        return any(self._group_needs_stats.values())

    def describe_statistics(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for group_name, group in self.groups.items():
            idx_list = group.indices.detach().cpu().tolist()
            comp_names = [
                self.component_names[i] if i < len(self.component_names) else f"dim{i}"
                for i in idx_list
            ]
            stats: Dict[str, Any] = {}
            if hasattr(group, "mean"):
                stats["mean"] = group.mean.detach().cpu().tolist()
            if hasattr(group, "std"):
                stats["std"] = group.std.detach().cpu().tolist()
            if hasattr(group, "min"):
                stats["min"] = group.min.detach().cpu().tolist()
            if hasattr(group, "max"):
                stats["max"] = group.max.detach().cpu().tolist()
            summary.append(
                {
                    "group": group_name,
                    "strategy": group.strategy,
                    "components": comp_names,
                    "stats": stats,
                }
            )
        return summary


def quat_ambiguity_resolved_loss(pred_q, target_q, mode='mse', delta=0.5):
    # pred_q   = F.normalize(pred_q, dim=-1, eps=1e-8)
    # target_q = F.normalize(target_q, dim=-1, eps=1e-8)

    if mode == 'mse':
        lp = F.mse_loss(pred_q,  target_q, reduction='none')
        ln = F.mse_loss(pred_q, -target_q, reduction='none')
        return torch.minimum(lp, ln).mean()

    elif mode == 'mae':
        lp = F.l1_loss(pred_q,  target_q, reduction='none')
        ln = F.l1_loss(pred_q, -target_q, reduction='none')
        return torch.minimum(lp, ln).mean()

    elif mode == 'mae_individual':
        lp = F.l1_loss(pred_q,  target_q, reduction='none')
        ln = F.l1_loss(pred_q, -target_q, reduction='none')
        return torch.minimum(lp, ln).mean(dim=(0, 1))

    else:
        raise ValueError("Unknown loss mode")


def quat_geodesic_loss(pred_q, target_q, reduction='mean', eps=1e-7):
    # Normalize
    pred_q = F.normalize(pred_q, dim=-1, eps=eps)
    target_q = F.normalize(target_q, dim=-1, eps=eps)

    # Dot in [-1,1], abs for double cover
    dot = torch.sum(pred_q * target_q, dim=-1).abs()

    # Clamp to avoid sqrt(negative) and atan2(0/0)
    dot = dot.clamp(min=eps, max=1.0 - eps)

    # Safe computation
    sin_half = torch.sqrt((1.0 - dot * dot).clamp(min=0.0))
    geo = 2.0 * torch.atan2(sin_half, dot)

    if reduction == 'mean':
        return geo.mean()
    elif reduction == 'sum':
        return geo.sum()
    return geo


class TrajectoryLosses:
    def __init__(self, discount_factor: float = 0.998):
        self.discount_factor = discount_factor

    def __call__(self, pred, gt, pred_log_var=None, is_quaternion=False):
        if is_quaternion:
            mse_loss = quat_ambiguity_resolved_loss(pred, gt, mode='mse')
            geodesic_loss = quat_geodesic_loss(pred, gt, reduction="mean")
            mae_individual_loss = quat_ambiguity_resolved_loss(pred, gt, mode='mae_individual')
        else:
            mse_loss = F.mse_loss(pred, gt)
            mae_individual_loss = F.l1_loss(pred, gt, reduction='none').mean(dim=(0, 1))

        aleatoric_uncertainty_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if pred_log_var is not None:
            var = torch.exp(pred_log_var)
            mse_unc = ((gt - pred) ** 2) / var
            reg_loss = pred_log_var
            aleatoric_uncertainty_loss = 0.5 * (mse_unc + reg_loss).mean()

        loss_dict = {
            "mse_loss": mse_loss,
            "mae_individual_loss": mae_individual_loss,
            "aleatoric_uncertainty_loss": aleatoric_uncertainty_loss,
        }

        if is_quaternion:
            loss_dict["geodesic_loss"] = geodesic_loss
        
        return loss_dict


class FKDLightningModule(L.LightningModule):
    def __init__(
        self,
        model_config: FKD_model_config,
        learning_config: Optional[Any] = None,
        dataset_size : Optional[int] = 1e6,
        eval: bool = False,
        eval_t_range=None,
        eval_images: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.l_cfg = _coerce_learning_config(learning_config)
        self.dataset_size = dataset_size
        
        # Set device from model config
        # CONFLICT WITH LIGHTNING MODULE DEVICE SETTING
        if hasattr(self.model_config, 'device'):
            print(self.model_config.device)
            if isinstance(self.model_config.device, int):
                self._model_device = f'cuda:{self.model_config.device}' if torch.cuda.is_available() else 'cpu'
            else:
                self._model_device = self.model_config.device
        else:
            self._model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model_dtype = resolve_dtype(getattr(self.model_config, "dtype", None))

        self.is_eval = eval
        self.eval_t_range = eval_t_range or [0, 150]
        self.eval_images = eval_images
        self.aleatoric_uncertainty = getattr(self.model_config, "aleatoric_uncertainty", False)
        self._field_normalizers = nn.ModuleDict()
        self._normalization_initialized = False
        self._build_normalizers_from_config_dims()

        self._component_specs = {
            "pos": {
                "slice": slice(0, 3),
                "is_quat": False,
                "prefix": "tr",
                "axis_labels": ["X", "Y", "Z"],
                "weights": {
                    "abs": {
                        "mse": "T_MSE_factor",
                        "aleatoric": "T_Aleatoric_factor",
                        "mae": ["T_MAE_factor_x", "T_MAE_factor_y", "T_MAE_factor_z"],
                    },
                    "delta": {
                        "mse": "T_DELTA_MSE_factor",
                        "aleatoric": "T_DELTA_Aleatoric_factor",
                        "mae": ["T_DELTA_MAE_factor_x", "T_DELTA_MAE_factor_y", "T_DELTA_MAE_factor_z"],
                    },
                },
            },
            "quat": {
                "slice": slice(3, 7),
                "is_quat": True,
                "prefix": "rot",
                "axis_labels": ["W", "X", "Y", "Z"],
                "weights": {
                    "abs": {
                        "mse": "R_MSE_factor",
                        "aleatoric": "R_Aleatoric_factor",
                        "mae": ["R_MAE_factor_w", "R_MAE_factor_x", "R_MAE_factor_y", "R_MAE_factor_z"],
                        "geodesic": "R_Geodesic_factor",
                    },
                    "delta": {
                        "mse": "R_DELTA_MSE_factor",
                        "aleatoric": "R_DELTA_Aleatoric_factor",
                        "mae": ["R_DELTA_MAE_factor_w", "R_DELTA_MAE_factor_x", "R_DELTA_MAE_factor_y", "R_DELTA_MAE_factor_z"],
                        "geodesic": "R_DELTA_Geodesic_factor",
                    },
                },
            },
            "vel": {
                "slice": slice(7, 10),
                "is_quat": False,
                "prefix": "vel",
                "axis_labels": ["X", "Y", "Z"],
                "weights": {
                    "abs": {
                        "mse": "T_Vel_MSE_factor",
                        "aleatoric": "T_Vel_Aleatoric_factor",
                        "mae": ["T_Vel_MAE_factor_x", "T_Vel_MAE_factor_y", "T_Vel_MAE_factor_z"],
                    },
                    "delta": {
                        "mse": "T_DELTA_Vel_MSE_factor",
                        "aleatoric": "T_DELTA_Vel_Aleatoric_factor",
                        "mae": ["T_DELTA_Vel_MAE_factor_x", "T_DELTA_Vel_MAE_factor_y", "T_DELTA_Vel_MAE_factor_z"],
                    },
                },
            },
            "angvel": {
                "slice": slice(10, 13),
                "is_quat": False,
                "prefix": "omg",
                "axis_labels": ["X", "Y", "Z"],
                "weights": {
                    "abs": {
                        "mse": "R_Vel_MSE_factor",
                        "aleatoric": "R_Vel_Aleatoric_factor",
                        "mae": ["R_Vel_MAE_factor_x", "R_Vel_MAE_factor_y", "R_Vel_MAE_factor_z"],
                    },
                    "delta": {
                        "mse": "R_DELTA_Vel_MSE_factor",
                        "aleatoric": "R_DELTA_Vel_Aleatoric_factor",
                        "mae": ["R_DELTA_Vel_MAE_factor_x", "R_DELTA_Vel_MAE_factor_y", "R_DELTA_Vel_MAE_factor_z"],
                    },
                },
            },
        }

        def _slice_to_tuple(s: slice) -> Tuple[int, int]:
            start = 0 if s.start is None else s.start
            stop = start if s.stop is None else s.stop
            return (start, stop)

        decoder_include_twist = getattr(self.model_config, "decoder_include_twist", False)
        self._active_components: List[str] = ["pos", "quat"]
        if decoder_include_twist:
            self._active_components.extend(["vel", "angvel"])

        self._pred_component_slices: Dict[str, Tuple[int, int]] = {
            comp: _slice_to_tuple(spec["slice"])
            for comp, spec in self._component_specs.items()
        }

        self._component_total_labels = {
            "tr": "translation_loss",
            "rot": "rotation_loss",
            "vel": "velocity_loss",
            "omg": "angular_velocity_loss",
        }

        self.Loss_function = TrajectoryLosses(getattr(self.l_cfg, "discount_gamma", 0.998))

        self.plot_val_epoch_freq = kwargs.get("plot_val_epoch_freq", 1)
        self.plot_val_batch_freq = kwargs.get("plot_val_batch_freq", 1)
        self.plot_test_epoch_freq = kwargs.get("plot_test_epoch_freq", 1)
        self.plot_test_batch_freq = kwargs.get("plot_test_batch_freq", 1)
        self.no_plot = kwargs.get("no_plot", False)

    def _normalization_enabled(self) -> bool:
        norm_cfg = self.model_config.normalization
        return isinstance(norm_cfg, dict) and len(norm_cfg) > 0

    def _build_normalizers_from_config_dims(self) -> None:
        """Build normalizers from configured dims to align with checkpoints."""
        if self._normalization_initialized or not self._normalization_enabled():
            return
        norm_cfg: Dict[str, Dict[str, str]] = self.model_config.normalization
        bounds_cfg: Dict[str, Dict[str, Any]] = getattr(self.model_config, "normalization_bounds", {})
        field_dims = {
            "history": self.model_config.input_state_dims,
            "future": self.model_config.output_state_dims,
            "delta_history": self.model_config.input_state_dims,
            "delta_future": self.model_config.output_state_dims,
            "history_actions": self.model_config.action_dims,
            "future_actions": self.model_config.action_dims,
        }
        for field, dim in field_dims.items():
            if dim is None:
                continue
            comps = _component_names_for_field(field, dim)
            field_cfg = norm_cfg.get(field, {})
            field_bounds = bounds_cfg.get(field, {})
            self._field_normalizers[field] = FieldNormalizer(field, comps, field_cfg, field_bounds=field_bounds)
        self._normalization_initialized = True

    def _build_normalizers_from_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        if not self._normalization_enabled():
            return
        if self._normalization_initialized:
            for field in ["history", "future", "delta_history", "delta_future", "history_actions", "future_actions"]:
                tensor = batch.get(field)
                if tensor is None:
                    continue
                if field in self._field_normalizers:
                    norm = self._field_normalizers[field]
                    if len(norm.component_names) != tensor.shape[-1]:
                        # If tensor has MORE dims than expected, it's fine (we just normalize the prefix)
                        if tensor.shape[-1] > len(norm.component_names):
                            pass
                        else:
                            # Rebuild with the correct dimensionality
                            self._field_normalizers = nn.ModuleDict()
                            self._normalization_initialized = False
                            break
            else:
                return
        norm_cfg: Dict[str, Dict[str, str]] = self.model_config.normalization
        bounds_cfg: Dict[str, Dict[str, Any]] = getattr(self.model_config, "normalization_bounds", {})
        for field in ["history", "future", "delta_history", "delta_future", "history_actions", "future_actions"]:
            tensor = batch.get(field)
            if tensor is None:
                continue
            dim = tensor.shape[-1]
            comps = _component_names_for_field(field, dim)
            field_cfg = norm_cfg.get(field, {})
            field_bounds = bounds_cfg.get(field, {})
            self._field_normalizers[field] = FieldNormalizer(field, comps, field_cfg, field_bounds=field_bounds)
        self._normalization_initialized = True

    def _reset_normalizer_accumulators(self) -> None:
        for normalizer in self._field_normalizers.values():
            normalizer.reset_accumulators()

    def _apply_field_normalization(self, batch: Dict[str, torch.Tensor], inverse: bool = False) -> None:
        if not self._normalization_enabled():
            return
        if not self._normalization_initialized:
            self._build_normalizers_from_batch(batch)
        for field, normalizer in self._field_normalizers.items():
            tensor = batch.get(field)
            if tensor is None:
                continue
            normalizer.apply(tensor, inverse=inverse)

    def _print_normalization_stats(self) -> None:
        if not self._field_normalizers:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] INFO: Normalization statistics summary:")
        rows = []
        for field, normalizer in self._field_normalizers.items():
            descriptions = normalizer.describe_statistics()
            if not descriptions:
                rows.append((field, "-", "-", "no normalization groups configured"))
                continue
            for desc in descriptions:
                stats = desc.get("stats", {})
                if stats:
                    parts = []
                    for stat_name, values in stats.items():
                        arr = np.asarray(values, dtype=np.float32)
                        arr_str = np.array2string(arr, precision=4, floatmode="fixed")
                        parts.append(f"{stat_name}={arr_str}")
                    stats_text = "; ".join(parts)
                else:
                    stats_text = "(no tracked statistics for this strategy)"
                comp_str = ", ".join(desc["components"])
                rows.append(
                    (
                        field,
                        f"{desc['group']} [{comp_str}]",
                        desc["strategy"],
                        stats_text,
                    )
                )
        if not rows:
            print("  (no normalization statistics collected)")
            return
        col_field = max(len("Field"), max(len(r[0]) for r in rows))
        col_group = max(len("Group"), max(len(r[1]) for r in rows))
        col_strategy = max(len("Strategy"), max(len(r[2]) for r in rows))
        header = f"  {'Field':<{col_field}}  {'Group':<{col_group}}  {'Strategy':<{col_strategy}}  Stats"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for field, group, strategy, stats_text in rows:
            print(f"  {field:<{col_field}}  {group:<{col_group}}  {strategy:<{col_strategy}}  {stats_text}")

    def collect_normalization_stats(self, dataloader, max_batches: Optional[int] = None) -> None:
        """One pass over a dataloader to populate mean/std/min/max buffers."""
        if not self._normalization_enabled():
            return
        if not any(norm.needs_statistics() for norm in self._field_normalizers.values()):
            return
        try:
            total_batches = len(dataloader)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total_batches = None
        progress_total = total_batches
        if max_batches is not None:
            progress_total = max_batches if progress_total is None else min(progress_total, max_batches)
        progress_bar = tqdm(
            total=progress_total,
            desc="Collecting normalization stats",
            leave=False,
        )
        accum_initialized = False
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    if not self._normalization_initialized:
                        self._build_normalizers_from_batch(batch)
                    if not accum_initialized:
                        if not any(norm.needs_statistics() for norm in self._field_normalizers.values()):
                            return
                        self._reset_normalizer_accumulators()
                        accum_initialized = True
                    for field, normalizer in self._field_normalizers.items():
                        tensor = batch.get(field)
                        if tensor is None:
                            continue
                        normalizer.accumulate(tensor.detach())
                    progress_bar.update(1)
        finally:
            progress_bar.close()
        for normalizer in self._field_normalizers.values():
            normalizer.finalize()
        self._print_normalization_stats()

    def pre_process_data(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Work on references (no clones). We’ll mutate these in-place.
        processed_batch_history          = batch["history"]
        processed_batch_history_actions  = batch["history_actions"]
        processed_batch_future_actions   = batch["future_actions"]
        processed_batch_hmap             = batch["hmap"]
        processed_batch_quat_wrt_world   = batch["quat_wrt_world"]
        processed_batch_vel_wrt_world    = batch["vel_wrt_world"]
        processed_batch_angvel_wrt_world = batch["angvel_wrt_world"]

        processed_batch_delta_history = None
        processed_batch_delta_future  = None
        if "delta_history" in batch:
            processed_batch_delta_history = batch["delta_history"]
            processed_batch_delta_future  = batch["delta_future"]

        processed_batch_future = batch.get("future", None)

        processed_batch_cmd_vel_history = None
        if self.model_config.input_cmd_vel:
            processed_batch_cmd_vel_history = batch["history_cmd_vel"]

        # ---- Config-driven normalization (in place) ----
        if self._normalization_enabled():
            self._build_normalizers_from_batch(batch)
            self._apply_field_normalization(batch, inverse=False)

        processed_dict = {
            "history": processed_batch_history,
            "history_actions": processed_batch_history_actions,
            "hmap": processed_batch_hmap,
            "future": processed_batch_future,
            "future_actions": processed_batch_future_actions,
            "quat_wrt_world": processed_batch_quat_wrt_world,
            "vel_wrt_world": processed_batch_vel_wrt_world,
            "angvel_wrt_world": processed_batch_angvel_wrt_world,
        }
        if self.model_config.input_cmd_vel:
            processed_dict["history_cmd_vel"] = processed_batch_cmd_vel_history
        if processed_batch_delta_history is not None:
            processed_dict["delta_history"] = processed_batch_delta_history
            processed_dict["delta_future"]  = processed_batch_delta_future

        return processed_dict


    def post_process_data_learning(
        self,
        model_output: torch.Tensor,
        current_vel_world: torch.Tensor,
        current_angvel_world: torch.Tensor,
    ):
        """Post-process model outputs for learning and visualization.

        Returns a dictionary with:
          - 'delta' (delta mode only): normalized delta prediction.
          - 'absolute_for_loss': absolute prediction in *normalized* space (used for gradients).
          - 'absolute_phys': absolute prediction in *physical* space (for visualization/metrics only).
        """
        model_output_dict: Dict[str, torch.Tensor] = {}

        if self.model_config.prediction_type == "delta":
            # Normalized delta output (direct model prediction)
            delta_out = model_output
            model_output_dict["delta"] = delta_out

            # Convert normalized deltas -> physical deltas for integration
            delta_for_abs = delta_out
            if self._normalization_enabled():
                self._build_normalizers_from_batch({"delta_future": delta_out})
                delta_norm = (
                    self._field_normalizers["delta_future"]
                    if "delta_future" in self._field_normalizers
                    else None
                )
                if delta_norm is not None:
                    delta_for_abs = delta_out.clone()
                    # For fnorm, inverse=True still enforces unit norm; for other
                    # strategies this un-normalizes back to physical space.
                    delta_norm.apply(delta_for_abs, inverse=True)

            # Integrate physical deltas to physical absolute trajectory
            absolute_phys = reconstruct_absolute_from_deltas(delta_for_abs)

            # Re-normalize absolute trajectory for loss computation
            absolute_for_loss = absolute_phys
            if self._normalization_enabled():
                self._build_normalizers_from_batch({"future": absolute_for_loss})
                future_norm = (
                    self._field_normalizers["future"]
                    if "future" in self._field_normalizers
                    else None
                )
                if future_norm is not None:
                    absolute_for_loss = absolute_for_loss.clone()
                    future_norm.apply(absolute_for_loss, inverse=False)

            # Always enforce unit quaternions for loss (out-of-place to avoid
            # in-place grad/version issues).
            if absolute_for_loss.shape[-1] >= 7:
                quat_norm = F.normalize(
                    absolute_for_loss[..., 3:7], p=2, dim=-1, eps=1e-8
                )
                absolute_for_loss = torch.cat(
                    [absolute_for_loss[..., :3], quat_norm, absolute_for_loss[..., 7:]],
                    dim=-1,
                )

            model_output_dict["absolute_for_loss"] = absolute_for_loss
            model_output_dict["absolute_phys"] = absolute_phys

        else:
            # Absolute model: model_output is already in normalized space
            absolute_for_loss = model_output

            # Always enforce unit quaternions for loss (out-of-place).
            if absolute_for_loss.shape[-1] >= 7:
                quat_norm = F.normalize(
                    absolute_for_loss[..., 3:7], p=2, dim=-1, eps=1e-8
                )
                absolute_for_loss = torch.cat(
                    [absolute_for_loss[..., :3], quat_norm, absolute_for_loss[..., 7:]],
                    dim=-1,
                )

            model_output_dict["absolute_for_loss"] = absolute_for_loss

            # For visualization, also expose a physical (unnormalized) trajectory
            if self._normalization_enabled():
                absolute_phys = self.post_process_data(absolute_for_loss.clone())
            else:
                absolute_phys = absolute_for_loss.clone()
            model_output_dict["absolute_phys"] = absolute_phys

        return model_output_dict

    def post_process_data(self, model_output: torch.Tensor):
        # model output -> (batch_size, seq_len, 7) pos, quat

        if self._normalization_enabled():
            self._build_normalizers_from_batch({"future": model_output})
            future_norm = self._field_normalizers["future"] if "future" in self._field_normalizers else None
            if future_norm is not None:
                future_norm.apply(model_output, inverse=True)

        # Quaternions in physical space should always be unit-norm (out-of-place).
        if model_output.shape[-1] >= 7:
            quat_norm = F.normalize(
                model_output[..., 3:7], p=2, dim=-1, eps=1e-8
            )
            model_output = torch.cat(
                [model_output[..., :3], quat_norm, model_output[..., 7:]],
                dim=-1,
            )

        return model_output

    def forward(self, batch: Dict[str, torch.Tensor]):  # pragma: no cover - abstract hook
        raise NotImplementedError()

    def _has_absolute_loss_weights(self):
        """Check if any absolute (T_*, R_*) loss weights are non-zero."""
        attrs = [
            "T_MSE_factor",
            "T_Aleatoric_factor",
            "T_MAE_factor_x",
            "T_MAE_factor_y",
            "T_MAE_factor_z",
            "T_Vel_MSE_factor",
            "T_Vel_Aleatoric_factor",
            "T_Vel_MAE_factor_x",
            "T_Vel_MAE_factor_y",
            "T_Vel_MAE_factor_z",
            "R_MSE_factor",
            "R_Aleatoric_factor",
            "R_Geodesic_factor",
            "R_MAE_factor_w",
            "R_MAE_factor_x",
            "R_MAE_factor_y",
            "R_MAE_factor_z",
            "R_Vel_MSE_factor",
            "R_Vel_Aleatoric_factor",
            "R_Vel_MAE_factor_x",
            "R_Vel_MAE_factor_y",
            "R_Vel_MAE_factor_z",
        ]
        return any(getattr(self.l_cfg, attr, 0.0) != 0.0 for attr in attrs)

    def _has_delta_loss_weights(self):
        """Check if any delta (T_DELTA_*, R_DELTA_*) loss weights are non-zero."""
        attrs = [
            "T_DELTA_MSE_factor",
            "T_DELTA_Aleatoric_factor",
            "T_DELTA_MAE_factor_x",
            "T_DELTA_MAE_factor_y",
            "T_DELTA_MAE_factor_z",
            "T_DELTA_Vel_MSE_factor",
            "T_DELTA_Vel_Aleatoric_factor",
            "T_DELTA_Vel_MAE_factor_x",
            "T_DELTA_Vel_MAE_factor_y",
            "T_DELTA_Vel_MAE_factor_z",
            "R_DELTA_MSE_factor",
            "R_DELTA_Aleatoric_factor",
            "R_DELTA_Geodesic_factor",
            "R_DELTA_MAE_factor_w",
            "R_DELTA_MAE_factor_x",
            "R_DELTA_MAE_factor_y",
            "R_DELTA_MAE_factor_z",
            "R_DELTA_Vel_MSE_factor",
            "R_DELTA_Vel_Aleatoric_factor",
            "R_DELTA_Vel_MAE_factor_x",
            "R_DELTA_Vel_MAE_factor_y",
            "R_DELTA_Vel_MAE_factor_z",
        ]
        return any(getattr(self.l_cfg, attr, 0.0) != 0.0 for attr in attrs)

    def _apply_loss_weights(self, loss_dict: Dict[str, torch.Tensor], weight_map: Dict[str, Any]) -> torch.Tensor:
        device = loss_dict["mse_loss"].device
        total = torch.tensor(0.0, device=device, dtype=loss_dict["mse_loss"].dtype)

        mse_attr = weight_map.get("mse")
        if mse_attr:
            total = total + loss_dict["mse_loss"] * getattr(self.l_cfg, mse_attr, 0.0)

        ale_attr = weight_map.get("aleatoric")
        if ale_attr:
            total = total + loss_dict["aleatoric_uncertainty_loss"] * getattr(self.l_cfg, ale_attr, 0.0)

        mae_weights = weight_map.get("mae")
        if mae_weights:
            mae_values = loss_dict["mae_individual_loss"]
            for idx, attr in enumerate(mae_weights):
                if attr and idx < mae_values.shape[0]:
                    total = total + mae_values[idx] * getattr(self.l_cfg, attr, 0.0)

        geo_attr = weight_map.get("geodesic")
        if geo_attr and "geodesic_loss" in loss_dict:
            total = total + loss_dict["geodesic_loss"] * getattr(self.l_cfg, geo_attr, 0.0)

        return total

    def _compute_component_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_log_var: Optional[torch.Tensor],
        step_type: str,
        loss_type: str,
        weight_key: str,
        *,
        component_pred: Optional[torch.Tensor] = None,
        component_log_var: Optional[torch.Tensor] = None,
        pred_slice_map: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        component_losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}

        for comp in self._active_components:
            spec = self._component_specs[comp]
            target_slice = spec["slice"]

            pred_slice_indices = None
            if component_pred is not None and pred_slice_map is not None:
                pred_slice_indices = pred_slice_map.get(comp)

            if pred_slice_indices is not None:
                start, end = pred_slice_indices
                pred_slice = component_pred[..., start:end]
                log_var_slice = (
                    component_log_var[..., start:end] if component_log_var is not None else None
                )
            else:
                pred_slice = pred[..., target_slice]
                log_var_slice = (
                    pred_log_var[..., target_slice] if pred_log_var is not None else None
                )

            target_tensor = target[..., target_slice].float()

            loss_dict = self.Loss_function(
                pred_slice,
                target_tensor,
                pred_log_var=log_var_slice,
                is_quaternion=spec["is_quat"],
            )

            prefix = spec["prefix"]
            metrics[f"{step_type}/{loss_type}/{prefix}_mse_loss"] = loss_dict["mse_loss"]

            mae_values = loss_dict["mae_individual_loss"]
            for idx, axis in enumerate(spec["axis_labels"]):
                if idx < mae_values.shape[0]:
                    metrics[f"{step_type}/{loss_type}/{prefix}_{axis}_mae_loss"] = mae_values[idx]

            metrics[f"{step_type}/{loss_type}/{prefix}_aleatoric_loss"] = loss_dict["aleatoric_uncertainty_loss"]

            if spec["is_quat"] and "geodesic_loss" in loss_dict:
                metrics[f"{step_type}/{loss_type}/{prefix}_geodesic_loss"] = loss_dict["geodesic_loss"]

            weight_map = spec["weights"].get(weight_key, {})
            component_losses[prefix] = self._apply_loss_weights(loss_dict, weight_map)

        return component_losses, metrics

    def _compute_cumulative_prediction_error(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if pred is None or target is None:
            return None
        if pred.shape[:-1] != target.shape[:-1]:
            return None
        pose_dim = 7
        dims = min(pred.shape[-1], target.shape[-1], pose_dim)

        if dims < pose_dim:
            return None

        pred = pred[..., :dims].float()
        target = target[..., :dims].float()

        # Two hypotheses: quaternion as-is vs negated
        pred_neg = pred.clone()
        pred_neg[..., 3:7] = -pred_neg[..., 3:7]

        mse_pos = F.mse_loss(pred, target)
        mse_neg = F.mse_loss(pred_neg, target)
        return torch.minimum(mse_pos, mse_neg)

    def calc_loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        step_type: str = "train",
        loss_type: str = "learn",
        pred_log_var: Optional[torch.Tensor] = None,
        pred_residual: Optional[torch.Tensor] = None,
        component_pred: Optional[torch.Tensor] = None,
        component_log_var: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Calculate loss for absolute predictions."""

        component_losses, metrics = self._compute_component_losses(
            pred,
            y,
            pred_log_var,
            step_type,
            loss_type,
            weight_key="abs",
            component_pred=component_pred,
            component_log_var=component_log_var,
            pred_slice_map=self._pred_component_slices,
        )

        return component_losses, metrics

    def calc_loss_delta(
        self,
        pred_delta: torch.Tensor,
        y_delta: torch.Tensor,
        step_type: str = "train",
        loss_type: str = "learn_delta",
        pred_log_var: Optional[torch.Tensor] = None,
        component_pred: Optional[torch.Tensor] = None,
        component_log_var: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Calculate loss for delta predictions."""

        component_losses, metrics = self._compute_component_losses(
            pred_delta,
            y_delta,
            pred_log_var,
            step_type,
            loss_type,
            weight_key="delta",
            component_pred=component_pred,
            component_log_var=component_log_var,
            pred_slice_map=self._pred_component_slices,
        )

        return component_losses, metrics

    def _step(self, batch_dict, step_type="train"):
        """Streamlined step function supporting both absolute and delta losses."""

        # Preserve original (unnormalized) ground-truth future for visualization
        original_future_phys: Optional[torch.Tensor] = batch_dict.get("future")
        if original_future_phys is not None:
            original_future_phys = original_future_phys.clone().detach()

        # Forward pass (normalizes batch_dict in-place)
        processed_batch_dict = self.pre_process_data(batch_dict)
        # Targets coming out of pre_process_data are in *normalized* space
        y_absolute = processed_batch_dict.get("future")
        if y_absolute is not None:
            y_absolute = y_absolute.float()
        y_delta = processed_batch_dict.get("delta_future")
        if y_delta is not None:
            y_delta = y_delta.float()
        output_dict = self(processed_batch_dict)
        
        model_future_output = output_dict["future_traj"]
        pred_traj_log_var = output_dict["future_traj_log_var"]
        pred_residual = output_dict.get("future_traj_residual", None)
        component_pred = output_dict.get("component_predictions")
        component_log_var = output_dict.get("component_log_var")

        # Determine which losses to compute based on config and step type
        is_training = (step_type == "train")
        is_delta_mode = (self.model_config.prediction_type == "delta")
        has_abs_weights = self._has_absolute_loss_weights()
        has_delta_weights = self._has_delta_loss_weights()

        need_absolute = False
        if y_absolute is not None:
            if not is_delta_mode:
                need_absolute = True
            elif not is_training:
                need_absolute = True
            elif has_abs_weights:
                need_absolute = True

        # Normalized
        pred_absolute: Optional[torch.Tensor] = None          # normalized (for loss)
        pred_delta: Optional[torch.Tensor] = None             # normalized deltas
        # Un-Normalized
        pred_absolute_phys: Optional[torch.Tensor] = None     # physical (for visualization)

        if is_delta_mode:
            pred_delta = model_future_output

        if need_absolute:
            processed_dict = self.post_process_data_learning(
                model_future_output,
                processed_batch_dict["vel_wrt_world"],
                processed_batch_dict["angvel_wrt_world"],
            )
            pred_absolute = processed_dict.get("absolute_for_loss")
            pred_absolute_phys = processed_dict.get("absolute_phys")
            if is_delta_mode:
                pred_delta = processed_dict.get("delta", model_future_output)
        elif not is_delta_mode:
            pred_absolute = model_future_output

        # Initialize loss components
        total_loss = torch.tensor(0.0, device=model_future_output.device, dtype=model_future_output.dtype)
        metrics = {}

        def _sum_losses(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            if not loss_dict:
                return model_future_output.new_zeros(())
            return torch.stack(list(loss_dict.values())).sum()

        # Compute absolute loss (for learning)
        compute_abs_loss = (
            need_absolute
            and (pred_absolute is not None)
            and (y_absolute is not None)
            and ((is_training and has_abs_weights) or (not is_training))
        )


        if compute_abs_loss and pred_absolute is not None and y_absolute is not None:
            abs_component_pred = component_pred if (component_pred is not None and not is_delta_mode) else None
            abs_component_log_var = (
                component_log_var if abs_component_pred is not None and component_log_var is not None else None
            )
            component_losses, component_metrics = self.calc_loss(
                pred_absolute,
                y_absolute,
                step_type,
                "learn",
                pred_traj_log_var,
                pred_residual,
                component_pred=abs_component_pred,
                component_log_var=abs_component_log_var,
            )

            component_total = _sum_losses(component_losses)

            if is_training:
                total_loss = total_loss + component_total

            metrics.update(component_metrics)
            for prefix, value in component_losses.items():
                label = self._component_total_labels.get(prefix, f"{prefix}_loss")
                metrics[f"{step_type}/learn/{label}"] = value

        # Variable-horizon masking: zero out predictions and targets beyond the
        # sampled horizon so those steps contribute zero to every loss term.
        future_horizon_mask = batch_dict.get("future_horizon_mask")  # (B, T) bool after collation
        if future_horizon_mask is not None and is_training:
            vmask = future_horizon_mask.unsqueeze(-1).to(model_future_output.dtype)  # (B, T, 1)
            if pred_delta is not None:
                pred_delta = pred_delta * vmask
            if y_delta is not None:
                y_delta = y_delta * vmask
            if pred_absolute is not None:
                pred_absolute = pred_absolute * vmask
            if y_absolute is not None:
                y_absolute = y_absolute * vmask

        # Compute delta loss (for learning)
        compute_delta_loss = has_delta_weights and pred_delta is not None and y_delta is not None
        if compute_delta_loss and pred_delta is not None and y_delta is not None:
            delta_component_losses, delta_component_metrics = self.calc_loss_delta(
                pred_delta,
                y_delta,
                step_type,
                "learn_delta",
                pred_traj_log_var,
                component_pred=component_pred,
                component_log_var=component_log_var,
            )

            delta_total = _sum_losses(delta_component_losses)

            if is_training and is_delta_mode:
                total_loss = total_loss + delta_total

            metrics.update(delta_component_metrics)
            for prefix, value in delta_component_losses.items():
                label = self._component_total_labels.get(prefix, f"{prefix}_loss")
                metrics[f"{step_type}/learn_delta/{label}"] = value

        # Compute visualization metrics (post-processed, in physical space only)
        processed_output: Optional[torch.Tensor] = None
        batch_future_copy: Optional[torch.Tensor] = None
        if (pred_absolute is not None or pred_absolute_phys is not None) and original_future_phys is not None:
            # Predicted trajectory in physical coordinates
            if pred_absolute_phys is not None:
                processed_output = pred_absolute_phys.clone().detach()
            else:
                processed_output = self.post_process_data(pred_absolute.clone().detach())

            # Ground-truth future in physical coordinates (original, unnormalized batch values)
            batch_future_copy = original_future_phys

            vis_component_losses, vis_metrics = self.calc_loss(
                processed_output, batch_future_copy, step_type, "vis", pred_traj_log_var
            )

            metrics.update(vis_metrics)
            for prefix, value in vis_component_losses.items():
                label = self._component_total_labels.get(prefix, f"{prefix}_loss")
                metrics[f"{step_type}/vis/{label}"] = value

            if step_type != "train":
                cumulative_error = self._compute_cumulative_prediction_error(
                    processed_output, batch_future_copy
                )
                if cumulative_error is not None:
                    metrics[f"{step_type}/overall/cumulative_prediction_error"] = cumulative_error
        
        metrics[f"{step_type}/learn/loss"] = total_loss
        metrics[f"{step_type}/learn/loss_epoch"] = total_loss
        
        return total_loss, metrics, processed_output, batch_future_copy, pred_traj_log_var

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        # with TimerBlock("Shifting devices in train step"):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        loss, metrics, pred, y, pred_log_var = self._step(batch, step_type="train")

        self._last_train_metrics = metrics
        self.train_metrics= metrics
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # safe to log here (outside torch.compile graph)
        metrics = getattr(self, "_last_train_metrics", None)
        if metrics is not None:
            self.log_dict(metrics, prog_bar=False, sync_dist=False, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self._last_train_metrics = None

    @torch.inference_mode()
    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation without autograd to avoid graph retention/OOM."""

        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        step_type = "val"
        if dataloader_idx == 1:
            step_type = "test"
        
        with torch.no_grad():
            loss, metrics, pred, y, pred_log_var = self._step(batch, step_type=step_type)

            # CTX diagnostics — only present on CTX models (get_decoder_memory sets _ctx_vec)
            if hasattr(self, "_ctx_vec") and self._ctx_vec is not None:
                # Variance across batch: near-zero = CTX collapsed, growing = CTX encoding info
                ctx_var = self._ctx_vec.std(dim=0).mean()
                metrics[f"{step_type}/ctx_token_variance"] = ctx_var

                # FiLM gamma std across batch per layer: near-zero = FiLM ignoring CTX
                if hasattr(self, "film_layers"):
                    gamma_stds = []
                    for film in self.film_layers:
                        gb = film.proj(self._ctx_vec)
                        gamma, _ = gb.chunk(2, dim=-1)
                        gamma_stds.append(gamma.std(dim=0).mean())
                    metrics[f"{step_type}/film_gamma_std"] = torch.stack(gamma_stds).mean()

        # Keep GPU payload only until on_validation_batch_end runs; metrics stay on CPU
        # self._last_val_payload = (metrics, pred, y, pred_log_var, batch, batch_idx)
        self._last_val_payload = (metrics, pred.detach(), y.detach(), 
                                pred_log_var.detach() if pred_log_var is not None else None, 
                                batch, batch_idx)
        return loss.detach()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):

        step_type = "val"
        if dataloader_idx == 1:
            step_type = "test"

        # Nothing to do if payload wasn't set
        if not hasattr(self, "_last_val_payload") or self._last_val_payload is None:
            return

        try:
            metrics, pred, y, pred_log_var, saved_batch, saved_idx = self._last_val_payload

            # # ---------------------------
            # # 1) Log metrics (avoid fancy structures)
            # # ---------------------------
            # # Ensure tensors are detached & on CPU-friendly dtype for Lightning’s logger
            # clean_metrics = {}
            # for k, v in metrics.items():
            #     if isinstance(v, torch.Tensor):
            #         # Keep device-agnostic, cast to float32 for safety
            #         clean_metrics[k] = v.detach().to(torch.float32)
            #     else:
            #         clean_metrics[k] = v
            metrics = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in metrics.items()
            }
            self.log_dict(
                metrics,
                prog_bar=False,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

            # ---------------------------
            # 2) Optional plotting on rank-0
            # ---------------------------

            if step_type == "val":
                plot_epoch_freq = self.plot_val_epoch_freq
                plot_batch_freq = self.plot_val_batch_freq
            elif step_type == "test":
                plot_epoch_freq = self.plot_test_epoch_freq
                plot_batch_freq = self.plot_test_batch_freq

            if (
                getattr(self, "global_rank", 0) == 0
                and (self.current_epoch % plot_epoch_freq == 0)
                and (saved_idx % plot_batch_freq == 0) and (self.no_plot == False)
            ):
                self._launch_async_visualization(
                    step_type,
                    pred,
                    y,
                    pred_log_var,
                    saved_batch,
                    saved_idx,
                )

                # pass

        finally:
            # Always clear payload so we don't reuse stale data
            self._last_val_payload = None
            # Force GPU memory cleanup to clear any PEFT/LoRA internal caches
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()


    def configure_optimizers(self):
        lr =self.l_cfg.lr
        weight_decay = self.l_cfg.adamW_weight_decay

        optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        use_cosine_annealing = self.l_cfg.use_cosine_annealing
        use_lr_warmup = self.l_cfg.use_lr_warmup

        T_max = int(self.dataset_size / self.l_cfg.batch_size) * self.l_cfg.epochs

        if not (use_cosine_annealing or use_lr_warmup):
            return optimizer

        cosine_scheduler = None
        warmup_scheduler = None
        if use_cosine_annealing:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, T_max),
            )
        
        if use_lr_warmup:
            if self.l_cfg.lr_warmup_factor > 0.0:
                warmup_steps = T_max * self.l_cfg.lr_warmup_factor
                def _warmup_lambda(step: int) -> float:
                    if warmup_steps <= 0:
                        return 1.0
                    return min(step / max(1, warmup_steps), 1.0)
                
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer,
                        lr_lambda=_warmup_lambda,
                    )
            else:
                use_lr_warmup = False  # disable if factor <= 0.0
        
        scheduler = None
        if use_cosine_annealing and use_lr_warmup:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                )
        elif use_cosine_annealing:
            scheduler = cosine_scheduler
        elif use_lr_warmup:
            scheduler = warmup_scheduler
        
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

    def _launch_async_visualization(
        self,
        phase: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_log_var: Optional[torch.Tensor],
        saved_batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        worker = threading.Thread(
            target=self._async_visualization_worker,
            args=(phase, pred, target, pred_log_var, saved_batch, batch_idx),
            daemon=True,
        )
        worker.start()

    def _async_visualization_worker(
        self,
        phase: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_log_var: Optional[torch.Tensor],
        saved_batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        try:
            with torch.no_grad():
                pred32 = pred.detach().to(torch.float32)
                target32 = target.detach().to(torch.float32)

                pos_pred = pred32[0, :, :3].cpu().numpy()
                pos_gt = target32[0, :, :3].cpu().numpy()

                quat_pred = F.normalize(pred32[0, :, 3:7], p=2, dim=-1, eps=1e-8).cpu().numpy()
                quat_gt = target32[0, :, 3:7].cpu().numpy()

                pos_log_var_np: Optional[np.ndarray] = None
                quat_log_var_np: Optional[np.ndarray] = None
                if self.aleatoric_uncertainty and (pred_log_var is not None):
                    plv32 = pred_log_var.detach().to(torch.float32)
                    pos_log_var_np = plv32[0, :, :3].cpu().numpy()
                    quat_log_var_np = plv32[0, :, 3:7].cpu().numpy()

                hmap_np: Optional[np.ndarray] = None
                future_actions_np: Optional[np.ndarray] = None
                if isinstance(saved_batch, dict):
                    if "hmap" in saved_batch and saved_batch["hmap"] is not None:
                        hmap_np = saved_batch["hmap"][0].detach().to(torch.float32).cpu().numpy()
                    if "future_actions" in saved_batch and saved_batch["future_actions"] is not None:
                        future_actions_np = saved_batch["future_actions"][0].detach().to(torch.float32).cpu().numpy()

                self.log_pos_and_quat_to_tensorboard(
                    phase,
                    pos_pred,
                    pos_gt,
                    quat_pred,
                    quat_gt,
                    hmap_np,
                    batch_idx,
                    future_actions_np,
                    pos_log_var_np,
                    quat_log_var_np,
                )
        except Exception:
            logging.getLogger(__name__).exception("Async visualization logging failed.")

    def log_pos_and_quat_to_tensorboard(
        self, phase: str, pred_pos: np.ndarray, target_pos: np.ndarray,
        pred_quat: np.ndarray, target_quat: np.ndarray, hmap, batch_idx: int,
        future_actions, pos_log_var, quat_log_var
    ):
        """
        Logs position (x, y, z) and quaternion (w, x, y, z) trajectories along with a BEV trajectory plot to TensorBoard.
        """
        x_axis = np.arange(len(pred_pos))  # Time steps or indices

        # Create the figure
        fig = plt.figure(figsize=(24, 20))

        # Create a GridSpec layout
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 3])  # Adjust height ratios

        if self.aleatoric_uncertainty:
            pred_pos_var = np.exp(pos_log_var)
            pred_quat_var = np.exp(quat_log_var)
            pred_pos_sigma = np.sqrt(pred_pos_var)
            pred_quat_sigma = np.sqrt(pred_quat_var)

        # Position plots (x, y, z) in the left column
        for i, coord in enumerate(["X", "Y", "Z"]):
            ax_pos = fig.add_subplot(gs[i, 0])  # Use gridspec for precise positioning

            position_errors = np.abs(pred_pos[:, i] - target_pos[:, i])

            error_colors = cmap_bg(norm_pos_cmap(position_errors))[:, :3]

            ax_pos.plot(x_axis, pred_pos[:, i], label=f"Predicted {coord}", color="green")
            ax_pos.plot(x_axis, target_pos[:, i], label=f"Target {coord}", color="black")

            if self.aleatoric_uncertainty:
                ax_pos.fill_between(x_axis, pred_pos[:, i] - pred_pos_sigma[:, i], pred_pos[:, i] + pred_pos_sigma[:, i], color='blue', alpha=0.3, label="Predicted Variance")

            ax_pos.set_title(f"Position {coord} Trajectory")
            # ax_pos.set_xlabel("Time step")
            ax_pos.set_ylabel(f"{coord} value")
            ax_pos.legend()
            finite_target_coord = np.isfinite(target_pos[:, i])
            if np.any(finite_target_coord):
                coord_min = min(-6.0, float(np.min(target_pos[finite_target_coord, i])))
                coord_max = max(6.0, float(np.max(target_pos[finite_target_coord, i])))
                ax_pos.set_ylim(coord_min, coord_max)
            else:
                ax_pos.set_ylim(-6, 6)
            ax_pos.grid(True)
            ax_pos.set_xticklabels([])

            cbar_ax = ax_pos.inset_axes([0, -0.1, 1, 0.05])  # Place directly below x-axis
            cbar_ax.imshow([error_colors], aspect="auto", extent=[x_axis[0], x_axis[-1], 0, 1], cmap=cmap_bg)
            
            # Remove y-axis labels/ticks from color bar subplot
            cbar_ax.set_yticks([])
            cbar_ax.set_xticks(ax_pos.get_xticks())  # Ensure x-axis ticks align with the main plot
            cbar_ax.set_xlim(ax_pos.get_xlim())
            cbar_ax.set_xlabel("Time step")

            # position_errors = np.abs(pred_pos[:, i] - target_pos[:, i])
            
            # error_colors = cmap_bg(norm_pos_cmap(position_errors))[:, :3]  # Extract RGB values
            # ax_pos.imshow([error_colors], extent=[x_axis[0], x_axis[-1], ax_pos.get_ylim()[0], ax_pos.get_ylim()[1]], 
            #                 aspect='auto', alpha=0.45, origin='lower')
            


        cbar_ax = fig.add_axes([0.45, 0.53, 0.01, 0.4])  # Manually placing the colorbar inside the left column
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_pos_cmap, cmap=cmap_bg), cax=cbar_ax, orientation='vertical')
        cbar.set_label("pos error", fontsize=10)
        cbar.set_ticks(np.arange(0, 3, 0.2))  # Set evenly spaced tick marks
        cbar.ax.tick_params(labelsize=10)  # Adjust font size for better readability

        gs_sub = gs[3, 0].subgridspec(1, 2, wspace=0.4)  # Divide gs[3, 0] into 1 row and 2 columns
        # Add throttle plot (future_actions[:, 0]) in the left column
        ax_throttle = fig.add_subplot(gs_sub[0, 0])  # Left column
        ax_throttle.plot(x_axis, future_actions[:, 0], label="Throttle", color="green")
        ax_throttle.set_title("Throttle Actions")
        ax_throttle.set_xlabel("Time step")
        ax_throttle.set_ylabel("Throttle")
        ax_throttle.grid(True)
        ax_throttle.legend()

        # Add steer plot (future_actions[:, 1]) in the right column
        ax_steer = fig.add_subplot(gs_sub[0, 1])  # Right column
        ax_steer.plot(x_axis, future_actions[:, 1], label="Steer", color="purple")
        ax_steer.set_title("Steer Actions")
        ax_steer.set_xlabel("Time step")
        ax_steer.set_ylabel("Steer")
        ax_steer.grid(True)
        ax_steer.legend()

        # Quaternion plots (w, x, y, z) in the right column
        for i, coord in enumerate(["W", "X", "Y", "Z"]):
            ax_quat = fig.add_subplot(gs[i, 1])

            # Original and negated quaternions
            pred_q = pred_quat[:, i]
            pred_q_alt = -pred_quat[:, i]
            target_q = target_quat[:, i]
            target_q_alt = -target_quat[:, i]

            # Error colors for original (not altered)
            quat_errors = np.abs(pred_q - target_q)
            error_colors = cmap_bg(norm_quat_cmap(quat_errors))[:, :3]

            # Plot all 4 lines
            ax_quat.plot(x_axis, pred_q, label=f"Predicted {coord}", color="green", linestyle="-")
            ax_quat.plot(x_axis, pred_q_alt, label=f"-Predicted {coord}", color="green", linestyle="--")

            ax_quat.plot(x_axis, target_q, label=f"Target {coord}", color="black", linestyle="-")
            ax_quat.plot(x_axis, target_q_alt, label=f"-Target {coord}", color="black", linestyle="--")

            if self.aleatoric_uncertainty:
                ax_quat.fill_between(
                    x_axis,
                    pred_q - pred_quat_sigma[:, i],
                    pred_q + pred_quat_sigma[:, i],
                    color='blue', alpha=0.3, label="Predicted Variance"
                )

            ax_quat.set_title(f"Quaternion {coord} Trajectory")
            ax_quat.set_ylabel(f"{coord} value")
            ax_quat.set_ylim(-1, 1)
            ax_quat.legend()
            ax_quat.grid(True)
            ax_quat.set_xticklabels([])

            cbar_ax = ax_quat.inset_axes([0, -0.1, 1, 0.05])
            cbar_ax.imshow([error_colors], aspect="auto", extent=[x_axis[0], x_axis[-1], 0, 1], cmap=cmap_bg)
            cbar_ax.set_yticks([])
            cbar_ax.set_xticks(ax_quat.get_xticks())
            cbar_ax.set_xlim(ax_quat.get_xlim())
            cbar_ax.set_xlabel("Time step")


            # quat_errors = np.abs(pred_quat[:, i] - target_quat[:, i])
            
            # error_colors = cmap_bg(norm_quat_cmap(quat_errors))[:, :3]  # Extract RGB values
            # ax_quat.imshow([error_colors], extent=[x_axis[0], x_axis[-1], ax_quat.get_ylim()[0], ax_quat.get_ylim()[1]], 
            #                 aspect='auto', alpha=0.45, origin='lower')

        cbar_ax = fig.add_axes([0.52, 0.53, 0.01, 0.4])  # Manually placing the colorbar inside the left column
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_quat_cmap, cmap=cmap_bg), cax=cbar_ax, orientation='vertical')
        cbar.set_label("quat error", fontsize=10)
        cbar.set_ticks(np.arange(0, 2, 0.2))  # Set evenly spaced tick marks
        cbar.ax.tick_params(labelsize=10)  # Adjust font size for better readability
        # BEV trajectory plot configuration

        if hmap is not None:
            x = np.arange(-hmap.shape[0] // 2, hmap.shape[0] // 2) * self.model_config.local_hmap_bf_res
            y = np.arange(-hmap.shape[1] // 2, hmap.shape[1] // 2) * self.model_config.local_hmap_bf_res

        x_traj_tgt = target_pos[:, 0]
        y_traj_tgt = target_pos[:, 1]
        x_traj_pred = pred_pos[:, 0]
        y_traj_pred = pred_pos[:, 1]

        ax_bev = fig.add_subplot(gs[4, :])
        if hmap is not None:
            ax_bev.imshow(
                hmap.T,
                extent=[x[0], x[-1], y[0], y[-1]],
                origin="lower",
                cmap="terrain",
                alpha=0.35,
                aspect="equal",
            )

        ax_bev.plot(x_traj_tgt, y_traj_tgt, label="Target Trajectory", color="red", linewidth=2.5)
        ax_bev.plot(x_traj_pred, y_traj_pred, label="Predicted Trajectory", color="blue", linewidth=2.0)
        ax_bev.scatter(x_traj_tgt[0], y_traj_tgt[0], label="Start", color="black", s=40, zorder=4)
        ax_bev.scatter(x_traj_tgt[-1], y_traj_tgt[-1], label="Target End", color="red", s=50, zorder=4)

        finite_target = np.isfinite(x_traj_tgt) & np.isfinite(y_traj_tgt)
        if np.any(finite_target):
            x_min = float(np.min(x_traj_tgt[finite_target]))
            x_max = float(np.max(x_traj_tgt[finite_target]))
            y_min = float(np.min(y_traj_tgt[finite_target]))
            y_max = float(np.max(y_traj_tgt[finite_target]))
            x_center = 0.5 * (x_min + x_max)
            y_center = 0.5 * (y_min + y_max)
            side = max(x_max - x_min, y_max - y_min, 1.0)
            half_side = 0.55 * side
            ax_bev.set_xlim(x_center - half_side, x_center + half_side)
            ax_bev.set_ylim(y_center - half_side, y_center + half_side)

        ax_bev.set_title("BEV Trajectory")
        ax_bev.set_xlabel("X (meters)")
        ax_bev.set_ylabel("Y (meters)")
        ax_bev.set_aspect("equal", adjustable="box")
        ax_bev.grid(True)
        ax_bev.legend(loc="best")

        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.6, wspace=0.4)  # Add space between plots

        self._fanout_add_figure(f"visualize/{phase}/pos_and_quat_traj_batch_{batch_idx}", fig, self.current_epoch)

        plt.close(fig)


    def _fanout_add_figure(self, tag: str, fig, step: int):
        # iterate over underlying loggers
        for lg in self.trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_figure(tag, fig, global_step=step)
            elif isinstance(lg, WandbLogger):
                lg.experiment.log({tag: wandb.Image(fig)}, step=step)

    def _fanout_add_text(self, tag: str, text: str, step: int):
        for lg in self.trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_text(tag, text, global_step=step)
            elif isinstance(lg, WandbLogger):
                # simplest: log as a text field
                lg.experiment.log({tag: text}, step=step)
