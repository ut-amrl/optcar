"""FKD model configurations."""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
    "64": torch.float64,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "16": torch.float16,
    "16-mixed": torch.float16,
    "16-true": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "bf16-true": torch.bfloat16,
}


def resolve_dtype(dtype_spec: Union[str, torch.dtype, None]) -> torch.dtype:
    """Return a torch.dtype given a user-provided spec (string or torch.dtype)."""
    if dtype_spec is None:
        return torch.float32
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    if isinstance(dtype_spec, str):
        key = dtype_spec.strip().lower()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
    raise ValueError(f"Unsupported dtype spec '{dtype_spec}'. Expected one of: {sorted(_DTYPE_ALIASES)}")


def dtype_to_trainer_precision(dtype_spec: Union[str, torch.dtype, None]) -> Union[str, int]:
    """Map a dtype spec to the corresponding Lightning precision flag."""
    dtype = resolve_dtype(dtype_spec)
    if dtype == torch.float16:
        return "16-mixed"
    if dtype == torch.bfloat16:
        return "bf16-mixed"
    if dtype == torch.float64:
        return 64
    return 32


@dataclass
class FKD_model_config:
    """Base configuration for FKD models."""

    future_horizon_steps: int = 150
    history_horizon_steps: int = 25
    input_state_dims: int = 13   # Full history/delta: pos(3) + quat(4) + vel(3) + angvel(3)
    output_state_dims: int = 7   # Default output: pos + quat
    action_dims: int = 2  # throttle, steer

    predict_dt: float = 0.01
    local_hmap_bf_res: float = 0.01
    local_hmap_bf_size: int = 10  # in meters

    hmap_img_size: int = 1000  # pixels

    learning_rate: float = 0.001

    embedding_dims: int = 64
    nhead: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    decoder_ff_dim: Optional[int] = None   # if None, falls back to ff_dim
    decoder_nhead: Optional[int] = None   # if None, falls back to nhead

    multilinear_dropout: float = 0.0
    encoder_types: dict = field(default_factory=lambda: {
        "history_states": [64, 128],
        "history_actions": [64, 128],
        "history_cmd_vel": [64, 128],
        "state_wrt_world": [64, 128],
        "map": 128
    })

    decoder_dims: List[int] = field(default_factory=lambda: [256, 256, 64])
    decoder_logvar_dims: List[int] = field(default_factory=lambda: [128, 64])
    decoder_include_twist: bool = False

    no_map: bool = False

    device: int = 0
    dtype: Union[str, torch.dtype] = "bf16-mixed"

    prediction_type: str = 'absolute'  # 'absolute' or 'delta'

    pretrained_model_path_without_residual: str = None
    start_non_residual_training_from_checkpoint: bool = False

    residual: bool = False
    rezero: bool = False

    max_vel: float = 10.0
    max_vel: float = 10.0
    max_steer: float = 1.0

    # Legacy-style trajectory normalization flags (for backward compatibility
    # with older training configs and pre-processing behavior).
    normalize_trajs: bool = False
    normalize_hist_quat: bool = False
    normalize_fut_quat: bool = False

    # Normalization strategies per field/component.
    # Accepted strategies:
    #   - Simple string: "none", "mean_std", "min_max", "hard_min_max",
    #     "hard_mean_std", "fnorm", or alias to another component key.
    #     Data-driven "mean_std" and "min_max" are populated from the train
    #     loader before training starts.
    #   - Or a dict/tuple specifying strategy + bounds, e.g.:
    #         {"strategy": "hard_min_max", "min": -5, "max": 5}
    #         {"strategy": "hard_mean_std", "mean": 0.0, "std": 0.5}
    #         ("hard_min_max", -5, 5)
    normalization: dict = field(default_factory=lambda: {
        "history": {
            "pos_x": "none",
            "pos_y": "none",
            "pos_z": "none",
            "quat_w": "none",
            "quat_x": "none",
            "quat_y": "none",
            "quat_z": "none",
            "vel_x": "none",
            "vel_y": "none",
            "vel_z": "none",
            "ang_x": "none",
            "ang_y": "none",
            "ang_z": "none",
        },
        "future": {
            "pos_x": "none",
            "pos_y": "none",
            "pos_z": "none",
            "quat_w": "none",
            "quat_x": "none",
            "quat_y": "none",
            "quat_z": "none",
        },
        "delta_history": {
            "pos_x": "none",
            "pos_y": "none",
            "pos_z": "none",
            "quat_w": "none",
            "quat_x": "none",
            "quat_y": "none",
            "quat_z": "none",
            "vel_x": "none",
            "vel_y": "none",
            "vel_z": "none",
            "ang_x": "none",
            "ang_y": "none",
            "ang_z": "none",
        },
        "delta_future": {
            "pos_x": "none",
            "pos_y": "none",
            "pos_z": "none",
            "quat_w": "none",
            "quat_x": "none",
            "quat_y": "none",
            "quat_z": "none",
        },
        "history_actions": {
            "throttle": "none",
            "steer": "none",
        },
        "future_actions": {
            "throttle": "none",
            "steer": "none",
        },
    })

    # Optional fixed bounds for "hard_min_max" / fixed (mean, std) for "hard_mean_std":
    # map field -> component/group -> (min, max) or (mean, std).
    # Normally users specify these via the combined `normalization` config
    # (see __post_init__), but this dict is still supported for backward
    # compatibility.
    normalization_bounds: dict = field(default_factory=dict)

    aleatoric_uncertainty: bool = False

    input_actions: bool = True
    input_cmd_vel: bool = False
    model_type: str = None

    # Random history attention masking during training.
    # Each forward pass, a random fraction of history memory tokens are masked from
    # decoder cross-attention (fraction sampled uniformly in [min, max]).
    # Set max_ratio=0.0 to disable (default).
    history_mask_min_ratio: float = 0.0
    history_mask_max_ratio: float = 0.0

    # When True, history positional encoding is reversed so the most recent
    # history token sits at "position 0". Recency becomes a strong inductive prior.
    history_pos_enc_flip: bool = False

    # Mutually exclusive history-encoding modes (both False = default, separate streams).
    #
    # pair_state_actions: encode each stream independently, then interleave into
    #   [s0, d0, a0, s1, d1, a1, ...] so per-timestep tokens land at adjacent
    #   positions. Total tokens = k * H.
    #
    # concat_state_actions: concatenate raw per-timestep inputs along the feature
    #   dim first, then encode through a single MLP. Produces one token per
    #   timestep (total = H). Captures nonlinear joint features in the encoder
    #   rather than relying on attention to discover them.
    pair_state_actions: bool = False
    concat_state_actions: bool = False
    # When concat_state_actions=True, hidden dims are taken from
    # encoder_types["concat_history"], e.g. encoder_types={"concat_history": [64, 128], ...}

    def __post_init__(self):
        self.local_hmap_bf_size_px = int(self.local_hmap_bf_size / self.local_hmap_bf_res)
        self._process_normalization_config()

    # ------------------------------------------------------------------
    # Normalization config helpers -------------------------------------
    # ------------------------------------------------------------------
    def _process_normalization_config(self) -> None:
        """Normalize `normalization` into (strategy-only, bounds in normalization_bounds).

        Supports both the legacy two-dict style:
            normalization[field][component] = "hard_min_max"
            normalization_bounds[field][component] = (min, max)

        and a combined style where bounds live alongside the strategy:
            normalization[field][component] = {
                "strategy": "hard_min_max",
                "min": -5,
                "max": 5,
            }
        or:
            normalization[field][component] = {
                "strategy": "hard_mean_std",
                "mean": 0.0,
                "std": 0.5,
            }
        or a tuple:
            normalization[field][component] = ("hard_min_max", -5, 5)

        After this runs:
            - self.normalization contains only strategy strings / aliases.
            - self.normalization_bounds collects any fixed bounds.
        """
        norm_cfg = self.normalization
        bounds_cfg: Dict[str, Dict[str, Any]] = dict(self.normalization_bounds) if self.normalization_bounds else {}

        if not isinstance(norm_cfg, dict):
            return

        for field, comp_map in norm_cfg.items():
            if not isinstance(comp_map, dict):
                continue
            for comp, spec in list(comp_map.items()):
                # Leave plain strings (strategies or aliases) as-is.
                if isinstance(spec, str):
                    continue

                strategy: Union[str, None] = None
                bound_min: Any = None
                bound_max: Any = None

                # Dict style: {"strategy": "hard_min_max", ...}
                if isinstance(spec, dict):
                    strategy = spec.get("strategy")
                    if strategy is None:
                        continue
                    if strategy in ("hard_min_max", "hard_mean_std"):
                        if "bounds" in spec:
                            b = spec["bounds"]
                            if isinstance(b, (list, tuple)) and len(b) >= 2:
                                bound_min, bound_max = b[0], b[1]
                        elif "min" in spec and "max" in spec:
                            bound_min, bound_max = spec["min"], spec["max"]
                        elif "mean" in spec and "std" in spec:
                            bound_min, bound_max = spec["mean"], spec["std"]

                # Tuple/list style: ("hard_min_max", -5, 5)
                elif isinstance(spec, (list, tuple)) and len(spec) >= 1:
                    strategy = spec[0]
                    if len(spec) >= 3 and strategy in ("hard_min_max", "hard_mean_std"):
                        bound_min, bound_max = spec[1], spec[2]

                if strategy is None:
                    continue

                # Update to strategy string
                comp_map[comp] = strategy

                # Attach bounds if provided and not already set
                if bound_min is not None and bound_max is not None and strategy in ("hard_min_max", "hard_mean_std"):
                    field_bounds = bounds_cfg.setdefault(field, {})
                    if comp not in field_bounds:
                        field_bounds[comp] = (bound_min, bound_max)

        self.normalization = norm_cfg
        self.normalization_bounds = bounds_cfg


@dataclass
class TransformerModelConfig(FKD_model_config):
    """Configuration for Transformer-based FKD model."""
    
    model_type: str = "TransformerModel_decoderonly"
    embedding_dims: int = 64
    nhead: int = 4
    num_layers: int = 4
    ff_dim: int = 512

    encoder_types: dict = field(default_factory=lambda: {
        "history_states": [64, 128],
        "history_actions": [64, 128],
        "history_cmd_vel": [64, 128],
        "state_wrt_world": [64, 128],
        "map": 128
    })

    def __post_init__(self):
        super().__post_init__()


@dataclass
class TransformerModelConfig_CTX(TransformerModelConfig):
    """TransformerModelConfig + a bidirectional history self-attention encoder.

    Adds a learnable [CTX] token that attends to all history tokens before the
    decoder cross-attention runs, giving the model an explicit global context
    vector for system identification (e.g. mass / inertia from trajectory history).
    """

    model_type: str = "TransformerModel_decoderonly_CTX"
    # Number of bidirectional self-attention layers applied to [CTX | history tokens].
    ctx_num_layers: int = 2
    # If True, decoder cross-attends only to the CTX token, forcing the history
    # encoder to compress all sys-ID information into that single vector.
    ctx_only: bool = False
    # If True, the CTX token is used for FiLM conditioning on the decoder
    # (gamma * x + beta per layer) instead of being included in cross-attention
    # memory.  With ctx_only=False, history feeds cross-attention and CTX
    # modulates via FiLM.
    ctx_film: bool = False

    def __post_init__(self):
        super().__post_init__()


@dataclass
class TransformerModel_bicycle_residual_Config(TransformerModelConfig):
    """Configuration for Transformer model with bicycle residual."""

    model_type: str = "TransformerModel_decoderonly_bicycle_residual"
    residual_predict_dt: float = 0.02
    residual_kinematic_wheelbase: float = 0.202  # e-revo 1/16 wheelbase dimension

    def __post_init__(self):
        super().__post_init__()
