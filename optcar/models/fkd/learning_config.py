"""Supervised learning configuration for FKD."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SupervisedLearningConfig:
    """Configuration for supervised learning."""

    # Reproducibility
    seed: int = 42  # Random seed for reproducibility
    deterministic: bool = False  # If True: 100% reproducible - will be slower
    
    batch_size: int = 10
    lr: float = 1e-4
    adamW_weight_decay: float = 1e-4
    epochs: int = 100

    # Training utilities
    use_cosine_annealing: bool = False
    use_lr_warmup: bool = False
    use_gradient_clip: bool = False
    
    gradient_clip_val: float = 1.0
    lr_warmup_factor: float = 0.0  # Fraction of cosine_T_max used for linear warmup (0.0 -> disabled)
    
    # DEPRECATED FIELDS ---------------
    cosine_eta_min: float = 0.0 # DEPRECATED
    cosine_warmup_steps: int = 0 # DEPRECATED
    cosine_warmup_start_factor: float = 1e-3 # DEPRECATED
    # DEPRECATED FIELDS ---------------

    log_time_with_metrics: bool = True
    save_checkpoint_every_n_epoch: int = 2
    run_initial_validation: bool = True
    checkpoint_folder: Optional[str] = None

    discount_gamma: float = 0.998

    # Absolute position (translation)
    T_MSE_factor: float = 0.0
    T_MAE_factor_x: float = 0.0
    T_MAE_factor_y: float = 0.0
    T_MAE_factor_z: float = 0.0
    T_Aleatoric_factor: float = 0.0

    # Absolute velocity
    T_Vel_MSE_factor: float = 0.0
    T_Vel_MAE_factor_x: float = 0.0
    T_Vel_MAE_factor_y: float = 0.0
    T_Vel_MAE_factor_z: float = 0.0
    T_Vel_Aleatoric_factor: float = 0.0

    # Absolute rotation (quaternion)
    R_MSE_factor: float = 0.0
    R_MAE_factor_w: float = 0.0
    R_MAE_factor_x: float = 0.0
    R_MAE_factor_y: float = 0.0
    R_MAE_factor_z: float = 0.0
    R_Aleatoric_factor: float = 0.0
    R_Geodesic_factor: float = 0.0

    # Absolute angular velocity
    R_Vel_MSE_factor: float = 0.0
    R_Vel_MAE_factor_x: float = 0.0
    R_Vel_MAE_factor_y: float = 0.0
    R_Vel_MAE_factor_z: float = 0.0
    R_Vel_Aleatoric_factor: float = 0.0

    # Delta position (translation)
    T_DELTA_MSE_factor: float = 0.0
    T_DELTA_MAE_factor_x: float = 0.0
    T_DELTA_MAE_factor_y: float = 0.0
    T_DELTA_MAE_factor_z: float = 0.0
    T_DELTA_Aleatoric_factor: float = 0.0

    # Delta velocity
    T_DELTA_Vel_MSE_factor: float = 0.0
    T_DELTA_Vel_MAE_factor_x: float = 0.0
    T_DELTA_Vel_MAE_factor_y: float = 0.0
    T_DELTA_Vel_MAE_factor_z: float = 0.0
    T_DELTA_Vel_Aleatoric_factor: float = 0.0

    # Delta rotation (quaternion)
    R_DELTA_MSE_factor: float = 0.0
    R_DELTA_MAE_factor_w: float = 0.0
    R_DELTA_MAE_factor_x: float = 0.0
    R_DELTA_MAE_factor_y: float = 0.0
    R_DELTA_MAE_factor_z: float = 0.0
    R_DELTA_Aleatoric_factor: float = 0.0
    R_DELTA_Geodesic_factor: float = 0.0

    # Delta angular velocity
    R_DELTA_Vel_MSE_factor: float = 0.0
    R_DELTA_Vel_MAE_factor_x: float = 0.0
    R_DELTA_Vel_MAE_factor_y: float = 0.0
    R_DELTA_Vel_MAE_factor_z: float = 0.0
    R_DELTA_Vel_Aleatoric_factor: float = 0.0



    training_dtype: type = torch.float32

    train_only_residual: bool = False
    L2_normalize_quat_before_loss: bool = False

    plot_validation_results: bool = True
    plot_idx_range: list = field(default_factory=lambda: [0, 1])

    early_stopping: bool = False
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    early_stop_monitor: str = "val/learn/loss_epoch"
    early_stop_mode: str = "min"

    add_noise_in_training: bool = False
    pos_sigma: float = 0.04
    vel_sigma: float = 0.04
    angvel_sigma: float = 0.02
    quat_sigma: float = 0.02
    act_vel_sigma: float = 0.1
    act_steer_sigma: float = 0.03
