# Adapted from WheeledLab wheeledlab_data/dataloaders/FKD/config.py.
"""Dataloader configuration for FKD."""

import torch
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class DataloaderConfig:
    """Configuration for the FKD dataloader."""
    
    dataset_path: str = None
    test_dataset_path: str = None
    num_runs: int = 1500
    test_num_runs: int = 0
    # Optional per-dataset trajectory caps. When provided, the dataset will
    # use at most this many trajectories per dataset (after num_runs filtering).
    num_trajectories: Optional[Union[int, List[int]]] = None
    test_num_trajectories: Optional[Union[int, List[int]]] = None

    sim_dt: float = 0.005  # dt at which replay was run

    debug: bool = False

    traj_body_frame: bool = True
    hmap_body_frame: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    is_cmd_vel_in_data: bool = False

    batch_size: int = None  # Will be set from learning_config if not specified

    no_plot: bool = False
    plot_val_epoch_freq: int = 15
    plot_val_batch_freq: int = 1
    check_val_every_n_epoch: int = 1
    plot_test_epoch_freq: int = 15
    plot_test_batch_freq: int = 1

    legacy_data: bool = False

    # When True, expect trailing 15 DBM-parameter dims tiled across time in the
    # processed .bin files (feature_dim becomes 41 + 15 = 56). The dataloader
    # returns the constant per-trajectory DBM vector as `dbm_params` (shape
    # (B, 15)) and leaves the leading 41 dims untouched.
    include_dbm_params: bool = False

    # Optional seed controlling which run files are selected when num_runs
    # is smaller than the total number of available files. When None, the
    # first num_runs files (sorted) are used for backward compatibility.
    file_selection_seed: Optional[int] = None

    # Variable-length teacher forcing. When True, each __getitem__ samples a
    # random prediction horizon in [1, future_horizon_steps] and returns a
    # boolean `future_horizon_mask` (shape T) where True = valid step.
    # The loss is zeroed for masked steps, up-weighting short-horizon accuracy.
    variable_horizon: bool = False
