"""Shared base parameter dataclasses for environment families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseParams:
    """Common starting pose parameters for all families."""

    vehicle_family: str = "ackermann"

    start_pos_x: float = 0.0
    start_pos_y: float = 0.0
    start_pos_z: float = 0.0
    start_rot_roll: float = 0.0
    start_rot_pitch: float = 0.0
    start_rot_yaw: float = 0.0

    # Episode reset behaviour.
    # A *run* spans max_steps and may contain multiple *episodes*.
    # An episode ends when an env is terminated (e.g. boundary collision).
    # auto_reset_episodes: if True, terminated envs are reset and a new
    #   episode begins within the same run.  If False (default), terminated
    #   envs freeze at their last position for the remainder of the run.
    # sync_reset: only used when auto_reset_episodes is True.
    #   If True, all envs wait until every env has terminated, then all
    #   reset together.  If False, each env resets independently as soon
    #   as it terminates.
    auto_reset_episodes: bool = True
    sync_reset: bool = True
