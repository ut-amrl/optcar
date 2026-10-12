"""Parameters for bicycle simulation and starting conditions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


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


@dataclass
class ObstacleConfig:
    """Obstacle spawning configuration.

    Total count (max_obstacles) is split across enabled types: a random
    weight in [0,1] is sampled per enabled type, normalised, then
    multiplied by max_obstacles.
    Override individual fields via Hydra, e.g.:
        +env_setup.params.obstacle_config.square_enabled=false
        +env_setup.params.obstacle_config.max_obstacles=20
    """

    max_obstacles: int = 0
    arrange_grid: bool = False

    # Optional base terrain PNG (white=free, black=obstacle). Loaded and
    # resized to the arena occupancy mask; random obstacles are drawn on top.
    base_map_path: Optional[str] = None
    base_map_threshold: int = 128

    # Square obstacles
    square_enabled: bool = True
    square_min_side: float = 0.4
    square_max_side: float = 0.8

    # Circle obstacles
    circle_enabled: bool = False
    circle_min_radius: float = 0.2
    circle_max_radius: float = 0.8

    # 2D Gaussian obstacles (drawn as filled ellipse at std_cutoff)
    gaussian_enabled: bool = False
    gaussian_x_std_min: float = 0.1
    gaussian_x_std_max: float = 0.5
    gaussian_y_std_min: float = 0.1
    gaussian_y_std_max: float = 0.5
    gaussian_std_cutoff: float = 2.0


@dataclass
class BicycleParams(BaseParams):
    """Physical and control parameters shared by KBM and DBM."""

    # Geometry
    len_f: float = 0.101
    len_r: float = 0.101

    # Control limits
    max_velocity: float = 5.0
    max_steering: float = 0.5
    # Body-frame lateral acceleration limit; None disables it for the dynamic bicycle model.
    max_lateral_acceleration_mps2: Optional[float] = 10.0

    # Dynamic model force coefficients
    C_throttle_a: float = 5.5
    C_rollf: float = 0.0
    C_drag: float = 0.0

    # Dynamic model tire parameters
    C_tire_f: float = 20.0
    C_tire_r: float = 20.0

    # Wheel velocity limits (dynamic model)
    min_wheel_velocity: float = -6.0
    max_wheel_velocity: float = 6.0

    # Inertial parameters (dynamic model)
    mass: float = 1.09
    inertia: float = 0.015

    # Wheel dynamics (dynamic model)
    wheel_time_constant: float = 0.05

    # Simulation step
    dt: float = 0.02

    # Action shaping / deadbands (scalar endpoints so they can be randomized)
    throttle_deadband_min: float = 0.0
    throttle_deadband_max: float = 0.0
    steer_deadband_min: float = 0.0
    steer_deadband_max: float = 0.0
    throttle_multiplier: float = 1.0
    throttle_bias: float = 0.0
    steer_pos_multiplier: float = 1.0
    steer_pos_bias: float = 0.0
    steer_neg_multiplier: float = 1.0
    steer_neg_bias: float = 0.0

    # Bounded arena
    bounded: bool = False
    arena_size: float = 10.0
    car_radius: float = 0.15
    visualizer_resolution_m_per_pixel: float = 0.0125
    local_elevation_map_size_m: float = 10.0
    local_elevation_map_resolution_m_per_pixel: float = 0.05
    elevation_free_value_m: float = 0.0
    elevation_obstacle_value_m: float = 10.0
    # Hand the whole obstacle raster to the controller in sensor_data, for
    # tasks that plan globally instead of only from the local crop.
    include_global_elevation_map: bool = False

    # Obstacles
    obstacle_config: ObstacleConfig = field(default_factory=ObstacleConfig)
