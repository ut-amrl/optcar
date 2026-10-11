"""WheeledLab batched open-loop collection using copied standalone components."""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch

from optcar.dataset import DataConfig, DatasetWriter, session_split
from optcar.models.dynamic_bicycle import DBMParams, DynamicBicycleModel
from optcar.utils.devices import resolve_device
from optcar.utils.file_io import write_json
from optcar.utils.math import yaw_from_quat_torch, quat_from_yaw_torch
from optcar.utils.state_struct import State_struct_torch
from optcar.datagen.wheeledlab.bicycle_params import BicycleParams
from optcar.datagen.wheeledlab.action_bias import ActionBiasSampler
from optcar.datagen.wheeledlab.param_randomizer import ParamRandomizer
from optcar.datagen.wheeledlab.parametric_open_loop import ParametricOpenLoopController
from optcar.datagen.wheeledlab.session import DataCollectionContext, DataCollector


ACTION_SHAPING_KEYS = ("throttle_multiplier", "throttle_bias", "steer_pos_multiplier", "steer_pos_bias", "steer_neg_multiplier", "steer_neg_bias")
ACTION_BIAS_KEYS = ("velocity_bias", "steering_bias")
RANDOMIZATION_KEYS = tuple(key for key in BicycleParams.__dataclass_fields__ if key in
    (set(DBMParams.__dataclass_fields__) - {"max_steering_angle"}) |
    {"max_steering", "max_velocity", "max_lateral_acceleration_mps2", "throttle_deadband_min",
     "throttle_deadband_max", "steer_deadband_min", "steer_deadband_max", "start_pos_x", "start_pos_y", "start_pos_z", "start_rot_yaw", *ACTION_SHAPING_KEYS}) + ACTION_BIAS_KEYS


def validate_parameter_ranges(ranges):
    if not isinstance(ranges, dict):
        raise ValueError("parameter_ranges must be a mapping of parameter names to ranges")
    normalized = dict(ranges)
    if "max_steering_angle" in normalized:
        if "max_steering" in normalized:
            raise ValueError("Specify only one of max_steering and max_steering_angle")
        normalized["max_steering"] = normalized.pop("max_steering_angle")
    unsupported = set(normalized) - set(RANDOMIZATION_KEYS)
    if unsupported:
        raise ValueError("Unsupported parameter ranges: " + ", ".join(sorted(unsupported)))
    for key, spec in normalized.items():
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            spec = [spec, spec]
        if not isinstance(spec, (list, tuple)) or len(spec) not in (2, 3):
            raise ValueError(f"{key}: use [min, max] or [mean, std, normal]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in spec[:2]):
            raise ValueError(f"{key}: range values must be finite numbers")
        kind = str(spec[2]).strip().lower() if len(spec) == 3 else "uniform"
        if kind not in ("uniform", "normal"):
            raise ValueError(f"{key}: distribution must be uniform or normal")
        if kind == "uniform" and spec[0] > spec[1]:
            raise ValueError(f"{key}: maximum must be at least the minimum")
        if key in ACTION_BIAS_KEYS and kind != "uniform":
            raise ValueError(f"{key}: WheeledLab samples per-environment command biases uniformly")
        if kind == "normal" and spec[1] < 0:
            raise ValueError(f"{key}: standard deviation must be nonnegative")
        normalized[key] = [float(spec[0]), float(spec[1]), kind]
    return normalized


def randomization_parameters(params=None):
    base = simulation_params(params, CommandConfig(), .02)
    return [{"key": key, "default": getattr(base, key, 0.0),
             "group": "pose" if key.startswith("start_") else "actuator" if key in (*ACTION_SHAPING_KEYS, *ACTION_BIAS_KEYS) or "deadband" in key else "physical"}
            for key in RANDOMIZATION_KEYS]


def model_params(params):
    values = {name: getattr(params, name) for name in DBMParams.__dataclass_fields__ if name != "max_steering_angle"}
    return DBMParams(**values, max_steering_angle=params.max_steering)


class DynamicBicycleEnv:
    """Headless adapter of BaseBicycleEnv's reset and one-step execution."""
    model_class = DynamicBicycleModel

    def __init__(self, num_envs=1, dt=0.02, params=None, device="cpu", *,
                 env_spacing=2.5, initial_yaw=None, seed=42):
        if num_envs < 1 or dt <= 0:
            raise ValueError("num_envs and dt must be positive")
        self.num_envs, self.dt = num_envs, dt
        params = params if isinstance(params, BicycleParams) else simulation_params(params, CommandConfig(), dt)
        self.model = self.model_class(model_params(params), device,
            max_lateral_acceleration_mps2=params.max_lateral_acceleration_mps2,
            action_shaping={
                "throttle_deadband": (params.throttle_deadband_min, params.throttle_deadband_max),
                "steer_deadband": (params.steer_deadband_min, params.steer_deadband_max),
                **{key: getattr(params, key) for key in ACTION_SHAPING_KEYS},
            })
        self.device = self.model.device
        self.max_velocity, self.max_steering = params.max_velocity, params.max_steering
        self.env_spacing = env_spacing
        self.start_pos_x, self.start_pos_y, self.start_pos_z = params.start_pos_x, params.start_pos_y, params.start_pos_z
        self.start_rot_yaw = params.start_rot_yaw
        self.start_pos_x_range = self.start_pos_y_range = None
        self.initial_yaw_range = initial_yaw
        self.visualizer = None
        self._rng = torch.Generator(device=self.device).manual_seed(seed)
        self.state = torch.zeros((num_envs, 13), device=self.device)
        self.vel_body = torch.zeros((num_envs, 3), device=self.device)
        self.wheel_speed = torch.zeros(num_envs, device=self.device)
        self.min_wheel_velocity, self.max_wheel_velocity = params.min_wheel_velocity, params.max_wheel_velocity
        self.env_terminated = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.auto_reset_episodes, self.sync_reset = params.auto_reset_episodes, params.sync_reset
        self.bounded, self.arena_size, self.car_radius = params.bounded, params.arena_size, params.car_radius
        self.last_reset_ids = None
        self.episode_length = 0
        self.reset_state()

    def reset(self, state=None, wheel_speed=None):
        self.reset_state()
        self.env_terminated.zero_()
        self.episode_length = 0
        if state is not None:
            self.state.copy_(torch.as_tensor(state, device=self.device))
        if wheel_speed is not None:
            self.wheel_speed.copy_(torch.as_tensor(wheel_speed, device=self.device))
        return self.state.clone()

    def current_state(self):
        return State_struct_torch(pos=self.state[:, :3], quat=self.state[:, 3:7],
            vel=self.state[:, 7:10], ang_vel=self.state[:, 10:13], n_envs=self.num_envs, device=self.device)

    def step(self, action):
        """Step the environment.

        Args:
            action: (num_envs, 2) physical commands [velocity (m/s), steering (rad)]

        Returns:
            The post-step world state, with the source environment reset behavior.
        """
        # SimRunner._to_env_action_space: controller actions are in m/s and rad.
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        normalized = action.clone()
        normalized[:, 0] = torch.clamp(action[:, 0] / self.max_velocity, -1.0, 1.0)
        normalized[:, 1] = torch.clamp(action[:, 1] / self.max_steering, -1.0, 1.0)
        action = normalized
        # Convert action to torch
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device, dtype=torch.float32)

        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Store last action
        self.last_action = action.clone()
        self.last_reset_ids = None

        # Snapshot state for already-terminated envs so we can freeze them.
        has_frozen = self.env_terminated.any()
        if has_frozen:
            frozen_mask = self.env_terminated.clone()
            frozen_state = self.state[frozen_mask].clone()
            frozen_vel_body = self.vel_body[frozen_mask].clone()
            frozen_wheel_speed = self.wheel_speed[frozen_mask].clone()

        # Denormalize actions from [-1, 1] to physical units
        throttle_normalized = action[:, 0]
        steering_normalized = action[:, 1]
        desired_wheel_speed = torch.clamp(
            throttle_normalized * self.max_velocity,
            self.min_wheel_velocity, self.max_wheel_velocity,
        )
        steering_angle = steering_normalized * self.max_steering

        # Current state
        pos = self.state[:, 0:3]
        quat = self.state[:, 3:7]

        # Prepare action for dynamic model: (N, 1, 2) where horizon=1
        actions_bicycle = torch.stack([desired_wheel_speed, steering_angle], dim=1).unsqueeze(1)

        # Rollout dynamic bicycle model for one step
        new_state, final_wheel = self.model.rollout(self.state, actions_bicycle, self.dt,
            self.wheel_speed, body_velocity=self.vel_body)

        # WheeledLab analytical rollouts use z=0 even when the reset pose has height.
        new_state[..., 2] = 0.0
        # Extract new state (squeeze time dimension)
        self.state[..., :13] = new_state[:, 0, :]

        # Update body frame velocities for next step
        new_quat = self.state[:, 3:7]
        vel_world = self.state[:, 7:10]
        ang_vel = self.state[:, 10:13]

        yaw = yaw_from_quat_torch(new_quat).squeeze(-1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        v_long = vel_world[:, 0] * cos_yaw + vel_world[:, 1] * sin_yaw
        v_lat = -vel_world[:, 0] * sin_yaw + vel_world[:, 1] * cos_yaw
        omega = ang_vel[:, 2]

        self.vel_body = torch.stack([v_long, v_lat, omega], dim=-1)

        wheel_speed = final_wheel
        if wheel_speed is not None:
            self.wheel_speed = wheel_speed.to(self.device, dtype=torch.float32)

        # Restore frozen envs (undo the physics step for already-terminated envs)
        if has_frozen:
            self.state[frozen_mask] = frozen_state
            self.vel_body[frozen_mask] = frozen_vel_body
            self.wheel_speed[frozen_mask] = frozen_wheel_speed

        boundary_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.bounded:
            half = self.arena_size / 2.0
            x = self.state[:, 0]
            y = self.state[:, 1]
            dist_to_boundary = torch.stack([
                x + half, half - x, y + half, half - y,
            ], dim=1).min(dim=1).values
            boundary_collision = dist_to_boundary < self.car_radius

        invalid_state = ~torch.isfinite(self.state).all(dim=1)

        obstacle_collision = self._check_obstacle_collisions()
        newly_terminated = (boundary_collision | obstacle_collision | invalid_state) & ~self.env_terminated
        self.env_terminated |= newly_terminated
        terminated = newly_terminated.cpu().numpy()

        # --- Auto-reset handling ---
        if self.auto_reset_episodes and self.env_terminated.any():
            if self.sync_reset:
                if self.env_terminated.all():
                    reset_ids = torch.arange(self.num_envs, device=self.device)
                    self._reset_envs(reset_ids)
                    self.last_reset_ids = reset_ids
            else:
                reset_ids = self.env_terminated.nonzero(as_tuple=False).squeeze(-1)
                if reset_ids.ndim == 0:
                    reset_ids = reset_ids.unsqueeze(0)
                self._reset_envs(reset_ids)
                self.last_reset_ids = reset_ids

        # Get observation (after potential resets so obs reflects fresh state)
        obs = self.state.clone()

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        truncated = np.zeros(self.num_envs, dtype=bool)
        info = {
            "boundary_collision": boundary_collision.cpu().numpy(),
            "obstacle_collision": obstacle_collision.cpu().numpy(),
            "invalid_state": invalid_state.cpu().numpy(),
        }
        self.episode_length += 1

        return obs

    def reset_state(self):
        """Reset state to initial conditions with environment spacing."""
        # Calculate grid layout for environments (like Isaac Lab)
        num_cols = int(np.ceil(np.sqrt(self.num_envs)))
        num_rows = int(np.ceil(self.num_envs / num_cols))
        idx = torch.arange(self.num_envs, device=self.device)
        row = idx // num_cols
        col = idx % num_cols

        x_offset = (col.to(torch.float32) - (num_cols - 1) / 2.0) * float(self.env_spacing)
        y_offset = (row.to(torch.float32) - (num_rows - 1) / 2.0) * float(self.env_spacing)

        self.state[:, 0] = x_offset + float(self.start_pos_x)
        self.state[:, 1] = y_offset + float(self.start_pos_y)
        self.state[:, 2] = float(self.start_pos_z)

        # Apply position randomization if ranges are set (per-environment sampling)
        if self.start_pos_x_range is not None:
            x_min, x_max = self.start_pos_x_range
            if x_min == x_max:
                offsets = torch.full((self.num_envs,), x_min, dtype=torch.float32, device=self.device)
            else:
                offsets = torch.empty(self.num_envs, dtype=torch.float32, device=self.device).uniform_(x_min, x_max)
            self.state[:, 0] += offsets

        if self.start_pos_y_range is not None:
            y_min, y_max = self.start_pos_y_range
            if y_min == y_max:
                offsets = torch.full((self.num_envs,), y_min, dtype=torch.float32, device=self.device)
            else:
                offsets = torch.empty(self.num_envs, dtype=torch.float32, device=self.device).uniform_(y_min, y_max)
            self.state[:, 1] += offsets

        # Orientation: quaternion from requested yaw
        # Sample per-environment yaw if randomization range is set
        if self.initial_yaw_range is not None:
            yaw_min, yaw_max = self.initial_yaw_range
            if yaw_min != yaw_max:
                # Sample different yaw for each environment
                yaw_tensor = torch.empty(self.num_envs, 1, dtype=torch.float32, device=self.device).uniform_(yaw_min, yaw_max, generator=self._rng)
            else:
                # Use provided constant yaw
                yaw_tensor = torch.full((self.num_envs, 1), yaw_min, dtype=torch.float32, device=self.device)
        else:
            # No randomization, use base start_rot_yaw
            yaw_tensor = torch.full((self.num_envs, 1), self.start_rot_yaw, dtype=torch.float32, device=self.device)

        quat = quat_from_yaw_torch(yaw_tensor)
        self.state[:, 3:7] = quat

        # Velocities: zero
        self.state[:, 7:13] = 0.0
        self.vel_body[:] = 0.0
        self.wheel_speed[:] = 0.0

        if self.visualizer is not None:
            self.visualizer.clear_trail()

    def _reset_envs(self, env_ids: torch.Tensor):
        """Reset a subset of environments by index.

        Mirrors reset_state() logic but only touches the given env indices.
        """
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        num_cols = int(np.ceil(np.sqrt(self.num_envs)))

        row = env_ids // num_cols
        col = env_ids % num_cols
        num_rows = int(np.ceil(self.num_envs / num_cols))

        x_offset = (col.to(torch.float32) - (num_cols - 1) / 2.0) * float(self.env_spacing)
        y_offset = (row.to(torch.float32) - (num_rows - 1) / 2.0) * float(self.env_spacing)

        self.state[env_ids, 0] = x_offset + float(self.start_pos_x)
        self.state[env_ids, 1] = y_offset + float(self.start_pos_y)
        self.state[env_ids, 2] = float(self.start_pos_z)

        if self.start_pos_x_range is not None:
            x_min, x_max = self.start_pos_x_range
            if x_min == x_max:
                self.state[env_ids, 0] += x_min
            else:
                self.state[env_ids, 0] += torch.empty(n, dtype=torch.float32, device=self.device).uniform_(x_min, x_max)

        if self.start_pos_y_range is not None:
            y_min, y_max = self.start_pos_y_range
            if y_min == y_max:
                self.state[env_ids, 1] += y_min
            else:
                self.state[env_ids, 1] += torch.empty(n, dtype=torch.float32, device=self.device).uniform_(y_min, y_max)

        if self.initial_yaw_range is not None:
            yaw_min, yaw_max = self.initial_yaw_range
            if yaw_min != yaw_max:
                yaw_tensor = torch.empty(n, 1, dtype=torch.float32, device=self.device).uniform_(yaw_min, yaw_max, generator=self._rng)
            else:
                yaw_tensor = torch.full((n, 1), yaw_min, dtype=torch.float32, device=self.device)
        else:
            yaw_tensor = torch.full((n, 1), self.start_rot_yaw, dtype=torch.float32, device=self.device)

        self.state[env_ids, 3:7] = quat_from_yaw_torch(yaw_tensor)
        self.state[env_ids, 7:13] = 0.0
        self.vel_body[env_ids] = 0.0
        self.wheel_speed[env_ids] = 0.0
        self.env_terminated[env_ids] = False

        if self.visualizer is not None:
            for idx in env_ids.cpu().tolist():
                self.visualizer.clear_trail(vehicle_idx=idx)

    def _check_obstacle_collisions(self):
        # The synthetic open-loop collection preset has no obstacle task.
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)



@dataclass
class CommandConfig:
    duration: float = 10.0
    action_bounds: list = field(default_factory=lambda: [[-2.0, 5.0], [-0.5, 0.5]])
    noise_std: list = field(default_factory=lambda: [0.2, 0.0])
    curve_methods: list = field(default_factory=lambda: [["bezier"], ["random_walk"]])
    offset_fraction: float = 0.0
    bezier_full_range: bool = True
    seed: int = 42
    initial_velocity: list = field(default_factory=lambda: [0.0, 0.0])
    initial_yaw: list | None = None
    parameter_randomization: float = 0.0
    parameter_ranges: dict = field(default_factory=dict)
    resample_params_each_run: bool = False
    fix_control_seed: bool = False
    fix_run_seed: bool = False
    env_spacing: float = 2.5
    max_velocity: float = 7.5
    max_lateral_acceleration_mps2: float | None = 10.0

    def __post_init__(self):
        if self.duration <= 0 or not math.isfinite(self.duration) or not 0 <= self.parameter_randomization < 1:
            raise ValueError("Invalid duration or parameter randomization fraction")
        if len(self.action_bounds) != 2 or len(self.noise_std) != 2 or len(self.curve_methods) != 2:
            raise ValueError("Configure exactly two command channels")
        if any(len(b) != 2 or b[0] > b[1] for b in self.action_bounds) or any(s < 0 for s in self.noise_std):
            raise ValueError("Invalid action bounds or noise")
        for interval in (*self.action_bounds, self.initial_velocity, *([self.initial_yaw] if self.initial_yaw is not None else [])):
            if len(interval) != 2 or interval[0] > interval[1] or not all(math.isfinite(v) for v in interval):
                raise ValueError("Initial-state and action ranges must have finite, ordered endpoints")
        if self.seed < 0 or not isinstance(self.seed, int) or not 0 <= self.offset_fraction <= 1:
            raise ValueError("Invalid seed or offset fraction")
        if not all(math.isfinite(v) for v in self.noise_std):
            raise ValueError("Noise standard deviations must be finite")
        if any(not methods or any(m not in ("bezier", "random_walk", "fourier", "polynomial") for m in methods) for methods in self.curve_methods):
            raise ValueError("Unsupported or empty curve method list")


        self.parameter_ranges = validate_parameter_ranges(self.parameter_ranges)
        if not isinstance(self.bezier_full_range, bool):
            raise ValueError("bezier_full_range must be true or false")
        if tuple(self.initial_velocity) != (0.0, 0.0):
            raise ValueError("WheeledLab resets body and wheel velocities to zero; initial_velocity must be [0, 0]")
        if not math.isfinite(self.env_spacing) or self.env_spacing < 0 or not math.isfinite(self.max_velocity) or self.max_velocity <= 0:
            raise ValueError("Invalid environment spacing or velocity limit")
        if self.max_lateral_acceleration_mps2 is not None and (not math.isfinite(self.max_lateral_acceleration_mps2) or self.max_lateral_acceleration_mps2 <= 0):
            raise ValueError("Lateral acceleration limit must be positive and finite")


def simulation_params(params, commands, dt):
    params = params if isinstance(params, DBMParams) else DBMParams(**(params or {}))
    values = asdict(params)
    values["max_steering"] = values.pop("max_steering_angle")
    return BicycleParams(**values, dt=dt, max_velocity=commands.max_velocity,
        max_lateral_acceleration_mps2=commands.max_lateral_acceleration_mps2)


def randomized_params(base, commands, run_index):
    ranges = dict(commands.parameter_ranges)
    if commands.parameter_randomization:
        for key in ("len_f", "len_r", "C_throttle_a", "C_rollf", "C_drag", "C_tire_f", "C_tire_r", "wheel_time_constant", "mass", "inertia"):
            value = getattr(base, key)
            ranges.setdefault(key, [value * (1-commands.parameter_randomization), value * (1+commands.parameter_randomization)])
    ranges = validate_parameter_ranges(ranges)
    ranged = replace(base, **{key: value for key, value in ranges.items() if key not in ACTION_BIAS_KEYS})
    # WheeledLab samples one scalar parameter set per run, shared by all envs.
    seed = commands.seed + (run_index if commands.resample_params_each_run and not commands.fix_run_seed else 0)
    resolved = ParamRandomizer(seed=seed).randomize_params(ranged)
    model_params(resolved)  # Validate sampled model parameters before allocation.
    if not math.isfinite(resolved.max_velocity) or resolved.max_velocity <= 0:
        raise ValueError("Sampled max_velocity must be positive and finite")
    for key in RANDOMIZATION_KEYS:
        if key in ACTION_BIAS_KEYS:
            continue
        value = getattr(resolved, key)
        if value is None and key == "max_lateral_acceleration_mps2":
            continue
        if not math.isfinite(value):
            raise ValueError(f"Sampled {key} must be finite")
    limit = resolved.max_lateral_acceleration_mps2
    if limit is not None and limit <= 0:
        raise ValueError("Sampled lateral acceleration limit must be positive")
    return resolved


def make_controller(commands, dt, num_steps, run_index, device):
    seed = commands.seed + (0 if commands.fix_control_seed else run_index)
    return ParametricOpenLoopController(action_dim=2, device=device, horizon_steps=num_steps,
        default_dt=dt, action_bounds=commands.action_bounds, curve_methods=commands.curve_methods,
        noise_std=commands.noise_std, clip_actions=True, offset_fraction=commands.offset_fraction, seed=seed,
        bezier_full_range=commands.bezier_full_range)


def generate(root, terrain, params=None, data=None, commands=None, episodes=None,
             batch_size=32, device="cpu", fractions=(0.8, 0.1, 0.1), progress=None, cancelled=None,
             calibration=None, process_windows=False, runs=None):
    """Each WheeledLab run produces one [batch_size, duration/dt, 15] raw bin.

    episodes is a legacy argument alias for runs; GUI/CLI now call the setting Runs.
    Only raw collection happens here; Dataset builder processes the saved bins.
    """
    device = resolve_device(device)
    runs = runs if runs is not None else episodes if episodes is not None else 10
    data, commands = data or DataConfig(), commands or CommandConfig()
    if not isinstance(runs, int) or not isinstance(batch_size, int) or runs < 1 or batch_size < 1:
        raise ValueError("Runs and batch_size must be positive integers")
    if process_windows:
        raise ValueError("Synthetic collection saves raw bins. Use Dataset builder for training windows.")
    steps = round(commands.duration / data.dt)
    if steps < 1 or abs(steps*data.dt - commands.duration) > 1e-6:
        raise ValueError("Duration must be an integer multiple of dt")
    session_split("validate", commands.seed, fractions)
    base = simulation_params(params, commands, data.dt)
    randomized_params(base, commands, 0)  # Validate range specs before creating output.
    provenance = {"pipeline": "wheeledlab_batched_v1", "params": asdict(model_params(base)),
        "commands": asdict(commands), "batch_size": batch_size, "fractions": list(fractions), "calibration": calibration or {}}
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(root)
    completed = 0
    started = time.monotonic()
    # Reuse the existing output lock, configuration guard, and manifest commits.
    with DatasetWriter(root, data, "sim", terrain, provenance, process_windows=False) as writer:
        if len(writer.manifest["episodes"]) > runs:
            raise ValueError("Output contains more completed runs than requested; choose a new output folder")
        writer.manifest.update(num_envs=batch_size, num_runs=runs)
        writer.manifest["schema"] = {**writer.manifest["schema"], "action_timing": "post-step state with the command applied in the preceding interval"}
        writer.dirty = True
        context = DataCollectionContext(project_name="optcar", dataset_name=root.name,
            device_label=device, num_envs=batch_size, control_dt=data.dt, randomization_yaml=None,
            env_file=None, env_cfg_file=None, missions_file=None, num_runs=runs,
            max_steps_per_run=steps, sim_dt=data.dt, decimation=1)
        collector = DataCollector(root, context, dataset_dir=root)
        write_json(root / "metadata/schema.json", writer.manifest["schema"])
        write_json(root / "metadata/collection.json", provenance)
        for run_index in range(runs):
            if cancelled and cancelled():
                break
            batch_id = f"batch_{identity}_{run_index:09d}"
            if writer.contains(batch_id):
                record = writer.manifest["episodes"][batch_id]
                if not (root / record["raw"]).is_file() or (root / record["raw"]).stat().st_size != batch_size*steps*15*4:
                    raise ValueError("A saved batch is missing or incomplete; choose a new output folder")
                completed += 1
                continue
            resolved = randomized_params(base, commands, run_index)
            env = DynamicBicycleEnv(batch_size, data.dt, resolved, device,
                env_spacing=commands.env_spacing, initial_yaw=commands.initial_yaw, seed=commands.seed+(0 if commands.fix_run_seed else run_index))
            controller = make_controller(commands, data.dt, steps, run_index, device)
            biases = ActionBiasSampler(commands.parameter_ranges, batch_size, device,
                commands.seed+(0 if commands.fix_run_seed else run_index))
            collector.start_run(run_index)
            interrupted = False
            for step in range(steps):
                if cancelled and cancelled():
                    interrupted = True
                    break
                with torch.inference_mode():
                    actions = controller.compute_control(env.current_state(), dt=data.dt)
                    actions = biases._apply_action_bias(actions)
                    env.step(actions)
                    state_vec = env.current_state().get_vector_state(["pos", "quat", "vel", "ang_vel"])
                    collector.record_step(state_vec, actions)
                if progress:
                    progress({"completed": completed*steps+step+1, "total": runs*steps, "new_windows": 0,
                        "current": f"Run {run_index+1}/{runs} · {batch_size} trajectories · step {step+1}/{steps} · {device}",
                        "elapsed_s": time.monotonic()-started})
            if interrupted:
                break
            file = collector.finish_run()
            writer.manifest["episodes"][batch_id] = {"session": batch_id,
                "split": session_split(batch_id, commands.seed, fractions),
                "raw": str(file.relative_to(root.resolve())), "raw_shape": [batch_size, steps, 15],
                "shards": [], "windows": 0, "metadata": {"run_index": run_index,
                "seed": commands.seed+(0 if commands.fix_run_seed else run_index), "params": asdict(resolved),
                "action_bias": torch.stack([biases._velocity_bias, biases._steering_bias], -1).cpu().tolist()}}
            writer.dirty = True
            writer.commit()
            completed += 1
        collector.finalize()
    return {"completed": completed, "total": runs, "trajectories": completed*batch_size,
        "batch_size": batch_size, "new_windows": 0, "stage": "raw", "dataset": str(root), "device": device}
