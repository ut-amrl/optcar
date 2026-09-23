"""Command curves and a headless DBM environment for synthetic FKD datasets."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np
import torch

from optcar.dataset import DataConfig, DatasetWriter, session_split
from optcar.models.dynamic_bicycle import DBMParams, DynamicBicycleModel


class DynamicBicycleEnv:
    model_class = DynamicBicycleModel

    def __init__(self, num_envs=1, dt=0.02, params=None, device="cpu"):
        if num_envs < 1 or dt <= 0:
            raise ValueError("num_envs and dt must be positive")
        self.num_envs, self.dt = num_envs, dt
        self.model = self.model_class(params, device)
        self.device = self.model.device
        self.reset()

    def reset(self, state=None, wheel_speed=None):
        self.state = torch.zeros((self.num_envs, 13), device=self.device)
        self.state[:, 3] = 1
        if state is not None:
            self.state.copy_(torch.as_tensor(state, device=self.device))
        self.wheel_speed = torch.zeros(self.num_envs, device=self.device)
        if wheel_speed is not None:
            self.wheel_speed.copy_(torch.as_tensor(wheel_speed, device=self.device))
        return self.state.clone()

    def step(self, action):
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        future, self.wheel_speed = self.model.rollout(self.state, action[:, None], self.dt, self.wheel_speed)
        self.state = future[:, -1]
        return self.state.clone()

    def collect(self, actions):
        """Batched episode rollout, including the initial state."""
        initial = self.state.clone()
        future, self.wheel_speed = self.model.rollout(self.state, actions, self.dt, self.wheel_speed)
        self.state = future[:, -1]
        if not torch.isfinite(future).all():
            raise ValueError("Nonfinite DBM trajectory; check parameters, initial state, or reduce dt")
        return torch.cat((initial[:, None], future), dim=1)


@dataclass
class CommandConfig:
    duration: float = 10.0
    action_bounds: list = field(default_factory=lambda: [[-2.0, 5.0], [-0.5, 0.5]])
    noise_std: list = field(default_factory=lambda: [0.2, 0.0])
    curve_methods: list = field(default_factory=lambda: [["bezier"], ["random_walk"]])
    offset_fraction: float = 0.0
    seed: int = 42
    initial_velocity: list = field(default_factory=lambda: [0.0, 0.0])
    initial_yaw: list = field(default_factory=lambda: [0.0, 0.0])
    parameter_randomization: float = 0.0

    def __post_init__(self):
        if self.duration <= 0 or not math.isfinite(self.duration) or not 0 <= self.parameter_randomization < 1:
            raise ValueError("Invalid duration or parameter randomization fraction")
        if len(self.action_bounds) != 2 or len(self.noise_std) != 2 or len(self.curve_methods) != 2:
            raise ValueError("Configure exactly two command channels")
        if any(len(b) != 2 or b[0] > b[1] for b in self.action_bounds) or any(s < 0 for s in self.noise_std):
            raise ValueError("Invalid action bounds or noise")
        for interval in (*self.action_bounds, self.initial_velocity, self.initial_yaw):
            if len(interval) != 2 or interval[0] > interval[1] or not all(math.isfinite(v) for v in interval):
                raise ValueError("Initial-state and action ranges must have finite, ordered endpoints")
        if self.seed < 0 or not isinstance(self.seed, int) or not 0 <= self.offset_fraction <= 1:
            raise ValueError("Invalid seed or offset fraction")
        if not all(math.isfinite(v) for v in self.noise_std):
            raise ValueError("Noise standard deviations must be finite")
        if any(not methods or any(m not in ("bezier", "random_walk", "fourier", "polynomial") for m in methods) for methods in self.curve_methods):
            raise ValueError("Unsupported or empty curve method list")


def _command_curve(steps, dt, rng, bounds, methods, noise_std, offset_fraction):
    """WheeledLab command curves without separate scalar/batch implementations."""
    low, high = bounds
    method = methods[0] if len(methods) == 1 else rng.choice(methods)
    time = np.linspace(0.0, 1.0, steps)
    if method == "bezier":
        count = int(rng.integers(4, 12))
        control = rng.uniform(low, high, size=(1, count))
        degree = count - 1
        basis = np.stack([math.comb(degree, i) * (1 - time)**(degree - i) * time**i
                          for i in range(count)])
        signal = (control @ basis)[0]
    elif method == "random_walk":
        noise = rng.uniform(-1.0, 1.0, steps)
        drift = 0.4 * np.sign(noise) * rng.uniform(0.5, 1.0, steps)
        signal = np.zeros(steps)
        signal[0] = rng.uniform(low, high)
        scale = (1.0 - 0.95) * (high - low) * 0.5
        for i in range(1, steps):
            signal[i] = 0.95 * signal[i - 1] + scale * noise[i] + drift[i]
    elif method == "fourier":
        coefficients = rng.uniform(-1.0, 1.0, int(rng.integers(1, 4)))
        signal = np.full(steps, coefficients[0])
        seconds = np.linspace(0.0, dt * (steps - 1), steps)
        for i, coefficient in enumerate(coefficients[1:], 1):
            signal += coefficient * np.sin(2 * math.pi * i * seconds / max(dt * steps, 1e-3))
    elif method == "polynomial":
        coefficients = rng.uniform(-1.0, 1.0, int(rng.integers(3, 10)) + 1)
        signal = np.polyval(coefficients, np.linspace(-1.0, 1.0, steps))
    else:
        raise ValueError(f"Unsupported curve method: {method}")

    signal = signal.astype(np.float32)
    span = signal.max() - signal.min()
    signal = ((signal - signal.min()) / span * (high - low) + low
              if span > 0 else np.full_like(signal, low + 0.5 * (high - low)))
    if offset_fraction > 0 and high > low:
        offset = np.float32(rng.uniform(-offset_fraction, offset_fraction)) * (high - low)
        signal = np.clip(signal + offset, low, high)
    if noise_std:
        signal += rng.normal(0.0, noise_std, steps).astype(np.float32)
    return np.clip(signal, low, high)


def sample_commands(config, dt, episode):
    steps = round(config.duration / dt)
    if steps < 1 or abs(steps * dt - config.duration) > 1e-6:
        raise ValueError("Duration must be an integer multiple of dt")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, episode]))
    channels = []
    for bounds, noise, methods in zip(config.action_bounds, config.noise_std, config.curve_methods):
        channels.append(_command_curve(steps, dt, rng, bounds, methods, noise, config.offset_fraction))
    return np.stack(channels, -1).astype(np.float32), rng


def generate(root, terrain, params=None, data=None, commands=None, episodes=1000,
             batch_size=128, device="cpu", fractions=(0.8, 0.1, 0.1), progress=None, cancelled=None,
             calibration=None):
    data = data or DataConfig()
    commands = commands or CommandConfig()
    params = params if isinstance(params, DBMParams) else DBMParams(**(params or {}))
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive")
    if round(commands.duration / data.dt) + 1 < data.window_steps:
        raise ValueError("Episode is shorter than the requested history + future window")
    provenance = {"params": asdict(params), "commands": asdict(commands), "fractions": list(fractions),
                  "calibration": calibration or {}}
    # JSON normalization keeps tuple/list metadata identical after a restart.
    provenance = json.loads(json.dumps(provenance))
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
    completed, windows = 0, 0
    with DatasetWriter(root, data, "sim", terrain, provenance) as writer:
        for first in range(0, episodes, batch_size):
            if cancelled and cancelled():
                break
            indices = [i for i in range(first, min(first + batch_size, episodes))
                       if not writer.contains(f"sim_{identity}_{i:09d}")]
            completed += min(batch_size, episodes - first) - len(indices)
            if not indices:
                continue
            signals, initial, wheels, rngs = [], [], [], []
            for index in indices:
                signal, rng = sample_commands(commands, data.dt, index)
                yaw = rng.uniform(*commands.initial_yaw)
                speed = rng.uniform(*commands.initial_velocity)
                state = np.zeros(13, np.float32)
                state[3], state[6] = np.cos(yaw / 2), np.sin(yaw / 2)
                state[7], state[8] = speed * np.cos(yaw), speed * np.sin(yaw)
                signals.append(signal)
                initial.append(state)
                wheels.append(speed)
                rngs.append(rng)
            env = DynamicBicycleEnv(len(indices), data.dt, params, device)
            # Per-episode parameters broadcast through the Torch reference functions.
            parameter_values = [asdict(params) for _ in indices]
            if commands.parameter_randomization:
                keys = ("mass", "inertia", "C_tire_f", "C_tire_r", "C_throttle_a", "C_rollf", "C_drag", "wheel_time_constant")
                f = commands.parameter_randomization
                for values, rng in zip(parameter_values, rngs):
                    for key in keys:
                        values[key] *= rng.uniform(1 - f, 1 + f)
                for key in keys:
                    env.model.tensors[key] = torch.tensor([v[key] for v in parameter_values], device=device, dtype=torch.float32)
            env.reset(np.stack(initial), np.asarray(wheels, np.float32))
            actions = env.model.clamp_actions(torch.as_tensor(np.stack(signals), device=device))
            with torch.inference_mode():
                states = env.collect(actions).cpu().numpy()
            actions = actions.cpu().numpy()
            for offset, index in enumerate(indices):
                if cancelled and cancelled():
                    break
                episode = f"sim_{identity}_{index:09d}"
                split = session_split(episode, commands.seed, fractions)
                windows += writer.write_episode(episode, states[offset], actions[offset], split,
                    metadata={"episode_index": index, "seed": commands.seed, "params": parameter_values[offset]})
                completed += 1
                if progress:
                    progress({"completed": completed, "total": episodes, "new_windows": windows})
            writer.commit()
    return {"completed": completed, "total": episodes, "new_windows": windows, "dataset": str(root)}
