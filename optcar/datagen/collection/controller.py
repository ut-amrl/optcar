"""Open-loop control signals for batched data collection."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, Any
import torch
import numpy as np
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

if TYPE_CHECKING:
    from optcar.utils.state_struct import State_struct_torch


class BaseController(ABC):
    """Abstract base class for all controllers."""
    
    def __init__(self, config, **kwargs):
        """
        Initialize base controller.
        
        Args:
            config: Controller configuration object
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self.cntrl_config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        self.dtype = config.dtype if hasattr(config, 'dtype') else torch.float32
        
        # Task can be set later
        self.task = None
        
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset controller state.

        Args:
            seed: Optional seed to reset any controller RNG. When None, the
                controller should keep its previously configured seed behavior.
        """
        pass
    
    @abstractmethod
    def compute_control(
        self, 
        curr_states: State_struct_torch,
        **kwargs
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute control action based on current state.
        
        Args:
            curr_states: Current state of the robot(s)
            **kwargs: Additional arguments (past_states, past_actions, etc.)
            
        Returns:
            actions: (batch_size, action_dim) control actions
            info: Dictionary with additional information
        """
        pass
    
    def set_task(self, task):
        """Set the current task for the controller."""
        self.task = task
    
    def get_task(self):
        """Get the current task."""
        return self.task


@dataclass
class SignalGeneratorConfig:
    """Configuration for a single action channel."""

    min_val: float
    max_val: float
    noise_std: float


class ParametricSignalGenerator:
    """Helper that creates smooth randomized 1-D signals."""

    def __init__(
        self,
        num_steps: int,
        dt: float,
        rng: np.random.Generator,
        config: SignalGeneratorConfig,
        methods: Sequence[str],
        max_offset_fraction: float = 0.25,
        bezier_full_range: bool = True,
    ) -> None:
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.rng = rng
        self.config = config
        self.methods: List[str] = list(methods)
        self.max_offset_fraction = float(max_offset_fraction)
        self.bezier_full_range = bezier_full_range
        self.time = np.linspace(0.0, self.dt * (self.num_steps - 1), self.num_steps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(self) -> np.ndarray:
        signal = self._generate_base_curve()
        if self.bezier_full_range or self._sampled_method != "bezier":
            signal = self._scale_to_bounds(signal)
        if self.max_offset_fraction > 0.0:
            span = self.config.max_val - self.config.min_val
            if span > 0.0:
                offset = self.rng.uniform(-self.max_offset_fraction, self.max_offset_fraction) * span
                signal = np.clip(signal + offset, self.config.min_val, self.config.max_val)
        return signal

    def add_gaussian_noise(self, signal: np.ndarray, clip: bool = True) -> np.ndarray:
        noise = self.rng.normal(loc=0.0, scale=self.config.noise_std, size=self.num_steps)
        noisy = signal + noise
        if clip:
            return np.clip(noisy, self.config.min_val, self.config.max_val)
        return noisy

    # ------------------------------------------------------------------
    # Curve generation methods
    # ------------------------------------------------------------------
    def _generate_base_curve(self) -> np.ndarray:
        method = self.rng.choice(self.methods)
        self._sampled_method = method
        if method == "fourier":
            return self._fourier_curve()
        if method == "bezier":
            return self._bezier_curve()
        if method == "random_walk":
            return self._random_walk()
        if method == "polynomial":
            return self._polynomial_curve()
        raise ValueError(f"Unknown curve method '{method}'")

    def _fourier_curve(self) -> np.ndarray:
        num_terms = int(self.rng.integers(1, 4))  # includes c0 term
        coeffs = self.rng.uniform(-1.0, 1.0, size=num_terms)
        signal = np.full(self.num_steps, coeffs[0])
        base_period = max(self.dt * self.num_steps, 1e-3)
        for i in range(1, num_terms):
            signal += coeffs[i] * np.sin(2.0 * math.pi * i * self.time / base_period)
        return signal

    def _bezier_curve(self) -> np.ndarray:
        num_points = int(self.rng.integers(4, 12))
        control = self.rng.uniform(self.config.min_val, self.config.max_val, size=num_points)
        t_norm = np.linspace(0.0, 1.0, self.num_steps)
        curve = np.zeros_like(t_norm)
        n = num_points - 1
        for i in range(num_points):
            binomial = math.comb(n, i)
            curve += binomial * ((1 - t_norm) ** (n - i)) * (t_norm ** i) * control[i]
        return curve

    def _random_walk(self) -> np.ndarray:
        noise = self.rng.uniform(-1.0, 1.0, size=self.num_steps)
        signal = np.zeros(self.num_steps)
        signal[0] = self.rng.uniform(self.config.min_val, self.config.max_val)
        # alpha = 0.95
        drift_rate = 0.1
        span = self.config.max_val - self.config.min_val
        for i in range(1, self.num_steps):
            # drift = drift_rate * np.sign(noise[i]) * self.rng.uniform(0.5, 1.0)
            # signal[i] = (
            #     alpha * signal[i - 1]
            #     + (1 - alpha) * noise[i] * span * 0.5
            #     + drift
            # )
            signal[i] += noise[i] * drift_rate
            signal[i] = np.clip(signal[i], self.config.min_val, self.config.max_val)

        return signal

    def _polynomial_curve(self) -> np.ndarray:
        order = int(self.rng.integers(3, 10))
        coeffs = self.rng.uniform(-1.0, 1.0, size=order + 1)
        x = np.linspace(-1.0, 1.0, self.num_steps)
        return np.polyval(coeffs, x)

    def _scale_to_bounds(self, signal: np.ndarray) -> np.ndarray:
        span = self.config.max_val - self.config.min_val
        if span <= 0.0:
            return np.full_like(signal, self.config.max_val)
        signal_min = float(signal.min())
        signal_max = float(signal.max())
        if math.isclose(signal_max, signal_min):
            return np.full_like(signal, self.config.min_val + 0.5 * span)
        normalized = (signal - signal_min) / (signal_max - signal_min)
        return normalized * span + self.config.min_val

    # ------------------------------------------------------------------
    # Vectorized (batch-over-envs) generation
    #
    # The per-env loop above is a Python bottleneck at large num_envs.
    # The methods below produce identical-shape signals but compute them
    # for an entire (num_envs, num_steps) block at once.
    #
    # Note: the order of RNG draws differs from the per-env version, so a
    # given seed will produce a different (still reproducible) ensemble of
    # trajectories than the loop. Per-env and per-step independence is
    # preserved.
    # ------------------------------------------------------------------
    def sample_batch(self, num_envs: int) -> np.ndarray:
        if num_envs <= 0:
            return np.zeros((0, self.num_steps), dtype=np.float32)

        # Pick a method per env. Group identical methods together so each
        # method's batched generator is called once per group.
        if len(self.methods) == 1:
            method_per_env = np.array([self.methods[0]] * num_envs)
        else:
            method_per_env = self.rng.choice(np.array(self.methods), size=num_envs)

        out = np.empty((num_envs, self.num_steps), dtype=np.float32)
        for method in np.unique(method_per_env):
            mask = method_per_env == method
            n_in = int(mask.sum())
            if method == "fourier":
                sub = self._fourier_batch(n_in)
            elif method == "bezier":
                sub = self._bezier_batch(n_in)
            elif method == "random_walk":
                sub = self._random_walk_batch(n_in)
            elif method == "polynomial":
                sub = self._polynomial_batch(n_in)
            else:
                raise ValueError(f"Unknown curve method '{method}'")
            out[mask] = sub.astype(np.float32, copy=False)

        # Optional OptCar mode: retain sampled Bézier control points
        # without stretching each curve to touch both command bounds.
        native_bezier = out.copy() if not self.bezier_full_range else None
        out = self._scale_to_bounds_batch(out)
        if native_bezier is not None:
            mask = method_per_env == "bezier"
            out[mask] = native_bezier[mask]
        if self.max_offset_fraction > 0.0:
            span = self.config.max_val - self.config.min_val
            if span > 0.0:
                offsets = self.rng.uniform(
                    -self.max_offset_fraction, self.max_offset_fraction, size=num_envs
                ).astype(np.float32) * span
                out = np.clip(
                    out + offsets[:, None], self.config.min_val, self.config.max_val
                )
        return out

    def add_gaussian_noise_batch(self, signals: np.ndarray, clip: bool = True) -> np.ndarray:
        if self.config.noise_std == 0.0:
            noisy = signals.copy()
        else:
            noise = self.rng.normal(
                loc=0.0, scale=self.config.noise_std, size=signals.shape
            ).astype(signals.dtype, copy=False)
            noisy = signals + noise
        if clip:
            return np.clip(noisy, self.config.min_val, self.config.max_val)
        return noisy

    def _scale_to_bounds_batch(self, signals: np.ndarray) -> np.ndarray:
        span = self.config.max_val - self.config.min_val
        if span <= 0.0:
            return np.full_like(signals, self.config.max_val)
        sig_min = signals.min(axis=1, keepdims=True)
        sig_max = signals.max(axis=1, keepdims=True)
        ranges = sig_max - sig_min
        safe = np.where(ranges > 0, ranges, 1.0)
        normalized = (signals - sig_min) / safe
        out = normalized * span + self.config.min_val
        constant_rows = (ranges.squeeze(1) <= 0)
        if constant_rows.any():
            out[constant_rows] = self.config.min_val + 0.5 * span
        return out

    def _fourier_batch(self, num_envs: int) -> np.ndarray:
        num_terms_per_env = self.rng.integers(1, 4, size=num_envs)
        base_period = max(self.dt * self.num_steps, 1e-3)
        out = np.zeros((num_envs, self.num_steps), dtype=np.float64)
        for k in np.unique(num_terms_per_env):
            mask = num_terms_per_env == k
            n_in = int(mask.sum())
            coeffs = self.rng.uniform(-1.0, 1.0, size=(n_in, int(k)))
            signals = np.broadcast_to(coeffs[:, 0:1], (n_in, self.num_steps)).copy()
            for i in range(1, int(k)):
                basis = np.sin(2.0 * math.pi * i * self.time / base_period)
                signals += coeffs[:, i:i + 1] * basis[None, :]
            out[mask] = signals
        return out

    def _bezier_batch(self, num_envs: int) -> np.ndarray:
        num_points_per_env = self.rng.integers(4, 12, size=num_envs)
        t_norm = np.linspace(0.0, 1.0, self.num_steps)
        out = np.zeros((num_envs, self.num_steps), dtype=np.float64)
        for k in np.unique(num_points_per_env):
            mask = num_points_per_env == k
            n_in = int(mask.sum())
            ctrl = self.rng.uniform(
                self.config.min_val, self.config.max_val, size=(n_in, int(k))
            )
            n = int(k) - 1
            basis = np.empty((int(k), self.num_steps))
            for i in range(int(k)):
                basis[i] = math.comb(n, i) * ((1.0 - t_norm) ** (n - i)) * (t_norm ** i)
            out[mask] = ctrl @ basis
        return out

    def _random_walk_batch(self, num_envs: int) -> np.ndarray:
        n = self.num_steps
        span = self.config.max_val - self.config.min_val
        noise = self.rng.uniform(-1.0, 1.0, size=(num_envs, n))
        drift_unif = self.rng.uniform(0.5, 1.0, size=(num_envs, n))
        drift_term = 0.4 * np.sign(noise) * drift_unif
        s0 = self.rng.uniform(self.config.min_val, self.config.max_val, size=num_envs)

        signal = np.zeros((num_envs, n), dtype=np.float64)
        signal[:, 0] = s0
        alpha = 0.95
        scale = (1.0 - alpha) * span * 0.5
        # Time loop is still serial in time, but each iter is one vector op.
        for i in range(1, n):
            signal[:, i] = alpha * signal[:, i - 1] + scale * noise[:, i] + drift_term[:, i]
        return signal

    def _polynomial_batch(self, num_envs: int) -> np.ndarray:
        order_per_env = self.rng.integers(3, 10, size=num_envs)
        x = np.linspace(-1.0, 1.0, self.num_steps)
        out = np.zeros((num_envs, self.num_steps), dtype=np.float64)
        for o in np.unique(order_per_env):
            mask = order_per_env == o
            n_in = int(mask.sum())
            coeffs = self.rng.uniform(-1.0, 1.0, size=(n_in, int(o) + 1))
            # Horner's rule, leading coefficient first (matches np.polyval).
            signals = np.zeros((n_in, self.num_steps))
            for c_idx in range(int(o) + 1):
                signals = signals * x + coeffs[:, c_idx:c_idx + 1]
            out[mask] = signals
        return out


class ParametricOpenLoopController(BaseController):
    """Controller that streams randomized open-loop actions per environment.
    
    Args:
        action_dim: Number of action dimensions
        device: Device to store tensors on
        horizon_steps: Number of timesteps per sequence
        default_dt: Default timestep duration
        action_bounds: Min/max bounds for each action dimension
        curve_methods: Curve generation methods. Can be either:
            - A single list of methods (e.g. ["fourier", "bezier"]) applied to all actions
            - A list of lists, one per action (e.g. [["fourier"], ["bezier", "polynomial"]])
              to use different methods for each action dimension
        noise_std: Standard deviation of Gaussian noise per action
        clip_actions: Whether to clip actions to bounds
        offset_fraction: Maximum fraction of range to offset signals
        seed: Random seed for reproducibility
    
    Example:
        # Use only fourier for action 0, only bezier for action 1
        controller = ParametricOpenLoopController(
            action_dim=2,
            curve_methods=[["fourier"], ["bezier"]]
        )
        
        # Use all methods for all actions (default behavior)
        controller = ParametricOpenLoopController(
            action_dim=2,
            curve_methods=["fourier", "bezier", "polynomial", "random_walk"]
        )
    """

    def __init__(
        self,
        action_dim: int = 2,
        device: str = "cuda",
        horizon_steps: int = 512,
        default_dt: float = 0.02,
        action_bounds: Optional[Sequence[Sequence[float]]] = None,
        curve_methods: Optional[Iterable[str] | Sequence[Iterable[str]]] = None,
        noise_std: Optional[Sequence[float]] = None,
        clip_actions: bool = True,
        offset_fraction: float = 0.25,
        seed: Optional[int] = None,
        bezier_full_range: bool = True,
    ) -> None:
        self.bezier_full_range = bezier_full_range
        self.device = device
        self.task = None  # For BaseController compatibility
        self.action_dim = action_dim
        self.horizon_steps = int(horizon_steps)
        self.default_dt = float(default_dt)
        self.dt = float(default_dt)
        
        # Parse curve_methods - can be single list or list of lists (one per action)
        if curve_methods is None:
            default_methods = ("fourier", "bezier", "polynomial", "random_walk")
            self.curve_methods_per_action = [default_methods] * action_dim
        else:
            # Check if it's a list of lists or a single list
            curve_methods_list = list(curve_methods)
            if curve_methods_list and isinstance(curve_methods_list[0], (list, tuple)):
                # List of lists - one set of methods per action
                if len(curve_methods_list) != action_dim:
                    raise ValueError(
                        f"curve_methods has {len(curve_methods_list)} entries but action_dim is {action_dim}. "
                        "Provide either a single list of methods (applied to all actions) or "
                        "one list per action dimension."
                    )
                self.curve_methods_per_action = [tuple(methods) for methods in curve_methods_list]
            else:
                # Single list - apply to all actions
                methods_tuple = tuple(curve_methods_list)
                self.curve_methods_per_action = [methods_tuple] * action_dim
        
        self.clip_actions = clip_actions
        self._offset_fraction = float(max(0.0, offset_fraction))
        self.seed = seed
        self._rng = np.random.default_rng(self._seed_from_system(seed))

        if action_bounds is None:
            action_bounds = ((-3.0, 5.0), (-0.5, 0.5))
        if len(action_bounds) != action_dim:
            raise ValueError("action_bounds length must match action_dim")

        if noise_std is None:
            std_values = [1.0, 0.5][:action_dim]
            if len(std_values) < action_dim:
                std_values.extend([std_values[-1]] * (action_dim - len(std_values)))
        else:
            std_values = [float(v) for v in noise_std]
            if len(std_values) == 1:
                std_values = std_values * action_dim
            if len(std_values) != action_dim:
                raise ValueError("noise_std sequence must match action_dim")

        span_configs: List[SignalGeneratorConfig] = []
        for idx, bounds in enumerate[Sequence[float]](action_bounds):
            if len(bounds) != 2:
                raise ValueError("Each entry in action_bounds must have (min, max)")
            span_configs.append(
                SignalGeneratorConfig(
                    min_val=float(bounds[0]),
                    max_val=float(bounds[1]),
                    noise_std=float(std_values[idx]),
                )
            )
        self._channel_configs = span_configs

        self._mean_sequences: Optional[torch.Tensor] = None
        self._current_sequences: Optional[torch.Tensor] = None
        self._step_index = 0
        self._num_envs = 0


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        # If a new seed is provided, use it; otherwise keep the previously
        # configured seed. If a fixed seed is configured, reseed the RNG so that
        # reset() restarts the same action sequences deterministically.
        if seed is not None:
            self.seed = int(seed)
        if self.seed is not None:
            self._rng = np.random.default_rng(int(self.seed))
        self._step_index = 0
        self._mean_sequences = None
        self._current_sequences = None
        self._num_envs = 0

    def compute_control(self, state, **kwargs) -> torch.Tensor:  # type: ignore[override]
        if hasattr(state, "n_envs"):
            num_envs = int(state.n_envs)
        else:
            num_envs = 1

        dt = kwargs.get("dt")
        if dt is not None:
            self.dt = float(dt)
        else:
            self.dt = self.default_dt

        if self._mean_sequences is None or num_envs != self._num_envs:
            self._initialize_sequences(num_envs)

        if self._step_index >= self._mean_sequences.shape[1]:
            self._initialize_sequences(num_envs)

        actions = self._current_sequences[:, self._step_index, :]
        self._step_index += 1
        return actions

    # ------------------------------------------------------------------
    # Sequence generation helpers
    # ------------------------------------------------------------------
    def _initialize_sequences(self, num_envs: int) -> None:
        mean_sequences = np.zeros((num_envs, self.horizon_steps, self.action_dim), dtype=np.float32)
        noisy_sequences = np.zeros_like(mean_sequences)
        for dim in range(self.action_dim):
            generator = ParametricSignalGenerator(
                num_steps=self.horizon_steps,
                dt=self.dt,
                rng=self._rng,
                config=self._channel_configs[dim],
                methods=self.curve_methods_per_action[dim],
                max_offset_fraction=self._offset_fraction,
                bezier_full_range=self.bezier_full_range,
            )
            mean_signal = generator.sample_batch(num_envs)
            noisy_signal = generator.add_gaussian_noise_batch(mean_signal, clip=self.clip_actions)
            if self.clip_actions:
                mean_signal = np.clip(
                    mean_signal,
                    self._channel_configs[dim].min_val,
                    self._channel_configs[dim].max_val,
                )
            mean_sequences[:, :, dim] = mean_signal.astype(np.float32, copy=False)
            noisy_sequences[:, :, dim] = noisy_signal.astype(np.float32, copy=False)

        self._mean_sequences = torch.tensor(mean_sequences, device=self.device)
        self._current_sequences = torch.tensor(noisy_sequences, device=self.device)
        self._num_envs = num_envs
        self._step_index = 0

    @staticmethod
    def _seed_from_system(seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        pid = os.getpid()
        timestamp = int(time.time_ns() & 0xFFFFFFFF)
        return (pid ^ timestamp) & 0xFFFFFFFF

