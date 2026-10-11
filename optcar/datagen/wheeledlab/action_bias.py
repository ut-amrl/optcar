"""SimRunner action bias methods copied from WheeledLab (quiet, isolated RNG)."""
from types import SimpleNamespace
import numpy as np
import torch

class ActionBiasSampler:
    def __init__(self, ranges, num_envs, device, seed):
        self.cfg = SimpleNamespace(env_setup=SimpleNamespace(num_envs=num_envs, params=ranges))
        self.tensor_device = device
        self._rng = np.random.RandomState(seed)
        self._resample_action_bias()

    def _apply_action_bias(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply sampled action biases before normalizing for the environment."""
        if actions is None or self._velocity_bias is None or self._steering_bias is None:
            return actions
        bias = torch.stack([self._velocity_bias, self._steering_bias], dim=1)
        if actions.dim() == 1:
            return actions + bias[0]
        return actions + bias

    def _resample_action_bias(self) -> None:
        """Sample new velocity and steering biases from params specs."""
        num_envs = self.cfg.env_setup.num_envs
        params = getattr(self.cfg.env_setup, 'params', {}) or {}
        vel_spec = params.get('velocity_bias', None)
        steer_spec = params.get('steering_bias', None)
        if vel_spec is None and steer_spec is None:
            self._velocity_bias = torch.zeros(num_envs, device=self.tensor_device)
            self._steering_bias = torch.zeros(num_envs, device=self.tensor_device)
            return

        def to_pair(spec, default=0.0):
            if spec is None:
                return (float(default), float(default))
            if isinstance(spec, (float, int)):
                v = float(spec)
                return (v, v)
            seq = list(spec)
            if len(seq) == 0:
                return (float(default), float(default))
            if len(seq) == 1:
                v = float(seq[0])
                return (v, v)
            return (float(seq[0]), float(seq[1]))
        (v_low, v_high) = to_pair(vel_spec, default=0.0)
        (s_low, s_high) = to_pair(steer_spec, default=0.0)
        vel_biases_list = [float(self._rng.uniform(v_low, v_high)) if v_low != v_high else v_low for _ in range(num_envs)]
        steer_biases_list = [float(self._rng.uniform(s_low, s_high)) if s_low != s_high else s_low for _ in range(num_envs)]
        self._velocity_bias = torch.tensor(vel_biases_list, dtype=torch.float32, device=self.tensor_device)
        self._steering_bias = torch.tensor(steer_biases_list, dtype=torch.float32, device=self.tensor_device)
