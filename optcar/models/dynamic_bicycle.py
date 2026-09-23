"""PyTorch dynamic bicycle model, using WheeledLab's wheel-actuated RK4 equations."""
from dataclasses import asdict, dataclass
from typing import Tuple
import math

import torch

from optcar.utils.math import qinverse_torch, qrotate_torch, wrap_to_pi, yaw_from_quat_torch


@torch.jit.script
def _state_derivative(
    x: torch.Tensor,
    u: torch.Tensor,          # [desired_wheel_velocity (m/s), steering_angle (rad)]
    len_f: torch.Tensor,
    len_r: torch.Tensor,
    mass: torch.Tensor,
    inertia: torch.Tensor,
    max_steering_angle: torch.Tensor,
    C_tire_r: torch.Tensor,
    C_tire_f: torch.Tensor,
    C_throttle_a: torch.Tensor,
    C_rollf: torch.Tensor,
    C_drag: torch.Tensor,
    wheel_time_constant: torch.Tensor,
    min_wheel_velocity: torch.Tensor,
    max_wheel_velocity: torch.Tensor,
) -> torch.Tensor:
    yaw      = x[..., 2]
    v_long   = x[..., 3]
    v_lat    = x[..., 4]
    ang_vel  = x[..., 5]
    v_wheel  = x[..., 6]

    raw_d_w = u[..., 0]
    raw_steering = u[..., 1]
    d_w = torch.clamp(raw_d_w, min_wheel_velocity, max_wheel_velocity)

    steering_angle = torch.clamp(raw_steering, -max_steering_angle, max_steering_angle)

    slip_f = steering_angle - torch.atan2(v_lat + ang_vel * len_f, v_long)
    slip_f = torch.where(
        slip_f.abs() > torch.pi / 2,
        wrap_to_pi(torch.pi - slip_f),
        slip_f,
    )
    slip_r = -torch.atan2(v_lat - ang_vel * len_r, v_long.abs())

    d_pose = torch.stack([
        v_long * torch.cos(yaw) - v_lat * torch.sin(yaw),
        v_long * torch.sin(yaw) + v_lat * torch.cos(yaw),
        ang_vel
    ], dim=-1)

    F_front_drive = 0.5 * C_throttle_a * (v_wheel - v_long)
    F_rear_drive  = 0.5 * C_throttle_a * (v_wheel - v_long)
    F_front_tire =       C_tire_f * slip_f
    F_rear_tire =        C_tire_r * slip_r
    F_rolling_resistance = C_rollf * torch.sign(v_long)
    F_drag = C_drag * v_long * v_long.abs()

    F_long = (
        F_front_drive * torch.cos(steering_angle)
        + F_rear_drive
        - F_front_tire * torch.sin(steering_angle)
        - F_rolling_resistance
        - F_drag
    )
    F_lat = (
        F_front_drive * torch.sin(steering_angle)
        + F_front_tire * torch.cos(steering_angle)
        + F_rear_tire
    )

    torque = (
        F_front_drive * len_f * torch.sin(steering_angle)
        + F_front_tire * len_f * torch.cos(steering_angle)
        - F_rear_tire * len_r
    )


    d_vel = torch.stack(
        [
            F_long / mass + v_lat * ang_vel,
            F_lat  / mass - v_long * ang_vel,
            torque    / inertia,
            (d_w - v_wheel) / wheel_time_constant
    ], dim=-1)

    if d_pose.shape[0] != d_vel.shape[0]:
        if d_pose.shape[0] == 1:
            d_pose = d_pose.repeat(d_vel.shape[0], 1)
        else:
            raise RuntimeError("Batch size mismatch between d_pose and d_vel")
    return torch.cat((d_pose, d_vel), dim=-1)

@torch.jit.script
def _rk4_step(
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    len_f: torch.Tensor,
    len_r: torch.Tensor,
    mass: torch.Tensor,
    inertia: torch.Tensor,
    max_steering_angle: torch.Tensor,
    C_tire_f: torch.Tensor,
    C_tire_r: torch.Tensor,
    C_throttle_a: torch.Tensor,
    C_rollf: torch.Tensor,
    C_drag: torch.Tensor,
    wheel_time_constant: torch.Tensor,
    min_wheel_velocity: torch.Tensor,
    max_wheel_velocity: torch.Tensor,
) -> torch.Tensor:

    k1 = _state_derivative(x, u, len_f, len_r, mass, inertia, max_steering_angle,
                                  C_tire_r, C_tire_f, C_throttle_a, C_rollf, C_drag,
                                  wheel_time_constant, min_wheel_velocity, max_wheel_velocity)
    k2 = _state_derivative(x + 0.5 * dt * k1, u, len_f, len_r, mass, inertia, max_steering_angle,
                                  C_tire_r, C_tire_f, C_throttle_a, C_rollf, C_drag,
                                  wheel_time_constant, min_wheel_velocity, max_wheel_velocity)
    k3 = _state_derivative(x + 0.5 * dt * k2, u, len_f, len_r, mass, inertia, max_steering_angle,
                                  C_tire_r, C_tire_f, C_throttle_a, C_rollf, C_drag,
                                  wheel_time_constant, min_wheel_velocity, max_wheel_velocity)
    k4 = _state_derivative(x + dt * k3, u, len_f, len_r, mass, inertia, max_steering_angle,
                                  C_tire_r, C_tire_f, C_throttle_a, C_rollf, C_drag,
                                  wheel_time_constant, min_wheel_velocity, max_wheel_velocity)
    new_x_tmp = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    new_theta = wrap_to_pi(new_x_tmp[..., 2])
    new_x = torch.cat([
        new_x_tmp[..., 0:2],
        new_theta.unsqueeze(-1),
        new_x_tmp[..., 3:],
    ], dim=-1)
    return new_x


@torch.jit.script
def _rollout_torch(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    current_vel_body: torch.Tensor,      # [v_long, v_lat, r]
    current_wheel_speed: torch.Tensor,   # [B] or [B,1]
    actions: torch.Tensor,               # [B, H, 2] = [desired_wheel_velocity (m/s), steering_angle (rad)]
    dt: float,
    len_f: torch.Tensor,
    len_r: torch.Tensor,
    mass: torch.Tensor,
    inertia: torch.Tensor,
    max_steering_angle: torch.Tensor,
    C_tire_f: torch.Tensor,
    C_tire_r: torch.Tensor,
    C_throttle_a: torch.Tensor,
    C_rollf: torch.Tensor,
    C_drag: torch.Tensor,
    wheel_time_constant: torch.Tensor,
    min_wheel_velocity: torch.Tensor,
    max_wheel_velocity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = actions.shape[0]
    H = actions.shape[1]

    trajectory = torch.zeros((B, H, 13), dtype=actions.dtype, device=actions.device)

    yaw0 = yaw_from_quat_torch(current_quat)

    v_wheel0 = current_wheel_speed if current_wheel_speed.dim() == 2 else current_wheel_speed.unsqueeze(-1)

    x = torch.cat([
        current_pos[:, 0:2],
        yaw0,
        current_vel_body[:, 0:2],
        current_vel_body[:, 2:3],
        v_wheel0
    ], dim=-1)

    for h in range(H):
        u_frac = actions[:, h, :]

        x = _rk4_step(
            x, u_frac, dt, len_f, len_r, mass, inertia, max_steering_angle,
            C_tire_f, C_tire_r, C_throttle_a, C_rollf, C_drag, wheel_time_constant,
            min_wheel_velocity, max_wheel_velocity
        )

        px, py, theta, v_long, v_lat, r, _ = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], x[:, 4:5], x[:, 5:6], x[:, 6:7]

        pos_3d = torch.cat([px, py, torch.zeros_like(px)], dim=-1)
        quat = torch.cat([torch.cos(theta * 0.5),
                          torch.zeros_like(theta),
                          torch.zeros_like(theta),
                          torch.sin(theta * 0.5)], dim=-1)

        vx_world = v_long * torch.cos(theta) - v_lat * torch.sin(theta)
        vy_world = v_long * torch.sin(theta) + v_lat * torch.cos(theta)
        vel_3d = torch.cat([vx_world, vy_world, torch.zeros_like(vx_world)], dim=-1)
        ang_vel = torch.cat([torch.zeros((B, 2), dtype=actions.dtype, device=actions.device), r], dim=-1)

        trajectory[:, h, :] = torch.cat([pos_3d, quat, vel_3d, ang_vel], dim=-1)

    return trajectory, x[:, 6].clone()


@dataclass
class DBMParams:
    len_f: float = 0.101
    len_r: float = 0.101
    mass: float = 1.09
    inertia: float = 0.015
    max_steering_angle: float = 0.5
    C_tire_f: float = 20.0
    C_tire_r: float = 20.0
    C_throttle_a: float = 5.5
    C_rollf: float = 0.0
    C_drag: float = 0.0
    wheel_time_constant: float = 0.05
    min_wheel_velocity: float = -6.0
    max_wheel_velocity: float = 6.0

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(float(v)) for v in values.values()):
            raise ValueError("DBM parameters must be finite")
        positive = ("len_f", "len_r", "mass", "inertia", "wheel_time_constant", "max_steering_angle")
        if any(values[k] <= 0 for k in positive):
            raise ValueError(f"These parameters must be positive: {positive}")
        if any(values[k] < 0 for k in ("C_tire_f", "C_tire_r", "C_throttle_a", "C_rollf", "C_drag")):
            raise ValueError("Force coefficients must be nonnegative")
        if self.min_wheel_velocity >= self.max_wheel_velocity:
            raise ValueError("Wheel velocity limits are reversed")


class DynamicBicycleModel:
    def __init__(self, params=None, device="cpu"):
        self.params = params if isinstance(params, DBMParams) else DBMParams(**(params or {}))
        self.device = torch.device(device)
        self.tensors = {k: torch.tensor(v, dtype=torch.float32, device=self.device)
                        for k, v in asdict(self.params).items()}

    def clamp_actions(self, actions):
        return torch.stack((actions[..., 0].clamp(self.params.min_wheel_velocity, self.params.max_wheel_velocity),
                            actions[..., 1].clamp(-self.params.max_steering_angle, self.params.max_steering_angle)), -1)

    def rollout(self, state, actions, dt=0.02, wheel_speed=None):
        """Return future world states [B,T,13] and final actuator state [B]."""
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if state.ndim != 2 or state.shape[-1] != 13 or actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError("Expected state [B,13] and actions [B,T,2]")
        if state.shape[0] != actions.shape[0]:
            raise ValueError("State and command batch sizes differ")
        body_v = qrotate_torch(qinverse_torch(state[:, 3:7]), state[:, 7:10])
        body_w = qrotate_torch(qinverse_torch(state[:, 3:7]), state[:, 10:13])
        velocity = torch.stack((body_v[:, 0], body_v[:, 1], body_w[:, 2]), -1)
        wheel = body_v[:, 0].clone() if wheel_speed is None else torch.as_tensor(wheel_speed, device=self.device, dtype=state.dtype)
        trajectory, wheel = _rollout_torch(
            state[:, :3], state[:, 3:7], velocity, wheel, self.clamp_actions(actions), dt,
            **self.tensors)
        # The reference operates in XY. Retain the initial altitude in bag previews.
        trajectory[..., 2] = state[:, None, 2]
        return trajectory, wheel
