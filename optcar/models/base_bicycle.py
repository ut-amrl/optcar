"""Base class for bicycle models."""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple
import torch


class BaseBicycleModel(ABC):
    """Abstract base class for bicycle models.
    
    All bicycle models share the same parameters, though some models
    (like kinematic) may not use all of them.
    
    Parameters:
        wheelbase: Distance between front and rear axles (meters)
        max_steering_angle: Maximum steering angle (radians)
        len_f: Distance from center of gravity to front axle (meters)
        len_r: Distance from center of gravity to rear axle (meters)
        mass: Vehicle mass (kg)
        inertia: Vehicle moment of inertia (kg*m^2)
        C_throttle_a: Throttle force coefficient - constant term (N)
        C_rollf: Rolling resistance coefficient (N)
        C_drag: Drag coefficient (N*s^2/m^2)
        tire_f: Front tire cornering stiffness (N/rad)
        tire_r: Rear tire cornering stiffness (N/rad)
        device: Torch device ('cuda' or 'cpu')
        dtype: Torch data type
    """
    pose_dim: int = 7  # [pos(3), quat(4)]
    twist_dim: int = 6  # [vel(3), angvel(3)]
    
    def __init__(
        self,
        len_f: float = 0.101,
        len_r: float = 0.101,
        max_steering_angle: float = 0.55,
        mass: float = 1.0,
        inertia: float = (0.34**2 + 0.16**2) * 1.0 / 12.0,  # 0.0147
        C_throttle_a: float = 5.5,
        C_rollf: float = 0.0,
        C_drag: float = 0.0,
        C_tire_f: float = 20.0,
        C_tire_r: float = 20.0,
        # tire_f: float = 55.0,
        # tire_r: float = 50.0,
        wheel_time_constant: float = 0.05,  # matches reference
        min_wheel_velocity: float = -6.0,
        max_wheel_velocity: float = 6.0,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        rollout_mode: str = 'absolute',
        include_twist: bool = False,
        throttle_deadband: Sequence[float] = (0.0, 0.0),
        steer_deadband: Sequence[float] = (0.0, 0.0),
        throttle_multiplier: float = 1.0,
        throttle_bias: float = 0.0,
        steer_pos_multiplier: float = 1.0,
        steer_pos_bias: float = 0.0,
        steer_neg_multiplier: float = 1.0,
        steer_neg_bias: float = 0.0,
    ):
        """Initialize bicycle model with physical parameters.
        
        Note: Kinematic models won't use all parameters (mass, inertia, tire stiffness, etc.)
        but we keep them for API consistency.
        
        Args:
            len_f: Distance from center of gravity to front axle (meters)
            len_r: Distance from center of gravity to rear axle (meters)
            max_steering_angle: Maximum steering angle (radians)
            rollout_mode: 'absolute' or 'delta' - whether to return absolute states or delta changes
            ... (other parameters)
        """
        # Geometric parameters - fundamental are len_f and len_r
        self.len_f = len_f
        self.len_r = len_r
        self.wheelbase = len_f + len_r  # Calculated from len_f and len_r
        self.max_steering_angle = max_steering_angle
        self.rollout_mode = rollout_mode
        self.include_twist = include_twist
        self.output_dim = self.pose_dim + (self.twist_dim if self.include_twist else 0)
        
        # Inertial parameters
        self.mass = mass
        self.inertia = inertia
        
        # Force/torque parameters
        self.C_throttle_a = C_throttle_a
        self.C_rollf = C_rollf
        self.C_drag = C_drag
        self.C_tire_f = C_tire_f
        self.C_tire_r = C_tire_r
        
        # Tire parameters
        # self.tire_f = tire_f
        # self.tire_r = tire_r
        self.wheel_time_constant = wheel_time_constant
        self.min_wheel_velocity = min_wheel_velocity
        self.max_wheel_velocity = max_wheel_velocity
        
        # Computation parameters
        self.device = device
        self.dtype = dtype
        self.throttle_deadband = self._make_deadband_tuple(throttle_deadband)
        self.steer_deadband = self._make_deadband_tuple(steer_deadband)
        self.throttle_multiplier = float(throttle_multiplier)
        self.throttle_bias = float(throttle_bias)
        self.steer_pos_multiplier = float(steer_pos_multiplier)
        self.steer_pos_bias = float(steer_pos_bias)
        self.steer_neg_multiplier = float(steer_neg_multiplier)
        self.steer_neg_bias = float(steer_neg_bias)
    
    @abstractmethod
    def rollout(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        actions: torch.Tensor,
        dt: float,
        current_vel_body: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rollout trajectory from initial conditions.
        
        Args:
            current_pos: Initial position (B, 3) [x, y, z]
            current_quat: Initial quaternion (B, 4) [w, x, y, z]
            actions: Control actions (B, H, 2) where H is horizon
            dt: Time step (seconds)
            current_vel_body: Initial velocity in body frame (B, 3) [vx, vy, vz] - optional
            
        Returns:
            Trajectory tensor of shape (B, H, D) where D depends on the model
        """
        raise NotImplementedError
    
    def __call__(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        actions: torch.Tensor,
        dt: float,
        current_vel_body: Optional[torch.Tensor] = None,
        current_wheel_speed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call interface for rollout - allows using model as a callable.
        
        Args:
            current_pos: Initial position (B, 3) [x, y, z]
            current_quat: Initial quaternion (B, 4) [w, x, y, z]
            actions: Control actions (B, H, 2) where H is horizon
            dt: Time step (seconds)
            current_vel_body: Initial velocity in body frame (B, 3) [vx, vy, vz] - optional
            current_wheel_speed: Initial wheel speed state (B,) - optional
            
        Returns:
            Trajectory tensor
        """
        return self.rollout(
            current_pos=current_pos,
            current_quat=current_quat,
            actions=actions,
            dt=dt,
            current_vel_body=current_vel_body,
            current_wheel_speed=current_wheel_speed,
        )
    
    def to(self, device: str):
        """Move model to specified device.
        
        Args:
            device: Device string ('cuda' or 'cpu')
        """
        self.device = device
        return self

    def _select_components(self, traj: torch.Tensor) -> torch.Tensor:
        """Select pose (and optional twist) from rollout tensor."""
        required_dim = self.pose_dim + self.twist_dim
        if traj.size(-1) < required_dim:
            raise ValueError(f"Trajectory tensor must have at least {required_dim} channels.")
        pose = traj[..., :self.pose_dim]
        if not self.include_twist:
            return pose
        return traj[..., :self.pose_dim + self.twist_dim]

    def _make_deadband_tuple(self, values: Sequence[float]) -> Tuple[float, float]:
        if not values or len(values) < 2:
            return (0.0, 0.0)
        low = float(values[0])
        high = float(values[1])
        if high < low:
            low, high = high, low
        return (low, high)

    def _apply_action_deadbands(self, actions: torch.Tensor) -> torch.Tensor:
        if actions is None or actions.size(-1) < 2:
            return actions

        thr_low, thr_high = self.throttle_deadband
        steer_low, steer_high = self.steer_deadband
        throttle_active = thr_high > thr_low
        steer_active = steer_high > steer_low
        if not (throttle_active or steer_active):
            return actions

        adjusted = actions.clone()
        if throttle_active:
            mask = (adjusted[..., 0] >= thr_low) & (adjusted[..., 0] <= thr_high)
            adjusted[..., 0] = torch.where(mask, torch.zeros_like(adjusted[..., 0]), adjusted[..., 0])
        if steer_active:
            mask = (adjusted[..., 1] >= steer_low) & (adjusted[..., 1] <= steer_high)
            adjusted[..., 1] = torch.where(mask, torch.zeros_like(adjusted[..., 1]), adjusted[..., 1])
        return adjusted

    def _apply_action_adjustments(self, actions: torch.Tensor) -> torch.Tensor:
        if actions is None or actions.size(-1) < 2:
            return actions

        throttle_active = abs(self.throttle_multiplier - 1.0) > 1e-6 or abs(self.throttle_bias) > 1e-6
        steer_pos_active = abs(self.steer_pos_multiplier - 1.0) > 1e-6 or abs(self.steer_pos_bias) > 1e-6
        steer_neg_active = abs(self.steer_neg_multiplier - 1.0) > 1e-6 or abs(self.steer_neg_bias) > 1e-6
        if not (throttle_active or steer_pos_active or steer_neg_active):
            return actions

        adjusted = actions.clone()
        if throttle_active:
            adjusted[..., 0] = (adjusted[..., 0] + self.throttle_bias) * self.throttle_multiplier

        if steer_pos_active or steer_neg_active:
            original_steer = adjusted[..., 1]
            if steer_pos_active:
                pos_mask = original_steer >= 0
                if pos_mask.any():
                    pos_adjusted = (original_steer + self.steer_pos_bias) * self.steer_pos_multiplier
                    adjusted[..., 1] = torch.where(pos_mask, pos_adjusted, adjusted[..., 1])
            if steer_neg_active:
                neg_mask = original_steer < 0
                if neg_mask.any():
                    neg_adjusted = (original_steer + self.steer_neg_bias) * self.steer_neg_multiplier
                    adjusted[..., 1] = torch.where(neg_mask, neg_adjusted, adjusted[..., 1])
        return adjusted

    def compute_action_bias(self, actions: torch.Tensor) -> torch.Tensor:
        """Return additive action bias for the provided actions."""
        if actions is None or actions.size(-1) < 2:
            return torch.zeros_like(actions) if actions is not None else torch.zeros(2, dtype=torch.float32)

        bias = torch.zeros_like(actions)
        if actions.dim() == 0:
            return bias

        throttle_bias = torch.as_tensor(self.throttle_bias, dtype=actions.dtype, device=actions.device)
        bias[..., 0] = throttle_bias

        steer_pos_bias = torch.as_tensor(self.steer_pos_bias, dtype=actions.dtype, device=actions.device)
        steer_neg_bias = torch.as_tensor(self.steer_neg_bias, dtype=actions.dtype, device=actions.device)
        bias[..., 1] = torch.where(actions[..., 1] >= 0, steer_pos_bias, steer_neg_bias)
        return bias
    
    def get_parameters(self) -> dict:
        """Get all model parameters as a dictionary.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'wheelbase': self.wheelbase,
            'max_steering_angle': self.max_steering_angle,
            'len_f': self.len_f,
            'len_r': self.len_r,
            'mass': self.mass,
            'inertia': self.inertia,
            'C_throttle_a': self.C_throttle_a,
            'C_rollf': self.C_rollf,
            'C_drag': self.C_drag,
            'C_tire_f': self.C_tire_f,
            'C_tire_r': self.C_tire_r,
        }
    def get_last_wheel_speed(self) -> Optional[torch.Tensor]:
        return None