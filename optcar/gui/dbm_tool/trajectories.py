"""Trajectory classes for MPPI-based controllers."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Type
import torch
import numpy as np
from abc import ABC, abstractmethod
import time
from .dubins_utils import Dubins_find_best_radii, resample_by_speed
from optcar.utils.math import yaw_from_quat_torch
from optcar.utils.math import quat_from_yaw_torch, wrap_to_pi, yaw_from_quat_torch, wrap_to_pi_math

class BaseTrajectory(ABC):
    """Base class for trajectory following with common functionality"""
    
    def __init__(self,
                 reference_speed: float = 3.0,
                 dt = 0.02,
                 current_state: Optional[object] = None,
                 device="cpu",
                 allow_far_detach: bool = False, # if car is too far from traj, then terminate
                 far_detach_threshold: float = 5.0,):
        # These will be set by derived classes
        self.reference_trajectory = None  # (N, 3) - x, y, orientation
        self.reference_arclen = None      # (N,) - cumulative arc length
        self.max_arclen = None           # scalar - total arc length
        self.device = device
        self.current_state: Optional[object] = current_state
        # Track last closest indices to prevent sudden jumps
        self._last_closest_idx = None  # (B, H) - last found indices
        
        # Default reference speed - can be overridden
        self.reference_speed = reference_speed

        self.dt = dt
        self.allow_far_detach = allow_far_detach
        self.far_detach_threshold = far_detach_threshold

    def set_current_state(self, state: Optional[object]) -> None:
        """Record the latest robot state metadata for derived classes."""
        self.current_state = state

    def has_goal(self) -> bool:
        """Whether this trajectory tracks a terminal goal."""
        return False

    def is_near_goal(self, curr_state) -> bool:
        """Check if the robot is near the terminal goal."""
        return False

    def get_capture_label(self) -> str:
        return self.__class__.__name__.lower()
    
    @abstractmethod
    def _generate_reference_trajectory(self):
        """Generate the reference trajectory points. Must be implemented by derived classes.
        
        Should set:
        - self.reference_trajectory: (N, 3) tensor of [x, y, orientation]
        - self.reference_arclen: (N,) tensor of cumulative arc lengths
        - self.max_arclen: scalar total arc length
        """
        return NotImplementedError
    
    def _compute_arc_length(self, points):
        """Compute cumulative arc length from trajectory points
        
        Args:
            points: (N, 2) tensor of x,y coordinates
            
        Returns:
            arc_lengths: (N,) tensor of cumulative arc lengths
        """
        deltas = points[1:] - points[:-1]
        seg_lens = torch.norm(deltas, dim=1)
        arc_lengths = torch.cat([
            torch.zeros(1, device=points.device), 
            torch.cumsum(seg_lens, dim=0)
        ])
        return arc_lengths

    def _get_trajectory_goals(
        self, x0s: torch.Tensor, speed: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Get sequence of trajectory goals starting from current positions
        
        Args:
            x0s: (batch_size, 1, 6) initial states
            speed: unused (kept for compatibility)
            n: number of future goals to generate
            
        Returns:
            goals: (batch_size, n, 3) sequence of trajectory goals
        """
        s0s = self._get_s(x0s)
        delta_s = self.reference_speed * self.dt  # use self.dt instead of hardcoded value
        goals_s = s0s + delta_s * torch.arange(n, device=s0s.device, dtype=s0s.dtype)
        goals_s = torch.fmod(goals_s, self.max_arclen)
        goal_idxs = self._get_ref_idx(goals_s)
        return self.reference_trajectory[goal_idxs]

    def _get_closest_reference_points(self, xs: torch.Tensor) -> torch.Tensor:
        """Get closest reference points for given states
        
        Args:
            xs: (batch_size, horizon_len, 6) states
            
        Returns:
            ref_points: (batch_size, horizon_len, 3) closest reference points
        """
        closest_ref_idx = self._get_closest_reference_idx(xs)
        return self.reference_trajectory[closest_ref_idx]

    def _get_closest_reference_idx(self, xs: torch.Tensor) -> torch.Tensor:
        """Get indices of closest reference points with cyclic-aware inverse decay bias
        
        Args:
            xs: (batch_size, horizon_len, 6) states
            
        Returns:
            indices: (batch_size, horizon_len) indices into reference trajectory
        """
        pos = xs[:, :, :3]
        pos[..., 2] = wrap_to_pi_math(pos[..., 2])
        B, H, _ = pos.shape
        num_ref_points = self.reference_trajectory.shape[0]

        param = 0.3
        diffs = pos.unsqueeze(2) - self.reference_trajectory.unsqueeze(0).unsqueeze(0)
        diffs[..., 2] = param * wrap_to_pi_math(diffs[..., 2])
        dists = torch.norm(diffs, dim=-1)  # (B, H, num_ref_points)

        # If no previous indices, just use pure distance
        if self._last_closest_idx is None or self._last_closest_idx.shape != (B, H):
            closest_ref_idx = torch.argmin(dists, dim=-1)  # (B, H)
            self._last_closest_idx = closest_ref_idx.clone()
            return closest_ref_idx

        # Add cyclic-aware penalty based on distance from last index
        all_indices = torch.arange(num_ref_points, device=pos.device)  # (num_ref_points,)
        last_indices = self._last_closest_idx.unsqueeze(2)  # (B, H, 1)
        
        # Compute cyclic distance between indices
        linear_distances = torch.abs(all_indices.unsqueeze(0).unsqueeze(0) - last_indices).float()  # (B, H, num_ref_points)
        wrap_distances = num_ref_points - linear_distances  # Distance going the other way around
        index_distances = torch.minimum(linear_distances, wrap_distances)  # Use minimum cyclic distance
        decay_penalty = 1.01 ** index_distances * 0.1
        
        # Apply penalty (bias toward indices near the last matched point)
        biased_dists = dists + decay_penalty
        
        closest_ref_idx = torch.argmin(biased_dists, dim=-1)  # (B, H)
        
        # Update last indices
        self._last_closest_idx = closest_ref_idx.clone()
        return closest_ref_idx

    def _get_ref_idx(self, s: torch.Tensor) -> torch.Tensor:
        """Get reference trajectory indices from arc length coordinates
        
        Args:
            s: (batch_size, horizon_len) arc length coordinates
            
        Returns:
            indices: (batch_size, horizon_len) indices into reference trajectory
        """
        s = torch.clamp(s, torch.zeros(1, device=s.device), self.max_arclen)
        diffs = torch.abs(
            s.unsqueeze(-1) - self.reference_arclen.unsqueeze(0).unsqueeze(0)
        )
        min_idx = torch.argmin(diffs, dim=-1)
        return min_idx

    def is_done(self, curr_state) -> bool:
        if (
            self.allow_far_detach
            and curr_state is not None
            and self.reference_trajectory is not None
        ):
            pos_tensor = curr_state.pos
            if pos_tensor.ndim > 1:
                pos_tensor = pos_tensor[0]
            pos_xy = pos_tensor[:2]
            ref_xy = self.reference_trajectory[:, :2]
            if hasattr(ref_xy, "device") and ref_xy.device != pos_xy.device:
                ref_xy = ref_xy.to(pos_xy.device)
            dists = torch.norm(pos_xy.unsqueeze(0) - ref_xy, dim=-1)
            min_dist = torch.min(dists).item()
            if min_dist > self.far_detach_threshold:
                return True
        return False

    def get_capture_label(self) -> str:
        return f"openloop_v={self.reference_speed:.2f}"

    def _get_s(self, xs: torch.Tensor) -> torch.Tensor:
        """Get arc length coordinates for given states
        
        Args:
            xs: (batch_size, horizon_len, 3) positions [x, y, orientation]
            
        Returns:
            s: (batch_size, horizon_len) arc length coordinates
        """
        closest_idx = self._get_closest_reference_idx(xs)
        return self.reference_arclen[closest_idx]
    
    def get_traj_goal(self, positions: torch.Tensor) -> torch.Tensor:
        """Get trajectory goals for given 2D positions.
        
        This is a simplified interface used by TrajTracking task.
        For each position, returns the closest point on the reference trajectory.
        
        Args:
            positions: (N, 2) tensor of [x, y] positions
            
        Returns:
            goals: (N, 3) tensor of [x, y, yaw] trajectory goals
        """
        N = positions.shape[0]
        
        # Compute distances to all reference points
        # positions: (N, 2), reference_trajectory[:, :2]: (M, 2)
        diffs = positions.unsqueeze(1) - self.reference_trajectory[:, :2].unsqueeze(0)  # (N, M, 2)
        dists = torch.norm(diffs, dim=-1)  # (N, M)
        
        # Find closest reference point for each position
        closest_idx = torch.argmin(dists, dim=-1)  # (N,)
        
        # Return the reference trajectory points at those indices
        return self.reference_trajectory[closest_idx]  # (N, 3)


class OpenLoopTrajectory(BaseTrajectory):
    """
    Open-loop reference trajectory compatible with BaseTrajectory interface.
    """

    def __init__(
        self,
        path: np.ndarray | torch.Tensor,  # shape (N,2) or (N,3)
        has_yaw: bool = False,
        v_ref: float = 1.5,
        dt_plan: float = 0.02,
        z: float = 0.0,
        current_state: Optional[Tuple[float, float, float]] = None,
        goal_threshold: float = 0.25,
        goal_yaw_threshold: Optional[float] = None,
        near_goal_threshold: Optional[float] = None,
        allow_far_detach: bool = False,
        far_detach_threshold: float = 5.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(reference_speed=v_ref, 
                         dt=dt_plan, 
                         current_state=current_state, 
                         device=device,
                         allow_far_detach=allow_far_detach,
                         far_detach_threshold=far_detach_threshold)

        device = torch.device(device)
        path = np.asarray(path, float)
        if path.ndim != 2 or path.shape[1] not in [2, 3] or len(path) < 2:
            raise ValueError("path must be an (N,2) or (N,3) array with N ≥ 2")

        # Prepare full pose: (x, y, yaw)
        if has_yaw and path.shape[1] == 3:
            path_xyyaw = path
        else:
            dx_dy = np.diff(path[:, :2], axis=0)
            yaw = np.arctan2(dx_dy[:, 1], dx_dy[:, 0])
            yaw = np.append(yaw, yaw[-1])
            path_xyyaw = np.hstack((path[:, :2], yaw[:, None]))

        # Resample path with yaw included
        path_xyyaw = resample_by_speed(path_xyyaw, speed=v_ref, dt=dt_plan)

        # Convert to torch
        xy = torch.as_tensor(path_xyyaw[:, :2], device=device, dtype=dtype)
        yaw = torch.as_tensor(path_xyyaw[:, 2], device=device, dtype=dtype)
        
        # Store original data for compatibility
        z_vec = torch.full((xy.shape[0], 1), z, device=device, dtype=dtype)
        self._pos = torch.cat((xy, z_vec), dim=-1)
        self._yaw = yaw
        self._quat = quat_from_yaw_torch(yaw.unsqueeze(-1))
        self.v_ref = v_ref
        self.goal_threshold = goal_threshold
        self.goal_yaw_threshold = goal_yaw_threshold
        self.near_goal_threshold = (
            near_goal_threshold if near_goal_threshold is not None else goal_threshold
        )
        self.goal_pose: Optional[torch.Tensor] = None
        
        # Set reference speed for BaseTrajectory
        self.reference_speed = v_ref

        # Generate BaseTrajectory-compatible attributes
        self._generate_reference_trajectory()
        self.goal_pose = self.reference_trajectory[-1].clone()
    
    def _generate_reference_trajectory(self):
        """Generate reference trajectory compatible with BaseTrajectory interface"""
        # Set the reference trajectory as (N, 3) with [x, y, orientation]
        self.reference_trajectory = torch.stack((
            self._pos[:, 0],  # x
            self._pos[:, 1],  # y
            self._yaw         # orientation
        ), dim=1).to(self.device)
        
        # Compute arc length using BaseTrajectory method
        self.reference_arclen = self._compute_arc_length(self._pos[:, :2])
        self.max_arclen = self.reference_arclen[-1]

    # def _get_trajectory_goals(
    #     self, x0s: torch.Tensor, speed: torch.Tensor, n: int
    # ) -> torch.Tensor:
    #     """Get sequence of trajectory goals starting from current positions
        
    #     Args:
    #         x0s: (batch_size, 1, 6) initial states
    #         speed: unused (kept for compatibility)
    #         n: number of future goals to generate
            
    #     Returns:
    #         goals: (batch_size, n, 3) sequence of trajectory goals
    #     """
    #     s0s = self._get_s(x0s)
    #     delta_s = self.reference_speed * 0.02  # hardcoded dt
    #     goals_s = s0s + delta_s * torch.arange(n, device=s0s.device, dtype=s0s.dtype)
        
    #     # For open-loop trajectory, clamp to max_arclen instead of wrapping
    #     goals_s = torch.clamp(goals_s, 0.0, self.max_arclen)
        
    #     goal_idxs = self._get_ref_idx(goals_s)
    #     return self.reference_trajectory[goal_idxs]
    def _get_trajectory_goals(
        self, x0s: torch.Tensor, speed: torch.Tensor, n: int
    ) -> torch.Tensor:
        s0s = self._get_s(x0s)
        delta_s = self.reference_speed * self.dt  # Changed from 0.02 to self.dt
        goals_s = s0s + delta_s * torch.arange(n, device=s0s.device, dtype=s0s.dtype)
        
        # For open-loop trajectory, clamp to max_arclen instead of wrapping
        goals_s = torch.clamp(goals_s, 0.0, self.max_arclen)
        
        goal_idxs = self._get_ref_idx(goals_s)
        return self.reference_trajectory[goal_idxs]

    def _get_ref_idx(self, s: torch.Tensor) -> torch.Tensor:
        """Get reference trajectory indices from arc length coordinates
        
        Args:
            s: (batch_size, horizon_len) arc length coordinates
            
        Returns:
            indices: (batch_size, horizon_len) indices into reference trajectory
        """
        # For open-loop trajectory, clamp to valid range instead of wrapping
        s = torch.clamp(s, 0.0, self.max_arclen)
        diffs = torch.abs(
            s.unsqueeze(-1) - self.reference_arclen.unsqueeze(0).unsqueeze(0)
        )
        min_idx = torch.argmin(diffs, dim=-1)
        return min_idx

    def has_goal(self) -> bool:
        return True

    def is_near_goal(self, curr_state) -> bool:
        if curr_state is None or self.goal_pose is None:
            return False

        pos_tensor = curr_state.pos
        if pos_tensor.ndim > 1:
            pos_tensor = pos_tensor[0]
        goal_xy = self.goal_pose[:2].to(pos_tensor.device)
        dist = torch.norm(pos_tensor[:2] - goal_xy)
        return bool(dist.item() <= self.near_goal_threshold)

    def is_done(self, curr_state) -> bool:
        if super().is_done(curr_state):
            return True

        if curr_state is None or self.goal_pose is None:
            return False

        pos_tensor = curr_state.pos
        quat_tensor = curr_state.quat
        if pos_tensor.ndim > 1:
            pos_tensor = pos_tensor[0]
        if quat_tensor.ndim > 1:
            quat_tensor = quat_tensor[0]

        goal_xy = self.goal_pose[:2].to(pos_tensor.device)
        dist = torch.norm(pos_tensor[:2] - goal_xy)
        if dist.item() > self.goal_threshold:
            return False

        if self.goal_yaw_threshold is None:
            return True

        yaw = yaw_from_quat_torch(quat_tensor.unsqueeze(0))[0, 0]
        goal_yaw = self.goal_pose[2]
        yaw_err = torch.abs(wrap_to_pi_math(yaw - goal_yaw))
        return bool(yaw_err.item() <= self.goal_yaw_threshold)


class DubinsTrajectory(OpenLoopTrajectory):
    """Open-loop trajectory generated from a Dubins-style recovery path."""

    @staticmethod
    def _state_to_pose(state: object) -> Tuple[float, float, float]:
        if state is None:
            raise ValueError("State is required to infer start pose")
        
        pos_tensor = state.pos
        quat_tensor = state.quat
        if pos_tensor.ndim > 1:
            pos_tensor = pos_tensor[0]
        if quat_tensor.ndim > 1:
            quat_tensor = quat_tensor[0]

        yaw_tensor = yaw_from_quat_torch(quat_tensor.unsqueeze(0))
        yaw_val = float(yaw_tensor.view(-1)[0].item())
        return float(pos_tensor[0].item()), float(pos_tensor[1].item()), yaw_val

    def __init__(
        self,
        start_pose: Optional[Sequence[float]] = None,
        goal_pose: Sequence[float] = (0.0, 0.0, 0.0),
        cand_r_start: Sequence[float] | np.ndarray = (1.0, 2.0, 3.0),
        cand_r_end: Sequence[float] | np.ndarray = (1.0, 2.0, 3.0),
        reference_speed: float = 1.5,
        dt_plan: float = 0.02,
        current_state: Optional[object] = None,
        goal_dist_thresh: float = 1.0,
        goal_yaw_thresh_deg: float = 15.0,
        near_goal_dist_thresh: Optional[float] = None,
        allow_far_detach: bool = False,
        far_detach_threshold: float = 5.0,
        allow_reverse: bool = True,
        n_jobs: int = -1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:

        if start_pose is None:
            if current_state is None:
                raise ValueError("DubinsTrajectory requires start_pose or current_state")
            start_pose = self._state_to_pose(current_state)
        else:
            start_pose = tuple(start_pose)
        goal_pose = tuple(goal_pose)
        if len(start_pose) != 3 or len(goal_pose) != 3:
            raise ValueError("start_pose and goal_pose must be (x, y, yaw)")

        result = Dubins_find_best_radii(
            start_pose,
            goal_pose,
            cand_r_start,
            cand_r_end,
            dist_thresh=goal_dist_thresh,
            angle_thresh_deg=goal_yaw_thresh_deg,
            n_jobs=n_jobs,
            return_yaw=True,
            reverse=allow_reverse,
        )
        resampled_path = resample_by_speed(result["path"], speed=reference_speed, dt=dt_plan)
        super().__init__(
            path=resampled_path,
            has_yaw=True,
            v_ref=reference_speed,
            dt_plan=dt_plan,
            current_state=current_state,
            goal_threshold=goal_dist_thresh,
            goal_yaw_threshold=goal_yaw_thresh_deg * torch.pi / 180.0,
            near_goal_threshold=near_goal_dist_thresh,
            allow_far_detach=allow_far_detach,
            far_detach_threshold=far_detach_threshold,
            device=device,
            dtype=dtype,
        )
        self.dubins_solution = result
        self.goal_pose = torch.tensor(goal_pose, dtype=torch.float32)
        self.goal_threshold = goal_dist_thresh
        self.goal_yaw_threshold = goal_yaw_thresh_deg * torch.pi / 180.0

    def is_done(self, curr_state) -> bool:
        if super().is_done(curr_state):
            return True

        if curr_state is None:
            return False

        pos = curr_state.pos[0, :2].detach().cpu()
        quat = curr_state.quat[0].detach().cpu()

        dist = torch.norm(pos - self.goal_pose[:2])
        yaw = yaw_from_quat_torch(quat.unsqueeze(0))[0, 0]

        yaw_err = torch.abs(wrap_to_pi_math(yaw - self.goal_pose[2]))

        return bool(dist.item() <= self.goal_threshold and yaw_err.item() <= self.goal_yaw_threshold)

    def get_capture_label(self) -> str:
        goal_x, goal_y = self.goal_pose[0].item(), self.goal_pose[1].item()
        return f"dubins_v={self.reference_speed:.2f}_x={goal_x:.2f}_y={goal_y:.2f}"


class Figure8Trajectory(BaseTrajectory):
    """Figure-8 trajectory implementation with direction control"""
    
    def __init__(self, 
                 amp=2, 
                 wide_factor=1.5,
                 v_ref=1.5,
                 z=0.0,
                 dt = 0.02,
                 center_xy=np.array([0.0, 0.0]),
                 ccw=True,
                 num_points=1000,
                 device="cpu",
                 current_state: Optional[object] = None,
                 allow_far_detach: bool = False,
                 far_detach_threshold: float = 5.0,
                 ):
        super().__init__(reference_speed=v_ref, dt=dt, current_state=current_state, device=device,
                         allow_far_detach=allow_far_detach, far_detach_threshold=far_detach_threshold)
        
        self.amp = amp
        self.wide_factor = wide_factor
        self.num_points = num_points
        self.clockwise = ccw
        self.center_xy = center_xy
        
        self._generate_reference_trajectory()
    def is_done(self, curr_state) -> bool:
        return super().is_done(curr_state)
    
    def _generate_reference_trajectory(self):
        """Generate figure-8 reference trajectory"""
        # Ensure center_xy is a tensor on correct device
        if isinstance(self.center_xy, np.ndarray):
            center_xy_t = torch.tensor(self.center_xy, device=self.device, dtype=torch.float32)
        else:
            center_xy_t = torch.as_tensor(self.center_xy, device=self.device, dtype=torch.float32)

        # Re-compute x, y with device-aware tensors
        theta = torch.linspace(-torch.pi, torch.pi, steps=self.num_points, device=self.device)
        
        if not self.clockwise:
            theta = torch.flip(theta, [0])
        
        x = self.amp * torch.cos(theta) / (1 + torch.sin(theta) ** 2)
        y = (self.wide_factor * self.amp * torch.sin(theta) * 
             torch.cos(theta) / (1 + torch.sin(theta) ** 2))
        
        x += center_xy_t[0]
        y += center_xy_t[1]
        
        dx_dtheta = (
            -self.amp * torch.sin(theta) * 
            (self.amp - torch.sin(theta) ** 2) / 
            (1 + torch.sin(theta) ** 2) ** 2
        )
        dy_dtheta = (
            self.wide_factor * self.amp *
            (torch.cos(theta) ** 2 - 2 * torch.sin(theta) ** 2) /
            (1 + torch.sin(theta) ** 2) ** 2
        )
        
        if not self.clockwise:
            dx_dtheta = -dx_dtheta
            dy_dtheta = -dy_dtheta
        
        orient = wrap_to_pi_math(torch.atan2(dy_dtheta, dx_dtheta))
        
        self.reference_trajectory = torch.stack((x, y, orient), dim=1).to(self.device)
        
        pts = self.reference_trajectory[:, :2]
        self.reference_arclen = self._compute_arc_length(pts).to(self.device)
        self.max_arclen = self.reference_arclen[-1].to(self.device)

    def get_capture_label(self) -> str:
        return f"fig8_v={self.reference_speed:.2f}_ccw={self.clockwise}_a={self.amp:.2f}_w={self.wide_factor:.2f}"


class CircleTrajectory(BaseTrajectory):
    """Circular trajectory with direction control"""
    
    def __init__(self, radius=2.0, 
                 num_points=200, 
                 center_xy=np.array([0.0, 0.0]),
                 v_ref=1.5,
                 ccw=True,
                 z=0.0,
                 dt = 0.02,
                 device="cpu",
                 current_state: Optional[object] = None,
                 allow_far_detach: bool = False,
                 far_detach_threshold: float = 5.0,
                 ):
        super().__init__(reference_speed=v_ref, dt=dt, current_state=current_state, device=device,
                         allow_far_detach=allow_far_detach, far_detach_threshold=far_detach_threshold)
        
        self.radius = radius
        self.num_points = num_points
        self.clockwise = ccw
        self.center_xy = center_xy
        self.reference_speed = v_ref
        
        self._generate_reference_trajectory()
    
    def get_capture_label(self) -> str:
        return f"circle_r={self.radius:.2f}_v={self.reference_speed:.2f}"
    def is_done(self, curr_state) -> bool:
        return super().is_done(curr_state)
    
    def _generate_reference_trajectory(self):
        """Generate circular reference trajectory"""
        if self.clockwise:
            theta = torch.linspace(0, -2 * torch.pi, steps=self.num_points)
        else:
            theta = torch.linspace(0, 2 * torch.pi, steps=self.num_points)
        
        x = self.radius * torch.cos(theta)
        y = self.radius * torch.sin(theta)
        x += self.center_xy[0]
        y += self.center_xy[1]
        
        if self.clockwise:
            orient = theta - torch.pi / 2
        else:
            orient = theta + torch.pi / 2
        
        orient = wrap_to_pi_math(orient)
        
        self.reference_trajectory = torch.stack((x, y, orient), dim=1).to(self.device)
        pts = self.reference_trajectory[:, :2]
        self.reference_arclen = self._compute_arc_length(pts)
        self.max_arclen = self.reference_arclen[-1]


class CustomClosedLoopTrajectory(BaseTrajectory):
    """Custom closed-loop trajectory from user-provided waypoints.
    
    Supports:
    - numpy array of shape (N, 2) with xy coordinates
    - txt file with line-by-line "x y" or "x,y" coordinates
    
    The trajectory is treated as a closed loop (cyclic).
    """
    
    def __init__(
        self,
        waypoints: np.ndarray | torch.Tensor | str | None = None,
        txt_file: str | None = None,
        v_ref: float = 1.5,
        dt_plan: float = 0.02,
        z: float = 0.0,
        current_state: Optional[object] = None,
        close_loop: bool = True,
        flip_x: bool = False,
        flip_y: bool = False,
        num_resample_points: int = 500,
        allow_far_detach: bool = False,
        far_detach_threshold: float = 5.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize custom closed-loop trajectory.
        
        Args:
            waypoints: (N, 2) array of [x, y] waypoints, or path to txt file
            txt_file: Path to txt file with "x y" or "x,y" coordinates per line
            v_ref: Reference speed for trajectory following
            dt_plan: Time step for planning
            z: Z coordinate (height)
            current_state: Current robot state
            close_loop: Whether to close the loop (connect last point to first)
            flip_x: If True, mirror trajectory across y-axis (x -> -x)
            flip_y: If True, mirror trajectory across x-axis (y -> -y)
            num_resample_points: Number of points to resample trajectory to
            allow_far_detach: If True, trajectory ends when robot is too far
            far_detach_threshold: Distance threshold for far_detach
            device: Torch device
            dtype: Torch dtype
        """
        super().__init__(
            reference_speed=v_ref,
            dt=dt_plan,
            current_state=current_state,
            device=device,
            allow_far_detach=allow_far_detach,
            far_detach_threshold=far_detach_threshold,
        )
        
        self.close_loop = close_loop
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.num_resample_points = num_resample_points
        self.v_ref = v_ref
        self.dtype = dtype
        self.z = z
        
        # Load waypoints
        if waypoints is not None:
            if isinstance(waypoints, str):
                # It's a file path
                xy = self._load_from_txt(waypoints)
            else:
                xy = np.asarray(waypoints, dtype=float)
        elif txt_file is not None:
            xy = self._load_from_txt(txt_file)
        else:
            raise ValueError("Must provide either waypoints or txt_file")
        
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"Waypoints must be (N, 2) array, got shape {xy.shape}")
        
        xy = xy[:, :2]  # Take only x, y columns

        if self.flip_x:
            xy = xy.copy()
            xy[:, 0] = -xy[:, 0]
        if self.flip_y:
            if "xy" not in locals():
                xy = xy.copy()
            xy[:, 1] = -xy[:, 1]
        
        if len(xy) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        # Close the loop if requested
        if close_loop and not np.allclose(xy[0], xy[-1], atol=1e-6):
            xy = np.vstack([xy, xy[0:1]])
        
        self._raw_waypoints = xy
        self._generate_reference_trajectory()
    
    def _load_from_txt(self, filepath: str) -> np.ndarray:
        """Load waypoints from txt file.
        
        Supports formats:
        - "x y" (space separated)
        - "x,y" (comma separated)
        - "x\ty" (tab separated)
        
        Lines starting with # are treated as comments.
        """
        waypoints = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try different separators
                if ',' in line:
                    parts = line.split(',')
                elif '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split()
                
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        waypoints.append([x, y])
                    except ValueError:
                        continue  # Skip malformed lines
        
        if len(waypoints) < 2:
            raise ValueError(f"Could not parse at least 2 waypoints from {filepath}")
        
        return np.array(waypoints)
    
    def _generate_reference_trajectory(self):
        """Generate reference trajectory from raw waypoints."""
        xy = self._raw_waypoints
        
        # Compute cumulative arc length of raw waypoints
        deltas = np.diff(xy, axis=0)
        seg_lens = np.linalg.norm(deltas, axis=1)
        raw_arclen = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = raw_arclen[-1]
        
        if total_len < 1e-6:
            raise ValueError("Trajectory has zero length")
        
        # Resample to uniform arc length spacing
        target_arclen = np.linspace(0, total_len, self.num_resample_points)
        x_resampled = np.interp(target_arclen, raw_arclen, xy[:, 0])
        y_resampled = np.interp(target_arclen, raw_arclen, xy[:, 1])
        
        # Compute yaw from trajectory direction
        dx = np.gradient(x_resampled)
        dy = np.gradient(y_resampled)
        yaw = np.arctan2(dy, dx)
        
        # Convert to torch
        xy_tensor = torch.tensor(
            np.stack([x_resampled, y_resampled], axis=1),
            device=self.device,
            dtype=self.dtype
        )
        yaw_tensor = torch.tensor(yaw, device=self.device, dtype=self.dtype)
        
        # Store position with z
        z_vec = torch.full((xy_tensor.shape[0], 1), self.z, device=self.device, dtype=self.dtype)
        self._pos = torch.cat((xy_tensor, z_vec), dim=-1)
        self._yaw = yaw_tensor
        
        # Set reference trajectory (x, y, yaw)
        self.reference_trajectory = torch.stack(
            (xy_tensor[:, 0], xy_tensor[:, 1], yaw_tensor), dim=1
        ).to(self.device)
        
        # Compute arc length
        self.reference_arclen = self._compute_arc_length(xy_tensor)
        self.max_arclen = self.reference_arclen[-1]
    
    def is_done(self, curr_state) -> bool:
        """Closed-loop trajectories are never 'done' unless far_detach triggers."""
        return super().is_done(curr_state)
    
    def get_capture_label(self) -> str:
        return f"custom_closedloop_v={self.reference_speed:.2f}_n={len(self._raw_waypoints)}"
    
    @classmethod
    def from_txt(
        cls,
        filepath: str,
        v_ref: float = 1.5,
        dt_plan: float = 0.02,
        **kwargs
    ) -> "CustomClosedLoopTrajectory":
        """Factory method to create trajectory from txt file.
        
        Args:
            filepath: Path to txt file with xy coordinates
            v_ref: Reference speed
            dt_plan: Time step
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            CustomClosedLoopTrajectory instance
        """
        return cls(txt_file=filepath, v_ref=v_ref, dt_plan=dt_plan, **kwargs)
    
    @classmethod
    def from_array(
        cls,
        xy_array: np.ndarray | torch.Tensor,
        v_ref: float = 1.5,
        dt_plan: float = 0.02,
        **kwargs
    ) -> "CustomClosedLoopTrajectory":
        """Factory method to create trajectory from numpy array.
        
        Args:
            xy_array: (N, 2) array of [x, y] waypoints
            v_ref: Reference speed
            dt_plan: Time step
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            CustomClosedLoopTrajectory instance
        """
        return cls(waypoints=xy_array, v_ref=v_ref, dt_plan=dt_plan, **kwargs)
