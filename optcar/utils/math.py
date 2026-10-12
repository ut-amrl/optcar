"""Math utilities for analytical models."""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import math

def wrap_to_pi_math(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


@torch.jit.script
def cross_product_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """3D cross product."""
    return torch.stack((
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    ), dim=-1)


@torch.jit.script
def qinverse_torch(q):
    """
    Quaternion inverse.
    q -> [w, x, y, z]
    """
    # Extract the scalar (w) and vector ([x, y, z]) parts
    scalar_part = q[..., 0:1]  # w
    vector_part = q[..., 1:]   # [x, y, z]
    q_inv = torch.cat([scalar_part, -vector_part], dim=-1)
    
    return q_inv


@torch.jit.script
def qrotate_torch(q, v):
    """Rotate vector v by quaternion q."""
    t = 2. * cross_product_3d(q[..., 1:], v)
    return v + q[..., 0:1] * t + cross_product_3d(q[..., 1:], t)


@torch.jit.script
def qmultiply_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiplication (JIT-scriptable).

    Args:
        q1: (..., 4) tensor [w, x, y, z]
        q2: (..., 4) tensor [w, x, y, z]

    Returns:
        (..., 4) tensor for q1 ⊗ q2
    """
    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w2 = q2[..., 0]
    x2 = q2[..., 1]
    y2 = q2[..., 2]
    z2 = q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


@torch.jit.script
def qrelative_torch(q1, q2):
    """
    q2 in q1 frame
    """
    return qmultiply_torch(qinverse_torch(q1), q2)


@torch.jit.script
def yaw_from_quat_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Compute yaw from quaternion for tensors with arbitrary leading dims.

    Parameters
    ----------
    q : (*, 4) tensor
        Quaternion in (w, x, y, z) order.  * can be any shape.

    Returns
    -------
    yaw : (*, 1) tensor
        Yaw angle (rad) in the same dtype/device as input.
    """
    assert q.size(-1) == 4, "last dim must be 4 (w,x,y,z)"
    w, x, y, z = q.unbind(dim=-1)
    sin_y = 2.0 * (w * z + x * y)
    cos_y = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(sin_y, cos_y)[..., None]   # keep singleton dim


@torch.jit.script
def quat_from_yaw_torch(yaw):
    """
    Create quaternion from yaw angle.
    yaw: tensor, (N, 1)
    output: tensor, (N, 4)
    """
    half_yaw = yaw * 0.5
    cos_yaw = torch.cos(half_yaw)
    sin_yaw = torch.sin(half_yaw)

    quaternions = torch.cat([cos_yaw, torch.zeros_like(yaw), torch.zeros_like(yaw), sin_yaw], dim=-1)

    return quaternions


def euler_to_quat_torch(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to quaternion.
    euler: (N, 3) [roll, pitch, yaw]
    output: (N, 4) [w, x, y, z]
    """
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    quat =  torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ), dim=-1)
    quat = torch.nn.functional.normalize(quat, p=2, dim=-1)
    return quat


def quaternion_continuity_filter_batch(
    quaternions: Union[torch.Tensor, np.ndarray],
    batch_dim: int = 0,
    time_dim: int = 1,
    inplace: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """Apply quaternion continuity filter with explicit batch and time dimensions.

    This is a more flexible version that allows you to specify which dimension
    is the batch and which is time.

    Args:
        quaternions: Quaternion array with shape containing batch and time dimensions
                    Last dimension should be 4 for [w, x, y, z]
        batch_dim: Dimension index for batch
        time_dim: Dimension index for time
        inplace: If True, modify the input array in place (only for torch.Tensor)

    Returns:
        Filtered quaternions with the same shape and type as input

    Example:
        >>> # Shape: (batch=32, time=150, quat=4)
        >>> quats = torch.randn(32, 150, 4)
        >>> quats_filtered = quaternion_continuity_filter_batch(quats, batch_dim=0, time_dim=1)

        >>> # Shape: (time=150, batch=32, quat=4)
        >>> quats = torch.randn(150, 32, 4)
        >>> quats_filtered = quaternion_continuity_filter_batch(quats, batch_dim=1, time_dim=0)
    """
    is_torch = isinstance(quaternions, torch.Tensor)

    if is_torch:
        if not inplace:
            quaternions = quaternions.clone()
    else:
        quaternions = np.array(quaternions, copy=not inplace)

    # Move time dimension to the second-to-last position for easier processing
    # Final shape will be: (..., time, 4)
    ndim = quaternions.ndim
    assert quaternions.shape[-1] == 4, "Last dimension must be 4 for quaternions [w,x,y,z]"

    # Create permutation to move time_dim to position -2
    perm = list(range(ndim))
    perm.pop(time_dim)
    perm.insert(-1, time_dim)

    if is_torch:
        q = quaternions.permute(*perm)
    else:
        q = np.transpose(quaternions, perm)

    # Now apply the filter with time at dimension -2
    N = q.shape[-2]

    if N < 2:
        return quaternions

    # Process each consecutive pair
    for i in range(1, N):
        q_prev = q[..., i-1, :]
        q_curr = q[..., i, :]

        # Compute dot product
        if is_torch:
            dot_product = torch.sum(q_prev * q_curr, dim=-1, keepdim=True)
            flip_mask = (dot_product < 0)
            # The copied implementation counted flips only for disabled logging.
            # Avoid synchronizing the GPU with .item() at every timestep.
            q[..., i, :] = torch.where(flip_mask, -q_curr, q_curr)
        else:
            dot_product = np.sum(q_prev * q_curr, axis=-1, keepdims=True)
            flip_mask = (dot_product < 0)
            q[..., i, :] = np.where(flip_mask, -q_curr, q_curr)

    # Permute back to original dimension order
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    if is_torch:
        quaternions = q.permute(*inv_perm)
    else:
        quaternions = np.transpose(q, inv_perm)

    return quaternions


@torch.jit.script
def quaternion_continuity_align_torchscript(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Align candidate quaternion sign to match reference quaternion."""
    dot = torch.sum(reference * candidate, dim=-1, keepdim=True)
    mask = (dot < 0).expand_as(candidate)
    return torch.where(mask, -candidate, candidate)


def quaternion_continuity_align(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Align candidate quaternion sign to match reference quaternion (eager)."""
    dot = torch.sum(reference * candidate, dim=-1, keepdim=True)
    mask = (dot < 0).expand_as(candidate)
    return torch.where(mask, -candidate, candidate)


@torch.jit.script
def horizon_world_to_body_frame(
    horizon_states: torch.Tensor,
    current_pos: torch.Tensor,
    current_quat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert horizon states from world frame to body frame.

    Args:
        horizon_states: States in world frame of shape (B, H, 13+)
                       [pos(3), quat(4), vel(3), ang_vel(3), ...]
        current_pos: Current position of shape (B, 3)
        current_quat: Current orientation of shape (B, 4) in [w, x, y, z]

    Returns:
        Tuple of (pos_body, quat_rel, vel_body, ang_vel_body) each of shape (B, H, 3/4)
    """
    B, H = horizon_states.shape[0], horizon_states.shape[1]
    current_quat_inverse = qinverse_torch(current_quat)

    pos_world = horizon_states[:, :, :3]
    quat_world = horizon_states[:, :, 3:7]
    vel_world = horizon_states[:, :, 7:10]
    angvel_world = horizon_states[:, :, 10:13]

    # Relative position and rotate to body frame
    pos_rel = pos_world - current_pos.unsqueeze(1)
    horizon_pos_body = qrotate_torch(current_quat_inverse.unsqueeze(1), pos_rel)

    # Rotate velocities to body frame
    horizon_vel_body = qrotate_torch(current_quat_inverse.unsqueeze(1), vel_world)
    horizon_ang_vel_body = qrotate_torch(current_quat_inverse.unsqueeze(1), angvel_world)

    # Relative orientation
    # NOTE: Do NOT standardize to positive w here - it breaks quaternion continuity
    # that was established by quaternion_continuity_filter in data processing
    horizon_quat_rel = qrelative_torch(
        current_quat.unsqueeze(1).expand(-1, H, -1),
        quat_world
    )

    return horizon_pos_body, horizon_quat_rel, horizon_vel_body, horizon_ang_vel_body

@torch.jit.script
def compute_delta_frame_representation(states: torch.Tensor) -> torch.Tensor:
    """Compute per-step delta representation: frame t w.r.t frame (t-1).

    Args:
        states: States in world frame of shape (B, H, 13)
               [pos(3), quat(4), vel(3), ang_vel(3)]

    Returns:
        Deltas of shape (B, H, 13) with index 0 = zeros + unit quaternion
    """
    B, H, D = states.shape
    assert D >= 13, "Input must have at least 13 dims: pos(3)+quat(4)+vel(3)+ang_vel(3)"

    pos = states[..., 0:3]
    quat = states[..., 3:7]
    vel = states[..., 7:10]
    ang_vel = states[..., 10:13]

    # Allocate result tensors
    pos_delta = torch.zeros_like(pos)
    quat_delta = torch.zeros_like(quat)
    vel_delta = torch.zeros_like(vel)
    angvel_delta = torch.zeros_like(ang_vel)

    # Initial frame placeholder
    quat_delta[:, 0] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=states.device,
        dtype=states.dtype
    )

    # Δpos in previous frame coordinates: R(q_{t-1})ᵀ (p_t - p_{t-1})
    pos_delta[:, 1:] = qrotate_torch(
        qinverse_torch(quat[:, :-1]),
        pos[:, 1:] - pos[:, :-1]
    )

    # Δquat = q_{t-1}⁻¹ ⊗ q_t
    quat_delta[:, 1:] = qrelative_torch(quat[:, :-1], quat[:, 1:])

    # Absolute vel in each step's own body frame (AnyCar-style, not differenced)
    vel_delta = qrotate_torch(qinverse_torch(quat), vel)
    angvel_delta = qrotate_torch(qinverse_torch(quat), ang_vel)

    # Concatenate into final delta representation
    deltas = torch.cat([pos_delta, quat_delta, vel_delta, angvel_delta], dim=-1)
    return deltas


@torch.compile(fullgraph=True, dynamic=True)  # type: ignore[arg-type]
def reconstruct_absolute_from_deltas(deltas: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct absolute (w.r.t. first frame) states from mixed delta/self-frame repr.
    Pos/quat at time t are deltas expressed in frame (t-1).
    Vel/angvel at time t are absolute, expressed in body frame of step t (AnyCar-style).

    Args:
        deltas: (B,H,7)  -> [Δpos(3), Δquat(4)]
                (B,H,13) -> [Δpos(3), Δquat(4), vel_self(3), angvel_self(3)]

    Returns:
        abs_states: (B,H,7) or (B,H,13)
    """
    B, H, D = deltas.shape
    has_vel = (D == 13)

    dpos  = deltas[..., 0:3]
    dquat = F.normalize(deltas[..., 3:7], dim=-1, eps=1e-8)

    q_identity = torch.zeros((B, 4), device=deltas.device, dtype=deltas.dtype)
    q_identity[:, 0] = 1.0

    q_abs = torch.zeros((B, H, 4), device=deltas.device, dtype=deltas.dtype)
    q_prev = q_identity

    # === sequential quaternion accumulation with continuity enforcement ===
    for t in range(H):
        q_prev = F.normalize(qmultiply_torch(q_prev, dquat[:, t]), dim=-1, eps=1e-8)
        if t == 0:
            ref = q_identity
        else:
            ref = q_abs[:, t - 1]
        if torch.jit.is_scripting():
            q_prev = quaternion_continuity_align_torchscript(ref, q_prev)
        else:
            q_prev = quaternion_continuity_align(ref, q_prev)
        q_abs[:, t] = q_prev

    q_prev_all = torch.cat([q_identity.unsqueeze(1), q_abs[:, :-1]], dim=1)
    dpos_world = qrotate_torch(q_prev_all, dpos)
    pos_abs = torch.cumsum(dpos_world, dim=1)

    if has_vel:
        # Vel/angvel stay in each step's own body frame (AnyCar-style pass-through).
        vel_self = deltas[..., 7:10]
        ang_self = deltas[..., 10:13]
        return torch.cat([pos_abs, q_abs, vel_self, ang_self], dim=-1)
    else:
        return torch.cat([pos_abs, q_abs], dim=-1)
