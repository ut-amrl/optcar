"""Frame transforms copied from WheeledLab; only local imports changed."""
from typing import Tuple
import torch
import torch.nn.functional as F
from optcar.utils.math import qinverse_torch, qrotate_torch, qrelative_torch, qmultiply_torch
from optcar.utils.filter import quaternion_continuity_align, quaternion_continuity_align_torchscript


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
