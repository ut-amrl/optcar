"""Math utilities for analytical models."""

import torch
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


def sqaured_distance_torch(x, x_d):
    """
    Squared distance between states.
    x: actual state, tensor, (N, m)
    x_d: desired, tensor, (m,)
    output: squared distance, tensor, (N,)
    """
    return torch.einsum('ij,ij->i', x-x_d, x-x_d)


def qdistance_torch(q1, q2):
    """
    Quaternion distance.
    q1: tensor, (..., 4)
    q2: tensor, (..., 4)
    output: tensor, (...,)
    distance = 1 - <q1,q2>^2
    """
    return 1 - torch.einsum('...i, ...i -> ...', q1, q2)**2


def qconjugate_torch(q):
    """Quaternion conjugate."""
    temp = torch.zeros_like(q)
    temp[:,0] = q[:,0]
    temp[:,1:] = -q[:,1:]
    return temp


def z_from_q(q):
    """
    Extract z-axis from quaternion.
    q: (N, H, 4)
    output: (N, H, 3)
    """
    e3 = torch.tensor((0, 0, 1.0)).view(1, 1, 3)
    temp = 2. * torch.cross(q[:, :, 1:], e3)
    return e3 + q[:, :, 0:1] * temp + torch.cross(q[:, :, 1:], temp)


def qexp_torch(q):
    """
    Quaternion exponential.
    q: tensor, (N, 4)
    output: tensor, (N, 4)
    """
    norm = torch.linalg.norm(q[:,1:], dim=1)
    e = torch.exp(q[:,0])
    result_w = e * torch.cos(norm)

    N = q.shape[0]
    result_v = e.view(N,1) * q[:,1:] / norm.view(N,1) * torch.sin(norm).view(N,1)
    result_v[torch.isnan(result_v)] = 0

    return torch.cat((result_w.view(N,1), result_v), dim=1)


def qintegrate_torch(q, v, dt, frame='body'):
    """
    Integrate quaternion with angular velocity.
    q: tensor, (N, 4)
    v: tensor, (N, 3)
    output: tensor, (N, 4)
    """
    quat_v = torch.zeros_like(q)
    quat_v[:,1:] = v * dt / 2.
    if frame == 'body':
        return qmultiply_torch(q, qexp_torch(quat_v))
    if frame == 'world':
        return qmultiply_torch(qexp_torch(quat_v), q)


def qstandardize_torch(q):
    """
    Standardize quaternion to positive w.
    q: tensor, (N, 4)
    output: tensor, (N, 4)
    """
    return torch.where(q[:, 0:1] < 0, -q, q)


def qtoR_torch(q):
    """
    Convert quaternion to rotation matrix.
    q: tensor, (N, B, 4)
    output: rotation matrix tensor, (N, B, 3, 3)
    """
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    
    R = torch.zeros(q.shape[0], q.shape[1], 3, 3, device=q.device, dtype=q.dtype)

    R[..., 0, 0] = 2 * (q0 * q0 + q1 * q1) - 1
    R[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)

    R[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[..., 1, 1] = 2 * (q0 * q0 + q2 * q2) - 1
    R[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)

    R[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[..., 2, 2] = 2 * (q0 * q0 + q3 * q3) - 1

    return R


def random_quat_gen(batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
    """
    Generates random quaternions representing rotations in 3D space.
    Args:
        batch_size (int): Number of random quaternions to generate.
        device (str): Device to create the tensor on.
    Returns:
        torch.Tensor: Random quaternions of shape (B, 4) with [w, x, y, z].
    """
    u1 = torch.rand(batch_size, device=device)
    u2 = torch.rand(batch_size, device=device)
    u3 = torch.rand(batch_size, device=device)

    w = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    x = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    y = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    z = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    q_random = torch.stack([w, x, y, z], dim=-1)
    q_random = q_random / torch.norm(q_random, dim=-1, keepdim=True)  # Normalize to unit quaternion
    return q_random


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


def quat_from_vec_torch(v: torch.Tensor) -> torch.Tensor:
    """
    Compute a quaternion aligning the x-axis to the given vector `v`.
    v :  (N, 3)
    output : Quaternion (w, x, y, z) (N, 4).
    """
    v = torch.nn.functional.normalize(v, dim=-1)  # Normalize the input vector
    ref_dir = torch.tensor([1.0, 0, 0], device=v.device, dtype=torch.float32)  # Reference direction (x-axis)

    # Compute cosine of angle and rotation axis
    cos_theta = torch.clamp(v[:, 0], -1.0, 1.0)  # Dot product with [1, 0, 0] is just v[:, 0]
    axis = torch.nn.functional.normalize(torch.cross(ref_dir.expand_as(v), v, dim=-1), dim=-1)

    # Handle edge cases where v is parallel to ref_dir
    axis[cos_theta.abs() == 1.0] = torch.tensor([0.0, 0.0, 1.0], device=v.device, dtype=torch.float32)

    # Compute quaternion components
    half_angle = torch.acos(cos_theta) / 2
    sin_half_angle, cos_half_angle = torch.sin(half_angle), torch.cos(half_angle)

    return torch.cat([cos_half_angle.unsqueeze(-1), axis * sin_half_angle.unsqueeze(-1)], dim=-1)
