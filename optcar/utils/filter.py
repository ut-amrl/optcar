"""Quaternion continuity filter copied from wheeledlab_utils/filter.py."""
from typing import Union
import numpy as np
import torch


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
