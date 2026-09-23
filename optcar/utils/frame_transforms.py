"""Body-frame transforms adapted from wheeledlab_utils/transform_utils.py."""
from typing import Tuple

import torch

from optcar.utils.math import qinverse_torch, qrotate_torch, qrelative_torch


@torch.jit.script
def horizon_world_to_body_frame(
    states: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express [B,T,13] world states in each batch's reference pose."""
    inverse = qinverse_torch(quaternion).unsqueeze(1)
    # Keep quaternion signs continuous across the sequence.
    return (
        qrotate_torch(inverse, states[..., :3] - position.unsqueeze(1)),
        qrelative_torch(quaternion.unsqueeze(1).expand(-1, states.shape[1], -1), states[..., 3:7]),
        qrotate_torch(inverse, states[..., 7:10]),
        qrotate_torch(inverse, states[..., 10:13]),
    )


@torch.jit.script
def compute_delta_frame_representation(states: torch.Tensor) -> torch.Tensor:
    """Previous-frame pose increments plus each state's own body velocities."""
    position, quaternion = states[..., :3], states[..., 3:7]
    inverse = qinverse_torch(quaternion)
    delta_position = torch.zeros_like(position)
    delta_quaternion = torch.zeros_like(quaternion)
    delta_quaternion[:, 0, 0] = 1
    delta_position[:, 1:] = qrotate_torch(inverse[:, :-1], position[:, 1:] - position[:, :-1])
    delta_quaternion[:, 1:] = qrelative_torch(quaternion[:, :-1], quaternion[:, 1:])
    return torch.cat((delta_position, delta_quaternion,
                      qrotate_torch(inverse, states[..., 7:10]),
                      qrotate_torch(inverse, states[..., 10:13])), dim=-1)
