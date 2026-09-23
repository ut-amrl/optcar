"""Open-loop prediction metrics; no simulated closed-loop control."""
from collections import defaultdict
import math

import torch
from torch.utils.data import DataLoader

from optcar.dataset import FKDDataset, integrate_deltas
from optcar.models.fkd_transformer import load_checkpoint
from optcar.utils.math import wrap_to_pi, yaw_from_quat_torch


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def prediction_loss(predicted_delta, batch, position_weight=1.0, quaternion_weight=0.3):
    predicted = integrate_deltas(predicted_delta)
    position = (predicted[..., :3] - batch["target_pose"][..., :3]).square().sum(-1).mean()
    q = predicted[..., 3:7]
    target = batch["target_pose"][..., 3:7]
    # Sign-invariant normalized quaternion chordal distance.
    quaternion = torch.minimum((q - target).square().sum(-1), (q + target).square().sum(-1)).mean()
    return position_weight * position + quaternion_weight * quaternion


@torch.inference_mode()
def evaluate_model(model, loader, device, position_weight=1.0, quaternion_weight=0.3):
    model.eval()
    groups = defaultdict(lambda: {"squared_position": 0., "position": 0., "heading": 0., "count": 0})
    total_loss, count = 0., 0
    for raw in loader:
        batch = to_device(raw, device)
        delta = model(batch["history"], batch["current"], batch["commands"])
        poses = integrate_deltas(delta)
        error = (poses[..., :3] - batch["target_pose"][..., :3]).norm(dim=-1)
        heading = wrap_to_pi(yaw_from_quat_torch(poses[..., 3:7]) - yaw_from_quat_torch(batch["target_pose"][..., 3:7])).abs().squeeze(-1)
        loss = prediction_loss(delta, batch, position_weight, quaternion_weight)
        total_loss += float(loss) * len(error)
        count += len(error)
        for i, (source, terrain) in enumerate(zip(raw["source"], raw["terrain"])):
            group = groups[f"{source}/{terrain}"]
            group["squared_position"] += float(error[i].square().sum())
            group["position"] += float(error[i].sum())
            group["heading"] += float(heading[i].sum())
            group["count"] += error.shape[1]
    if not count:
        raise ValueError("Evaluation dataset is empty")
    result = {}
    for key, group in groups.items():
        n = group["count"]
        result[key] = {"position_rmse_m": math.sqrt(group["squared_position"] / n),
                       "position_mean_m": group["position"] / n,
                       "heading_mae_rad": group["heading"] / n, "predicted_steps": n}
    return {"loss": total_loss / count, "windows": count, "by_source_terrain": result}


def evaluate(checkpoint, roots, split="test", device="cpu", batch_size=256, real_only=False):
    model, config, _ = load_checkpoint(checkpoint, device)
    dataset = FKDDataset(roots, config, split, sources=["real"] if real_only else None)
    return evaluate_model(model, DataLoader(dataset, batch_size=batch_size), device)
