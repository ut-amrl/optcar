"""FiLM/cross-attention FKD models, checkpoints, and trajectory prediction."""
from dataclasses import dataclass
from pathlib import Path
import math
import os
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from optcar.dataset import SCHEMA, DataConfig, integrate_deltas, pack_windows
from optcar.utils.math import wrap_to_pi, yaw_from_quat_torch


@dataclass
class ModelConfig:
    architecture: str = "film"
    history_steps: int = 250
    future_steps: int = 50
    embed_dim: int = 64
    heads: int = 4
    history_layers: int = 2
    decoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        if self.architecture not in ("film", "cross_attention"):
            raise ValueError("architecture must be film or cross_attention")
        if self.heads < 1 or self.embed_dim % self.heads or self.embed_dim % 2:
            raise ValueError("embed_dim must be even and divisible by heads")
        if min(self.history_layers, self.decoder_layers, self.future_steps) < 1 or self.history_steps < 2:
            raise ValueError("Invalid model depth or horizon")


def positional_encoding(length, dim):
    positions = torch.arange(length).float()[:, None]
    scales = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(positions * scales)
    encoding[:, 1::2] = torch.cos(positions * scales)
    return encoding[None]


class FiLMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(config.embed_dim, config.heads, config.feedforward_dim,
                                                config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.modulation = nn.Linear(config.embed_dim, 2 * config.embed_dim)
        nn.init.normal_(self.modulation.weight, std=1e-3)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, value, context, mask):
        value = self.block(value, src_mask=mask)
        gamma, beta = self.modulation(context).chunk(2, -1)
        return value * (1 + gamma[:, None]) + beta[:, None]


class FKDTransformer(nn.Module):
    """Both variants use identical observed-history and command interfaces.

    FiLM has no cross-attention path around its single-context bottleneck.
    Cross-attention exposes encoded history tokens to the rollout decoder.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        def embedding(width):
            return nn.Sequential(nn.Linear(width, c.embed_dim), nn.GELU(), nn.Linear(c.embed_dim, c.embed_dim))
        self.history_embedding = embedding(15)
        self.current_embedding = embedding(13)
        self.command_embedding = embedding(2)
        self.context_token = nn.Parameter(torch.zeros(1, 1, c.embed_dim))
        nn.init.normal_(self.context_token, std=0.02)
        layer = nn.TransformerEncoderLayer(c.embed_dim, c.heads, c.feedforward_dim, c.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, c.history_layers, norm=nn.LayerNorm(c.embed_dim))
        if c.architecture == "film":
            self.decoder = nn.ModuleList([FiLMBlock(c) for _ in range(c.decoder_layers)])
        else:
            layer = nn.TransformerDecoderLayer(c.embed_dim, c.heads, c.feedforward_dim, c.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(layer, c.decoder_layers)
        self.output = nn.Sequential(nn.LayerNorm(c.embed_dim), nn.Linear(c.embed_dim, 7))
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer("history_position", positional_encoding(c.history_steps, c.embed_dim))
        self.register_buffer("future_position", positional_encoding(c.future_steps, c.embed_dim))
        for name, size in (("history", 15), ("current", 13), ("commands", 2)):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_std", torch.ones(size))

    def set_normalization(self, statistics):
        for name in ("history", "current", "commands"):
            for kind in ("mean", "std"):
                target = getattr(self, f"{name}_{kind}")
                target.copy_(torch.as_tensor(statistics[name][kind], device=target.device))

    def normalize(self, name, value):
        return (value - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std").clamp_min(1e-6)

    def forward(self, history, current, commands):
        c = self.config
        if history.shape[1:] != (c.history_steps - 1, 15):
            raise ValueError("History shape does not match checkpoint")
        if commands.shape[1] > c.future_steps or commands.shape[1] < 1:
            raise ValueError("Prediction horizon exceeds checkpoint horizon")
        encoded = self.history_embedding(self.normalize("history", history))
        encoded = torch.cat((self.context_token.expand(history.shape[0], -1, -1), encoded), 1)
        encoded = self.history_encoder(encoded + self.history_position)
        query = self.command_embedding(self.normalize("commands", commands))
        query = query + self.current_embedding(self.normalize("current", current))[:, None]
        query = query + self.future_position[:, :commands.shape[1]]
        mask = torch.ones(commands.shape[1], commands.shape[1], dtype=torch.bool, device=commands.device).triu(1)
        if c.architecture == "film":
            for layer in self.decoder:
                query = layer(query, encoded[:, 0], mask)
        else:
            query = self.decoder(query, encoded[:, 1:], tgt_mask=mask)
        value = self.output(query)
        identity = value.new_tensor([1., 0., 0., 0.])
        quaternion = F.normalize(value[..., 3:7] + identity, dim=-1)
        return torch.cat((value[..., :3], quaternion), -1)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("schema") != SCHEMA:
        raise ValueError("Expected an OptCar checkpoint with the supported data schema")
    model = FKDTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, DataConfig(**checkpoint["data_config"]), checkpoint


def trajectory_metrics(predicted, measured):
    predicted = torch.as_tensor(predicted, dtype=torch.float32)
    measured = torch.as_tensor(measured, dtype=torch.float32)
    distance = (predicted[..., :3] - measured[..., :3]).norm(dim=-1)
    heading = wrap_to_pi(yaw_from_quat_torch(predicted[..., 3:7]) - yaw_from_quat_torch(measured[..., 3:7])).abs()
    return {"position_mean_m": float(distance.mean()), "position_rmse_m": float(distance.square().mean().sqrt()),
            "heading_mae_rad": float(heading.mean()), "final_position_error_m": float(distance[-1])}


@torch.inference_mode()
def predict_window(model, config, states, actions, current, horizon=None):
    h = config.history_steps
    horizon = config.future_steps if horizon is None else int(horizon)
    if current < h - 1 or current + horizon >= len(states) or not 1 <= horizon <= config.future_steps:
        raise ValueError("Select a window with sufficient recorded history and future, within the checkpoint horizon")
    device = next(model.parameters()).device
    # Build features solely from observed history through current; no future states.
    history = torch.as_tensor(states[current - h + 1:current + 1], device=device).float()[None]
    commands = torch.as_tensor(actions[current - h + 1:current], device=device).float()[None]
    padded = torch.cat((commands, commands.new_zeros((1, 1, 2))), 1)
    packed = pack_windows(history, padded, h)
    tokens = torch.cat((packed[:, 1:h, 26:39], commands), -1)
    future_commands = torch.as_tensor(actions[current:current + horizon], device=device).float()[None]
    delta = model(tokens, packed[:, h - 1, 13:26], future_commands)
    poses = integrate_deltas(delta, history[:, -1, :7])[0].cpu().numpy()
    measured = np.asarray(states[current + 1:current + horizon + 1, :7])
    return {"predicted": poses.tolist(), "measured": measured.tolist(),
            "metrics": trajectory_metrics(poses, measured), "current": current, "horizon": horizon}
