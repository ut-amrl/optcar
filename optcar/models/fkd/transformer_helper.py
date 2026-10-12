"""Transformer helper utilities for FKD models.

This module provides reusable building blocks shared across the FKD
transformer models (multi-layer perceptrons, positional encodings,
masking utilities, and the heightmap encoder).
"""

from __future__ import annotations

import torch
from torch import nn


def generate_markov_mask(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a causal mask that allows attending to the current and previous step only.

    Args:
        seq_len: Target sequence length.
        device: Device where the mask should live.

    Returns:
        A ``(seq_len, seq_len)`` mask filled with ``0`` on the diagonal and the
        lower sub-diagonal, and ``-inf`` elsewhere. This is compatible with the
        PyTorch transformer API.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    indices = torch.arange(seq_len, device=device)
    mask[indices, indices] = 0.0
    if seq_len > 1:
        mask[indices[1:], indices[:-1]] = 0.0
    return mask


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding with optional reversal and dropout."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1, *, flip: bool = False) -> None:
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.flip = flip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to ``x``.

        Args:
            x: Tensor shaped ``(batch, seq_len, d_model)``.
        """
        seq_len = x.size(1)
        pos = self.pos_enc[:seq_len]
        if self.flip:
            pos = torch.flip(pos, dims=[0])
        return self.dropout(x + pos)


class MultiLinear(nn.Module):
    """Configurable MLP with optional ReLU and dropout between layers."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        hidden_dims: list[int],
        *,
        relu: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if relu:
                layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MapEncoder_ViT(nn.Module):
    """Lightweight ViT-style encoder for local height maps."""

    def __init__(self, embed_dim: int, *, patch_size: int) -> None:
        super().__init__()
        if patch_size % 32 != 0:
            raise ValueError("patch_size must be divisible by 32 for the current architecture")

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        red = patch_size // 32
        flattened_dim = 16 * red * red

        self.map_patch_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, hmap: torch.Tensor) -> torch.Tensor:
        """Encode height maps into a sequence of embeddings.

        Args:
            hmap: Tensor of shape ``(B, C, H, W)`` or ``(B, H, W)`` with ``C`` equal to 1.
        """
        if hmap.dim() == 3:
            hmap = hmap.unsqueeze(1)
        elif hmap.size(1) != 1:
            raise ValueError("Height map tensor must have a single channel")

        B, _, H, W = hmap.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("Height map dimensions must be multiples of patch_size")

        patches = hmap.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        B, C, n_h, n_w, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * n_h * n_w, C, self.patch_size, self.patch_size)

        embeddings = self.map_patch_encoder(patches)
        n_patches = n_h * n_w
        return embeddings.reshape(B, n_patches, self.embed_dim)
