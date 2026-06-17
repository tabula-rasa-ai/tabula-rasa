"""Normalisation layers and activation helpers for the transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(dim: int, norm_type: str = "rmsnorm") -> nn.Module:
    """Factory: returns an ``RMSNorm`` or ``LayerNorm`` module.

    Args:
        dim: Feature dimension to normalise over.
        norm_type: ``'rmsnorm'`` (default) or ``'layernorm'``.

    Returns:
        A normalisation module.
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=1e-6)
    return RMSNorm(dim)


def _make_activation(name: str = "swiglu") -> callable[[torch.Tensor], torch.Tensor]:
    """Factory: returns an activation function.

    Args:
        name: One of ``'swiglu'``, ``'relu'``, ``'gelu'``.

    Returns:
        A callable activation.
    """
    if name == "swiglu":
        return _swiglu
    if name == "gelu":
        return F.gelu
    return F.relu


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: element-wise multiply with sigmoid-gated variant."""
    return x * torch.sigmoid(x)


def _compute_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute head-specific ALiBi slopes (geometric sequence of 2^(-8k/n_heads))."""
    def get_slopes(n: int) -> list[float]:
        if n == 1:
            return [0.1]
        if n == 2:
            return [1.0 / 2**0.5, 1.0]
        # Recursive: interleave slopes from n//2
        slopes_half = get_slopes(n // 2)
        return [s / 2.0 for s in slopes_half] + slopes_half
    return torch.tensor(get_slopes(n_heads), dtype=torch.float32)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    Reference:
        Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
