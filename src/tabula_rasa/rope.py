"""Position encoding modules for the transformer.

Supports RoPE (Rotary Position Embedding), learned absolute positions,
Abacus place-value embeddings, and ALiBi-style arithmetic bias helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of *x* by half (standard RoPE operation)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_pos_encoding(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    pos_type: str = "rope",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply position encoding to query and key tensors.

    Args:
        q: Query tensor ``(batch, heads, seq, head_dim)``.
        k: Key tensor ``(batch, heads, seq, head_dim)``.
        cos: Cosine precomputed for RoPE.
        sin: Sine precomputed for RoPE.
        pos_type: ``'rope'`` or ``'none'``.

    Returns:
        Tuple ``(q, k)`` with position encoding applied.
    """
    if pos_type == "none":
        return q, k
    # RoPE (default)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Pre-computes sinusoidal frequencies for a given dimension and maximum
    sequence length. Forward pass returns ``(cos, sin)`` for a requested
    sequence length on the target device.
    """

    def __init__(self, dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.dim = dim
        inv_freq: torch.Tensor = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(seq_len, device=device)
        freqs = torch.einsum("i,j->ij", position_ids.float(), self.inv_freq.float())
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


class LearnedPositionEmbedding(nn.Module):
    """Learnable absolute position embeddings."""

    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.pe(positions)


class AbacusEmbedding(nn.Module):
    """Place-value position embeddings for length generalisation.

    Instead of encoding absolute sequence position (RoPE), Abacus embeddings
    encode **digit place value** (ones=0, tens=1, ...). This lets the model
    generalise to longer unseen digit lengths because the position encoding
    is identical regardless of sequence length.

    Non-digit tokens receive fixed positions:
        operators -> ``max_digits``
        ``=``     -> ``max_digits + 1``
        other     -> ``max_digits + 2``

    Reference:
        McLeish et al., "Transformers Can Do Arithmetic with the Right
        Embeddings", NeurIPS 2024.
    """

    def __init__(self, max_digits: int, dim: int) -> None:
        super().__init__()
        self.max_digits = max_digits
        npos = max_digits + 3
        self.embed = nn.Embedding(npos, dim)
        self.op_id = max_digits
        self.eq_id = max_digits + 1
        self.other_id = max_digits + 2

    def forward(self, place_values: torch.Tensor) -> torch.Tensor:
        pos = place_values.clone()
        pos[pos < 0] = self.other_id
        pos[pos == -1] = self.op_id
        pos[pos == -2] = self.eq_id
        pos = pos.clamp(0, self.max_digits + 2)
        return self.embed(pos)
