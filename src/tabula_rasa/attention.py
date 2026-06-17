"""Multi-head self-attention with configurable position encoding and KV cache."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabula_rasa.norms import _compute_alibi_slopes
from tabula_rasa.rope import (
    RotaryEmbedding,
    LearnedPositionEmbedding,
    _apply_pos_encoding,
)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE / learned / none position encoding.

    Supports:
    - RoPE, learned absolute, or no position encoding.
    - ALiBi-style arithmetic bias (from ``_compute_arithmetic_bias``).
    - KV-cached generation for efficient autoregressive decoding.
    - Optional Flash Attention via ``torch.nn.functional.scaled_dot_product_attention``.
    - GQA (grouped-query attention) when ``n_kv_heads < n_heads``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = getattr(config, "n_kv_heads", config.n_heads)
        self.head_dim = config.d_model // config.n_heads
        self.pos_type = config.pos_encoding
        self.dropout = getattr(config, "dropout", 0.0)
        self.is_causal = True  # default; overridden when explicit mask is passed

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        # Position encoding
        if self.pos_type == "rope":
            self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        elif self.pos_type == "learned":
            self.learned_pos = LearnedPositionEmbedding(config.max_seq_len, config.d_model)
        else:
            self.rope = None
            self.learned_pos = None

        # RASA (Relation-Aware Sparse Attention) — learned relative position biases
        self.rasa_bias: nn.Embedding | None = None
        if getattr(config, "use_rasa", False):
            rasa_max = getattr(config, "rasa_max_relative", 16)
            self.rasa_bias = nn.Embedding(2 * rasa_max + 1, self.n_heads)

        # ALiBi slopes
        if config.pos_encoding == "alibi_arithmetic":
            alibi_slopes = _compute_alibi_slopes(config.n_heads)
            self.register_buffer("alibi_slopes", alibi_slopes)
        else:
            self.alibi_slopes = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        arithmetic_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = x.shape
        device = x.device

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Pre-compute RoPE cos/sin or learned embeddings
        cos = sin = None
        pos_offset = 0
        if self.pos_type == "rope":
            total_len = seq_len + (past_kv[0].shape[2] if past_kv is not None else 0)
            cos, sin = self.rope(total_len, device)
            if past_kv is not None:
                pos_offset = past_kv[0].shape[2]

        # Apply position encoding
        if self.pos_type not in ("none", "abacus") and cos is not None:
            seq_cos = cos[pos_offset:pos_offset + seq_len].unsqueeze(0).unsqueeze(1)
            seq_sin = sin[pos_offset:pos_offset + seq_len].unsqueeze(0).unsqueeze(1)
            q, k = _apply_pos_encoding(q, k, seq_cos, seq_sin, self.pos_type)

        # KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        # Expand KV heads for GQA
        if self.n_kv_heads < self.n_heads:
            n_reps = self.n_heads // self.n_kv_heads
            k = k[:, :, None, :, :].expand(-1, -1, n_reps, -1, -1).reshape(
                batch, self.n_heads, -1, self.head_dim
            )
            v = v[:, :, None, :, :].expand(-1, -1, n_reps, -1, -1).reshape(
                batch, self.n_heads, -1, self.head_dim
            )

        # Build combined attention bias
        attn_bias = None
        if mask is not None:
            attn_bias = mask
            self.is_causal = False
        if arithmetic_bias is not None:
            attn_bias = arithmetic_bias if attn_bias is None else attn_bias + arithmetic_bias

        # RASA bias
        if self.rasa_bias is not None:
            total_k = k.shape[2]
            rasa_max = (self.rasa_bias.num_embeddings - 1) // 2
            q_pos = torch.arange(seq_len, device=device)
            k_pos = torch.arange(total_k, device=device)
            rel_pos = q_pos[:, None] - k_pos[None, :]
            rel_pos = rel_pos.clamp(-rasa_max, rasa_max) + rasa_max
            rasa_bias = self.rasa_bias(rel_pos).permute(2, 0, 1).unsqueeze(0)
            attn_bias = rasa_bias if attn_bias is None else attn_bias + rasa_bias

        # Add ALiBi slopes
        if self.alibi_slopes is not None and arithmetic_bias is None:
            total_len = k.shape[2]
            pos = torch.arange(total_len, device=device)
            rel_pos = pos[:, None] - pos[None, :]
            alibi_bias = -self.alibi_slopes.view(-1, 1, 1) * rel_pos.abs().float()
            attn_bias = alibi_bias.unsqueeze(0) if attn_bias is None else attn_bias + alibi_bias.unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            is_causal=self.is_causal if attn_bias is None and past_kv is None else False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.wo(out)
        return out, (k, v)
