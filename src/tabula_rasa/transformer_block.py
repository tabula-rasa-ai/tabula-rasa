"""Transformer block — layer of attention + feed-forward with residual connections."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tabula_rasa.attention import Attention
from tabula_rasa.ffn import FeedForward, MoEFeedForward
from tabula_rasa.norms import _make_norm


class TransformerBlock(nn.Module):
    """Single transformer block: pre-norm attention → residual → pre-norm FFN → residual.

    Supports:
    - Pre-normalisation (RMSNorm or LayerNorm).
    - SwiGLU / ReLU / GELU feed-forward.
    - MoE feed-forward (top-k expert routing).
    - StreamBP: selective attention checkpointing for memory efficiency.
    - RASA attention bias.
    """

    def __init__(self, config: Any, layer_id: int) -> None:
        super().__init__()
        self.use_streambp = getattr(config, "use_streambp", True)
        self.use_moe = getattr(config, "use_moe", False)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.0))

        self.attention_norm = _make_norm(config.d_model, config.norm_type)
        self.attention = Attention(config)
        self.ffn_norm = _make_norm(config.d_model, config.norm_type)

        if self.use_moe:
            self.feed_forward = MoEFeedForward(config)
        else:
            self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        arithmetic_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if self.use_streambp and self.training and past_kv is None:
            attn_out, current_kv = torch.utils.checkpoint.checkpoint(
                self.attention, self.attention_norm(x), mask, None, arithmetic_bias,
                use_reentrant=False,
            )
        else:
            attn_out, current_kv = self.attention(
                self.attention_norm(x), mask, past_kv=past_kv, arithmetic_bias=arithmetic_bias
            )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x, current_kv
