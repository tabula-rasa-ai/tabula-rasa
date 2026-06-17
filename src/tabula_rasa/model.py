"""Tabula Rasa Transformer — core model.

Public API (backward-compatible re-exports):

    MathTransformer       Full autoregressive transformer.
    alphazero_loss        Loss function combining policy + value targets.
    count_parameters      Utility to count model parameters.

Sub-modules (also importable directly):

    tabula_rasa.rope       Position encodings (RoPE, learned, Abacus).
    tabula_rasa.norms      Normalisation layers (RMSNorm, LayerNorm).
    tabula_rasa.attention  Multi-head self-attention.
    tabula_rasa.ffn        Feed-forward blocks (SwiGLU, MoE).
    tabula_rasa.transformer_block  Single transformer block.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Sub-module imports ──────────────────────────────────────────────
from tabula_rasa.norms import _make_activation, _make_norm
from tabula_rasa.rope import (
    AbacusEmbedding,
    LearnedPositionEmbedding,
    RotaryEmbedding,
    _apply_pos_encoding,
    _rotate_half,
)
from tabula_rasa.transformer_block import TransformerBlock

# Re-export for backward compat (scripts that import from tabula_rasa.model)
__all__ = [
    "MathTransformer",
    "alphazero_loss",
    "count_parameters",
    # Sub-module symbols kept accessible for backward compat
    "_make_norm",
    "_make_activation",
    "_apply_pos_encoding",
    "_rotate_half",
    "AbacusEmbedding",
    "RotaryEmbedding",
    "LearnedPositionEmbedding",
    "TransformerBlock",
]


class MathTransformer(nn.Module):
    """Tiny autoregressive transformer — architecture set by Config.

    Supports:
    - Multi-head self-attention with RoPE / learned / Abacus / ALiBi.
    - Pre-normalisation (RMSNorm or LayerNorm).
    - SwiGLU / ReLU / GELU feed-forward blocks.
    - MoE (Mixture of Experts) feed-forward.
    - KV-cached generation for efficient decoding.
    - AlphaZero-style value head.
    - StreamBP memory-efficient attention checkpointing.
    """

    def __init__(self, config: Any) -> None:
        """Initialise the transformer model.

        Args:
            config: Configuration object (``vocab_size``, ``d_model``,
                ``n_layers``, etc.).
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)

        # Abacus position embeddings (place-value based, for length generalisation)
        if config.pos_encoding == "abacus":
            max_digits = getattr(config, "max_digits", 4)
            self.abacus_embedding = AbacusEmbedding(max_digits, config.d_model)
        else:
            self.abacus_embedding = None

        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.norm = _make_norm(config.d_model, config.norm_type)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # AlphaZero-style Value Head: outputs scalar -1 to +1 (intuition)
        if getattr(config, "use_value_head", False):
            self.value_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.Tanh(),
                nn.Linear(config.d_model // 2, 1),
                nn.Tanh(),
            )
        else:
            self.value_head = None

        # Weight tying (shared embedding/head) — OFF by default for algorithmic tasks
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialise weights
        self.apply(self._init_weights)

        # torch.compile() — PyTorch 2.0+ JIT compilation for 2-3x speedup
        if getattr(config, "use_compile", False) and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(self.forward, mode="reduce-overhead")
                print("  [*] torch.compile() enabled (mode=reduce-overhead)")
            except Exception as e:
                print(f"  [!] torch.compile() failed: {e}")

        if getattr(config, "use_compile", False) and hasattr(torch, "compile"):
            try:
                self._generate_step_impl = torch.compile(self._generate_step_impl, mode="reduce-overhead")
            except Exception:
                pass

    # ── Weight initialisation ───────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        init_type = getattr(self.config, "weight_init", "normal")
        init_std = getattr(self.config, "init_std", 0.02)
        if isinstance(module, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if init_type == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)

    # ── Attention mask helpers ───────────────────────────────────────

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    # Digit ranges in the MathTokenizer vocabulary:
    #   10-29 = carry-digit combined tokens ("00"-"19")
    #   30-39 = single digit tokens ("0"-"9")
    #   40-43 = operators (+ - * /)
    #   44    = equals
    _DIGIT_IDS = frozenset(range(10, 40))  # carry + single digits
    _OP_IDS = frozenset(range(40, 44))
    _EQ_ID = 44

    def _compute_place_values(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute decimal place values for each token in a batch.

        Digit tokens (single ``0``-``9`` or carry-digit ``00``-``19``)
        are assigned place values (0=ones, 1=tens, …) reading right-to-left
        within each operand. Non-digit tokens get fixed values.

        Returns:
            Place values ``(batch, seq_len)``: ``-1``=operator, ``-2``=equals,
            ``-3``=other.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        places = torch.full_like(input_ids, -3, dtype=torch.long)
        digit_ids = self._DIGIT_IDS
        op_ids = self._OP_IDS
        eq_id = self._EQ_ID
        for b in range(batch):
            ids = input_ids[b]
            eq_pos = (ids == eq_id).nonzero(as_tuple=True)[0]
            eq = eq_pos[0].item() if len(eq_pos) > 0 else seq_len
            # Operand side (before '=' or end): assign place values right-to-left
            place = 0
            for i in range(eq - 1, -1, -1):
                tid = ids[i].item()
                if tid in digit_ids:
                    places[b, i] = place
                    place += 1
                elif tid in op_ids:
                    places[b, i] = -1
                    place = 0
            # Mark the '=' token itself
            if eq < seq_len:
                places[b, eq] = -2
            # Output side (after '='): left-to-right, ones first
            place = 0
            for i in range(eq + 1, seq_len):
                tid = ids[i].item()
                if tid in digit_ids:
                    places[b, i] = place
                    place += 1
                else:
                    break
        return places

    def _compute_arithmetic_bias(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.config.pos_encoding != "alibi_arithmetic":
            return None
        places = self._compute_place_values(input_ids)
        p_i = places.unsqueeze(2)
        p_j = places.unsqueeze(1)
        diff = torch.abs(p_i - p_j).float()
        return -diff.unsqueeze(1)

    # ── Token difficulty weighting ───────────────────────────────────

    def _compute_token_weights(self, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(targets, dtype=torch.float)
        cfg = self.config
        if not getattr(cfg, "use_difficulty_weighting", False):
            return weights
        carry_mult = getattr(cfg, "difficulty_carry_mult", 2.0)
        op_mult = getattr(cfg, "difficulty_op_mult", 1.5)
        carry_mask = (targets >= 10) & (targets <= 29)
        weights = torch.where(carry_mask, weights * carry_mult, weights)
        op_mask = (targets >= 40) & (targets <= 44)
        weights = torch.where(op_mask, weights * op_mult, weights)
        eq_mask = targets == 44
        return weights

    # ── Forward pass ─────────────────────────────────────────────────

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None, use_cache: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch, seq_len = input_ids.shape
        device = input_ids.device

        arithmetic_bias = self._compute_arithmetic_bias(input_ids)

        x = self.token_embedding(input_ids)
        # Abacus: add digit-place embeddings for length generalisation
        if self.abacus_embedding is not None:
            places = self._compute_place_values(input_ids)
            x = x + self.abacus_embedding(places)

        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            x, layer_kv = layer(x, mask=None, arithmetic_bias=arithmetic_bias)
            kv_caches.append(layer_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        value: torch.Tensor | None = None
        if self.value_head is not None:
            value = self.value_head(x[:, -1, :])

        loss: torch.Tensor | None = None
        if targets is not None:
            token_weights = self._compute_token_weights(targets)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                label_smoothing=getattr(self.config, "label_smoothing", 0.0),
                reduction='none',
            )
            w = token_weights.reshape(-1)
            w[targets.reshape(-1) == -100] = 0.0
            loss = (loss * w).sum() / (w.sum() + 1e-8)

        return logits, loss, value

    # ── Generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 5,
        repetition_penalty: float = 1.0,
        clean_output: bool = True,
    ) -> str:
        self.eval()
        device = next(self.parameters()).device

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        seq = prompt_tensor

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(seq, use_cache=False)
            next_logits = logits[0, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0 and len(seq[0]) > 1:
                for prev_id in set(seq[0, -8:].tolist()):
                    next_logits[prev_id] /= repetition_penalty

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_k > 0:
                    top_indices = torch.topk(probs, top_k).indices
                    mask = torch.ones_like(probs, dtype=torch.bool)
                    mask[top_indices] = False
                    probs[mask] = 0.0
                    probs = probs / probs.sum()
                next_id = torch.multinomial(probs, 1).unsqueeze(0)
            else:
                next_id = next_logits.argmax(dim=-1).view(1, 1)

            seq = torch.cat([seq, next_id], dim=-1)

            if next_id.item() == tokenizer.eos_id:
                break

        raw = tokenizer.decode(seq[0].tolist())
        if clean_output:
            return self._clean_generated_output(raw)
        return raw

    def generate_step(
        self,
        input_ids: torch.Tensor,
        past_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        return self._generate_step_impl(input_ids, past_kv_caches)

    def _generate_step_impl(
        self,
        input_ids: torch.Tensor,
        past_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len = input_ids.shape
        device = input_ids.device

        arithmetic_bias = self._compute_arithmetic_bias(input_ids)

        x = self.token_embedding(input_ids)
        if self.abacus_embedding is not None:
            places = self._compute_place_values(input_ids)
            x = x + self.abacus_embedding(places)

        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past = past_kv_caches[i] if past_kv_caches else None
            x, kv = layer(x, mask=None, past_kv=past, arithmetic_bias=arithmetic_bias if past_kv_caches is None else None)
            new_kvs.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kvs

    @staticmethod
    def _clean_generated_output(raw: str) -> str:
        text = raw.replace("<EOS>", "").replace("<PAD>", "").replace("<BOS>", "").strip()
        if "=" not in text:
            return text[:4]
        after = text.split("=", 1)[-1].strip()
        return after[:8]

    # ── MoE aux loss ─────────────────────────────────────────────────

    def get_moe_aux_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            ff = layer.feed_forward
            if hasattr(ff, "last_aux_loss"):
                total = total + ff.last_aux_loss
        return total


def alphazero_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    value_pred: torch.Tensor | None = None,
    value_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute AlphaZero-style combined policy + value loss.

    Args:
        logits: Policy logits ``(batch, seq, vocab)``.
        targets: Target token IDs ``(batch, seq)``.
        mask: Optional padding mask.
        value_pred: Predicted value scores ``(batch, 1)``.
        value_target: Target value scores ``(batch, 1)``.

    Returns:
        Combined loss scalar.
    """
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction='mean',
    )

    vl = torch.tensor(0.0, device=logits.device)
    if value_pred is not None and value_target is not None:
        vl = F.mse_loss(value_pred.squeeze(-1), value_target.squeeze(-1))

    return ce + 0.5 * vl


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: A PyTorch module.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
