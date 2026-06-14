"""Transformer from scratch in PyTorch.
All architecture choices controlled by Config — edit config.py or use Model Config dashboard.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(dim: int, norm_type: str = 'rmsnorm') -> nn.Module:
    """Create a normalisation layer based on configuration.

    Args:
        dim: Feature dimension to normalise over.
        norm_type: One of ``'layernorm'`` or ``'rmsnorm'`` (default).

    Returns:
        An ``nn.Module`` that normalises along the last dimension.

    Raises:
        ValueError: If *norm_type* is not recognised (falls through to
            RMSNorm by default).
    """
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim, eps=1e-6)

    # RMSNorm (default)
    class RMSNorm(nn.Module):
        def __init__(self, d: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight

    return RMSNorm(dim)


def _make_activation(name: str = 'swiglu') -> callable[[torch.Tensor], torch.Tensor]:
    """Return an activation function by name.

    Args:
        name: One of ``'relu'``, ``'gelu'``, or ``'swiglu'`` (default).

    Returns:
        A callable activation function operating on tensors.
    """
    if name == 'relu':
        return F.relu
    elif name == 'gelu':
        return F.gelu
    return F.silu  # swiglu (default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension in two halves for RoPE.

    Args:
        x: Input tensor of shape ``(..., d)`` where ``d`` is even.

    Returns:
        Tensor of the same shape with the two halves rotated.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_pos_encoding(q: torch.Tensor, k: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor,
                        pos_type: str = 'rope') -> tuple[torch.Tensor, torch.Tensor]:
    """Apply position encoding to query and key tensors.

    Supports RoPE (default) and pass-through for ``'none'``.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine precomputed for RoPE.
        sin: Sine precomputed for RoPE.
        pos_type: ``'rope'`` or ``'none'``.

    Returns:
        Tuple ``(q, k)`` with position encoding applied.
    """
    if pos_type == 'none':
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
        """Initialise RoPE with inverse frequency bands.

        Args:
            dim: Head dimension (must be even).
            max_seq_len: Maximum supported sequence length (default: 512).
        """
        super().__init__()
        self.dim = dim
        inv_freq: torch.Tensor = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine position encodings.

        Args:
            seq_len: Sequence length to generate encodings for.
            device: Target device for the tensors.

        Returns:
            Tuple ``(cos, sin)`` each of shape ``(seq_len, dim)``.
        """
        position_ids = torch.arange(seq_len, device=device)
        freqs = torch.einsum('i,j->ij', position_ids.float(), self.inv_freq.float())
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


class LearnedPositionEmbedding(nn.Module):
    """Learnable absolute position embeddings."""

    def __init__(self, max_seq_len: int, dim: int) -> None:
        """Initialise learned position embeddings.

        Args:
            max_seq_len: Maximum number of position indices.
            dim: Embedding dimension.
        """
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Retrieve position embeddings for the first *seq_len* positions.

        Args:
            seq_len: Number of positions to retrieve.
            device: Target device for the tensor.

        Returns:
            Tensor of shape ``(seq_len, dim)``.
        """
        positions = torch.arange(seq_len, device=device)
        return self.pe(positions)


class Attention(nn.Module):
    """Multi-head self-attention with configurable position encoding and KV cache."""

    def __init__(self, config: Any) -> None:
        """Initialise attention projections and position encoding.

        Args:
            config: Configuration object with attributes ``d_model``,
                ``n_heads``, ``pos_encoding``, ``max_seq_len``, and
                ``dropout``.
        """
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert config.d_model % config.n_heads == 0, 'd_model must be divisible by n_heads'

        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

        self.pos_type = config.pos_encoding
        if self.pos_type == 'rope':
            self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len * 2)
        elif self.pos_type == 'learned':
            self.pos_embed = LearnedPositionEmbedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask (not used when ``past_kv`` is
                ``None`` — causal mask is built automatically).
            past_kv: Optional ``(k, v)`` tuple from previous generation
                steps for KV-cached decoding.

        Returns:
            Tuple ``(output, current_kv)`` where ``output`` has shape
            ``(batch, seq_len, d_model)`` and ``current_kv`` is the updated
            ``(k, v)`` tuple.
        """
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Position offset: when using KV cache, this token is at position len(past_kv) + current_pos
        kv_len = past_kv[0].size(2) if past_kv is not None else 0

        if self.pos_type == 'rope':
            cos, sin = self.rope(seq_len + kv_len, x.device)
            # Only apply to current positions
            cos = cos[kv_len:kv_len + seq_len].view(1, 1, seq_len, self.head_dim)
            sin = sin[kv_len:kv_len + seq_len].view(1, 1, seq_len, self.head_dim)
            q, k = _apply_pos_encoding(q, k, cos, sin, 'rope')
        elif self.pos_type == 'learned':
            pe = self.pos_embed(seq_len, x.device)
            x = x + pe
            q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Concatenate with past KV if available
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        current_kv = (k, v)

        # ── FlashAttention-2 via PyTorch 2.0+ scaled_dot_product_attention ──
        # Uses FlashAttention (CUDA), Memory-Efficient Attention, or fallback
        # based on hardware and input characteristics.
        if past_kv is not None:
            # KV cache scenario: build shifted causal mask manually
            q_len = seq_len
            total_len = k.size(2)
            causal = torch.tril(torch.ones(q_len, total_len, device=x.device))
            for i in range(q_len):
                causal[i, kv_len + i + 1:] = 0
            attn_mask = causal.view(1, 1, q_len, total_len).bool()
        else:
            # No KV cache: use built-in causal masking
            attn_mask = None

        # SDPA: is_causal=True when no explicit mask and no KV cache
        use_causal = (mask is None and past_kv is None)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out), current_kv


class FeedForward(nn.Module):
    """Feed-forward network with configurable activation (SwiGLU-style)."""

    def __init__(self, config: Any) -> None:
        """Initialise the feed-forward block.

        Args:
            config: Configuration object with attributes ``d_model``,
                ``d_ff``, and ``activation``.
        """
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.act = _make_activation(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            Tensor of the same shape as input.
        """
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """One transformer decoder block with pre-normalisation and optional KV cache."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Initialise the transformer block.

        Args:
            config: Configuration object with all model hyperparameters.
            layer_idx: Index of this layer in the stack (currently unused
                but available for future use).
        """
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = _make_norm(config.d_model, config.norm_type)
        self.ffn_norm = _make_norm(config.d_model, config.norm_type)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply self-attention and feed-forward with residual connections.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask.
            past_kv: Optional ``(k, v)`` tuple from previous generation steps.

        Returns:
            Tuple ``(output, current_kv)``.
        """
        attn_out, current_kv = self.attention(self.attention_norm(x), mask, past_kv=past_kv)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x, current_kv


class MathTransformer(nn.Module):
    """Tiny autoregressive transformer — architecture set by Config.

    Supports:
    - Multi-head self-attention with RoPE or learned position encodings.
    - Pre-normalisation (RMSNorm or LayerNorm).
    - SwiGLU / ReLU / GELU feed-forward blocks.
    - KV-cached generation for efficient decoding.
    - Optional AlphaZero-style value head.
    """

    def __init__(self, config: Any) -> None:
        """Initialise the transformer model.

        Args:
            config: Configuration object with all model hyperparameters
                (``vocab_size``, ``d_model``, ``n_layers``, etc.).
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])
        self.norm = _make_norm(config.d_model, config.norm_type)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # AlphaZero-style Value Head: outputs scalar -1 to +1 (intuition)
        if getattr(config, 'use_value_head', False):
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise weights for ``Linear`` and ``Embedding`` modules.

        Uses the initialisation strategy from ``self.config.weight_init``
        (``'normal'``, ``'xavier'``, or ``'kaiming'``).

        Args:
            module: A PyTorch module whose weights will be initialised.
        """
        init_type = getattr(self.config, 'weight_init', 'normal')
        init_std = getattr(self.config, 'init_std', 0.02)
        if isinstance(module, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            else:  # normal
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if init_type == 'normal':
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            # xavier/kaiming for embeddings often just uses uniform

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal (upper-triangular) attention mask.

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Boolean mask of shape ``(1, 1, seq_len, seq_len)``.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None,
                use_cache: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward pass through the full transformer stack.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.
            targets: Optional target ID tensor of the same shape for
                computing cross-entropy loss.
            use_cache: If ``True``, returns per-layer KV caches (not used
                in this training path — retained for API compatibility).

        Returns:
            Tuple ``(logits, loss, value)``:
            - **logits**: Raw logits of shape ``(batch, seq_len, vocab_size)``.
            - **loss**: Cross-entropy loss scalar, or ``None`` if *targets*
              is ``None``.
            - **value**: Value-head scalar ``(batch, 1)`` or ``None`` if
              ``use_value_head`` is ``False``.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids)
        mask = self._causal_mask(seq_len, device)

        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            x, layer_kv = layer(x, mask)
            kv_caches.append(layer_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        # Value head: intuition score for the current state (-1 to +1)
        value: torch.Tensor | None = None
        if self.value_head is not None:
            value = self.value_head(x[:, -1, :])  # use last token's hidden state

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=getattr(self.config, 'label_smoothing', 0.0),
            )

        return logits, loss, value

    @torch.no_grad()
    def generate_step(self, input_ids: torch.Tensor,
                      past_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
                      ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Single forward step for generation with KV cache.

        Args:
            input_ids: Token IDs to process of shape ``(batch, seq_len)``
                — usually just the latest token.
            past_kv_caches: List of ``(K, V)`` tuples from previous steps,
                one per layer, or ``None`` for the first step.

        Returns:
            Tuple ``(logits, new_kv_caches)`` where logits are for every
            position in the input and KV caches include both past and
            current keys/values.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids)

        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past = past_kv_caches[i] if past_kv_caches else None
            x, kv = layer(x, mask=None, past_kv=past)
            new_kvs.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kvs

    @torch.no_grad()
    def generate(self, tokenizer: Any, prompt: str, max_new_tokens: int = 20,
                 temperature: float = 1.0, top_k: int = 5) -> str:
        """Generate text from a prompt with KV cache for efficiency.

        KV cache avoids recomputing attention over previous tokens on each
        step, making generation 5-10× faster for long sequences.

        Args:
            tokenizer: A ``MathTokenizer``-like object with ``encode``,
                ``decode``, and ``eos_id`` attributes.
            prompt: Input prompt string.
            max_new_tokens: Maximum number of new tokens to generate
                (default: 20).
            temperature: Sampling temperature (``0.0`` = greedy, default:
                ``1.0``).
            top_k: Top-K filtering threshold (default: 5).

        Returns:
            Decoded generated string (prompt + continuation).
        """
        self.eval()
        device = next(self.parameters()).device

        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)

        # Step 1: Process the full prompt to build initial KV cache
        prompt_logits, kv_caches = self.generate_step(input_ids, past_kv_caches=None)
        next_logits = prompt_logits[0, -1, :]

        # Sample first token from prompt
        if temperature < 0.01:
            next_id = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_vals[-1]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).unsqueeze(0)

        # next_id shape: (1, 1) — matches input_ids (1, seq_len)
        all_ids = torch.cat([input_ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_id:
            return tokenizer.decode(all_ids[0].tolist())

        # Step 2: Generate remaining tokens one at a time using KV cache
        for _ in range(max_new_tokens - 1):
            # Only feed the last token, not the whole sequence
            logits, kv_caches = self.generate_step(next_id, past_kv_caches=kv_caches)
            next_logits = logits[0, -1, :]

            if all_ids.size(1) >= self.config.max_seq_len:
                break  # Don't exceed max sequence length

            if temperature < 0.01:
                next_id = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < top_vals[-1]] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).unsqueeze(0)

            all_ids = torch.cat([all_ids, next_id], dim=1)

            if next_id.item() == tokenizer.eos_id:
                break

        return tokenizer.decode(all_ids[0].tolist())


def alphazero_loss(policy_logits: torch.Tensor, value_pred: torch.Tensor | None,
                   targets: torch.Tensor,
                   value_target_z: torch.Tensor | None,
                   config: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
    """AlphaZero-style dual loss: policy cross-entropy + value MSE.

    Args:
        policy_logits: Raw logits from ``lm_head`` of shape
            ``(batch, seq_len, vocab)``.
        value_pred: Value head output of shape ``(batch, 1)``, range
            ``-1`` to ``+1``, or ``None``.
        targets: Target token indices of shape ``(batch, seq_len)`` for
            policy loss.
        value_target_z: Target value tensor — ``+1`` (correct),
            ``-1`` (wrong), ``0`` (draw), or ``None``.
        config: Optional configuration object (for ``label_smoothing``).

    Returns:
        Tuple ``(policy_loss, value_loss)``, each a scalar tensor.
    """
    # Policy loss: cross-entropy on tokens
    policy_loss = F.cross_entropy(
        policy_logits.view(-1, policy_logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
        label_smoothing=getattr(config, 'label_smoothing', 0.0) if config else 0.0,
    )

    # Value loss: MSE between predicted intuition and actual outcome
    if value_pred is not None and value_target_z is not None:
        vp = value_pred.squeeze(-1)  # (batch,) if input was (batch, 1)
        if vp.dim() != value_target_z.dim():
            value_target_z = value_target_z.view_as(vp)
        value_loss = F.mse_loss(vp, value_target_z)
    else:
        value_loss = torch.tensor(0.0, device=policy_logits.device)

    return policy_loss, value_loss


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: A PyTorch ``nn.Module``.

    Returns:
        Total number of parameters that require gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer

    cfg = Config()
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size

    model = MathTransformer(cfg)
    print(f'Model params: {count_parameters(model):,}')
    print(f'Arch: d={cfg.d_model} L={cfg.n_layers} ff={cfg.d_ff} heads={cfg.n_heads}')
    print(f'Act: {cfg.activation}  Norm: {cfg.norm_type}  Pos: {cfg.pos_encoding}')
    print(f'Init: {cfg.weight_init}  Tie: {cfg.tie_embeddings}')

    ids = tok.encode('12+34=46', add_special_tokens=True)
    x = torch.tensor([ids])
    logits, loss, value = model(x, x)
    print(f'Logits shape: {logits.shape}')
    print(f'Loss (random init): {loss.item():.4f}')
    print(f'Value (random init): {value}')
