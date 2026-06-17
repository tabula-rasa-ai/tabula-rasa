"""
Captum-based interpretability for Tabula Rasa's MathTransformer.

Provides:
- analyze_prediction(): token-level attribution using LayerIntegratedGradients
- visualize_attention(): extract attention weights per layer/head
- generate_html_report(): HTML visualization with colored token highlights
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Ensure project root and src/ are on the path
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "src"))

try:
    import captum
    from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
except ImportError:
    captum = None  # type: ignore[assignment]

# ── Helpers ─────────────────────────────────────────────────────


def _get_embedding_layer(model: nn.Module) -> nn.Embedding:
    """Return the token embedding layer from a MathTransformer."""
    return model.token_embedding


def _construct_input_ref(
    token_ids: list[int],
    ref_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build input and reference tensors for integrated gradients.

    Args:
        token_ids: List of input token IDs.
        ref_token_id: Token ID to use as baseline (default 0 = <PAD>).

    Returns:
        (input_tensor, ref_tensor) each of shape (1, seq_len).
    """
    inp = torch.tensor([token_ids], dtype=torch.long)
    ref = torch.full_like(inp, ref_token_id)
    return inp, ref


def _forward_wrapper(
    model: nn.Module,
    target_id: int | None = None,
) -> callable:
    """Wrap model.forward so it returns logits for a single target position.

    Args:
        model: A MathTransformer instance.
        target_id: If set, return logit for this token ID at the last position.
                   If None, return the logit for the argmax at the last position.

    Returns:
        A function(input_embeds) -> torch.Tensor scalar suitable for Captum.
    """

    def wrapped(input_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds: (batch, d_model) or (batch, seq_len, d_model)
        # Captum passes single positions (2D) during integrated gradients
        if input_embeds.dim() == 2:
            batch, d_model = input_embeds.shape
            seq_len = 1
            x = input_embeds.unsqueeze(1)  # (batch, 1, d_model)
        else:
            batch, seq_len, d_model = input_embeds.shape
            x = input_embeds

        # Pass through each transformer block
        for layer in model.layers:
            x, _ = layer(x)

        # Final norm
        x = model.norm(x)
        # LM head
        logits = model.lm_head(x)  # (batch, seq_len, vocab_size)

        # Target: last position
        last_logits = logits[:, -1, :]  # (batch, vocab_size)

        if target_id is not None:
            return last_logits[:, target_id].sum()
        # Default: argmax
        pred_id = last_logits.argmax(dim=-1)
        return last_logits.gather(1, pred_id.unsqueeze(1)).sum()

    return wrapped


# ── Public API ──────────────────────────────────────────────────


def analyze_prediction(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    target_id: int | None = None,
) -> dict:
    """Analyze which input tokens most influence the model's prediction.

    Uses Captum's LayerIntegratedGradients on the token embedding layer.

    Args:
        model: A MathTransformer model (loaded and in eval mode).
        tokenizer: A MathTokenizer instance.
        prompt: Input string (e.g. '12+34=').
        target_id: Optional target token ID to score against. If None,
                   uses the model's argmax prediction at the last position.

    Returns:
        Dict with keys:
            - 'tokens': list of token strings
            - 'attributions': list of {token, score, abs_score} dicts
            - 'scores_raw': raw attribution scores tensor
    """
    if captum is None:
        return {
            "tokens": list(prompt),
            "attributions": [{"token": t, "score": 0.0, "abs_score": 0.0} for t in prompt],
            "scores_raw": [],
            "warning": "Captum not installed. Run: pip install captum",
        }

    # Tokenize
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor, ref_tensor = _construct_input_ref(
        token_ids, ref_token_id=tokenizer.pad_id
    )

    # Build forward function
    forward_fn = _forward_wrapper(model, target_id)

    # LayerIntegratedGradients on the embedding layer
    embedding_layer = _get_embedding_layer(model)
    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    # Compute attributions
    result = lig.attribute(
        inputs=input_tensor,
        baselines=ref_tensor,
        target=None,
        n_steps=50,
        return_convergence_delta=True,
    )
    attributions = result[0] if isinstance(result, tuple) else result
    _delta = result[1] if len(result) >= 2 else None

    # attributions shape: (1, seq_len, d_model) — sum over embedding dim
    scores = attributions.squeeze(0).sum(dim=-1)  # (seq_len,)
    scores = scores.detach().cpu()

    # Build token list
    token_strs = [
        tokenizer.itos.get(tid, f"<{tid}>") for tid in token_ids
    ]

    attrib_list = []
    for i, (token_str, score_val) in enumerate(zip(token_strs, scores)):
        attrib_list.append({
            "position": i,
            "token": token_str,
            "score": round(float(score_val), 6),
            "abs_score": round(abs(float(score_val)), 6),
        })

    return {
        "tokens": token_strs,
        "attributions": attrib_list,
        "scores_raw": scores.tolist(),
        "input_ids": token_ids,
    }


def visualize_attention(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
) -> dict:
    """Extract attention weights from every layer and head.

    For each transformer block, runs a forward pass and captures the
    attention scores before softmax, then returns per-layer/per-head matrices.

    Args:
        model: A MathTransformer model in eval mode.
        tokenizer: A MathTokenizer instance.
        prompt: Input string.

    Returns:
        Dict of {layer_idx: {head_idx: attention_matrix (list of lists)}}
        where attention_matrix[q][k] = weight from query pos q to key pos k.
    """
    from tabula_rasa.model import _apply_pos_encoding  # safe import now

    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    x = torch.tensor([token_ids], dtype=torch.long)
    batch, seq_len = x.shape

    # Embed
    embeds = model.token_embedding(x)

    # Hook: capture attention weights from each layer
    attn_store: dict[int, list[torch.Tensor]] = {}

    def _save_attn(layer_idx: int) -> callable:
        def hook(module, inputs, output):
            # output is (attended, kv_cache)
            # Recompute attention weights manually
            hidden = inputs[0]  # (batch, seq_len, d_model)
            b, sl, _ = hidden.shape
            # Store q, k for this forward to compute weights
            q = module.wq(hidden).view(b, sl, module.n_heads, module.head_dim).transpose(1, 2)
            k = module.wk(hidden).view(b, sl, module.n_heads, module.head_dim).transpose(1, 2)

            if module.pos_type == "rope":
                cos, sin = module.rope(sl, hidden.device)
                cos = cos.view(1, 1, sl, module.head_dim)
                sin = sin.view(1, 1, sl, module.head_dim)
                q, k = _apply_pos_encoding(q, k, cos, sin, "rope")

            scale = module.head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            # Causal mask
            causal = torch.tril(torch.ones(sl, sl, device=hidden.device)).view(1, 1, sl, sl)
            attn_scores = attn_scores.masked_fill(causal == 0, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_store.setdefault(layer_idx, []).append(attn_weights.detach().cpu())
        return hook

    # Register forward hooks on each Attention module
    hooks = []
    for i, layer in enumerate(model.layers):
        hook = layer.attention.register_forward_hook(_save_attn(i))
        hooks.append(hook)

    # Run forward pass
    with torch.no_grad():
        model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Format results
    result: dict = {}
    for layer_idx, attn_list in attn_store.items():
        if not attn_list:
            continue
        # Take the last captured attention (there should be one)
        attn = attn_list[-1]  # (1, n_heads, seq_len, seq_len)
        n_heads = attn.size(1)
        head_dict: dict[str, list[list[float]]] = {}
        for h in range(n_heads):
            mat = attn[0, h].tolist()  # (seq_len, seq_len)
            head_dict[str(h)] = mat
        result[str(layer_idx)] = head_dict

    # Token labels for reference
    token_labels = [tokenizer.itos.get(tid, f"<{tid}>") for tid in token_ids]
    result["_tokens"] = token_labels

    return result


def generate_html_report(analysis: dict) -> str:
    """Generate an HTML visualization of token attributions.

    Tokens with higher absolute attribution scores are highlighted more
    intensely (green for positive influence, red for negative influence).

    Args:
        analysis: Dict from analyze_prediction() containing 'tokens'
                  and 'attributions'.

    Returns:
        HTML string with colored token highlights.
    """
    tokens = analysis.get("tokens", [])
    attributions = analysis.get("attributions", [])

    if not tokens:
        return "<p>No tokens to visualize.</p>"

    # Normalize scores between 0 and 1 for colour intensity
    abs_scores = [abs(a["score"]) for a in attributions] if attributions else [0] * len(tokens)
    max_abs = max(abs_scores) if abs_scores and max(abs_scores) > 0 else 1.0
    normalized = [s / max_abs for s in abs_scores]

    # Build HTML
    html_parts = [
        '<div style="font-family: monospace; padding: 16px; background: #0f172a; border-radius: 8px; '
        'line-height: 2.5;">'
    ]

    for i, token_str in enumerate(tokens):
        score = attributions[i]["score"] if i < len(attributions) else 0.0
        norm_intensity = normalized[i] if i < len(normalized) else 0.0

        if score > 0:
            # Green: positive influence
            intensity = int(55 + 200 * norm_intensity)
            bg = f"rgba(34, 197, 94, {0.15 + 0.6 * norm_intensity:.2f})"
            border = f"rgba(34, 197, 94, {0.4 + 0.6 * norm_intensity:.2f})"
        elif score < 0:
            # Red: negative influence
            bg = f"rgba(239, 68, 68, {0.15 + 0.6 * norm_intensity:.2f})"
            border = f"rgba(239, 68, 68, {0.4 + 0.6 * norm_intensity:.2f})"
        else:
            bg = "#1e293b"
            border = "#334155"

        display_token = token_str.replace("<", "&lt;").replace(">", "&gt;")
        if display_token == " ":
            display_token = "&nbsp;"

        html_parts.append(
            f'<span style="background:{bg}; border:1px solid {border}; '
            f'border-radius:4px; padding:4px 8px; margin:2px; '
            f'color:#f1f5f9; display:inline-block; '
            f'title="{token_str}: {score:.4f}">'
            f'{display_token}'
            f'<span style="font-size:0.65em; color:#94a3b8; margin-left:4px;">'
            f'{score:.3f}</span></span>'
        )

    html_parts.append("</div>")

    # Add a legend
    html_parts.append(
        '<div style="margin-top: 12px; font-size: 0.8em; color: #94a3b8; '
        'font-family: sans-serif;">'
        '<span style="display:inline-block; width:12px; height:12px; '
        'background:rgba(34,197,94,0.5); border-radius:2px; margin-right:4px;"></span> '
        'Positive influence &nbsp;&nbsp;'
        '<span style="display:inline-block; width:12px; height:12px; '
        'background:rgba(239,68,68,0.5); border-radius:2px; margin-right:4px;"></span> '
        'Negative influence &nbsp;&nbsp;'
        '<span>Number = attribution score</span>'
        '</div>'
    )

    return "\n".join(html_parts)
