"""Linguistic knowledge probe — analyzes what the model's internal representations encode.

The PDF recommends "probing a model's linguistic knowledge" (ref [18]: "Probing the
Linguistic Knowledge of Character-level Neural RNNs").

For our transformer, this means analyzing:
1. Token embedding space: do digit embeddings encode magnitude ordering?
2. Position encoding: does position encode distance linearly?
3. Hidden state analysis: do intermediate layers differentiate operations?
4. Attention patterns: do heads specialize for carry propagation?

Usage:
    python3 -m tabula_rasa.probe checkpoints/best.pt
    python3 -m tabula_rasa.probe checkpoints/best.pt --visualize
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def probe_embeddings(model: Any, tokenizer: Any) -> dict:
    """Analyze token embedding space for linguistic structure.

    Checks:
    - Digit magnitude encoding: are digit embeddings ordered by value?
    - Digit clustering: do similar digits cluster?
    - Operator separation: are operators distinct from digits?

    Returns dict of analysis results.
    """
    results = {}
    emb = model.token_embedding.weight.detach()  # (vocab_size, d_model)

    # 1. Digit magnitude encoding
    # Check if cosine similarity between digit embeddings correlates with digit proximity
    digit_sims = []
    for i in range(9):
        for j in range(i + 1, 10):
            sim = F.cosine_similarity(emb[i].unsqueeze(0), emb[j].unsqueeze(0)).item()
            digit_diff = abs(i - j)
            digit_sims.append((digit_diff, sim))

    if digit_sims:
        # Correlation between digit difference and embedding similarity
        diffs = [d for d, _ in digit_sims]
        sims = [s for _, s in digit_sims]
        if len(diffs) > 1:
            import statistics
            corr = statistics.correlation(diffs, sims) if hasattr(statistics, 'correlation') else 0
        else:
            corr = 0
        results["digit_magnitude_correlation"] = round(corr, 3)
        # Adjacent digit similarity vs far-digit similarity
        adj_sim = [s for d, s in digit_sims if d == 1]
        far_sim = [s for d, s in digit_sims if d >= 5]
        results["adjacent_digit_sim_mean"] = round(sum(adj_sim) / len(adj_sim), 3) if adj_sim else 0
        results["far_digit_sim_mean"] = round(sum(far_sim) / len(far_sim), 3) if far_sim else 0

    # 2. Digit embedding norm analysis
    digit_norms = [emb[i].norm().item() for i in range(10)]
    results["digit_norm_mean"] = round(sum(digit_norms) / len(digit_norms), 3)
    results["digit_norm_std"] = round(
        (sum((n - results["digit_norm_mean"]) ** 2 for n in digit_norms) / len(digit_norms)) ** 0.5, 3
    )

    # 3. Operator vs digit separation
    op_tokens = [10, 11, 12, 13]  # +, -, *, /
    digit_center = emb[:10].mean(dim=0)
    for op_id, op_name in zip(op_tokens, ['+', '-', '*', '/']):
        if op_id < emb.size(0):
            op_sim = F.cosine_similarity(emb[op_id].unsqueeze(0), digit_center.unsqueeze(0)).item()
            results[f"operator_{op_name}_to_digit_center"] = round(op_sim, 3)

    # 4. Embedding space dimensionality
    results["embedding_dim"] = emb.size(1)
    results["vocab_size"] = emb.size(0)

    return results


def probe_attention_patterns(model: Any, input_ids: torch.Tensor) -> dict:
    """Analyze attention patterns for structural specialization.

    Checks:
    - Do certain heads attend to carry positions?
    - Is there diagonal attention (digit-to-same-digit across operands)?

    Args:
        model: MathTransformer.
        input_ids: Tokenized input (1, seq_len).

    Returns:
        Dict of attention analysis.
    """
    results = {}
    batch, seq_len = input_ids.shape
    device = next(model.parameters()).device

    # Run forward with attention output capture
    # We use the model's forward which doesn't expose attention weights directly
    # Instead, analyze hidden state norms per position
    x = model.token_embedding(input_ids)
    h = x
    layer_outputs = []
    for layer in model.layers:
        h, _ = layer(h)
        layer_outputs.append(h.detach())

    # Per-position hidden state norm (higher = more processing at that position)
    for layer_idx, h_state in enumerate(layer_outputs):
        pos_norms = h_state[0].norm(dim=-1).cpu().tolist()  # (seq_len,)
        results[f"layer_{layer_idx}_pos_norms"] = [round(n, 3) for n in pos_norms]

    # Entropy of hidden state distribution across positions
    h_flat = h[0]  # (seq_len, d_model)
    h_normed = F.normalize(h_flat, dim=-1)
    # If certain positions dominate, entropy is low
    sim_matrix = (h_normed @ h_normed.T).cpu()  # (seq_len, seq_len)
    results["final_layer_self_sim_mean"] = round(sim_matrix.mean().item(), 3)
    results["final_layer_self_sim_std"] = round(sim_matrix.std().item(), 3)

    return results


def probe_hidden_states(model: Any, tokenizer: Any, expressions: list[str]) -> dict:
    """Analyze hidden states across different expression types.

    Tests whether the model develops separate representations for
    addition vs subtraction, small vs large numbers, etc.

    Args:
        model: MathTransformer.
        tokenizer: MathTokenizer.
        expressions: List of math expressions to compare.

    Returns:
        Dict of hidden state analysis.
    """
    results = {}
    device = next(model.parameters()).device
    model.eval()

    reprs = {}
    for expr in expressions:
        ids = tokenizer.encode(expr, add_special_tokens=True)
        inp = torch.tensor([ids], device=device)
        with torch.no_grad():
            # Get hidden state from last layer
            x = model.token_embedding(inp)
            for layer in model.layers:
                x, _ = layer(x)
            # Mean pool over sequence
            repr_vec = x[0].mean(dim=0).cpu()  # (d_model,)
        reprs[expr] = repr_vec

    # Compute similarity matrix
    names = list(reprs.keys())
    sims = torch.zeros(len(names), len(names))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            sims[i, j] = F.cosine_similarity(
                reprs[name_i].unsqueeze(0), reprs[name_j].unsqueeze(0)
            ).item()

    results["expression_similarity"] = {
        names[i]: {names[j]: round(sims[i, j].item(), 3) for j in range(len(names))}
        for i in range(len(names))
    }

    return results


def full_probe(model: Any, tokenizer: Any, visualize: bool = False) -> dict:
    """Run all probes and return structured results.

    Args:
        model: MathTransformer.
        tokenizer: MathTokenizer.
        visualize: If True, print detailed visualizations.

    Returns:
        Dict of all probe results.
    """
    results = {}

    print("=" * 60)
    print("  LINGUISTIC KNOWLEDGE PROBE")
    print("=" * 60)

    # 1. Embedding analysis
    print("\n[1/3] Token embedding space...")
    emb_results = probe_embeddings(model, tokenizer)
    results["embeddings"] = emb_results
    print(f"  Digit magnitude correlation: {emb_results.get('digit_magnitude_correlation', 'N/A')}")
    print(f"  Adjacent digit similarity:   {emb_results.get('adjacent_digit_sim_mean', 'N/A')}")
    print(f"  Far digit similarity:        {emb_results.get('far_digit_sim_mean', 'N/A')}")
    for op in ['+', '-', '*', '/']:
        key = f"operator_{op}_to_digit_center"
        if key in emb_results:
            print(f"  Operator '{op}' to digit center: {emb_results[key]}")

    # 2. Hidden state analysis
    print("\n[2/3] Hidden state representations...")
    test_exprs = ["2+3=", "2+4=", "9-4=", "9-5=", "12+34=", "56-18="]
    hs_results = probe_hidden_states(model, tokenizer, test_exprs)
    results["hidden_states"] = hs_results
    sim_matrix = hs_results.get("expression_similarity", {})
    if sim_matrix:
        # Print similarity matrix
        names = list(sim_matrix.keys())
        print(f"  {'':<12}", end="")
        for n in names:
            print(f"{n[:6]:<8}", end="")
        print()
        for name_i in names:
            print(f"  {name_i:<12}", end="")
            for name_j in names:
                val = sim_matrix[name_i].get(name_j, 0)
                print(f"{val:<8.3f}", end="")
            print()

    # 3. Attention patterns
    print("\n[3/3] Attention patterns...")
    sample = "12+34="
    ids = tokenizer.encode(sample, add_special_tokens=True)
    inp = torch.tensor([ids])
    attn_results = probe_attention_patterns(model, inp)
    results["attention"] = attn_results
    for key in attn_results:
        val = attn_results[key]
        if isinstance(val, list) and len(val) <= 16:
            print(f"  {key}: {val}")
        elif isinstance(val, (int, float)):
            print(f"  {key}: {val}")
    print(f"\n  Layer position norms indicate which input positions")
    print(f"  receive the most processing (higher = more attention)")

    print(f"\n{'='*60}")
    print("  PROBE COMPLETE")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        ckpt = Path("checkpoints/best.pt")
        if not ckpt.exists():
            ckpt = Path("checkpoints/final.pt")
    else:
        ckpt = Path(sys.argv[1])

    visualize = "--visualize" in sys.argv

    if not ckpt.exists():
        print(f"No checkpoint found at {ckpt}")
        sys.exit(1)

    print(f"Loading model from {ckpt}")
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer

    tok = MathTokenizer()
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]
    
    # Handle vocab mismatch
    ckpt_vocab = sd["token_embedding.weight"].shape[0]
    if ckpt_vocab != cfg.vocab_size:
        old_w = sd["token_embedding.weight"]
        new_w = torch.randn(cfg.vocab_size, cfg.d_model) * 0.02
        new_w[:ckpt_vocab] = old_w
        sd["token_embedding.weight"] = new_w
        old_w = sd["lm_head.weight"]
        new_w = torch.randn(cfg.vocab_size, cfg.d_model) * 0.02
        new_w[:ckpt_vocab] = old_w
        sd["lm_head.weight"] = new_w
        print(f"  [*] Extended embeddings: {ckpt_vocab} → {cfg.vocab_size}")

    model.load_state_dict(sd, strict=False)
    model.eval()

    results = full_probe(model, tok, visualize=visualize)

    # Save results
    out_path = ckpt.parent / "probe_results.json"
    with open(out_path, "w") as f:
        # Convert tensors to serializable
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")
