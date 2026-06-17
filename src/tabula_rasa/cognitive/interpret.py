"""Mechanistic interpretability for Tabula Rasa's arithmetic transformer.

Probes a trained MathTransformer to answer:
- Which attention heads carry digit information between columns?
- Do specific neurons act as "carry registers"?
- How does the model construct the "borrow" circuit in subtraction?

Usage:
    python3 -m egefalos.interpret specialists/math/add/best.pt --op add
    python3 -m egefalos.interpret specialists/math/sub/best.pt --op sub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer

# ─── Hooks ──────────────────────────────────────────────────────────


class AttentionCapture:
    """Forward hook that captures attention weights and outputs per layer."""

    def __init__(self):
        self.attentions: list[torch.Tensor] = []  # per-head attention weights
        self.outputs: list[torch.Tensor] = []  # post-attention hidden states

    def hook_fn(self, module, inputs, output):
        """Capture the attention output and weights.

        For Attention.forward(), output is (attended, kv_cache).
        We capture the attended tensor (before residual add).
        """
        self.outputs.append(output[0].detach())

        # Extract attention weights via a separate forward pass on q, k
        # SDPA doesn't return attention weights, so we compute them manually
        # if the module has stored q and k
        if hasattr(module, "_last_q") and hasattr(module, "_last_k"):
            q, k = module._last_q, module._last_k
            scale = module.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.tril(
                torch.ones(q.size(-2), k.size(-2), device=q.device)
            ).view(1, 1, q.size(-2), k.size(-2))
            attn = attn.masked_fill(causal == 0, float("-inf"))
            attn_weights = F.softmax(attn, dim=-1)
            self.attentions.append(attn_weights.detach())


class ActivationCapture:
    """Forward hook that captures neuron activations after the gate/up projection."""

    def __init__(self):
        self.activations: list[torch.Tensor] = []

    def hook_fn(self, module, inputs, output):
        self.activations.append(output.detach())


def patch_model_for_interpretability(model: MathTransformer):
    """Register hooks on all attention and FFN layers.

    Returns dict of {layer_idx: {'attn_weights': list, 'attn_outputs': list, 'ffn_acts': list}}
    """
    captures = {}

    for i, layer in enumerate(model.layers):
        # Store attention data per layer
        layer_data = {
            "attn_weights": [],  # list of (n_heads, seq_len, seq_len) tensors
            "attn_outputs": [],  # list of (1, seq_len, d_model) tensors
            "ffn_acts": [],      # list of (1, seq_len, d_ff) tensors
        }

        # Monkey-patch Attention.forward to capture weights
        original_attn_forward = layer.attention.forward

        def make_attn_forward(layer_idx, orig_fn, store):
            def patched_forward(x, mask=None, past_kv=None):
                batch, seq_len, _ = x.shape
                attn = model.layers[layer_idx].attention

                # Compute q, k, v manually
                q = attn.wq(x).view(batch, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
                k = attn.wk(x).view(batch, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
                v = attn.wv(x).view(batch, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)

                kv_len = past_kv[0].size(2) if past_kv is not None else 0
                if attn.pos_type == "rope":
                    cos, sin = attn.rope(seq_len + kv_len, x.device)
                    cos = cos[kv_len : kv_len + seq_len].view(1, 1, seq_len, attn.head_dim)
                    sin = sin[kv_len : kv_len + seq_len].view(1, 1, seq_len, attn.head_dim)
                    from tabula_rasa.model import _apply_pos_encoding
                    q_pe, k_pe = _apply_pos_encoding(q, k, cos, sin, "rope")
                else:
                    q_pe, k_pe = q, k

                if past_kv is not None:
                    k_pe = torch.cat([past_kv[0], k_pe], dim=2)
                    v = torch.cat([past_kv[1], v], dim=2)
                current_kv = (k_pe, v)

                # Manual attention to capture weights
                scale = attn.head_dim ** -0.5
                attn_scores = torch.matmul(q_pe, k_pe.transpose(-2, -1)) * scale

                # Causal mask
                q_len, kv_len_full = attn_scores.size(-2), attn_scores.size(-1)
                if past_kv is not None:
                    causal = torch.tril(torch.ones(q_len, kv_len_full, device=x.device))
                    for j in range(q_len):
                        causal[j, kv_len + j + 1:] = 0
                    attn_mask = causal.view(1, 1, q_len, kv_len_full).bool()
                else:
                    causal = torch.tril(torch.ones(q_len, q_len, device=x.device))
                    attn_mask = causal.view(1, 1, q_len, q_len).bool()

                attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
                attn_weights = F.softmax(attn_scores, dim=-1)

                # Store attention weights and output
                store["attn_weights"].append(attn_weights[0].detach())  # (n_heads, q_len, kv_len)

                # Compute output
                attn_out = torch.matmul(attn_weights, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
                out = attn.wo(attn_out)
                store["attn_outputs"].append(out.detach())

                return out, current_kv

            return patched_forward

        layer.attention.forward = make_attn_forward(i, original_attn_forward, layer_data)

        # FFN activation capture via forward hook
        original_ffn_forward = layer.feed_forward.forward

        def make_ffn_forward(store):
            def hooked_forward(x):
                # Store pre-down-projection activations (from w1 * w3)
                # Capture the gate*up values
                gate = layer.feed_forward.act(layer.feed_forward.w1(x))
                up = layer.feed_forward.w3(x)
                hidden = gate * up
                store["ffn_acts"].append(hidden.detach())
                return layer.feed_forward.w2(hidden)
            return hooked_forward

        layer.feed_forward.forward = make_ffn_forward(layer_data)

        captures[i] = layer_data

    return captures


# ─── Analysis ───────────────────────────────────────────────────────


def run_problem(model, tok, expr: str, captures: dict) -> dict:
    """Run a single problem and return captures."""
    # Clear previous captures
    for c in captures.values():
        c["attn_outputs"] = []
        c["attn_weights"] = []
        c["ffn_acts"] = []

    ids = tok.encode(expr + "=", add_special_tokens=True)
    x = torch.tensor([ids])
    with torch.no_grad():
        logits, _, _ = model(x)

    return {
        "logits": logits,
        "attentions": {i: c["attn_weights"] for i, c in captures.items()},
        "ffn_acts": {i: c["ffn_acts"] for i, c in captures.items()},
    }


def find_carry_neurons(
    model, tok, captures: dict, num_problems: int = 50
) -> dict:
    """Find neurons that activate differently on carry vs no-carry problems.

    For addition: carry problems (e.g., 5+7=12 where 5+7>=10) vs no-carry (2+3=5).
    Compares FFN activations across layers to find carry-sensitive neurons.
    """
    import random

    carry_acts: list[torch.Tensor] = []
    no_carry_acts: list[torch.Tensor] = []

    for _ in range(num_problems):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        is_carry = (a + b) >= 10

        # Clear captures
        for c in captures.values():
            c["attn_outputs"] = []
            c["attn_weights"] = []
            c["ffn_acts"] = []

        prompt = f"{a}+{b}="
        ids = tok.encode(prompt, add_special_tokens=True)
        x = torch.tensor([ids])
        with torch.no_grad():
            model(x)

        # Collect FFN activations (post-gate*up, pre-down projection)
        batch_acts = []
        for i in range(model.config.n_layers):
            acts = captures[i]["ffn_acts"]
            if acts:
                # acts is list of (1, seq_len, d_ff) tensors
                # Take the activation at the output position (after `=`)
                # The last position in the sequence
                batch_acts.append(acts[-1][0, -1, :].unsqueeze(0))

        if batch_acts:
            combined = torch.cat(batch_acts, dim=-1)  # concatenate all layers' activations
            if is_carry:
                carry_acts.append(combined)
            else:
                no_carry_acts.append(combined)

    if not carry_acts or not no_carry_acts:
        return {"error": "Need both carry and no-carry examples"}

    carry_mean = torch.stack(carry_acts).mean(dim=0).squeeze(0)
    no_carry_mean = torch.stack(no_carry_acts).mean(dim=0).squeeze(0)
    diff = (carry_mean - no_carry_mean).abs()

    # Find top neurons by activation difference
    d_ff = model.config.d_ff
    top_k = 10
    # diff is (total_neurons,) tensor — find top-k
    flat_diff = diff.view(-1)
    top_diffs, top_indices = torch.topk(flat_diff, min(top_k, flat_diff.size(0)))

    results = []
    for i in range(top_diffs.size(0)):
        idx_val = top_indices[i].item()
        diff_val = top_diffs[i].item()
        layer = idx_val // d_ff
        neuron = idx_val % d_ff
        c_mean = carry_mean[idx_val].item()
        nc_mean = no_carry_mean[idx_val].item()
        results.append(
            {
                "layer": layer,
                "neuron": neuron,
                "diff": round(diff_val, 4),
                "carry_mean": round(c_mean, 4),
                "no_carry_mean": round(nc_mean, 4),
            }
        )

    return {
        "top_carry_neurons": results,
        "total_neurons": diff.size(0),
        "layers": model.config.n_layers,
        "d_ff": d_ff,
        "samples_per_group": len(carry_acts),
    }


def analyze_attention_patterns(model, tok, captures: dict) -> dict:
    """Analyze attention patterns for carry propagation.

    For a problem like 5+7=12 (with carry), examine which positions
    each head attends to — specifically whether there's cross-position
    attention between the ones column and tens column.
    """
    # Test a carry problem
    prompt = "5+7="
    ids = tok.encode(prompt, add_special_tokens=True)
    x = torch.tensor([ids])

    for c in captures.values():
        c["attn_outputs"] = []
        c["attn_weights"] = []
        c["ffn_acts"] = []

    with torch.no_grad():
        model(x)

    # Token mapping for interpretability
    token_labels = [tok.itos[i.item()] for i in x[0]]
    # Map position to role
    # Format: BOS a + b = (reversed: BOS a_digit b_digit + ...)
    # For LSD-first: positions 0=BOS, 1=b_ones, 2=b_tens, ..., N=+, N+1=a_ones, ...
    # Actually the input is just "5+7=" so: BOS, 5, +, 7, =

    positions = []
    for i, label in enumerate(token_labels):
        role = "unknown"
        if label == "<BOS>":
            role = "bos"
        elif label == "+":
            role = "op"
        elif label == "=":
            role = "eq"
        elif label.isdigit():
            role = "digit"
        positions.append({"pos": i, "label": label, "role": role})

    # Extract attention patterns
    attn_patterns = {}
    for layer_idx, cap in captures.items():
        if cap["attn_weights"]:
            attn_weights = cap["attn_weights"][-1]  # (n_heads, q_len, kv_len)
            # For each head, show the attention to each position
            for head in range(attn_weights.size(0)):
                head_attn = attn_weights[head]  # (q_len, kv_len)
                # Most important: what does the position after `=` attend to?
                eq_pos = [p["pos"] for p in positions if p["role"] == "eq"]
                if eq_pos:
                    # Check what the first output position attends to
                    out_pos = eq_pos[0] + 1
                    if out_pos < head_attn.size(0):
                        attn_dist = head_attn[out_pos]
                        top_attended = torch.topk(attn_dist, min(3, attn_dist.size(0)))
                        top_positions = [
                            {
                                "pos": p.item(),
                                "label": token_labels[p.item()],
                                "weight": round(w.item(), 3),
                            }
                            for p, w in zip(top_attended.indices, top_attended.values)
                        ]
                        attn_patterns[f"L{layer_idx}H{head}"] = top_positions

    return {
        "prompt": prompt,
        "positions": positions,
        "attention_patterns": attn_patterns,
    }


# ─── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Mechanistic interpretability for Tabula Rasa arithmetic models"
    )
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument("--op", default="add", choices=["add", "sub"],
                        help="Operation (for tokenizer selection)")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of carry/no-carry samples for neuron analysis")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Guess architecture from state dict
    sd = ckpt.get("model_state_dict", ckpt)
    d_model = sd["token_embedding.weight"].shape[1]
    n_layers = len(
        [k for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight")]
    )
    d_ff = sd["layers.0.feed_forward.w1.weight"].shape[0]
    n_heads = d_model // (
        sd.get("layers.0.attention.rope.inv_freq", torch.tensor([0])).shape[0] * 2
        or 4
    )
    if n_heads == 0:
        n_heads = 4

    tok_dir = ckpt_path.parent
    tok_path = tok_dir / "tokenizer.json"
    tok = MathTokenizer.load(str(tok_path))

    cfg = Config()
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.d_ff = d_ff
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = 32

    model = MathTransformer(cfg)
    model.load_state_dict(sd)
    model.eval()

    acc = ckpt.get("acc", "?")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: d={d_model} L={n_layers} ff={d_ff} heads={n_heads}")
    print(f"  Params: {total_params:,}")
    print(f"  Reported accuracy: {acc}%")

    # Patch model
    print("Patching model for interpretability...")
    captures = patch_model_for_interpretability(model)

    # 1. Attention pattern analysis
    print("\n─── Attention Pattern Analysis ───")
    attn_results = analyze_attention_patterns(model, tok, captures)
    print(f"  Prompt: {attn_results['prompt']}")
    print(f"  Positions: {[p['label'] for p in attn_results['positions']]}")

    if attn_results["attention_patterns"]:
        print("\n  Per-head attention from first output position (after `=`):")
        for head_name, top_positions in sorted(attn_results["attention_patterns"].items()):
            attended_str = ", ".join(
                f"{p['label']} (pos {p['pos']}, w={p['weight']})"
                for p in top_positions
            )
            print(f"    {head_name}: {attended_str}")
    else:
        print("  (No attention data captured)")

    # 2. Carry neuron analysis
    print(f"\n─── Carry Neuron Analysis ({args.num_samples} samples/group) ───")
    neuron_results = find_carry_neurons(
        model, tok, captures, num_problems=args.num_samples
    )

    if "error" in neuron_results:
        print(f"  Error: {neuron_results['error']}")
    else:
        print(f"  Sampled {neuron_results['samples_per_group']} carry + {neuron_results['samples_per_group']} no-carry")
        print(f"\n  Top {len(neuron_results['top_carry_neurons'])} carry-sensitive neurons:")
        print(f"  {'Layer':>5} {'Neuron':>7} {'Diff':>8} {'Carry μ':>8} {'NoCarry μ':>8}")
        print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        for n in neuron_results["top_carry_neurons"]:
            print(
                f"  {n['layer']:>5} {n['neuron']:>7} {n['diff']:>8.4f} "
                f"{n['carry_mean']:>8.4f} {n['no_carry_mean']:>8.4f}"
            )

    # 3. Summary statistics
    print("\n─── Summary ───")
    if "error" not in neuron_results:
        total_l4 = sum(
            1 for n in neuron_results["top_carry_neurons"] if n["layer"] < 2
        )
        total_h4 = sum(
            1 for n in neuron_results["top_carry_neurons"] if n["layer"] >= 2
        )
        print(f"  Carry-sensitive neurons concentrated in:")
        print(f"    Lower layers (0-1): {total_l4}")
        print(f"    Higher layers (2-3): {total_h4}")

        # Mean diff by layer
        from collections import defaultdict

        by_layer = defaultdict(list)
        for n in neuron_results["top_carry_neurons"]:
            by_layer[n["layer"]].append(n["diff"])
        for layer in sorted(by_layer.keys()):
            avg = sum(by_layer[layer]) / len(by_layer[layer])
            print(f"    Layer {layer}: avg diff = {avg:.4f}")

    # Save if requested
    if args.output:
        output = {
            "architecture": {
                "d_model": d_model,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "n_heads": n_heads,
                "params": total_params,
            },
            "attention": attn_results,
            "carry_neurons": neuron_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
