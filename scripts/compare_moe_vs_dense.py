#!/usr/bin/env python3
"""Dense vs MoE comparison — trains both architectures on identical data
and runs the full benchmark suite on each.

Usage:
    python3 scripts/compare_moe_vs_dense.py --steps 5000 --digits 2
    python3 scripts/compare_moe_vs_dense.py --quick        # 1000 steps, tiny model
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from tabula_rasa.config import Config
from tabula_rasa.dataset import MathDataset, generate_problem
from tabula_rasa.eval import full_benchmark, evaluate_accuracy, evaluate_per_position
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer


def create_model(
    cfg: Config,
    use_moe: bool = False,
    num_experts: int = 4,
    top_k: int = 2,
    seed: int = 42,
) -> MathTransformer:
    """Create a MathTransformer with optional MoE, seeded for reproducibility."""
    torch.manual_seed(seed)
    cfg.use_moe = use_moe
    cfg.num_experts = num_experts
    cfg.top_k = top_k

    model = MathTransformer(cfg)
    return model


def train_model(
    model: MathTransformer,
    tok: MathTokenizer,
    cfg: Config,
    num_steps: int,
    device: torch.device,
    log_interval: int = 200,
) -> float:
    """Train a model for num_steps, returning final eval accuracy."""
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Cosine LR with warmup
    warmup = min(100, num_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup) if step < warmup
        else 0.5 * (1.0 + torch.cos(torch.tensor((step - warmup) / max(1, num_steps - warmup) * 3.14159))),
    )

    # Create dataset
    dataset = MathDataset(tok, min(cfg.train_samples, num_steps * cfg.batch_size),
                          cfg.min_digits, cfg.max_digits, seed=42)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    best_acc = 0.0
    global_step = 0
    data_iter = iter(loader)
    start = time.time()

    while global_step < num_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % log_interval == 0 or global_step == num_steps:
            acc, _ = evaluate_accuracy(
                model, tok, num_problems=50, max_digits=cfg.max_digits, verbose=False
            )
            model.train()
            if acc > best_acc:
                best_acc = acc
            elapsed = time.time() - start
            steps_s = global_step / elapsed if elapsed > 0 else 0
            print(f"  Step {global_step:>5d}/{num_steps} | loss={loss.item():.4f} | acc={acc:.1f}% | {steps_s:.0f} steps/s", flush=True)

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Dense vs MoE comparison")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per model")
    parser.add_argument("--digits", type=int, default=2, help="Max digits")
    parser.add_argument("--quick", action="store_true", help="Tiny model, 500 steps")
    parser.add_argument("--preset", type=str, default="1M", choices=["1M", "5M"],
                        help="Architecture preset")
    args = parser.parse_args()

    steps = 500 if args.quick else args.steps
    max_digits = 1 if args.quick else args.digits
    preset = args.preset

    print("=" * 70)
    print("  DENSE vs MoE COMPARISON")
    print(f"  Preset: {preset} | Steps: {steps} | Max digits: {max_digits}")
    print("=" * 70)

    # Common config
    cfg = Config()
    cfg.apply_preset(preset)
    cfg.batch_size = 128
    cfg.learning_rate = 0.001
    cfg.max_digits = max_digits
    cfg.min_digits = 1
    cfg.train_samples = max(2000, steps * 2)
    cfg.max_steps = steps
    cfg.log_every = max(50, steps // 20)
    cfg.use_curriculum = False  # fixed difficulty for fair comparison

    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    device = torch.device("cpu")
    print(f"\nDevice: {device}")
    print(f"Config: d={cfg.d_model}, L={cfg.n_layers}, h={cfg.n_heads}, ff={cfg.d_ff}")

    # ── Train Dense ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  [1/2] Training DENSE model...")
    print(f"{'─'*70}")
    dense_model = create_model(cfg, use_moe=False, seed=42)
    dense_params = count_parameters(dense_model)
    print(f"  Parameters: {dense_params:,}")
    t0 = time.time()
    dense_best_acc = train_model(dense_model, tok, cfg, steps, device)
    dense_time = time.time() - t0
    print(f"  Done in {dense_time:.0f}s | Best accuracy: {dense_best_acc:.1f}%")

    # ── Train MoE ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  [2/2] Training MoE model (4 experts, top-2)...")
    print(f"{'─'*70}")
    moe_model = create_model(cfg, use_moe=True, num_experts=4, top_k=2, seed=42)
    moe_params = count_parameters(moe_model)
    # Note: despite more total params, compute per token is ~2x dense (top_k=2)
    compute_ratio = cfg.top_k / 1  # top_k=2 → ~2x dense compute per token
    print(f"  Parameters: {moe_params:,} ({(moe_params/dense_params-1)*100:+.0f}% vs dense)")
    print(f"  Active params per token: ~{cfg.top_k * (moe_params // cfg.num_experts // cfg.n_layers):,}")
    t0 = time.time()
    moe_best_acc = train_model(moe_model, tok, cfg, steps, device)
    moe_time = time.time() - t0
    print(f"  Done in {moe_time:.0f}s | Best accuracy: {moe_best_acc:.1f}%")

    # ── Full Benchmark Comparison ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FULL BENCHMARK COMPARISON")
    print(f"{'='*70}")

    results = {
        "dense": {"params": dense_params},
        "moe": {"params": moe_params},
        "config": {
            "preset": preset,
            "steps": steps,
            "max_digits": max_digits,
            "num_experts": 4,
            "top_k": 2,
        },
    }

    for name, model, params in [
        ("DENSE", dense_model, dense_params),
        ("MoE", moe_model, moe_params),
    ]:
        print(f"\n  >>> {name} Model ({params:,} params) <<<")
        bench = full_benchmark(model, tok, train_max_digits=max_digits, verbose=False)
        results[name.lower()]["benchmark"] = {
            k: (v.accuracy if hasattr(v, "accuracy") else
                v.overall_accuracy if hasattr(v, "overall_accuracy") else 0.0)
            for k, v in bench.items()
        }
        # Also store per-position data
        if "per_position" in bench:
            pp = bench["per_position"]
            results[name.lower()]["per_position"] = {
                "overall": pp.overall_accuracy,
                "columns": pp.position_accuracy,
                "carry": pp.carry_accuracy,
            }

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Dense':<12} {'MoE':<12} {'Δ':<10}")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*10}")

    d = results["dense"]["benchmark"]
    m = results["moe"]["benchmark"]
    all_metrics = set(list(d.keys()) + list(m.keys()))
    for metric in sorted(all_metrics):
        dv = d.get(metric, 0)
        mv = m.get(metric, 0)
        delta = mv - dv
        delta_str = f"{delta:+.1f}%" if abs(delta) > 0.1 else "—"
        print(f"  {metric:<35} {dv:>6.1f}%   {mv:>6.1f}%   {delta_str:>8}")

    # Per-position detail
    dp = results["dense"].get("per_position", {})
    mp = results["moe"].get("per_position", {})
    if dp and mp:
        print(f"\n  {'Per-position':<35} {'Dense':<12} {'MoE':<12} {'Δ':<10}")
        print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*10}")
        print(f"  {'Overall':<35} {dp.get('overall',0):>6.1f}%   {mp.get('overall',0):>6.1f}%   {mp.get('overall',0)-dp.get('overall',0):+.1f}%")
        for col in sorted(set(list(dp.get('columns',{}).keys()) + list(mp.get('columns',{}).keys()))):
            dv = dp.get('columns', {}).get(col, 0)
            mv = mp.get('columns', {}).get(col, 0)
            label = ["units", "tens", "hundreds", "thousands"][min(col, 3)]
            print(f"  {'  Column '+str(col)+' ('+label+')':<35} {dv:>6.1f}%   {mv:>6.1f}%   {mv-dv:+.1f}%")
        if dp.get('carry') is not None and mp.get('carry') is not None:
            print(f"  {'  Carry accuracy':<35} {dp['carry']:>6.1f}%   {mp['carry']:>6.1f}%   {mp['carry']-dp['carry']:+.1f}%")

    print(f"\n  {'Params':<35} {dense_params:>10,}   {moe_params:>10,}   {moe_params-dense_params:>+9,}")
    print(f"  {'Training time':<35} {dense_time:>6.0f}s   {moe_time:>6.0f}s   {moe_time-dense_time:+.0f}s")
    print(f"  {'Best eval accuracy':<35} {dense_best_acc:>6.1f}%   {moe_best_acc:>6.1f}%   {moe_best_acc-dense_best_acc:+.1f}%")
    print(f"{'='*70}")

    # Save results
    out_path = f"experiments/comparison_dense_vs_moe_{preset}_{max_digits}d.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
