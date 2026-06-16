#!/usr/bin/env python3
"""run_benchmarks.py — Systematic benchmarks for Tabula Rasa.

Trains specialists across different configurations and reports throughput,
accuracy, and convergence time. Results appended to RESULTS.md.

Usage:
    python3 run_benchmarks.py                         # Run all quick benchmarks
    python3 run_benchmarks.py --quick                  # Fast smoke-test benchmarks
    python3 run_benchmarks.py --full                   # Full suite (hours)
    python3 run_benchmarks.py --throughput             # Forward-pass throughput only
    python3 run_benchmarks.py --op add                 # Bench a specific operation
    python3 run_benchmarks.py --output results.json    # Export as JSON
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

# ═══════════════════════════════════════════════════════════════════
# Benchmark Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_model(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    vocab_size=44,
    activation="relu",
    pos_encoding="rope",
    norm_type="rmsnorm",
):
    """Create a model with the given architecture."""
    cfg = Config()
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.d_ff = d_ff
    cfg.vocab_size = vocab_size
    cfg.activation = activation
    cfg.pos_encoding = pos_encoding
    cfg.norm_type = norm_type
    return MathTransformer(cfg)


def bench_throughput(model, batch_size=128, seq_len=16, steps=200, warmup=20) -> dict:
    """Measure forward-pass throughput (steps/second)."""
    device = next(model.parameters()).device
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(warmup):
        model(x, y)

    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(steps):
        model(x, y)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - t0

    steps_per_sec = steps / elapsed

    # Memory
    mem = 0
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "steps_per_sec": round(steps_per_sec, 2),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "elapsed_seconds": round(elapsed, 2),
        "peak_memory_mb": round(mem, 1) if mem else 0,
    }


def bench_architecture(configs: list[dict], steps=200) -> list[dict]:
    """Benchmark multiple architecture configurations."""
    results = []
    for cfg_dict in configs:
        label = (
            f"d={cfg_dict.get('d_model',128)} "
            f"L={cfg_dict.get('n_layers',4)} "
            f"act={cfg_dict.get('activation','relu')} "
            f"pos={cfg_dict.get('pos_encoding','rope')}"
        )
        print(f"  Bench: {label}...", end=" ", flush=True)

        model = _make_model(**cfg_dict)
        params = count_parameters(model)
        tp = bench_throughput(model, steps=steps)

        result = {
            "label": label,
            "params": params,
            **cfg_dict,
            **tp,
        }
        results.append(result)
        print(f"{tp['steps_per_sec']} st/s, {params:,} params")

    return results


def bench_evals() -> dict:
    """Benchmark per-digit evaluation on all 4 operations."""
    from train_specialist import evaluate, evaluate_per_digit

    results = {}
    for op in ["add", "sub", "mul", "div"]:
        ckpt = Path(f"specialists/math/{op}/best.pt")
        if not ckpt.exists():
            results[op] = {"error": "no checkpoint"}
            continue

        tok = MathTokenizer.load(str(ckpt.parent / "tokenizer.json"))
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        model = _make_model(vocab_size=tok.vocab_size)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        t0 = time.time()
        acc = evaluate(model, tok, cfg, op, num=100)
        eval_time = (time.time() - t0) * 1000

        t0 = time.time()
        per_digit = evaluate_per_digit(model, tok, cfg, op, per_digit_samples=15)
        per_digit_time = (time.time() - t0) * 1000

        results[op] = {
            "accuracy_pct": round(acc, 1),
            "eval_100_samples_ms": round(eval_time, 1),
            "per_digit_15_samples_ms": round(per_digit_time, 1),
            "per_digit": {str(k): round(v, 1) for k, v in per_digit.items()},
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Benchmark Suite Definitions
# ═══════════════════════════════════════════════════════════════════

QUICK_CONFIGS = [
    # (label, d_model, n_layers, n_heads, d_ff, activation, pos_encoding)
    dict(d_model=128, n_layers=4, n_heads=4, d_ff=512, activation="relu", pos_encoding="rope"),
    dict(d_model=128, n_layers=4, n_heads=4, d_ff=512, activation="swiglu", pos_encoding="rope"),
]

FULL_CONFIGS = QUICK_CONFIGS + [
    dict(d_model=64, n_layers=8, n_heads=4, d_ff=256, activation="relu", pos_encoding="rope"),
    dict(d_model=128, n_layers=2, n_heads=4, d_ff=512, activation="relu", pos_encoding="rope"),
    dict(d_model=128, n_layers=6, n_heads=4, d_ff=512, activation="relu", pos_encoding="rope"),
    dict(d_model=128, n_layers=4, n_heads=4, d_ff=512, activation="relu", pos_encoding="learned"),
    dict(
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        activation="relu",
        pos_encoding="rope",
        norm_type="layernorm",
    ),
]


def append_results(results: list[dict], output_path: str = "RESULTS.md"):
    """Append benchmark results to RESULTS.md as a formatted table."""
    header = f"\n## Benchmark Run — {time.strftime('%Y-%m-%d %H:%M')}\n\n"
    header += "| Config | Params | Steps/s | Batch | Peak Mem (MB) |\n"
    header += "|--------|--------|---------|-------|--------------|\n"

    rows = []
    for r in results:
        rows.append(
            f"| {r.get('label', '?')} "
            f"| {r.get('params', 0):,} "
            f"| {r.get('steps_per_sec', 0)} "
            f"| {r.get('batch_size', 0)} "
            f"| {r.get('peak_memory_mb', 0)} |"
        )

    with open(output_path, "a") as f:
        f.write(header)
        for row in rows:
            f.write(row + "\n")

    print(f"\n  Results appended to {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Systematic benchmarks for Tabula Rasa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick throughput benchmarks (2 configs, 200 steps each)",
    )
    parser.add_argument("--full", action="store_true", help="Full benchmark suite (all configs)")
    parser.add_argument("--throughput", action="store_true", help="Forward-pass throughput only")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluation benchmarks on trained checkpoints"
    )
    parser.add_argument(
        "--op", type=str, default=None, help="Operation to benchmark (add/sub/mul/div)"
    )
    parser.add_argument(
        "--output", type=str, default="RESULTS.md", help="Output file for results table"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Export results as JSON to this path"
    )
    args = parser.parse_args()

    print("Tabula Rasa — Benchmark Runner")
    print("=" * 50)
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    all_results = []

    # ── Throughput benchmarks ──
    configs = FULL_CONFIGS if args.full else QUICK_CONFIGS
    steps = 500 if not args.quick else 200

    bench_label = {
        "throughput": args.throughput,
    }

    print(f"  Throughput benchmarks ({len(configs)} configs, {steps} steps)...")
    print()
    tp_results = bench_architecture(configs, steps=steps)
    all_results.extend(tp_results)

    # ── Evaluation benchmarks ──
    if args.eval or (not args.throughput and not args.quick):
        print(f"\n  Evaluation benchmarks...")
        eval_results = bench_evals()
        print(f"\n  {'='*40}")
        for op, data in eval_results.items():
            if "error" in data:
                print(f"  {op}: {data['error']}")
            else:
                print(
                    f"  {op}: {data['accuracy_pct']}% acc "
                    f"({data['eval_100_samples_ms']}ms eval)"
                )
        all_results.append({"eval": eval_results})

    # ── Output ──
    append_results(all_results, args.output)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results exported to {args.json}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
