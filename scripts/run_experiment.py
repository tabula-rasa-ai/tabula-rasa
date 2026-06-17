#!/usr/bin/env python3
"""Run a complete experiment: train → eval → probe → export → log — one command.

Usage:
    python3 scripts/run_experiment.py --name my_first_moe --moe --steps 5000
    python3 scripts/run_experiment.py --name rasa_10M --rasa --preset 10M --steps 30000
    python3 scripts/run_experiment.py --name lora_finetune --lora --lora-rank 16
    python3 scripts/run_experiment.py --list                              # show history
    python3 scripts/run_experiment.py --help
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def run_experiment(args):
    """Train a model, run benchmarks, probe, export, and log results."""
    name = args.name or f"experiment_{int(time.time())}"
    print("=" * 70)
    print(f"  EXPERIMENT: {name}")
    print("=" * 70)

    # ── 1. Build training command ──────────────────────────────────
    cmd = [sys.executable, "train.py", "--steps", str(args.steps)]

    if args.preset:
        cmd.extend(["--preset", args.preset])
    if args.moe:
        cmd.extend(["--use-moe", "--num-experts", str(args.num_experts), "--top-k", str(args.top_k)])
    if args.rasa:
        cmd.append("--use-rasa")
    if args.lora:
        cmd.append("--use-lora")
    if args.lora_rank:
        cmd.extend(["--lora-rank", str(args.lora_rank)])
    if args.value_head:
        cmd.append("--use-value-head")
    if args.max_digits:
        cmd.extend(["--max-digits", str(args.max_digits)])
    if args.batch:
        cmd.extend(["--batch", str(args.batch)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.resume:
        cmd.extend(["--resume", args.resume])

    print(f"\n  [1/5] Training: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=args.timeout)
    train_elapsed = time.time() - t0
    print(f"  Training completed in {train_elapsed:.0f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(f"  Training failed. stderr:\n{proc.stderr[-500:]}")
        return

    # Find the checkpoint
    ckpt = Path("checkpoints/best.pt")
    if not ckpt.exists():
        ckpt = Path("checkpoints/final.pt")
    if not ckpt.exists():
        print("  No checkpoint found after training.")
        return

    # ── 2. Run full benchmark ──────────────────────────────────────
    print("\n  [2/5] Benchmarking...")
    bench_path = ROOT / "experiments" / "runs" / f"benchmark_{name}.json"
    bench_cmd = [
        sys.executable, "-m", "tabula_rasa.eval", str(ckpt),
        str(args.max_digits or 4), "--save", str(bench_path),
    ]
    proc = subprocess.run(bench_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=args.timeout)
    print(f"  Benchmark: exit code {proc.returncode}")

    # ── 3. Linguistic probe ────────────────────────────────────────
    print("\n  [3/5] Linguistic probe...")
    probe_path = ROOT / "experiments" / "runs" / f"probe_{name}.json"
    try:
        from tabula_rasa.eval import load_model
        from tabula_rasa.probe import full_probe
        model, tok = load_model(str(ckpt))
        probe_results = full_probe(model, tok, visualize=False)
        with open(probe_path, "w") as f:
            json.dump(probe_results, f, indent=2, default=str)
        print(f"  Probe saved to {probe_path}")
    except Exception as e:
        print(f"  Probe skipped: {e}")

    # ── 4. Export to ProofGrid ─────────────────────────────────────
    print("\n  [4/5] Benchmark export...")
    export_dir = ROOT / "exports"
    export_dir.mkdir(exist_ok=True)
    try:
        from tabula_rasa.benchmark_export import full_export
        from tabula_rasa.eval import load_model
        model, tok = load_model(str(ckpt))
        full_export(model, tok, output_dir=str(export_dir), num_problems=args.export_problems, max_digits=args.max_digits or 4)
    except Exception as e:
        print(f"  Export skipped: {e}")

    # ── 5. Log experiment ──────────────────────────────────────────
    print("\n  [5/5] Logging experiment...")
    try:
        from tabula_rasa.eval import load_model
        from tabula_rasa.experiments import log_run
        model, tok = load_model(str(ckpt))
        from tabula_rasa.eval import evaluate_accuracy
        acc, _ = evaluate_accuracy(model, tok, num_problems=100, max_digits=args.max_digits or 4, verbose=False)

        # Load benchmark metrics if available
        metrics = {"accuracy": round(acc, 1)}
        if bench_path.exists():
            try:
                with open(bench_path) as f:
                    bench_data = json.load(f)
                metrics.update(bench_data)
            except:
                pass

        log_run(
            name=name,
            config={
                "preset": args.preset or "1M",
                "use_moe": args.moe, "num_experts": args.num_experts,
                "use_rasa": args.rasa,
                "use_lora": args.lora, "lora_rank": args.lora_rank,
                "use_value_head": args.value_head,
                "max_digits": args.max_digits,
                "steps": args.steps, "batch_size": args.batch,
                "learning_rate": args.lr,
            },
            metrics=metrics,
            checkpoint=str(ckpt),
            notes=f"Auto experiment: {name}",
            duration_s=train_elapsed,
        )
    except Exception as e:
        print(f"  Logging skipped: {e}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE: {name}")
    print(f"  Training: {train_elapsed:.0f}s")
    if 'acc' in locals():
        print(f"  Accuracy: {acc:.1f}%")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Benchmark: {bench_path}")
    print(f"  Probe: {probe_path}")
    print(f"  Exports: {export_dir}")
    print(f"{'='*70}")
    print("\n  Compare with: python3 scripts/cli.py experiments --compare")
    print("  View all:     python3 scripts/cli.py experiments")


def main():
    parser = argparse.ArgumentParser(description="Run a complete experiment: train → eval → probe → export → log")
    parser.add_argument("--name", type=str, help="Experiment name (auto-generated if omitted)")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--preset", type=str, choices=["1M", "5M", "10M"], help="Architecture preset")
    parser.add_argument("--moe", action="store_true", help="Enable MoE layers")
    parser.add_argument("--num-experts", type=int, default=4, help="MoE expert count")
    parser.add_argument("--top-k", type=int, default=2, help="MoE top-K routing")
    parser.add_argument("--rasa", action="store_true", help="Enable RASA attention")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--value-head", action="store_true", help="Enable value head")
    parser.add_argument("--max-digits", type=int, default=4, help="Max digits")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--export-problems", type=int, default=200, help="Export problem count")
    parser.add_argument("--timeout", type=int, default=3600, help="Max runtime (seconds)")
    parser.add_argument("--list", action="store_true", help="Show experiment history and exit")
    args = parser.parse_args()

    if args.list:
        from tabula_rasa.experiments import list_runs, print_runs
        print_runs(list_runs(limit=30))
        return

    run_experiment(args)


if __name__ == "__main__":
    main()
