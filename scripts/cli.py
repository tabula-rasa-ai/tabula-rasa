#!/usr/bin/env python3
"""Tabula Rasa CLI — unified command interface for all tools.

Usage:
    python3 scripts/cli.py status              # Show system status
    python3 scripts/cli.py benchmark <ckpt>    # Run full benchmark
    python3 scripts/cli.py probe <ckpt>        # Linguistic knowledge probe
    python3 scripts/cli.py export <ckpt>       # ProofGrid/MLE-bench export
    python3 scripts/cli.py serve <ckpt>        # Start math server
    python3 scripts/cli.py rag                  # Start RAG server
    python3 scripts/cli.py experiments          # Show experiment history
    python3 scripts/cli.py list                # List available checkpoints
    python3 scripts/cli.py compare <a> <b>     # Compare two benchmarks
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def cmd_status(args):
    """Show system status — available components, checkpoints, experiments."""
    print("=" * 60)
    print("  TABULA RASA — SYSTEM STATUS")
    print("=" * 60)

    # Checkpoints
    print("\n  Checkpoints:")
    ckpt_dirs = [Path("checkpoints"), Path("gpu_output"), Path("/output")]
    found = False
    for d in ckpt_dirs:
        if d.exists():
            for f in sorted(d.glob("*.pt")):
                size = f.stat().st_size / 1e6
                age = time.time() - f.stat().st_mtime
                print(f"    {f}  ({size:.1f} MB, {age/3600:.0f}h old)")
                found = True
    if not found:
        print("    No checkpoints found. Train first: python3 train.py")

    # Experiments
    print("\n  Experiments:")
    runs_dir = ROOT / "experiments" / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("*.json"))
        print(f"    {len(runs)} experiment runs logged")
        if runs:
            print(f"    Latest: {runs[-1].stem}")
    else:
        print("    No experiments yet.")

    # Available tools
    print("\n  Available tools:")
    tools = [
        ("Benchmark", "python3 scripts/cli.py benchmark <ckpt>"),
        ("Probe", "python3 scripts/cli.py probe <ckpt>"),
        ("Export", "python3 scripts/cli.py export <ckpt>"),
        ("Train", "python3 train.py [--use-moe] [--use-rasa]"),
        ("GPU Train", "python3 scripts/train_gpu.py [--moe] [--rasa]"),
        ("RAG Server", "python3 -m tabula_rasa.rag --port 8300"),
        ("Orchestrator", "python3 -m tabula_rasa.orchestrator --port 8400"),
        ("Experiments", "python3 scripts/cli.py experiments"),
        ("Validate", "python3 scripts/validate_all.py"),
    ]
    for name, cmd in tools:
        print(f"    {name:<15} {cmd}")

    print()
    print("  PDF recommendations implemented: 22/25")
    print("  Not done: DeepSpeed ZeRO (multi-GPU), differential privacy, multimodal")
    print("=" * 60)


def cmd_benchmark(args):
    ckpt = args.checkpoint or _find_checkpoint()
    if not ckpt:
        print("No checkpoint specified or found.")
        sys.exit(1)
    print(f"Running benchmark on {ckpt}...")
    from tabula_rasa.eval import full_benchmark, load_model
    model, tok = load_model(ckpt)
    results = full_benchmark(model, tok, train_max_digits=args.digits, verbose=True)
    if args.save:
        import json
        summary = {}
        for k, v in results.items():
            if hasattr(v, "accuracy"): summary[k] = round(v.accuracy, 1)
            elif hasattr(v, "overall_accuracy"): summary[k] = round(v.overall_accuracy, 1)
        for ak in ["adversarial/boundary","adversarial/format_attacks","adversarial/noise","adversarial/fgsm"]:
            if ak in results: summary[ak.replace("/","_")] = round(results[ak].accuracy, 1)
        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {args.save}")


def cmd_probe(args):
    ckpt = args.checkpoint or _find_checkpoint()
    if not ckpt:
        print("No checkpoint specified or found.")
        sys.exit(1)
    print(f"Running linguistic probe on {ckpt}...")
    from tabula_rasa.eval import load_model
    from tabula_rasa.probe import full_probe
    model, tok = load_model(ckpt)
    full_probe(model, tok, visualize=True)


def cmd_export(args):
    ckpt = args.checkpoint or _find_checkpoint()
    if not ckpt:
        print("No checkpoint specified or found.")
        sys.exit(1)
    print(f"Running benchmark export on {ckpt}...")
    from tabula_rasa.benchmark_export import full_export
    from tabula_rasa.eval import load_model
    model, tok = load_model(ckpt)
    full_export(model, tok, output_dir=args.dir, num_problems=args.num, max_digits=args.digits)


def cmd_experiments(args):
    from tabula_rasa.experiments import compare_runs, export_csv, list_runs, print_runs
    runs = list_runs(limit=args.limit)
    print_runs(runs)
    if args.compare and len(runs) >= 2:
        print()
        compare_runs(runs[1], runs[0])
    if args.csv:
        export_csv(args.csv)


def cmd_list(args):
    """List available checkpoints with details."""
    for pattern in ["checkpoints/*.pt", "gpu_output/*/best.pt", "gpu_output/*.pt", "/output/*/best.pt"]:
        for f in sorted(Path().glob(pattern)):
            size = f.stat().st_size / 1e6
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
            name = str(f)
            # Try to read config from checkpoint
            try:
                state = __import__("torch").load(f, map_location="cpu", weights_only=True)
                acc = state.get("acc", state.get("best_acc", "?"))
                step = state.get("step", "?")
                cfg_info = state.get("config", {})
                preset = cfg_info.get("config_preset", cfg_info.get("preset", ""))
                print(f"  {name:<50} {size:>5.1f}MB  acc={acc}  step={step}  {mtime}")
            except:
                print(f"  {name:<50} {size:>5.1f}MB  {mtime}")


def cmd_serve(args):
    """Start the math server."""
    ckpt = args.checkpoint or _find_checkpoint()
    if not ckpt:
        print("No checkpoint found. Train first.")
        sys.exit(1)
    port = args.port or 8002
    print(f"Starting math server on port {port} with {ckpt}...")
    os.execvp(sys.executable, [sys.executable, "serve.py", "--port", str(port), "--checkpoint", str(ckpt)])


def cmd_rag(args):
    """Start the RAG server."""
    port = args.port or 8300
    print(f"Starting RAG server on port {port}...")
    os.execvp(sys.executable, [sys.executable, "-m", "tabula_rasa.rag", "--port", str(port)])


def cmd_compare(args):
    """Compare two benchmark JSON files."""
    if len(args.files) < 2:
        print("Need two files to compare: cli.py compare <a.json> <b.json>")
        sys.exit(1)
    from scripts.compare_benchmarks import compare
    compare(args.files[0], args.files[1], name_a=args.files[0], name_b=args.files[1])


def _find_checkpoint():
    """Find the best available checkpoint."""
    candidates = [
        "checkpoints/best.pt",
        "checkpoints/final.pt",
        "gpu_output/generalist/best.pt",
        "/output/generalist/best.pt",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="Tabula Rasa CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("status", help="Show system status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("benchmark", help="Run full benchmark")
    p.add_argument("checkpoint", nargs="?", help="Path to checkpoint")
    p.add_argument("--digits", type=int, default=4, help="Max digits")
    p.add_argument("--save", type=str, help="Save results to JSON")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("probe", help="Linguistic knowledge probe")
    p.add_argument("checkpoint", nargs="?", help="Path to checkpoint")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("export", help="Export to ProofGrid/MLE-bench")
    p.add_argument("checkpoint", nargs="?", help="Path to checkpoint")
    p.add_argument("--dir", default="exports", help="Output directory")
    p.add_argument("--num", type=int, default=200, help="Number of problems")
    p.add_argument("--digits", type=int, default=4, help="Max digits")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("experiments", help="Show experiment history")
    p.add_argument("--limit", type=int, default=20, help="Max runs")
    p.add_argument("--compare", action="store_true", help="Compare last two")
    p.add_argument("--csv", type=str, help="Export to CSV")
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("list", help="List available checkpoints")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("serve", help="Start math server")
    p.add_argument("checkpoint", nargs="?", help="Path to checkpoint")
    p.add_argument("--port", type=int, default=8002, help="Server port")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("rag", help="Start RAG server")
    p.add_argument("--port", type=int, default=8300, help="Server port")
    p.set_defaults(func=cmd_rag)

    p = sub.add_parser("compare", help="Compare two benchmark JSONs")
    p.add_argument("files", nargs="+", help="Two benchmark JSON files")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
