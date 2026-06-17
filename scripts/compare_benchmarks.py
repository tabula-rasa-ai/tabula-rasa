#!/usr/bin/env python3
"""Compare two benchmark result JSONs side by side.

Usage:
    python3 scripts/compare_benchmarks.py results_v1.json results_v2.json
    python3 scripts/compare_benchmarks.py --dir experiments/  # compare all
    python3 scripts/compare_benchmarks.py --latest             # last 2 from gpu_output/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_benchmark(path):
    """Load a benchmark JSON file, extracting metrics robustly."""
    with open(path) as f:
        data = json.load(f)
    
    # Flatten: handle nested dicts and flat files
    metrics = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)):
                        metrics[f"{k}_{sk}"] = sv
    
    # Also try loading from experiments/benchmark_results.json format
    if "results" in data and isinstance(data["results"], list):
        for r in data["results"]:
            name = r.get("test", "unknown")
            passed = r.get("passed", False)
            metrics[f"test_{name}"] = 1.0 if passed else 0.0
    
    return metrics, path


def compare(file_a, file_b, name_a="A", name_b="B"):
    """Compare two benchmark files and print a formatted table."""
    metrics_a, path_a = load_benchmark(file_a)
    metrics_b, path_b = load_benchmark(file_b)
    
    all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    
    print(f"{'='*70}")
    print(f"  BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"  A: {name_a} ({path_a})")
    print(f"  B: {name_b} ({path_b})")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'A':>10} {'B':>10} {'Δ':>10}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}")
    
    deltas = []
    for key in all_keys:
        va = metrics_a.get(key, None)
        vb = metrics_b.get(key, None)
        if va is None and vb is None:
            continue
        
        if isinstance(va, float) or isinstance(vb, float):
            va_f = va if va is not None else 0.0
            vb_f = vb if vb is not None else 0.0
            delta = vb_f - va_f
            deltas.append(delta)
            delta_str = f"{delta:+.2f}" if abs(delta) > 0.001 else "—"
            print(f"  {key:<35} {va_f:>8.2f}%  {vb_f:>8.2f}%  {delta_str:>8}")
        else:
            va_s = str(va) if va is not None else "—"
            vb_s = str(vb) if vb is not None else "—"
            print(f"  {key:<35} {va_s:>10} {vb_s:>10}")
    
    if deltas:
        avg = sum(deltas) / len(deltas)
        print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'Average Δ':<35} {'':>10} {'':>10} {avg:>+8.2f}%")
    
    print(f"{'='*70}")
    
    # Return overall improvement
    return sum(deltas) / len(deltas) if deltas else 0


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("files", nargs="*", help="Two benchmark JSON files to compare")
    parser.add_argument("--dir", type=str, help="Directory with benchmark JSONs (compares all pairs)")
    parser.add_argument("--latest", action="store_true", help="Compare last 2 in gpu_output/")
    args = parser.parse_args()
    
    if args.dir:
        d = Path(args.dir)
        benches = sorted(d.glob("benchmark_*.json"))
        if len(benches) < 2:
            print(f"Need at least 2 benchmark files in {d}, found {len(benches)}")
            sys.exit(1)
        print(f"Found {len(benches)} benchmark files in {d}")
        for i in range(0, len(benches), 2):
            if i + 1 < len(benches):
                print()
                compare(benches[i], benches[i + 1],
                        name_a=benches[i].stem, name_b=benches[i + 1].stem)
    
    elif args.latest:
        out = Path("gpu_output")
        if not out.exists():
            out = Path("/output")
        benches = sorted(out.glob("benchmark_*.json"))
        if len(benches) < 2:
            print(f"Need at least 2 benchmark files in {out}, found {len(benches)}")
            sys.exit(1)
        compare(benches[-2], benches[-1], name_a=benches[-2].stem, name_b=benches[-1].stem)
    
    elif len(args.files) >= 2:
        compare(args.files[0], args.files[1], name_a=args.files[0], name_b=args.files[1])
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
