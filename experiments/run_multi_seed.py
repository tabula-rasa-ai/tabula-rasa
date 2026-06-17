#!/usr/bin/env python3
"""Multi-seed experiment runner for error bars and baselines.

Runs training with N random seeds for a given operation and config,
collecting accuracy per seed. Reports mean ± std across seeds.

Usage:
    python3 experiments/run_multi_seed.py add --seeds 5 --quick
    python3 experiments/run_multi_seed.py mul --seeds 3 --steps 10000 --scratchpad
    python3 experiments/run_multi_seed.py add sub mul --seeds 5 --full
    python3 experiments/run_multi_seed.py --list  # Show last results
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = os.environ.get("TABULA_RASA_PYTHON", sys.executable)

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT / 'experiments' / 'multi_seed_results.json'


def run_one(op: str, seed: int, steps: int, scratchpad: bool, quick: bool) -> dict:
    """Run a single training session and return its accuracy.

    Args:
        op: Operation name (add/sub/mul).
        seed: Random seed for reproducibility.
        steps: Number of training steps.
        scratchpad: Enable scratchpad format.
        quick: Quick smoke-test mode (200 steps, 2K samples).

    Returns:
        Dict with ``accuracy``, ``steps``, ``seed``, and ``duration_s``.
    """
    cmd = [
        PYTHON, 'scripts/train_specialist.py', op,
        '--seed', str(seed),
        '--steps', str(steps),
    ]
    if quick:
        cmd.append('--quick')
    if scratchpad:
        cmd.append('--scratchpad')

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    # Parse accuracy from stdout (last eval line: "Eval: 87.5%")
    acc = 0.0
    for line in (result.stdout or '').splitlines():
        if 'Eval:' in line and '%' in line:
            parts = line.split()
            for p in parts:
                if p.endswith('%'):
                    try:
                        acc = float(p.rstrip('%'))
                    except ValueError:
                        pass

    return {
        'op': op,
        'seed': seed,
        'steps': steps,
        'accuracy': acc,
        'duration_s': round(elapsed, 1),
        'exit_code': result.returncode,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def show_results(results: list[dict]) -> None:
    """Display a summary table with mean ± std per operation."""
    from collections import defaultdict
    by_op: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_op[r['op']].append(r['accuracy'])

    print()
    print(f'  {"Operation":<12} {"Samples":>8} {"Mean":>8} {"Std":>8} {"Best":>8} {"Worst":>8}')
    print(f'  {"-"*12} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for op, accs in sorted(by_op.items()):
        mean = sum(accs) / len(accs)
        std = math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs)) if len(accs) > 1 else 0.0
        print(f'  {op:<12} {len(accs):>8} {mean:>7.1f}% {std:>6.2f}% {max(accs):>7.1f}% {min(accs):>7.1f}%')
    print()

    # Save to results file
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
    existing.extend(results)
    RESULTS_FILE.write_text(json.dumps(existing, indent=2))
    print(f'  Results appended to {RESULTS_FILE}')


def main():
    parser = argparse.ArgumentParser(description='Multi-seed experiment runner')
    parser.add_argument('ops', nargs='*', default=['add'], help='Operations to test')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds (default: 5)')
    parser.add_argument('--steps', type=int, default=0, help='Training steps (0=config default)')
    parser.add_argument('--scratchpad', action='store_true', help='Enable scratchpad format')
    parser.add_argument('--quick', action='store_true', help='Quick smoke test (200 steps)')
    parser.add_argument('--list', action='store_true', help='Show last results and exit')
    args = parser.parse_args()

    if args.list:
        if RESULTS_FILE.exists():
            results = json.loads(RESULTS_FILE.read_text())
            show_results(results)
        else:
            print(f'  No results found at {RESULTS_FILE}')
        return

    print(f'  Multi-seed experiment: ops={args.ops} seeds={args.seeds} steps={args.steps}')
    print()

    all_results: list[dict] = []
    for op in args.ops:
        for seed in range(1, args.seeds + 1):
            print(f'  [{time.strftime("%H:%M:%S")}] Running {op} seed={seed}...')
            r = run_one(op, seed, args.steps, args.scratchpad, args.quick)
            status = 'OK' if r['exit_code'] == 0 else f'FAIL({r["exit_code"]})'
            print(f'  [{time.strftime("%H:%M:%S")}] {op} seed={seed}: {r["accuracy"]:.1f}% ({r["duration_s"]:.0f}s) {status}')
            all_results.append(r)

    show_results(all_results)


if __name__ == '__main__':
    main()
