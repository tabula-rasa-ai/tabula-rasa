#!/usr/bin/env python3
"""Length generalization experiment for Tabula Rasa arithmetic.

Trains on N-digit problems, evaluates on N+1 and N+2 digits to measure
out-of-distribution generalization. Tests multiple position encoding modes:
rope, abacus, alibi_arithmetic.

Usage:
    python3 experiments/run_length_gen.py add --train-digits 2 --eval-digits 4 --steps 5000
    python3 experiments/run_length_gen.py add sub --train-digits 1 --eval-digits 3 --steps 10000 --pe abacus
    python3 experiments/run_length_gen.py mul --train-digits 1 --eval-digits 2 --steps 15000 --pe all
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT / 'experiments' / 'length_gen_results.json'


def run_one(
    op: str,
    train_digits: int,
    eval_digits: int,
    steps: int,
    pe: str,
    seed: int = 42,
    scratchpad: bool = True,
) -> dict:
    """Train on train_digits, eval on eval_digits with given position encoding.

    Returns dict with training and evaluation metrics.
    """
    import os
    _PY = os.environ.get("TABULA_RASA_PYTHON", sys.executable)
    cmd = [
        _PY, 'scripts/train_specialist.py', op,
        '--seed', str(seed),
        '--steps', str(steps),
        '--max-digits', str(train_digits),
    ]
    if pe != 'rope':
        cmd.extend(['--pos-encoding', pe])
    if scratchpad:
        pass  # scratchpad is ON by default, no CLI flag
    # Use --test-hard to get OOD eval at max_digits+1 from training built-in
    cmd.append('--test-hard')

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    # Parse training accuracy (last eval line)
    train_acc = 0.0
    for line in (result.stdout or '').splitlines():
        if 'Eval' in line and '%' in line and 'step' in line:
            # Line format: "    Eval step 500: 76.0% (best: 76.0%) | ..."
            for p in line.split():
                if p.endswith('%') and p.rstrip('%').replace('.', '').isdigit():
                    try:
                        train_acc = float(p.rstrip('%'))
                        break
                    except ValueError:
                        pass

    # Parse OOD accuracy from training --test-hard output
    ood_acc = 0.0
    for line in (result.stdout or '').splitlines():
        if 'hard' in line and '%' in line:
            # e.g. "1d:100% 2d:100% hard:50%"
            for p in line.split():
                if p.startswith('hard:'):
                    try:
                        ood_acc = float(p.split(':')[1].rstrip('%'))
                    except (ValueError, IndexError):
                        pass

    return {
        'op': op,
        'pe': pe,
        'train_digits': train_digits,
        'eval_digits': eval_digits,
        'steps': steps,
        'seed': seed,
        'train_accuracy': train_acc,
        'ood_accuracy': ood_acc,
        'duration_s': round(elapsed, 1),
        'exit_code': result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description='Length generalization experiment')
    parser.add_argument('ops', nargs='+', help='Operations (add/sub/mul)')
    parser.add_argument('--train-digits', type=int, default=2, help='Training digit length')
    parser.add_argument('--eval-digits', type=int, default=4, help='OOD eval digit length')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--pe', choices=['rope', 'abacus', 'none', 'alibi_arithmetic', 'all'],
                        default='all', help='Position encoding mode')
    parser.add_argument('--seeds', type=int, default=3, help='Seeds per config')
    parser.add_argument('--scratchpad', action='store_true', default=True,
                        help='Enable scratchpad (default: on)')
    args = parser.parse_args()

    pe_list = (
        ['rope', 'abacus', 'none', 'alibi_arithmetic']
        if args.pe == 'all' else [args.pe]
    )

    print(f'  Length generalization: ops={args.ops} train={args.train_digits}d '
          f'eval={args.eval_digits}d steps={args.steps}')
    print(f'  Position encodings: {pe_list}')
    print()

    all_results = []
    for op in args.ops:
        for pe in pe_list:
            for seed in range(1, args.seeds + 1):
                print(f'  [{time.strftime("%H:%M:%S")}] {op} pe={pe} seed={seed}...')
                r = run_one(op, args.train_digits, args.eval_digits,
                           args.steps, pe, seed, args.scratchpad)
                status = 'OK' if r['exit_code'] == 0 else f'FAIL({r["exit_code"]})'
                print(f'  [{time.strftime("%H:%M:%S")}] {op} pe={pe} seed={seed}: '
                      f'train={r["train_accuracy"]:.1f}% ood={r["ood_accuracy"]:.1f}% '
                      f'({r["duration_s"]:.0f}s) {status}')
                all_results.append(r)

    # Summary
    print()
    print(f'  {"Op":<6} {"PE":<18} {"Train Acc":>10} {"OOD Acc":>10} {"N":>4}')
    print(f'  {"-"*6} {"-"*18} {"-"*10} {"-"*10} {"-"*4}')
    for r in all_results:
        print(f'  {r["op"]:<6} {r["pe"]:<18} {r["train_accuracy"]:>8.1f}% '
              f'{r["ood_accuracy"]:>8.1f}% {r["train_digits"]:>2}→{r["eval_digits"]}')

    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if RESULTS_FILE.exists():
        existing = json.loads(RESULTS_FILE.read_text())
    existing.extend(all_results)
    RESULTS_FILE.write_text(json.dumps(existing, indent=2))
    print(f'\n  Results saved to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
