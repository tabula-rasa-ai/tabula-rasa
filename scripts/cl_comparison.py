"""Continual Learning methods comparison — EWC vs OGD vs LwF vs GEM.

Compares all 4 CL methods on a sequential 2-task learning benchmark:
  Task 1: Addition (50 steps)
  Task 2: Subtraction (50 steps)

Each method is evaluated on:
  - Task 1 accuracy AFTER learning Task 2 (forgetting measure)
  - Task 2 accuracy (new-task learning)
  - Training time overhead

Usage:
    python3 scripts/cl_comparison.py              # Quick test (50 steps each)
    python3 scripts/cl_comparison.py --steps 200  # Full comparison
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from egefalos.lwf_gem import GEM, LwF
from egefalos.ogd import OGD
from egefalos.online_ewc import OnlineEWC
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer

# ─── Task Data ───────────────────────────────────────────────

def make_dataset(op: str, num_samples: int = 200, max_digits: int = 2):
    """Generate a dataset for a given operation."""
    import random

    from tabula_rasa.tokenizer import MathTokenizer
    tok = MathTokenizer()

    seq_len = 16  # fixed sequence length
    inputs_list, targets_list = [], []
    for _ in range(num_samples):
        a = random.randint(10**(max_digits-1) if max_digits > 1 else 1, 10**max_digits - 1)
        b = random.randint(10**(max_digits-1) if max_digits > 1 else 1, 10**max_digits - 1)
        if op == 'sub':
            a, b = max(a, b), min(a, b)
        prompt = f'{a}{op}{b}='
        if op == 'add': answer = str(a + b)
        elif op == 'sub': answer = str(a - b)
        elif op == 'mul': answer = str(a * b)
        elif op == 'div': answer = str(a // b) if b != 0 else '0'
        else: answer = '0'
        input_ids = tok.encode(prompt, add_special_tokens=True)
        target_ids = tok.encode(answer, add_special_tokens=True)
        # Pad/truncate to fixed seq_len
        input_padded = (input_ids + [tok.pad_id] * seq_len)[:seq_len]
        target_padded = (target_ids + [tok.pad_id] * seq_len)[:seq_len]
        inputs_list.append(torch.tensor(input_padded))
        targets_list.append(torch.tensor(target_padded))

    inputs = torch.stack(inputs_list)
    targets = torch.stack(targets_list)
    return DataLoader(list(zip(inputs, targets)), batch_size=16, shuffle=True)


def eval_accuracy(model, tok, op: str, num_samples: int = 50, max_digits: int = 2) -> float:
    """Evaluate accuracy on a given operation."""
    import random
    model.eval()
    correct = 0
    total = 0

    for _ in range(num_samples):
        a = random.randint(10**(max_digits-1) if max_digits > 1 else 1, 10**max_digits - 1)
        b = random.randint(10**(max_digits-1) if max_digits > 1 else 1, 10**max_digits - 1)
        if op == 'sub':
            a, b = max(a, b), min(a, b)
        prompt = f'{a}{op}{b}='
        if op == 'add': expected = str(a + b)
        elif op == 'sub': expected = str(a - b)
        elif op == 'mul': expected = str(a * b)
        elif op == 'div': expected = str(a // b) if b != 0 else '0'
        else: expected = '0'

        output = model.generate(tok, prompt, max_new_tokens=10, temperature=0.0)
        # Extract answer after '='
        if '=' in output:
            raw = output.split('=', 1)[-1].replace('<EOS>', '').replace('<PAD>', '').strip()
        else:
            raw = output.strip()

        # Parse result
        result = raw.lstrip('0') or '0'
        if result == expected:
            correct += 1
        total += 1

    model.train()
    return correct / max(1, total) * 100


def run_cl_method(name: str, model, tok, task1_op: str, task2_op: str,
                  steps: int, verbose: bool = True) -> dict:
    """Run a CL method through 2 sequential tasks and measure forgetting."""
    cfg = Config()
    opt = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Task 1 data
    dl1 = make_dataset(task1_op, num_samples=200)
    task1_acc_before = eval_accuracy(model, tok, task1_op)

    method_obj = None
    results = {
        'method': name,
        'task1_op': task1_op,
        'task2_op': task2_op,
        'task1_acc_before': task1_acc_before,
        'task1_acc_after': 0,
        'task2_acc': 0,
        'forgetting': 0,
        'time_seconds': 0,
    }

    # ── Initialize CL method ──
    if name == 'EWC':
        method_obj = OnlineEWC(model, gamma=0.9)
    elif name == 'OGD':
        method_obj = OGD(model, alpha=0.5)
    elif name == 'LwF':
        method_obj = LwF(model, lambda_distill=0.5, temperature=2.0)
    elif name == 'GEM':
        method_obj = GEM(model, memory_size=256)
    else:
        raise ValueError(f"Unknown method: {name}")

    # ── Store pre-task-2 snapshot (for LwF) ──
    if name == 'LwF':
        method_obj.snapshot_teacher()

    # ── Store reference gradients (for OGD / GEM) ──
    if name == 'OGD':
        method_obj.store_gradients(dl1, num_batches=5)
    elif name == 'GEM':
        for batch in dl1:
            x, y = batch
            method_obj.add_to_memory(x, y)
        method_obj.store_gradients(dl1, num_batches=5)
        method_obj.new_task()

    # ── Task 2 training ──
    t0 = time.time()
    dl2 = make_dataset(task2_op, num_samples=200)
    steps_done = 0

    for batch in dl2:
        if steps_done >= steps:
            break
        x, y = batch

        opt.zero_grad()
        logits, loss, _ = model(x, targets=y)

        # ── Apply CL method ──
        if name == 'LwF':
            loss = method_obj.get_total_loss(loss, x)

        loss.backward()

        if name == 'OGD':
            sim = method_obj.project_gradients()
        elif name == 'GEM':
            sim = method_obj.project_gradients()

        opt.step()
        steps_done += 1

    elapsed = time.time() - t0

    # ── Evaluate ──
    results['task1_acc_after'] = eval_accuracy(model, tok, task1_op)
    results['task2_acc'] = eval_accuracy(model, tok, task2_op)
    results['forgetting'] = results['task1_acc_before'] - results['task1_acc_after']
    results['time_seconds'] = round(elapsed, 2)

    if verbose:
        print(f"  [{name:>4}] Task1: {results['task1_acc_before']:.0f}% → "
              f"{results['task1_acc_after']:.0f}%  |  Task2: {results['task2_acc']:.0f}%  |  "
              f"Forgetting: {results['forgetting']:+.1f}pp  |  {elapsed:.1f}s")

    return results


def compare_methods(steps: int = 50, task1: str = 'add', task2: str = 'sub',
                    verbose: bool = True):
    """Run all 4 CL methods on the same sequential task and compare."""
    tok = MathTokenizer()
    cfg = Config()
    # Reset config to default each time (seed doesn't matter for comparison direction)
    seed = 42

    methods = ['EWC', 'OGD', 'LwF', 'GEM']
    all_results = []

    print(f"\n{'='*70}")
    print("  CONTINUAL LEARNING BENCHMARK")
    print(f"  Task 1: {task1.upper()}  →  Task 2: {task2.upper()}")
    print(f"  Steps per task: {steps}")
    print(f"{'='*70}")

    for method in methods:
        torch.manual_seed(seed)
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)
        # Warm-start: train briefly on task 1
        opt = AdamW(model.parameters(), lr=0.001)
        dl1 = make_dataset(task1, num_samples=100)
        for _ in range(30):
            for x, y in dl1:
                opt.zero_grad()
                _, loss, _ = model(x, targets=y)
                loss.backward()
                opt.step()

        result = run_cl_method(method, model, tok, task1, task2,
                               steps=steps, verbose=verbose)
        all_results.append(result)

    # ── Results Table ──
    print(f"\n{'='*70}")
    print("  RESULTS: Continual Learning Methods Comparison")
    print(f"{'='*70}")
    print(f"  {'Method':<8} {'Before':<10} {'After':<10} {'Forget':<10} {'Task2':<10} {'Time':<8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for r in all_results:
        print(f"  {r['method']:<8} {r['task1_acc_before']:<8.0f}% "
              f"{r['task1_acc_after']:<8.0f}% {r['forgetting']:<+8.1f}pp "
              f"{r['task2_acc']:<8.0f}% {r['time_seconds']:<6.1f}s")

    # Save results
    out = ROOT / 'results' / 'cl_comparison.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        'config': {'task1': task1, 'task2': task2, 'steps': steps},
        'results': all_results,
    }, out, indent=2)
    print(f"\n  Results saved to: {out}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CL methods comparison')
    parser.add_argument('--steps', type=int, default=50, help='Training steps per task')
    parser.add_argument('--task1', type=str, default='add')
    parser.add_argument('--task2', type=str, default='sub')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    compare_methods(steps=args.steps, task1=args.task1, task2=args.task2,
                    verbose=not args.quiet)
