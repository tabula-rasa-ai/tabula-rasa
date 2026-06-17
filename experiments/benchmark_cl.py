"""CL Method Benchmark: EWC vs MAS vs OGD.

Standardised comparison of continual learning methods on sequential
arithmetic tasks (add -> sub -> mul).

Metrics:
  - Accuracy on each task after all training
  - BWT (Backward Transfer): acc_on_A_after_C - acc_on_A_after_A
  - FWT (Forward Transfer): acc_on_B - baseline_acc_on_B
  - Computational overhead (time per method)
  - Memory used (importance matrix size)

Usage:
  python3 experiments/benchmark_cl.py --steps 2000 --methods none ewc ogd mas
  python3 experiments/benchmark_cl.py --quick  (500 steps each)

Output:
  experiments/cl_benchmark_results.json — full results table
  Console prints a formatted comparison table.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer
from scripts.train_specialist import (
    SpecialistDataset,
    generate_problem,
    _get_lr,
    evaluate,
)

# ─── Config ───────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Benchmark device: {DEVICE}")

QUICK_STEPS = 500
FULL_STEPS = 2000

TASKS = [
    ("add", "Addition"),
    ("sub", "Subtraction"),
    ("mul", "Multiplication"),
]


def make_model(tok):
    """Create a fresh model for each benchmark run."""
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.n_heads = 4
    cfg.d_ff = 512
    cfg.max_seq_len = 32
    cfg.max_digits = 1
    cfg.min_digits = 1
    cfg.use_reversed = True
    cfg.use_scratchpad = True
    cfg.use_loss_masking = True
    cfg.dropout = 0.1
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg).to(DEVICE)
    return model, cfg


def evaluate_accuracy(model, tok, cfg, operation, num=100):
    """Evaluate exact-match accuracy on a single operation."""
    return evaluate(model, tok, cfg, operation, num=num)


def evaluate_all_ops(model, tok, cfg, ops, num=100):
    """Evaluate on all given operations."""
    return {op: evaluate_accuracy(model, tok, cfg, op, num) for op in ops}


def _infinite_batches(ds, bs, dev):
    """Infinite DataLoader generator."""
    while True:
        for x, y in DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True):
            yield x.to(dev), y.to(dev)


def train_steps(model, cfg, tok, operation, steps, penalty_fn=None):
    """Train model on one operation for a fixed number of steps.

    Args:
        penalty_fn: Optional callable that returns a penalty tensor
            added to the task loss (for EWC, MAS etc.).

    Returns:
        (final_acc, steps_to_mastery)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    scheduler = lambda s: 0.001 * _get_lr(s, 200, steps, "cosine")

    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, cfg.batch_size, DEVICE)
    global_step = 0
    steps_to_mastery = None
    MASTERY = 80.0

    while global_step < steps:
        x, y = next(batch_iter)
        _, task_loss, _ = model(x, y)

        total_loss = task_loss
        if penalty_fn is not None:
            p = penalty_fn()
            total_loss = task_loss + p

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for pg in optimizer.param_groups:
            pg["lr"] = scheduler(global_step)
        global_step += 1

        if global_step % 50 == 0 or global_step == steps:
            acc = evaluate_accuracy(model, tok, cfg, operation, num=50)
            if steps_to_mastery is None and acc >= MASTERY:
                steps_to_mastery = global_step

    final_acc = evaluate_accuracy(model, tok, cfg, operation, num=100)
    return final_acc, steps_to_mastery or steps


def compute_fisher(model, tok, cfg, operation, num_samples=100):
    """Compute Fisher diagonal on one operation."""
    model.train()
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p)
    samples = 0
    while samples < num_samples:
        inputs, targets = [], []
        for _ in range(min(16, num_samples - samples)):
            expr, ans = generate_problem(
                operation, 1, 1, reversed=True, scratchpad=True
            )
            full = f"{expr}={ans}"
            ids = tok.encode(full, add_special_tokens=True)
            x = ids[:-1]
            y = ids[1:]
            pad = tok.pad_id
            inputs.append(x + [pad] * (tok.max_seq_len - len(x)))
            targets.append(y + [-100] * (tok.max_seq_len - len(y)))
        if not inputs:
            break
        x = torch.tensor(inputs, dtype=torch.long, device=DEVICE)
        y = torch.tensor(targets, dtype=torch.long, device=DEVICE)
        model.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None and name in fisher:
                fisher[name] += p.grad.detach() ** 2
        samples += x.size(0)
    if samples > 0:
        for name in fisher:
            fisher[name] /= samples
    return fisher


def make_dataloader(tok, cfg, operation, num_samples=100):
    """Create a small DataLoader for computing importance/Fisher."""
    inputs, targets = [], []
    for _ in range(num_samples):
        expr, ans = generate_problem(
            operation, 1, 1, reversed=True, scratchpad=True
        )
        full = f"{expr}={ans}"
        ids = tok.encode(full, add_special_tokens=True)
        x = ids[:-1]
        y = ids[1:]
        pad = tok.pad_id
        inputs.append(x + [pad] * (tok.max_seq_len - len(x)))
        targets.append(y + [-100] * (tok.max_seq_len - len(y)))
    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True)


def estimate_memory_usage(importance_dict):
    """Estimate memory used by importance matrices in MB."""
    total_bytes = sum(
        v.numel() * v.element_size() for v in importance_dict.values()
    )
    return total_bytes / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════
# Method Implementations
# ═══════════════════════════════════════════════════════════════════


def method_none(model, cfg, tok, task_seq, steps):
    """No continual learning: fine-tune sequentially (baseline)."""
    results = {}
    for i, (op, _) in enumerate(task_seq):
        acc, mastery = train_steps(model, cfg, tok, op, steps)
        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(
                model, tok, cfg, [t[0] for t in task_seq]
            ),
            "steps_to_mastery": mastery,
        }
    results["memory_mb"] = 0.0
    return results


def method_ewc(model, cfg, tok, task_seq, steps, gamma=0.9, lam=1000.0):
    """Online EWC with merged Fisher."""
    from egefalos.online_ewc import OnlineEWC

    ewc = OnlineEWC(model, gamma=gamma, use_archive=False)
    results = {}
    memory_mb = 0.0

    for i, (op, _) in enumerate(task_seq):
        if i == 0:
            # First task: train normally, compute Fisher
            acc, mastery = train_steps(model, cfg, tok, op, steps)
            ewc.save_anchor_weights()
            fisher = compute_fisher(model, tok, cfg, op, num_samples=100)
            ewc.fisher_dict = {k: v.clone() for k, v in fisher.items()}
            ewc.task_count = 1
            memory_mb = estimate_memory_usage(fisher)
        else:
            # Subsequent tasks: train with EWC penalty
            acc, mastery = train_steps(
                model, cfg, tok, op, steps,
                penalty_fn=lambda: ewc.compute_ewc_penalty(lam),
            )
            fisher = compute_fisher(model, tok, cfg, op, num_samples=100)
            ewc.merge_fisher(fisher)
            ewc.save_anchor_weights()
            memory_mb = max(memory_mb, estimate_memory_usage(fisher))

        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(
                model, tok, cfg, [t[0] for t in task_seq]
            ),
            "steps_to_mastery": mastery,
        }

    results["memory_mb"] = round(memory_mb, 2)
    return results


def method_mas(model, cfg, tok, task_seq, steps, lambda_reg=1000.0):
    """Memory Aware Synapses."""
    from egefalos.mas import MAS

    mas = MAS(model, lambda_reg=lambda_reg, merge_mode="max")
    results = {}
    memory_mb = 0.0

    for i, (op, _) in enumerate(task_seq):
        if i == 0:
            # First task: train normally, compute importance
            acc, mastery = train_steps(model, cfg, tok, op, steps)
            dataloader = make_dataloader(tok, cfg, op, num_samples=100)
            omega = mas.compute_importance(dataloader, num_samples=100)
            mas.consolidate(new_omega=omega)
            memory_mb = estimate_memory_usage(omega)
        else:
            # Subsequent tasks: train with MAS penalty
            acc, mastery = train_steps(
                model, cfg, tok, op, steps,
                penalty_fn=lambda: mas.compute_penalty(model),
            )
            # Compute and store importance for new task
            dataloader = make_dataloader(tok, cfg, op, num_samples=100)
            omega = mas.compute_importance(dataloader, num_samples=100)
            mas.consolidate(new_omega=omega)
            memory_mb = max(memory_mb, estimate_memory_usage(omega))

        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(
                model, tok, cfg, [t[0] for t in task_seq]
            ),
            "steps_to_mastery": mastery,
        }

    results["memory_mb"] = round(memory_mb, 2)
    return results


def method_ogd(model, cfg, tok, task_seq, steps, alpha=0.5):
    """Orthogonal Gradient Descent."""
    from egefalos.ogd import OGD

    ogd = OGD(model, alpha=alpha)
    results = {}
    memory_mb = 0.0

    for i, (op, _) in enumerate(task_seq):
        if i == 0:
            # First task: train normally
            acc, mastery = _train_ogd_first(model, cfg, tok, op, steps, ogd)
        else:
            # Subsequent tasks: train with gradient projection
            acc, mastery = _train_ogd(model, cfg, tok, op, steps, ogd)

        # Estimate memory: total gradient vector storage
        if i >= 0:
            ref_size = 0
            for refs in ogd._reference_grads:
                for v in refs.values():
                    ref_size += v.numel() * v.element_size()
            memory_mb = max(memory_mb, ref_size / (1024 * 1024))

        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(
                model, tok, cfg, [t[0] for t in task_seq]
            ),
            "steps_to_mastery": mastery,
        }

    results["memory_mb"] = round(memory_mb, 2)
    return results


def _train_ogd_first(model, cfg, tok, operation, steps, ogd):
    """First task training with OGD (collects reference gradients)."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    scheduler = lambda s: 0.001 * _get_lr(s, 200, steps, "cosine")
    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, cfg.batch_size, DEVICE)
    global_step = 0
    steps_to_mastery = None
    MASTERY = 80.0

    while global_step < steps:
        x, y = next(batch_iter)
        _, task_loss, _ = model(x, y)
        optimizer.zero_grad()
        task_loss.backward()
        ogd.project_gradients()  # no-op on first task (no refs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for pg in optimizer.param_groups:
            pg["lr"] = scheduler(global_step)
        global_step += 1
        if global_step % 50 == 0 or global_step == steps:
            acc = evaluate_accuracy(model, tok, cfg, operation, num=50)
            if steps_to_mastery is None and acc >= MASTERY:
                steps_to_mastery = global_step

    # Store reference gradients
    dataloader = make_dataloader(tok, cfg, operation, num_samples=100)
    ogd.store_gradients(dataloader, num_batches=5)

    final_acc = evaluate_accuracy(model, tok, cfg, operation, num=100)
    return final_acc, steps_to_mastery or steps


def _train_ogd(model, cfg, tok, operation, steps, ogd):
    """Subsequent task training with OGD gradient projection."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    scheduler = lambda s: 0.001 * _get_lr(s, 200, steps, "cosine")
    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, cfg.batch_size, DEVICE)
    global_step = 0
    steps_to_mastery = None
    MASTERY = 80.0

    while global_step < steps:
        x, y = next(batch_iter)
        _, task_loss, _ = model(x, y)
        optimizer.zero_grad()
        task_loss.backward()
        ogd.project_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for pg in optimizer.param_groups:
            pg["lr"] = scheduler(global_step)
        global_step += 1
        if global_step % 50 == 0 or global_step == steps:
            acc = evaluate_accuracy(model, tok, cfg, operation, num=50)
            if steps_to_mastery is None and acc >= MASTERY:
                steps_to_mastery = global_step

    # Store reference gradients for new task
    dataloader = make_dataloader(tok, cfg, operation, num_samples=100)
    ogd.store_gradients(dataloader, num_batches=5)

    final_acc = evaluate_accuracy(model, tok, cfg, operation, num=100)
    return final_acc, steps_to_mastery or steps


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════


def compute_bwt(results, task_names):
    """Backward Transfer: how much does task j forget task i (i < j).

    BWT_i = acc_on_task_i_after_C - acc_on_task_i_after_A
    """
    bwt = {}
    for i in range(len(task_names)):
        name_i = task_names[i]
        acc_after = []
        for j in range(i + 1, len(task_names)):
            phase = f"after_task_{j+1}"
            if phase in results:
                acc_after.append(
                    results[phase]["accuracies"].get(name_i, 0)
                )
        if acc_after:
            initial = results.get("after_task_1", {}).get("accuracies", {}).get(name_i, 0)
            bwt[name_i] = min(acc_after) - initial
        else:
            bwt[name_i] = 0.0
    return bwt


def compute_fwt(results, task_names, baseline_accs):
    """Forward Transfer: how much does previous training help task j.

    FWT_j = acc_on_j_with_CL - baseline_acc_on_j
    """
    fwt = {}
    for i in range(1, len(task_names)):
        name_i = task_names[i]
        phase = f"after_task_{i+1}"
        cl_acc = results.get(phase, {}).get("accuracies", {}).get(name_i, 0)
        base_acc = baseline_accs.get(name_i, 0)
        fwt[name_i] = cl_acc - base_acc
    return fwt


# ═══════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

METHODS = {
    "none": method_none,
    "ewc": method_ewc,
    "mas": method_mas,
    "ogd": method_ogd,
}


def run_benchmark(methods_to_run, task_names, steps_per_task):
    """Run CL benchmark for specified methods and tasks."""
    tok = MathTokenizer()

    print(f"\n{'='*60}")
    print(f"  CL BENCHMARK -- {len(task_names)} tasks, {steps_per_task} steps/task")
    print(f"  Tasks: {', '.join(task_names)}")
    print(f"  Methods: {', '.join(methods_to_run)}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    # Compute baselines: train each task from scratch
    baseline_accs = {}
    print(f"\n  Computing baselines (no CL, train from scratch)...")
    for op_name in task_names:
        model, cfg = make_model(tok)
        acc, mastery = train_steps(model, cfg, tok, op_name, steps_per_task)
        baseline_accs[op_name] = acc
        print(f"    {op_name}: {acc:.1f}% (mastery at step {mastery})")

    task_seq = [(op, baseline_accs.get(op, 0)) for op in task_names]
    all_results = {}

    for method_name in methods_to_run:
        print(f"\n  {'---'*12}")
        print(f"  Method: {method_name.upper()}")
        print(f"  {'---'*12}")

        model, cfg = make_model(tok)
        t0 = time.time()

        if method_name == "none":
            results = method_none(model, cfg, tok, task_seq, steps_per_task)
        elif method_name == "ewc":
            results = method_ewc(model, cfg, tok, task_seq, steps_per_task)
        elif method_name == "mas":
            results = method_mas(model, cfg, tok, task_seq, steps_per_task)
        elif method_name == "ogd":
            results = method_ogd(model, cfg, tok, task_seq, steps_per_task)
        else:
            print(f"  Unknown method: {method_name}")
            continue

        elapsed = time.time() - t0
        results["initial"] = {"add": 0.0, "sub": 0.0, "mul": 0.0}

        # Compute BWT
        bwt = compute_bwt(results, task_names)
        results["backward_transfer"] = bwt

        # Compute FWT
        fwt = compute_fwt(results, task_names, baseline_accs)
        results["forward_transfer"] = fwt

        # Summary metrics
        final_phase = f"after_task_{len(task_names)}"
        final_accs = results.get(final_phase, {}).get("accuracies", {})
        avg_retention = sum(final_accs.values()) / max(1, len(final_accs))
        total_steps = sum(
            results.get(f"after_task_{i+1}", {}).get("steps_to_mastery", steps_per_task)
            for i in range(len(task_names))
        )

        results["summary"] = {
            "avg_final_accuracy": round(avg_retention, 1),
            "total_steps_to_mastery": total_steps,
            "time_seconds": round(elapsed, 1),
            "min_bwt": round(min(bwt.values()), 1) if bwt else 0,
            "avg_fwt": round(sum(fwt.values()) / max(1, len(fwt)), 1) if fwt else 0,
            "memory_mb": results.get("memory_mb", 0),
        }

        all_results[method_name] = results

        # Print per-method summary
        print(f"\n  [{method_name}] Results:")
        print(f"    Final accuracies: {', '.join(f'{k}={v:.1f}%' for k, v in final_accs.items())}")
        print(f"    Avg retention: {avg_retention:.1f}%")
        print(f"    BWT: {', '.join(f'{k}={v:.1f}%' for k, v in bwt.items())}" if bwt else "    BWT: N/A")
        print(f"    FWT: {', '.join(f'{k}={v:.1f}%' for k, v in fwt.items())}" if fwt else "    FWT: N/A")
        print(f"    Total steps: {total_steps}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Memory (importance): {results.get('memory_mb', 0):.2f} MB")

    # ── Final comparison table ──
    print(f"\n{'='*90}")
    print(f"  CL BENCHMARK COMPARISON")
    print(f"{'='*90}")

    header = (
        f"  {'Method':<10} | {'Avg Ret%':>8} | {'BWT%':>6} | {'FWT%':>6} | "
        f"{'Steps':>7} | {'Time':>6} | {'Mem MB':>7}"
    )
    print(header)
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

    for method_name in methods_to_run:
        s = all_results[method_name]["summary"]
        print(
            f"  {method_name:<10} | {s['avg_final_accuracy']:>7.1f}% | "
            f"{s['min_bwt']:>5.1f}% | {s['avg_fwt']:>5.1f}% | "
            f"{s['total_steps_to_mastery']:>7} | {s['time_seconds']:>5.1f}s | "
            f"{s['memory_mb']:>6.2f} MB"
        )

    # Also print per-task accuracy table
    print(f"\n{'='*90}")
    print(f"  PER-TASK ACCURACIES (Final)")
    print(f"{'='*90}")

    task_header = f"  {'Method':<10}"
    for op_name in task_names:
        task_header += f" | {op_name:>6}"
    task_header += f" | {'Avg':>6}"
    print(task_header)
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for method_name in methods_to_run:
        fp = all_results[method_name].get(
            f"after_task_{len(task_names)}", {}
        ).get("accuracies", {})
        accs = [fp.get(op, 0) for op in task_names]
        avg = sum(accs) / max(1, len(accs))
        acc_str = " | ".join(f"{a:>5.1f}%" for a in accs)
        print(f"  {method_name:<10} | {acc_str} | {avg:>5.1f}%")

    print(f"{'='*90}")

    # Save results
    out_path = Path("experiments/cl_benchmark_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="CL Method Benchmark: EWC vs MAS vs OGD"
    )
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Steps per task (default: 500 for quick, 2000 for full)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 500 steps each"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full benchmark: 2000 steps each"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=list(METHODS.keys()),
        help=f"Methods: {', '.join(METHODS.keys())}"
    )
    args = parser.parse_args()

    steps = args.steps
    if steps == 0:
        if args.quick:
            steps = QUICK_STEPS
        elif args.full:
            steps = FULL_STEPS
        else:
            steps = QUICK_STEPS  # default: quick

    # Validate methods
    for m in args.methods:
        if m not in METHODS:
            print(f"Unknown method: {m}. Available: {list(METHODS.keys())}")
            sys.exit(1)

    run_benchmark(args.methods, [t[0] for t in TASKS], steps)


if __name__ == "__main__":
    main()
