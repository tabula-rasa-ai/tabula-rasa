"""Scaling Law Ablation: Does 5M params break the 3-task collapse?

Hypothesis: The 3-task collapse (add->sub->mul) at 1M params is a
strict representational capacity limit, not a Fisher-merging artifact.
At 5M params, the model should retain all 3 tasks above 80%.

Protocol:
1. Create fresh 5M model (d=256, L=5, h=8, ff=1024, 5.27M params)
2. Train on addition from scratch (no EWC - first task)
3. Compute Fisher on addition, save anchors
4. Train subtraction with EWC + Fisher Archive (unmerged per-task)
5. Train multiplication with EWC + Fisher Archive
6. Measure retention of ALL tasks after each step

Expected:
  1M baseline (from ablation_three_task_results.json):
    After Sub: Add=40%, Sub=58%
    After Mul: Add=0%,  Sub=2%,  Mul=14%

  5M prediction:
    After Sub: Add>90%, Sub>90%
    After Mul: Add>80%, Sub>80%, Mul>80%
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egefalos.online_ewc import OnlineEWC
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer
from train_specialist import (
    SpecialistDataset,
    _get_lr,
    evaluate,
    evaluate_per_digit,
    generate_problem,
)


def _infinite_batches(ds, bs, dev):
    """Infinite DataLoader generator - never exhausts."""
    while True:
        for x, y in DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4):
            yield x.to(dev), y.to(dev)


# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRESET = "5M"
LAMBDA_EWC = 500
GAMMA = 0.9
STEPS_PER_TASK = 5000
EVAL_EVERY = 500
BATCH = 128

print(f"Device: {DEVICE}  Preset: {PRESET}  l={LAMBDA_EWC} g={GAMMA}")
print(f"Steps/task: {STEPS_PER_TASK}  Batch: {BATCH}")


def make_model(tok):
    cfg = Config().apply_preset(PRESET)
    cfg.vocab_size = tok.vocab_size
    cfg.max_digits = 1
    cfg.min_digits = 1
    cfg.train_samples = 5000
    cfg.use_reversed = True
    cfg.use_scratchpad = True
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Created model: {total:,} params ({total/1e6:.2f}M)  train_samples={cfg.train_samples}")
    return model, cfg


def train_fresh(model, cfg, tok, operation, steps=STEPS_PER_TASK):
    """Train from scratch (no EWC) using infinite batches."""
    print(f"\n  Training {operation} from scratch ({steps} steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    cfg.max_digits = 1
    cfg.min_digits = 1
    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, BATCH, DEVICE)
    t0 = time.time()
    global_step = 0
    best_acc = 0.0
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    use_amp = DEVICE == "cuda"

    while global_step < steps:
        x, y = next(batch_iter)
        if use_amp:
            with torch.amp.autocast("cuda"):
                _, loss, _ = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss, _ = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
        for pg in optimizer.param_groups:
            pg["lr"] = 0.001 * _get_lr(global_step, 200, steps, "cosine")
        global_step += 1

        if global_step % EVAL_EVERY == 0 or global_step == steps:
            model.eval()
            acc = evaluate(model, tok, cfg, operation, num=100)
            elapsed = time.time() - t0
            print(f"    Step {global_step}/{steps}: {operation}={acc:.1f}% [{elapsed:.0f}s]")
            if acc > best_acc:
                best_acc = acc
            model.train()

    elapsed = time.time() - t0
    print(f"  Done: {operation} best={best_acc:.1f}% [{elapsed:.0f}s]")
    return best_acc


def compute_fisher(model, tok, cfg, operation, num_samples=200):
    """Compute Fisher diagonal on one operation."""
    model.train()
    device = next(model.parameters()).device
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p)
    use_amp = device.type == "cuda"
    samples = 0
    while samples < num_samples:
        inputs, targets = [], []
        for _ in range(min(16, num_samples - samples)):
            expr, ans = generate_problem(operation, 1, 1, reversed=True, scratchpad=True)
            full = f"{expr}={ans}"
            ids = tok.encode(full, add_special_tokens=True)
            x = ids[:-1]
            y = ids[1:]
            pad = tok.pad_id
            inputs.append(x + [pad] * (tok.max_seq_len - len(x)))
            targets.append(y + [-100] * (tok.max_seq_len - len(y)))
        if not inputs:
            break
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        model.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                _, loss, _ = model(x, y)
        else:
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


def train_with_ewc(model, cfg, tok, operation, ewc, lambda_val, steps=STEPS_PER_TASK):
    """Train one operation with EWC protection using infinite batches."""
    print(f"\n  Training {operation} with EWC (l={lambda_val})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    cfg.max_digits = 1
    cfg.min_digits = 1
    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, BATCH, DEVICE)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    t0 = time.time()
    global_step = 0
    trajectory = []
    epoch = 0

    while global_step < steps:
        x, y = next(batch_iter)
        if use_amp:
            with torch.amp.autocast("cuda"):
                _, task_loss, _ = model(x, y)
                ewc_pen = ewc.compute_ewc_penalty(lambda_ewc=lambda_val)
                total = task_loss + ewc_pen
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, task_loss, _ = model(x, y)
            ewc_pen = ewc.compute_ewc_penalty(lambda_ewc=lambda_val)
            total = task_loss + ewc_pen
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
        for pg in optimizer.param_groups:
            pg["lr"] = 0.001 * _get_lr(global_step, 200, steps, "cosine")
        global_step += 1
        epoch += 1

        if epoch % 2 == 0:
            model.eval()
            metrics = {}
            for op_name in ["add", "sub", "mul"]:
                metrics[op_name] = evaluate(model, tok, cfg, op_name, num=100)
            trajectory.append({"epoch": epoch, **metrics})
            elapsed = time.time() - t0
            vals = "  ".join(f"{o}={metrics[o]:.0f}%" for o in ["add", "sub", "mul"])
            print(f"    Epoch {epoch}: {vals} [{elapsed:.0f}s]")
            model.train()

    return trajectory


def measure_all(model, tok, cfg):
    """Measure accuracy on all operations."""
    model.eval()
    results = {}
    for op in ["add", "sub", "mul"]:
        acc = evaluate(model, tok, cfg, op, num=200)
        pd = evaluate_per_digit(model, tok, cfg, op, per_digit_samples=50)
        results[op] = {"accuracy": acc, "per_digit": {str(k): v for k, v in pd.items()}}
    return results


def main():
    print("=" * 60)
    print("  SCALING LAW ABLATION: 1M vs 5M")
    print("  Fresh 5M model -> Add -> Sub -> Mul with Fisher Archive")
    print("=" * 60)

    tok = MathTokenizer()
    all_results = {
        "task_sequence": ["addition", "subtraction", "multiplication"],
        "preset": PRESET,
        "params": 5_269_248,
        "config": {"d_model": 256, "n_layers": 5, "n_heads": 8, "d_ff": 1024, "max_seq_len": 48},
    }

    # Phase 0: Create + train addition
    print("\n--- Phase 0: Create Fresh 5M Model ---")
    model, cfg = make_model(tok)
    print("\n--- Phase 0b: Train Addition (first task, no EWC) ---")
    add_acc = train_fresh(model, cfg, tok, "add", steps=STEPS_PER_TASK)
    all_results["phase0_add_baseline"] = measure_all(model, tok, cfg)
    print(f"  After add training: add={all_results['phase0_add_baseline']['add']['accuracy']:.1f}%")

    # Step 1: Fisher on addition
    print("\n--- Step 1: Compute Fisher on Addition ---")
    fisher_add = compute_fisher(model, tok, cfg, "add", num_samples=200)
    print(f"  Fisher computed: {len(fisher_add)} parameter groups")
    ewc = OnlineEWC(model, gamma=GAMMA, use_archive=True)
    ewc.fisher_dict = {k: v.clone() for k, v in fisher_add.items()}
    ewc.task_count = 1
    ewc.save_anchor_weights()
    fisher_norm_add = sum(v.norm().item() for v in ewc.fisher_dict.values())
    print(f"  Fisher norm: {fisher_norm_add:.4f}")

    # Step 2: Train subtraction
    print("\n--- Step 2: Train Subtraction (task 2) ---")
    traj_sub = train_with_ewc(model, cfg, tok, "sub", ewc, LAMBDA_EWC)
    step2 = {"operation": "subtraction", "action": "train_with_ewc"}
    step2["trajectory"] = traj_sub
    step2["measurements"] = measure_all(model, tok, cfg)
    print(f"  After sub: add={step2['measurements']['add']['accuracy']:.1f}%  "
          f"sub={step2['measurements']['sub']['accuracy']:.1f}%")
    all_results["step2_train_sub"] = step2

    # Step 3: Merge Fisher
    print("\n--- Step 3: Compute Fisher on Subtraction & Merge ---")
    fisher_sub = compute_fisher(model, tok, cfg, "sub", num_samples=200)
    old_count = ewc.task_count
    ewc.merge_fisher(fisher_sub)
    ewc.save_anchor_weights()
    merged_norm = sum(v.norm().item() for v in ewc.fisher_dict.values())
    print(f"  Fisher merged: {old_count} -> {ewc.task_count} tasks")
    print(f"  Norm: {fisher_norm_add:.4f} -> {merged_norm:.4f} ({merged_norm/fisher_norm_add:.2f}x)")
    all_results["step3_merge"] = {
        "task_count_before": old_count, "task_count_after": ewc.task_count,
        "fisher_norm_after": merged_norm, "fisher_growth": merged_norm / fisher_norm_add,
    }

    # Step 4: Train multiplication
    print("\n--- Step 4: Train Multiplication (task 3) ---")
    traj_mul = train_with_ewc(model, cfg, tok, "mul", ewc, LAMBDA_EWC)
    step4 = {"operation": "multiplication", "action": "train_with_ewc"}
    step4["trajectory"] = traj_mul
    step4["measurements"] = measure_all(model, tok, cfg)
    print(f"  After mul: add={step4['measurements']['add']['accuracy']:.1f}%  "
          f"sub={step4['measurements']['sub']['accuracy']:.1f}%  "
          f"mul={step4['measurements']['mul']['accuracy']:.1f}%")
    all_results["step4_train_mul"] = step4

    # Summary
    after_add = all_results["phase0_add_baseline"]
    after_sub = all_results["step2_train_sub"]["measurements"]
    after_mul = all_results["step4_train_mul"]["measurements"]
    add_ret = after_mul["add"]["accuracy"]
    sub_ret = after_mul["sub"]["accuracy"]
    mul_acq = after_mul["mul"]["accuracy"]
    min_ret = min(add_ret, sub_ret)

    all_results["summary"] = {
        "addition_retained_after_mul": add_ret,
        "subtraction_retained_after_mul": sub_ret,
        "multiplication_acquired": mul_acq,
        "min_retained": min_ret,
        "ewc_effective": min_ret > 80,
        "three_task_success": min_ret > 80 and mul_acq > 80,
    }

    print(f"\n{'='*60}")
    print(f"  SCALING LAW SUMMARY: 5M params")
    print(f"  {'='*50}")
    print(f"  {'Phase':<16} | {'Add':>6} | {'Sub':>6} | {'Mul':>6}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    print(f"  {'After Add':<16} | {after_add['add']['accuracy']:>5.1f}% | {'-':>6} | {'-':>6}")
    print(f"  {'After Sub':<16} | {after_sub['add']['accuracy']:>5.1f}% | {after_sub['sub']['accuracy']:>5.1f}% | {'-':>6}")
    print(f"  {'After Mul':<16} | {add_ret:>5.1f}% | {sub_ret:>5.1f}% | {mul_acq:>5.1f}%")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    print(f"\n  Min retained: {min_ret:.1f}%")
    if min_ret > 80 and mul_acq > 80:
        print(f"  5M MODEL HANDLES 3 TASKS - Capacity boundary confirmed!")
        print(f"  The 3-task collapse at 1M is a strict representational limit.")
    elif min_ret > 50:
        print(f"  Partial success: {min_ret:.1f}% min retention")
        print(f"  Improvement over 1M baseline but not full 3-task mastery.")
    else:
        print(f"  3-task persists at 5M")
        print(f"  Possible causes: EWC hyperparameters, need more steps, or structural limit.")
    print(f"  {'='*50}")

    output_path = Path("experiments/scaling_ablation_5m_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
