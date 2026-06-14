"""
EWC Benchmark: Custom Online EWC vs Avalanche-lib EWC on Sequential Arithmetic.

Compares the project's OnlineEWC (merged Fisher matrix with exponential decay)
against avalanche-lib's EwcPlugin on an addition → subtraction sequential task.

Protocol:
1. Train a tiny model (d=32, L=1, h=2, ff=64) on addition for 3000 steps
   — yields a 'fresh' addition model.
2. Compute Fisher matrix using project's OnlineEWC, save anchor, then train
   subtraction for 2000 steps with EWC penalty (custom method).
3. Train a fresh addition model (same config), then train subtraction using
   avalanche-lib's EwcPlugin for 2000 steps.
4. Compare:
   - Addition accuracy retention (both methods)
   - Subtraction accuracy (both methods)
   - Training time (both methods)
   - Fisher / memory usage (both methods)

Usage:
    python experiments/run_ewc_benchmark.py

Requires:
    - torch, numpy (standard)
    - avalanche-lib (optional — prints install instructions if missing)

Output:
    experiments/ewc_benchmark_results.json
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from egefalos.online_ewc import OnlineEWC
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer
from train_specialist import SpecialistDataset, _get_lr, generate_problem

# ─── Tiny model config for fast benchmarking ────────────────────────────
MODEL_D = 32
MODEL_L = 1
MODEL_H = 2
MODEL_FF = 64
ADD_STEPS = 3000
SUB_STEPS = 2000
BATCH_SIZE = 32
LAMBDA_EWC = 500
GAMMA = 0.9
EVAL_NUM = 100
SEED = 42
DEVICE = "cpu"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_tiny_config(tok) -> Config:
    """Create a tiny model Config with d=32, L=1, h=2, ff=64."""
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.d_model = MODEL_D
    cfg.n_layers = MODEL_L
    cfg.n_heads = MODEL_H
    cfg.d_ff = MODEL_FF
    cfg.max_seq_len = 32
    cfg.use_reversed = True
    cfg.use_scratchpad = True
    cfg.max_digits = 1
    cfg.min_digits = 1
    cfg.train_samples = 5000
    cfg.force_carry_ratio = 0.0
    cfg.use_loss_masking = True
    cfg.use_curriculum = False
    tok.max_seq_len = cfg.max_seq_len
    return cfg


def make_model(cfg) -> MathTransformer:
    """Create a fresh tiny MathTransformer."""
    model = MathTransformer(cfg)
    return model


def evaluate_model(model, tokenizer, cfg: Config, op: str, num: int = EVAL_NUM) -> float:
    """Evaluate accuracy on N random problems. Returns percentage 0–100."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num):
            expr, ans = generate_problem(
                op,
                cfg.min_digits,
                cfg.max_digits,
                reversed=cfg.use_reversed,
                scratchpad=cfg.use_scratchpad,
            )
            full = f"{expr}={ans}"
            ids = tokenizer.encode(full, add_special_tokens=True)
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y_target = ids[1:]
            logits, _, _ = model(x)
            pred_ids = logits[0].argmax(dim=-1).tolist()
            correct += int(pred_ids == y_target)
            total += 1
    return 100.0 * correct / total if total > 0 else 0.0


def compute_fisher_on_model(model, tok, cfg, op: str, num_samples: int = 100):
    """Compute Fisher matrix on a given operation using the OnlineEWC helper."""
    ewc = OnlineEWC(model, gamma=GAMMA)
    model.train()

    inputs, targets = [], []
    for _ in range(num_samples):
        expr, ans = generate_problem(
            op,
            cfg.min_digits,
            cfg.max_digits,
            reversed=cfg.use_reversed,
            scratchpad=cfg.use_scratchpad,
        )
        full = f"{expr}={ans}"
        ids = tok.encode(full, add_special_tokens=True)
        x = ids[:-1]
        y = ids[1:]
        inputs.append(x + [tok.pad_id] * (cfg.max_seq_len - len(x)))
        targets.append(y + [-100] * (cfg.max_seq_len - len(y)))

    loader = DataLoader(
        list(zip(torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))),
        batch_size=16,
        shuffle=True,
    )

    fisher = ewc.compute_fisher(loader, num_samples=num_samples)
    ewc.fisher_dict = {k: v.clone() for k, v in fisher.items()}
    ewc.task_count = 1
    ewc.save_anchor_weights()
    return ewc


def train_model(
    model, cfg, tok, op: str, steps: int, ewc: OnlineEWC = None, lambda_ewc: float = 0.0
):
    """Train model on an operation, optionally with EWC penalty."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
    )
    cfg.max_digits = 1
    cfg.min_digits = 1
    ds = SpecialistDataset(tok, op, cfg)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    global_step = 0
    t0 = time.time()
    for epoch in range(50):
        for x, y in loader:
            if global_step >= steps:
                break
            _, task_loss, _ = model(x, y)
            total_loss = task_loss
            if ewc is not None and lambda_ewc > 0:
                total_loss = total_loss + ewc.compute_ewc_penalty(lambda_ewc=lambda_ewc)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for pg in optimizer.param_groups:
                pg["lr"] = 0.001 * _get_lr(global_step, 200, steps, "cosine")
            global_step += 1
    elapsed = time.time() - t0
    return global_step, elapsed


def measure_fisher_bytes(model) -> float:
    """Return rough memory usage of Fisher-like params (anchor + Fisher pair copy)."""
    total_bytes = 0.0
    for p in model.parameters():
        if p.requires_grad:
            # Two copies: one for Fisher diagonal, one for anchor weights
            total_bytes += 2 * p.numel() * 4  # float32
    return total_bytes / (1024 * 1024)  # MB


# ─────────────────────────────────────────────────────────────────────────
#  AVALANCHE EWC WRAPPER
# ─────────────────────────────────────────────────────────────────────────


def try_run_avalanche_benchmark(model, cfg, tok):
    """
    Train addition → subtraction using avalanche-lib's EwcPlugin.

    Returns a dict with results or error string.
    """
    try:
        import avalanche
        from avalanche.benchmarks.generators import benchmark_from_datasets
        from avalanche.benchmarks.utils import make_avalanche_dataset
        from avalanche.training import Ewc as EwcStrategy
        from avalanche.training.plugins import EwcPlugin
        from avalanche.training.strategies import BaseStrategy, Naive
    except ImportError as e:
        return {"error": f"avalanche-lib not installed: {e}"}

    set_seed(SEED)

    # Build datasets for avalanche
    # Train on addition for ADD_STEPS batches, then subtract for SUB_STEPS
    # avalanche uses experiences - we create two experiences: add then sub

    # Create datasets
    cfg.max_digits = 1
    cfg.min_digits = 1

    add_ds_raw = SpecialistDataset(tok, "add", cfg)
    sub_ds_raw = SpecialistDataset(tok, "sub", cfg)

    # Convert to tensors
    add_x_all, add_y_all = [], []
    for i in range(min(len(add_ds_raw), ADD_STEPS * BATCH_SIZE)):
        x, y = add_ds_raw[i % len(add_ds_raw)]
        add_x_all.append(x.unsqueeze(0))
        add_y_all.append(y.unsqueeze(0))
    add_x = torch.cat(add_x_all[: ADD_STEPS * BATCH_SIZE])
    add_y = torch.cat(add_y_all[: ADD_STEPS * BATCH_SIZE])

    sub_x_all, sub_y_all = [], []
    for i in range(min(len(sub_ds_raw), SUB_STEPS * BATCH_SIZE)):
        x, y = sub_ds_raw[i % len(sub_ds_raw)]
        sub_x_all.append(x.unsqueeze(0))
        sub_y_all.append(y.unsqueeze(0))
    sub_x = torch.cat(sub_x_all[: SUB_STEPS * BATCH_SIZE])
    sub_y = torch.cat(sub_y_all[: SUB_STEPS * BATCH_SIZE])

    # Create avalanche datasets
    add_dataset = make_avalanche_dataset(add_x, add_y, task_labels=0)
    sub_dataset = make_avalanche_dataset(sub_x, sub_y, task_labels=1)

    # Create benchmark from datasets
    benchmark = benchmark_from_datasets(
        train=[add_dataset, sub_dataset], test=[add_dataset, sub_dataset]
    )

    # Create fresh model for avalanche
    model_av = make_model(cfg)
    # Reset seed for fair comparison
    set_seed(SEED)
    model_av = make_model(cfg)

    # Avalanche EWC strategy
    ewc_plugin = EwcPlugin(ewc_lambda=LAMBDA_EWC, mode="separate")
    ewc_strategy = EwcStrategy(
        model=model_av,
        optimizer=torch.optim.AdamW(
            model_av.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
        ),
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        train_mb_size=BATCH_SIZE,
        train_epochs=1,
        plugins=[ewc_plugin],
        device=DEVICE,
    )

    # Measure pre-training accuracy
    add_before_av = evaluate_model(model_av, tok, cfg, "add", num=EVAL_NUM)

    t0 = time.time()

    # Task 1: train addition
    ewc_strategy.train(benchmark.train_stream[0])

    time_task1 = time.time() - t0
    add_after_task1_av = evaluate_model(model_av, tok, cfg, "add", num=EVAL_NUM)

    t1 = time.time()

    # Task 2: train subtraction
    ewc_strategy.train(benchmark.train_stream[1])

    time_task2 = time.time() - t1
    total_time_av = time.time() - t0

    add_final_av = evaluate_model(model_av, tok, cfg, "add", num=EVAL_NUM)
    sub_final_av = evaluate_model(model_av, tok, cfg, "sub", num=EVAL_NUM)

    # Estimate Fisher memory for avalanche (it stores per-task fisher)
    fisher_memory_av = measure_fisher_bytes(model_av) * 2  # avalanche stores separate per-task

    return {
        "add_before": add_before_av,
        "add_after_task1": add_after_task1_av,
        "add_retention": add_final_av,
        "sub_accuracy": sub_final_av,
        "training_time_task1": time_task1,
        "training_time_task2": time_task2,
        "total_training_time": total_time_av,
        "fisher_memory_mb": fisher_memory_av,
    }


# ─────────────────────────────────────────────────────────────────────────
#  CUSTOM ONLINE EWC BENCHMARK
# ─────────────────────────────────────────────────────────────────────────


def run_custom_ewc_benchmark(model, cfg, tok):
    """Train addition → subtraction using project's OnlineEWC. Returns dict."""
    set_seed(SEED)

    # Baseline addition accuracy (untrained model)
    add_before = evaluate_model(model, tok, cfg, "add", num=EVAL_NUM)

    # ─── Phase 1: Train addition (3000 steps) ───
    print("  [Custom EWC] Training addition...")
    steps_done, time_task1 = train_model(model, cfg, tok, "add", ADD_STEPS, ewc=None)
    add_after_train = evaluate_model(model, tok, cfg, "add", num=EVAL_NUM)
    print(f"    add acc after training: {add_after_train:.1f}%  [{time_task1:.1f}s]")

    # ─── Phase 2: Compute Fisher + save anchor ───
    print("  [Custom EWC] Computing Fisher on addition...")
    t_fisher_start = time.time()
    ewc = compute_fisher_on_model(model, tok, cfg, "add", num_samples=100)
    fisher_time = time.time() - t_fisher_start
    fisher_mem = measure_fisher_bytes(model)
    print(f"    Fisher computed: {fisher_time:.2f}s, ~{fisher_mem:.2f}MB")

    # ─── Phase 3: Train subtraction with EWC ───
    print("  [Custom EWC] Training subtraction with EWC penalty...")
    steps_done2, time_task2 = train_model(
        model, cfg, tok, "sub", SUB_STEPS, ewc=ewc, lambda_ewc=LAMBDA_EWC
    )
    total_time = time_task1 + fisher_time + time_task2

    add_final = evaluate_model(model, tok, cfg, "add", num=EVAL_NUM)
    sub_final = evaluate_model(model, tok, cfg, "sub", num=EVAL_NUM)
    print(
        f"    add retention: {add_final:.1f}%, sub accuracy: {sub_final:.1f}%  [{time_task2:.1f}s]"
    )

    return {
        "add_before": add_before,
        "add_after_training": add_after_train,
        "add_retention": add_final,
        "sub_accuracy": sub_final,
        "training_time_task1": time_task1,
        "fisher_compute_time": fisher_time,
        "training_time_task2": time_task2,
        "total_training_time": total_time,
        "fisher_memory_mb": fisher_mem,
    }


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("  EWC BENCHMARK: Custom OnlineEWC vs Avalanche-lib EwcPlugin")
    print(
        "  Tiny model: d=32, L=1, h=2, ff=64  (~{:.0f}K params)".format(
            (MODEL_D * 4 * MODEL_D + MODEL_D * MODEL_FF * 2) / 1000
        )
    )
    print("  Addition: {} steps → Subtraction: {} steps".format(ADD_STEPS, SUB_STEPS))
    print("  λ={}, γ={}, seed={}".format(LAMBDA_EWC, GAMMA, SEED))
    print("=" * 65)

    tok = MathTokenizer()
    cfg = make_tiny_config(tok)

    # ─── Benchmark A: Custom Online EWC ────────────────────────────
    print("\n" + "─" * 65)
    print("  BENCHMARK A: Custom Online EWC")
    print("─" * 65)
    model_a = make_model(cfg)
    custom_results = run_custom_ewc_benchmark(model_a, cfg, tok)

    # ─── Benchmark B: Avalanche-lib EWC ────────────────────────────
    print("\n" + "─" * 65)
    print("  BENCHMARK B: Avalanche-lib EwcPlugin")
    print("─" * 65)
    model_b = make_model(cfg)
    avalanche_results = try_run_avalanche_benchmark(model_b, cfg, tok)

    # ─── Compile results ───────────────────────────────────────────
    output = {
        "config": {
            "d_model": MODEL_D,
            "n_layers": MODEL_L,
            "n_heads": MODEL_H,
            "d_ff": MODEL_FF,
            "add_steps": ADD_STEPS,
            "sub_steps": SUB_STEPS,
            "lambda_ewc": LAMBDA_EWC,
            "gamma": GAMMA,
            "seed": SEED,
        },
        "custom_online_ewc": custom_results,
        "avalanche_ewc": avalanche_results,
    }

    # ─── Print comparison table ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  === EWC Implementation Comparison ===")
    print("=" * 65)

    c_add_ret = custom_results.get("add_retention", 0.0)
    c_sub_acc = custom_results.get("sub_accuracy", 0.0)
    c_time = custom_results.get("total_training_time", 0.0)
    c_fisher_mem = custom_results.get("fisher_memory_mb", 0.0)

    av_error = avalanche_results.get("error")
    if av_error:
        a_add_ret = 0.0
        a_sub_acc = 0.0
        a_time = 0.0
        a_fisher_mem = 0.0
    else:
        a_add_ret = avalanche_results.get("add_retention", 0.0)
        a_sub_acc = avalanche_results.get("sub_accuracy", 0.0)
        a_time = avalanche_results.get("total_training_time", 0.0)
        a_fisher_mem = avalanche_results.get("fisher_memory_mb", 0.0)

    print(f"  {'':>25} {'Custom Online EWC':>18} {'Avalanche EWC':>18}")
    print(f"  {'-'*25}-+-{'-'*18}-+-{'-'*18}")
    print(f"  {'Add retention:':>25} {c_add_ret:>15.1f}%   {a_add_ret:>15.1f}%")
    print(f"  {'Sub accuracy:':>25} {c_sub_acc:>15.1f}%   {a_sub_acc:>15.1f}%")
    print(f"  {'Training time:':>25} {c_time:>14.1f}s   {a_time:>14.1f}s")
    print(f"  {'Fisher compute:':>25} {c_fisher_mem:>13.2f}MB   {a_fisher_mem:>13.2f}MB")

    if av_error:
        print(f"\n  ⚠️  Avalanche EWC: {av_error}")

    print(f"\n  {'=' * 55}")

    # ─── Interpretation ────────────────────────────────────────────
    add_delta = c_add_ret - a_add_ret
    sub_delta = c_sub_acc - a_sub_acc
    if not av_error:
        if abs(add_delta) < 5 and abs(sub_delta) < 5:
            print(f"\n  ✅ Comparable performance (|Δ| < 5pp)")
        else:
            print(
                f"\n  📊 Performance difference: add Δ={add_delta:+.1f}pp, sub Δ={sub_delta:+.1f}pp"
            )
        faster_text = "Custom OnlineEWC" if c_time < a_time else "Avalanche EWC"
        print(f"  ⏱  Faster: {faster_text} ({abs(c_time - a_time):.1f}s difference)")

    # ─── Save results ──────────────────────────────────────────────
    out_path = Path("experiments/ewc_benchmark_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
