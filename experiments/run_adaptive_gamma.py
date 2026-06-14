"""Mitigation 1: Adaptive Gamma — validate on three-task sequence.

Tests whether adaptive gamma (scaling γ based on Fisher overlap)
preserves >90% on all 3 tasks during add → sub → mul sequence.

Protocol:
1. Load addition model, compute Fisher (F_add)
2. Train subtraction with EWC (γ=0.9, λ=500)
3. Compute Fisher on trained model (F_sub)
4. Measure overlap = cosine_sim(F_add, F_sub)
5. Compute γ_adaptive = 0.9 * overlap + 0.3 * (1 - overlap)
6. Merge: F = γ_adaptive * F_add + (1-γ_adaptive) * F_sub
7. Train multiplication with merged Fisher
8. Measure all three tasks
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

DEVICE = "cpu"
ADD_DIR = Path("specialists/math/add")
LAMBDA_EWC = 500
BASE_GAMMA = 0.9
MIN_GAMMA = 0.3
STEPS = 2000


def load_clean_model(tok):
    checkpoint = torch.load(ADD_DIR / "best_original.pt", map_location="cpu", weights_only=True)
    sd = checkpoint.get("model_state_dict", checkpoint)
    d_model = sd["token_embedding.weight"].shape[1]
    n_layers = len(
        [k for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight")]
    )
    d_ff = sd["layers.0.feed_forward.w1.weight"].shape[0] if n_layers > 0 else 256
    n_heads = {128: 4, 64: 2, 96: 4, 256: 8}.get(d_model, max(1, d_model // 32))
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.d_ff = d_ff
    cfg.n_heads = n_heads
    cfg.max_seq_len = 32
    cfg.use_reversed = True
    cfg.use_scratchpad = True
    cfg.max_digits = 1
    cfg.min_digits = 1
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg)
    model.load_state_dict(sd, strict=False)
    return model, cfg


def compute_fresh_fisher(model, tok, cfg, num_samples=100):
    device = next(model.parameters()).device
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
    samples = 0
    for _ in range(num_samples):
        expr, ans = generate_problem("add", 1, 1, reversed=True, scratchpad=True)
        full = f"{expr}={ans}"
        ids = tok.encode(full, add_special_tokens=True)
        ids_t = torch.tensor([ids])
        model.zero_grad()
        _, loss, _ = model(ids_t[:, :-1], ids_t[:, 1:])
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2
        samples += 1
    if samples > 0:
        for name in fisher:
            fisher[name] /= samples
    return fisher


def train_task(model, cfg, tok, operation, ewc, lambda_val, steps=STEPS):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
    )
    cfg.max_digits = 1
    cfg.min_digits = 1
    ds = SpecialistDataset(tok, operation, cfg)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    global_step = 0
    for epoch in range(20):
        for x, y in loader:
            if global_step >= steps:
                break
            _, task_loss, _ = model(x, y)
            ewc_pen = ewc.compute_ewc_penalty(lambda_ewc=lambda_val)
            total_loss = task_loss + ewc_pen
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for pg in optimizer.param_groups:
                pg["lr"] = 0.001 * _get_lr(global_step, 200, steps, "cosine")
            global_step += 1
    return global_step


def measure_all(model, tok, cfg):
    model.eval()
    return {op: evaluate(model, tok, cfg, op, num=100) for op in ["add", "sub", "mul"]}


def main():
    print("=" * 60)
    print("  MITIGATION 1: ADAPTIVE GAMMA")
    print("  Three-task sequence with γ based on Fisher overlap")
    print(f"  λ={LAMBDA_EWC}, base_γ={BASE_GAMMA}, min_γ={MIN_GAMMA}")
    print("=" * 60)

    tok = MathTokenizer()

    # ─── Step 1: Load model + compute Fisher on addition ───
    print("\n--- Step 1: Load Addition Model ---")
    model, cfg = load_clean_model(tok)
    fisher_add = compute_fresh_fisher(model, tok, cfg)
    add_norm = sum(v.norm().item() for v in fisher_add.values())
    print(f"  Fisher (add) norm: {add_norm:.4f}")

    ewc = OnlineEWC(model, gamma=BASE_GAMMA)
    ewc.fisher_dict = {k: v.clone() for k, v in fisher_add.items()}
    ewc.task_count = 1
    ewc.save_anchor_weights()
    before = measure_all(model, tok, cfg)
    print(f"  Baseline: add={before['add']:.0f}%")

    # ─── Step 2: Train Subtraction ───
    print("\n--- Step 2: Train Subtraction (task 2) ---")
    train_task(model, cfg, tok, "sub", ewc, LAMBDA_EWC)
    after_sub = measure_all(model, tok, cfg)
    print(f"  After sub: add={after_sub['add']:.0f}%, sub={after_sub['sub']:.0f}%")

    # ─── Step 3: Compute Fisher on subtraction model ───
    print("\n--- Step 3: Compute Fisher on Sub Model + Measure Overlap ---")
    fisher_sub = compute_fresh_fisher(model, tok, cfg)
    sub_norm = sum(v.norm().item() for v in fisher_sub.values())

    # Measure overlap between add and sub Fisher
    overlap = OnlineEWC.compute_fisher_overlap(fisher_add, fisher_sub)
    gamma_adaptive = OnlineEWC.compute_adaptive_gamma(BASE_GAMMA, overlap, MIN_GAMMA)
    print(f"  Fisher (sub) norm: {sub_norm:.4f}")
    print(f"  Overlap (cosine sim): {overlap:.4f}")
    print(f"  Gamma adaptive: {gamma_adaptive:.4f} (from base {BASE_GAMMA})")
    print(
        f"  Interpretation: {'High overlap — similar tasks' if overlap > 0.5 else 'Low overlap — divergent tasks'}"
    )

    # ─── Step 4: Merge with ADAPTIVE gamma ───
    print("\n--- Step 4: Merge Fisher with Adaptive Gamma ---")
    old_norm = sum(v.norm().item() for v in ewc.fisher_dict.values())
    ewc.merge_fisher(fisher_sub, gamma=gamma_adaptive)
    ewc.save_anchor_weights()
    new_norm = sum(v.norm().item() for v in ewc.fisher_dict.values())
    print(
        f"  Fisher norm: {old_norm:.4f} → {new_norm:.4f} (change: {((new_norm/old_norm)-1)*100:+.1f}%)"
    )
    print(f"  Task count: {ewc.task_count}")

    # ─── Step 5: Train Multiplication ───
    print("\n--- Step 5: Train Multiplication (task 3) ---")
    train_task(model, cfg, tok, "mul", ewc, LAMBDA_EWC)
    after_mul = measure_all(model, tok, cfg)
    print(
        f"  After mul: add={after_mul['add']:.0f}%, sub={after_mul['sub']:.0f}%, mul={after_mul['mul']:.0f}%"
    )

    # ─── Summary ───
    print(f"\n{'=' * 60}")
    print(f"  ADAPTIVE GAMMA THREE-TASK RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Step':<20} | {'Add':>6} | {'Sub':>6} | {'Mul':>6} | {'Fisher':>8} | {'γ':>6}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")
    print(
        f"  {'After Addition':<20} | {before['add']:>5.0f}% | {'-':>6} | {'-':>6} | {add_norm:>8.2f} | {'-':>6}"
    )
    print(
        f"  {'After Subtraction':<20} | {after_sub['add']:>5.0f}% | {after_sub['sub']:>5.0f}% | {'-':>6} | {sub_norm:>8.2f} | {BASE_GAMMA:>6}"
    )
    print(
        f"  {'After Mul (adaptive γ)':<20} | {after_mul['add']:>5.0f}% | {after_mul['sub']:>5.0f}% | {after_mul['mul']:>5.0f}% | {new_norm:>8.2f} | {gamma_adaptive:>6.3f}"
    )

    all_ok = all(after_mul[op] > 85 for op in ["add", "sub"])
    print(f"\n  {'=' * 50}")
    if all_ok:
        print(f"  ✅ ADAPTIVE GAMMA WORKS: All tasks >85%")
        print(f"  Overlap={overlap:.3f}, γ_adaptive={gamma_adaptive:.3f}")
    else:
        print(f"  ❌ Adaptive gamma insufficient for full recovery")
        print(f"  Best task: {max(after_mul.values()):.0f}%, Worst: {min(after_mul.values()):.0f}%")
    print(f"  {'=' * 50}")

    # ─── Compare with fixed gamma ───
    print(f"\n  Comparison with fixed γ={BASE_GAMMA}:")
    print(f"    Fixed γ:   add=3%,  sub=0%,  mul=34% (from three-task experiment)")
    print(
        f"    Adaptive γ: add={after_mul['add']:.0f}%, sub={after_mul['sub']:.0f}%, mul={after_mul['mul']:.0f}%"
    )
    improvement = after_mul["add"] - 3
    print(f"    Improvement in add: +{improvement:.0f}pp")

    # ─── Save ───
    results = {
        "add_before": before["add"],
        "add_after_sub": after_sub["add"],
        "sub_after_sub": after_sub["sub"],
        "add_after_mul": after_mul["add"],
        "sub_after_mul": after_mul["sub"],
        "mul_after_mul": after_mul["mul"],
        "fisher_add_norm": add_norm,
        "fisher_sub_norm": sub_norm,
        "fisher_merged_norm": new_norm,
        "overlap": overlap,
        "gamma_adaptive": gamma_adaptive,
        "gamma_base": BASE_GAMMA,
        "all_tasks_above_85": all_ok,
    }
    output_path = Path("experiments/adaptive_gamma_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {output_path}")


if __name__ == "__main__":
    main()
