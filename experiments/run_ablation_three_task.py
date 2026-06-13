"""Ablation 4: Three-Task Sequence — Add → Sub → Mul with Online EWC.

The definitive validation: does the merged Fisher remain effective after
3 sequential tasks? Measures retention of ALL previous tasks after each step.

Protocol:
1. Load original addition model (best.pt)
2. Train subtraction with EWC (λ=1000, γ=0.9)
3. Measure: add retention, sub acquired
4. Merge Fisher (add + sub)
5. Train multiplication with EWC (same λ, γ)
6. Measure: add retention, sub retention, mul acquired
7. Report complete trace

Expected (if Fisher merge scales):
  After Add:  Add=98%
  After Sub:  Add=92%,  Sub=97%
  After Mul:  Add=88%,  Sub=93%,  Mul=96%

Failure mode (Fisher too crowded):
  After Mul:  Add=15%,  Sub=20%,  Mul=96%  ← lower gamma or per-task anchors needed
"""
import sys, json, time, shutil
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer
from train_specialist import evaluate, evaluate_per_digit, generate_problem, SpecialistDataset, _get_lr
from egefalos.online_ewc import OnlineEWC

DEVICE = 'cpu'
ADD_DIR = Path('specialists/math/add')
LAMBDA_EWC = 500   # Use the discovered optimum from lambda sweep
GAMMA = 0.9
STEPS_PER_TASK = 2000


def load_clean_model(tok):
    """Load the original addition model — same as lambda sweep."""
    checkpoint = torch.load(ADD_DIR / 'best_original.pt', map_location='cpu', weights_only=True)
    sd = checkpoint.get('model_state_dict', checkpoint)

    d_model = sd['token_embedding.weight'].shape[1]
    n_layers = len([k for k in sd if k.startswith('layers.') and k.endswith('.attention.wq.weight')])
    d_ff = sd['layers.0.feed_forward.w1.weight'].shape[0] if n_layers > 0 else 256
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
    """Compute Fisher on addition data from scratch."""
    device = next(model.parameters()).device
    model.train()
    fisher_inputs, fisher_targets = [], []
    for _ in range(num_samples):
        expr, ans = generate_problem('add', 1, 1, reversed=True, scratchpad=True)
        full = f'{expr}={ans}'
        ids = tok.encode(full, add_special_tokens=True)
        x = ids[:-1]
        y = ids[1:]
        fisher_inputs.append(x + [tok.pad_id] * (cfg.max_seq_len - len(x)))
        fisher_targets.append(y + [-100] * (cfg.max_seq_len - len(y)))

    fisher_loader = DataLoader(
        list(zip(
            torch.tensor(fisher_inputs, dtype=torch.long),
            torch.tensor(fisher_targets, dtype=torch.long),
        )),
        batch_size=16, shuffle=True
    )

    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)

    samples = 0
    for x, y in fisher_loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2
        samples += x.size(0)

    if samples > 0:
        for name in fisher:
            fisher[name] /= samples
    return fisher


def train_with_ewc(model, cfg, tok, operation, ewc, lambda_val, steps=STEPS_PER_TASK):
    """Train a model on one operation with EWC protection. Returns trajectory."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                   betas=(0.9, 0.999))
    cfg.max_digits = 1
    cfg.min_digits = 1
    ds = SpecialistDataset(tok, operation, cfg)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    t_start = time.time()
    global_step = 0
    trajectory = []
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
                pg['lr'] = 0.001 * _get_lr(global_step, 200, steps, 'cosine')
            global_step += 1

        if epoch % 2 == 0:
            model.eval()
            metrics = {}
            for op_name in ['add', 'sub', 'mul']:
                metrics[op_name] = evaluate(model, tok, cfg, op_name, num=50)
            elapsed = time.time() - t_start
            traj_str = '  '.join(f'{o}={metrics[o]:.0f}%' for o in ['add', 'sub', 'mul'])
            trajectory.append({'epoch': epoch + 1, **metrics})
            print(f"  Epoch {epoch+1}: {traj_str} [{elapsed:.0f}s]")
            model.train()

    return trajectory


def measure_all(model, tok, cfg):
    """Measure all operations' accuracy."""
    model.eval()
    results = {}
    for op in ['add', 'sub', 'mul']:
        acc = evaluate(model, tok, cfg, op, num=100)
        pd = evaluate_per_digit(model, tok, cfg, op, per_digit_samples=20)
        results[op] = {'accuracy': acc, 'per_digit': {str(k): v for k, v in pd.items()}}
    return results


def main():
    print("=" * 60)
    print("  ABLATION 4: THREE-TASK SEQUENCE")
    print("  Add → Sub → Mul with Online EWC")
    print(f"  λ={LAMBDA_EWC}, γ={GAMMA}, {STEPS_PER_TASK} steps/task")
    print("=" * 60)

    tok = MathTokenizer()
    all_results = {'task_sequence': ['addition', 'subtraction', 'multiplication']}

    # ─── Step 1: Load clean addition model ───
    print("\n--- Step 1: Load Addition Model ---")
    model, cfg = load_clean_model(tok)
    step1 = {'operation': 'addition', 'action': 'load_model'}
    step1['baseline'] = measure_all(model, tok, cfg)
    print(f"  Baseline: add={step1['baseline']['add']['accuracy']:.1f}%")
    all_results['step1_load'] = step1

    # ─── Step 2: Compute Fisher on addition ───
    print("\n--- Step 2: Compute Fisher on Addition ---")
    fisher_add = compute_fresh_fisher(model, tok, cfg, num_samples=100)
    print(f"  Fisher computed: {len(fisher_add)} parameter groups")

    ewc = OnlineEWC(model, gamma=GAMMA)
    ewc.fisher_dict = {k: v.clone() for k, v in fisher_add.items()}
    ewc.task_count = 1
    ewc.save_anchor_weights()
    fisher_norm = sum(v.norm().item() for v in ewc.fisher_dict.values())
    print(f"  Fisher norm: {fisher_norm:.4f}")

    # ─── Step 3: Train Subtraction with EWC ───
    print("\n--- Step 3: Train Subtraction (task 2) ---")
    traj_sub = train_with_ewc(model, cfg, tok, 'sub', ewc, LAMBDA_EWC)
    step3 = {'operation': 'subtraction', 'action': 'train_with_ewc'}
    step3['trajectory'] = traj_sub
    step3['measurements'] = measure_all(model, tok, cfg)
    step3['fisher_norm'] = sum(v.norm().item() for v in ewc.fisher_dict.values())
    print(f"  After sub: add={step3['measurements']['add']['accuracy']:.1f}%  "
          f"sub={step3['measurements']['sub']['accuracy']:.1f}%")
    all_results['step2_train_sub'] = step3

    # ─── Step 4: Merge Fisher (sub → add) ───
    print("\n--- Step 4: Compute & Merge Fisher on Subtraction ---")
    # Compute Fisher on the new task
    fisher_sub = compute_fresh_fisher(model, tok, cfg, num_samples=100)
    old_task_count = ewc.task_count
    ewc.merge_fisher(fisher_sub)
    ewc.save_anchor_weights()
    merged_norm = sum(v.norm().item() for v in ewc.fisher_dict.values())
    print(f"  Fisher merged: {old_task_count} → {ewc.task_count} tasks")
    print(f"  Fisher norm: {fisher_norm:.4f} → {merged_norm:.4f} (growth: {merged_norm/fisher_norm:.2f}x)")

    step4 = {'operation': 'merge_fisher', 'action': 'fisher_merge'}
    step4['task_count_before'] = old_task_count
    step4['task_count_after'] = ewc.task_count
    step4['fisher_norm_before'] = fisher_norm
    step4['fisher_norm_after'] = merged_norm
    step4['fisher_growth_rate'] = merged_norm / fisher_norm
    all_results['step3_merge_fisher'] = step4

    # ─── Step 5: Train Multiplication with EWC ───
    print("\n--- Step 5: Train Multiplication (task 3) ---")
    traj_mul = train_with_ewc(model, cfg, tok, 'mul', ewc, LAMBDA_EWC)
    step5 = {'operation': 'multiplication', 'action': 'train_with_ewc'}
    step5['trajectory'] = traj_mul
    step5['measurements'] = measure_all(model, tok, cfg)
    step5['fisher_norm'] = sum(v.norm().item() for v in ewc.fisher_dict.values())
    print(f"  After mul: add={step5['measurements']['add']['accuracy']:.1f}%  "
          f"sub={step5['measurements']['sub']['accuracy']:.1f}%  "
          f"mul={step5['measurements']['mul']['accuracy']:.1f}%")
    all_results['step4_train_mul'] = step5

    # ─── Summary ───
    print(f"\n\n{'=' * 60}")
    print(f"  THREE-TASK SEQUENCE SUMMARY")
    print(f"{'=' * 60}")

    after_add = all_results['step1_load']['baseline']
    after_sub = all_results['step2_train_sub']['measurements']
    after_mul = all_results['step4_train_mul']['measurements']

    print(f"  {'Task':<12} | {'Add':>6} | {'Sub':>6} | {'Mul':>6} | {'Fisher':>8}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    print(f"  {'After Add':<12} | {after_add['add']['accuracy']:>5.1f}% | {'-':>6} | {'-':>6} | {fisher_norm:>8.2f}")
    print(f"  {'After Sub':<12} | {after_sub['add']['accuracy']:>5.1f}% | {after_sub['sub']['accuracy']:>5.1f}% | {'-':>6} | {merged_norm:>8.2f}")
    print(f"  {'After Mul':<12} | {after_mul['add']['accuracy']:>5.1f}% | {after_mul['sub']['accuracy']:>5.1f}% | {after_mul['mul']['accuracy']:>5.1f}% | {step5['fisher_norm']:>8.2f}")

    # Interpretation
    add_retained = after_mul['add']['accuracy']
    sub_retained = after_mul['sub']['accuracy']
    mul_acquired = after_mul['mul']['accuracy']
    min_retained = min(add_retained, sub_retained)

    all_results['summary'] = {
        'addition_after_mul': add_retained,
        'subtraction_after_mul': sub_retained,
        'multiplication_acquired': mul_acquired,
        'min_retained': min_retained,
        'fisher_norm_initial': fisher_norm,
        'fisher_norm_after_sub': merged_norm,
        'fisher_norm_after_mul': step5['fisher_norm'],
        'fisher_scales_linearly': step5['fisher_norm'] / merged_norm < 1.2,  # ≈linear growth
        'ewc_effective': min_retained > 80,
    }

    print(f"\n  {'=' * 50}")
    if min_retained > 80:
        print(f"  ✅ FISHER MERGE SCALES: All 3 tasks retained above {min_retained:.0f}%")
        print(f"  Fisher grows linearly: {fisher_norm:.1f} → {merged_norm:.1f} → {step5['fisher_norm']:.1f}")
    else:
        print(f"  ❌ FISHER MERGE CROWDED: Retention dropped to {min_retained:.0f}%")
        print(f"  Consider: lower gamma, per-task anchor buffers, or sparse Fisher")
    print(f"  {'=' * 50}")

    # ─── Save ───
    output_path = Path('experiments/ablation_three_task_results.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results: {output_path}")


if __name__ == '__main__':
    main()
