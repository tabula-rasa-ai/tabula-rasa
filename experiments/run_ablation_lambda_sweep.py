"""Ablation 3: Lambda EWC Sensitivity Sweep (Self-Contained)

Each lambda run starts from the SAME original model (loaded from model.pt)
and computes Fisher FRESH — no cross-contamination between runs.

Lambda directly scales: (λ/2)·ΣF·(θ-θ*)² penalty.
λ = [100, 500, 1000, 2000]
"""
import sys, json, time, shutil
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from tokenizer import MathTokenizer
from model import MathTransformer
from train_specialist import evaluate, evaluate_per_digit, generate_problem, SpecialistDataset, _get_lr
from egefalos.online_ewc import OnlineEWC


LAMBDAS = [100, 250, 500, 1000, 2000]
STEPS = 2000
ADD_DIR = Path('specialists/math/add')


def get_model_and_config(tok):
    """Load the original addition model (best.pt — pre-EWC)."""
    checkpoint = torch.load(ADD_DIR / 'best.pt', map_location='cpu', weights_only=True)
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
    """Compute Fisher matrix from scratch on addition data — no prior contamination."""
    device = next(model.parameters()).device
    model.train()

    # Build a small fisher dataloader
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

    # Compute Fisher
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


def run_at_lambda(lambda_val, tok, cfg, device='cpu'):
    """Run a single sequential retention test with a specific lambda value.
    Starts from the ORIGINAL model every time — no contamination."""
    print(f"\n{'=' * 60}")
    print(f"  LAMBDA = {lambda_val}")
    print(f"{'=' * 60}")

    results = {'lambda': lambda_val, 'ewc_used': True}

    # ─── Load model FRESH from best.pt ───
    model, model_cfg = get_model_and_config(tok)
    model.train()

    # ─── Baseline ───
    add_baseline = evaluate(model, tok, model_cfg, 'add', num=100)
    results['addition_baseline_1d'] = add_baseline
    print(f"  Addition baseline (max_digits={model_cfg.max_digits}): {add_baseline:.1f}%")
    assert add_baseline > 50, f"Baseline too low ({add_baseline:.1f}%) — model or config broken"

    # ─── Compute Fisher FRESH ───
    print(f"  Computing Fisher matrix from scratch...")
    fisher = compute_fresh_fisher(model, tok, model_cfg)

    # ─── Set up EWC with fresh Fisher ───
    ewc = OnlineEWC(model, gamma=0.9)
    ewc.fisher_dict = {k: v.clone() for k, v in fisher.items()}
    ewc.task_count = 1
    ewc.save_anchor_weights()
    print(f"  EWC ready: {len(ewc.fisher_dict)} fisher params, λ={lambda_val}")

    # ─── Train subtraction WITH EWC ───
    print(f"\n  Training subtraction WITH EWC (λ={lambda_val})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                   betas=(0.9, 0.999))

    t_start = time.time()
    sub_ds = SpecialistDataset(tok, 'sub', model_cfg)
    loader = DataLoader(sub_ds, batch_size=32, shuffle=True, drop_last=True)

    global_step = 0
    trajectory = []
    for epoch in range(20):
        for x, y in loader:
            if global_step >= STEPS:
                break
            _, task_loss, _ = model(x, y)
            ewc_pen = ewc.compute_ewc_penalty(lambda_ewc=lambda_val)
            total_loss = task_loss + ewc_pen
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for pg in optimizer.param_groups:
                pg['lr'] = 0.001 * _get_lr(global_step, 200, STEPS, 'cosine')
            global_step += 1

        if epoch % 2 == 0:
            model.eval()
            add_a = evaluate(model, tok, model_cfg, 'add', num=50)
            sub_a = evaluate(model, tok, model_cfg, 'sub', num=50)
            elapsed = time.time() - t_start
            trajectory.append({'epoch': epoch + 1, 'add': add_a, 'sub': sub_a})
            print(f"  Epoch {epoch+1}: add={add_a:.1f}% sub={sub_a:.1f}% [{elapsed:.0f}s]")
            model.train()

    results['trajectory'] = trajectory
    results['total_steps'] = global_step

    # ─── Final measurements ───
    model.eval()
    add_final = evaluate(model, tok, model_cfg, 'add', num=100)
    sub_final = evaluate(model, tok, model_cfg, 'sub', num=100)
    per_digit = evaluate_per_digit(model, tok, model_cfg, 'add', per_digit_samples=20)

    results['addition_final_1d'] = add_final
    results['subtraction_final_1d'] = sub_final
    results['retention_drop_pp'] = add_baseline - add_final
    results['retention_preserved'] = (add_baseline - add_final) < 5.0
    results['per_digit_final'] = {str(k): v for k, v in per_digit.items()}

    print(f"\n  λ={lambda_val}: add={add_baseline:.1f}% → {add_final:.1f}% "
          f"(drop={results['retention_drop_pp']:+.1f}pp), sub={sub_final:.1f}%")

    return results


def main():
    print("=" * 60)
    print("  ABLATION 3: LAMBDA EWC SENSITIVITY SWEEP")
    print(f"  Testing λ = {LAMBDAS}")
    print("  Each run starts from model.pt with fresh Fisher (no cross-contamination)")
    print("=" * 60)

    device = 'cpu'
    tok = MathTokenizer()
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    print("  Each run starts from clean best.pt with fresh Fisher (no cross-contamination)")
    print(f"  Model loaded from: {ADD_DIR / 'best_original.pt'}")

    # Verify best.pt exists
    if not (ADD_DIR / 'best.pt').exists():
        print(f"  ERROR: {ADD_DIR / 'best.pt'} not found!")
        print("  Run 'python3 train_specialist.py add' first to train the baseline.")
        return

    # Back up the original if not already done
    orig_best = ADD_DIR / 'best.pt'
    bak_best = ADD_DIR / 'best_original.pt'
    if not bak_best.exists():
        import shutil
        shutil.copy2(str(orig_best), str(bak_best))
        print(f"  Backed up original to {bak_best}")

    all_results = {}
    for lr_val in LAMBDAS:
        result = run_at_lambda(lr_val, tok, cfg, device)
        all_results[str(lr_val)] = result

    # ─── Summary ───
    print(f"\n\n{'=' * 60}")
    print(f"  LAMBDA SWEEP SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'λ':>6} | {'Add Baseline':>13} | {'Add Final':>10} | {'Drop':>7} | {'Sub Final':>10}")
    print(f"  {'-'*6}-+-{'-'*13}-+-{'-'*10}-+-{'-'*7}-+-{'-'*10}")

    best_combined = 0
    best_lambda = None
    for lr_val in LAMBDAS:
        r = all_results[str(lr_val)]
        drop = r['retention_drop_pp']
        drop_str = f"{drop:+.1f}pp"
        combined = r['addition_final_1d'] + r['subtraction_final_1d']
        if combined > best_combined:
            best_combined = combined
            best_lambda = lr_val
        print(f"  {lr_val:>6} | {r['addition_baseline_1d']:>12.1f}% | {r['addition_final_1d']:>9.1f}% | {drop_str:>7} | {r['subtraction_final_1d']:>9.1f}%")

    print(f"\n  Best λ by combined accuracy: {best_lambda} "
          f"(add={all_results[str(best_lambda)]['addition_final_1d']:.1f}% + "
          f"sub={all_results[str(best_lambda)]['subtraction_final_1d']:.1f}% = {best_combined:.1f}%)")

    # ─── Save ───
    output_path = Path('experiments/ablation_lambda_sweep_results.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {output_path}")


if __name__ == '__main__':
    main()
