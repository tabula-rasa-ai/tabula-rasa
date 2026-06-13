"""Standard EWC (non-merged) baseline for three-task sequence.

Stores per-task Fisher matrices and applies separate EWC penalties.
This isolates whether the collapse at task 3 is caused by the MERGE
or by something else (model capacity, λ, task difficulty).

Standard EWC: 3 tasks, each with its own Fisher + anchor.
Penalty = λ · Σ(F₁ · (θ−θ*₁)² + F₂ · (θ−θ*₂)² + ...)

If Standard EWC preserves >85% on all tasks: the merge IS the problem.
If Standard EWC also collapses: the problem is model capacity or λ.
"""
import sys, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from tokenizer import MathTokenizer
from model import MathTransformer
from train_specialist import evaluate, evaluate_per_digit, generate_problem, SpecialistDataset, _get_lr
from egefalos.online_ewc import OnlineEWC

DEVICE = 'cpu'
ADD_DIR = Path('specialists/math/add')
LAMBDA_EWC = 500
GAMMA = 0.9
STEPS = 2000
SEEDS = [42, 123, 256]


class StandardEWC:
    """Standard (non-merged) EWC — separate Fisher per task."""

    def __init__(self, model, lambda_ewc=500):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_list = []  # list of fisher_dicts
        self.anchor_list = []  # list of anchor_dicts

    def add_task(self, fisher, anchor):
        self.fisher_list.append({k: v.clone() for k, v in fisher.items()})
        self.anchor_list.append({k: v.clone() for k, v in anchor.items()})

    def compute_penalty(self):
        if not self.fisher_list:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        device = next(self.model.parameters()).device
        total = torch.tensor(0.0, device=device)
        for fisher, anchor in zip(self.fisher_list, self.anchor_list):
            pen = torch.tensor(0.0, device=device)
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in fisher and name in anchor:
                    f = fisher[name].to(device)
                    delta = param - anchor[name].to(device)
                    pen += (f * delta ** 2).sum()
            total += pen
        return (self.lambda_ewc / 2.0) * total


def load_clean_model(tok):
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


def compute_fisher(model, tok, cfg, num_samples=100):
    device = next(model.parameters()).device
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
    samples = 0
    for _ in range(num_samples):
        expr, ans = generate_problem('add', 1, 1, reversed=True, scratchpad=True)
        full = f'{expr}={ans}'
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


def get_anchor(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def train_task(model, cfg, tok, operation, ewc, steps=STEPS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
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
            penalty = ewc.compute_penalty()
            total_loss = task_loss + penalty
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for pg in optimizer.param_groups:
                pg['lr'] = 0.001 * _get_lr(global_step, 200, steps, 'cosine')
            global_step += 1
    return global_step


def measure_all(model, tok, cfg):
    model.eval()
    return {op: evaluate(model, tok, cfg, op, num=100) for op in ['add', 'sub', 'mul']}


def run_seed(seed):
    """Run one seed of Standard EWC three-task sequence."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n  --- Seed {seed} ---')
    tok = MathTokenizer()
    model, cfg = load_clean_model(tok)
    std_ewc = StandardEWC(model, lambda_ewc=LAMBDA_EWC)

    results = {'seed': seed}

    # ─── Addition baseline ───
    before = measure_all(model, tok, cfg)
    results['after_addition'] = before
    print(f'  Baseline: add={before["add"]:.0f}%')

    # Compute Fisher on addition model
    fisher_add = compute_fisher(model, tok, cfg)
    anchor_add = get_anchor(model)
    std_ewc.add_task(fisher_add, anchor_add)

    # ─── Train subtraction ───
    train_task(model, cfg, tok, 'sub', std_ewc)
    after_sub = measure_all(model, tok, cfg)
    results['after_subtraction'] = after_sub
    print(f'  After sub: add={after_sub["add"]:.0f}%, sub={after_sub["sub"]:.0f}%')

    # Compute Fisher on subtraction model + add as second task
    fisher_sub = compute_fisher(model, tok, cfg)
    anchor_sub = get_anchor(model)
    std_ewc.add_task(fisher_sub, anchor_sub)

    # ─── Train multiplication ───
    train_task(model, cfg, tok, 'mul', std_ewc)
    after_mul = measure_all(model, tok, cfg)
    results['after_multiplication'] = after_mul
    print(f'  After mul: add={after_mul["add"]:.0f}%, sub={after_mul["sub"]:.0f}%, mul={after_mul["mul"]:.0f}%')

    return results


def main():
    print("=" * 60)
    print("  STANDARD EWC BASELINE (non-merged)")
    print("  3 tasks × 3 seeds = 9 runs")
    print(f"  λ={LAMBDA_EWC}")
    print("=" * 60)

    all_seeds = []
    for seed in SEEDS:
        r = run_seed(seed)
        all_seeds.append(r)

    # Stats
    def avg_std(vals):
        import statistics
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0

    print(f'\n\n{"=" * 60}')
    print('  STANDARD EWC — SUMMARY')
    print(f'{"=" * 60}')

    for task in ['add', 'sub', 'mul']:
        after_add = [s['after_addition'][task] for s in all_seeds]
        after_sub = [s['after_subtraction'][task] for s in all_seeds]
        after_mul = [s['after_multiplication'][task] for s in all_seeds]
        m1, s1 = avg_std(after_add)
        m2, s2 = avg_std(after_sub)
        m3, s3 = avg_std(after_mul)
        print(f'  {task:<5} | After add: {m1:>5.1f}±{s1:.1f} | After sub: {m2:>5.1f}±{s2:.1f} | After mul: {m3:>5.1f}±{s3:.1f}')

    # Compare with merged EWC (from three-task experiment)
    print(f'\n  Comparison with Merged EWC (previous experiment):')
    merged_add = [3.0]  # from three-task experiment
    merged_sub = [0.0]
    merged_mul = [34.0]
    std_add = [s['after_multiplication']['add'] for s in all_seeds]
    std_sub = [s['after_multiplication']['sub'] for s in all_seeds]
    std_mul = [s['after_multiplication']['mul'] for s in all_seeds]
    m_sa, _ = avg_std(std_add)
    m_ss, _ = avg_std(std_sub)
    m_sm, _ = avg_std(std_mul)
    print(f'  {"":<8} | {"Merged":>8} | {"Standard":>8} | {"Delta":>8}')
    print(f'  {"Add":<8} | {"3.0%":>8} | {m_sa:>7.1f}% | {m_sa-3:>+7.1f}pp')
    print(f'  {"Sub":<8} | {"0.0%":>8} | {m_ss:>7.1f}% | {m_ss-0:>+7.1f}pp')
    print(f'  {"Mul":<8} | {"34.0%":>8} | {m_sm:>7.1f}% | {m_sm-34:>+7.1f}pp')

    # Interpretation
    if m_sa > 85 and m_ss > 85:
        print(f'\n  ✅ STANDARD EWC RECOVERS >85% — the merge IS the cause of collapse')
    elif m_sa > 50:
        print(f'\n  ⚠️  Standard EWC partially recovers (>{m_sa:.0f}%) — merge is a factor, but λ or capacity also matter')
    else:
        print(f'\n  ❌ Standard EWC also collapses — the problem is not the merge, it\'s capacity or λ')

    # Save
    output = {
        'seeds': SEEDS,
        'lambda_ewc': LAMBDA_EWC,
        'results': all_seeds,
    }
    std_add_vals = [s['after_multiplication']['add'] for s in all_seeds]
    std_sub_vals = [s['after_multiplication']['sub'] for s in all_seeds]
    std_mul_vals = [s['after_multiplication']['mul'] for s in all_seeds]
    m_a, s_a = avg_std(std_add_vals)
    m_s, s_s = avg_std(std_sub_vals)
    m_m, s_m = avg_std(std_mul_vals)

    output['summary'] = {
        'add_mean': m_a, 'add_std': s_a,
        'sub_mean': m_s, 'sub_std': s_s,
        'mul_mean': m_m, 'mul_std': s_m,
    }

    import json
    out_path = Path('experiments/standard_ewc_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n  Results: {out_path}')


if __name__ == '__main__':
    main()
