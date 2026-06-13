"""Adaptive Gamma — additional seeds for reliability.
Runs 2 additional seeds to complement the first run (already in progress).
"""
import sys, json, time, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from tokenizer import MathTokenizer
from model import MathTransformer
from train_specialist import evaluate, generate_problem, SpecialistDataset, _get_lr
from egefalos.online_ewc import OnlineEWC

DEVICE = 'cpu'
ADD_DIR = Path('specialists/math/add')
LAMBDA_EWC = 500
BASE_GAMMA = 0.9
MIN_GAMMA = 0.3
STEPS = 2000
SEEDS = [123, 256]  # Seed 42 is already running


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


def train_task(model, cfg, tok, operation, ewc, lambda_val, steps=STEPS):
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
            ewc_pen = ewc.compute_ewc_penalty(lambda_ewc=lambda_val)
            total_loss = task_loss + ewc_pen
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
    random.seed(seed)
    torch.manual_seed(seed)
    tok = MathTokenizer()
    model, cfg = load_clean_model(tok)

    # Step 1: Fisher on addition
    fisher_add = compute_fisher(model, tok, cfg)
    add_norm = sum(v.norm().item() for v in fisher_add.values())

    ewc = OnlineEWC(model, gamma=BASE_GAMMA)
    ewc.fisher_dict = {k: v.clone() for k, v in fisher_add.items()}
    ewc.task_count = 1
    ewc.save_anchor_weights()
    before = measure_all(model, tok, cfg)

    # Step 2: Train subtraction
    train_task(model, cfg, tok, 'sub', ewc, LAMBDA_EWC)
    after_sub = measure_all(model, tok, cfg)

    # Step 3: Fisher on sub model + overlap
    fisher_sub = compute_fisher(model, tok, cfg)
    sub_norm = sum(v.norm().item() for v in fisher_sub.values())
    overlap = OnlineEWC.compute_fisher_overlap(fisher_add, fisher_sub)
    gamma_ad = OnlineEWC.compute_adaptive_gamma(BASE_GAMMA, overlap, MIN_GAMMA)

    old_norm = sum(v.norm().item() for v in ewc.fisher_dict.values())
    ewc.merge_fisher(fisher_sub, gamma=gamma_ad)
    ewc.save_anchor_weights()
    new_norm = sum(v.norm().item() for v in ewc.fisher_dict.values())

    # Step 4: Train multiplication
    train_task(model, cfg, tok, 'mul', ewc, LAMBDA_EWC)
    after_mul = measure_all(model, tok, cfg)

    return {
        'seed': seed,
        'add_before': before['add'],
        'add_after_sub': after_sub['add'],
        'sub_after_sub': after_sub['sub'],
        'add_after_mul': after_mul['add'],
        'sub_after_mul': after_mul['sub'],
        'mul_after_mul': after_mul['mul'],
        'fisher_add_norm': add_norm,
        'fisher_sub_norm': sub_norm,
        'fisher_merged_norm': new_norm,
        'overlap': overlap,
        'gamma_adaptive': gamma_ad,
    }


def main():
    print("=" * 60)
    print("  ADAPTIVE GAMMA — Additional Seeds")
    print(f"  Seeds: {SEEDS}")
    print(f"  λ={LAMBDA_EWC}, base_γ={BASE_GAMMA}")
    print("=" * 60)

    all_results = [run_seed(s) for s in SEEDS]

    import statistics
    def stats(vals):
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0

    print(f'\n{"="*60}')
    print('  RESULTS')
    print(f'{"="*60}')
    for r in all_results:
        print(f'  Seed {r["seed"]}: add={r["add_after_mul"]:.0f}%, sub={r["sub_after_mul"]:.0f}%, mul={r["mul_after_mul"]:.0f}%, γ={r["gamma_adaptive"]:.3f}, overlap={r["overlap"]:.3f}')

    add_vals = [r['add_after_mul'] for r in all_results]
    sub_vals = [r['sub_after_mul'] for r in all_results]
    mul_vals = [r['mul_after_mul'] for r in all_results]
    m_a, s_a = stats(add_vals)
    m_s, s_s = stats(sub_vals)
    m_m, s_m = stats(mul_vals)
    print(f'\n  Mean ± Std:')
    print(f'  Add: {m_a:.1f} ± {s_a:.1f}%')
    print(f'  Sub: {m_s:.1f} ± {s_s:.1f}%')
    print(f'  Mul: {m_m:.1f} ± {s_m:.1f}%')
    print(f'  Overlap mean: {statistics.mean([r["overlap"] for r in all_results]):.3f}')

    # Save
    output = {'seeds': SEEDS, 'results': all_results, 'add_mean': m_a, 'add_std': s_a, 'sub_mean': m_s, 'sub_std': s_s, 'mul_mean': m_m, 'mul_std': s_m}
    out_path = Path('experiments/adaptive_gamma_seeds.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n  Results: {out_path}')


if __name__ == '__main__':
    main()
