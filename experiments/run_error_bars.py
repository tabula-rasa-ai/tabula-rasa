"""Run error-bar experiments: 3 seeds each for critical conditions.

Runs sequentially (each ~25 min) to collect mean ± std for paper tables.
Targets:
  1. EWC sequential retention (add→sub, λ=1000) — 3 seeds
  2. No-EWC control (add→sub without EWC) — 3 seeds
  3. Lambda=500 (optimal, add→sub) — 3 seeds

Total: 9 runs × ~25 min ≈ 3.75 hours
"""
import sys, json, time, random
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer
from train_specialist import evaluate, evaluate_per_digit, generate_problem, SpecialistDataset, _get_lr
from egefalos.online_ewc import OnlineEWC

ADD_DIR = Path('specialists/math/add')
DEVICE = 'cpu'
STEPS = 2000
NUM_SEEDS = 3
SEEDS = [42, 123, 256]


def get_model_and_config(tok):
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


def train_subtraction(model, cfg, tok, use_ewc, lambda_val, seed):
    """Train subtraction with or without EWC. Returns (add_final, sub_final, trajectory)."""
    random.seed(seed)
    torch.manual_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                   betas=(0.9, 0.999))

    if use_ewc:
        fisher = compute_fresh_fisher(model, tok, cfg)
        ewc = OnlineEWC(model, gamma=0.9)
        ewc.fisher_dict = {k: v.clone() for k, v in fisher.items()}
        ewc.task_count = 1
        ewc.save_anchor_weights()

    cfg.max_digits = 1
    cfg.min_digits = 1
    ds = SpecialistDataset(tok, 'sub', cfg)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    global_step = 0
    trajectory = []
    t_start = time.time()
    for epoch in range(20):
        for x, y in loader:
            if global_step >= STEPS:
                break
            _, task_loss, _ = model(x, y)
            if use_ewc:
                ewc_pen = ewc.compute_ewc_penalty(lambda_ewc=lambda_val)
                total_loss = task_loss + ewc_pen
            else:
                total_loss = task_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for pg in optimizer.param_groups:
                pg['lr'] = 0.001 * _get_lr(global_step, 200, STEPS, 'cosine')
            global_step += 1
        if epoch % 4 == 0:
            model.eval()
            add_a = evaluate(model, tok, cfg, 'add', num=50)
            sub_a = evaluate(model, tok, cfg, 'sub', num=50)
            trajectory.append({'epoch': epoch+1, 'add': add_a, 'sub': sub_a})
            model.train()

    model.eval()
    add_final = evaluate(model, tok, cfg, 'add', num=100)
    sub_final = evaluate(model, tok, cfg, 'sub', num=100)
    return add_final, sub_final, trajectory, global_step


def run_experiment(label, use_ewc, lambda_val):
    """Run one condition across NUM_SEEDS seeds."""
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'  λ={lambda_val}, EWC={use_ewc}, seeds={SEEDS}')
    print(f'{"="*60}')

    tok = MathTokenizer()
    results = []
    for seed in SEEDS:
        print(f'\n--- Seed {seed} ---')
        model, cfg = get_model_and_config(tok)
        model.train()

        # Baseline
        add_base = evaluate(model, tok, cfg, 'add', num=100)
        print(f'  Baseline add: {add_base:.1f}%')

        # Train
        add_final, sub_final, traj, steps = train_subtraction(model, cfg, tok, use_ewc, lambda_val, seed)
        results.append({
            'seed': seed,
            'addition_baseline_1d': add_base,
            'addition_final_1d': add_final,
            'subtraction_final_1d': sub_final,
            'retention_drop_pp': add_base - add_final,
            'steps': steps,
            'trajectory': traj,
        })
        print(f'  add={add_base:.0f}%→{add_final:.0f}%, sub={sub_final:.0f}%, steps={steps}')

    # Compute stats
    add_finals = [r['addition_final_1d'] for r in results]
    sub_finals = [r['subtraction_final_1d'] for r in results]
    drops = [r['retention_drop_pp'] for r in results]

    import statistics
    stats = {
        'label': label,
        'use_ewc': use_ewc,
        'lambda_val': lambda_val,
        'num_seeds': NUM_SEEDS,
        'seeds': SEEDS,
        'results': results,
        'addition_mean': statistics.mean(add_finals),
        'addition_std': statistics.stdev(add_finals) if len(add_finals) > 1 else 0,
        'subtraction_mean': statistics.mean(sub_finals),
        'subtraction_std': statistics.stdev(sub_finals) if len(sub_finals) > 1 else 0,
        'retention_drop_mean': statistics.mean(drops),
        'retention_drop_std': statistics.stdev(drops) if len(drops) > 1 else 0,
    }
    return stats


def main():
    print('=' * 60)
    print('  ERROR BAR EXPERIMENTS')
    print(f'  {NUM_SEEDS} seeds × 3 conditions = {NUM_SEEDS*3} runs')
    print(f'  Estimated: ~{NUM_SEEDS*3*25//60}h{NUM_SEEDS*3*25%60}m')
    print('=' * 60)

    all_results = {}

    # 1. EWC sequential retention (λ=1000)
    all_results['ewc_retention'] = run_experiment('EWC Retention', use_ewc=True, lambda_val=1000)

    # 2. No-EWC control
    all_results['no_ewc_control'] = run_experiment('No-EWC Control', use_ewc=False, lambda_val=0)

    # 3. λ=500 optimal
    all_results['lambda_500'] = run_experiment('Lambda=500 (Optimal)', use_ewc=True, lambda_val=500)

    # ─── Summary ───
    print(f'\n\n{"="*60}')
    print(f'  ERROR BAR SUMMARY')
    print(f'{"="*60}')

    print(f'\n  {"Condition":<25} | {"Add Mean":>9} | {"Add Std":>8} | {"Sub Mean":>9} | {"Sub Std":>8}')
    print(f'  {"-"*25}-+-{"-"*9}-+-{"-"*8}-+-{"-"*9}-+-{"-"*8}')
    for key, r in all_results.items():
        print(f'  {r["label"]:<25} | {r["addition_mean"]:>7.1f}%  | ±{r["addition_std"]:.1f}% | {r["subtraction_mean"]:>7.1f}%  | ±{r["subtraction_std"]:.1f}%')

    # ─── Format for paper ───
    print(f'\n\n  --- PAPER TABLE ENTRIES ---')
    for key, r in all_results.items():
        print(f'\n  {r["label"]}:')
        print(f'    Addition:  {r["addition_mean"]:.1f} ± {r["addition_std"]:.1f}%')
        print(f'    Subtraction: {r["subtraction_mean"]:.1f} ± {r["subtraction_std"]:.1f}%')
        print(f'    Retention drop: {r["retention_drop_mean"]:.1f} ± {r["retention_drop_std"]:.1f} pp')

    # ─── Save ───
    output_path = Path('experiments/error_bar_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Saved: {output_path}')


if __name__ == '__main__':
    main()
