"""Ablation 1: No-EWC Control — Train subtraction WITHOUT EWC protection.

Hypothesis: Without EWC, addition accuracy should drop to ~30-50%,
proving EWC is essential for catastrophic forgetting prevention.

Protocol:
1. Load existing addition model (best.pt — 83% 1-digit baseline)
2. Measure baseline addition accuracy
3. Train on subtraction WITHOUT EWC (2000 steps, same config as EWC run)
4. Measure addition retention + subtraction acquired
"""
import sys, json, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer
from train_specialist import evaluate, evaluate_per_digit, generate_problem, SpecialistDataset, _make_optimizer, _get_lr


def main():
    print("=" * 60)
    print("  ABLATION 1: NO-EWC CONTROL")
    print("  Train subtraction WITHOUT EWC - measure forgetting")
    print("=" * 60)

    device = 'cpu'
    tok = MathTokenizer()
    add_dir = Path('specialists/math/add')
    results = {}

    # ─── Load addition model from best.pt ───
    print("\n--- Loading addition checkpoint (best.pt) ---")
    checkpoint = torch.load(add_dir / 'best.pt', map_location='cpu', weights_only=True)
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
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    model.load_state_dict(sd, strict=False)
    model.train()

    # ─── Phase 1: Baseline accuracy ───
    print("\n--- Phase 1: Baseline ---")
    cfg.max_digits = 1
    cfg.min_digits = 1
    add_baseline = evaluate(model, tok, cfg, 'add', num=100)
    results['addition_baseline_1d'] = add_baseline
    print(f"  1-digit addition: {add_baseline:.1f}%")

    per_digit = evaluate_per_digit(model, tok, cfg, 'add', per_digit_samples=20)
    digit_str = ' '.join(f'{d}d:{v:.0f}%' for d, v in per_digit.items())
    print(f"  Per digit (add, before): {digit_str}")
    results['per_digit_baseline'] = {str(k): v for k, v in per_digit.items()}

    # ─── Phase 2: Train on subtraction WITHOUT EWC ───
    print("\n--- Phase 2: Train Subtraction WITHOUT EWC ---")
    print("  ⚠️  No Fisher matrix, no EWC penalty, no protection")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                   betas=(0.9, 0.999))

    steps = 2000
    t_start = time.time()

    # Build subtraction dataset
    from train_specialist import SpecialistDataset as SpecDS
    cfg.max_digits = 1
    cfg.min_digits = 1
    sub_ds = SpecDS(tok, 'sub', cfg)
    loader = DataLoader(sub_ds, batch_size=32, shuffle=True, drop_last=True)

    global_step = 0
    for epoch in range(20):
        for x, y in loader:
            if global_step >= steps:
                break

            # NO EWC — just task loss
            _, task_loss, _ = model(x, y)

            optimizer.zero_grad()
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for pg in optimizer.param_groups:
                pg['lr'] = 0.001 * _get_lr(global_step, 200, steps, 'cosine')

            global_step += 1

        if epoch % 2 == 0:
            model.eval()
            add_a = evaluate(model, tok, cfg, 'add', num=50)
            sub_a = evaluate(model, tok, cfg, 'sub', num=50)
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1}: add={add_a:.1f}% sub={sub_a:.1f}% [{elapsed:.0f}s]")
            model.train()

    elapsed = time.time() - t_start
    print(f"  Training done: {elapsed:.0f}s, {global_step} steps")

    # ─── Phase 3: Final measurements ───
    print("\n--- Phase 3: Results ---")
    model.eval()

    add_final = evaluate(model, tok, cfg, 'add', num=100)
    sub_final = evaluate(model, tok, cfg, 'sub', num=100)
    per_digit_final = evaluate_per_digit(model, tok, cfg, 'add', per_digit_samples=20)
    digit_str_final = ' '.join(f'{d}d:{v:.0f}%' for d, v in per_digit_final.items())

    results['addition_final_1d'] = add_final
    results['subtraction_final_1d'] = sub_final
    results['retention_drop_pp'] = add_baseline - add_final
    results['retention_preserved'] = (add_baseline - add_final) < 5.0
    results['per_digit_final'] = {str(k): v for k, v in per_digit_final.items()}
    results['ewc_used'] = False
    results['total_steps'] = global_step

    print(f"\n  === COMPARISON ===")
    print(f"  Addition baseline:          {add_baseline:.1f}%")
    print(f"  After NO-EWC + subtraction: {add_final:.1f}%")
    print(f"  Retention drop:             {results['retention_drop_pp']:.1f} pp")
    print(f"  Subtraction acquired:       {sub_final:.1f}%")
    print(f"  Per digit (add, after):     {digit_str_final}")
    print(f"  Retention preserved (<5pp): {'YES ✅' if results['retention_preserved'] else 'NO ❌'}")

    # Comparison with EWC run
    ewc_path = Path('experiments/sequential_retention_results.json')
    if ewc_path.exists():
        try:
            with open(ewc_path) as f:
                ewc_results = json.load(f)
            print(f"\n  === VS EWC RUN ===")
            print(f"  With EWC:    add={ewc_results.get('addition_with_ewc', ewc_results.get('addition_final_1d', '?'))}%  sub={ewc_results.get('subtraction_acquired', ewc_results.get('subtraction_final_1d', '?'))}%")
            print(f"  Without EWC: add={add_final:.1f}%  sub={sub_final:.1f}%")
            results['ewc_comparison_add'] = ewc_results.get('addition_with_ewc', ewc_results.get('addition_final_1d', None))
            results['ewc_comparison_sub'] = ewc_results.get('subtraction_acquired', ewc_results.get('subtraction_final_1d', None))
        except (json.JSONDecodeError, KeyError):
            pass

    # ─── Save ───
    output_path = Path('experiments/ablation_no_ewc_results.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {output_path}")

    # Echo the key finding
    drop = results['retention_drop_pp']
    print(f"\n{'=' * 60}")
    if drop >= 5.0:
        print(f"  ✅ KEY FINDING: Addition dropped {drop:.1f} pp — EWC IS ESSENTIAL")
    else:
        print(f"  ❌ Addition dropped only {drop:.1f} pp — forgetting less than expected")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
