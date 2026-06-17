"""Train an algebra specialist — solves linear equations symbolically.

Demonstrates that Tabula Rasa's transformer framework extends beyond
pure arithmetic to symbolic algebra (solving for x).

Usage:
    python3 scripts/train_algebra.py                     # Full training
    python3 scripts/train_algebra.py --quick              # Smoke test
    python3 scripts/train_algebra.py --type linear        # Specific problem type
"""

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tabula_rasa.algebra_dataset import AlgebraDataset, AlgebraTokenizer, evaluate_algebra
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters


def train_algebra(opts):
    """Train a transformer on algebraic equation solving."""
    cfg = Config()
    cfg.max_seq_len = 32
    cfg.d_model = opts.d_model
    cfg.n_layers = opts.n_layers
    cfg.n_heads = opts.n_heads
    cfg.d_ff = opts.d_ff
    cfg.max_steps = opts.steps
    cfg.batch_size = opts.batch
    cfg.learning_rate = opts.lr
    cfg.use_reversed = False
    cfg.use_loss_masking = True
    cfg.eval_every = max(100, opts.steps // 10)
    cfg.save_every = max(500, opts.steps // 5)
    cfg.eval_samples = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizer with algebra variables
    tok = AlgebraTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    # Dataset
    train_ds = AlgebraDataset(
        num_samples=cfg.max_steps * opts.samples_per_step,
        problem_type=opts.type,
        max_coeff=opts.max_coeff,
        max_constant=opts.max_const,
    )

    print(f'\n{"="*60}')
    print('  ALGEBRA SPECIALIST — Solving for x')
    print(f'  Problem type: {opts.type}')
    print(f'  Device: {device}')
    print(f'  Steps: {cfg.max_steps} | Batch: {cfg.batch_size} | LR: {cfg.learning_rate}')
    print(f'  Arch: d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads}, ff={cfg.d_ff}')
    print(f'  Tokenizer: {cfg.vocab_size} tokens')
    print(f'  Model: {count_parameters.__name__} params')
    print(f'  Coeff max: {opts.max_coeff} | Constant max: {opts.max_const}')
    print(f'{"="*60}')

    # Model
    model = MathTransformer(cfg)
    model.to(device)
    total_params = count_parameters(model)
    print(f'  Model: {total_params:,} params')

    if opts.eval_only:
        ckpt = torch.load(opts.eval_only, map_location=device, weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
        print(f'  Loaded checkpoint: {opts.eval_only}')
        results = evaluate_algebra(model, tok, train_ds, num_samples=100)
        print(f'  Overall accuracy: {results["overall"]:.1f}%')
        for ptype, acc in sorted(results['per_type'].items()):
            print(f'    {ptype}: {acc:.1f}%')
        return

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    # Data preparation
    def collate_fn(batch):
        xs, ys = zip(*batch)
        max_len = max(max(len(x), len(y)) for x, y in batch)
        pad_id = tok.pad_id
        padded_x = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_id
        padded_y = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_id
        for i, (x, y) in enumerate(zip(xs, ys)):
            padded_x[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            padded_y[i, :len(y)] = torch.tensor(y, dtype=torch.long)
        return padded_x, padded_y

    def _infinite_batches(dataset, batch_size):
        while True:
            items = [(tok.encode(p, add_special_tokens=True),
                      tok.encode(a, add_special_tokens=True))
                     for p, a in [dataset[i] for i in range(min(2000, len(dataset)))]]
            loader = DataLoader(items, batch_size=batch_size,
                                shuffle=True, drop_last=True, collate_fn=collate_fn)
            yield from loader

    batch_iter = _infinite_batches(train_ds, cfg.batch_size)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training
    best_acc = 0.0
    t0 = time.time()
    recent_losses = []

    def sigint_handler(sig, frame):
        print(f'\n  [Interrupt] Saving checkpoint at step {step}...')
        Path('specialists/algebra').mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'global_step': step,
            'best_acc': best_acc,
        }, ROOT / 'specialists/algebra' / 'checkpoint.pt')
        print('  Checkpoint saved.')
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    for step in range(1, cfg.max_steps + 1):
        x, y = next(batch_iter)
        x, y = x.to(device), y.to(device)

        logits, _, _ = model(x)

        # Loss masking: only keep answer tokens (after '=')
        y_masked = y.clone()
        for b in range(y.size(0)):
            eq_positions = (x[b] == tok.stoi.get('=', 0))
            if eq_positions.any():
                first_eq = eq_positions.nonzero(as_tuple=True)[0][0].item()
                y_masked[b, :first_eq + 1] = -100

        loss = criterion(logits.view(-1, cfg.vocab_size), y_masked.view(-1))
        recent_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluation
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            results = evaluate_algebra(model, tok, train_ds, num_samples=50)
            overall = results['overall']
            best_acc = max(best_acc, overall)
            elapsed = time.time() - t0
            sps = step / max(1, elapsed)
            per_type = ' '.join(f'{k}:{v:.0f}%' for k, v in sorted(results['per_type'].items()))
            print(f'  Step {step:>6d} | loss={loss.item():.4f} | '
                  f'acc={overall:.1f}% (best={best_acc:.1f}%) | '
                  f'{sps:.1f} st/s | {per_type}')

        # Save
        if step % cfg.save_every == 0:
            Path('specialists/algebra').mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'global_step': step,
                'best_acc': best_acc,
            }, ROOT / 'specialists/algebra' / 'best.pt')

    # Final save
    Path('specialists/algebra').mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'global_step': cfg.max_steps,
        'best_acc': best_acc,
    }, ROOT / 'specialists/algebra' / 'best.pt')

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  DONE! {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'  Best accuracy: {best_acc:.1f}%')
    print('  Model saved: specialists/algebra/best.pt')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train algebra specialist')
    parser.add_argument('--quick', action='store_true', help='Smoke test')
    parser.add_argument('--eval-only', type=str, default=None)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=256)
    parser.add_argument('--type', type=str, default='linear',
                        choices=['linear', 'two-step', 'division', 'neg-coeff', 'mix'])
    parser.add_argument('--max-coeff', type=int, default=12)
    parser.add_argument('--max-const', type=int, default=20)
    parser.add_argument('--samples-per-step', type=int, default=2)
    args = parser.parse_args()

    if args.quick:
        args.steps = 100
        args.batch = 32
        args.max_coeff = 5
        args.max_const = 10

    train_algebra(args)
