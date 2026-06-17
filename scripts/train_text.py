"""Train a specialist on string reversal — validates the BPETokenizer for text.

Demonstrates that the Tabula Rasa transformer can learn a simple text task
(string reversal) using the BPE tokenizer, proving the framework extends
beyond math arithmetic.

Usage:
    python3 scripts/train_text.py                          # Train on reversal
    python3 scripts/train_text.py --quick                  # Smoke test (100 steps)
    python3 scripts/train_text.py --eval-only              # Evaluate existing model
    python3 scripts/train_text.py --steps 5000 --bpe 30    # Custom run with BPE merges
"""

import argparse
import signal
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters

# ─── Text Reversal Dataset ─────────────────────────────────────

class TextReversalDataset(Dataset):
    """Generate random string reversal problems."""

    CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789'

    def __init__(self, num_samples=5000, min_len=2, max_len=8):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        import random
        length = random.randint(self.min_len, self.max_len)
        s = ''.join(random.choice(self.CHARS) for _ in range(length))
        prompt = s + '='
        answer = s[::-1]
        return prompt, answer


# ─── Training Loop ─────────────────────────────────────────────

def setup_tokenizer(cfg, texts=None):
    """Create and optionally train a BPETokenizer."""
    tok = BPETokenizer()
    tok.max_seq_len = cfg.max_seq_len

    if texts and cfg.text_num_merges > 0:
        print(f'  Learning {cfg.text_num_merges} BPE merges from {len(texts)} samples...')
        tok.learn_bpe_from_texts(texts, num_merges=cfg.text_num_merges,
                                 min_frequency=2, verbose=True)

    cfg.vocab_size = tok.vocab_size
    print(f'  Tokenizer: {tok.vocab_size} tokens ({tok.base_vocab_size} base + '
          f'{tok.bpe_vocab_size} BPE)')
    return tok


def train_reversal(opts):
    """Train a transformer on string reversal."""
    cfg = Config()
    cfg.max_seq_len = 64
    cfg.d_model = opts.d_model
    cfg.n_layers = opts.n_layers
    cfg.n_heads = opts.n_heads
    cfg.d_ff = opts.d_ff
    cfg.max_steps = opts.steps
    cfg.batch_size = opts.batch
    cfg.learning_rate = opts.lr
    cfg.text_num_merges = opts.bpe
    cfg.use_reversed = False
    cfg.use_loss_masking = True
    cfg.eval_every = max(100, opts.steps // 20)
    cfg.save_every = max(500, opts.steps // 10)
    cfg.eval_samples = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print('  TEXT SPECIALIST — String Reversal')
    print(f'  Device: {device} | Params: {count_parameters.__name__}')
    print(f'  Steps: {cfg.max_steps} | Batch: {cfg.batch_size} | LR: {cfg.learning_rate}')
    print(f'  Arch: d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads}, ff={cfg.d_ff}')
    print(f'  BPE merges: {cfg.text_num_merges}')
    print(f'  Eval every: {cfg.eval_every} | Save every: {cfg.save_every}')
    print(f'{"="*60}')

    # Dataset
    train_ds = TextReversalDataset(num_samples=cfg.max_steps * 2,
                                   max_len=opts.max_len)
    eval_ds = TextReversalDataset(num_samples=cfg.eval_samples,
                                  max_len=opts.max_len)

    # Build vocabulary from training samples
    sample_texts = [s[0] + s[1] for s in [train_ds[i] for i in range(min(200, len(train_ds)))]]
    tok = setup_tokenizer(cfg, sample_texts)

    # Model
    model = MathTransformer(cfg)
    model.to(device)
    print(f'  Model: {count_parameters(model):,} params ({cfg.d_model}d, {cfg.n_layers}L, {cfg.n_heads}H)')

    if opts.eval_only:
        ckpt = torch.load(opts.eval_only, map_location=device, weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
        print(f'  Loaded checkpoint: {opts.eval_only}')
        eval_acc = evaluate_reversal(model, tok, eval_ds, device)
        print(f'  Eval accuracy: {eval_acc["overall"]:.1f}%')
        for l, acc in sorted(eval_acc['per_length'].items()):
            print(f'    {l}-char: {acc:.1f}%')
        return

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=0.01)

    # DataLoader
    def collate_fn(batch):
        xs, ys = zip(*batch)
        # Pad to max length in batch
        max_x = max(len(x) for x in xs)
        max_y = max(len(y) for y in ys)
        max_len = max(max_x, max_y)
        padded_x = torch.zeros(len(batch), max_len, dtype=torch.long) + tok.pad_id
        padded_y = torch.zeros(len(batch), max_len, dtype=torch.long) + tok.pad_id
        for i, (x, y) in enumerate(zip(xs, ys)):
            padded_x[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            padded_y[i, :len(y)] = torch.tensor(y, dtype=torch.long)
        return padded_x, padded_y

    # Infinite batch generator (avoids DataLoader exhaustion)
    def _infinite_batches(dataset, tok, batch_size):
        while True:
            items = [(tok.encode(p, add_special_tokens=True),
                      tok.encode(a, add_special_tokens=True))
                     for p, a in [dataset[i] for i in range(min(2000, len(dataset)))]]
            loader = DataLoader(items, batch_size=batch_size,
                                shuffle=True, drop_last=True, collate_fn=collate_fn)
            yield from loader

    batch_iter = _infinite_batches(train_ds, tok, cfg.batch_size)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training
    best_acc = 0.0
    t0 = time.time()

    def sigint_handler(sig, frame):
        print(f'\n  [Interrupt] Saving checkpoint at step {step}...')
        Path('specialists/text').mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'global_step': step,
            'best_acc': best_acc,
            'config': cfg.__dict__,
        }, ROOT / 'specialists/text' / 'checkpoint.pt')
        print('  Checkpoint saved.')
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    for step in range(1, cfg.max_steps + 1):
        x, y = next(batch_iter)
        x, y = x.to(device), y.to(device)

        # Forward with loss masking (mask prompt tokens)
        logits, _, _ = model(x)
        # Mask everything before and including '=' (prompt), keep answer tokens
        y_masked = y.clone()
        for b in range(y.size(0)):
            eq_positions = (x[b] == tok.stoi.get('=', -1))
            if eq_positions.any():
                first_eq = eq_positions.nonzero(as_tuple=True)[0][0].item()
                y_masked[b, :first_eq + 1] = -100

        loss = criterion(logits.view(-1, cfg.vocab_size), y_masked.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluation
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            acc = evaluate_reversal(model, tok, eval_ds, device)
            overall = acc['overall']
            best_acc = max(best_acc, overall)
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            per_len = ' '.join(f'{l}:{acc["per_length"].get(l, 0):.0f}%'
                               for l in sorted(acc['per_length'].keys()))
            print(f'  Step {step:>6d} | loss={loss.item():.4f} | '
                  f'acc={overall:.1f}% (best={best_acc:.1f}%) | '
                  f'{sps:.1f} st/s | {per_len}')

        # Save
        if step % cfg.save_every == 0:
            Path('specialists/text').mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'global_step': step,
                'best_acc': best_acc,
                'config': cfg.__dict__,
            }, ROOT / 'specialists/text' / f'step_{step}.pt')

    # Final save
    Path('specialists/text').mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'global_step': cfg.max_steps,
        'best_acc': best_acc,
        'config': cfg.__dict__,
    }, ROOT / 'specialists/text' / 'best.pt')

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  DONE! {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'  Best accuracy: {best_acc:.1f}%')
    print('  Model saved: specialists/text/best.pt')
    print(f'{"="*60}')


def evaluate_reversal(model, tok, dataset, device='cpu'):
    """Evaluate string reversal accuracy per length."""
    model.eval()
    per_length = {}

    with torch.no_grad():
        for i in range(min(len(dataset), 100)):
            prompt, expected = dataset[i]
            output = model.generate(tok, prompt, max_new_tokens=len(expected) + 5,
                                    temperature=0.0)

            # Extract answer after '='
            if '=' in output:
                raw = output.split('=', 1)[-1].replace('<EOS>', '').replace('<PAD>', '').strip()
                # If BPE tokens present, decode properly from scratchpad
            else:
                raw = output.strip()

            # Remove special tokens from raw
            for t in ['<EOS>', '<PAD>', '<BOS>', '<UNK>']:
                raw = raw.replace(t, '')

            length = len(expected)
            if length not in per_length:
                per_length[length] = {'correct': 0, 'total': 0}
            per_length[length]['total'] += 1
            if raw == expected:
                per_length[length]['correct'] += 1

    model.train()
    overall = sum(d['correct'] for d in per_length.values()) / max(1, sum(d['total'] for d in per_length.values())) * 100
    return {
        'overall': overall,
        'per_length': {str(l): d['correct'] / max(1, d['total']) * 100
                       for l, d in per_length.items()}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train text specialist (string reversal)')
    parser.add_argument('--quick', action='store_true', help='Quick test: 100 steps')
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Path to checkpoint for evaluation only')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=256)
    parser.add_argument('--max-len', type=int, default=6, help='Max string length')
    parser.add_argument('--bpe', type=int, default=20, help='Number of BPE merges to learn')
    args = parser.parse_args()

    if args.quick:
        args.steps = 100
        args.batch = 32
        args.bpe = 5
        args.max_len = 4

    opts = parser.parse_args([])  # get defaults for display
    opts.__dict__.update(vars(args))
    train_reversal(opts)
