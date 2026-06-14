"""MCTS Ablation Study: Compare simulation counts on addition accuracy and speed.

Methods:
  A: Greedy decoding (temperature=0) — baseline
  B: Micro-MCTS (16 simulations) — current project default
  C: MCTS-32 (32 simulations)
  D: MCTS-64 (64 simulations)

Usage:
    python3 experiments/run_mcts_ablation.py              # 100 problems
    python3 experiments/run_mcts_ablation.py --quick      # 10 problems (fast validation)
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path

# ─── sys.path hack: add ../src + ../
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / 'src'))
sys.path.insert(0, str(_SCRIPT_DIR.parent))

import torch
import torch.nn.functional as F

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.dataset import generate_problem
from egefalos.mcts import micro_mcts_search


# ═════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════

def train_addition_model(d_model=64, n_layers=2, n_heads=4, steps=3000,
                         save_path='checkpoints/mcts_ablation.pt',
                         max_digits=1):
    """Train a tiny addition model with value head for MCTS."""
    cfg = Config()
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.d_ff = d_model * 4
    cfg.max_seq_len = 32
    cfg.use_value_head = True       # MCTS needs value head for rollout scoring
    cfg.use_curriculum = False      # Simple fixed difficulty
    cfg.max_steps = steps
    cfg.batch_size = 32
    cfg.learning_rate = 0.001
    cfg.weight_decay = 0.01
    cfg.warmup_steps = 100
    cfg.log_every = 500
    cfg.eval_every = 500
    cfg.save_every = 100000        # Don't save intermediate checkpoints
    cfg.use_reversed = True
    cfg.force_carry_ratio = 0.0
    cfg.activation = 'relu'
    cfg.norm_type = 'rmsnorm'
    cfg.pos_encoding = 'rope'
    cfg.tie_embeddings = False
    cfg.label_smoothing = 0.0

    device = torch.device('cpu')
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg).to(device)
    n_params = count_parameters(model)
    print(f'    Model: d={d_model} L={n_layers} heads={n_heads} params={n_params:,}')
    print(f'    Vocab: {tok.vocab_size} tokens')
    print(f'    Value head: {model.value_head is not None}')

    # Generate a fixed training set: simple addition problems
    from torch.utils.data import Dataset, DataLoader

    class AdditionDataset(Dataset):
        def __init__(self, tok, num_samples, max_digits=2, seed=42):
            self.tok = tok
            self.samples = []
            rng = random.Random(seed)
            self.max_len = 0
            for _ in range(num_samples):
                # Only addition problems
                max_val = 10**max_digits - 1
                a = rng.randint(0, max_val)
                b = rng.randint(0, max_val)
                ans = a + b
                text = f'{a}+{b}={ans}'
                ids = tok.encode(text, add_special_tokens=True)
                # Input = all but last token, target = all but first token
                x = ids[:-1]
                y = ids[1:]
                self.samples.append((x, y))
                self.max_len = max(self.max_len, len(x))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            x, y = self.samples[idx]
            # Pad to max_len
            x_pad = x + [tok.pad_id] * (self.max_len - len(x))
            y_pad = y + [-100] * (self.max_len - len(y))
            return torch.tensor(x_pad), torch.tensor(y_pad)

    import random
    train_ds = AdditionDataset(tok, 5000, max_digits=2, seed=42)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def get_lr_scheduler(opt, warmup, max_steps):
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps)

    model.train()
    global_step = 0
    losses = []

    print(f'\n    Training for {steps} steps...')
    start = time.time()
    while global_step < cfg.max_steps:
        for x, y in train_loader:
            if global_step >= cfg.max_steps:
                break
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg_loss = sum(losses[-cfg.log_every:]) / len(losses[-cfg.log_every:])
                elapsed = time.time() - start
                print(f'      Step {global_step}/{cfg.max_steps} | loss={avg_loss:.4f} | {elapsed:.0f}s', flush=True)

    elapsed = time.time() - start
    print(f'    Training done in {elapsed:.0f}s ({elapsed/60:.1f} min)')

    # Quick evaluate
    model.eval()
    correct = 0
    for _ in range(50):
        a = random.randint(0, 10**max_digits - 1)
        b = random.randint(0, 10**max_digits - 1)
        ans = a + b
        prompt = f'{a}+{b}='
        # Use our custom greedy (no EOS at end of prompt)
        generated = generate_greedy(model, tok, prompt, max_new_tokens=10)
        if '=' in generated:
            pred = generated.split('=')[-1].strip()
            pred = ''.join(c for c in pred if c.isdigit() or c == '-')
            if pred == str(ans):
                correct += 1
    acc = correct / 50 * 100
    print(f'    Quick eval accuracy: {acc:.1f}%')

    # Save
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config_vars': {
            'd_model': cfg.d_model, 'n_layers': cfg.n_layers,
            'n_heads': cfg.n_heads, 'd_ff': cfg.d_ff,
            'vocab_size': cfg.vocab_size,
            'max_seq_len': cfg.max_seq_len,
            'use_value_head': cfg.use_value_head,
        }
    }, save_path)
    tok.save(str(save_dir / 'tokenizer.json'))

    return model, tok


# ═════════════════════════════════════════════════════════════════════
# MCTS GENERATION
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_greedy(model, tok, prompt, max_new_tokens=10):
    """Generate answer using greedy decoding (temperature=0).

    Encodes prompt WITHOUT adding EOS at the end (unlike model.generate).
    """
    device = next(model.parameters()).device
    model.eval()

    input_ids = tok.encode(prompt, add_special_tokens=True)  # BOS ... = EOS
    # Remove trailing EOS so the model generates the answer, not EOS
    if input_ids and input_ids[-1] == tok.eos_id:
        input_ids = input_ids[:-1]
    input_tensor = torch.tensor([input_ids], device=device)

    # Process full prompt to build KV cache
    prompt_logits, kv_caches = model.generate_step(input_tensor, past_kv_caches=None)
    next_logits = prompt_logits[0, -1, :]

    # Greedy: argmax
    next_id = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
    all_ids = torch.cat([input_tensor, next_id], dim=1)

    if next_id.item() == tok.eos_id:
        return tok.decode(all_ids[0].tolist())

    for _ in range(max_new_tokens - 1):
        logits, kv_caches = model.generate_step(next_id, past_kv_caches=kv_caches)
        next_logits = logits[0, -1, :]

        if all_ids.size(1) >= model.config.max_seq_len:
            break

        next_id = next_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        all_ids = torch.cat([all_ids, next_id], dim=1)

        if next_id.item() == tok.eos_id:
            break

    return tok.decode(all_ids[0].tolist())


@torch.no_grad()
def generate_with_mcts(model, tok, prompt, n_simulations=16, max_new_tokens=10,
                       top_k=5, max_rollout_tokens=8):
    """Generate answer token-by-token using MCTS for each step.

    The concept passed to MCTS is a string describing the addition problem.
    """
    device = next(model.parameters()).device
    input_ids = tok.encode(prompt, add_special_tokens=True)
    # Remove trailing EOS so the model generates the answer
    if input_ids and input_ids[-1] == tok.eos_id:
        input_ids = input_ids[:-1]
    input_tensor = torch.tensor([input_ids], device=device)
    full_ids = input_ids.copy()

    concept_str = prompt  # Use the prompt as the 'concept'

    for step in range(max_new_tokens):
        current_tensor = torch.tensor([full_ids], device=device)

        result = micro_mcts_search(
            model, tok, current_tensor, concept_str,
            temperature=0.8,
            top_k=top_k,
            simulations=n_simulations,
            max_rollout_tokens=max_rollout_tokens,
            grammar_rules=None,
            c_puct=1.0,
        )

        next_token = result['best_token']
        full_ids.append(next_token)

        if next_token == tok.eos_id:
            break

        if len(full_ids) > tok.max_seq_len:
            break

    decoded = tok.decode(full_ids, skip_special=True)
    return decoded


# ═════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════

def extract_answer(text):
    """Extract the numeric answer from a generated string."""
    if '=' in text:
        pred = text.split('=')[-1].strip()
        pred = ''.join(c for c in pred if c.isdigit() or c == '-')
        return pred
    return ''


@torch.no_grad()
def evaluate_method(model, tok, problems, method_fn, method_name, num_problems):
    """Run a method on all problems and measure accuracy + speed."""
    correct = 0
    total_time = 0.0
    total_tokens = 0

    for expr, ans in problems[:num_problems]:
        prompt = f'{expr}='

        t0 = time.time()
        generated = method_fn(model, tok, prompt)
        elapsed = time.time() - t0

        # Count tokens in the generated part (strip special tokens)
        gen_ids = tok.encode(generated, add_special_tokens=False)
        total_tokens += len(gen_ids)

        pred = extract_answer(generated)
        if pred == ans:
            correct += 1

        total_time += elapsed

    accuracy = correct / num_problems * 100
    avg_time = total_time / num_problems
    avg_tokens = total_tokens / num_problems

    return accuracy, avg_time, avg_tokens


def generate_problems_list(num_problems, max_digits=1):
    """Generate a fixed list of addition problems for fair comparison."""
    import random
    rng = random.Random(42)  # Fixed seed for reproducibility
    problems = []
    for _ in range(num_problems):
        a = rng.randint(0, 10**max_digits - 1)
        b = rng.randint(0, 10**max_digits - 1)
        ans = a + b
        problems.append((f'{a}+{b}', str(ans)))
    return problems


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='MCTS Ablation Study')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation mode: 10 problems only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional; trains fresh if not provided)')
    parser.add_argument('--train-steps', type=int, default=3000,
                        help='Training steps for fresh model (default: 3000)')
    args = parser.parse_args()

    num_problems = 10 if args.quick else 100
    print(f'{"="*60}')
    print(f'  MCTS Ablation Study')
    print(f'  Problems: {num_problems}  |  Quick mode: {args.quick}')
    print(f'{"="*60}')

    # ─── Load or train model ───
    checkpoint_path = args.checkpoint or 'checkpoints/mcts_ablation.pt'
    ckpt_file = Path(checkpoint_path)

    if ckpt_file.exists():
        print(f'\n[*] Loading existing model from {ckpt_file}')
        sd = torch.load(str(ckpt_file), map_location='cpu', weights_only=True)

        if 'config_vars' in sd:
            cv = sd['config_vars']
            cfg = Config()
            cfg.d_model = cv['d_model']
            cfg.n_layers = cv['n_layers']
            cfg.n_heads = cv['n_heads']
            cfg.d_ff = cv['d_ff']
            cfg.vocab_size = cv['vocab_size']
            cfg.max_seq_len = cv['max_seq_len']
            cfg.use_value_head = cv.get('use_value_head', False)
        else:
            # Infer from state dict
            d_model = sd['token_embedding']['weight'].shape[1] if isinstance(sd, dict) else 32
            # Fallback
            cfg = Config()
            cfg.d_model = 32
            cfg.n_layers = 1
            cfg.n_heads = 4
            cfg.d_ff = 128
            cfg.use_value_head = True

        tok_path = ckpt_file.parent / 'tokenizer.json'
        if tok_path.exists():
            tok = MathTokenizer.load(str(tok_path))
        else:
            tok = MathTokenizer()
        cfg.vocab_size = tok.vocab_size
        tok.max_seq_len = cfg.max_seq_len

        model = MathTransformer(cfg)
        model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd, strict=False)
        model.eval()
        print(f'    Loaded: d={cfg.d_model} L={cfg.n_layers} value_head={cfg.use_value_head}')

        # Quick sanity check
        test_gen = generate_greedy(model, tok, '1+1=', max_new_tokens=5)
        print(f'    Sanity: "1+1=" -> {repr(test_gen)}')
    else:
        print(f'\n[*] Training fresh model ({args.train_steps} steps)...')
        model, tok = train_addition_model(
            d_model=64, n_layers=2, n_heads=4,
            steps=args.train_steps,
            max_digits=1,
            save_path=str(ckpt_file),
        )

    # Move to CPU for consistent timing
    model = model.to('cpu')
    device = torch.device('cpu')

    # ─── Generate fixed problem set ───
    problems = generate_problems_list(num_problems, max_digits=1)
    print(f'\n[*] Generated {len(problems)} addition problems (1-digit)')

    # ─── Run all methods ───
    methods = [
        ('Greedy', lambda m, tok, prompt: generate_greedy(m, tok, prompt)),
        ('Micro-MCTS(16)', lambda m, tok, prompt: generate_with_mcts(m, tok, prompt, n_simulations=16)),
        ('MCTS(32)', lambda m, tok, prompt: generate_with_mcts(m, tok, prompt, n_simulations=32)),
        ('MCTS(64)', lambda m, tok, prompt: generate_with_mcts(m, tok, prompt, n_simulations=64)),
    ]

    results = {}
    print(f'\n{"="*60}')
    print('  Running evaluations...')
    print(f'{"="*60}\n')

    for method_name, method_fn in methods:
        print(f'  [{method_name}] Evaluating on {num_problems} problems...', end=' ', flush=True)
        t0 = time.time()
        acc, avg_time, avg_tokens = evaluate_method(
            model, tok, problems, method_fn, method_name, num_problems
        )
        wall = time.time() - t0
        results[method_name] = {
            'accuracy': round(acc, 1),
            'accuracy_pct': f'{acc:.1f}%',
            'avg_time_per_problem_s': round(avg_time, 4),
            'avg_tokens': round(avg_tokens, 1),
            'total_wall_time_s': round(wall, 2),
        }
        print(f'acc={acc:.1f}% | time={avg_time:.4f}s/token | {avg_tokens:.1f} tok | wall={wall:.1f}s')

    # ─── Print comparison table ───
    print(f'\n{"="*60}')
    print('  === MCTS Ablation Results ===')
    print(f'{"="*60}')
    print(f'  {"Method":<20} {"Accuracy":>10} {"Time/problem":>14} {"Tokens/problem":>16}')
    print(f'  {"-"*20} {"-"*10} {"-"*14} {"-"*16}')
    for method_name, _ in methods:
        r = results[method_name]
        print(f'  {method_name:<20} {r["accuracy_pct"]:>10} {r["avg_time_per_problem_s"]:>8.4f}s {r["avg_tokens"]:>12.1f}')

    # ─── Save results ───
    output_path = _SCRIPT_DIR / 'mcts_ablation_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'num_problems': num_problems,
                'quick_mode': args.quick,
                'train_steps': args.train_steps,
                'model': {
                    'd_model': model.config.d_model,
                    'n_layers': model.config.n_layers,
                    'use_value_head': getattr(model.config, 'use_value_head', False),
                }
            },
            'results': results,
        }, f, indent=2)
    print(f'\n[*] Results saved to {output_path}')

    print(f'\n{"="*60}')
    print('  Done!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
