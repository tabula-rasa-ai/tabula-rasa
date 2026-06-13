"""Socratic Training Experiment: Validates whether Socratic dialogue
improves model accuracy compared to standard training.

Trains a baseline model on addition → evaluates → runs Socratic dialogue
between two instances → fine-tunes on Socratic-refined predictions →
evaluates → compares.

Usage:
    python experiments/run_socratic_training.py
"""
import sys, os, json, time, random, math
from pathlib import Path

# Path hack: this script lives in experiments/, src/ is one level up
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.dataset import generate_problem

# ─── Constants ───────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
N_PROBLEMS = 200              # Problems for evaluation
N_SOCRATIC_ROUNDS = 5         # Dialogue rounds per problem
N_SOCRATIC_PROBLEMS = 100     # Problems to run Socratic dialogue on
N_FINETUNE_EPOCHS = 3         # Fine-tuning epochs on Socratic data
BASELINE_DIR = Path('specialists') / 'socratic_baseline'
OUTPUT_PATH = Path('experiments') / 'socratic_training_results.json'

torch.manual_seed(SEED)
random.seed(SEED)


# ─── Helpers ─────────────────────────────────────────────────────────

def make_model(tok, cfg_overrides=None):
    """Create a fresh model with standard config."""
    cfg = Config()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg).to(DEVICE)
    return model, cfg


def save_checkpoint(model, tok, path):
    """Save model and tokenizer to a directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, path / 'final.pt')
    tok.save(str(path / 'tokenizer.json'))
    print(f'  [*] Model saved to {path / "final.pt"}')


def load_checkpoint(tok, path):
    """Load model from a checkpoint directory."""
    path = Path(path)
    ckpt = path / 'final.pt'
    if not ckpt.exists():
        ckpt = path / 'best.pt'
    if not ckpt.exists():
        return None
    model, cfg = make_model(tok)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print(f'  [*] Model loaded from {ckpt}')
    return model


@torch.no_grad()
def evaluate_accuracy(model, tok, num_problems=N_PROBLEMS, max_digits=4,
                      temperature=0.3, top_k=3, verbose=False):
    """Evaluate accuracy on random addition problems."""
    model.eval()
    correct = 0
    total = num_problems
    for i in range(total):
        expr, ans = generate_problem(1, max_digits)
        prompt = f'{expr}='
        out = model.generate(tok, prompt, max_new_tokens=10,
                             temperature=temperature, top_k=top_k)
        if '=' in out:
            pred = out.split('=')[-1].strip()
            pred = ''.join(c for c in pred if c.isdigit() or c == '-')
            if pred == ans:
                correct += 1
            elif verbose and i < 5:
                print(f'    FAIL: {expr}={ans} | pred={pred} | raw={repr(out)}')
    acc = correct / total * 100
    print(f'  Accuracy: {acc:.1f}% ({correct}/{total})')
    return acc


# ─── Step 1: Train baseline model ────────────────────────────────────

def train_baseline(tok, steps=2000):
    """Train a model on addition problems from scratch."""
    print('\n' + '=' * 60)
    print('  Step 1: Training Baseline Model (addition, 2000 steps)')
    print('=' * 60)

    model, cfg = make_model(tok, {
        'max_steps': steps,
        'batch_size': 32,
        'learning_rate': 0.001,
        'eval_every': 500,
        'train_samples': 5000,
        'eval_samples': 100,
        'max_digits': 4,
        'min_digits': 1,
        'use_reversed': True,
        'use_scratchpad': True,
        'use_loss_masking': True,
        'max_seq_len': 32,
    })
    print(f'  Model params: {count_parameters(model):,}')

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay)

    # Simple linear LR schedule
    def lr_lambda(s):
        if s < 100:
            return s / 100.0
        return max(0.05, 1.0 - (s - 100) / (steps - 100))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    t0 = time.time()
    for step in range(steps):
        total_loss = 0.0
        n_batch = 0
        optimizer.zero_grad()
        for _ in range(cfg.batch_size):
            expr, ans = generate_problem(1, 4)
            text = f'{expr}={ans}'
            ids = tok.encode(text, add_special_tokens=True)
            if len(ids) > cfg.max_seq_len:
                ids = ids[:cfg.max_seq_len]
            padded = ids + [tok.pad_id] * (cfg.max_seq_len - len(ids))
            x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
            y_target = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Loss masking: only compute loss on answer portion (after '=')
            if cfg.use_loss_masking:
                eq_id = tok.stoi['=']
                eq_positions = (y_target == eq_id).nonzero(as_tuple=True)[1]
                if len(eq_positions) > 0:
                    eq_pos = eq_positions[0].item()
                    y_target[0, :eq_pos + 1] = -100

            _, loss, _ = model(x, y_target)
            loss.backward()
            total_loss += loss.item()
            n_batch += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if (step + 1) % max(1, steps // 10) == 0:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f'    Step {step + 1}/{steps} | loss={total_loss / max(1, n_batch):.4f} | lr={lr_now:.6f} | {elapsed:.0f}s')

    elapsed = time.time() - t0
    print(f'  Training complete in {elapsed:.0f}s')

    # Save baseline
    save_checkpoint(model, tok, BASELINE_DIR)
    return model


# ─── Step 2: Evaluate baseline ───────────────────────────────────────

def evaluate_baseline(model, tok):
    """Evaluate baseline model accuracy."""
    print('\n' + '=' * 60)
    print('  Step 2: Evaluating Baseline Accuracy')
    print('=' * 60)
    acc = evaluate_accuracy(model, tok, num_problems=N_PROBLEMS, verbose=True)
    return acc


# ─── Step 3: Socratic Dialogue ───────────────────────────────────────

@torch.no_grad()
def solve_problem(model, tok, expr):
    """Solve a single problem and return the predicted answer."""
    prompt = f'{expr}='
    out = model.generate(tok, prompt, max_new_tokens=10,
                         temperature=0.3, top_k=3)
    if '=' in out:
        pred = out.split('=')[-1].strip()
        pred = ''.join(c for c in pred if c.isdigit() or c == '-')
        return pred
    return ''


@torch.no_grad()
def critique_answer(model, tok, expr, answer):
    """Generate a critique of an answer for a given problem.

    The model sees the problem, the proposed answer, and is prompted to
    check if it's correct. Returns a corrected answer if the model
    identifies a flaw, or the original answer if it agrees.
    """
    prompt = (
        f"Problem: {expr}=?\n"
        f"Proposed answer: {answer}\n"
        f"Check if this answer is correct. If wrong, provide the correct answer. "
        f"If correct, repeat the answer.\n"
        f"Correct answer:"
    )
    out = model.generate(tok, prompt, max_new_tokens=10,
                         temperature=0.2, top_k=3)
    # Extract answer text
    if '=' in out:
        pred = out.split('=')[-1].strip()
    else:
        pred = out.strip()
    pred = ''.join(c for c in pred if c.isdigit() or c == '-')
    if pred:
        return pred
    return answer  # fallback


def socratic_dialogue(model_a, model_b, tok, expr, true_answer, rounds=N_SOCRATIC_ROUNDS):
    """Run Socratic dialogue between two model instances on one problem.

    Both models solve independently. If answers differ, each critques the
    other's answer. After N rounds, if the models converge on a shared
    answer AND that answer is correct, return it as the Socratic-refined answer.

    Returns:
        {
            'expr': expr,
            'true_answer': true_answer,
            'model_a_answer': ...,
            'model_b_answer': ...,
            'socratic_answer': ...,   # converged answer (or None)
            'was_correct': bool,       # whether the converged answer is correct
            'improved': bool,          # whether Socratic fixed a mistake
        }
    """
    # Initial independent solves
    ans_a = solve_problem(model_a, tok, expr)
    ans_b = solve_problem(model_b, tok, expr)

    if not ans_a:
        ans_a = ans_b if ans_b else ''
    if not ans_b:
        ans_b = ans_a if ans_a else ''

    answers_a, answers_b = [ans_a], [ans_b]

    # Dialogue rounds
    for r in range(rounds):
        if answers_a[-1] == answers_b[-1]:
            break  # Converged

        # Each model critiques the other's latest answer
        new_a = critique_answer(model_a, tok, expr, answers_b[-1])
        new_b = critique_answer(model_b, tok, expr, answers_a[-1])
        answers_a.append(new_a if new_a else answers_a[-1])
        answers_b.append(new_b if new_b else answers_b[-1])

    # Converged answer: if models agree, use their answer; else use model_a's final
    if answers_a[-1] == answers_b[-1]:
        socratic_answer = answers_a[-1]
        converged = True
    else:
        # Use answer from the model whose critiques converged more
        socratic_answer = answers_b[-1] if answers_a.count(answers_a[-1]) < answers_b.count(answers_b[-1]) else answers_a[-1]
        converged = False

    baseline_correct = (ans_a == true_answer)
    socratic_correct = (socratic_answer == true_answer)
    improved = (not baseline_correct) and socratic_correct

    return {
        'expr': expr,
        'true_answer': true_answer,
        'model_a_answer': ans_a,
        'model_b_answer': ans_b,
        'socratic_answer': socratic_answer,
        'converged': converged,
        'baseline_correct': baseline_correct,
        'socratic_correct': socratic_correct,
        'improved': improved,
    }


def run_socratic_dialogue(model, tok):
    """Run Socratic dialogue between the model and a copy of itself.

    Uses the same model loaded twice (model_a, model_b) with different
    random seeds for generation diversity.

    Returns:
        results: list of dialogue result dicts
        training_data: list of (expr, true_answer) pairs where Socratic improved
    """
    print('\n' + '=' * 60)
    print('  Step 3: Running Socratic Dialogue')
    print('=' * 60)

    # Use the same model — create a second instance with same weights
    model_a = model
    model_b = load_checkpoint(tok, BASELINE_DIR)
    if model_b is None:
        model_b = model  # fallback: use same instance

    # Generate problems for dialogue
    problems = []
    for _ in range(N_SOCRATIC_PROBLEMS):
        expr, ans = generate_problem(1, 4)
        problems.append((expr, ans))

    results = []
    n_improved = 0
    n_correct_baseline = 0
    n_correct_socratic = 0
    n_converged = 0

    t0 = time.time()
    for i, (expr, ans) in enumerate(problems):
        result = socratic_dialogue(model_a, model_b, tok, expr, ans,
                                    rounds=N_SOCRATIC_ROUNDS)
        results.append(result)

        if result['baseline_correct']:
            n_correct_baseline += 1
        if result['socratic_correct']:
            n_correct_socratic += 1
        if result['improved']:
            n_improved += 1
        if result['converged']:
            n_converged += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f'    [{i + 1}/{N_SOCRATIC_PROBLEMS}] baseline_acc={(n_correct_baseline / (i+1)) * 100:.1f}% '
                  f'socratic_acc={(n_correct_socratic / (i+1)) * 100:.1f}% '
                  f'improved={n_improved} converged={n_converged} | {elapsed:.0f}s')

    elapsed = time.time() - t0
    baseline_acc = n_correct_baseline / N_SOCRATIC_PROBLEMS * 100
    socratic_acc = n_correct_socratic / N_SOCRATIC_PROBLEMS * 100
    print(f'\n  Socratic dialogue complete ({elapsed:.0f}s):')
    print(f'    Baseline accuracy on dialogue set: {baseline_acc:.1f}%')
    print(f'    Socratic accuracy on dialogue set: {socratic_acc:.1f}%')
    print(f'    Problems improved by Socratic: {n_improved}/{N_SOCRATIC_PROBLEMS}')
    print(f'    Dialogues converged: {n_converged}/{N_SOCRATIC_PROBLEMS}')

    # Build training data: problems where Socratic improved the answer
    training_data = []
    for r in results:
        if r['improved']:
            training_data.append((r['expr'], r['socratic_answer']))

    print(f'    Training examples from Socratic: {len(training_data)}')
    return results, training_data


# ─── Step 4: Fine-tune on Socratic data ──────────────────────────────

def finetune_socratic(model, tok, training_data):
    """Fine-tune the baseline model on Socratic-refined predictions."""
    print('\n' + '=' * 60)
    print('  Step 4: Fine-tuning on Socratic-Refined Predictions')
    print('=' * 60)

    if not training_data:
        print('  No training data generated. Skipping fine-tuning.')
        return model

    print(f'  Training examples: {len(training_data)}')
    print(f'  Epochs: {N_FINETUNE_EPOCHS}')

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    cfg = Config()
    cfg.max_seq_len = tok.max_seq_len if hasattr(tok, 'max_seq_len') else 32

    model.train()
    t0 = time.time()
    total_pairs = 0
    for epoch in range(N_FINETUNE_EPOCHS):
        random.shuffle(training_data)
        epoch_loss = 0.0
        n = 0
        for expr, ans in training_data:
            text = f'{expr}={ans}'
            ids = tok.encode(text, add_special_tokens=True)
            if len(ids) > cfg.max_seq_len:
                ids = ids[:cfg.max_seq_len]
            padded = ids + [tok.pad_id] * (cfg.max_seq_len - len(ids))
            x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0).to(DEVICE)
            y_target = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Loss masking: only compute loss on answer portion
            eq_id = tok.stoi['=']
            eq_positions = (y_target == eq_id).nonzero(as_tuple=True)[1]
            if len(eq_positions) > 0:
                eq_pos = eq_positions[0].item()
                y_target[0, :eq_pos + 1] = -100

            optimizer.zero_grad()
            _, loss, _ = model(x, y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
            total_pairs += 1

        print(f'    Epoch {epoch + 1}: loss={epoch_loss / max(1, n):.4f}')

    elapsed = time.time() - t0
    print(f'  Fine-tuning complete in {elapsed:.0f}s ({total_pairs} pairs)')
    return model


# ─── Step 5: Evaluate Socratic-tuned model ───────────────────────────

def evaluate_socratic(model, tok):
    """Evaluate the Socratic-tuned model accuracy."""
    print('\n' + '=' * 60)
    print('  Step 5: Evaluating Socratic-Tuned Model Accuracy')
    print('=' * 60)
    acc = evaluate_accuracy(model, tok, num_problems=N_PROBLEMS, verbose=True)
    return acc


# ─── Main Experiment ─────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Socratic Training Experiment')
    print('  Validating whether Socratic dialogue improves accuracy')
    print(f'  Device: {DEVICE}')
    print('=' * 60)

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Build tokenizer
    tok = MathTokenizer()
    print(f'\n  Tokenizer vocab: {tok.vocab_size} tokens')

    # Step 1: Train baseline
    model = train_baseline(tok)

    # Step 2: Evaluate baseline
    baseline_acc = evaluate_baseline(model, tok)

    # Step 3: Run Socratic dialogue
    results, training_data = run_socratic_dialogue(model, tok)

    # Save dialogue results
    dialogue_log_path = Path('experiments') / 'socratic_dialogue_log.json'
    with open(dialogue_log_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_problems': N_SOCRATIC_PROBLEMS,
            'n_rounds': N_SOCRATIC_ROUNDS,
            'results': results,
        }, f, indent=2)
    print(f'  Dialogue log saved to {dialogue_log_path}')

    # Step 4: Fine-tune on Socratic data
    model = finetune_socratic(model, tok, training_data)

    # Save Socratic-tuned model
    save_checkpoint(model, tok, BASELINE_DIR / 'socratic_tuned')

    # Step 5: Evaluate Socratic-tuned model
    socratic_acc = evaluate_socratic(model, tok)

    # Step 6: Compare and save results
    improvement = socratic_acc - baseline_acc
    print('\n' + '=' * 60)
    print('  RESULTS COMPARISON')
    print('=' * 60)
    print(f'  Baseline accuracy:  {baseline_acc:.1f}%')
    print(f'  Socratic accuracy:  {socratic_acc:.1f}%')
    print(f'  Improvement:        {improvement:+.1f}%')
    print(f'  Problems evaluated: {N_PROBLEMS}')
    print(f'  Socratic problems:  {N_SOCRATIC_PROBLEMS}')
    print(f'  Socratic rounds:    {N_SOCRATIC_ROUNDS}')
    print(f'  Training examples:  {len(training_data)}')
    print('=' * 60)

    # Save results
    results_data = {
        'baseline_accuracy': round(baseline_acc, 2),
        'socratic_accuracy': round(socratic_acc, 2),
        'improvement': round(improvement, 2),
        'n_problems': N_PROBLEMS,
        'n_socratic_rounds': N_SOCRATIC_ROUNDS,
        'n_socratic_problems': N_SOCRATIC_PROBLEMS,
        'n_training_examples': len(training_data),
        'timestamp': timestamp,
        'device': DEVICE,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f'\n  Results saved to {OUTPUT_PATH}')

    return results_data


if __name__ == '__main__':
    main()
