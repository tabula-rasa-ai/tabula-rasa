"""Evaluate trained model on arithmetic problems."""

import torch
from pathlib import Path

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.dataset import generate_problem

# Operations and their expected formats
MULTI_STEP_PROBLEMS = [
    # Chained arithmetic
    ("2+3+4=", "9"),
    ("10-3-2=", "5"),
    ("5+3*2=", "11"),  # order of ops matters - model must learn this
    # Parentheses
    ("(2+3)*4=", "20"),
    ("(10-3)*2=", "14"),
    # Two-digit
    ("12+34=", "46"),
    ("56-18=", "38"),
    ("23*4=", "92"),
    # Three-digit
    ("123+456=", "579"),
    ("500-123=", "377"),
    # Larger
    ("99+1=", "100"),
    ("1000-1=", "999"),
]


def load_model(checkpoint_path):
    cfg = Config()
    tok = MathTokenizer.load(str(Path(checkpoint_path).parent / 'tokenizer.json'))
    cfg.vocab_size = tok.vocab_size  # type: ignore[misc]
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tok


@torch.no_grad()
def evaluate_accuracy(model, tok, num_problems=200, max_digits=4, verbose=True):
    """Evaluate raw accuracy on random problems."""
    correct = 0
    total = num_problems

    for i in range(num_problems):
        expr, ans = generate_problem(1, max_digits)
        prompt = f'{expr}='
        generated = model.generate(tok, prompt, max_new_tokens=10, temperature=0.3, top_k=3)

        if '=' in generated:
            pred = generated.split('=')[-1].strip()
            pred = ''.join(c for c in pred if c.isdigit() or c == '-')
            if pred == ans:
                correct += 1
            elif verbose and i < 10:
                print(f'  FAIL: {expr}={ans} | pred: {pred} | raw: {repr(generated)}')

    acc = correct / total * 100
    print(f'\nAccuracy on {total} problems: {acc:.1f}%')
    return acc


@torch.no_grad()
def solve_problems(model, tok, problems=None):
    """Solve specific problems and show results."""
    if problems is None:
        problems = MULTI_STEP_PROBLEMS

    print(f'\n{"Problem":<20} {"Expected":<12} {"Predicted":<12} {"Result":<8}')
    print('-' * 52)

    for expr, expected in problems:
        generated = model.generate(tok, expr, max_new_tokens=10, temperature=0.3, top_k=3)
        if '=' in generated:
            pred = generated.split('=')[-1].strip()
            pred = ''.join(c for c in pred if c.isdigit() or c == '-')
        else:
            pred = ''

        result = 'PASS' if pred == expected else 'FAIL'
        print(f'{expr:<20} {expected:<12} {pred:<12} {result:<8}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        # Try best or final checkpoint
        ckpt = Path('checkpoints/best.pt')
        if not ckpt.exists():
            ckpt = Path('checkpoints/final.pt')
        if not ckpt.exists():
            print('No checkpoint found. Train first: python3 train.py')
            sys.exit(1)
    else:
        ckpt = Path(sys.argv[1])

    print(f'Loading model from {ckpt}')
    model, tok = load_model(ckpt)
    print(f'Model params: {count_parameters(model):,}')

    # Test on random problems
    evaluate_accuracy(model, tok, num_problems=200, max_digits=4)

    # Test on specific problems
    solve_problems(model, tok)
