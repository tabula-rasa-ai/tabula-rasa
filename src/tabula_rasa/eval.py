"""Evaluate trained model on arithmetic problems."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Any

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.dataset import generate_problem

# Operations and their expected formats
MULTI_STEP_PROBLEMS: list[tuple[str, str]] = [
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


def load_model(checkpoint_path: str | Path) -> tuple[MathTransformer, MathTokenizer]:
    """Load a trained model and matching tokenizer from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file. The
            corresponding tokenizer JSON is expected at
            ``{checkpoint_dir}/tokenizer.json``.

    Returns:
        Tuple ``(model, tokenizer)`` with the model in evaluation mode.
    """
    cfg = Config()
    tok = MathTokenizer.load(str(Path(checkpoint_path).parent / 'tokenizer.json'))
    cfg.vocab_size = tok.vocab_size  # type: ignore[misc]
    tok.max_seq_len = cfg.max_seq_len  # type: ignore[attr-defined]

    model = MathTransformer(cfg)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tok


@torch.no_grad()
def evaluate_accuracy(model: MathTransformer, tok: MathTokenizer,
                      num_problems: int = 200, max_digits: int = 4,
                      verbose: bool = True) -> float:
    """Evaluate raw accuracy on random arithmetic problems.

    Args:
        model: The trained transformer model.
        tok: The tokenizer.
        num_problems: Number of random problems to evaluate (default: 200).
        max_digits: Maximum operand digits for generated problems
            (default: 4).
        verbose: If ``True``, print up to 10 failure examples.

    Returns:
        Accuracy as a percentage (``0.0`` – ``100.0``).
    """
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
def solve_problems(model: MathTransformer, tok: MathTokenizer,
                   problems: list[tuple[str, str]] | None = None) -> None:
    """Solve specific problems and display results in a table.

    Args:
        model: The trained transformer model.
        tok: The tokenizer.
        problems: List of ``(expression, expected_answer)`` tuples. If
            ``None``, uses the built-in ``MULTI_STEP_PROBLEMS`` list.
    """
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
