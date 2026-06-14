"""Generate synthetic math problems for training."""
from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from tabula_rasa.tokenizer import MathTokenizer


OPERATIONS: list[str] = ['+', '-', '*', '/']


def generate_problem(min_digits: int = 1, max_digits: int = 4) -> tuple[str, str]:
    """Generate an arithmetic problem and its answer.

    Creates a random arithmetic expression using one of ``+``, ``-``, ``*``,
    or ``/`` with operands having between *min_digits* and *max_digits*
    digits. Subtraction results are always positive; division always yields
    integer results.

    Args:
        min_digits: Minimum number of digits for each operand (default: 1).
        max_digits: Maximum number of digits for each operand (default: 4).

    Returns:
        A tuple ``(expression_str, answer_str)``, e.g. ``('12+34', '46')``.
    """
    op = random.choice(OPERATIONS)

    a_digits = random.randint(min_digits, max_digits)
    b_digits = random.randint(min_digits, max_digits)

    a = random.randint(10 ** (a_digits - 1), 10**a_digits - 1)
    b = random.randint(10 ** (b_digits - 1), 10**b_digits - 1)

    if op == '+':
        ans = a + b
    elif op == '-':
        # Ensure positive result for simplicity
        if a < b:
            a, b = b, a
        ans = a - b
    elif op == '*':
        ans = a * b
    elif op == '/':
        # Division must yield integer
        ans = random.randint(1, max(1, a // 2))
        b = random.randint(1, max(1, a // 2))
        # Make it exact: a = b * ans -> a = b * ans
        a = b * ans
        if a == 0:
            a = b * max(1, ans)
            ans = a // b if b != 0 else 1
        # Ensure b isn't too crazy
        if b > 10000:
            b = random.randint(1, 100)

    expr = f'{a}{op}{b}'
    answer = str(ans)
    return expr, answer


def format_math_sample(expr: str, answer: str) -> str:
    """Format problem and answer as a single string.

    Args:
        expr: The expression string (e.g. ``'12+34'``).
        answer: The answer string (e.g. ``'46'``).

    Returns:
        Formatted string ``'{expr}={answer}'`` (e.g. ``'12+34=46'``).
    """
    return f'{expr}={answer}'


class MathDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of synthetic math problems for autoregressive training.

    Each sample is a tokenised string of the form ``'{expr}={answer}'``,
    wrapped in special tokens. Sequences are padded to *max_seq_len* and
    returned as ``(input_ids, target_ids)`` where targets are shifted right
    by one position.
    """

    def __init__(self, tokenizer: MathTokenizer, num_samples: int,
                 min_digits: int = 1, max_digits: int = 4,
                 seed: int = 42) -> None:
        """Initialise the dataset by generating *num_samples* problems.

        Args:
            tokenizer: Tokenizer used to encode problem strings.
            num_samples: Number of problems to generate.
            min_digits: Minimum operand digits per problem.
            max_digits: Maximum operand digits per problem.
            seed: Random seed for reproducible problem generation.
        """
        self.tokenizer = tokenizer
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.max_seq_len: int = tokenizer.max_seq_len if hasattr(tokenizer, 'max_seq_len') else 64

        rng = random.Random(seed)
        self.samples: list[list[int]] = []
        for _ in range(num_samples):
            expr, ans = generate_problem(min_digits, max_digits)
            text = format_math_sample(expr, ans)
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= self.max_seq_len:
                self.samples.append(ids)

        print(f'Created {len(self.samples)} valid samples (filtered from {num_samples})')

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(input, target)`` pair for sample at index *idx*.

        The input is the token sequence without the last token; the target
        is the sequence without the first token (shifted right). Both are
        padded to *max_seq_len*.

        Args:
            idx: Sample index.

        Returns:
            A tuple ``(input_ids, target_ids)``, each a ``torch.long`` tensor
            of shape ``(max_seq_len,)``.
        """
        ids = self.samples[idx]
        # Pad to max_seq_len
        padded = ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(ids))
        # Input: all tokens except last
        x = padded[:-1]
        # Target: all tokens except first (shifted)
        y = padded[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def generate_test_set(num_samples: int = 100, seed: int = 123) -> list[tuple[str, str]]:
    """Generate a clean test set of problems for evaluation.

    Problems are generated with operand digits ranging from 1 to 5 to
    test generalisation beyond training difficulty.

    Args:
        num_samples: Number of test problems to generate (default: 100).
        seed: Random seed for reproducibility (default: 123).

    Returns:
        List of ``(expression, answer)`` tuples.
    """
    rng = random.Random(seed)
    problems: list[tuple[str, str]] = []
    for _ in range(num_samples):
        expr, ans = generate_problem(1, 5)
        problems.append((expr, ans))
    return problems


if __name__ == '__main__':
    from tabula_rasa.config import Config

    cfg = Config()
    tok = MathTokenizer()
    # Test dataset creation
    ds = MathDataset(tok, 1000)
    x, y = ds[0]
    print(f'Input shape: {len(x)}, Target shape: {len(y)}')
    print(f'Input: {tok.decode([i for i in x if i != tok.pad_id])}')
    print(f'Target (non-pad): {tok.decode([i for i in y if i != tok.pad_id])}')

    # Test a few problems
    print('\nSample problems:')
    for _ in range(5):
        expr, ans = generate_problem(1, 3)
        print(f'  {expr} = {ans}')
