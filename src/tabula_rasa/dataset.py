"""Generate synthetic math problems for training."""

import random
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tabula_rasa.tokenizer import MathTokenizer


OPERATIONS = ['+', '-', '*', '/']


def generate_problem(min_digits=1, max_digits=4):
    """Generate an arithmetic problem and its answer.

    Returns (expression_str, answer_str) e.g. ('12+34', '46')
    """
    op = random.choice(OPERATIONS)

    a_digits = random.randint(min_digits, max_digits)
    b_digits = random.randint(min_digits, max_digits)

    a = random.randint(10**(a_digits-1), 10**a_digits - 1)
    b = random.randint(10**(b_digits-1), 10**b_digits - 1)

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


def format_math_sample(expr, answer):
    """Format as 'question=answer' e.g. '12+34=46'"""
    return f'{expr}={answer}'


class MathDataset(Dataset):
    """Dataset of synthetic math problems."""

    def __init__(self, tokenizer: MathTokenizer, num_samples: int,
                 min_digits: int = 1, max_digits: int = 4,
                 seed: int = 42):
        self.tokenizer = tokenizer
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.max_seq_len = tokenizer.max_seq_len if hasattr(tokenizer, 'max_seq_len') else 64

        rng = random.Random(seed)
        self.samples = []
        for _ in range(num_samples):
            expr, ans = generate_problem(min_digits, max_digits)
            text = format_math_sample(expr, ans)
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= self.max_seq_len:
                self.samples.append(ids)

        print(f'Created {len(self.samples)} valid samples (filtered from {num_samples})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # Pad to max_seq_len
        padded = ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(ids))
        # Input: all tokens except last
        x = padded[:-1]
        # Target: all tokens except first (shifted)
        y = padded[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def generate_test_set(num_samples=100, seed=123):
    """Generate a clean test set of problems for evaluation."""
    rng = random.Random(seed)
    problems = []
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
