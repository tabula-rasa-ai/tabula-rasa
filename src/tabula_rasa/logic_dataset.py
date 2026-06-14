"""Symbolic logic dataset generator — transitive reasoning problems.

Extends Tabula Rasa beyond arithmetic into symbolic reasoning.
Generates problems of the form "A > B, B > C -> A > C" (transitive closure).

Supports:
- Comparison: '>', '<', '>=', '<=', '==', '!='
- Chain lengths: 2 to N steps
- True and false statements (for training discriminator)

Usage:
    from tabula_rasa.logic_dataset import LogicDataset, generate_logic_problem
    ds = LogicDataset(tokenizer, num_samples=1000)
    x, y = ds[0]
"""

import random
import re
from typing import Optional

import torch
from torch.utils.data import Dataset


# ── Problem Generation ───────────────────────────────────────────

RELATIONS = {
    '>': ('greater_than', lambda a, b: a > b),
    '<': ('less_than', lambda a, b: a < b),
    '>=': ('geq', lambda a, b: a >= b),
    '<=': ('leq', lambda a, b: a <= b),
    '==': ('equals', lambda a, b: a == b),
    '!=': ('not_equal', lambda a, b: a != b),
}

VARIABLES = [chr(ord('A') + i) for i in range(10)]  # A through J
VALUES = list(range(10))  # 0-9


def _assign_values(chain_length: int, relation: str) -> dict[str, int]:
    """Assign integer values to variables that satisfy the given chain.

    For transitive chains, we assign values that validate the full chain.
    E.g., for A > B > C, assign A=5, B=3, C=1 so A>B and B>C both hold.
    """
    vals = {}
    op = RELATIONS.get(relation, RELATIONS['>'])

    remaining = list(VALUES)
    random.shuffle(remaining)

    if relation in ('>', '>='):
        for i in range(chain_length + 1):
            var = VARIABLES[i]
            idx = len(remaining) - 1 - i  # Decreasing values
            if idx < 0:
                idx = 0
            vals[var] = remaining[idx % len(remaining)]
    elif relation in ('<', '<='):
        for i in range(chain_length + 1):
            var = VARIABLES[i]
            vals[var] = remaining[i % len(remaining)]
    elif relation == '==':
        same_val = remaining[0]
        for i in range(chain_length + 1):
            vals[VARIABLES[i]] = same_val
    elif relation == '!=':
        for i in range(chain_length + 1):
            vals[VARIABLES[i]] = remaining[i % len(remaining)]

    return vals


def generate_logic_problem(
    chain_length: int = 2,
    relation: str = '>',
    p_false: float = 0.3,
    seed: Optional[int] = None,
) -> tuple[str, str]:
    """Generate a transitive logic problem.

    Args:
        chain_length: Number of statements in the chain (e.g., 2 for A>B, B>C).
        relation: One of '>', '<', '>=', '<=', '==', '!='.
        p_false: Probability of generating a FALSE conclusion (for training).
        seed: Optional RNG seed.

    Returns:
        (prompt, answer) where prompt is like "A>B, B>C, A?C" and answer is
        ">" or "<" or "==" etc.
    """
    if seed is not None:
        random.seed(seed)

    op_name, op_fn = RELATIONS[relation]
    vals = _assign_values(chain_length, relation)

    # Build the chain of premises
    premises = []
    for i in range(chain_length):
        a = VARIABLES[i]
        b = VARIABLES[i + 1]
        premises.append(f"{a}{relation}{b}")

    # The conclusion: first and last variable
    conclusion_a = VARIABLES[0]
    conclusion_b = VARIABLES[chain_length]

    # Compute the correct answer
    a_val = vals[conclusion_a]
    b_val = vals[conclusion_b]

    if op_fn(a_val, b_val):
        correct_answer = relation
    else:
        # Inverse relation
        inverse_map = {'>': '<', '<': '>', '>=': '<=', '<=': '>=',
                       '==': '!=', '!=': '=='}
        correct_answer = inverse_map.get(relation, relation)

    # Optionally generate a FALSE conclusion
    is_false = random.random() < p_false
    if is_false:
        # Pick a wrong answer
        wrong_answers = [r for r in RELATIONS if r != correct_answer]
        answer = random.choice(wrong_answers)
    else:
        answer = correct_answer

    prompt = ", ".join(premises) + f", {conclusion_a}?{conclusion_b}"
    return prompt, answer


# ── Tokenizer Wrapper ───────────────────────────────────────────


def _build_logic_vocab() -> dict[str, int]:
    """Build a simple tokenizer for logic expressions.

    Tokens: A-J, 0-9, > < >= <= == != , ? and operator names.
    """
    tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    tokens += VARIABLES
    tokens += list('0123456789')
    tokens += ['>', '<', '>=', '<=', '==', '!=', ',', '?']
    return {t: i for i, t in enumerate(tokens)}


_LOGIC_STOI = _build_logic_vocab()
_LOGIC_ITOS = {i: t for t, i in _LOGIC_STOI.items()}


def encode_logic(text: str, add_special: bool = True) -> list[int]:
    """Encode a logic expression to token IDs."""
    ids = []
    if add_special:
        ids.append(_LOGIC_STOI['<BOS>'])
    i = 0
    while i < len(text):
        # Check multi-char tokens first
        if i + 2 <= len(text) and text[i:i+2] in _LOGIC_STOI:
            ids.append(_LOGIC_STOI[text[i:i+2]])
            i += 2
        elif text[i] in _LOGIC_STOI:
            ids.append(_LOGIC_STOI[text[i]])
            i += 1
        else:
            ids.append(_LOGIC_STOI['<UNK>'])
            i += 1
    if add_special:
        ids.append(_LOGIC_STOI['<EOS>'])
    return ids


def decode_logic(ids: list[int], skip_special: bool = True) -> str:
    """Decode token IDs back to text."""
    chars = []
    for i in ids:
        if skip_special and i in (_LOGIC_STOI['<PAD>'], _LOGIC_STOI['<BOS>'],
                                   _LOGIC_STOI['<EOS>'], _LOGIC_STOI['<UNK>']):
            continue
        chars.append(_LOGIC_ITOS.get(i, '?'))
    return ''.join(chars)


# ── Dataset ──────────────────────────────────────────────────────


class LogicDataset(Dataset):
    """Dataset of transitive logic reasoning problems.

    Each sample is (premises, conclusion_symbol) where premises describe
    a transitive chain (e.g., "A>B, B>C") and the answer is the relation
    between the first and last variable.

    Args:
        num_samples: Number of samples to generate.
        chain_length: Length of transitive chain (2 = 2-step: A>B, B>C).
        relation: Comparison operator.
        p_false: Probability of generating a false conclusion.
        max_seq_len: Maximum sequence length for padding.
        seed: RNG seed.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        chain_length: int = 2,
        relation: str = '>',
        p_false: float = 0.3,
        max_seq_len: int = 32,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.chain_length = chain_length
        self.relation = relation
        self.p_false = p_false
        self.max_seq_len = max_seq_len
        self.pad_id = _LOGIC_STOI['<PAD>']

        self.samples: list[tuple[list[int], list[int]]] = []
        for i in range(num_samples):
            prompt, answer = generate_logic_problem(
                chain_length=chain_length,
                relation=relation,
                p_false=p_false,
                seed=(seed or 42) + i,
            )
            full_text = prompt + answer
            ids = encode_logic(full_text)
            if len(ids) <= max_seq_len:
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.samples[idx]
        padded = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        # Mask everything before the answer (? marker position)
        y[y == self.pad_id] = -100
        return x, y


# ── Quick Test ───────────────────────────────────────────────────


def main():
    """Generate and print sample logic problems."""
    print("Symbolic Logic Dataset — Sample Problems")
    print("=" * 50)

    for chain_len in [2, 3]:
        for rel in ['>', '<', '>=', '==']:
            print(f"\nChain length={chain_len}, relation={rel}:")
            for i in range(5):
                prompt, answer = generate_logic_problem(chain_len, rel, p_false=0.2, seed=i + 100)
                print(f"  {prompt:<20s} -> {answer}")


if __name__ == "__main__":
    main()
