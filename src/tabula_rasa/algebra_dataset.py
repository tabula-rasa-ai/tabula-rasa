"""Algebra dataset — linear equations, fractions, and symbolic math.

Extends Tabula Rasa beyond arithmetic into symbolic algebra.
Generates problems like:
  - Linear:        x + 3 = 7   →  x = 4
  - Two-step:      2x + 1 = 9  →  x = 4
  - Division:      x / 2 = 5   →  x = 10
  - Multiplication: 3x = 12    →  x = 4
  - Negative coeff: -x + 5 = 2 →  x = 3

The dataset maps naturally to the existing scratchpad approach:
the model outputs the solution as "x=N" where N is the answer digit.
"""

from __future__ import annotations

import random
from typing import ClassVar

from tabula_rasa.tokenizer import MathTokenizer


class AlgebraTokenizer(MathTokenizer):
    """Tokenizer that adds algebraic variable tokens to the math vocabulary.

    Extends MathTokenizer with variable names (x, y, z) for
    solving symbolic equations.

    Vocab: 4 special + 20 carry-digit + 18 math chars + 3 variables = 45 tokens
    """

    VARIABLE_CHARS: ClassVar[list[str]] = list("xyz")

    def _build_vocab(self) -> None:
        all_tokens: list[str] = (
            self.SPECIAL_TOKENS + self.CARRY_TOKENS + self.MATH_CHARS + self.VARIABLE_CHARS
        )
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.itos: dict[int, str] = {i: t for i, t in enumerate(all_tokens)}
        self.vocab_size: int = len(all_tokens)

        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]

        self._tokens_sorted = sorted(all_tokens, key=len, reverse=True)


class AlgebraDataset:
    """Generate algebraic equation problems with solutions.

    Problem types:
      - 'linear':    x + 3 = 7           (single step)
      - 'two-step':  2x + 1 = 9          (coefficient + constant)
      - 'division':  x / 2 = 5           (fraction coefficient)
      - 'neg-coeff': -x + 5 = 2          (negative coefficient)
      - 'mix':       random from all types
    """

    OPERATORS = ["+", "-", "*", "/"]
    VARIABLES = ["x", "y", "z"]

    def __init__(
        self,
        num_samples: int = 5000,
        problem_type: str = "mix",
        max_coeff: int = 12,
        max_constant: int = 20,
        allow_negative: bool = False,
    ):
        self.num_samples = num_samples
        self.problem_type = problem_type
        self.max_coeff = max_coeff
        self.max_constant = max_constant
        self.allow_negative = allow_negative

    def __len__(self) -> int:
        return self.num_samples

    def _rand_positive(self, max_val: int) -> int:
        return random.randint(1, max_val)

    def _generate_linear(self) -> tuple[str, str]:
        """Generate: x + a = b  →  x = b - a"""
        var = random.choice(self.VARIABLES)
        a = self._rand_positive(self.max_constant)
        x_val = self._rand_positive(self.max_constant)

        # Pick an operation
        op = random.choice([op for op in self.OPERATORS if op in ("+", "-")])

        if op == "+":
            # x + a = (x + a)
            b = x_val + a
            prompt = f"{var}+{a}="
            answer = f"{var}={x_val}"
        elif op == "-":
            # x - a = (x - a) where x > a for positive
            if x_val <= a:
                x_val, a = a + 1, x_val - 1
                if x_val <= 0:
                    x_val, a = 5, 2
            b = x_val - a
            prompt = f"{var}-{a}="
            answer = f"{var}={x_val}"
        else:
            prompt, answer = f"{var}+{a}=", f"{var}={x_val}"

        return prompt, answer

    def _generate_two_step(self) -> tuple[str, str]:
        """Generate: ax + b = c  →  x = (c - b) / a"""
        var = random.choice(self.VARIABLES)
        a = self._rand_positive(min(self.max_coeff, 10))
        x_val = self._rand_positive(10)
        op = random.choice(["+", "-"])

        ax = a * x_val
        if op == "+":
            b = self._rand_positive(20)
            c = ax + b
            prompt = f"{a}{var}+{b}="
        else:
            b = self._rand_positive(min(ax - 1, 20))
            c = ax - b
            prompt = f"{a}{var}-{b}="

        answer = f"{var}={x_val}"
        return prompt, answer

    def _generate_division(self) -> tuple[str, str]:
        """Generate: x / a = b  →  x = a * b"""
        var = random.choice(self.VARIABLES)
        a = self._rand_positive(min(self.max_coeff, 10))
        x_val = a * self._rand_positive(10)
        b = x_val // a
        prompt = f"{var}/{a}="
        answer = f"{var}={x_val}"
        return prompt, answer

    def _generate_neg_coeff(self) -> tuple[str, str]:
        """Generate: -x + a = b  →  x = a - b"""
        var = random.choice(self.VARIABLES)
        a = self._rand_positive(self.max_constant)
        x_val = self._rand_positive(min(self.max_constant, 10))

        # -x + a = (a - x_val)
        if x_val >= a:
            x_val = max(1, a - 1)
        b = a - x_val
        prompt = f"-{var}+{a}="
        answer = f"{var}={x_val}"
        return prompt, answer

    def __getitem__(self, idx: int) -> tuple[str, str]:
        ptype = self.problem_type
        if ptype == "mix":
            ptype = random.choice(["linear", "two-step", "division", "neg-coeff"])
        generators = {
            "linear": self._generate_linear,
            "two-step": self._generate_two_step,
            "division": self._generate_division,
            "neg-coeff": self._generate_neg_coeff,
        }
        gen = generators.get(ptype, self._generate_linear)
        return gen()


def evaluate_algebra(model, tok, dataset, num_samples: int = 50) -> dict:
    """Evaluate algebraic equation solving accuracy.

    Returns per-type accuracy and overall.
    """
    import torch

    model.eval()
    per_type: dict[str, dict] = {}
    overall_correct = 0

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            prompt, expected = dataset[i]
            # Determine problem type
            ptype = "other"
            if "--" not in prompt:
                if "/" in prompt:
                    ptype = "division"
                elif prompt.startswith("-"):
                    ptype = "neg-coeff"
                elif any(c.isalpha() for c in prompt) and any(c.isdigit() for c in prompt):
                    ptype = "two-step"
                else:
                    ptype = "linear"

            output = model.generate(tok, prompt, max_new_tokens=15, temperature=0.0)

            # Extract answer
            if "=" in output:
                raw = output.split("=", 1)[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
            else:
                raw = output.strip()

            is_correct = raw.strip() == expected.split("=")[-1].strip()

            if ptype not in per_type:
                per_type[ptype] = {"correct": 0, "total": 0}
            per_type[ptype]["total"] += 1
            if is_correct:
                per_type[ptype]["correct"] += 1
                overall_correct += 1

    model.train()
    overall = overall_correct / max(1, num_samples) * 100
    details = {k: v["correct"] / max(1, v["total"]) * 100 for k, v in per_type.items()}
    return {"overall": overall, "per_type": details}


# ══════════════════════════════════════════════════════════════
# Quick Test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test tokenizer
    tok = AlgebraTokenizer()
    print(f"AlgebraTokenizer: {tok.vocab_size} tokens")

    for test in ["x+3=", "2x+1=", "x/2=", "-x+5=", "x+y=z"]:
        ids = tok.encode(test, add_special_tokens=True)
        dec = tok.decode(ids, skip_special=True)
        print(f"  {test:>10} -> {ids} -> {dec}")

    # Test dataset
    ds = AlgebraDataset(num_samples=10, problem_type="mix")
    print("\nAlgebraDataset samples:")
    for i in range(8):
        p, a = ds[i]
        print(f"  {p:>12} -> {a}")
