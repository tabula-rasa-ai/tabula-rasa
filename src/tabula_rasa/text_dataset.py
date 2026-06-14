"""Text dataset — string reversal for validating the BPETokenizer.

A simple text task that proves the transformer + BPE tokenizer work for
non-math domains. The model learns to reverse strings character by
character, testing its ability to handle arbitrary Unicode text.

Usage:
    ds = TextReversalDataset(num_samples=1000, max_len=8)
    prompt, answer = ds[0]   # e.g. ('hello=', 'olleh')
"""

from __future__ import annotations

import random
import string


class TextReversalDataset:
    """Dataset for string reversal — simplest general-text task.

    Generates random alphanumeric strings and their reversals.
    The model sees "abcde=" and must output "edcba".

    Supports curriculum: short strings first, then longer ones.
    """

    CHARS = string.ascii_lowercase + string.digits

    def __init__(
        self, num_samples: int = 5000, max_len: int = 8, min_len: int = 2, chars: str | None = None
    ):
        self.num_samples = num_samples
        self.max_len = max_len
        self.min_len = min_len
        self.chars = chars or self.CHARS

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[str, str]:
        length = random.randint(self.min_len, self.max_len)
        s = "".join(random.choice(self.chars) for _ in range(length))
        prompt = s + "="
        answer = s[::-1]
        return prompt, answer

    def generate_problem(self, length: int | None = None) -> tuple[str, str]:
        """Generate a single problem with optional fixed length."""
        if length is None:
            length = random.randint(self.min_len, self.max_len)
        s = "".join(random.choice(self.chars) for _ in range(length))
        return s + "=", s[::-1]


def evaluate_reversal(model, tokenizer, num_samples: int = 100, max_len: int = 8) -> dict:
    """Evaluate string reversal accuracy.

    Returns per-length accuracy dict and overall accuracy.
    """
    ds = TextReversalDataset(num_samples, max_len=max_len)
    per_length = {l: {"correct": 0, "total": 0} for l in range(2, max_len + 1)}

    total_correct = 0
    for i in range(num_samples):
        prompt, expected = ds[i]
        output = model.generate(
            tokenizer, prompt, max_new_tokens=len(expected) + 3, temperature=0.0
        )
        # Extract answer after '='
        if "=" in output:
            raw = output.split("=", 1)[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
        else:
            raw = output.strip()

        length = len(expected)
        is_correct = raw == expected
        if is_correct:
            total_correct += 1
            per_length[length]["correct"] += 1
        per_length[length]["total"] += 1

    overall = total_correct / num_samples * 100
    details = {
        str(l): (d["correct"] / d["total"] * 100) if d["total"] > 0 else 0
        for l, d in per_length.items()
    }
    return {"overall": overall, "per_length": details}


if __name__ == "__main__":
    ds = TextReversalDataset(100, max_len=6)
    for i in range(5):
        p, a = ds[i]
        print(f"  {p:12} -> {a}")
    print(f"\nTotal samples: {len(ds)}")

    # Test BPETokenizer compatibility
    from bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    text = "hello=olleh"
    ids = tok.encode(text)
    dec = tok.decode(ids)
    print(f"\nTokenizer test: {text} -> {ids} -> {dec}")
    print(f"  Vocab size: {tok.vocab_size}")
