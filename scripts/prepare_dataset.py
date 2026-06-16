"""Pre-tokenize dataset and save as memory-mapped tensor for fast training.

Usage:
    python3 prepare_dataset.py         # Tokenize math dataset
    python3 prepare_dataset.py --op add  # Tokenize addition only
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import json
import sys
from pathlib import Path

import torch

from tabula_rasa.config import Config
from tabula_rasa.dataset import MathDataset
from tabula_rasa.tokenizer import MathTokenizer


def prepare_dataset(op: str = None, num_samples=100000, min_digits=1, max_digits=4):
    cfg = Config()
    tok = MathTokenizer()
    tok.max_seq_len = cfg.max_seq_len

    if op:
        # Specialist dataset: filter to one operation
        from train_specialist import SpecialistDataset

        print(f"Tokenizing {op} dataset ({num_samples} samples)...")
        ds = SpecialistDataset(tok, op, num_samples, min_digits, max_digits)
        out_dir = Path("specialists") / "math" / op / "data"
    else:
        # Generalist dataset
        print(f"Tokenizing general dataset ({num_samples} samples)...")
        ds = MathDataset(tok, num_samples, min_digits, max_digits)
        out_dir = Path("data")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert all samples to tensors
    all_x, all_y = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        all_x.append(x.unsqueeze(0))
        all_y.append(y.unsqueeze(0))

    X = torch.cat(all_x, dim=0).to(torch.int16)
    Y = torch.cat(all_y, dim=0).to(torch.int16)

    # Save as memory-mapped files
    torch.save(X, str(out_dir / "inputs.pt"))
    torch.save(Y, str(out_dir / "targets.pt"))

    # Save metadata
    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "num_samples": len(ds),
                "seq_len": cfg.max_seq_len - 1,
                "vocab_size": tok.vocab_size,
                "op": op,
                "dtype": "int16",
            },
            f,
        )

    print(f"Saved {len(ds)} samples to {out_dir}/")
    print(f"  Inputs:  {X.shape} ({X.element_size() * X.nelement() / 1e6:.1f} MB)")
    print(f"  Targets: {Y.shape} ({Y.element_size() * Y.nelement() / 1e6:.1f} MB)")


if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else None
    prepare_dataset(op)
