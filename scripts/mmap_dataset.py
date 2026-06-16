"""Memory-mapped dataset — zero-copy loading for fast training."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MMapDataset(Dataset):
    """Dataset backed by memory-mapped tensors. Loads in microseconds."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.inputs = torch.load(str(self.data_dir / "inputs.pt"), mmap=True, weights_only=True)
        self.targets = torch.load(str(self.data_dir / "targets.pt"), mmap=True, weights_only=True)
        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)

    def __len__(self):
        return self.meta["num_samples"]

    def __getitem__(self, idx):
        return self.inputs[idx].long(), self.targets[idx].long()


def get_loader(data_dir: str, batch_size: int = 256, shuffle: bool = True):
    """Create a DataLoader with optimizations for CPU training."""
    ds = MMapDataset(data_dir)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # No workers — mmap loads instantly on main thread
        pin_memory=False,  # No benefit on CPU
        drop_last=True,
    )
