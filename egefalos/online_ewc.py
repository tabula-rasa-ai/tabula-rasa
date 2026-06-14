"""Online Elastic Weight Consolidation for continual learning.
Merged Fisher matrix prevents catastrophic forgetting across sequential tasks.

Includes adaptive lambda/gamma tuning based on loss trajectory.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional
import json
from pathlib import Path


class OnlineEWC:
    """Online EWC: merge all past-task Fisher matrices into one running total."""

    def __init__(self, model: nn.Module, gamma: float = 0.9):
        """
        Args:
            model: The model whose parameters we're protecting
            gamma: Decay rate for old Fisher info (0.9 = keep 90% of old, add 10% new)
        """
        self.model = model
        self.gamma = gamma

        # merged fisher: F_combined = gamma * F_old + F_new
        self.fisher_dict: Dict[str, torch.Tensor] = {}

        # Anchor weights (theta*) — parameters at last consolidation
        self.anchor_dict: Dict[str, torch.Tensor] = {}

        # Metadata
        self.task_count = 0
        self.consolidation_steps = 0

    def compute_fisher(self, dataloader, num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix on a data sample.

        Fisher[param] = E[(d log p / d param)^2]

        We approximate: Fisher ≈ (1/N) Σ(grad²) over N examples
        """
        device = next(self.model.parameters()).device
        fisher = {name: torch.zeros_like(param)
                  for name, param in self.model.named_parameters()
                  if param.requires_grad}

        total_samples = num_samples if num_samples is not None else len(dataloader) if hasattr(dataloader, '__len__') else 100
        samples_seen = 0

        self.model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * (dataloader.batch_size if hasattr(dataloader, 'batch_size') else 1) >= total_samples:
                break

            # Handle both (input_ids, targets) tuples and dict-style batches
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_ids, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('input'))
                targets = batch.get('targets', batch.get('labels'))
            else:
                continue

            input_ids = input_ids.to(device) if torch.is_tensor(input_ids) else torch.tensor(input_ids).to(device)
            targets = targets.to(device) if torch.is_tensor(targets) else torch.tensor(targets).to(device)

            # Forward
            logits, loss, _ = self.model(input_ids, targets=targets)

            # Backward
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            samples_seen += input_ids.size(0)

        # Normalize by number of samples
        if samples_seen > 0:
            for name in fisher:
                fisher[name] /= samples_seen

        return fisher

    def merge_fisher(self, new_fisher: Dict[str, torch.Tensor], gamma: Optional[float] = None):
        """Merge new Fisher into running total using exponential decay.

        F_combined = gamma * F_old + (1 - gamma) * F_new

        Args:
            new_fisher: Fisher matrix for the new task
            gamma: Optional override for self.gamma. If provided, uses this
                   value for the merge (enables adaptive gamma per-merge).
        """
        gamma = self.gamma if gamma is None else gamma

        if not self.fisher_dict:
            # First task: just use the new Fisher
            self.fisher_dict = {k: v.clone() for k, v in new_fisher.items()}
        else:
            # Merge with decay
            for name in self.fisher_dict:
                if name in new_fisher:
                    self.fisher_dict[name] = (
                        gamma * self.fisher_dict[name] +
                        (1.0 - gamma) * new_fisher[name]
                    )

        self.task_count += 1

    @staticmethod
    def compute_fisher_overlap(fisher_a: Dict[str, torch.Tensor],
                                fisher_b: Dict[str, torch.Tensor]) -> float:
        """Compute cosine similarity between two Fisher matrices as a measure of task overlap.

        Returns a value in [0, 1] where:
        - 1.0 = identical Fisher structure (tasks share all parameter importance)
        - 0.0 = completely orthogonal Fisher structure (no parameter overlap)

        Uses the normalized inner product of flattened Fisher diagonals.
        """
        a_flat = torch.cat([v.flatten() for v in fisher_a.values()])
        b_flat = torch.cat([v.flatten() for v in fisher_b.values()])
        cos_sim = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))
        return cos_sim.item()

    @staticmethod
    def compute_adaptive_gamma(base_gamma: float, overlap: float,
                                min_gamma: float = 0.3, max_gamma: float = 0.95) -> float:
        """Compute adaptive gamma based on Fisher overlap.

        When overlap is high (similar tasks), keep gamma close to base_gamma.
        When overlap is low (divergent tasks), reduce gamma to blend more new info.

        gamma_adaptive = base_gamma * overlap + min_gamma * (1 - overlap)

        Args:
            base_gamma: The user-specified base decay rate (e.g., 0.9)
            overlap: Fisher cosine similarity in [0, 1]
            min_gamma: Minimum gamma when overlap → 0
            max_gamma: Maximum gamma (clamped)
        """
        adaptive = base_gamma * overlap + min_gamma * (1 - overlap)
        return max(min_gamma, min(max_gamma, adaptive))

    def save_anchor_weights(self):
        """Save current model weights as anchor (theta*) for EWC penalty."""
        self.anchor_dict = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def compute_ewc_penalty(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """Compute EWC penalty term: λ/2 · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²"""
        if not self.fisher_dict or not self.anchor_dict:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict and name in self.anchor_dict:
                # fisher_dict and anchor_dict are already on the correct device
                # (load() moves them; save_anchor_weights() clones in-place)
                f = self.fisher_dict[name]
                theta_delta = param - self.anchor_dict[name]
                penalty += (f * theta_delta ** 2).sum()

        return (lambda_ewc / 2.0) * penalty

    def save(self, path):
        """Serialize Fisher matrix and anchor weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        device = next(self.model.parameters()).device

        # Move to CPU for storage
        fisher_cpu = {}
        for k, v in self.fisher_dict.items():
            fisher_cpu[k] = v.detach().cpu() if v.device != torch.device('cpu') else v.detach().clone()

        anchor_cpu = {}
        for k, v in self.anchor_dict.items():
            anchor_cpu[k] = v.detach().cpu() if v.device != torch.device('cpu') else v.detach().clone()

        torch.save({
            'fisher': fisher_cpu,
            'anchor': anchor_cpu,
            'task_count': self.task_count,
            'gamma': self.gamma,
            'consolidation_steps': self.consolidation_steps,
        }, path)

    def load(self, path):
        """Load Fisher matrix and anchor weights from disk."""
        path = Path(path)
        if not path.exists():
            print(f"  [EWC] No checkpoint at {path}. Starting fresh.")
            return

        device = next(self.model.parameters()).device
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        self.fisher_dict = checkpoint['fisher']
        self.anchor_dict = checkpoint['anchor']
        self.task_count = checkpoint['task_count']
        self.gamma = checkpoint['gamma']
        self.consolidation_steps = checkpoint['consolidation_steps']

        # Move to model device
        for k in self.fisher_dict:
            self.fisher_dict[k] = self.fisher_dict[k].to(device)
        for k in self.anchor_dict:
            self.anchor_dict[k] = self.anchor_dict[k].to(device)

        print(f"  [EWC] Loaded {len(self.fisher_dict)} params, {self.task_count} tasks")

    # ── Adaptive Tuning ─────────────────────────────────────────

    def tune_lambda(self, recent_losses: list[float],
                    base_lambda: float = 1000.0,
                    window: int = 5,
                    spike_threshold: float = 2.0,
                    min_lambda: float = 100.0,
                    max_lambda: float = 5000.0) -> float:
        """Dynamically adjust EWC lambda based on loss trajectory.

        When the loss spikes suddenly (new task is very different),
        increase lambda to protect old knowledge. When loss is stable,
        decrease lambda to allow faster learning.

        Args:
            recent_losses: List of recent loss values (newest last).
            base_lambda: The user-configured base lambda.
            window: Lookback window for spike detection.
            spike_threshold: Multiple of rolling mean that counts as a spike.
            min_lambda: Minimum lambda (never go below).
            max_lambda: Maximum lambda (cap for safety).

        Returns:
            Adjusted lambda value.
        """
        if len(recent_losses) < window * 2:
            return base_lambda

        # Non-overlapping windows: compare the last `window` values
        # with the `window` values immediately before them.
        recent = recent_losses[-window:]
        prev = recent_losses[-(window * 2):-window]
        mean = sum(recent) / len(recent)
        prev_mean = sum(prev) / len(prev)

        # Relative change: spike ratio
        spike_ratio = mean / max(prev_mean, 1e-8)

        if spike_ratio > spike_threshold:
            # Loss jumped → protect old tasks aggressively
            multiplier = min(spike_ratio, 4.0)
            return min(base_lambda * multiplier, max_lambda)
        elif spike_ratio < 1.0 / spike_threshold:
            # Loss dropped → safe to reduce constraint
            return max(base_lambda * 0.5, min_lambda)

        return base_lambda

    def tune_gamma(self, overlap: Optional[float] = None,
                   base_gamma: float = 0.9) -> float:
        """Dynamically adjust gamma based on Fisher overlap.

        When overlap is high (similar tasks), keep gamma high to
        preserve old Fisher. When overlap is low (divergent tasks),
        lower gamma to blend in more new information.

        Args:
            overlap: Fisher cosine similarity in [0, 1], or None to
                     use the stored overlap heuristic.
            base_gamma: The user-configured base gamma.

        Returns:
            Adjusted gamma value.
        """
        if overlap is None:
            return base_gamma

        # Linear interpolation: high overlap → keep base gamma
        # Low overlap → reduce gamma toward 0.5 (blend more new info)
        return base_gamma * overlap + 0.5 * (1.0 - overlap)
