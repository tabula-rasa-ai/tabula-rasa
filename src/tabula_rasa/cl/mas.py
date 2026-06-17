"""Memory Aware Synapses (MAS) for continual learning.

MAS is a regularization-based method that estimates parameter importance
by measuring how much the squared L2 norm of the output changes when a
parameter is perturbed. Unlike EWC, MAS does not require access to task
labels or log-probabilities — it only needs the model's output.

The importance weight for parameter theta_i is:

    Omega_i = (1/N) * sum_k |d(||f(x_k)||^2) / d_theta_i|

where f(x_k) is the model's output for input x_k.

Reference:
    Aljundi et al., "Memory Aware Synapses: Learning what (not) to forget"
    ECCV 2018. https://arxiv.org/abs/1711.09601

Usage:
    mas = MAS(model, lambda_reg=1000)
    mas.compute_importance(dataloader, num_samples=100)
    mas.consolidate()
    # During training:
    loss = task_loss + mas.compute_penalty(model)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class MAS:
    """Memory Aware Synapses -- regularisation-based continual learning.

    Estimates per-parameter importance by measuring the sensitivity of
    the squared L2 output norm to parameter perturbations. The penalty
    term Omega_i * (theta_i - theta*_i)^2 discourages changes to
    important parameters.

    Supports both single-task consolidation (merge importance matrices
    via exponential decay or max) and multi-task archive mode.

    Args:
        model: The neural network whose parameters we protect.
        lambda_reg: Regularisation strength (default 1000).
        merge_mode: How to merge importance across tasks:
            'max' -- take element-wise maximum (preserves history)
            'sum' -- additive accumulation
            'decay' -- exponential decay (gamma * old + new)
        gamma: Decay rate for 'decay' merge mode (default 0.9).
        use_archive: If True, store per-task importance matrices
            separately (like Fisher Archive for EWC).
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_reg: float = 1000.0,
        merge_mode: str = "max",
        gamma: float = 0.9,
        use_archive: bool = False,
    ):
        self.model = model
        self.lambda_reg = lambda_reg
        self.merge_mode = merge_mode
        self.gamma = gamma
        self.use_archive = use_archive

        # Merged importance (standard mode)
        self.omega_dict: Dict[str, torch.Tensor] = {}

        # Importance Archive (per-task, when use_archive=True)
        self.omega_archive: list[Dict[str, torch.Tensor]] = []

        # Anchor weights: theta* -- parameters at last consolidation
        self.anchor_dict: Dict[str, torch.Tensor] = {}

        # Anchor Archive: per-task anchors (when use_archive=True)
        self.anchor_archive: list[Dict[str, torch.Tensor]] = []

        # Per-task lambda weights for the archive
        self.archive_lambdas: list[float] = []

        # Metadata
        self.task_count = 0

    def compute_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute MAS importance weights from model outputs.

        For each sample, computes Omega = ||model(x)||^2 (squared L2 norm of
        the output logits), then gets dOmega/dtheta for each parameter.
        The importance is the mean absolute gradient across samples.

        This is O(N) per sample and requires only the model's forward
        pass -- no labels or log-probabilities are needed.

        Args:
            dataloader: DataLoader yielding batches of (input_ids, targets)
                or dict-style batches. Only input_ids are used.
            num_samples: Number of samples to process. If None, uses the
                full dataloader.

        Returns:
            Dict mapping parameter name -> importance tensor (Omega_i).
        """
        device = next(self.model.parameters()).device
        omega: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                omega[name] = torch.zeros_like(param)

        total_samples = (
            num_samples
            if num_samples is not None
            else (len(dataloader) if hasattr(dataloader, "__len__") else 100)
        )
        samples_seen = 0

        self.model.train()

        for batch_idx, batch in enumerate(dataloader):
            if samples_seen >= total_samples:
                break

            # Extract inputs (we don't need targets for MAS)
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                input_ids = batch[0]
            elif isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch.get("input"))
            else:
                continue

            if not torch.is_tensor(input_ids):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(device)
            batch_size = input_ids.size(0)

            # Forward: get logits (no targets = no loss computation,
            # but model still returns logits, loss=None, value=None)
            logits, _, _ = self.model(input_ids)

            # Compute Omega = ||output||^2 (squared L2 norm of logits)
            # We use the squared L2 norm of the logit vector per token,
            # summed across all tokens, averaged over batch.
            output_sq = logits.norm(dim=-1)  # (batch, seq_len) -- L2 norm per token
            omega_loss = output_sq.pow(2).sum(dim=-1)  # (batch,) -- sum over seq
            omega_loss = omega_loss.mean()  # scalar -- mean over batch

            # Backward to get dOmega/dtheta
            self.model.zero_grad()
            omega_loss.backward()

            # Accumulate |dOmega/dtheta|
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in omega:
                    omega[name] += param.grad.detach().abs()

            samples_seen += batch_size

        if samples_seen > 0:
            for name in omega:
                omega[name] /= samples_seen

        return omega

    def consolidate(
        self,
        new_omega: Optional[Dict[str, torch.Tensor]] = None,
        lambda_task: Optional[float] = None,
    ):
        """Store current parameters as anchors and merge importance.

        After computing importance on a task, call this to save the
        parameters that should be protected and merge the new importance
        into the running estimate.

        Args:
            new_omega: Importance dict from ``compute_importance()``.
                If None, uses ``self.omega_dict`` (must have been set
                via ``set_omega()`` or a prior call).
            lambda_task: Per-task lambda weight (archive mode only).
                If provided, stored for per-task weighting in the
                penalty computation.
        """
        # Determine which omega to store
        omega_to_store = new_omega if new_omega is not None else self.omega_dict

        if self.use_archive:
            # Archive mode: store per-task importance + anchor snapshot
            if omega_to_store:
                self.omega_archive.append(
                    {k: v.clone() for k, v in omega_to_store.items()}
                )
                self.anchor_archive.append(
                    {
                        name: param.detach().clone()
                        for name, param in self.model.named_parameters()
                        if param.requires_grad
                    }
                )
                self.archive_lambdas.append(lambda_task or self.lambda_reg)
                self.task_count += 1
            return

        # Standard mode: merge with running total
        if not self.omega_dict:
            # First task: store omega and anchor
            if new_omega is not None:
                self.omega_dict = {k: v.clone() for k, v in new_omega.items()}
            self.anchor_dict = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        else:
            # Merge new omega into running total
            source_omega = new_omega if new_omega is not None else omega_to_store
            if source_omega and self.omega_dict:
                if self.merge_mode == "max":
                    for name in self.omega_dict:
                        if name in source_omega:
                            self.omega_dict[name] = torch.max(
                                self.omega_dict[name], source_omega[name]
                            )
                elif self.merge_mode == "sum":
                    for name in self.omega_dict:
                        if name in source_omega:
                            self.omega_dict[name] = (
                                self.omega_dict[name] + source_omega[name]
                            )
                elif self.merge_mode == "decay":
                    for name in self.omega_dict:
                        if name in source_omega:
                            self.omega_dict[name] = (
                                self.gamma * self.omega_dict[name]
                                + (1.0 - self.gamma) * source_omega[name]
                            )

        self.task_count += 1

    def compute_penalty(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute MAS regularisation penalty.

        Penalty = sum_i Omega_i * (theta_i - theta*_i)^2 / 2

        Where Omega_i is the importance of parameter i, theta_i is the
        current value, and theta*_i is the anchor value.

        In archive mode, sums over all archived per-task importance
        matrices:

            penalty = sum_i lambda_i/2 * sum_j Omega_ij * (theta_j - theta*_ij)^2

        Args:
            model: The model to compute penalty for. If None, uses
                self.model.

        Returns:
            Scalar penalty tensor.
        """
        model = model or self.model
        device = next(model.parameters()).device

        if self.use_archive and self.omega_archive:
            # Archive mode: sum over all per-task penalties
            terms = []
            for task_idx, omega_dict in enumerate(self.omega_archive):
                anchor_dict = self.anchor_archive[task_idx]
                lam = (
                    self.archive_lambdas[task_idx]
                    if task_idx < len(self.archive_lambdas)
                    else self.lambda_reg
                )
                for name, param in model.named_parameters():
                    if (
                        param.requires_grad
                        and name in omega_dict
                        and name in anchor_dict
                    ):
                        delta = param - anchor_dict[name]
                        terms.append(
                            (lam / 2.0) * (omega_dict[name] * delta ** 2).sum()
                        )
            return sum(terms) if terms else torch.tensor(0.0, device=device)

        # Standard mode: single merged importance
        if not self.omega_dict or not self.anchor_dict:
            return torch.tensor(0.0, device=device)

        terms = []
        for name, param in model.named_parameters():
            if (
                param.requires_grad
                and name in self.omega_dict
                and name in self.anchor_dict
            ):
                delta = param - self.anchor_dict[name]
                terms.append((self.omega_dict[name] * delta ** 2).sum())

        return (self.lambda_reg / 2.0) * (
            sum(terms) if terms else torch.tensor(0.0, device=device)
        )

    def save_anchor_weights(self):
        """Save current model weights as anchor (theta*) for MAS penalty."""
        self.anchor_dict = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def set_omega(self, omega: Dict[str, torch.Tensor]):
        """Set the importance matrix directly (e.g., after loading)."""
        self.omega_dict = {k: v.clone() for k, v in omega.items()}

    def save(self, path):
        """Serialize importance matrix and anchor weights to disk."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "lambda_reg": self.lambda_reg,
            "merge_mode": self.merge_mode,
            "gamma": self.gamma,
            "use_archive": self.use_archive,
            "task_count": self.task_count,
            "archive_lambdas": self.archive_lambdas,
        }

        if self.use_archive:
            archive_omega = []
            for o_dict in self.omega_archive:
                archive_omega.append(
                    {k: v.detach().cpu() for k, v in o_dict.items()}
                )
            archive_a = []
            for a_dict in self.anchor_archive:
                archive_a.append(
                    {k: v.detach().cpu() for k, v in a_dict.items()}
                )
            save_dict["omega_archive"] = archive_omega
            save_dict["anchor_archive"] = archive_a
        else:
            omega_cpu = {k: v.detach().cpu() for k, v in self.omega_dict.items()}
            anchor_cpu = {k: v.detach().cpu() for k, v in self.anchor_dict.items()}
            save_dict["omega"] = omega_cpu
            save_dict["anchor"] = anchor_cpu

        torch.save(save_dict, path)

    def load(self, path):
        """Load importance matrix and anchor weights from disk."""
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            print(f"  [MAS] No checkpoint at {path}. Starting fresh.")
            return

        device = next(self.model.parameters()).device
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        self.lambda_reg = checkpoint.get("lambda_reg", self.lambda_reg)
        self.merge_mode = checkpoint.get("merge_mode", self.merge_mode)
        self.gamma = checkpoint.get("gamma", self.gamma)
        self.task_count = checkpoint.get("task_count", 0)
        self.use_archive = checkpoint.get("use_archive", False)
        self.archive_lambdas = checkpoint.get("archive_lambdas", [])

        if self.use_archive and "omega_archive" in checkpoint:
            self.omega_archive = []
            for o_dict in checkpoint["omega_archive"]:
                self.omega_archive.append(
                    {k: v.to(device) for k, v in o_dict.items()}
                )
            self.anchor_archive = []
            for a_dict in checkpoint["anchor_archive"]:
                self.anchor_archive.append(
                    {k: v.to(device) for k, v in a_dict.items()}
                )
            print(
                f"  [MAS] Loaded archive: {len(self.omega_archive)} tasks, "
                f"{self.task_count} total"
            )
        else:
            self.omega_dict = checkpoint.get("omega", {})
            self.anchor_dict = checkpoint.get("anchor", {})
            for k in self.omega_dict:
                self.omega_dict[k] = self.omega_dict[k].to(device)
            for k in self.anchor_dict:
                self.anchor_dict[k] = self.anchor_dict[k].to(device)
            print(
                f"  [MAS] Loaded {len(self.omega_dict)} params, "
                f"{self.task_count} tasks"
            )

    def get_summary(self) -> dict:
        """Return diagnostic info about stored importance matrices."""
        return {
            "lambda_reg": self.lambda_reg,
            "merge_mode": self.merge_mode,
            "gamma": self.gamma,
            "use_archive": self.use_archive,
            "task_count": self.task_count,
            "archive_tasks": len(self.omega_archive),
            "param_groups": len(self.omega_dict),
            "total_importance_params": sum(
                v.numel() for v in self.omega_dict.values()
            ),
        }


# ============================================================
# Quick Test
# ============================================================


def main():
    """Test MAS with a small model."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from torch.utils.data import DataLoader, TensorDataset

    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer

    print("MAS -- Quick Test")
    print("=" * 50)

    cfg = Config()
    cfg.d_model = 32
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.d_ff = 64
    cfg.vocab_size = 44

    model = MathTransformer(cfg)
    mas = MAS(model, lambda_reg=1000.0)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Lambda: {mas.lambda_reg}")

    # Create dummy data
    x = torch.randint(0, 44, (4, 16))
    y = torch.randint(0, 44, (4, 16))

    # Test penalty without importance (should be zero)
    penalty = mas.compute_penalty(model)
    print(f"  Penalty (no importance): {penalty.item():.4f} (expected ~0.0)")

    # Compute importance
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2)
    omega = mas.compute_importance(loader, num_samples=4)
    print(f"  Importance computed: {len(omega)} param groups")

    # Consolidate with new omega
    mas.consolidate(new_omega=omega)
    print(f"  Consolidated: {mas.task_count} task(s)")

    # Test penalty with importance
    penalty = mas.compute_penalty(model)
    print(f"  Penalty (with importance, same params): {penalty.item():.4f} (expected ~0.0)")

    print(f"\n  Summary: {mas.get_summary()}")
    print("\n  OK!")


if __name__ == "__main__":
    main()
