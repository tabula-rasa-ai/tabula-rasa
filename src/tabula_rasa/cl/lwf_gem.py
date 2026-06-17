"""Learning without Forgetting (LwF) for continual learning.

Distillation-based approach: when training on a new task, the model's
previous-task predictions are used as soft targets (knowledge distillation).
The loss becomes: L_new + lambda * L_distill(old_preds, new_preds).

This prevents catastrophic forgetting by forcing the model to maintain
its old-task output distribution while adapting to new tasks.

Usage:
    lwf = LwF(model, lambda_distill=0.5, temperature=2.0)

    # Before training on new task:
    old_logits = lwf.snapshot_old_model(dataloader_old)

    # During training (after loss.backward(), before optimizer.step()):
    # The LwF loss is added automatically via compute_lwf_loss()

Reference:
    Li & Hoiem (2017), "Learning without Forgetting"
    https://arxiv.org/abs/1606.09282
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LwF:
    """Learning without Forgetting — knowledge distillation for CL.

    Maintains a frozen copy of the model from before training on a new task.
    The student (current model) is trained to minimize both the new-task loss
    and the distillation loss with the frozen teacher.

    Args:
        model: The current (student) model being trained.
        lambda_distill: Weight of the distillation loss (0 = no distillation).
        temperature: Softmax temperature for distillation (higher = softer targets).
                     Standard values: 1.0-4.0. Higher values spread probability mass.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_distill: float = 0.5,
        temperature: float = 2.0,
    ):
        self.model = model
        self.lambda_distill = lambda_distill
        self.temperature = temperature
        self.teacher: Optional[nn.Module] = None
        self.device = next(model.parameters()).device

    def snapshot_teacher(self) -> None:
        """Create a frozen copy of the current model as the teacher.

        Call this BEFORE training on a new task to capture the old-task
        prediction distribution.
        """
        import copy
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        print(f"  [LwF] Teacher snapshot taken ({sum(p.numel() for p in self.teacher.parameters()):,} params frozen)")

    def compute_distillation_loss(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute distillation loss between teacher and student logits.

        Uses KL divergence between the softmax outputs of teacher and student,
        scaled by temperature^2 to keep gradient magnitudes consistent.

        Args:
            x: Input batch (token IDs).
            temperature: Override temperature. Uses self.temperature if None.

        Returns:
            Scalar distillation loss tensor, or 0 if no teacher set.
        """
        if self.teacher is None:
            return torch.tensor(0.0, device=self.device)

        T = temperature if temperature is not None else self.temperature

        with torch.no_grad():
            teacher_logits, _, _ = self.teacher(x, targets=None)
            teacher_logits = teacher_logits.detach()

        student_logits, _, _ = self.model(x, targets=None)

        # KL divergence: softmax(student/T) vs softmax(teacher/T)
        # Multiply by T^2 to scale gradients properly
        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean',
            log_target=False,
        ) * (T ** 2)

        return loss

    def get_total_loss(
        self,
        task_loss: torch.Tensor,
        x: torch.Tensor,
        lambda_distill: Optional[float] = None,
    ) -> torch.Tensor:
        """Combine task loss with distillation loss.

        Args:
            task_loss: Standard cross-entropy or task-specific loss.
            x: Input batch (for teacher forward pass).
            lambda_distill: Override lambda. Uses self.lambda_distill if None.

        Returns:
            Combined loss tensor.
        """
        lam = lambda_distill if lambda_distill is not None else self.lambda_distill
        distill = self.compute_distillation_loss(x)

        if distill.item() == 0.0:
            return task_loss

        total = task_loss + lam * distill
        return total

    def get_summary(self) -> dict:
        """Return diagnostic info."""
        return {
            'lambda_distill': self.lambda_distill,
            'temperature': self.temperature,
            'has_teacher': self.teacher is not None,
            'device': str(self.device),
        }


class GEM:
    """Gradient Episodic Memory — project gradients to prevent forgetting.

    Maintains a small episodic memory of past tasks. Before each optimizer step,
    computes whether the proposed gradient would increase loss on any past task.
    If so, projects the gradient onto the closest direction that doesn't increase
    any past-task loss.

    This is a simplified version of the full GEM algorithm (Lopez-Paz & Ranzato, 2017),
    using gradient projection similar to OGD but constrained by actual task losses.

    Args:
        model: The model being trained.
        memory_size: Max examples stored per task in episodic memory.
        margin: Allow small loss increase (default 0.0 = strict non-increasing).
        gamma: Reference gradient decay (0.9 = keep 90% of old direction).
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 256,
        margin: float = 0.0,
        gamma: float = 0.9,
    ):
        self.model = model
        self.memory_size = memory_size
        self.margin = margin
        self.gamma = gamma
        self.device = next(model.parameters()).device

        # Episodic memory: list of dicts, one per task
        # Each dict: {'inputs': tensor, 'targets': tensor}
        self._memory: list[dict[str, torch.Tensor]] = []

        # Reference gradients per task (for projection)
        self._ref_grads: list[dict[str, torch.Tensor]] = []
        self._task_count = 0

    def add_to_memory(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Add examples to the episodic memory for the current task.

        Maintains a fixed-size reservoir per task — oldest examples are evicted
        when the memory is full.

        Args:
            inputs: Input token IDs (batch x seq_len).
            targets: Target token IDs (batch x seq_len).
        """
        if not self._memory or self._task_count >= len(self._memory):
            # New task — add a new memory slot
            self._memory.append({'inputs': inputs.cpu(), 'targets': targets.cpu()})
            self._ref_grads.append({})
            self._task_count = max(self._task_count, len(self._memory) - 1)
        else:
            # Existing task — reservoir sample
            mem = self._memory[self._task_count]
            n_current = mem['inputs'].size(0) if mem['inputs'].dim() > 1 else 0
            n_new = inputs.size(0)

            if n_current + n_new <= self.memory_size:
                # Append
                mem['inputs'] = torch.cat([mem['inputs'], inputs.cpu()], dim=0)
                mem['targets'] = torch.cat([mem['targets'], targets.cpu()], dim=0)
            else:
                # Reservoir sampling: replace random old examples
                import random
                idxs = random.sample(range(n_current), min(n_new, n_current))
                for i, idx in enumerate(idxs):
                    if i < n_new:
                        mem['inputs'][idx] = inputs[i].cpu()
                        mem['targets'][idx] = targets[i].cpu()

    def store_gradients(self, dataloader, num_batches: int = 5) -> None:
        """Compute and store reference gradient directions from episodic memory.

        Args:
            dataloader: DataLoader for current task data.
            num_batches: Number of batches to sample.
        """
        if not self._memory:
            return

        self.model.train()
        batch_grads: dict[str, torch.Tensor] = {}
        batch_count = 0

        for batch in dataloader:
            if batch_count >= num_batches:
                break

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                continue

            x = x.to(self.device) if torch.is_tensor(x) else torch.tensor(x).to(self.device)
            y = y.to(self.device) if torch.is_tensor(y) else torch.tensor(y).to(self.device)

            self.model.zero_grad()
            _, loss, _ = self.model(x, targets=y)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    grad_vec = param.grad.detach().clone().view(-1).cpu()
                    if name not in batch_grads:
                        batch_grads[name] = grad_vec
                    else:
                        batch_grads[name] = (
                            (batch_grads[name] * batch_count + grad_vec) /
                            (batch_count + 1)
                        )
            batch_count += 1

        self.model.zero_grad()

        if batch_grads:
            # Merge with existing refs using gamma decay
            task_idx = max(0, len(self._ref_grads) - 1)
            existing = self._ref_grads[task_idx] if self._ref_grads else {}

            merged = {}
            for name, grad in batch_grads.items():
                if name in existing:
                    merged[name] = self.gamma * existing[name] + (1 - self.gamma) * grad
                else:
                    merged[name] = grad

            if self._ref_grads:
                self._ref_grads[task_idx] = merged
            else:
                self._ref_grads.append(merged)

            print(f"  [GEM] Stored ref gradients for task {task_idx + 1} "
                  f"({len(merged)} param groups, {batch_count} batches)")

    def project_gradients(self) -> float:
        """Project gradients to prevent increasing past-task loss.

        For each parameter group, checks if the proposed gradient would increase
        loss on any past task (using stored ref gradients). If so, projects to
        the nearest direction that doesn't increase any past loss.

        Returns mean cosine similarity with original gradients.
        """
        if not self._ref_grads:
            return 1.0

        total_sim = 0.0
        n_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            g = param.grad.data
            flat_g = g.view(-1)
            original_norm = flat_g.norm().item()

            if original_norm < 1e-8:
                continue

            # Check each past task's reference gradient
            for task_refs in self._ref_grads[:-1]:  # Skip current task
                if name not in task_refs:
                    continue
                ref = task_refs[name].to(g.device)
                ref_norm = ref.norm()
                if ref_norm < 1e-8:
                    continue

                # If g·ref < 0, the gradient decreases past-task loss (OK, no projection)
                # If g·ref > 0, the gradient increases past-task loss (project away)
                dot = torch.dot(flat_g, ref.view(-1))
                if dot > self.margin:
                    # Project out the component parallel to ref
                    ref_unit = ref.view(-1) / ref_norm
                    proj = dot * ref_unit
                    flat_g.sub_(proj)

            param.grad.data = flat_g.view(g.shape)

            new_norm = flat_g.norm().item()
            if original_norm > 1e-8 and new_norm > 1e-8:
                cos_sim = (flat_g * g.view(-1)).sum() / (original_norm * new_norm)
                total_sim += cos_sim.item()
                n_params += 1

        return total_sim / max(1, n_params)

    def new_task(self) -> None:
        """Signal that a new task is starting."""
        self._task_count = len(self._memory)
        print(f"  [GEM] New task {self._task_count + 1} — memory has {self._task_count} stored tasks")

    def get_summary(self) -> dict:
        """Return diagnostic info."""
        return {
            'memory_size': self.memory_size,
            'n_tasks_in_memory': len(self._memory),
            'n_ref_tasks': len(self._ref_grads),
            'margin': self.margin,
            'gamma': self.gamma,
            'device': str(self.device),
        }


# ─── Quick Test ──────────────────────────────────────────────

def main():
    """Quick test of LwF and GEM with a tiny model."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer

    print("CL Methods — LwF + GEM Quick Test")
    print("=" * 50)

    cfg = Config()
    cfg.d_model = 32
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.d_ff = 64
    cfg.vocab_size = 44

    model = MathTransformer(cfg)
    x = torch.randint(0, 44, (4, 16))

    # Test LwF
    print("\n  [LwF]")
    lwf = LwF(model, lambda_distill=0.5, temperature=2.0)
    lwf.snapshot_teacher()
    total = lwf.get_total_loss(torch.tensor(1.0, requires_grad=True), x)
    print(f"    Total loss (task=1.0 + distill): {total.item():.4f}")
    print(f"    Summary: {lwf.get_summary()}")
    print(f"  → LwF OK")

    # Test GEM
    print("\n  [GEM]")
    gem = GEM(model, memory_size=128)
    gem.add_to_memory(x, x)
    print(f"    Memory: {gem._memory[0]['inputs'].shape}")
    y = torch.randint(0, 44, (4, 16))
    gem.add_to_memory(y, y)
    print(f"    Memory after 2 calls: {gem._memory[0]['inputs'].shape}")
    print(f"    Summary: {gem.get_summary()}")

    # Test gradient projection without stored refs (no-op)
    loss = model(x, targets=x)[1]
    loss.backward()
    sim = gem.project_gradients()
    print(f"    No refs projection: cos_sim={sim:.4f} (expected ~1.0)")
    print(f"  → GEM OK")

    print(f"\n{'='*50}")
    print("  LwF + GEM — Quick Test PASSED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
