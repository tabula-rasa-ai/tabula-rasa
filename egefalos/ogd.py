"""Orthogonal Gradient Descent (OGD) for continual learning.

Projects gradients onto the orthogonal subspace of previous tasks' gradient
directions, preventing updates that would interfere with old knowledge.

Works alongside EWC: EWC adds a penalty term to the loss, OGD modifies
the gradients directly. They're complementary.

Usage:
    ogd = OGD(model, alpha=0.5)
    
    # After loss.backward(), before optimizer.step():
    ogd.project_gradients()
    optimizer.step()
    optimizer.zero_grad()
    
    # After training on a task, store the gradient subspace:
    ogd.store_gradients(dataloader, num_batches=10)
"""

import torch
import torch.nn as nn
from typing import Optional


class OGD:
    """Orthogonal Gradient Descent for continual learning.

    Maintains a set of reference gradient directions from previous tasks.
    Before each optimizer step, projects the current gradients onto the
    orthogonal complement of the reference subspace, so only gradient
    components that don't interfere with past tasks are applied.

    The reference subspace is built from the gradient directions that
    were active during past-task training.

    Args:
        model: The neural network model.
        alpha: Projection strength (0 = no projection, 1 = full).
               Default 0.5 balances old-task protection with new-task learning.
        use_cuda: Whether to use GPU (auto-detected from model).
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.5,
        use_cuda: Optional[bool] = None,
    ):
        self.model = model
        self.alpha = alpha
        self.device = next(model.parameters()).device
        self.use_cuda = use_cuda if use_cuda is not None else self.device.type == 'cuda'

        # Reference gradient subspace: list of (param_name, grad_vector) per task
        # Each task stores a list of gradient vectors (one per stored batch).
        self._reference_grads: list[dict[str, torch.Tensor]] = []

        # Flag to skip projection during gradient accumulation
        self._active = True

    def project_gradients(self) -> float:
        """Project current gradients onto the orthogonal complement of
        the reference subspace.

        For each parameter, computes:
          g_proj = g - alpha * sum_over_refs( proj(g, ref_i) )

        Where proj(a, b) = (a·b / b·b) * b  (vector projection).

        Returns the mean cosine similarity between current and projected
        gradients (for monitoring how much the gradient changed).
        """
        if not self._reference_grads or not self._active:
            return 1.0

        total_sim = 0.0
        n_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if not param.requires_grad:
                continue

            g = param.grad.data
            flat_g = g.view(-1)
            original_norm = flat_g.norm().item()

            if original_norm < 1e-8:
                continue

            # Subtract projections onto each reference direction
            for task_refs in self._reference_grads:
                if name not in task_refs:
                    continue
                ref = task_refs[name].to(g.device)

                # Vector projection: (g · r_hat) * r_hat
                ref_norm = ref.norm()
                if ref_norm < 1e-8:
                    continue
                ref_unit = ref / ref_norm
                dot = torch.dot(flat_g, ref_unit)
                proj = self.alpha * dot * ref_unit

                flat_g.sub_(proj)

            # Reshape projected gradient back
            param.grad.data = flat_g.view(g.shape)

            # Compute cosine similarity for monitoring
            new_norm = flat_g.norm().item()
            if original_norm > 1e-8 and new_norm > 1e-8:
                cos_sim = (flat_g * g.view(-1)).sum() / (original_norm * new_norm)
                total_sim += cos_sim.item()
                n_params += 1

        return total_sim / max(1, n_params)

    def store_gradients(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
    ) -> None:
        """Compute and store reference gradient directions from a task's data.

        Runs forward-backward on batches from the dataloader without
        actually applying the optimizer step, and stores the resulting
        gradient directions for future projection.

        Args:
            dataloader: DataLoader for the task's training data.
            num_batches: Number of batches to sample for reference gradients.
        """
        self._active = False  # Don't project during reference collection
        self.model.train()

        batch_grads: dict[str, torch.Tensor] = {}
        batch_count = 0

        for batch in dataloader:
            if batch_count >= num_batches:
                break

            # Handle batch format
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x = batch.get('input_ids', batch.get('input'))
                y = batch.get('targets', batch.get('labels'))
            else:
                continue

            x = x.to(self.device) if torch.is_tensor(x) else torch.tensor(x).to(self.device)
            y = y.to(self.device) if torch.is_tensor(y) else torch.tensor(y).to(self.device)

            self.model.zero_grad()
            _, loss, _ = self.model(x, targets=y)
            loss.backward()

            # Accumulate or store gradient per parameter
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    grad_vec = param.grad.detach().clone().view(-1).cpu()
                    if name not in batch_grads:
                        batch_grads[name] = grad_vec
                    else:
                        # Average gradient across batches
                        batch_grads[name] = (
                            (batch_grads[name] * batch_count + grad_vec) /
                            (batch_count + 1)
                        )

            batch_count += 1

        self.model.zero_grad()
        self._active = True

        if batch_grads:
            self._reference_grads.append(batch_grads)
            param_count = sum(v.numel() for v in batch_grads.values())
            print(f"  [OGD] Stored {len(batch_grads)} param groups "
                  f"({param_count:,} params) from {batch_count} batches")

    def clear_references(self) -> None:
        """Clear all stored reference gradients."""
        self._reference_grads.clear()
        print("  [OGD] Reference gradients cleared")

    @property
    def num_tasks(self) -> int:
        """Number of tasks stored in the reference set."""
        return len(self._reference_grads)

    def get_summary(self) -> dict:
        """Return diagnostic info about stored references."""
        return {
            'num_tasks': self.num_tasks,
            'param_groups_per_task': [
                len(refs) for refs in self._reference_grads
            ],
            'alpha': self.alpha,
            'device': str(self.device),
        }


# ═══════════════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════════════


def main():
    """Test OGD with a small model."""
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer
    from tabula_rasa.model import MathTransformer

    print("OGD — Quick Test")
    print("=" * 50)

    cfg = Config()
    cfg.d_model = 32
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.d_ff = 64
    cfg.vocab_size = 44

    model = MathTransformer(cfg)
    ogd = OGD(model, alpha=0.5)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Alpha: {ogd.alpha}")

    # Create dummy data
    tok = MathTokenizer()
    x = torch.randint(0, 44, (4, 16))
    y = torch.randint(0, 44, (4, 16))

    # Test projection without references (should be no-op)
    loss = model(x, targets=y)[1]
    loss.backward()
    sim = ogd.project_gradients()
    print(f"  No references: cos_sim={sim:.4f} (expected ~1.0)")

    # Store a reference gradient
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2)
    ogd.store_gradients(loader, num_batches=2)
    print(f"  Stored: {ogd.num_tasks} task(s)")

    # Test projection with references
    model.zero_grad()
    loss = model(x, targets=y)[1]
    loss.backward()
    sim = ogd.project_gradients()
    print(f"  With references: cos_sim={sim:.4f} (expected < 1.0)")

    print(f"\n  Summary: {ogd.get_summary()}")
    print("\n  OK!")


if __name__ == "__main__":
    main()
