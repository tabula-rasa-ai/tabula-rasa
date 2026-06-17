"""Per-expert Elastic Weight Consolidation for MoE models.

Tracks separate Fisher Information matrices per expert so that training
a new task (e.g. multiplication) doesn't corrupt the expert weights that
a previous task (e.g. addition) learned. Shared parameters (attention,
embedding, router, LM head) still use a single global Fisher.

Usage:
    from tabula_rasa.cl.expert_ewc import ExpertEWC

    ewc = ExpertEWC(model, gamma=0.9, num_experts=4)
    ewc.save_anchor_weights()           # Before training new task
    fisher = ewc.compute_fisher(loader)  # On old-task data
    ewc.merge_fisher(fisher)             # Merge into running total
    ...
    loss = task_loss + ewc.compute_ewc_penalty(lambda_ewc=1000.0)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _is_expert_param(name: str) -> bool:
    """Check if a parameter name belongs to an expert feed-forward.

    Matches names like::
        layers.2.feed_forward.expert_gate.1.weight
        layers.0.feed_forward.expert_up.3.weight
    """
    return any(x in name for x in ("expert_gate", "expert_up", "expert_down"))


def _parse_expert_ids(name: str) -> tuple[int, int] | None:
    """Extract ``(layer_idx, expert_idx)`` from an expert parameter name.

    Args:
        name: Parameter name like ``layers.2.feed_forward.expert_gate.1.weight``.

    Returns:
        ``(layer_idx, expert_idx)`` tuple, or ``None`` if not an expert param.
    """
    m = re.search(r"layers\.(\d+).*?expert_\w+\.(\d+)\.weight", name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _flatten_key(name: str) -> str:
    """Normalise a parameter name for use in *expert_fisher* dicts.

    Strips the expert index from expert params so the same key works
    for all experts::

        layers.2.feed_forward.expert_gate.1.weight
            → layers.2.feed_forward.expert_gate.<idx>.weight
    """
    return re.sub(r"(expert_\w+)\.\d+\.(weight|bias)", r"\1.<idx>.\2", name)


class ExpertEWC:
    """Per-expert Elastic Weight Consolidation for MoE models.

    Three tiers of protection:

    1. **Global Fisher** — shared parameters (token embedding, attention,
       norms, router weights, LM head). One merged matrix for all tasks.
    2. **Per-expert Fisher** — each expert's gate/up/down weights tracked
       separately, keyed by ``(layer_idx, expert_idx)``.
    3. **Per-expert anchor** — snapshot of expert weights at consolidation.

    Attributes:
        model: The PyTorch model being protected.
        gamma: Exponential decay rate for merging old Fishers (default 0.9).
        num_experts: Number of experts per MoE layer.
        global_fisher: Fisher diagonal for shared (non-expert) params.
        expert_fisher: Per-expert Fishers keyed by ``(layer, expert)``.
        global_anchor: Anchor weights for shared params.
        expert_anchor: Per-expert anchor weights.
        task_count: Number of tasks merged so far.
    """

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.9,
        num_experts: int = 4,
    ) -> None:
        self.model = model
        self.gamma = gamma
        self.num_experts = num_experts

        # ── Global (shared) Fisher ───────────────────────────────
        self.global_fisher: Dict[str, torch.Tensor] = {}
        self.global_anchor: Dict[str, torch.Tensor] = {}

        # ── Per-expert Fisher ────────────────────────────────────
        # expert_fisher[(layer, expert)] = {flat_key: tensor}
        self.expert_fisher: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.expert_anchor: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}

        self.task_count: int = 0

    # ── Public API ───────────────────────────────────────────────

    def save_anchor_weights(self) -> None:
        """Snapshot current model weights as EWC anchors (theta*).

        Must be called **before** training a new task.
        """
        self.global_anchor = {}
        self.expert_anchor = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_expert_param(name):
                expert_key = _parse_expert_ids(name)
                if expert_key is None:
                    continue
                if expert_key not in self.expert_anchor:
                    self.expert_anchor[expert_key] = {}
                fk = _flatten_key(name)
                self.expert_anchor[expert_key][fk] = param.detach().clone()
            else:
                self.global_anchor[name] = param.detach().clone()

    def compute_fisher(
        self, dataloader, num_samples: Optional[int] = None
    ) -> None:
        """Compute Fisher matrix on a data sample and separate it into
        global and per-expert components. Results are stored internally
        and can be merged with :meth:`merge_fisher`.

        Args:
            dataloader: DataLoader yielding ``(input_ids, targets)``.
            num_samples: Max samples to use (default: all).
        """
        device = next(self.model.parameters()).device

        # Initialise accumulators
        global_fisher: Dict[str, torch.Tensor] = {}
        expert_fisher: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_expert_param(name):
                expert_key = _parse_expert_ids(name)
                if expert_key is None:
                    continue
                if expert_key not in expert_fisher:
                    expert_fisher[expert_key] = {}
                fk = _flatten_key(name)
                expert_fisher[expert_key][fk] = torch.zeros_like(param)
            else:
                global_fisher[name] = torch.zeros_like(param)

        total_samples = (
            num_samples
            if num_samples is not None
            else (
                len(dataloader)
                if hasattr(dataloader, "__len__")
                else 100
            )
        )
        samples_seen = 0
        self.model.train()

        for batch_idx, batch in enumerate(dataloader):
            if (
                batch_idx
                * (
                    dataloader.batch_size
                    if hasattr(dataloader, "batch_size")
                    else 1
                )
                >= total_samples
            ):
                break

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_ids, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch.get("input"))
                targets = batch.get("targets", batch.get("labels"))
            else:
                continue

            if torch.is_tensor(input_ids):
                input_ids = input_ids.to(device)
            else:
                input_ids = torch.tensor(input_ids).to(device)
            if torch.is_tensor(targets):
                targets = targets.to(device)
            else:
                targets = torch.tensor(targets).to(device)

            logits, loss, _ = self.model(input_ids, targets=targets)
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients — separate global vs expert
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                grad_sq = param.grad.detach() ** 2
                if _is_expert_param(name):
                    ek = _parse_expert_ids(name)
                    if ek is not None and ek in expert_fisher:
                        fk = _flatten_key(name)
                        expert_fisher[ek][fk] += grad_sq
                elif name in global_fisher:
                    global_fisher[name] += grad_sq

            samples_seen += input_ids.size(0)

        # Normalize
        if samples_seen > 0:
            for name in global_fisher:
                global_fisher[name] /= samples_seen
            for ek in expert_fisher:
                for fk in expert_fisher[ek]:
                    expert_fisher[ek][fk] /= samples_seen

        # Store internally for merge_fisher
        self._last_global_fisher = global_fisher
        self._last_expert_fisher = expert_fisher

    def merge_fisher(self, gamma: Optional[float] = None) -> None:
        """Merge the last computed Fisher into the running total.

        Global and per-expert Fishers are merged separately with the
        same exponential decay: ``F_combined = gamma * F_old + F_new``.

        Args:
            gamma: Override for ``self.gamma`` (adaptive gamma).
        """
        gamma = self.gamma if gamma is None else gamma

        if not hasattr(self, "_last_global_fisher"):
            raise RuntimeError("Call compute_fisher() before merge_fisher()")

        # ── Merge global Fisher ──────────────────────────────────
        if not self.global_fisher:
            self.global_fisher = {
                k: v.clone() for k, v in self._last_global_fisher.items()
            }
        else:
            for name in self.global_fisher:
                if name in self._last_global_fisher:
                    self.global_fisher[name] = (
                        gamma * self.global_fisher[name]
                        + self._last_global_fisher[name]
                    )

        # ── Merge per-expert Fishers ─────────────────────────────
        for ek, new_exp_f in self._last_expert_fisher.items():
            if ek not in self.expert_fisher:
                self.expert_fisher[ek] = {
                    k: v.clone() for k, v in new_exp_f.items()
                }
            else:
                for fk in self.expert_fisher[ek]:
                    if fk in new_exp_f:
                        self.expert_fisher[ek][fk] = (
                            gamma * self.expert_fisher[ek][fk]
                            + new_exp_f[fk]
                        )

        self.task_count += 1
        # Clear temporary storage
        del self._last_global_fisher
        del self._last_expert_fisher

    def compute_ewc_penalty(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """Compute per-expert EWC penalty.

        ``penalty = λ/2 · (global_penalty + Σ_expert_penalty)``

        The global term protects shared parameters (attention, norms,
        embedding, router, LM head). The per-expert terms protect each
        expert's gate/up/down independently, so unrelated tasks don't
        interfere with each other's expert subspaces.

        Args:
            lambda_ewc: Penalty strength.

        Returns:
            Scalar penalty tensor.
        """
        device = next(self.model.parameters()).device

        if not self.global_fisher:
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=device)

        # ── Global term ──────────────────────────────────────────
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_expert_param(name):
                continue
            if name in self.global_fisher and name in self.global_anchor:
                f = self.global_fisher[name]
                theta_delta = param - self.global_anchor[name]
                penalty += (f * theta_delta ** 2).sum()

        # ── Per-expert term ──────────────────────────────────────
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not _is_expert_param(name):
                continue
            ek = _parse_expert_ids(name)
            if ek is None:
                continue
            if ek not in self.expert_fisher or ek not in self.expert_anchor:
                continue
            fk = _flatten_key(name)
            if fk in self.expert_fisher[ek] and fk in self.expert_anchor[ek]:
                f = self.expert_fisher[ek][fk]
                theta_delta = param - self.expert_anchor[ek][fk]
                penalty += (f * theta_delta ** 2).sum()

        return (lambda_ewc / 2.0) * penalty

    # ── Serialization ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize all Fishers and anchors to disk.

        Separates global and per-expert structures for clean
        deserialization.

        Args:
            path: File path to save to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Move to CPU for storage
        global_f_cpu = {
            k: v.detach().cpu() for k, v in self.global_fisher.items()
        }
        global_a_cpu = {
            k: v.detach().cpu() for k, v in self.global_anchor.items()
        }

        expert_f_cpu = {}
        for ek, fdict in self.expert_fisher.items():
            expert_f_cpu[f"{ek[0]}_{ek[1]}"] = {
                k: v.detach().cpu() for k, v in fdict.items()
            }

        expert_a_cpu = {}
        for ek, adict in self.expert_anchor.items():
            expert_a_cpu[f"{ek[0]}_{ek[1]}"] = {
                k: v.detach().cpu() for k, v in adict.items()
            }

        torch.save(
            {
                "global_fisher": global_f_cpu,
                "global_anchor": global_a_cpu,
                "expert_fisher": expert_f_cpu,
                "expert_anchor": expert_a_cpu,
                "task_count": self.task_count,
                "gamma": self.gamma,
                "num_experts": self.num_experts,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load Fishers and anchors from disk.

        Args:
            path: File path to load from. No-op if file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            print(f"  [ExpertEWC] No checkpoint at {path}. Starting fresh.")
            return

        device = next(self.model.parameters()).device
        ckpt = torch.load(path, map_location=device, weights_only=True)

        # Restore global
        self.global_fisher = {k: v.to(device) for k, v in ckpt["global_fisher"].items()}
        self.global_anchor = {k: v.to(device) for k, v in ckpt["global_anchor"].items()}

        # Restore per-expert (keys stored as "layer_expert" strings)
        self.expert_fisher = {}
        for key_str, fdict in ckpt["expert_fisher"].items():
            parts = key_str.split("_")
            ek = (int(parts[0]), int(parts[1]))
            self.expert_fisher[ek] = {k: v.to(device) for k, v in fdict.items()}

        self.expert_anchor = {}
        for key_str, adict in ckpt["expert_anchor"].items():
            parts = key_str.split("_")
            ek = (int(parts[0]), int(parts[1]))
            self.expert_anchor[ek] = {k: v.to(device) for k, v in adict.items()}

        self.task_count = ckpt["task_count"]
        self.gamma = ckpt["gamma"]
        self.num_experts = ckpt["num_experts"]

        print(
            f"  [ExpertEWC] Loaded {len(self.global_fisher)} global params, "
            f"{len(self.expert_fisher)} expert Fishers, "
            f"{self.task_count} tasks"
        )

    # ── Convenience ──────────────────────────────────────────────

    @property
    def consolidated(self) -> bool:
        """``True`` if at least one Fisher has been merged."""
        return bool(self.global_fisher)
