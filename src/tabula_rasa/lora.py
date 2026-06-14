"""Low-Rank Adaptation (LoRA) for efficient task-specific fine-tuning.

Freezes base model weights and adds trainable rank-decomposition matrices
to attention projections (q, k, v, o). Multiple LoRA adapters can be
swapped at inference time for different tasks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """LoRA adapter for a single nn.Linear layer.

    Adds: output = base(x) + alpha * (x @ A @ B)
    where A is (in_features, rank) and B is (rank, out_features).
    Base weights are frozen; only A and B are trainable.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        # Validate
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f'LoRALayer expects nn.Linear base, got {type(base_layer).__name__}'
            )
        if rank < 1:
            raise ValueError(f'LoRA rank must be >= 1, got {rank}')

        self.base = base_layer
        self.base.requires_grad_(False)  # freeze base weights
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA matrices: A is random init (Gaussian), B is zero init
        # This ensures the update is zero at initialisation.
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank) * 0.02
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + LoRA update.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f'in={self.base.in_features}, out={self.base.out_features}, '
            f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}'
        )


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
) -> list[LoRALayer]:
    """Replace target linear layers with LoRA-wrapped versions.

    Searches all modules in the model and wraps any ``nn.Linear`` whose name
    contains one of the *target_modules* substrings.  The base linear layer
    is frozen; only the LoRA ``A`` and ``B`` matrices are trainable.

    Args:
        model: The transformer model (e.g. ``MathTransformer``).
        rank: LoRA rank (default ``8``).
        alpha: LoRA scaling constant (default ``1.0``).
        target_modules: Which weight types to adapt.  Default is
            ``['wq', 'wk', 'wv', 'wo']`` (attention projections).

    Returns:
        List of ``LoRALayer`` instances that were inserted.  The caller
        should hold onto this list for subsequent save/load/switch calls.
    """
    if target_modules is None:
        target_modules = ['wq', 'wk', 'wv', 'wo']

    lora_layers: list[LoRALayer] = []

    for name, module in model.named_modules():
        if not any(t in name for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Build the LoRA wrapper
        lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
        lora_layers.append(lora_layer)

        # Walk the module hierarchy to set the attribute at the right level
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lora_layer)

    return lora_layers


def save_lora_adapters(
    lora_layers: list[LoRALayer],
    path: str,
):
    """Save LoRA adapter weights (A and B matrices only).

    The saved file contains only the low-rank matrices, making it very
    lightweight compared to a full model checkpoint.

    Args:
        lora_layers: List of ``LoRALayer`` instances (from
            ``apply_lora_to_model``).
        path: Destination file path (e.g. ``'specialists/math/add/lora.pt'``).
    """
    state: dict = {}
    for i, layer in enumerate(lora_layers):
        state[f'lora_{i}_A'] = layer.lora_A.detach().cpu()
        state[f'lora_{i}_B'] = layer.lora_B.detach().cpu()
        state[f'lora_{i}_rank'] = layer.rank
        state[f'lora_{i}_alpha'] = layer.alpha
        state[f'lora_{i}_in_features'] = layer.base.in_features
        state[f'lora_{i}_out_features'] = layer.base.out_features
    torch.save(state, path)


def load_lora_adapters(
    model: nn.Module,
    path: str,
    rank: int = 8,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
) -> list[LoRALayer]:
    """Load LoRA adapter weights into an already-wrapped model.

    Applies LoRA wrapping to *model* (if not already applied) and then
    populates the ``A`` and ``B`` matrices from the saved file.

    Args:
        model: The transformer model.
        path: Path to a saved ``.pt`` file produced by
            ``save_lora_adapters``.
        rank: LoRA rank to use if the saved file does not contain ranks
            (backward compat).
        alpha: LoRA alpha to use if the saved file does not contain alphas.
        target_modules: Substrings identifying which layers to adapt.

    Returns:
        The list of ``LoRALayer`` instances (now populated).
    """
    state = torch.load(path, map_location='cpu', weights_only=True)

    # Determine rank/alpha from saved state if available
    if f'lora_0_rank' in state:
        rank = int(state['lora_0_rank'])
    if f'lora_0_alpha' in state:
        alpha = float(state['lora_0_alpha'])

    lora_layers = apply_lora_to_model(model, rank=rank, alpha=alpha,
                                      target_modules=target_modules)

    for i, layer in enumerate(lora_layers):
        key_a = f'lora_{i}_A'
        key_b = f'lora_{i}_B'
        if key_a in state:
            with torch.no_grad():
                layer.lora_A.data.copy_(state[key_a].to(layer.lora_A.device))
        if key_b in state:
            with torch.no_grad():
                layer.lora_B.data.copy_(state[key_b].to(layer.lora_B.device))

    return lora_layers


def set_lora_trainable(model: nn.Module, trainable: bool = True):
    """Set only LoRA parameters as trainable (freeze everything else).

    When *trainable* is ``True`` (default), all parameters whose name
    contains ``'lora_'`` are set to require gradients, and all other
    parameters are frozen.  When *trainable* is ``False``, the entire
    model is frozen.

    Args:
        model: The model (with LoRA layers applied).
        trainable: Whether LoRA parameters should be trainable.
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad_(trainable)
        else:
            param.requires_grad_(not trainable)


def switch_lora_adapters(
    model: nn.Module,
    path: str,
) -> list[LoRALayer]:
    """Swap LoRA adapters in-place, returning the updated layer list.

    Unlike ``load_lora_adapters`` (which applies LoRA wrapping from scratch),
    this function assumes the model already has ``LoRALayer`` instances in
    the target positions and simply overwrites their ``A`` and ``B`` matrices
    with the saved values.

    Args:
        model: The model with existing ``LoRALayer`` wrappers.
        path: Path to a saved ``.pt`` file produced by ``save_lora_adapters``.

    Returns:
        The list of ``LoRALayer`` instances (now pointing to the new adapter).
    """
    state = torch.load(path, map_location='cpu', weights_only=True)

    # Collect existing LoRALayers
    lora_layers: list[LoRALayer] = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_layers.append(module)

    # Overwrite A and B matrices
    for i, layer in enumerate(lora_layers):
        key_a = f'lora_{i}_A'
        key_b = f'lora_{i}_B'
        if key_a in state:
            with torch.no_grad():
                layer.lora_A.data.copy_(state[key_a].to(layer.lora_A.device))
        if key_b in state:
            with torch.no_grad():
                layer.lora_B.data.copy_(state[key_b].to(layer.lora_B.device))

    return lora_layers
