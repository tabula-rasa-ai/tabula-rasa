"""Neural Circuit Architectural Priors — composable circuit modules as inductive bias.

Reference [2] from the PDF: "NEURAL CIRCUIT ARCHITECTURAL PRIORS" (OpenReview).
The idea: bake small, predefined circuit-like patterns into the architecture
so the model doesn't have to learn basic operations from scratch.

For arithmetic, this provides strong inductive bias:
  - CarryDetector: AND-like — both digits >= 5 trigger carry
  - BorrowDetector: XOR-like — a_digit < b_digit triggers borrow  
  - DigitComparator: returns which operand is larger
  - Incrementer: adds 1 (for handling carries)
  - ColumnSummer: a + b + carry_in → digit_out, carry_out

Each circuit is a tiny, interpretable module. The model learns to route
information through the right circuits rather than learning arithmetic from scratch.

Usage:
    from tabula_rasa.circuit_priors import ArithmeticCircuitLayer
    layer = ArithmeticCircuitLayer(d_model=128)
    output = layer(hidden_states)  # adds circuit-processed information
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  Individual Circuit Primitives
#  Each is a small differentiable module that implements a basic operation.
# ═══════════════════════════════════════════════════════════════════════


class CarryDetector(nn.Module):
    """AND-like circuit: detects when a carry is needed (both digits >= 5).

    In arithmetic, a carry occurs when a_digit + b_digit >= 10.
    This is approximately: (a >= 5) AND (b >= 5).
    Provides a strong inductive bias for column-wise addition.

    Input:  (batch, 2*d) — two digit representations concatenated
    Output: (batch, d)   — carry signal
    """

    def __init__(self, dim: int = 128, hidden: int = 16):
        super().__init__()
        # Tiny circuit: down-project to hidden, compute AND-like gate
        self.project = nn.Linear(dim * 2, hidden, bias=False)
        self.gate = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.project(x))
        # AND-like gating: sigmoid product creates AND behavior
        g = torch.sigmoid(self.gate(h))
        h = h * g
        return self.output(h)


class BorrowDetector(nn.Module):
    """XOR-like circuit: detects when a borrow is needed (a_digit < b_digit).

    In subtraction, a borrow occurs when the top digit is smaller than
    the bottom digit in a column. This is essentially XOR-like:
    (a < b) → borrow.

    Input:  (batch, 2*d) — two digit representations
    Output: (batch, d)   — borrow signal
    """

    def __init__(self, dim: int = 128, hidden: int = 16):
        super().__init__()
        self.project = nn.Linear(dim * 2, hidden, bias=False)
        self.compare = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.project(x))
        # Comparison gate: learn relative ordering
        c = torch.tanh(self.compare(h))
        h = h * c
        return self.output(h)


class Incrementer(nn.Module):
    """Adds 1 to a digit value (handles carry propagation).

    When a carry propagates to the next column, we need to increment
    that column's result. This module implements that operation.

    Input:  (batch, d) — digit representation
    Output: (batch, d) — incremented digit
    """

    def __init__(self, dim: int = 128, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ColumnSummer(nn.Module):
    """Computes a + b + carry_in → digit_out, carry_out.

    This is the core column operation for addition. It takes two digit
    representations and a carry signal, and produces the result digit
    and next carry.

    Input:  (batch, 3*d) — a_digit, b_digit, carry_in
    Output: (batch, 2*d) — digit_out, carry_out
    """

    def __init__(self, dim: int = 128, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, dim * 2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DigitProjector(nn.Module):
    """Projects token embeddings into digit-space for circuit processing.

    Extracts the "digit-ness" from a token embedding so circuit modules
    can operate on it.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Tanh(),  # squash to [-1, 1] for circuit operations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
#  ArithmeticCircuitLayer — drop-in replacement for part of FFN
# ═══════════════════════════════════════════════════════════════════════


class ArithmeticCircuitLayer(nn.Module):
    """A transformer block enhancement that adds circuit-based inductive bias.

    Processes hidden states through small, interpretable circuit modules
    alongside the normal feed-forward network. The idea is that arithmetic
    operations are easier to learn when the architecture already has
    carry/borrow detectors built in.

    Args:
        d_model: Transformer hidden dimension.
        hidden: Circuit internal dimension (small, default 16).
        use_carry: Enable carry detector circuit.
        use_borrow: Enable borrow detector circuit.
        use_increment: Enable incrementer circuit.
        use_column_sum: Enable column summer circuit.

    Usage in model.py:
        self.circuit_layer = ArithmeticCircuitLayer(config.d_model)
        h = h + self.circuit_layer(h)  # residual addition
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden: int = 16,
        use_carry: bool = True,
        use_borrow: bool = True,
        use_increment: bool = True,
        use_column_sum: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Digit projector — converts any hidden state to circuit-space
        self.projector = DigitProjector(d_model)

        # Circuit modules (each is tiny — 16-dim hidden)
        self.carry_detector = CarryDetector(d_model, hidden) if use_carry else None
        self.borrow_detector = BorrowDetector(d_model, hidden) if use_borrow else None
        self.incrementer = Incrementer(d_model, hidden) if use_increment else None
        self.column_summar = ColumnSummer(d_model, hidden) if use_column_sum else None

        # Gating: learn how much circuit signal to mix in
        self.gate = nn.Parameter(torch.zeros(1))

        # Which circuits are active
        self.active_circuits = sum([use_carry, use_borrow, use_increment, use_column_sum])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply circuit priors to hidden states.

        Args:
            x: Hidden states (batch, seq_len, d_model).

        Returns:
            Circuit-enhanced states (batch, seq_len, d_model).
        """
        batch, seq, dim = x.shape
        circuit_out = torch.zeros_like(x)

        # Project to circuit space
        h = self.projector(x)  # (batch, seq, d_model)

        # Apply circuits to consecutive token pairs
        # This simulates digit-column processing
        for i in range(seq - 1):
            pair = torch.cat([h[:, i:i+1], h[:, i+1:i+2]], dim=-1)  # (batch, 1, 2*d)

            if self.carry_detector is not None:
                carry = self.carry_detector(pair)  # (batch, 1, d_model)
                circuit_out[:, i:i+1] = circuit_out[:, i:i+1] + carry

            if self.borrow_detector is not None:
                borrow = self.borrow_detector(pair)
                circuit_out[:, i:i+1] = circuit_out[:, i:i+1] + borrow

            if self.incrementer is not None and i > 0:
                inc = self.incrementer(h[:, i:i+1])
                circuit_out[:, i:i+1] = circuit_out[:, i:i+1] + inc

            if self.column_summar is not None and i < seq - 1 and i > 0:
                prev = h[:, i-1:i]
                curr = h[:, i:i+1]
                nxt = h[:, i+1:i+2]
                triple = torch.cat([prev, curr, nxt], dim=-1)  # (batch, 1, 3*d)
                col_out = self.column_summar(triple)  # (batch, 1, 2*d)
                digit_out, carry_out = col_out.chunk(2, dim=-1)
                circuit_out[:, i:i+1] = circuit_out[:, i:i+1] + digit_out
                if i + 1 < seq:
                    circuit_out[:, i+1:i+2] = circuit_out[:, i+1:i+2] + carry_out

        # Gated residual mixing
        gate_val = torch.sigmoid(self.gate)
        return gate_val * circuit_out


# ═══════════════════════════════════════════════════════════════════════
#  Integration helper — adds circuit priors to existing MathTransformer
# ═══════════════════════════════════════════════════════════════════════


def add_circuit_priors(model: nn.Module, hidden: int = 16) -> list[ArithmeticCircuitLayer]:
    """Add ArithmeticCircuitLayer to each transformer block.

    This modifies the model in-place, inserting circuit layers
    after each block's feed-forward network.

    Args:
        model: MathTransformer instance.
        hidden: Circuit internal dimension.

    Returns:
        List of added ArithmeticCircuitLayer instances.
    """
    layers = []
    d_model = model.config.d_model if hasattr(model, "config") else 128
    for block in model.layers:
        circuit = ArithmeticCircuitLayer(d_model=d_model, hidden=hidden)
        # Store on the block for forward pass integration
        block.circuit_layer = circuit
        layers.append(circuit)
    return layers


# ═══════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  NEURAL CIRCUIT ARCHITECTURAL PRIORS")
    print("=" * 60)

    layer = ArithmeticCircuitLayer(d_model=128, hidden=16)
    total = sum(p.numel() for p in layer.parameters())
    print(f"  Total circuit params: {total:,}")
    print(f"  Active circuits: {layer.active_circuits}")

    x = torch.randn(2, 8, 128)
    out = layer(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Output norm: {out.norm().item():.4f}")
    print()

    # Test integration with a small transformer
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 32; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 64
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 32
    model = MathTransformer(cfg)

    circuits = add_circuit_priors(model, hidden=8)
    print(f"  Added {len(circuits)} circuit layers ({sum(p.numel() for c in circuits for p in c.parameters()):,} extra params)")

    # Modify forward: inject circuit output into each block
    # (Integration with TransformerBlock forward is handled externally)

    print()
    print("  Neural Circuit Architectural Priors: OK")
    print("=" * 60)
