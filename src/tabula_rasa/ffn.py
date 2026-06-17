"""Feed-forward blocks: standard SwiGLU/ReLU/GELU and Mixture-of-Experts."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabula_rasa.norms import _make_activation


class FeedForward(nn.Module):
    """Standard feed-forward block with gated (SwiGLU) or plain activation.

    SwiGLU::

        output = (gate(x) * up(x)).down(x)

    ReLU / GELU::

        output = up(x).activation().down(x)
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.activation = _make_activation(config.activation)

        if config.activation == "swiglu":
            # Gated variant: gate + up → element-wise multiply
            self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        else:
            self.gate = None
            self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate is not None:
            return self.down(self.activation(self.gate(x)) * self.up(x))
        return self.down(self.activation(self.up(x)))


class MoEFeedForward(nn.Module):
    """Mixture of Experts feed-forward — replaces a single FFN with top-k routing.

    Each token is routed to the top-k experts (by gating score). Only those
    experts run, keeping compute cost constant while expanding capacity.

    Different tasks naturally specialise different experts, so Fisher
    matrices stop overlapping — solving the 3-task EWC collapse.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = min(config.top_k, config.num_experts)
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        # Router: projects d_model → logits over experts
        self.router = nn.Linear(config.d_model, self.num_experts, bias=False)

        # Expert FFNs: each is a SwiGLU-style FFN (gate + up projections)
        self.expert_gate = nn.ModuleList([
            nn.Linear(config.d_model, config.d_ff, bias=False)
            for _ in range(self.num_experts)
        ])
        self.expert_up = nn.ModuleList([
            nn.Linear(config.d_model, config.d_ff, bias=False)
            for _ in range(self.num_experts)
        ])
        self.expert_down = nn.ModuleList([
            nn.Linear(config.d_ff, config.d_model, bias=False)
            for _ in range(self.num_experts)
        ])

        # Auxiliary loss (load balancing)
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        orig_shape = x.shape
        x_2d = x.view(-1, self.d_model)

        # Router logits and top-k selection
        router_logits = self.router(x_2d)  # (batch*seq, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute aux loss (load balance)
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            tokens_per_expert[e] = (top_k_indices == e).any(dim=-1).float().sum()
        # Importance: fraction of router probability mass per expert
        importance = routing_weights.mean(dim=0)
        aux_loss = (tokens_per_expert.float() / tokens_per_expert.sum() * importance).sum()
        self.last_aux_loss = aux_loss * self.num_experts

        # Dispatch tokens to experts
        final_output = torch.zeros_like(x_2d)
        for e in range(self.num_experts):
            mask = (top_k_indices == e).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_2d[mask]
            gate_out = F.silu(self.expert_gate[e](expert_input))
            up_out = self.expert_up[e](expert_input)
            expert_out = self.expert_down[e](gate_out * up_out)

            # Weight by routing scores
            expert_weights = top_k_weights[mask][(top_k_indices[mask] == e)]
            final_output[mask] += expert_out * expert_weights.unsqueeze(-1)

        return final_output.view(orig_shape)
