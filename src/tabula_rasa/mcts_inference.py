"""Inference-time MCTS for MathTransformer — bridges pattern-matching to reasoning.

Instead of greedy/beam decoding, runs a lightweight MCTS search at generation time.
Each simulation explores alternative token paths, using the model's logits as the
policy prior and an optional value head to evaluate partial solutions.

Usage:
    from tabula_rasa.mcts_inference import mcts_generate
    answer = mcts_generate(model, tokenizer, "12+34", num_simulations=32)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F


@dataclass
class MCTSNode:
    """Node in the inference-time MCTS tree.

    Each node = a token prefix. Children = possible next tokens.
    """

    token_id: int = 0
    parent: MCTSNode | None = None
    visit_count: int = 0
    total_value: float = 0.0
    prior_log_prob: float = 0.0  # log-prob from model policy
    children: dict[int, MCTSNode] = field(default_factory=dict)
    is_terminal: bool = False
    depth: int = 0

    @property
    def mean_value(self) -> float:
        """Q(s,a) — average value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb_score(self, c_puct: float, parent_visit: int) -> float:
        """PUCT score: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)."""
        q = self.mean_value
        prior = math.exp(self.prior_log_prob)  # convert log-prob to prob
        if parent_visit == 0:
            return q + c_puct * prior
        u = c_puct * prior * math.sqrt(parent_visit) / (1.0 + self.visit_count)
        return q + u


class MCTSInference:
    """Inference-time MCTS for math transformer decoding.

    Args:
        model: MathTransformer (with or without value head).
        tokenizer: MathTokenizer.
        num_simulations: MCTS rollouts per token position (default 16).
        c_puct: Exploration constant (default 1.0).
        temperature: Softmax temperature for action selection (default 0.5).
        value_fn: Optional custom value function. If None, uses:
            1. Model's value head if available (use_value_head=True)
            2. Normalized logit confidence as fallback
        max_new_tokens: Max tokens to generate (default 20).
        top_k: Only consider top-k tokens at each step (default 10).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        num_simulations: int = 16,
        c_puct: float = 1.0,
        temperature: float = 0.5,
        value_fn: Callable | None = None,
        max_new_tokens: int = 20,
        top_k: int = 10,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.value_fn = value_fn or self._default_value_fn
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k

        # Determine device
        self.device = next(model.parameters()).device

        # Check if model has a value head
        self.has_value_head = hasattr(model, "value_head") and model.value_head is not None

    def _default_value_fn(self, model: Any, token_ids: list[int]) -> float:
        """Default value function: use value head or logit confidence.

        Returns a scalar in [-1, 1] indicating estimated outcome quality.
        """
        if self.has_value_head:
            # Use the trained value head
            inp = torch.tensor([token_ids], device=self.device)
            with torch.no_grad():
                _, _, v = model(inp, inp)
            return float(v.item())
        else:
            # Fallback: softmax confidence of the last token
            inp = torch.tensor([token_ids], device=self.device)
            with torch.no_grad():
                logits, _, _ = model(inp, inp)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            top_p, _ = probs.topk(1)
            # Normalize confidence to [-1, 1]: confidence 0→-1, confidence 1→+1
            confidence = float(top_p[0])
            return confidence * 2.0 - 1.0

    @torch.no_grad()
    def _get_logits(self, token_ids: list[int]) -> torch.Tensor:
        """Get logits for the next token given the current prefix."""
        inp = torch.tensor([token_ids], device=self.device)
        logits, _, _ = self.model(inp, inp)
        return logits[0, -1]  # (vocab_size,)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse to a leaf using PUCT."""
        while not node.is_leaf:
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = child.ucb_score(self.c_puct, node.visit_count)
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    def _expand(self, node: MCTSNode, logits: torch.Tensor) -> None:
        """Expansion phase: create child nodes for top-k tokens."""
        if node.is_terminal:
            return

        # Apply temperature scaling
        if self.temperature > 0:
            logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)

        # Get top-k tokens (filter out special tokens)
        special_ids = {
            self.tokenizer.pad_id if hasattr(self.tokenizer, "pad_id") else 0,
        }
        top_probs, top_tokens = probs.topk(min(self.top_k, len(probs)))
        for prob, token in zip(top_probs, top_tokens):
            tid = int(token)
            if tid in special_ids:
                continue
            log_prob = math.log(max(float(prob), 1e-10))
            node.children[tid] = MCTSNode(
                token_id=tid,
                parent=node,
                prior_log_prob=log_prob,
                depth=node.depth + 1,
                is_terminal=tid == self.tokenizer.eos_id
                if hasattr(self.tokenizer, "eos_id")
                else False,
            )

    def _simulate(self, node: MCTSNode) -> float:
        """Simulation phase: estimate value of a leaf node.

        Uses the value function (value head or confidence heuristic).
        """
        if node.is_terminal:
            return 1.0 if self._is_valid_answer_node(node) else -1.0

        # Build token sequence
        tokens = self._node_path_tokens(node)
        return self.value_fn(self.model, tokens)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation phase: update visit counts and values up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def _node_path_tokens(self, node: MCTSNode) -> list[int]:
        """Collect token IDs along path from root to this node."""
        tokens = []
        current = node
        while current is not None:
            if current.token_id != 0 or current.parent is None:  # skip root
                tokens.append(current.token_id)
            current = current.parent
        tokens.reverse()
        return tokens

    def _is_valid_answer_node(self, node: MCTSNode) -> bool:
        """Check if terminal node represents a plausible answer (has '=' in path)."""
        tokens = self._node_path_tokens(node)
        text = self.tokenizer.decode(tokens) if hasattr(self.tokenizer, "decode") else ""
        return "=" in text

    def _most_visited_path(self, root: MCTSNode) -> list[int]:
        """Follow the most-visited child at each level."""
        tokens = []
        node = root
        while node.children:
            best = max(node.children.values(), key=lambda c: c.visit_count)
            tokens.append(best.token_id)
            node = best
            if node.is_terminal:
                break
        return tokens

    @torch.no_grad()
    def generate(self, prompt: str) -> tuple[str, dict]:
        """Run MCTS inference to generate an answer.

        Args:
            prompt: Input expression (e.g. '12+34').

        Returns:
            Tuple of (decoded_answer, stats_dict).
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt)
        if hasattr(self.tokenizer, "bos_id"):
            prompt_ids = [self.tokenizer.bos_id] + prompt_ids
        prompt_ids = prompt_ids[: self.max_new_tokens]

        # Get logits for the full prompt (to know where to start generating)
        root_logits = self._get_logits(prompt_ids)

        # Root node = prompt prefix (no action token, just a container)
        root = MCTSNode(token_id=0, depth=0)
        self._expand(root, root_logits)

        # ── MCTS Loop ────────────────────────────────────────────────
        for sim in range(self.num_simulations):
            # 1. Select
            leaf = self._select(root)

            # 2. Expand (if not terminal)
            if not leaf.is_terminal:
                tokens = prompt_ids + self._node_path_tokens(leaf)[:-1]  # exclude leaf itself
                # If we don't have children yet, get logits and expand
                if leaf.is_leaf:
                    logits = self._get_logits(tokens)
                    self._expand(leaf, logits)

            # 3. Simulate (estimate value)
            value = self._simulate(leaf)

            # 4. Backpropagate
            self._backpropagate(leaf, value)

        # ── Extract best path ────────────────────────────────────────
        gen_tokens = self._most_visited_path(root)

        # Truncate at EOS if present
        if hasattr(self.tokenizer, "eos_id"):
            eos_id = self.tokenizer.eos_id
            if eos_id in gen_tokens:
                gen_tokens = gen_tokens[: gen_tokens.index(eos_id)]

        # Decode
        full_text = self.tokenizer.decode(prompt_ids + gen_tokens)

        # Extract answer
        if "=" in full_text:
            answer = full_text.split("=")[-1].strip()
            answer = "".join(c for c in answer if c.isdigit() or c == "-")
        else:
            answer = full_text

        stats = {
            "num_simulations": self.num_simulations,
            "tree_size": root.visit_count,
            "total_nodes": self._count_nodes(root),
            "has_value_head": self.has_value_head,
        }

        return answer, stats

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in the tree."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


def mcts_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_simulations: int = 16,
    top_k: int = 10,
    temperature: float = 0.5,
) -> str:
    """Convenience function — run MCTS inference on a math problem.

    Args:
        model: MathTransformer instance.
        tokenizer: MathTokenizer instance.
        prompt: Math expression like "12+34".
        num_simulations: MCTS rollouts (default 16).
        top_k: Tokens to consider at each step (default 10).
        temperature: Softmax temperature (default 0.5).

    Returns:
        Predicted answer string.
    """
    mcts = MCTSInference(
        model=model,
        tokenizer=tokenizer,
        num_simulations=num_simulations,
        top_k=top_k,
        temperature=temperature,
    )
    answer, _ = mcts.generate(prompt)
    return answer
