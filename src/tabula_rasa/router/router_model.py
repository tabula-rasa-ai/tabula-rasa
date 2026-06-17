"""RouterModel — tiny transformer that routes prompts to specialists via intent tags.

Instead of generating human text (which requires learning grammar, syntax,
vocabulary, and world knowledge simultaneously), the router outputs structured
intent tags. This reduces learning complexity by ~99%, allowing a tiny model
trained from scratch to master deep semantic nuance.

Architecture:
  - BPE tokenizer → Embedding → 2 transformer blocks → Mean pool → Classifier
  - Phase 1: softmax over 4 intents (MATH, CODE, MEMORY, CHAT)
  - Phase 3: sigmoid multi-label (allows "80% MATH + 60% MEMORY")

Intent labels:
  0 = MATH    (calculations, arithmetic, equations, numbers)
  1 = CODE    (programming, algorithms, loops, functions)
  2 = MEMORY  (remember/remind me, store facts, recall)
  3 = CHAT    (greetings, opinions, casual conversation, questions)
"""

from __future__ import annotations

# Reuse existing transformer blocks from model.py
import sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src.tabula_rasa.model import TransformerBlock, _make_norm

# ─── Intent definitions ────────────────────────────────────────────

INTENT_NAMES = ["MATH", "CODE", "MEMORY", "CHAT"]
NUM_INTENTS = len(INTENT_NAMES)


def intents_to_vector(intents: list[str], multi_label: bool = False) -> torch.Tensor:
    """Convert intent strings to a target vector.

    Softmax mode: returns the index of the primary intent (scalar).
    Sigmoid mode: returns a multi-hot vector (num_intents,).

    Args:
        intents: List of intent names (e.g. ['MATH'] or ['MATH', 'MEMORY'])
        multi_label: If True, return multi-hot vector for sigmoid training

    Returns:
        Long tensor (scalar index) for softmax, or float tensor (num_intents,) for sigmoid
    """
    if multi_label:
        vec = torch.zeros(NUM_INTENTS)
        for name in intents:
            if name in INTENT_NAMES:
                vec[INTENT_NAMES.index(name)] = 1.0
        return vec
    else:
        return torch.tensor(INTENT_NAMES.index(intents[0]) if intents else 3)


def vector_to_intents(
    logits: torch.Tensor, multi_label: bool = False, threshold: float = 0.5
) -> dict:
    """Convert model output logits to structured intent dict.

    Args:
        logits: Raw logits tensor (num_intents,)
        multi_label: If True, use sigmoid activation (multi-label)
        threshold: Confidence threshold for sigmoid mode

    Returns:
        dict with 'intents' (list), 'confidence' (float or dict),
        and 'primary' (str) fields.
    """
    if multi_label:
        probs = torch.sigmoid(logits)
        masks = (probs >= threshold).tolist()
        intents = [INTENT_NAMES[i] for i, m in enumerate(masks) if m]
        confidence = {INTENT_NAMES[i]: round(p, 3) for i, p in enumerate(probs.tolist())}
        primary = INTENT_NAMES[probs.argmax().item()] if intents else "CHAT"
        return {
            "intents": intents or ["CHAT"],
            "confidence": confidence,
            "primary": primary,
            "multi_label": True,
        }
    else:
        probs = F.softmax(logits, dim=-1)
        max_prob, pred_idx = probs.max(dim=-1)
        intent = INTENT_NAMES[pred_idx.item()]
        return {
            "intents": [intent],
            "confidence": round(max_prob.item(), 3),
            "primary": intent,
            "multi_label": False,
        }


# ─── Lightweight config adapter ─────────────────────────────────────


class RouterConfig:
    """Pulls router settings from main Config — no new config file needed."""

    def __init__(self, config: Any) -> None:
        self.vocab_size = getattr(config, "router_vocab_size", 768)
        self.d_model = getattr(config, "router_d_model", 128)
        self.n_layers = getattr(config, "router_n_layers", 2)
        self.n_heads = getattr(config, "router_n_heads", 4)
        self.d_ff = getattr(config, "router_d_ff", 512)
        self.max_seq_len = getattr(config, "router_max_seq_len", 64)
        self.dropout = getattr(config, "router_dropout", 0.1)
        self.num_intents = getattr(config, "router_num_intents", 4)
        self.multi_label = getattr(config, "router_multi_label", False)
        self.weight_init = getattr(config, "weight_init", "normal")
        self.init_std = getattr(config, "init_std", 0.02)
        self.norm_type = getattr(config, "norm_type", "rmsnorm")
        self.pos_encoding = getattr(config, "pos_encoding", "rope")
        self.activation = getattr(config, "activation", "swiglu")
        self.router_confidence_threshold = getattr(config, "router_confidence_threshold", 0.6)


# ─── RouterModel — fast transformer classifier ─────────────────────


class RouterModel(nn.Module):
    """Tiny transformer that classifies prompts into intent categories.

    Architecture: Embed → 2 TransformerBlocks → Mean Pool → Linear classifier.
    Mean pooling over all tokens (non-causal) gives better classification than
    last-token pooling, especially for short prompts.

    Forward returns logits only — no autoregressive generation needed.
    """

    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])

        self.norm = _make_norm(config.d_model, config.norm_type)

        # Classification head
        self.classifier = nn.Linear(config.d_model, config.num_intents, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with normal distribution."""
        init_std = self.config.init_std
        for module in [self.token_embedding, self.classifier]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    torch.nn.init.normal_(param, mean=0.0, std=init_std)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, None]:
        """Forward pass — classify prompt into intent(s).

        Uses mean pooling over all non-padding tokens for robust
        classification regardless of prompt length.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional (batch, seq_len) — 1=keep, 0=ignore
            targets: Optional targets for loss computation (EWC compatibility)

        Returns:
            Logits tensor (batch, num_intents), or (logits, loss, None) when targets provided
            for compatibility with OnlineEWC.compute_fisher
        """
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        for layer in self.layers:
            x, _ = layer(x, mask=None)

        x = self.norm(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        logits = self.classifier(x)

        if targets is not None:
            loss = self.compute_loss(logits, targets)
            return logits, loss, None

        return logits

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss.

        Phase 1 (softmax): CrossEntropyLoss (targets are class indices)
        Phase 3 (sigmoid): BCEWithLogitsLoss (targets are multi-hot vectors)

        Args:
            logits: (batch, num_intents)
            targets: (batch,) for softmax or (batch, num_intents) for sigmoid

        Returns:
            Scalar loss tensor
        """
        if self.config.multi_label:
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        else:
            return F.cross_entropy(logits, targets.long())

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Run inference and return structured intent predictions.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: Optional (batch, seq_len)
            threshold: Confidence threshold for sigmoid mode

        Returns:
            List of dicts with 'intents', 'confidence', 'primary'
        """
        logits = self.forward(input_ids, attention_mask)
        return [
            vector_to_intents(logits[i], multi_label=self.config.multi_label, threshold=threshold)
            for i in range(logits.size(0))
        ]

    def get_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute confidence scores.

        Softmax: max probability per sample.
        Sigmoid: max probability per sample.

        Args:
            logits: (batch, num_intents)

        Returns:
            (batch,) confidence in [0, 1]
        """
        if self.config.multi_label:
            return torch.sigmoid(logits).max(dim=-1).values
        else:
            return F.softmax(logits, dim=-1).max(dim=-1).values

    def is_low_confidence(
        self, logits: torch.Tensor, threshold: Optional[float] = None
    ) -> torch.Tensor:
        """Check which samples have confidence below threshold (high surprise).

        Low-confidence prompts should be routed to hippocampus for storage
        and later consolidation during sleep.

        Args:
            logits: (batch, num_intents)
            threshold: Confidence threshold (defaults to config value)

        Returns:
            Boolean mask (batch,) — True = low confidence → store in hippocampus
        """
        threshold = threshold or self.config.router_confidence_threshold
        return self.get_confidence(logits) < threshold

    # ─── Save / Load ───────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "d_ff": self.config.d_ff,
                "max_seq_len": self.config.max_seq_len,
                "dropout": self.config.dropout,
                "num_intents": self.config.num_intents,
                "multi_label": self.config.multi_label,
            },
        }
        torch.save(data, path)

    @classmethod
    def load(
        cls, path: str | Path, config: Optional[RouterConfig] = None
    ) -> "RouterModel":
        """Load model weights from checkpoint.

        Args:
            path: Path to checkpoint (.pt file)
            config: RouterConfig to use. If None, reconstructed from saved config.

        Returns:
            Loaded RouterModel
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved_cfg = checkpoint.get("config", {})

        if config is None:
            from src.tabula_rasa.config import Config as MainConfig
            base_cfg = MainConfig()
            for k, v in saved_cfg.items():
                setattr(base_cfg, f"router_{k}", v)
            config = RouterConfig(base_cfg)
        else:
            # Override vocab_size from checkpoint to ensure match
            if "vocab_size" in saved_cfg:
                config.vocab_size = saved_cfg["vocab_size"]

        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        return model


def count_router_params(model: RouterModel) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.tabula_rasa.config import Config

    cfg = Config()
    cfg.router_n_layers = 2
    rcfg = RouterConfig(cfg)
    model = RouterModel(rcfg)

    batch = torch.randint(0, rcfg.vocab_size, (4, rcfg.max_seq_len))
    logits = model(batch)

    targets = torch.randint(0, rcfg.num_intents, (4,))
    loss = model.compute_loss(logits, targets)
    preds = model.predict(batch)

    print(f"RouterModel — {count_router_params(model):,} params")
    print(f"  Input:  {batch.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss:   {loss.item():.4f}")
    print(f"  Preds:  {preds}")
    print(f"  Conf:   {model.get_confidence(logits)}")
