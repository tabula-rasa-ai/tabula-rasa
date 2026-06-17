"""Router-Hippocampus Bridge — confidence-based surprise detection and storage.

When the router encounters a prompt it's unsure about (confidence < threshold),
it stores the experience in the hippocampus for later consolidation during sleep.

This is the "High Surprise Routing" mechanism:
  1. User sends a prompt
  2. Router predicts intents with confidence scores
  3. If confidence < threshold (default 60%), flag as "High Surprise"
  4. Store prompt + predicted intents + confidence in hippocampus
  5. During sleep cycle, these are replayed with EWC to learn the nuance
"""

from __future__ import annotations

from typing import Optional

import torch

from tabula_rasa.router.router_model import INTENT_NAMES, RouterModel, vector_to_intents


def route_with_hippocampus(
    model: RouterModel,
    input_ids: torch.Tensor,
    prompt_text: str,
    attention_mask: Optional[torch.Tensor] = None,
    confidence_threshold: Optional[float] = None,
    store_low_confidence: bool = True,
) -> dict:
    """Route a prompt through the model and optionally store low-confidence results.

    This is the main entry point for routing with hippocampus integration.

    Args:
        model: Trained RouterModel
        input_ids: Tokenized prompt (1, seq_len)
        prompt_text: Original text for hippocampus storage
        attention_mask: Optional attention mask
        confidence_threshold: Override for confidence threshold. Defaults to model.config.router_confidence_threshold
        store_low_confidence: If True, store low-confidence predictions in hippocampus

    Returns:
        dict with routing result + hippocampus metadata
    """
    model.eval()
    threshold = confidence_threshold or getattr(
        model.config, "router_confidence_threshold", 0.6
    )

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        result = vector_to_intents(
            logits[0],
            multi_label=model.config.multi_label,
            threshold=0.5,
        )
        confidence = result["confidence"]
        if model.config.multi_label:
            # For multi-label, use the max sigmoid probability as overall confidence
            max_conf = max(confidence.values()) if isinstance(confidence, dict) else confidence
        else:
            max_conf = confidence

    # Check if this is a low-confidence prediction (high surprise)
    is_low_confidence = max_conf < threshold if isinstance(max_conf, (int, float)) else False

    hippocampus_info = {
        "stored": False,
        "reason": None,
        "confidence": max_conf,
        "threshold": threshold,
    }

    if is_low_confidence and store_low_confidence:
        # Store in hippocampus
        try:
            from tabula_rasa.memory.hippocampus import store_experience

            prediction_error = 1.0 - max_conf  # Surprise = 1 - confidence
            store_experience(
                input_text=prompt_text,
                prediction_error=prediction_error,
                output_text=str(result),
                skill="router",
                is_correction=False,
                tier="active",
            )
            hippocampus_info["stored"] = True
            hippocampus_info["reason"] = f"low_confidence ({max_conf:.3f} < {threshold})"
            hippocampus_info["prediction_error"] = prediction_error
        except Exception as e:
            hippocampus_info["stored"] = False
            hippocampus_info["reason"] = f"hippocampus_error: {e}"

    result["hippocampus"] = hippocampus_info
    result["is_low_confidence"] = is_low_confidence

    return result


@torch.no_grad()
def batch_route_with_hippocampus(
    model: RouterModel,
    input_ids: torch.Tensor,
    prompt_texts: list[str],
    attention_mask: Optional[torch.Tensor] = None,
    confidence_threshold: Optional[float] = None,
    store_low_confidence: bool = True,
) -> list[dict]:
    """Route a batch of prompts with hippocampus integration.

    Args:
        model: Trained RouterModel
        input_ids: (batch, seq_len) token IDs
        prompt_texts: Original text for each prompt
        attention_mask: Optional (batch, seq_len)
        confidence_threshold: Override for confidence threshold
        store_low_confidence: If True, store low-confidence predictions

    Returns:
        List of routing result dicts
    """
    results = []
    for i in range(len(prompt_texts)):
        single_ids = input_ids[i : i + 1]
        single_mask = attention_mask[i : i + 1] if attention_mask is not None else None
        result = route_with_hippocampus(
            model, single_ids, prompt_texts[i],
            attention_mask=single_mask,
            confidence_threshold=confidence_threshold,
            store_low_confidence=store_low_confidence,
        )
        results.append(result)
    return results


def get_router_surprise_experiences(
    limit: int = 100,
    min_error: float = 0.4,
) -> list[dict]:
    """Get router experiences with high prediction error from hippocampus.

    These are the "High Surprise" prompts that the router struggled with,
    ready for consolidation during the sleep cycle.

    Args:
        limit: Max experiences to return
        min_error: Minimum prediction error (1 - confidence) to filter

    Returns:
        List of experience dicts with 'input_text', 'output_text', 'prediction_error'
    """
    try:
        from tabula_rasa.memory.hippocampus import get_unconsolidated

        experiences = get_unconsolidated(limit=limit, tier="active", include_gradients=False)
        # Filter for router skill and high error
        router_exps = [
            e for e in experiences
            if e.get("skill") == "router" and e.get("prediction_error", 0) >= min_error
        ]
        return router_exps[:limit]
    except Exception:
        return []


def get_router_consolidation_data(
    model: RouterModel,
    tokenizer: any,
    limit: int = 100,
    min_error: float = 0.4,
    max_seq_len: int = 64,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], list[dict]]:
    """Get and tokenize surprise experiences for router consolidation.

    Returns:
        (input_ids, targets, metadata) tuple or (None, None, []) if nothing to consolidate.
        targets are the INTENT class indices for training.
    """
    experiences = get_router_surprise_experiences(limit=limit, min_error=min_error)
    if not experiences:
        return None, None, []

    input_ids_list = []
    targets_list = []
    metadata = []

    for exp in experiences:
        try:
            ids = tokenizer.encode(exp["input_text"], add_special_tokens=True)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            pad_len = max_seq_len - len(ids)
            padded_ids = ids + [tokenizer.pad_id] * pad_len
            input_ids_list.append(padded_ids)

            # Parse stored intent prediction to get target label
            # For consolidation we use the stored prediction as the target
            # (it was stored as the best guess at the time)
            try:
                import json
                output = exp.get("output_text", "{'intents': ['CHAT']}")
                output_data = json.loads(output.replace("'", '"'))
                primary = output_data.get("primary", "CHAT")
                target = INTENT_NAMES.index(primary)
            except Exception:
                target = 3  # Default to CHAT

            targets_list.append(target)
            metadata.append(exp)
        except Exception:
            continue

    if not input_ids_list:
        return None, None, []

    return (
        torch.tensor(input_ids_list, dtype=torch.long),
        torch.tensor(targets_list, dtype=torch.long),
        metadata,
    )
