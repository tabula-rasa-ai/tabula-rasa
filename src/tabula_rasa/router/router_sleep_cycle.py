"""Router Sleep Cycle — consolidate high-surprise experiences into long-term memory.

During sleep:
1. Sample high-surprise router experiences from hippocampus
2. Compute Fisher information on the surprise data
3. Micro-train router with EWC penalty to learn nuance without forgetting
4. Mark experiences as consolidated

This is how the router achieves "General Language Routing" from scratch:
it encounters nuance in the wild, stores it in the Hippocampus,
and learns it one edge-case at a time while it "sleeps."
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.tabula_rasa.config import Config
from src.tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.router.router_model import RouterModel, RouterConfig, count_router_params, INTENT_NAMES
from tabula_rasa.router.router_hippocampus import (
    get_router_consolidation_data,
    get_router_surprise_experiences,
)
from tabula_rasa.cl.online_ewc import OnlineEWC


def consolidate_router_sleep(
    model_path: Path,
    tokenizer_path: Path,
    ewc_path: Optional[Path] = None,
    num_samples: int = 100,
    epochs: int = 5,
    lambda_ewc: float = 1000.0,
    gamma: float = 0.9,
    learning_rate: float = 5e-4,
    batch_size: int = 16,
    max_seq_len: int = 64,
    min_error: float = 0.4,
) -> dict:
    """Run one sleep cycle for the router.

    Samples high-surprise router experiences from hippocampus and
    micro-trains the router with EWC to consolidate nuance.

    Args:
        model_path: Path to router checkpoint
        tokenizer_path: Path to BPE tokenizer JSON
        ewc_path: Path to existing EWC state (optional)
        num_samples: Max surprise experiences to consume
        epochs: Micro-training epochs
        lambda_ewc: EWC penalty strength
        gamma: Fisher merge decay rate
        learning_rate: Training learning rate
        batch_size: Micro-training batch size
        max_seq_len: Max token sequence length
        min_error: Minimum prediction error to filter

    Returns:
        Dict of sleep cycle results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "experiences_found": 0,
        "experiences_used": 0,
        "consolidated": 0,
        "train_loss": None,
        "train_acc": None,
        "ewc_penalty": None,
        "hippocampus_stats": {},
    }

    print(f"\n{'='*60}")
    print(f"  ROUTER SLEEP CYCLE — Nuance Consolidation")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── Load hippocampus stats ──────────────────────────────────
    try:
        from tabula_rasa.memory.hippocampus import get_stats
        stats = get_stats()
        results["hippocampus_stats"] = stats
        print(f"  Hippocampus: {stats['total_experiences']} total, "
              f"{stats['unconsolidated']} unconsolidated")
    except Exception as e:
        print(f"  Hippocampus stats: {e}")

    # ── Check for surprise experiences ──────────────────────────
    raw_exps = get_router_surprise_experiences(
        limit=num_samples, min_error=min_error
    )
    results["experiences_found"] = len(raw_exps)
    if not raw_exps:
        print("  No high-surprise router experiences. Nothing to consolidate.")
        return results
    print(f"  Found {len(raw_exps)} high-surprise router experiences "
          f"(error >= {min_error})")

    # ── Load model and tokenizer ────────────────────────────────
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return results

    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer.max_seq_len = max_seq_len
    print(f"  Tokenizer: {tokenizer.vocab_size} tokens")

    model = RouterModel.load(model_path)
    model.to(device)
    model.train()
    params = count_router_params(model)
    print(f"  Router: {params:,} params")

    # ── Tokenize surprise experiences ───────────────────────────
    input_ids, targets, metadata = get_router_consolidation_data(
        model, tokenizer,
        limit=num_samples,
        min_error=min_error,
        max_seq_len=max_seq_len,
    )
    if input_ids is None:
        print("  No valid experiences to tokenize.")
        return results

    input_ids = input_ids.to(device)
    targets = targets.to(device)
    results["experiences_used"] = len(metadata)
    print(f"  Tokenized {len(metadata)} experiences for training")

    # ── Initialize EWC ──────────────────────────────────────────
    ewc = OnlineEWC(model, gamma=gamma, use_archive=True)
    if ewc_path and ewc_path.exists():
        ewc.load(ewc_path)
        print(f"  [EWC] Loaded {ewc.task_count} prior tasks")

    # ── Phase 1: Compute Fisher on surprise data ────────────────
    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("  Phase 1: Computing Fisher Information Matrix...")
    new_fisher = ewc.compute_fisher(dataloader)
    ewc.merge_fisher(new_fisher)
    ewc.save_anchor_weights()
    print(f"  Fisher computed across {len(new_fisher)} parameter groups")

    # ── Phase 2: Train with EWC penalty ─────────────────────────
    print(f"  Phase 2: Training with EWC penalty (λ={lambda_ewc})...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epoch_losses = []
    epoch_accs = []
    epoch_penalties = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_penalty = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for batch_ids, batch_targets in dataloader:
            batch_ids = batch_ids.to(device)
            batch_targets = batch_targets.to(device)

            # Forward
            logits = model(batch_ids)
            cls_loss = model.compute_loss(logits, batch_targets)

            # EWC penalty
            ewc_penalty = ewc.compute_ewc_penalty(lambda_ewc=lambda_ewc)

            total_loss_val = cls_loss + ewc_penalty

            # Backward
            optimizer.zero_grad()
            total_loss_val.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += cls_loss.item()
            total_penalty += ewc_penalty.item()
            _, preds = logits.max(1)
            correct += (preds == batch_targets).sum().item()
            total += batch_targets.size(0)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_penalty = total_penalty / max(num_batches, 1)
        avg_acc = correct / max(total, 1)

        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
        epoch_penalties.append(avg_penalty)

        print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  "
              f"acc={avg_acc:.3f}  ewc_penalty={avg_penalty:.4f}")

    results["train_loss"] = epoch_losses[-1] if epoch_losses else None
    results["train_acc"] = epoch_accs[-1] if epoch_accs else None
    results["ewc_penalty"] = epoch_penalties[-1] if epoch_penalties else None

    # ── Phase 3: Save consolidated model ────────────────────────
    print("  Phase 3: Saving consolidated model and EWC state...")
    model.save(model_path)
    ewc.save(ewc_path or (model_path.parent / "ewc_fisher.pt"))

    # ── Mark experiences as consolidated ────────────────────────
    exp_ids = [meta.get("id") for meta in metadata if meta.get("id") is not None]
    if exp_ids:
        from tabula_rasa.memory.hippocampus import mark_consolidated
        mark_consolidated(exp_ids)
        results["consolidated"] = len(exp_ids)
        print(f"  Consolidated {len(exp_ids)} experiences")

    # ── Synaptic decay ──────────────────────────────────────────
    try:
        from tabula_rasa.memory.hippocampus import decay_memories
        decay_result = decay_memories(
            min_access=3, max_age_days=7,
            demote_working=True, cleanup_active=True,
        )
        print(f"  Synaptic decay: {decay_result}")
    except Exception as e:
        print(f"  Synaptic decay skipped: {e}")

    print(f"  Router sleep cycle complete. Model saved to {model_path}")
    print(f"{'='*60}\n")

    return results


def run_daemon_cycle(args):
    """Run one router sleep cycle with auto-finding the latest checkpoint."""
    router_dir = Path(args.router_dir)
    model_path = router_dir / "best.pt"
    if not model_path.exists():
        model_path = router_dir / "final.pt"

    if not model_path.exists():
        print(f"  No router checkpoint found in {router_dir}. Skipping cycle.")
        return

    tokenizer_path = router_dir / "tokenizer.json"
    ewc_path = router_dir / "ewc_fisher.pt"

    consolidate_router_sleep(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        ewc_path=ewc_path if ewc_path.exists() else None,
        num_samples=args.num_samples,
        epochs=args.epochs,
        lambda_ewc=args.lambda_ewc,
        gamma=args.gamma,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        min_error=args.min_error,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Router sleep cycle — consolidate nuance via EWC"
    )
    parser.add_argument("--router-dir", type=str, default="checkpoints/router",
                        help="Router checkpoint directory")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Max surprise experiences to consume")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Micro-training epochs")
    parser.add_argument("--lambda-ewc", type=float, default=1000.0,
                        help="EWC penalty strength")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Fisher merge decay rate")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Micro-training learning rate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--min-error", type=float, default=0.4,
                        help="Minimum prediction error to filter")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously in daemon mode")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between cycles in daemon mode")
    args = parser.parse_args()

    if args.daemon:
        print(f"Router sleep daemon starting — interval: {args.interval}s")
        cycle = 0
        while True:
            cycle += 1
            print(f"\n--- Cycle #{cycle} ---")
            run_daemon_cycle(args)
            print(f"\nNext cycle in {args.interval}s...")
            time.sleep(args.interval)
    else:
        run_daemon_cycle(args)
