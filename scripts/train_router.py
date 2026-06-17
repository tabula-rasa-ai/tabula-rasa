"""train_router.py — Train the neural semantic parser router from scratch.

Curriculum:
  Phase 1 (Toddler): Learn 4 basic intents (MATH, CODE, MEMORY, CHAT)
  Phase 2 (Nuance):  Freeze with EWC, train on ambiguous edge cases
  Phase 3 (Compound): Swap to sigmoid multi-label, train on multi-intent

Usage:
    python3 train_router.py --dataset-size 5000 --epochs 50
    python3 train_router.py --phase 2 --ewc-checkpoint checkpoints/router/best.pt
    python3 train_router.py --phase 3 --multi-label

The router uses a BPE tokenizer (learning word-piece merges from the
dataset text) so it can read English words without needing a huge vocab.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import json
import math
import random
from typing import Optional

import torch
from egefalos.router_dataset import (
    generate_multi_label_dataset,
    generate_router_dataset,
    load_dataset,
    print_dataset_stats,
    save_dataset,
)
from egefalos.router_model import (
    INTENT_NAMES,
    RouterConfig,
    RouterModel,
    count_router_params,
)
from torch.utils.data import DataLoader, TensorDataset

from src.tabula_rasa.bpe_tokenizer import BPETokenizer
from src.tabula_rasa.config import Config

# ─── Paths ─────────────────────────────────────────────────────────

ROUTER_DIR = Path("checkpoints/router")
DATA_DIR = Path("data")


# ─── Tokenizer setup ────────────────────────────────────────────────


def build_router_tokenizer(
    texts: list[str],
    num_merges: int = 50,
    min_frequency: int = 2,
) -> BPETokenizer:
    """Build a BPE tokenizer for the router from dataset texts.

    Uses a random subset of texts for BPE learning to keep it fast on CPU.

    Args:
        texts: List of training prompt strings
        num_merges: BPE merge operations to learn
        min_frequency: Minimum pair frequency for merging

    Returns:
        Trained BPETokenizer
    """
    import random
    tok = BPETokenizer()
    print(f"  [Tokenizer] Base vocab: {tok.vocab_size} tokens")

    # Use a subset for BPE learning — 1000 texts is enough for good merges
    sample_size = min(1000, len(texts))
    sample_texts = random.sample(texts, sample_size) if len(texts) > sample_size else texts
    print(f"  [Tokenizer] Learning BPE from {len(sample_texts)} texts (sampled from {len(texts)})")

    learned = tok.learn_bpe_from_texts(
        sample_texts, num_merges=num_merges,
        min_frequency=min_frequency, verbose=True
    )
    print(f"  [Tokenizer] Learned {learned} BPE merges — vocab now {tok.vocab_size}")

    # Set max sequence length for router
    tok.max_seq_len = 64
    return tok


def tokenize_dataset(
    dataset: list[dict],
    tokenizer: BPETokenizer,
    max_seq_len: int = 64,
    multi_label: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize the full dataset into padded tensors.

    Args:
        dataset: List of dicts with 'text' and 'label' or 'multi_hot'
        tokenizer: BPETokenizer instance
        max_seq_len: Max token sequence length
        multi_label: If True, targets are multi-hot vectors

    Returns:
        (input_ids, targets) tensors
    """
    input_ids_list = []
    targets_list = []

    for item in dataset:
        text = item["text"]
        ids = tokenizer.encode(text, add_special_tokens=True)

        # Truncate
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        # Pad
        pad_len = max_seq_len - len(ids)
        padded_ids = ids + [tokenizer.pad_id] * pad_len
        input_ids_list.append(padded_ids)

        if multi_label:
            targets_list.append(item["multi_hot"])
        else:
            targets_list.append(item["label"])

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    if multi_label:
        targets_tensor = torch.tensor(targets_list, dtype=torch.float)
    else:
        targets_tensor = torch.tensor(targets_list, dtype=torch.long)

    return input_ids_tensor, targets_tensor


# ─── Training ──────────────────────────────────────────────────────


def train_router(
    model: RouterModel,
    train_ids: torch.Tensor,
    train_targets: torch.Tensor,
    eval_ids: Optional[torch.Tensor] = None,
    eval_targets: Optional[torch.Tensor] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    label_smoothing: float = 0.1,
    curriculum_phase: int = 1,
    output_dir: Path = ROUTER_DIR,
    log_every: int = 10,
    save_every: int = 50,
    ewc: Optional["OnlineEWC"] = None,
    lambda_ewc: float = 1000.0,
) -> dict:
    """Train the router model.

    Args:
        model: RouterModel instance
        train_ids: (N, seq_len) token IDs
        train_targets: (N,) scalar labels or (N, num_intents) multi-hot
        eval_ids: Optional eval set
        eval_targets: Optional eval targets
        epochs: Number of training epochs
        batch_size: Minibatch size
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        warmup_steps: Linear warmup steps
        label_smoothing: Label smoothing for cross-entropy
        curriculum_phase: Which phase (1, 2, or 3)
        output_dir: Model save directory
        log_every: Log metrics every N steps
        save_every: Save checkpoint every N epochs
        ewc: Optional OnlineEWC instance for Phase 2+
        lambda_ewc: EWC penalty strength

    Returns:
        Training history dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    multi_label = model.config.multi_label
    num_intents = model.config.num_intents

    # Create dataset and dataloader
    dataset = TensorDataset(train_ids, train_targets)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler with linear warmup + cosine decay
    total_steps = epochs * len(dataloader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Tracking
    history = {
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": [],
        "confidence": [],
        "low_conf_ratio": [],
        "learning_rate": [],
        "epochs_completed": 0,
        "total_steps": 0,
        "curriculum_phase": curriculum_phase,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_conf = 0.0
        epoch_low_conf = 0
        num_batches = 0

        for batch_ids, batch_targets in dataloader:
            batch_ids = batch_ids.to(device)
            batch_targets = batch_targets.to(device)

            # Forward
            logits = model(batch_ids)

            # Classification loss
            cls_loss = model.compute_loss(logits, batch_targets)

            # EWC penalty (Phase 2+)
            ewc_penalty = torch.tensor(0.0, device=device)
            if ewc is not None:
                ewc_penalty = ewc.compute_ewc_penalty(lambda_ewc=lambda_ewc)

            total_loss = cls_loss + ewc_penalty

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            batch_size_actual = batch_ids.size(0)
            epoch_loss += cls_loss.item() * batch_size_actual

            if multi_label:
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc = (preds == batch_targets).float().mean().item()
            else:
                _, preds = logits.max(dim=1)
                acc = (preds == batch_targets).float().mean().item()

            epoch_acc += acc * batch_size_actual
            conf = model.get_confidence(logits)
            epoch_conf += conf.sum().item()
            epoch_low_conf += (conf < model.config.router_confidence_threshold).sum().item()
            num_batches += 1
            global_step += 1

        # Epoch averages
        n_samples = len(dataset)
        avg_loss = epoch_loss / n_samples
        avg_acc = epoch_acc / n_samples
        avg_conf = epoch_conf / n_samples
        low_conf_ratio = epoch_low_conf / n_samples
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)
        history["confidence"].append(avg_conf)
        history["low_conf_ratio"].append(low_conf_ratio)
        history["learning_rate"].append(current_lr)

        # Eval
        eval_loss = avg_loss
        eval_acc = avg_acc
        if eval_ids is not None and eval_targets is not None:
            eval_loss, eval_acc = evaluate_router(
                model, eval_ids, eval_targets, batch_size
            )
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            ewc_str = f" | EWC λ={lambda_ewc}" if ewc else ""
            print(
                f"  Epoch {(epoch+1):3d}/{epochs} "
                f"| loss={avg_loss:.4f} | acc={avg_acc:.3f} "
                f"| eval_acc={eval_acc:.3f} "
                f"| conf={avg_conf:.3f} | low%={low_conf_ratio:.3f}"
                f" | lr={current_lr:.2e}{ewc_str}"
            )

        # Save best
        if eval_acc > best_acc:
            best_acc = eval_acc
            model.save(output_dir / "best.pt")
            if (epoch + 1) % 5 == 0:
                print(f"    [*] New best: {best_acc:.3f}")

        # Periodic save
        if (epoch + 1) % save_every == 0:
            model.save(output_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Final save
    model.save(output_dir / "final.pt")
    history["epochs_completed"] = epochs
    history["total_steps"] = global_step
    history["best_acc"] = best_acc

    print(f"\n  [*] Training complete. Best eval acc: {best_acc:.3f}")
    print(f"  [*] Model saved to {output_dir / 'final.pt'}")

    return history


@torch.no_grad()
def evaluate_router(
    model: RouterModel,
    eval_ids: torch.Tensor,
    eval_targets: torch.Tensor,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Evaluate router on a held-out set.

    Returns:
        (avg_loss, avg_accuracy)
    """
    device = next(model.parameters()).device
    model.eval()

    dataset = TensorDataset(eval_ids, eval_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    total_acc = 0.0
    n_samples = 0

    for batch_ids, batch_targets in dataloader:
        batch_ids = batch_ids.to(device)
        batch_targets = batch_targets.to(device)

        logits = model(batch_ids)
        loss = model.compute_loss(logits, batch_targets)

        batch_size_actual = batch_ids.size(0)
        total_loss += loss.item() * batch_size_actual

        if model.config.multi_label:
            preds = (torch.sigmoid(logits) >= 0.5).float()
            acc = (preds == batch_targets).float().mean().item()
        else:
            _, preds = logits.max(dim=1)
            acc = (preds == batch_targets).float().mean().item()

        total_acc += acc * batch_size_actual
        n_samples += batch_size_actual

    model.train()
    return total_loss / n_samples, total_acc / n_samples


# ─── Phase 2 EWC setup ──────────────────────────────────────────────


def setup_ewc_for_router(
    model: RouterModel,
    train_ids: torch.Tensor,
    train_targets: torch.Tensor,
    gamma: float = 0.9,
    use_archive: bool = True,
    ewc_path: Optional[Path] = None,
) -> "OnlineEWC":
    """Initialize OnlineEWC and compute Fisher on existing training data.

    Args:
        model: Already-trained RouterModel
        train_ids: Training token IDs
        train_targets: Training targets
        gamma: Fisher decay rate
        use_archive: Use Fisher Archive mode
        ewc_path: Optional path to load existing EWC state

    Returns:
        OnlineEWC instance with Fisher computed
    """
    from egefalos.online_ewc import OnlineEWC

    device = next(model.parameters()).device
    ewc = OnlineEWC(model, gamma=gamma, use_archive=use_archive)

    if ewc_path and ewc_path.exists():
        ewc.load(ewc_path)
        print(f"  [EWC] Loaded existing state: {ewc.task_count} prior tasks")
        return ewc

    # Compute Fisher on training data
    dataset = TensorDataset(train_ids, train_targets)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True,
    )

    print("  [EWC] Computing Fisher on training data...")
    fisher = ewc.compute_fisher(dataloader)
    ewc.merge_fisher(fisher)
    ewc.save_anchor_weights()
    ewc.task_count = 1
    print(f"  [EWC] Fisher computed and merged. {len(fisher)} param groups")

    return ewc


# ─── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train the neural router (semantic parser)"
    )
    parser.add_argument("--dataset-size", type=int, default=5000,
                        help="Number of synthetic prompts to generate")
    parser.add_argument("--ambiguous-ratio", type=float, default=0.1,
                        help="Ratio of ambiguous multi-intent samples")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    # Curriculum phases
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Curriculum phase (1=basic, 2=nuance+EWC, 3=multi-label)")
    parser.add_argument("--multi-label", action="store_true",
                        help="Enable multi-label sigmoid output (Phase 3)")
    parser.add_argument("--ewc-checkpoint", type=str, default=None,
                        help="Load Phase 1 checkpoint for EWC setup (Phase 2)")
    parser.add_argument("--lambda-ewc", type=float, default=1000.0,
                        help="EWC penalty strength (Phase 2+)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Fisher decay rate for online EWC")
    parser.add_argument("--load-data", type=str, default=None,
                        help="Load existing dataset JSON instead of generating")

    # Paths
    parser.add_argument("--output-dir", type=str, default="checkpoints/router")
    parser.add_argument("--data-dir", type=str, default="data")

    args = parser.parse_args()

    ROUTER_DIR = Path(args.output_dir)
    DATA_DIR = Path(args.data_dir)
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"{'='*60}")
    print(f"  ROUTER TRAINING — Phase {args.phase}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Output: {ROUTER_DIR}")
    print(f"{'='*60}\n")

    # ── 1. Dataset ──────────────────────────────────────────────
    multi_label = args.multi_label or (args.phase == 3)

    if args.load_data:
        print(f"[*] Loading dataset from {args.load_data}")
        dataset = load_dataset(args.load_data)
    elif multi_label:
        print("[*] Generating multi-label dataset (Phase 3)...")
        dataset = generate_multi_label_dataset(
            total=args.dataset_size,
            ambiguous_ratio=args.ambiguous_ratio or 0.3,
            seed=args.seed,
        )
    else:
        print("[*] Generating single-label dataset (Phase 1)...")
        dataset = generate_router_dataset(
            total=args.dataset_size,
            ambiguous_ratio=args.ambiguous_ratio or 0.1,
            seed=args.seed,
        )

    print_dataset_stats(dataset)
    save_dataset(dataset, DATA_DIR / "router_dataset.json")

    # Split train/eval (90/10)
    split = int(len(dataset) * 0.9)
    train_data = dataset[:split]
    eval_data = dataset[split:]
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    # ── 2. Config ───────────────────────────────────────────────
    cfg = Config()
    cfg.router_vocab_size = 768  # Will be updated after tokenizer
    cfg.router_multi_label = multi_label
    rcfg = RouterConfig(cfg)

    # ── 3. Tokenizer ────────────────────────────────────────────
    all_texts = [item["text"] for item in dataset]
    tokenizer = build_router_tokenizer(
        all_texts, num_merges=50, min_frequency=2,
    )
    cfg.router_vocab_size = tokenizer.vocab_size
    rcfg = RouterConfig(cfg)
    tokenizer.max_seq_len = rcfg.max_seq_len

    # Save tokenizer
    tokenizer.save(str(ROUTER_DIR / "tokenizer.json"))
    print(f"  [Tokenizer] Saved to {ROUTER_DIR / 'tokenizer.json'}")

    # ── 3. Tokenize ─────────────────────────────────────────────
    print("\n[*] Tokenizing dataset...")
    train_ids, train_targets = tokenize_dataset(
        train_data, tokenizer, max_seq_len=rcfg.max_seq_len,
        multi_label=multi_label,
    )
    eval_ids, eval_targets = tokenize_dataset(
        eval_data, tokenizer, max_seq_len=rcfg.max_seq_len,
        multi_label=multi_label,
    )
    print(f"  Train: {train_ids.shape} | Eval: {eval_ids.shape}")

    # ── 4. Model ────────────────────────────────────────────────
    model = RouterModel(rcfg)

    # Load Phase 1 checkpoint for Phase 2/3
    if args.ewc_checkpoint:
        ckpt_path = Path(args.ewc_checkpoint)
        if ckpt_path.exists():
            model = RouterModel.load(ckpt_path, rcfg)
            print(f"  [Model] Loaded checkpoint from {ckpt_path}")
        else:
            print(f"  [Warning] Checkpoint not found: {ckpt_path}. Starting fresh.")

    params = count_router_params(model)
    print(f"\n  RouterModel: {params:,} params")
    print(f"  Intent classes: {INTENT_NAMES}")
    print(f"  Multi-label: {multi_label}")

    # ── 5. EWC setup (Phase 2+) ─────────────────────────────────
    ewc = None
    if args.phase >= 2:

        ewc_path = ROUTER_DIR / "ewc_fisher.pt"
        if args.ewc_checkpoint:
            # Load Phase 1 model, compute Fisher on Phase 1 data
            ewc_path = Path(args.ewc_checkpoint).parent / "ewc_fisher.pt"

        ewc = setup_ewc_for_router(
            model, train_ids, train_targets,
            gamma=args.gamma, use_archive=True,
            ewc_path=ewc_path if ewc_path.exists() else None,
        )
        ewc.save(ROUTER_DIR / "ewc_fisher.pt")
        print(f"  [EWC] State saved to {ROUTER_DIR / 'ewc_fisher.pt'}")

    # ── 6. Train ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING — Phase {args.phase}")
    print(f"{'='*60}\n")

    history = train_router(
        model=model,
        train_ids=train_ids,
        train_targets=train_targets,
        eval_ids=eval_ids,
        eval_targets=eval_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        curriculum_phase=args.phase,
        output_dir=ROUTER_DIR,
        log_every=max(1, args.epochs // 10),
        save_every=max(1, args.epochs // 2),
        ewc=ewc,
        lambda_ewc=args.lambda_ewc if ewc else 1000.0,
    )

    # ── 7. Save history ─────────────────────────────────────────
    history_path = ROUTER_DIR / "training_history.json"
    # Convert tensors to floats for JSON serialization
    json_safe = {}
    for k, v in history.items():
        if isinstance(v, list):
            json_safe[k] = [float(x) if isinstance(x, (torch.Tensor,)) else x for x in v]
        elif isinstance(v, (torch.Tensor,)):
            json_safe[k] = float(v)
        else:
            json_safe[k] = v
    with open(history_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\n  [*] Training history saved to {history_path}")
    print(f"  [*] Best accuracy: {history.get('best_acc', 0):.3f}")
    print("  [*] Done!")


if __name__ == "__main__":
    main()
