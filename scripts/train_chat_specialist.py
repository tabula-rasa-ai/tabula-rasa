"""Train a chat specialist — a transformer for general conversation.

Uses the BPETokenizer (character-level, supports English) and the
existing MathTransformer architecture. Trains on chat prompt/response
pairs with loss masking so the model only learns to generate responses.

Usage:
    python3 train_chat_specialist.py               # Train with defaults
    python3 train_chat_specialist.py --quick        # Quick smoke test
    python3 train_chat_specialist.py --resume       # Resume from checkpoint
    python3 train_chat_specialist.py --steps 5000   # Custom step count
    python3 train_chat_specialist.py --preset 10M   # Model size (default: 10M)
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.chat_dataset import ChatDataset, encode_chat_sample, evaluate_chat
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters

# ── Signal handling ──────────────────────────────────────────────
_INTERRUPTED = False


def _signal_handler(sig, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n  [!] Interrupt received — saving checkpoint before exit...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Dataset wrapper that returns tensors ──────────────────────────
class TokenizedChatDataset(Dataset):
    """Wraps ChatDataset and produces encoded (input, target) tensors."""

    def __init__(self, chat_ds: ChatDataset, tokenizer, max_seq_len: int):
        self.chat_ds = chat_ds
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_loss_masking = True

    def __len__(self) -> int:
        return len(self.chat_ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt, response = self.chat_ds[idx]
        return encode_chat_sample(
            prompt, response, self.tokenizer, self.max_seq_len, use_loss_masking=True
        )


# ── Training helpers ──────────────────────────────────────────────
def _make_optimizer(model, lr: float, weight_decay: float = 0.01):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _get_lr(step, warmup, max_steps, schedule="cosine"):
    if schedule == "constant":
        return 1.0
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, max_steps - warmup)
    if schedule == "linear":
        return max(0.01, 1.0 - p)
    return 0.5 * (1.0 + math.cos(math.pi * p))  # cosine


def _save_checkpoint(model, optimizer, global_step, best_acc, save_dir, name="checkpoint.pt"):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "best_acc": best_acc,
        },
        save_dir / name,
    )


def _load_checkpoint(save_dir, model, optimizer):
    ckpt_path = save_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        return None
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd, strict=False)
        if optimizer is not None and "optimizer_state_dict" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception:
                pass
        return state.get("global_step", 0), state.get("best_acc", 0.0)
    except Exception as e:
        print(f"  [!] Failed to load checkpoint: {e}")
        return None


# ── Main training function ───────────────────────────────────────
def train_chat_specialist(
    steps: int = 0,
    batch_size: int = 0,
    lr: float = 0,
    preset: str = "10M",
    max_seq_len: int = 128,
    num_samples: int = 5000,
    quick: bool = False,
    resume: bool = False,
    data_path: str | None = None,
):
    """Train a chat specialist model.

    Args:
        steps: Training steps (0=config default).
        batch_size: Batch size (0=auto).
        lr: Learning rate (0=auto).
        preset: Model size preset ("1M", "5M", "10M").
        max_seq_len: Context window.
        num_samples: Number of training samples.
        quick: Quick smoke test mode.
        resume: Resume from checkpoint.
        data_path: Optional JSONL file with chat data.
    """
    global _INTERRUPTED
    _INTERRUPTED = False

    print("\n" + "=" * 60)
    print("  Training Chat Specialist")
    print("=" * 60)

    # ── Config ──
    cfg = Config()

    # Apply preset for model architecture
    cfg.apply_preset(preset)
    cfg.max_seq_len = max_seq_len
    cfg.use_reversed = False
    cfg.use_scratchpad = False
    cfg.cot_scratchpad = False
    cfg.use_curriculum = False
    cfg.use_loss_masking = True

    # Quick mode
    if quick:
        steps = 500
        num_samples = 500
        batch_size = 32
        cfg.eval_every = 100
        cfg.save_every = 200
        print("  Quick mode: 500 steps, 500 samples")

    cfg.max_steps = steps or 10000
    cfg.train_samples = num_samples
    if batch_size > 0:
        cfg.batch_size = batch_size
    if lr > 0:
        cfg.learning_rate = lr
    else:
        cfg.learning_rate = 0.001

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"  Device: CUDA ({device_name})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
        print(f"  Device: MPS ({device_name})")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
        print(f"  Device: CPU ({n_threads} threads)")

    # ── Tokenizer (BPETokenizer for English text) ──
    tok = BPETokenizer()
    tok.max_seq_len = cfg.max_seq_len
    cfg.vocab_size = tok.vocab_size
    print(f"  Tokenizer: BPETokenizer (vocab={tok.vocab_size} tokens)")

    # ── Dataset ──
    t0 = time.time()
    chat_ds = ChatDataset(
        num_samples=cfg.train_samples,
        max_len=cfg.max_seq_len // 2,
        data_path=data_path,
    )
    train_ds = TokenizedChatDataset(chat_ds, tok, cfg.max_seq_len)
    print(f"  DataLoader: {len(train_ds)} samples, batch={cfg.batch_size}")
    ds_time = time.time() - t0

    # ── Model ──
    model = MathTransformer(cfg).to(device)
    n_params = count_parameters(model)
    print(f"  Model: {n_params:,} params")
    print(f"    d_model={cfg.d_model} L={cfg.n_layers} ff={cfg.d_ff} heads={cfg.n_heads}")
    print(f"    max_seq_len={cfg.max_seq_len} vocab={cfg.vocab_size}")
    print(f"    Preset: {preset}")

    optimizer = _make_optimizer(model, cfg.learning_rate)

    # ── Output directory ──
    save_dir = Path("specialists") / "chat"
    save_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_dir / "tokenizer.json"))

    # ── Resume ──
    global_step = 0
    best_acc = 0.0
    if resume:
        result = _load_checkpoint(save_dir, model, optimizer)
        if result:
            global_step, best_acc = result
            print(f"  Resumed from step {global_step} (best: {best_acc:.1f}%)")
        else:
            print("  No checkpoint found — starting fresh")

    # ── DataLoader ──
    nw = 0 if device.type == "cpu" else 2
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=nw
    )
    data_iter = iter(loader)

    # ── Training log ──
    log_path = save_dir / "training.log"
    log_file = open(log_path, "w")
    header = f'=== Chat Specialist — {time.strftime("%Y-%m-%d %H:%M")} ==='
    header += f"\n  Device: {device_name} | Params: {n_params:,}"
    header += f"\n  Steps: {global_step} -> {cfg.max_steps} | Batch: {cfg.batch_size} | LR: {cfg.learning_rate}"
    header += f"\n  Dataset: {len(train_ds)} samples | Max seq: {cfg.max_seq_len}"
    header += f"\n"
    print(header)
    log_file.write(header + "\n")
    log_file.flush()

    t_start = time.time()
    recent_losses = []

    print(f"  Training from step {global_step} to {cfg.max_steps}...")
    print(f"  Press Ctrl+C to save and exit gracefully.\n")

    # ── Training loop ──
    try:
        while global_step < cfg.max_steps:
            if _INTERRUPTED:
                break

            # Get batch
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # Forward
            _, loss, _ = model(x, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            # LR schedule
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * _get_lr(
                    global_step, cfg.warmup_steps, cfg.max_steps, cfg.lr_schedule
                )

            global_step += 1
            recent_losses.append(loss.item())

            # Progress logging
            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                sps = global_step / max(1, elapsed)
                eta = (cfg.max_steps - global_step) / max(0.1, sps)
                avg_loss = sum(recent_losses[-100:]) / max(1, len(recent_losses[-100:]))
                line = (
                    f"  Step {global_step:>6}/{cfg.max_steps} | "
                    f"loss={avg_loss:.4f} | "
                    f"lr={cfg.learning_rate * _get_lr(global_step, cfg.warmup_steps, cfg.max_steps, cfg.lr_schedule):.6f} | "
                    f"{sps:.1f} st/s | ETA: {eta/60:.0f}m"
                )
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()

            # Periodic checkpoint
            if global_step % cfg.save_every == 0:
                _save_checkpoint(model, optimizer, global_step, best_acc, save_dir)

            # Evaluation
            if global_step % cfg.eval_every == 0:
                eval_t = time.time()
                results = evaluate_chat(model, tok, chat_ds, num_samples=50)
                acc = results["accuracy"]

                eval_line = (
                    f"    Eval step {global_step}: {acc:.1f}% "
                    f"(best: {max(best_acc, acc):.1f}%) "
                    f"[{time.time() - t_start:.0f}s]"
                )
                print(eval_line, flush=True)
                log_file.write(eval_line + "\n")
                log_file.flush()

                # Show some sample outputs
                correct_samples = [s for s in results["samples"] if s["correct"]][:2]
                wrong_samples = [s for s in results["samples"] if not s["correct"]][:2]
                if correct_samples:
                    for s in correct_samples:
                        print(f"      OK: {s['prompt']:30s} -> '{s['got']}'", flush=True)
                if wrong_samples:
                    for s in wrong_samples:
                        print(f"      XX: {s['prompt']:30s} expected='{s['expected']}' got='{s['got']}'", flush=True)

                # Save best model
                if acc > best_acc:
                    best_acc = acc
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "acc": acc,
                            "global_step": global_step,
                        },
                        save_dir / "best.pt",
                    )

    except KeyboardInterrupt:
        _INTERRUPTED = True

    # ── Final save ──
    if _INTERRUPTED:
        print(f"\n  Saving checkpoint at step {global_step}...", flush=True)
    _save_checkpoint(model, optimizer, global_step, best_acc, save_dir)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "acc": best_acc,
            "global_step": global_step,
        },
        save_dir / "final.pt",
    )

    elapsed = time.time() - t_start
    done_line = f"\n  Done! {elapsed:.0f}s ({elapsed/60:.1f} min) | Best: {best_acc:.1f}% | Steps: {global_step}"
    print(done_line, flush=True)
    log_file.write(done_line + "\n")
    log_file.close()

    # ── Final evaluation ──
    print("\n  Final evaluation (100 samples):", flush=True)
    final_results = evaluate_chat(model, tok, chat_ds, num_samples=100)
    print(f"    Accuracy: {final_results['accuracy']:.1f}% ({final_results['correct']}/{final_results['total']})", flush=True)

    # Show examples
    print("\n  Sample generations:", flush=True)
    test_prompts = [
        "Hello!=",
        "What is your name?=",
        "What is 2+2?=",
        "Tell me a joke.=",
        "Thanks!=",
        "What is AI?=",
    ]
    for prompt in test_prompts:
        output = model.generate(tok, prompt, max_new_tokens=24, temperature=0.0)
        response = output.split("=", 1)[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
        print(f"    {prompt[:-1]:30s} -> {response}", flush=True)

    return model, tok


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a chat specialist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python3 train_chat_specialist.py                          # Full training
  python3 train_chat_specialist.py --quick                  # Quick smoke test
  python3 train_chat_specialist.py --resume                 # Resume from checkpoint
  python3 train_chat_specialist.py --steps 5000             # Custom steps
  python3 train_chat_specialist.py --preset 1M              # Small model (1M params)
  python3 train_chat_specialist.py --preset 10M             # Default 10M params
  python3 train_chat_specialist.py --data chat_data.jsonl   # Use custom data
        """,
    )
    parser.add_argument("--steps", type=int, default=0, help="Training steps (0=default 10000)")
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0=auto)")
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0=auto)")
    parser.add_argument(
        "--preset", type=str, default="10M", choices=["1M", "5M", "10M"],
        help="Model size preset (default: 10M)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=128,
        help="Max sequence length (default: 128)"
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Training samples (default: 5000)"
    )
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to JSONL chat data")

    args = parser.parse_args()

    train_chat_specialist(
        steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        preset=args.preset,
        max_seq_len=args.max_seq_len,
        num_samples=args.samples,
        quick=args.quick,
        resume=args.resume,
        data_path=args.data,
    )
