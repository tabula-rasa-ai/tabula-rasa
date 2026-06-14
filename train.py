"""Training loop for tiny math transformer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tabula_rasa.config import Config
from tabula_rasa.dataset import MathDataset, format_math_sample, generate_problem
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer


def get_lr_scheduler(optimizer, warmup_steps, max_steps):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, tokenizer, num_problems=50, max_digits=4):
    """Evaluate model on fresh arithmetic problems.
    Returns (accuracy, list of (value_pred, value_target_z)) if value head exists.
    """
    model.eval()
    total = 0
    correct = 0
    device = next(model.parameters()).device
    value_pairs = []  # (predicted_value, target_z)

    for _ in range(num_problems):
        expr, ans = generate_problem(1, max_digits)
        prompt = f"{expr}="
        generated = model.generate(tokenizer, prompt, max_new_tokens=10, temperature=0.5, top_k=3)
        # Extract answer part
        if "=" in generated:
            pred = generated.split("=")[-1].strip()
            # Extract just the answer digits
            pred = "".join(c for c in pred if c.isdigit() or c == "-")
            is_correct = pred == ans
            if is_correct:
                correct += 1

            # Value head evaluation: get value prediction for this input
            if model.value_head is not None:
                input_ids = tokenizer.encode(prompt, add_special_tokens=True)
                inp = torch.tensor([input_ids], device=device)
                _, _, v = model(inp, inp)
                z = 1.0 if is_correct else -1.0
                value_pairs.append((v.item(), z))
        total += 1

    acc = correct / total * 100
    return acc, value_pairs


def train(resume_from=None, resume_steps=None):
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}", flush=True)

    # Build tokenizer
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size  # type: ignore[misc]
    tok.max_seq_len = cfg.max_seq_len

    # Create datasets
    print(f"Generating training data...", flush=True)
    train_ds = MathDataset(tok, cfg.train_samples, cfg.min_digits, cfg.max_digits, seed=42)
    eval_ds = MathDataset(tok, cfg.eval_samples, cfg.min_digits, cfg.max_digits, seed=99)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )

    # Build or load model
    model = MathTransformer(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Resume from checkpoint
    global_step = 0
    best_acc = 0.0
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            try:
                if "optimizer_state_dict" in state:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as e:
                print(f"  [*] Optimizer state incompatible ({e}). Starting fresh optimizer.")
            global_step = state.get("step", 0)
            best_acc = state.get("acc", 0.0)
            print(
                f"[*] Resumed from {resume_from} (step {global_step}, acc {best_acc:.1f}%)",
                flush=True,
            )
        else:
            print(f"[!] Checkpoint not found: {resume_from}. Starting from scratch.")

    if resume_steps:
        cfg.max_steps = global_step + resume_steps
        print(
            f"[*] Will train for {resume_steps} additional steps (total: {cfg.max_steps})",
            flush=True,
        )

    print(f"Model parameters: {count_parameters(model):,}", flush=True)
    print(f"Model created, starting scheduler...", flush=True)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps)

    # Save dir
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Save tokenizer with config
    tok.save(str(save_dir / "tokenizer.json"))

    # Training state (only init if not already set from resume)
    if "global_step" not in dir() or global_step is None:
        global_step = 0
    if "best_acc" not in dir() or best_acc is None:
        best_acc = 0.0
    losses = []
    start_time = time.time()

    # Open training log for progress
    log_path = Path("training.log")
    log_file = open(log_path, "w")

    print(f'\n{"="*60}')
    print(f"Training for {cfg.max_steps} steps...")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f'{"="*60}\n')

    while global_step < cfg.max_steps:
        for batch in train_loader:
            if global_step >= cfg.max_steps:
                break

            x, y = batch
            x, y = x.to(device), y.to(device)

            _, loss, _ = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            global_step += 1

            # Log
            if global_step % cfg.log_every == 0:
                avg_loss = sum(losses[-cfg.log_every :]) / len(losses[-cfg.log_every :])
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                line = f"Step {global_step:>6d}/{cfg.max_steps} | loss={avg_loss:.4f} | lr={lr:.2e} | {steps_per_sec:.1f} steps/s"
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()

            # Evaluate
            if global_step % cfg.eval_every == 0:
                acc, value_pairs = evaluate(model, tok, num_problems=50, max_digits=cfg.max_digits)
                eval_line = f"[*] Eval accuracy: {acc:.1f}% (best: {max(best_acc, acc):.1f}%)"
                print(f"  {eval_line}", flush=True)
                log_file.write(eval_line + "\n")
                log_file.flush()

                # AlphaZero value head training
                if cfg.use_value_head and value_pairs:
                    v_preds = torch.tensor([p[0] for p in value_pairs], device=device)
                    v_targets = torch.tensor([p[1] for p in value_pairs], device=device)
                    v_loss = F.mse_loss(v_preds, v_targets)
                    v_weight = getattr(cfg, "alphazero_loss_weight", 0.5)
                    total_v_loss = v_loss * v_weight
                    total_v_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    print(
                        f"  [*] Value loss: {v_loss.item():.4f} | Target z: {v_targets.mean().item():.2f} avg",
                        flush=True,
                    )
                    log_file.write(f"[*] Value loss: {v_loss.item():.4f}\n")
                    log_file.flush()

                if acc > best_acc:
                    best_acc = acc
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.item(),
                            "acc": acc,
                        },
                        save_dir / "best.pt",
                    )
                    print(f"  [*] New best model saved!")

            # Save checkpoint
            if global_step % cfg.save_every == 0:
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                    },
                    save_dir / f"checkpoint_{global_step}.pt",
                )
                print(f"  [*] Checkpoint saved at step {global_step}")

    # Save final model
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
            "final_loss": sum(losses[-1000:]) / 1000,
        },
        save_dir / "final.pt",
    )

    elapsed = time.time() - start_time
    print(f"============================================================")
    print(f"Training complete! {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best eval accuracy: {best_acc:.1f}%")
    print(f"Final model saved to checkpoints/final.pt")
    print(f"Best model saved to checkpoints/best.pt")
    print(f"============================================================")
    log_file.write(f"Training complete! Best accuracy: {best_acc:.1f}%\n")
    log_file.close()

    return model, tok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or resume the math transformer")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g. checkpoints/best.pt)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of additional steps when resuming (ignored for fresh training)",
    )
    parser.add_argument("--use-value-head", action="store_true", help="Enable AlphaZero value head")
    args = parser.parse_args()
    cfg = Config()
    if args.use_value_head:
        cfg.use_value_head = True
    model, tok = train(resume_from=args.resume, resume_steps=args.steps)
