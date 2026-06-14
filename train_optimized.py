"""Optimized training — reads all settings from Config.
Easy to customize: just edit config.py or use Model Config dashboard.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

LOG_FILE = Path("training.log")
OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def make_batch(tok, cfg, op, bs, eq_token, use_reversed, use_loss_masking):
    xs, ys = [], []
    ignore_val = -100 if use_loss_masking else tok.pad_id
    for _ in range(bs):
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        if op == "add":
            ans = a + b
        elif op == "sub":
            if a < b:
                a, b = b, a
            ans = a - b
        elif op == "mul":
            ans = a * b
        elif op == "div":
            b = random.randint(1, max(1, a // 2))
            a = b * random.randint(1, 100)
            ans = a // b if b != 0 else 1
        else:
            from train_specialist import generate_problem

            op2 = random.choice(["add", "sub", "mul", "div"])
            e, a2 = generate_problem(op2, 1, 4)
            text = f"{e}={a2}"
            ids = tok.encode(text, add_special_tokens=True)
            ids = (ids + [tok.pad_id] * cfg.max_seq_len)[: cfg.max_seq_len]
            xs.append(torch.tensor(ids[:-1]))
            y = list(ids[1:])
            eq = ids.index(eq_token) if eq_token in ids else -1
            for j in range(len(y)):
                if j < eq or y[j] == tok.pad_id:
                    y[j] = ignore_val
            ys.append(torch.tensor(y))
            continue

        rev = use_reversed and op in ("add", "sub")
        a_s = str(a)[::-1] if rev else str(a)
        b_s = str(b)[::-1] if rev else str(b)
        ans_s = str(ans)[::-1] if rev else str(ans)
        text = f"{a_s}{OPS[op]}{b_s}={ans_s}"
        ids = tok.encode(text, add_special_tokens=True)
        ids = (ids + [tok.pad_id] * cfg.max_seq_len)[: cfg.max_seq_len]
        xs.append(torch.tensor(ids[:-1]))
        y = list(ids[1:])
        eq = ids.index(eq_token) if eq_token in ids else -1
        for j in range(len(y)):
            if j < eq or y[j] == tok.pad_id:
                y[j] = ignore_val
        ys.append(torch.tensor(y))
    return torch.stack(xs), torch.stack(ys)


def quick_eval(model, tok, cfg, op=None, num=100):
    model.eval()
    from train_specialist import generate_problem

    correct = 0
    ops_t = [op] if op else ["add", "sub", "mul", "div"]
    for op_n in ops_t:
        for _ in range(num // len(ops_t)):
            e, a = generate_problem(
                op_n, 1, 4, reversed=(cfg.use_reversed and op_n in ("add", "sub"))
            )
            out = model.generate(
                tok, e, max_new_tokens=cfg.eval_max_tokens, temperature=cfg.eval_temperature
            )
            p = "".join(
                c for c in (out.split("=")[-1] if "=" in out else "") if c.isdigit() or c == "-"
            )
            if p == a:
                correct += 1
    # Diagnostic: print first 3 samples when accuracy is 0
    if correct == 0 and num >= 3:
        print("  [diag] eval sample outputs:")
        for op_n in ops_t[:1]:
            for _ in range(3):
                e, a = generate_problem(
                    op_n, 1, 4, reversed=(cfg.use_reversed and op_n in ("add", "sub"))
                )
                out = model.generate(
                    tok, e, max_new_tokens=cfg.eval_max_tokens, temperature=cfg.eval_temperature
                )
                print(f"    prompt={repr(e)} expected={repr(a)} output={repr(out)}")
    model.train()
    return correct / num * 100


def _make_optimizer(model, cfg, lr):
    """Create optimizer based on config settings."""
    if cfg.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
    else:  # adamw (default)
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )


def _get_lr(s, warmup, steps, schedule="cosine"):
    """LR schedule: cosine (default), linear, or constant."""
    if schedule == "constant":
        return 1.0
    if s < warmup:
        return s / max(1, warmup)
    p = (s - warmup) / max(1, steps - warmup)
    if schedule == "linear":
        return 1.0 - p
    else:  # cosine
        return 0.5 * (1.0 + math.cos(math.pi * p))


def train_optimized(op=None, steps=None, batch_size=None):
    cfg = Config()
    steps = steps or cfg.max_steps
    batch_size = batch_size or cfg.batch_size
    lr = cfg.learning_rate
    warmup = cfg.warmup_steps
    name = f"{op} specialist" if op else "generalist"

    LOG_FILE.write_text("")
    log("=" * 60)
    log(f"  {name}  |  Reversed: {cfg.use_reversed}  |  Loss masking: {cfg.use_loss_masking}")
    log(f"  Batch: {batch_size}  Steps: {steps}")
    log(
        f"  Optimizer: {cfg.optimizer}  |  Schedule: {cfg.lr_schedule}  |  Grad clip: {cfg.grad_clip_norm}"
    )
    log("=" * 60)

    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    eq_token = tok.stoi.get("=", None)

    model = MathTransformer(cfg)
    params = count_parameters(model)
    log(
        f"  Params: {params:,} | d={cfg.d_model} L={cfg.n_layers} ff={cfg.d_ff} | act={cfg.activation} norm={cfg.norm_type} pos={cfg.pos_encoding}"
    )

    opt = _make_optimizer(model, cfg, lr)

    step = 0
    best_acc = 0.0
    t0 = time.time()
    losses = []

    log("")
    while step < steps:
        x, y = make_batch(
            tok, cfg, op, batch_size, eq_token, cfg.use_reversed, cfg.use_loss_masking
        )
        _, loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        opt.step()
        for pg in opt.param_groups:
            pg["lr"] = lr * _get_lr(step, warmup, steps, cfg.lr_schedule)
        losses.append(loss.item())
        step += 1

        if step % max(steps // 10, 100) == 0:
            elapsed = time.time() - t0
            rate = step / max(0.001, elapsed)
            avg_l = sum(losses[-50:]) / max(len(losses[-50:]), 1)
            rem = (steps - step) / max(rate, 0.001)
            log(
                f"  Step {step:>5d}/{steps} | loss={avg_l:.4f} | {rate:.1f} st/s | ETA {int(rem//60):02d}:{int(rem%60):02d}"
            )

        if step % max(steps // 5, 100) == 0:
            acc = quick_eval(model, tok, cfg, op)
            log(f'  {"":>32} Acc: {acc:.1f}% (best: {max(best_acc, acc):.1f}%)')
            sd = Path("specialists") / "math" / (op or "general")
            sd.mkdir(parents=True, exist_ok=True)
            if acc > best_acc:
                best_acc = acc
                torch.save({"model_state_dict": model.state_dict(), "acc": acc}, sd / "best.pt")
                tok.save(str(sd / "tokenizer.json"))
                log(f'  {"":>32} New best — checkpoint saved')
            # Always save a periodic checkpoint for resume/loading
            torch.save(
                {"model_state_dict": model.state_dict(), "acc": acc, "step": step},
                sd / f"checkpoint_{step}.pt",
            )
            log(f'  {"":>32} Auto-saved checkpoint_{step}.pt')

    # Final checkpoint on completion
    sd = Path("specialists") / "math" / (op or "general")
    sd.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "acc": best_acc, "step": step}, sd / "final.pt"
    )
    tok.save(str(sd / "tokenizer.json"))
    elapsed = time.time() - t0
    log(f"\n  Done! {elapsed:.0f}s ({elapsed/60:.1f} min)  Best acc: {best_acc:.1f}%")
    log(f'  Final checkpoint saved to {sd / "final.pt"}')
    return model, tok


if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else None
    if op == "":
        op = None
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    train_optimized(op, steps=steps or None)
