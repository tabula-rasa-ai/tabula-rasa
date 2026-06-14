"""Train a specialist model for one arithmetic operation.
All settings from Config — edit config.py or use Model Config dashboard.

Usage:
    python3 train_specialist.py add                 # Train addition
    python3 train_specialist.py sub                 # Train subtraction
    python3 train_specialist.py mul                 # Train multiplication
    python3 train_specialist.py div                 # Train division
    python3 train_specialist.py all                 # Train ALL operations sequentially
    python3 train_specialist.py add --quick         # Quick smoke test (500 steps)
    python3 train_specialist.py add --resume        # Resume from checkpoint
    python3 train_specialist.py add --steps 5000    # Custom step count
    python3 train_specialist.py add --log-level DEBUG  # Debug logging
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import json
import logging
import math
import random
import signal
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ── Structured Logging Setup ────────────────────────────────────
logger = logging.getLogger("train_specialist")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure structured logging with optional file output."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
    )


from egefalos.online_ewc import OnlineEWC
from tabula_rasa.config import Config
from tabula_rasa.lora import (
    apply_lora_to_model,
    load_lora_adapters,
    save_lora_adapters,
    set_lora_trainable,
)
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}
OP_NAMES = {"add": "Addition", "sub": "Subtraction", "mul": "Multiplication", "div": "Division", "pow": "Power"}

# ─── Interrupted signal handling ─────────────────────────────────
_INTERRUPTED = False


def _signal_handler(sig, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n  [!] Interrupt received — saving checkpoint before exit...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def generate_problem(
    op: str, min_digits=1, max_digits=4, reversed: bool = False, scratchpad: bool = True
):
    """Generate a problem for a specific operation.

    V2: Combined carry-digit tokens f'{carry}{digit}' per column.
    Each column is one token (e.g. "04" = carry=0, digit=4).
    """
    a_digits = random.randint(min_digits, max_digits)
    b_digits = random.randint(min_digits, max_digits)
    a = random.randint(10 ** (a_digits - 1), 10**a_digits - 1)
    b = random.randint(10 ** (b_digits - 1), 10**b_digits - 1)

    if op == "add":
        ans = a + b
    elif op == "sub":
        if a < b:
            a, b = b, a
        ans = a - b
    elif op == "mul":
        ans = a * b
    elif op == "div":
        ans = random.randint(1, max(1, a // 2))
        b = random.randint(1, max(1, a // 2))
        if b == 0:
            b = 1
        a = b * ans
        if a == 0:
            a = b * max(1, ans)
        ans = a // b if b != 0 else 1
    elif op == "pow":
        # Small exponents to keep results manageable
        a = random.randint(2, 12)
        exp = random.randint(1, 4)
        b = exp
        ans = a ** exp

    if reversed and op in ("add", "sub"):
        a_str = str(a)[::-1]
        b_str = str(b)[::-1]
        if scratchpad:
            sp = ""
            carry = 0
            max_len = max(len(str(a)), len(str(b)))
            ra, rb = str(a)[::-1], str(b)[::-1]
            for i in range(max_len):
                da = int(ra[i]) if i < len(ra) else 0
                db = int(rb[i]) if i < len(rb) else 0
                total = da + db + carry
                carry = total // 10
                digit = total % 10
                sp += f"{carry}{digit}"
            ans_str = sp
        else:
            ans_str = str(ans)[::-1]
        return f"{a_str}{OPS[op]}{b_str}", ans_str
    return f"{a}{OPS[op]}{b}", str(ans)


class SpecialistDataset(Dataset):
    """Dataset of problems for one operation."""

    def __init__(self, tokenizer, op: str, cfg: Config):
        self.tokenizer = tokenizer
        self.max_seq_len = cfg.max_seq_len
        self.use_loss_masking = cfg.use_loss_masking
        self.pad_id = tokenizer.pad_id
        force_carry = getattr(cfg, "force_carry_ratio", 0.0) if op in ("add", "sub") else 0.0
        self.samples = []
        target_carry = int(cfg.train_samples * force_carry)
        carry_count = 0
        max_attempts = cfg.train_samples * 3
        for _ in range(max_attempts):
            if len(self.samples) >= cfg.train_samples:
                break
            expr, ans = generate_problem(
                op,
                cfg.min_digits,
                cfg.max_digits,
                reversed=(cfg.use_reversed and op in ("add", "sub")),
                scratchpad=(cfg.use_scratchpad and op in ("add", "sub")),
            )
            ids = tokenizer.encode(f"{expr}={ans}", add_special_tokens=True)
            if len(ids) > self.max_seq_len:
                continue
            has_carry = "+" in expr and len(ans) >= 2 and ans[0] == "1"
            if force_carry > 0 and has_carry and carry_count < target_carry:
                self.samples.append(ids)
                carry_count += 1
            elif (
                force_carry > 0
                and not has_carry
                and (len(self.samples) - carry_count) < (cfg.train_samples - target_carry)
            ):
                self.samples.append(ids)
            elif force_carry == 0:
                self.samples.append(ids)
        print(f"  Created {len(self.samples)} {op} samples (carry: {carry_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        padded = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        if self.use_loss_masking:
            y[y == self.pad_id] = -100
            eq_id = self.tokenizer.stoi["="]
            eq_positions = (y == eq_id).nonzero(as_tuple=True)[0]
            if len(eq_positions) > 0:
                eq_pos = eq_positions[0].item()
                y[: eq_pos + 1] = -100
        return x, y


def parse_scratchpad_answer(text: str) -> str:
    """Extract the final answer digits from a scratchpad output.

    V2 format: "0406" = carry=0,digit=4 | carry=0,digit=6
    Each column is 2 chars: carry(0-1) + digit(0-9).
    Extracts every second char: "4" then "6" -> "46" (LSD-first).
    """
    text = text.replace("<EOS>", "").replace("<PAD>", "").replace("<BOS>", "").strip()
    if "=" in text:
        text = text.split("=")[-1]
    # Ensure even length
    if len(text) % 2 != 0:
        text = text[:-1]
    return text[1::2]


def _count_digits(expr: str, op: str) -> int:
    """Get the max number of digits in a problem expression."""
    if op in OPS:
        parts = expr.replace(OPS[op], " ").split()
        if len(parts) == 2:
            return max(len(parts[0]), len(parts[1]))
    return max(
        len(p)
        for p in expr.replace("+", " ")
        .replace("-", " ")
        .replace("*", " ")
        .replace("/", " ")
        .split()
        if p.isdigit()
    )


@torch.no_grad()
def evaluate(model, tokenizer, cfg: Config, op: str, num=0, min_digits=0, max_digits=0):
    """Evaluate model accuracy on fresh problems.

    Args:
        num: Number of problems (0=cfg.eval_samples)
        min_digits: Override min digit count (0=cfg.min_digits)
        max_digits: Override max digit count (0=cfg.max_digits)
    """
    model.eval()
    num = num or cfg.eval_samples
    min_d = min_digits or cfg.min_digits
    max_d = max_digits or cfg.max_digits
    correct = 0
    for _ in range(num):
        expr, ans_raw = generate_problem(
            op,
            min_d,
            max_d,
            reversed=(cfg.use_reversed and op in ("add", "sub")),
            scratchpad=(cfg.use_scratchpad and op in ("add", "sub")),
        )
        prompt = f"{expr}="
        out = model.generate(
            tokenizer,
            prompt,
            max_new_tokens=cfg.eval_max_tokens,
            temperature=cfg.eval_temperature,
            top_k=3,
        )
        if cfg.use_scratchpad and op in ("add", "sub"):
            pred = parse_scratchpad_answer(out)
            true_answer = parse_scratchpad_answer(ans_raw)
            if pred == true_answer:
                correct += 1
        else:
            pred = "".join(
                c for c in (out.split("=")[-1] if "=" in out else "") if c.isdigit() or c == "-"
            )
            if pred == ans_raw:
                correct += 1
    return correct / max(1, num) * 100


@torch.no_grad()
def evaluate_per_digit(model, tokenizer, cfg: Config, op: str, per_digit_samples=25, hard=False):
    """Evaluate accuracy broken down by number of digits.

    Returns dict like {1: 85.0, 2: 42.0, 3: 12.0, 4: 0.0, 'hard': 0.0}
    """
    model.eval()
    results = {}
    max_d = cfg.max_digits
    if hard:
        max_d = min(cfg.max_digits + 1, 6)  # Test one digit harder

    for d in range(cfg.min_digits, max_d + 1):
        correct = 0
        for _ in range(per_digit_samples):
            expr, ans_raw = generate_problem(
                op,
                d,
                d,
                reversed=(cfg.use_reversed and op in ("add", "sub")),
                scratchpad=(cfg.use_scratchpad and op in ("add", "sub")),
            )
            prompt = f"{expr}="
            out = model.generate(
                tokenizer,
                prompt,
                max_new_tokens=cfg.eval_max_tokens,
                temperature=cfg.eval_temperature,
                top_k=3,
            )
            if cfg.use_scratchpad and op in ("add", "sub"):
                pred = parse_scratchpad_answer(out)
                true_answer = parse_scratchpad_answer(ans_raw)
                if pred == true_answer:
                    correct += 1
            else:
                pred = "".join(
                    c for c in (out.split("=")[-1] if "=" in out else "") if c.isdigit() or c == "-"
                )
                if pred == ans_raw:
                    correct += 1
        results[d] = correct / max(1, per_digit_samples) * 100

    return results


def _make_optimizer(model, cfg, lr):
    if cfg.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )


def _get_lr(step, warmup, steps, schedule="cosine"):
    if schedule == "constant":
        return 1.0
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, steps - warmup)
    if schedule == "linear":
        return max(0.01, 1.0 - p)  # Floor at 1% of LR to avoid dead zones
    return 0.5 * (1.0 + math.cos(math.pi * p))


def _validate_config(cfg, op):
    """Check for known bad configurations before wasting compute."""
    warnings = []

    if cfg.use_scratchpad and op not in ("add", "sub"):
        warnings.append(f"Scratchpad only works for add/sub, not {op}. Disabling.")

    if cfg.use_reversed and op not in ("add", "sub"):
        warnings.append(f"Reversed digits only meaningful for add/sub, not {op}. Disabling.")

    if cfg.max_digits < 2:
        warnings.append("max_digits=1 gives false 100% accuracy. Use >=4 for meaningful results.")

    if cfg.train_samples > 50000 and cfg.max_steps < 20000:
        warnings.append(
            f"Dataset ({cfg.train_samples}) much larger than steps ({cfg.max_steps}). Consider fewer samples."
        )

    if cfg.d_model % cfg.n_heads != 0:
        warnings.append(
            f"d_model ({cfg.d_model}) not divisible by n_heads ({cfg.n_heads}). Training will CRASH."
        )

    return warnings


def _estimate_time(steps, device_str):
    """Rough time estimate based on hardware."""
    if "cuda" in device_str:
        secs_per_step = 0.02
    else:
        secs_per_step = 1.2  # ~0.8 steps/s on typical CPU
    total_sec = steps * secs_per_step
    if total_sec > 3600:
        return f"~{total_sec/3600:.1f} hours"
    elif total_sec > 60:
        return f"~{total_sec/60:.0f} min"
    return f"~{total_sec:.0f}s"


def _save_checkpoint(
    model, optimizer, global_step, best_acc, op_dir, name="checkpoint.pt", lora_layers=None
):
    """Save a resume-able checkpoint.

    When *lora_layers* is provided, saves only the LoRA adapter weights
    (lightweight) and a small metadata checkpoint.
    """
    if lora_layers is not None:
        # LoRA mode: save adapters + metadata
        save_lora_adapters(lora_layers, str(op_dir / "lora.pt"))
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "best_acc": best_acc,
            },
            op_dir / name,
        )
    else:
        # Full checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "best_acc": best_acc,
            },
            op_dir / name,
        )


def _load_checkpoint(op_dir, model, optimizer):
    """Load a resume checkpoint. Returns (global_step, best_acc) or None."""
    ckpt_path = op_dir / "checkpoint.pt"
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
                pass  # Optimizer state mismatch is non-fatal
        return state.get("global_step", 0), state.get("best_acc", 0.0)
    except Exception as e:
        print(f"  [!] Failed to load checkpoint: {e}")
        return None


def _log_experiment(exp):
    """Send experiment result to API server if running."""
    try:
        body = json.dumps(exp).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/experiments",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        print(f"  [*] Experiment logged to comparator", flush=True)
    except Exception:
        pass  # Server not running is fine


def train_specialist(
    op,
    steps=0,
    batch_size=0,
    lr=0,
    use_reversed=None,
    use_loss_masking=None,
    quick=False,
    resume=False,
    test_hard=False,
    deep=False,
    use_ewc=False,
    ewc_lambda=1000.0,
    ewc_gamma=0.9,
    use_amp=False,
    gradient_accumulation_steps=0,
    use_lora=False,
    lora_rank=8,
    socratic=False,
    socratic_steps=500,
    socratic_problems=200,
):
    """Train a specialist for one arithmetic operation.

    Args:
        op: Operation (add, sub, mul, div)
        steps: Max training steps (0=use Config default)
        batch_size: Batch size (0=use Config default)
        lr: Learning rate (0=use Config default)
        use_reversed: Override reversed digits (None=use Config)
        use_loss_masking: Override loss masking (None=use Config)
        quick: Quick smoke test mode (500 steps, small dataset)
        resume: Resume from last checkpoint
        test_hard: Enable hard eval (max_digits+1, tests generalization)
        use_lora: Enable LoRA fine-tuning (freezes base, trains adapters)
        lora_rank: LoRA rank (default 8)
    """
    global _INTERRUPTED
    _INTERRUPTED = False

    print(f'\n{"="*60}')
    print(f"  Training {OP_NAMES[op]} Specialist ({OPS[op]})")
    print(f'{"="*60}')

    cfg = Config()

    # Quick mode overrides
    if quick:
        steps = 200
        cfg.train_samples = 2000
        cfg.batch_size = 64
        cfg.eval_every = 100
        cfg.eval_samples = 20
        cfg.save_every = 100
        cfg.use_curriculum = False
        logger.info("Quick mode: 200 steps, 2K samples, 20 eval")

    cfg.max_steps = steps or cfg.max_steps
    if batch_size > 0:
        cfg.batch_size = batch_size
    if lr > 0:
        cfg.learning_rate = lr
    if use_reversed is not None:
        cfg.use_reversed = use_reversed
    if use_loss_masking is not None:
        cfg.use_loss_masking = use_loss_masking
    if test_hard:
        cfg.test_hard = True

    # Deep model override: d=64 L=8 ff=256 (~500K params, better for algorithms)
    if deep:
        cfg.d_model = 64
        cfg.n_layers = 8
        cfg.n_heads = 4
        cfg.d_ff = 256
        print(f"  *** DEEP MODEL: d=64 L=8 ff=256 (depth > width for algorithms) ***")

    # ── Validate ──
    warnings = _validate_config(cfg, op)
    for w in warnings:
        logger.warning(w)
    if any("CRASH" in w for w in warnings):
        logger.error("Aborting — fix config issues first.")
        return None, None

    # ── Device detection ──
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
        print(f"  Device: CPU (no CUDA detected)")
        # CPU thread tuning — avoid oversubscription on many-core machines
        n_threads = min(6, os.cpu_count() or 4)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(2)
        print(f"  CPU threads: {n_threads} intra / 2 inter")

    # ── Tokenizer ──
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    # ── Mixed precision (AMP) ──
    cfg.use_amp = use_amp or cfg.use_amp
    use_amp_enabled = cfg.use_amp and device.type == "cuda"
    if use_amp_enabled:
        print(f"  AMP: enabled (mixed precision)")
    elif cfg.use_amp and device.type not in ("cuda",):
        print(f"  AMP: disabled (requires CUDA; MPS does not support amp.GradScaler)")

    # ── Gradient accumulation ──
    if gradient_accumulation_steps > 0:
        cfg.gradient_accumulation_steps = gradient_accumulation_steps
    if cfg.gradient_accumulation_steps > 1:
        print(
            f"  Grad Accum: {cfg.gradient_accumulation_steps} steps (effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps})"
        )

    # ── Disable scratchpad/reversed for mul/div ──
    if op not in ("add", "sub"):
        cfg.use_scratchpad = False
        cfg.use_reversed = False

    # ── Curriculum setup ──
    use_curriculum = cfg.use_curriculum and not quick
    curriculum_max_digits = cfg.min_digits  # Start easy
    if use_curriculum:
        print(f"  Curriculum: {len(cfg.curriculum_phases)} phases")
        for i, (steps, digits) in enumerate(cfg.curriculum_phases):
            print(f"    Phase {i+1}: {steps} steps at max_digits={digits}")

    # ── Dataset (curriculum: start with easy digits) ──
    t0 = time.time()
    current_max_digits = cfg.min_digits if use_curriculum else cfg.max_digits
    print(
        f"  Building dataset ({cfg.train_samples} samples, max_digits={current_max_digits})...",
        end=" ",
        flush=True,
    )
    # Temporarily override max_digits for initial dataset
    orig_max_digits = cfg.max_digits
    cfg.max_digits = current_max_digits
    train_ds = SpecialistDataset(tok, op, cfg)
    cfg.max_digits = orig_max_digits  # Restore
    ds_time = time.time() - t0
    print(f"done in {ds_time:.1f}s")

    if len(train_ds) < 10:
        print(f"  [X] Dataset too small ({len(train_ds)} samples). Check config.")
        return None, None

    # ── Model ──
    model = MathTransformer(cfg).to(device)
    use_compile = args.compile or os.environ.get("TABULA_RASA_COMPILE", "").lower() in ("1", "true")
    if use_compile:
        if device.type == "cpu":
            print("  [!] torch.compile is not recommended on CPU — no speedup expected.")
        else:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print(f"  torch.compile: enabled (mode=reduce-overhead)")
            except Exception as e:
                print(f"  [!] torch.compile failed: {e}")

    # ── LoRA (low-rank adaptation) ──
    lora_layers = None
    if use_lora:
        target_modules = [m.strip() for m in cfg.lora_target_modules.split(",")]
        print(f"  LoRA: enabled (rank={lora_rank}, targets={target_modules})")
        lora_layers = apply_lora_to_model(
            model, rank=lora_rank, alpha=cfg.lora_alpha, target_modules=target_modules
        )
        set_lora_trainable(model, trainable=True)

    n_params = count_parameters(model)
    print(
        f"  Params: {n_params:,} | d={cfg.d_model} L={cfg.n_layers} ff={cfg.d_ff} heads={cfg.n_heads}"
    )
    print(
        f"  Optimizer: {cfg.optimizer}  |  LR schedule: {cfg.lr_schedule}  |  Grad clip: {cfg.grad_clip_norm}"
    )
    print(
        f"  Reversed: {cfg.use_reversed}  |  Loss masking: {cfg.use_loss_masking}  |  Scratchpad: {cfg.use_scratchpad}"
    )
    print(f"  Estimated time: {_estimate_time(cfg.max_steps, str(device))}")

    optimizer = _make_optimizer(model, cfg, cfg.learning_rate)

    # ── Output directory ──
    op_dir = Path("specialists") / "math" / op
    op_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(op_dir / "tokenizer.json"))

    # ── EWC (continual learning) ──
    ewc = None
    if use_ewc:
        ewc = OnlineEWC(model, gamma=ewc_gamma)
        ewc_path = op_dir / "ewc_fisher.pt"
        if ewc_path.exists():
            ewc.load(ewc_path)
            print(
                f"  EWC loaded: {ewc.task_count} prior tasks, {ewc.consolidation_steps} consolidations"
            )
        # Save anchor weights before training begins
        ewc.save_anchor_weights()
        print(f"  EWC enabled (λ={ewc_lambda}, γ={ewc_gamma})")

    # ── Resume or fresh start ──
    global_step = 0
    best_acc = 0.0
    if resume:
        if use_lora:
            # Load base model + LoRA adapters
            lora_path = op_dir / "lora.pt"
            if lora_path.exists():
                target_modules = [m.strip() for m in cfg.lora_target_modules.split(",")]
                lora_layers = load_lora_adapters(
                    model,
                    str(lora_path),
                    rank=lora_rank,
                    alpha=cfg.lora_alpha,
                    target_modules=target_modules,
                )
                set_lora_trainable(model, trainable=True)
                # Load optimizer state from checkpoint if available
                result = _load_checkpoint(op_dir, model, optimizer)
                if result:
                    global_step, best_acc = result
                print(f"  Resumed LoRA from step {global_step} (best: {best_acc:.1f}%)")
            else:
                print(f"  No LoRA adapter found at {lora_path} — starting fresh")
        else:
            result = _load_checkpoint(op_dir, model, optimizer)
            if result:
                global_step, best_acc = result
                print(f"  Resumed from step {global_step} (best: {best_acc:.1f}%)")
            else:
                print(f"  No checkpoint found — starting fresh")

    # ── Training log (overwrite, not append) ──
    log_path = op_dir / "training.log"
    log_file = open(log_path, "w")
    header = f'=== {OP_NAMES[op]} Specialist — {time.strftime("%Y-%m-%d %H:%M")} ==='
    header += f"\n  Device: {device_name} | Params: {n_params:,}"
    header += f"\n  Steps: {global_step} -> {cfg.max_steps} | Batch: {cfg.batch_size} | LR: {cfg.learning_rate}"
    header += f"\n  Reversed: {cfg.use_reversed} | Loss masking: {cfg.use_loss_masking} | Scratchpad: {cfg.use_scratchpad}"
    header += f"\n  Activation: {cfg.activation} | Norm: {cfg.norm_type} | Pos: {cfg.pos_encoding}"
    header += f"\n"
    print(header)
    log_file.write(header + "\n")
    log_file.flush()

    t_start = time.time()
    last_eval_time = t_start
    eval_history = []  # Track recent accuracies for early stopping
    loss_streak_count = 0  # Consecutive evals with no improvement
    recent_losses = []  # Recent loss values for entropy-based curriculum
    entropy_msg = ""  # Entropy trigger message (for curriculum logging)
    curriculum_phase = 0  # Current curriculum phase index
    curriculum_step_offset = global_step  # Steps at start of current phase

    print(f"  Training from step {global_step} to {cfg.max_steps}...", flush=True)
    if use_curriculum:
        print(
            f"  Curriculum: phase 1/{len(cfg.curriculum_phases)} (max_digits={cfg.curriculum_phases[0][1]})"
        )
    print(f"  Press Ctrl+C to save and exit gracefully.\\n")

    # ── Persistent batch generator (avoids recreating DataLoader every epoch) ──
    _nw = 4 if device.type == "cuda" else 0

    def _infinite_batches(dataset, batch_size):
        while True:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=_nw, drop_last=True
            )
            yield from loader

    scaler = GradScaler() if use_amp_enabled else None
    batch_iter = _infinite_batches(train_ds, cfg.batch_size)

    try:
        while global_step < cfg.max_steps:
            if _INTERRUPTED:
                break

            # ── Curriculum phase transition ──
            if use_curriculum and curriculum_phase < len(cfg.curriculum_phases):
                phase_steps, phase_digits = cfg.curriculum_phases[curriculum_phase]
                steps_in_phase = global_step - curriculum_step_offset

                # Check if it's time to advance
                advance = steps_in_phase >= phase_steps

                # Entropy-based advancement: advance when model is confident enough
                if cfg.use_entropy_curriculum and not advance:
                    # Use running average loss as entropy proxy
                    # Loss < threshold indicates high confidence
                    if len(recent_losses) >= cfg.entropy_window:
                        avg_loss = sum(recent_losses[-cfg.entropy_window :]) / cfg.entropy_window
                        if avg_loss < cfg.entropy_threshold and steps_in_phase >= max(
                            100, phase_steps // 10
                        ):
                            advance = True
                            entropy_msg = f" (entropy-triggered: avg_loss={avg_loss:.4f} < threshold={cfg.entropy_threshold})"
                        else:
                            entropy_msg = ""

                if advance and curriculum_phase + 1 < len(cfg.curriculum_phases):
                    # Move to next phase
                    curriculum_phase += 1
                    new_steps, new_digits = cfg.curriculum_phases[curriculum_phase]
                    curriculum_step_offset = global_step
                    entropy_info = entropy_msg if cfg.use_entropy_curriculum and entropy_msg else ""
                    curr_msg = f"\n  >>> Curriculum phase {curriculum_phase + 1}/{len(cfg.curriculum_phases)}: max_digits={new_digits} ({new_steps} steps){entropy_info} <<<\n"
                    print(curr_msg, flush=True)
                    log_file.write(curr_msg + "\n")
                    log_file.flush()
                    # Rebuild dataset with new difficulty
                    t_rebuild = time.time()
                    cfg.max_digits = new_digits
                    train_ds = SpecialistDataset(tok, op, cfg)
                    cfg.max_digits = orig_max_digits
                    batch_iter = _infinite_batches(
                        train_ds, cfg.batch_size
                    )  # Reset iterator for new dataset
                    rebuild_msg = f"  Dataset rebuilt: {len(train_ds)} samples in {time.time()-t_rebuild:.1f}s"
                    print(rebuild_msg, flush=True)
                    log_file.write(rebuild_msg + "\n")
                    log_file.flush()

            # ── Single training step ──
            if global_step >= cfg.max_steps or _INTERRUPTED:
                break

            grad_accum_steps = cfg.gradient_accumulation_steps
            last_micro_loss = 0.0
            acc_counter = 0

            # Accumulate micro-batches from the persistent generator
            for _ in range(grad_accum_steps):
                if global_step >= cfg.max_steps or _INTERRUPTED:
                    break
                x, y = next(batch_iter)
                x, y = x.to(device), y.to(device)

                if scaler is not None:
                    with autocast():
                        _, loss, _ = model(x, y)
                else:
                    _, loss, _ = model(x, y)

                last_micro_loss = loss.item()
                recent_losses.append(last_micro_loss)
                loss_for_backward = loss / grad_accum_steps

                if ewc is not None:
                    ewc_penalty = ewc.compute_ewc_penalty(lambda_ewc=ewc_lambda)
                    total_loss = loss_for_backward + ewc_penalty / grad_accum_steps
                else:
                    total_loss = loss_for_backward

                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                acc_counter += 1

            # ── Optimizer step (after accumulation completes) ──
            if cfg.grad_clip_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            global_step += 1

            # Learning rate schedule (based on optimizer steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * _get_lr(
                    global_step, cfg.warmup_steps, cfg.max_steps, cfg.lr_schedule
                )

            # ── Progress logging (per-micro-batch loss) ──
            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                sps = global_step / max(1, elapsed)
                eta = (cfg.max_steps - global_step) / max(0.1, sps)
                current_lr = cfg.learning_rate * _get_lr(
                    global_step, cfg.warmup_steps, cfg.max_steps, cfg.lr_schedule
                )
                phase_info = f" P{curriculum_phase+1}" if use_curriculum else ""
                line = (
                    f"  Step {global_step:>6}/{cfg.max_steps} | "
                    f"loss={last_micro_loss:.4f} | lr={current_lr:.6f} | "
                    f"{sps:.1f} st/s | ETA: {eta/60:.0f}m{phase_info}"
                )
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()

            # ── Periodic checkpoint (for resume) ──
            if global_step % cfg.save_every == 0:
                _save_checkpoint(
                    model, optimizer, global_step, best_acc, op_dir, lora_layers=lora_layers
                )

            # ── Evaluation ──
            if global_step % cfg.eval_every == 0:
                eval_t = time.time()
                # Use fewer samples early in training (model is near-random)
                fast_eval = global_step < cfg.max_steps * 0.3
                n_eval = max(25, cfg.eval_samples // 4) if fast_eval else cfg.eval_samples
                # Tight token budget per digit count
                if cfg.use_scratchpad and op in ("add", "sub"):
                    eval_max_tokens = cfg.max_digits * 2 + 4
                else:
                    eval_max_tokens = cfg.max_digits + 3
                cfg_eval_max = cfg.eval_max_tokens
                cfg.eval_max_tokens = eval_max_tokens
                acc = evaluate(model, tok, cfg, op, num=n_eval)
                cfg.eval_max_tokens = cfg_eval_max
                elapsed = time.time() - t_start

                # Per-digit breakdown: only useful once model is learning
                if acc > 50.0 or global_step > cfg.max_steps * 0.7:
                    per_digit_n = max(5, 15 if not fast_eval else 5)
                    per_digit = evaluate_per_digit(
                        model, tok, cfg, op, per_digit_samples=per_digit_n, hard=cfg.test_hard
                    )
                    digit_str = " ".join(f"{d}d:{v:.0f}%" for d, v in per_digit.items())
                else:
                    per_digit = {}
                    digit_str = "(acc<50%, skip per-digit)"

                eval_history.append(acc)
                eval_line = f"    Eval step {global_step}: {acc:.1f}% (best: {max(best_acc, acc):.1f}%) | {digit_str} [{elapsed:.0f}s]"
                print(eval_line, flush=True)
                log_file.write(eval_line + "\n")
                log_file.flush()

                # ── W&B logging (optional) ──
                try:
                    wandb.log({
                        "step": global_step,
                        "eval_accuracy": acc,
                        "best_accuracy": best_acc,
                        "training_loss": last_micro_loss,
                        "learning_rate": current_lr,
                    })
                except (NameError, AttributeError):
                    pass

                if acc > best_acc:
                    best_acc = acc
                    loss_streak_count = 0
                    if use_lora and lora_layers:
                        save_lora_adapters(lora_layers, str(op_dir / "best_lora.pt"))
                        torch.save(
                            {
                                "acc": acc,
                                "global_step": global_step,
                                "per_digit": per_digit,
                                "config": {
                                    k: getattr(cfg, k, None)
                                    for k in [
                                        "d_model",
                                        "n_layers",
                                        "n_heads",
                                        "d_ff",
                                        "use_reversed",
                                        "use_loss_masking",
                                        "use_scratchpad",
                                    ]
                                },
                            },
                            op_dir / "best.pt",
                        )
                    else:
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "acc": acc,
                                "global_step": global_step,
                                "per_digit": per_digit,
                                "config": {
                                    k: getattr(cfg, k, None)
                                    for k in [
                                        "d_model",
                                        "n_layers",
                                        "n_heads",
                                        "d_ff",
                                        "use_reversed",
                                        "use_loss_masking",
                                        "use_scratchpad",
                                    ]
                                },
                            },
                            op_dir / "best.pt",
                        )
                else:
                    loss_streak_count += 1

                # ── Early stopping: 5 evals with no improvement ──
                if loss_streak_count >= 5 and global_step > cfg.max_steps * 0.3:
                    early_msg = f"  [!] Early stop at step {global_step} — no improvement for {loss_streak_count} evals (best: {best_acc:.1f}%)"
                    print(early_msg, flush=True)
                    log_file.write(early_msg + "\n")
                    log_file.flush()
                    _INTERRUPTED = True
                    break

    except KeyboardInterrupt:
        _INTERRUPTED = True

    # ── Final save ──
    if _INTERRUPTED:
        print(f"\n  Saving checkpoint at step {global_step}...", flush=True)
        _save_checkpoint(model, optimizer, global_step, best_acc, op_dir, lora_layers=lora_layers)
        if use_lora:
            print(
                f"  LoRA adapters saved. Resume with: python3 train_specialist.py {op} --resume --lora",
                flush=True,
            )
        else:
            print(
                f"  Checkpoint saved. Resume with: python3 train_specialist.py {op} --resume",
                flush=True,
            )

    if use_lora and lora_layers:
        save_lora_adapters(lora_layers, str(op_dir / "final_lora.pt"))
        torch.save(
            {
                "acc": best_acc,
                "global_step": global_step,
            },
            op_dir / "final.pt",
        )
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "acc": best_acc,
                "global_step": global_step,
            },
            op_dir / "final.pt",
        )

    elapsed = time.time() - t_start
    done_line = f"\n  Done! {elapsed:.0f}s ({elapsed/60:.1f} min) | Best: {best_acc:.1f}% | Steps: {global_step}"
    print(done_line, flush=True)
    log_file.write(done_line + "\n")
    log_file.close()

    # ── Save EWC state ──
    if ewc is not None:
        ewc_path = op_dir / "ewc_fisher.pt"
        ewc.save(ewc_path)
        print(f"  EWC saved ({ewc.task_count} tasks, {ewc.consolidation_steps} consolidations)")

    # ── Socratic Self-Improvement (post-training refinement) ──
    if socratic and not _INTERRUPTED and best_acc > 30.0:
        print(f'\n{"="*60}')
        print(f"  Socratic Self-Improvement Phase")
        print(f"  Refining {OP_NAMES[op]} specialist via critique loop")
        print(f"  {socratic_problems} problems, {socratic_steps} training steps")
        print(f'{"="*60}')
        try:
            from egefalos.socratic_trainer import SocraticSelfTrainer

            socratic_op = OPS.get(op, "+")
            trainer = SocraticSelfTrainer(
                model=model,
                tokenizer=tok,
                device=device,
                op=socratic_op,
                max_digits=cfg.max_digits,
                lr=cfg.learning_rate,
            )
            socratic_results = trainer.run_self_improvement(
                iterations=1,
                problems_per_iter=socratic_problems,
                train_steps_per_iter=socratic_steps,
                batch_size=cfg.batch_size,
            )
            if socratic_results and socratic_results[0].get("training", {}).get("trained"):
                socratic_acc = socratic_results[0]["training"].get("acc_after_pct", 0)
                if socratic_acc > best_acc:
                    best_acc = socratic_acc
                    # Save the refined model as best
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "acc": best_acc,
                            "global_step": global_step,
                            "socratic_refined": True,
                        },
                        op_dir / "best.pt",
                    )
                    print(f"  [Socratic] New best accuracy: {best_acc:.1f}%")
                print(
                    f'  [Socratic] Improvement: {socratic_results[0]["training"].get("improvement_pct", 0):+.1f}%'
                )
        except Exception as e:
            print(f"  [!] Socratic refinement failed: {e}")

    # ── Log experiment for comparator ──
    _log_experiment(
        {
            "operation": op,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "dropout": cfg.dropout,
            "activation": cfg.activation,
            "norm_type": cfg.norm_type,
            "pos_encoding": cfg.pos_encoding,
            "use_reversed": cfg.use_reversed,
            "use_loss_masking": cfg.use_loss_masking,
            "use_hard_negative": cfg.use_hard_negative,
            "optimizer": cfg.optimizer,
            "lr_schedule": cfg.lr_schedule,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "warmup_steps": cfg.warmup_steps,
            "max_steps": cfg.max_steps,
            "best_accuracy": round(best_acc, 2),
            "training_time_seconds": round(elapsed),
            "steps_per_sec": round(global_step / max(1, elapsed), 2),
            "total_params": n_params,
            "train_samples": cfg.train_samples,
            "device": device_name,
            "notes": "quick" if quick else "",
        }
    )

    return model, tok


# ─── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a specialist model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 train_specialist.py add --quick        # Quick test (30 seconds)
  python3 train_specialist.py add                # Full training (hours)
  python3 train_specialist.py add --resume       # Resume from checkpoint
  python3 train_specialist.py all                # Train all 4 operations
  python3 train_specialist.py add --steps 5000 --batch 64 --lr 0.0005
        """,
    )
    parser.add_argument("op", nargs="?", default="add", help="Operation: add|sub|mul|div|all")
    parser.add_argument(
        "--steps", type=int, default=0, help="Max training steps (0=config default)"
    )
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0=config default)")
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0=config default)")
    parser.add_argument("--no-reversed", action="store_true", help="Disable reversed digits")
    parser.add_argument("--no-loss-mask", action="store_true", help="Disable loss masking")
    parser.add_argument(
        "--quick", action="store_true", help="Quick smoke test (500 steps, small dataset)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--test-hard", action="store_true", help="Also test on max_digits+1 (generalization check)"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use deeper model: d=64 L=8 ff=256 (better for algorithms)",
    )
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="Enable Online EWC for continual learning (preserves prior tasks)",
    )
    parser.add_argument(
        "--ewc-lambda", type=float, default=1000.0, help="EWC penalty strength (default: 1000)"
    )
    parser.add_argument(
        "--ewc-gamma", type=float, default=0.9, help="Fisher merge decay rate (default: 0.9)"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable mixed precision (AMP) training (requires CUDA)"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Enable torch.compile model optimization (GPU only)"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases experiment tracking"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="tabula-rasa", help="W&B project name"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview config and estimated time, then exit"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=0,
        help="Gradient accumulation steps (0=config default, default: 1)",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA fine-tuning (freezes base, trains low-rank adapters)",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument(
        "--socratic", action="store_true", help="Enable Socratic self-improvement after training"
    )
    parser.add_argument(
        "--socratic-steps",
        type=int,
        default=500,
        help="Training steps for Socratic refinement (default: 500)",
    )
    parser.add_argument(
        "--socratic-problems",
        type=int,
        default=200,
        help="Problems per Socratic iteration (default: 200)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level, log_file="training_specialist.log")
    logger.info("Starting train_specialist with log level: %s", args.log_level)

    if args.op == "all":
        print("\n  Training ALL specialists (add, sub, mul, div)")
        print(f"  This will take a while. Press Ctrl+C to stop between ops.\n")
        results = {}
        for op_name in ["add", "sub", "mul", "div"]:
            if _INTERRUPTED:
                print(f"\n  Stopped by user before {op_name}.")
                break
            model, tok = train_specialist(
                op_name,
                steps=args.steps,
                batch_size=args.batch,
                lr=args.lr,
                use_reversed=None if not args.no_reversed else False,
                use_loss_masking=None if not args.no_loss_mask else False,
                quick=args.quick,
                resume=args.resume,
                test_hard=args.test_hard,
                deep=args.deep,
                use_ewc=args.ewc,
                ewc_lambda=args.ewc_lambda,
                ewc_gamma=args.ewc_gamma,
                use_amp=args.amp,
                gradient_accumulation_steps=args.grad_accum,
                use_lora=args.lora,
                lora_rank=args.lora_rank,
                socratic=args.socratic,
                socratic_steps=args.socratic_steps,
                socratic_problems=args.socratic_problems,
            )
            results[op_name] = "OK" if model is not None else "FAILED"
        print(f"\n  Results: {results}")
        sys.exit(0)

    if args.op not in OPS:
        print(f"Usage: python3 train_specialist.py [add|sub|mul|div|all] [options]")
        print(f"       python3 train_specialist.py add --quick      (smoke test)")
        print(f"       python3 train_specialist.py add --resume     (continue)")
        sys.exit(1)

    # ── Optional Weights & Biases ──
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config={
                    "op": args.op, "steps": args.steps, "batch": args.batch,
                    "lr": args.lr, "amp": args.amp, "compile": args.compile,
                    "reversed": not args.no_reversed,
                    "loss_masking": not args.no_loss_mask,
                    "socratic": args.socratic,
                    "ewc": args.ewc, "deep": args.deep, "lora": args.lora,
                },
            )
            print(f"  W&B: enabled (project={args.wandb_project})")
        except ImportError:
            print("  [!] W&B requested but not installed. pip install wandb")

    # ── Dry-run: show config and exit ──
    import torch
    if args.dry_run:
        steps = args.steps or 30000
        batch = args.batch or 128
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None"
        sps = 200 if has_gpu else 0.9
        est_min = steps / sps / 60
        print(f"\n{'='*50}")
        print(f"  DRY RUN — {args.op}")
        print(f"{'='*50}")
        print(f"  Steps:   {steps}")
        print(f"  Batch:   {batch}")
        print(f"  Device:  {'GPU (' + gpu_name + ')' if has_gpu else 'CPU'}")
        print(f"  AMP:     {'ON' if args.amp else 'OFF'}")
        print(f"  Compile: {'ON' if args.compile else 'OFF'}")
        print(f"  MoE:     {'ON (config.use_moe)' if hasattr(args, 'moe') and args.moe else 'OFF'}")
        print(f"  EWC:     {'ON (lambda=' + str(args.ewc_lambda) + ')' if args.ewc else 'OFF'}")
        print(f"  Socratic: {'ON (steps=' + str(args.socratic_steps) + ')' if args.socratic else 'OFF'}")
        print(f"  LoRA:    {'ON (rank=' + str(args.lora_rank) + ')' if args.lora else 'OFF'}")
        print(f"  WandB:   {'ON' if args.wandb else 'OFF'}")
        print(f"  Est. time: {est_min:.0f} min ({est_min/60:.1f} hours)")
        print(f"{'='*50}")
        print(f"  To train: python3 train_specialist.py {args.op} --steps {steps}")
        sys.exit(0)

    train_specialist(
        args.op,
        steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        use_reversed=None if not args.no_reversed else False,
        use_loss_masking=None if not args.no_loss_mask else False,
        quick=args.quick,
        resume=args.resume,
        test_hard=args.test_hard,
        deep=args.deep,
        use_ewc=args.ewc,
        ewc_lambda=args.ewc_lambda,
        ewc_gamma=args.ewc_gamma,
        use_amp=args.amp,
        gradient_accumulation_steps=args.grad_accum,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        socratic=args.socratic,
        socratic_steps=args.socratic_steps,
        socratic_problems=args.socratic_problems,
    )
