#!/usr/bin/env python3
"""math_benchmarks.py — Standardized math benchmarks for Tabula Rasa specialist models.

Evaluates a trained specialist on addition (3 difficulty levels),
subtraction, and word problems. Reports per-category accuracy,
total accuracy, and per-digit breakdowns.

Usage:
    python3 scripts/math_benchmarks.py --checkpoint specialists/math/add/best.pt
    python3 scripts/math_benchmarks.py --checkpoint specialists/math/general/best.pt --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Ensure src/ is on the Python path
_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

# ═══════════════════════════════════════════════════════════════════
# Constants & Helpers
# ═══════════════════════════════════════════════════════════════════

SEED = 42


@dataclass
class BenchmarkResult:
    """Stores results for one benchmark category."""

    name: str
    correct: int = 0
    total: int = 0
    per_digit_correct: dict[int, int] = field(default_factory=dict)
    per_digit_total: dict[int, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total

    def per_digit_accuracy(self, digit: int) -> float:
        total = self.per_digit_total.get(digit, 0)
        if total == 0:
            return float("nan")
        return 100.0 * self.per_digit_correct.get(digit, 0) / total


# ═══════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════


def _infer_config_from_state(state: dict[str, Any], tok: MathTokenizer) -> Config:
    """Infer architecture hyperparameters from a checkpoint state_dict.

    Scans the keys to determine d_model, n_layers, n_heads, d_ff, etc.
    Falls back to default 1M preset if inference fails.
    """
    cfg = Config()
    cfg.vocab_size = tok.vocab_size

    # Collect all weight shapes
    shapes: dict[str, tuple[int, ...]] = {}
    for key, tensor in state.items():
        if isinstance(tensor, torch.Tensor):
            shapes[key] = tuple(tensor.shape)

    # Try to infer d_model from token_embedding or first layer norm
    d_model = None
    if "token_embedding.weight" in shapes:
        d_model = shapes["token_embedding.weight"][1]
    elif "layers.0.attention_norm.weight" in shapes:
        d_model = shapes["layers.0.attention_norm.weight"][0]
    elif "norm.weight" in shapes:
        d_model = shapes["norm.weight"][0]

    if d_model is not None:
        cfg.d_model = d_model

    # Infer n_layers from the number of layer blocks
    n_layers = 0
    for key in shapes:
        if key.startswith("layers.") and ".attention.wq.weight" in key:
            parts = key.split(".")
            try:
                idx = int(parts[1])
                n_layers = max(n_layers, idx + 1)
            except (ValueError, IndexError):
                pass
    if n_layers > 0:
        cfg.n_layers = n_layers

    # Infer n_heads from wq weight shape
    if "layers.0.attention.wq.weight" in shapes:
        wq_shape = shapes["layers.0.attention.wq.weight"]
        if d_model and wq_shape[0] % d_model == 0:
            n_heads = wq_shape[0] // d_model
            cfg.n_heads = n_heads

    # Infer d_ff from feed-forward weights
    if "layers.0.feed_forward.w1.weight" in shapes:
        cfg.d_ff = shapes["layers.0.feed_forward.w1.weight"][0]
    elif "layers.0.feed_forward.w2.weight" in shapes:
        cfg.d_ff = shapes["layers.0.feed_forward.w2.weight"][1]

    # Detect pos_encoding from presence of rope.inv_freq
    if any("rope.inv_freq" in k for k in shapes):
        cfg.pos_encoding = "rope"
    elif any("pos_embed.pe.weight" in k for k in shapes):
        cfg.pos_encoding = "learned"

    # Detect MoE
    if any("feed_forward.router.weight" in k for k in shapes):
        cfg.use_moe = True

    # Detect value head
    if any(k.startswith("value_head.") for k in shapes):
        cfg.use_value_head = True

    # Infer max_seq_len from token_embedding or positions
    if "token_embedding.weight" in shapes:
        cfg.max_seq_len = max(cfg.max_seq_len, 32)

    return cfg


def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[MathTransformer, MathTokenizer, Config]:
    """Load a trained model and tokenizer from a checkpoint.

    Auto-detects architecture from the checkpoint state_dict.

    Args:
        checkpoint_path: Path to the .pt checkpoint file. The corresponding
            tokenizer JSON is expected at ``{checkpoint_dir}/tokenizer.json``.
        device: Device to load the model onto ('cpu' or 'cuda').

    Returns:
        Tuple ``(model, tokenizer, config)`` with the model in eval mode.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        print(f"  [ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    tok_path = ckpt.parent / "tokenizer.json"
    if not tok_path.exists():
        print(f"  [ERROR] Tokenizer not found at {tok_path}")
        sys.exit(1)

    print(f"  Loading tokenizer from {tok_path}")
    # Use Path().read_text() with encoding='utf-8' for Windows compatibility
    tokenizer_data = Path(tok_path).read_text(encoding="utf-8")
    tok = MathTokenizer.__new__(MathTokenizer)
    data = json.loads(tokenizer_data)
    tok.stoi = data["stoi"]
    tok.itos = {int(k): v for k, v in data["itos"].items()}
    tok.vocab_size = len(tok.stoi)
    tok.pad_id = tok.stoi["<PAD>"]
    tok.bos_id = tok.stoi["<BOS>"]
    tok.eos_id = tok.stoi["<EOS>"]
    tok.unk_id = tok.stoi["<UNK>"]
    tok.step_id = tok.stoi.get("<STEP>", tok.eos_id)
    tok.end_id = tok.stoi.get("<END>", tok.eos_id)
    tok._tokens_sorted = sorted(tok.stoi.keys(), key=len, reverse=True)

    print(f"  Loading checkpoint from {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if "model_state_dict" in state:
        model_state = state["model_state_dict"]
    elif "state_dict" in state:
        model_state = state["state_dict"]
    else:
        # Assume the entire state dict is the model
        model_state = state

    # Infer config from state dict
    cfg = _infer_config_from_state(model_state, tok)
    cfg.vocab_size = tok.vocab_size

    print("\n  Inferred architecture:")
    print(f"    d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"    pos_encoding={cfg.pos_encoding}, max_seq_len={cfg.max_seq_len}")
    print(f"    vocab_size={cfg.vocab_size}")

    # Build model and load weights
    model = MathTransformer(cfg)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    model.to(device)

    print(f"  Model parameters: {count_parameters(model):,}")
    print()

    return model, tok, cfg


# ═══════════════════════════════════════════════════════════════════
# Problem Generation
# ═══════════════════════════════════════════════════════════════════


def gen_level1(num: int = 100) -> list[tuple[str, int, int, int]]:
    """Single-digit addition: 0-9 + 0-9 (no carries for single-digit)."""
    rng = random.Random(SEED)
    problems: list[tuple[str, int, int, int]] = []
    for _ in range(num):
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        ans = a + b
        problems.append((f"{a}+{b}", a, b, ans))
    return problems


def gen_level2(num: int = 100) -> list[tuple[str, int, int, int]]:
    """Two-digit addition without carry.

    Both operands are 10-99. The ones digits sum < 10 and tens digits sum < 10,
    so no carry occurs.
    """
    rng = random.Random(SEED + 1)
    problems: list[tuple[str, int, int, int]] = []
    attempts = 0
    while len(problems) < num and attempts < num * 20:
        attempts += 1
        a_tens = rng.randint(1, 9)
        a_ones = rng.randint(0, 9)
        b_tens = rng.randint(1, 9)
        b_ones = rng.randint(0, 9)
        # No carry condition: ones + ones < 10 AND tens + tens < 10
        if a_ones + b_ones >= 10 or a_tens + b_tens >= 10:
            continue
        a = a_tens * 10 + a_ones
        b = b_tens * 10 + b_ones
        ans = a + b
        problems.append((f"{a}+{b}", a, b, ans))
    return problems


def gen_level3(num: int = 100) -> list[tuple[str, int, int, int]]:
    """Two-digit addition WITH carry.

    Forces a carry in either the ones or tens column (or both).
    """
    rng = random.Random(SEED + 2)
    problems: list[tuple[str, int, int, int]] = []
    attempts = 0
    while len(problems) < num and attempts < num * 20:
        attempts += 1
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        # Check if any column produces a carry
        a_str = str(a).zfill(2)
        b_str = str(b).zfill(2)
        ones_sum = int(a_str[1]) + int(b_str[1])
        tens_sum = int(a_str[0]) + int(b_str[0]) + (1 if ones_sum >= 10 else 0)
        if ones_sum < 10 and tens_sum < 10:
            continue  # No carry at all, skip
        ans = a + b
        problems.append((f"{a}+{b}", a, b, ans))
    return problems


def gen_subtraction(num: int = 50) -> list[tuple[str, int, int, int]]:
    """1-digit subtraction with positive result (a >= b)."""
    rng = random.Random(SEED + 3)
    problems: list[tuple[str, int, int, int]] = []
    for _ in range(num):
        a = rng.randint(1, 9)
        b = rng.randint(0, a)  # b <= a ensures positive result
        ans = a - b
        problems.append((f"{a}-{b}", a, b, ans))
    return problems


def gen_word_problems(num: int = 10) -> list[tuple[str, str, int]]:
    """Simple GSM8K-style word problems requiring basic arithmetic.

    Returns list of (prompt, expected_answer_str, expected_value).
    """
    rng = random.Random(SEED + 4)
    templates = [
        # Addition templates
        lambda r: (
            f"Tom has {r.randint(2,9)} apples. He gets {r.randint(1,5)} more. "
            f"How many apples does he have?",
            str(r.randint(2,9) + r.randint(1,5)),
        ),
        lambda r: (
            f"There are {r.randint(3,8)} birds on a tree. "
            f"{r.randint(2,5)} more birds fly to the tree. "
            f"How many birds are on the tree now?",
            str(r.randint(3,8) + r.randint(2,5)),
        ),
        lambda r: (
            f"Sara has {r.randint(5,10)} candies. Her friend gives her "
            f"{r.randint(2,4)} more. How many candies does she have?",
            str(r.randint(5,10) + r.randint(2,4)),
        ),
        # Subtraction templates
        lambda r: (
            f"Jake has {r.randint(8,15)} crayons. He gives "
            f"{r.randint(2,5)} to his sister. How many crayons does he have left?",
            str(r.randint(8,15) - r.randint(2,5)),
        ),
        lambda r: (
            f"A farmer has {r.randint(10,20)} eggs. He sells "
            f"{r.randint(3,8)} eggs at the market. How many eggs are left?",
            str(r.randint(10,20) - r.randint(3,8)),
        ),
        # Two-step (simple)
        lambda r: (
            f"Lisa picks {r.randint(2,5)} flowers on Monday and "
            f"{r.randint(3,6)} on Tuesday. How many flowers does she pick in total?",
            str(r.randint(2,5) + r.randint(3,6)),
        ),
        lambda r: (
            f"A class has {r.randint(15,25)} students. "
            f"{r.randint(5,10)} students wear glasses. "
            f"How many students do NOT wear glasses?",
            str(r.randint(15,25) - r.randint(5,10)),
        ),
        # Multiplication template (simple)
        lambda r: (
            f"There are {r.randint(2,5)} bags. Each bag has "
            f"{r.randint(2,5)} cookies. How many cookies are there in total?",
            str(r.randint(2,5) * r.randint(2,5)),
        ),
        lambda r: (
            f"A bike has {r.randint(2,3)} wheels. There are "
            f"{r.randint(3,6)} bikes. How many wheels are there in total?",
            str(r.randint(2,3) * r.randint(3,6)),
        ),
        lambda r: (
            f"Each box holds {r.randint(3,6)} pencils. There are "
            f"{r.randint(2,4)} boxes. How many pencils are there?",
            str(r.randint(3,6) * r.randint(2,4)),
        ),
    ]

    problems: list[tuple[str, str, int]] = []
    for i in range(num):
        prompt, expected = templates[i % len(templates)](rng)
        problems.append((prompt, expected, int(expected)))
    return problems


# ═══════════════════════════════════════════════════════════════════
# Output Parsing — uses math_parser for robust extraction
# ═══════════════════════════════════════════════════════════════════


def _extract_answer_from_output(output: str, expr_prompt: str) -> str | None:
    """Extract the predicted answer from the model's generated output.

    Uses the project's own math parser (verify_equation / verify_scratchpad)
    to robustly extract the answer, handling both plain and scratchpad formats.

    Args:
        output: The full generated output string (may include the prompt).
        expr_prompt: The original expression prompt (e.g. '12+34').

    Returns:
        Predicted answer as a string, or None if extraction fails.
    """
    from tabula_rasa.math_parser import verify_equation, verify_scratchpad

    text = output.strip()

    # Remove the prompt prefix if present
    if text.startswith(expr_prompt):
        text = text[len(expr_prompt):].strip()

    if "=" not in text:
        # No '=' found — maybe model output is scratchpad or bare answer
        digits = "".join(c for c in text if c.isdigit())
        if not digits:
            return None
        # Heuristic: if the digit string clearly looks like scratchpad
        # (all 'carry' positions are 0 or 1), treat as scratchpad
        if len(digits) >= 4 and len(digits) % 2 == 0:
            # Check if first char of each pair is 0 or 1 (carry digit)
            carries = [digits[i] for i in range(0, len(digits), 2)]
            if all(c in "01" for c in carries):
                # Scratchpad format: LSD-first, extract digits and reverse
                pair_digits = [digits[i + 1] for i in range(0, len(digits), 2)]
                return "".join(reversed(pair_digits))
        # Plain answer — return as-is
        return digits

    after_eq = text.split("=")[-1].strip()

    if not after_eq:
        return None

    # Build the full equation string for verification
    # The expr_prompt may look like "12+34" — use it to build a verifiable equation
    eq_base = expr_prompt.rstrip("=")

    # Try plain equation: 12+34=46
    full_eq = f"{eq_base}={after_eq}"
    v = verify_equation(full_eq)
    if v.get("valid"):
        return v["got"]

    # Try scratchpad format: 12+34=0406
    v = verify_scratchpad(full_eq)
    if v.get("valid"):
        # Extract answer from scratchpad — digits from the pairs
        # In reversed training mode, scratchpad is LSD-first:
        # pairs are (carry, digit) with first pair = LSD of answer
        scratch = after_eq
        if scratch.isdigit() and len(scratch) % 2 == 0:
            # Digits from pairs (second char of each pair)
            digits = [scratch[i + 1] for i in range(0, len(scratch), 2)]
            # LSD-first → reverse for human-readable
            return "".join(reversed(digits))
        return after_eq

    # Try with the raw output (model may output reversed operands: 21+43=0406)
    # Split on '=' more carefully
    parts = text.split("=")
    if len(parts) >= 2:
        last_part = parts[-1].strip()
        digits_only = "".join(c for c in last_part if c.isdigit())
        if digits_only:
            # Heuristic: if it looks like scratchpad (carry chars are 0/1), parse it
            if len(digits_only) >= 4 and len(digits_only) % 2 == 0:
                carries = [digits_only[i] for i in range(0, len(digits_only), 2)]
                if all(c in "01" for c in carries):
                    pair_digits = [digits_only[i + 1] for i in range(0, len(digits_only), 2)]
                    return "".join(reversed(pair_digits))
            return digits_only

    # Fallback: strip non-digits
    stripped = "".join(c for c in after_eq if c.isdigit())
    return stripped if stripped else None


def _compute_per_digit(
    expected: int, predicted: int,
) -> dict[int, tuple[bool, bool]]:
    """Compute per-digit correctness for expected vs predicted answer.

    Returns dict mapping digit_position (0=ones, 1=tens, ...) to
    (expected_correct, predicted_correct).
    """
    exp_str = str(expected)
    pred_str = str(predicted)

    max_len = max(len(exp_str), len(pred_str))
    exp_padded = exp_str.zfill(max_len)
    pred_padded = pred_str.zfill(max_len)

    results: dict[int, tuple[bool, bool]] = {}
    for i in range(max_len):
        pos = max_len - 1 - i  # 0=ones, 1=tens, ...
        exp_digit = int(exp_padded[i])
        pred_digit = int(pred_padded[i])
        results[pos] = (exp_digit == pred_digit, True)
    return results


# ═══════════════════════════════════════════════════════════════════
# Word Problem Parsing
# ═══════════════════════════════════════════════════════════════════


def _extract_number_from_text(text: str) -> int | None:
    """Extract the last number from generated text."""
    # Strip prompt and split
    import re
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


# ═══════════════════════════════════════════════════════════════════
# Test Runners
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_addition_problems(
    model: MathTransformer,
    tok: MathTokenizer,
    problems: list[tuple[str, int, int, int]],
    max_new_tokens: int = 20,
) -> BenchmarkResult:
    """Evaluate model on a list of addition problems.

    Args:
        model: The trained transformer model.
        tok: The tokenizer.
        problems: List of (expr, a, b, expected_answer).
        max_new_tokens: Max tokens to generate.

    Returns:
        BenchmarkResult with accuracy and per-digit breakdown.
    """
    result = BenchmarkResult(name="addition")
    result.total = len(problems)

    for expr, a, b, expected in problems:
        prompt = f"{expr}="
        generated = model.generate(
            tok, prompt, max_new_tokens=max_new_tokens,
            temperature=0.0, top_k=5,
        )

        predicted_str = _extract_answer_from_output(generated, prompt)
        expected_str = str(expected)

        if predicted_str is not None and predicted_str == expected_str:
            result.correct += 1
        else:
            # Truncate long outputs for display
            display = generated[:60] + "..." if len(generated) > 60 else generated
            result.failures.append(
                f"  FAIL: {expr}={expected} | pred='{predicted_str}' "
                f"| raw={repr(display)}"
            )

        # Per-digit breakdown (even on wrong answers)
        if predicted_str is not None and predicted_str.isdigit():
            try:
                predicted_int = int(predicted_str)
                digit_info = _compute_per_digit(expected, predicted_int)
                for pos, (correct_flag, _) in digit_info.items():
                    result.per_digit_total[pos] = result.per_digit_total.get(pos, 0) + 1
                    if correct_flag:
                        result.per_digit_correct[pos] = result.per_digit_correct.get(pos, 0) + 1
            except ValueError:
                pass
        else:
            # Count all expected digits as wrong
            for pos in range(len(expected_str)):
                result.per_digit_total[pos] = result.per_digit_total.get(pos, 0) + 1

    return result


@torch.no_grad()
def evaluate_subtraction_problems(
    model: MathTransformer,
    tok: MathTokenizer,
    problems: list[tuple[str, int, int, int]],
    max_new_tokens: int = 20,
) -> BenchmarkResult:
    """Evaluate model on subtraction problems."""
    result = BenchmarkResult(name="subtraction")
    result.total = len(problems)

    for expr, a, b, expected in problems:
        prompt = f"{expr}="
        generated = model.generate(
            tok, prompt, max_new_tokens=max_new_tokens,
            temperature=0.0, top_k=5,
        )

        predicted_str = _extract_answer_from_output(generated, prompt)
        expected_str = str(expected)

        if predicted_str is not None and predicted_str == expected_str:
            result.correct += 1
        else:
            display = generated[:60] + "..." if len(generated) > 60 else generated
            result.failures.append(
                f"  FAIL: {expr}={expected} | pred='{predicted_str}' "
                f"| raw={repr(display)}"
            )

        if predicted_str is not None and predicted_str.isdigit():
            try:
                predicted_int = int(predicted_str)
                digit_info = _compute_per_digit(expected, predicted_int)
                for pos, (correct_flag, _) in digit_info.items():
                    result.per_digit_total[pos] = result.per_digit_total.get(pos, 0) + 1
                    if correct_flag:
                        result.per_digit_correct[pos] = result.per_digit_correct.get(pos, 0) + 1
            except ValueError:
                pass
        else:
            for pos in range(len(expected_str)):
                result.per_digit_total[pos] = result.per_digit_total.get(pos, 0) + 1

    return result


@torch.no_grad()
def evaluate_word_problems(
    model: MathTransformer,
    tok: MathTokenizer,
    problems: list[tuple[str, str, int]],
    max_new_tokens: int = 30,
) -> BenchmarkResult:
    """Evaluate model on word problems.

    For word problems, we prompt the model with the question and then parse
    the generated text for the answer number.

    Args:
        model: The trained transformer model.
        tok: The tokenizer.
        problems: List of (prompt, expected_str, expected_int).
        max_new_tokens: Max tokens to generate.

    Returns:
        BenchmarkResult with accuracy.
    """
    result = BenchmarkResult(name="word_problems")
    result.total = len(problems)

    for prompt_text, expected_str, expected_int in problems:
        generated = model.generate(
            tok, prompt_text, max_new_tokens=max_new_tokens,
            temperature=0.0, top_k=5,
        )

        # Try to extract a number from the generated response
        predicted_int = _extract_number_from_text(generated)
        predicted_str = str(predicted_int) if predicted_int is not None else None

        if predicted_str is not None and predicted_str == expected_str:
            result.correct += 1
        else:
            display = generated[:80] + "..." if len(generated) > 80 else generated
            result.failures.append(
                f"  FAIL: '{prompt_text[:40]}...' | expected={expected_str} "
                f"| pred='{predicted_str}' | raw={repr(display)}"
            )

    return result


# ═══════════════════════════════════════════════════════════════════
# Results Printing
# ═══════════════════════════════════════════════════════════════════


def _print_separator(char: str = "=", width: int = 70) -> None:
    print(char * width)


def _print_category_result(result: BenchmarkResult) -> None:
    """Print a formatted result table for one category."""
    acc = result.accuracy
    status = "PASS" if acc >= 50.0 else "FAIL"
    color = "" if acc >= 50.0 else "[LOW] "

    print(f"  {result.name}:")
    print(f"    Accuracy:  {color}{acc:5.1f}%  ({result.correct}/{result.total})  [{status}]")

    if result.per_digit_total:
        print("    Per-digit breakdown:")
        for pos in sorted(result.per_digit_total.keys()):
            total = result.per_digit_total[pos]
            correct = result.per_digit_correct.get(pos, 0)
            pct = 100.0 * correct / total if total > 0 else 0.0
            place = {0: "ones", 1: "tens", 2: "hundreds", 3: "thousands"}.get(pos, f"pos={pos}")
            print(f"      {place:>10}: {pct:5.1f}%  ({correct}/{total})")

    if result.failures:
        max_show = min(5, len(result.failures))
        print(f"    Sample failures ({max_show}/{len(result.failures)} shown):")
        for f in result.failures[:max_show]:
            print(f"      {f}")

    print()


def _print_summary(results: list[BenchmarkResult]) -> bool:
    """Print summary table and return True if all categories pass (>=50%)."""
    _print_separator()
    print("  SUMMARY")
    _print_separator("-")

    all_pass = True
    total_correct = 0
    total_problems = 0

    # Table header
    print(f"  {'Category':<30} {'Accuracy':<12} {'Correct':<12} {'Status'}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")

    for r in results:
        if r.total == 0:
            continue
        acc = r.accuracy
        passes = acc >= 50.0
        if not passes:
            all_pass = False
        total_correct += r.correct
        total_problems += r.total
        status = "PASS" if passes else "FAIL (<50%)"
        print(f"  {r.name:<30} {acc:>5.1f}%{'':<6} "
              f"{r.correct}/{r.total:<5} {status}")

    _print_separator("-")
    overall = 100.0 * total_correct / total_problems if total_problems > 0 else 0.0
    print(f"  {'OVERALL':<30} {overall:>5.1f}%{'':<6} "
          f"{total_correct}/{total_problems}")
    _print_separator()

    if not all_pass:
        print("  [!] Some categories below 50% threshold")
    else:
        print("  [✓] All categories pass (>=50%)")

    return all_pass


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardized math benchmarks for Tabula Rasa specialist models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/math_benchmarks.py --checkpoint specialists/math/add/best.pt
  python3 scripts/math_benchmarks.py --checkpoint specialists/math/general/best.pt --all
        """,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="specialists/math/add/best.pt",
        help="Path to model checkpoint (default: specialists/math/add/best.pt)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run ALL benchmark categories (addition L1/L2/L3, subtraction, word problems)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum new tokens to generate per problem (default: 30)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override sample count for all categories (default: category-specific)",
    )

    args = parser.parse_args()

    # ── Header ──────────────────────────────────────────────────
    _print_separator()
    print("  Tabula Rasa — Math Benchmarks")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print(f"  Generated:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print_separator()

    # ── Load model ──────────────────────────────────────────────
    model, tok, cfg = load_model(args.checkpoint, device=args.device)
    max_tokens = args.max_tokens

    # ── Run benchmarks ──────────────────────────────────────────
    results: list[BenchmarkResult] = []

    if args.all:
        # Level 1: Single-digit addition
        print("  Generating level 1 problems (single-digit addition)...")
        n = args.samples or 100
        probs_l1 = gen_level1(n)
        r1 = evaluate_addition_problems(model, tok, probs_l1, max_tokens)
        results.append(r1)
        _print_category_result(r1)

        # Level 2: Two-digit addition without carry
        print("  Generating level 2 problems (2-digit, no carry)...")
        n = args.samples or 100
        probs_l2 = gen_level2(n)
        r2 = evaluate_addition_problems(model, tok, probs_l2, max_tokens)
        results.append(r2)
        _print_category_result(r2)

        # Level 3: Two-digit addition with carry
        print("  Generating level 3 problems (2-digit, with carry)...")
        n = args.samples or 100
        probs_l3 = gen_level3(n)
        r3 = evaluate_addition_problems(model, tok, probs_l3, max_tokens)
        results.append(r3)
        _print_category_result(r3)

        # Subtraction
        print("  Generating subtraction problems (1-digit)...")
        n = args.samples or 50
        probs_sub = gen_subtraction(n)
        r4 = evaluate_subtraction_problems(model, tok, probs_sub, max_tokens)
        results.append(r4)
        _print_category_result(r4)

        # Word problems
        print("  Generating word problems...")
        n = args.samples or 10
        probs_wp = gen_word_problems(n)
        r5 = evaluate_word_problems(model, tok, probs_wp, max_tokens + 10)
        results.append(r5)
        _print_category_result(r5)

    else:
        # Default: just run addition level 1
        print("  Generating addition problems (single-digit)...")
        n = args.samples or 100
        probs_l1 = gen_level1(n)
        r1 = evaluate_addition_problems(model, tok, probs_l1, max_tokens)
        results.append(r1)
        _print_category_result(r1)

    # ── Summary ─────────────────────────────────────────────────
    all_pass = _print_summary(results)

    # Exit code: 1 if any category below 50%
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
