"""Comprehensive benchmark suite for Tabula-Rasa models.

Provides multi-dimensional evaluation: raw accuracy, per-position (digit-column)
accuracy, out-of-distribution generalization, compositional reasoning,
scratchpad correctness, and robustness to perturbation.

Usage:
    python3 -m tabula_rasa.eval checkpoints/best.pt
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tabula_rasa.config import Config
from tabula_rasa.dataset import generate_problem
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer


# ═══════════════════════════════════════════════════════════════════════
#  Results containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """Per-dimension evaluation result."""

    name: str
    accuracy: float
    num_problems: int
    details: list[dict] = field(default_factory=list)  # per-problem breakdowns


@dataclass
class PerPositionResult:
    """Accuracy broken down by digit column (units=col 0, tens=col 1, ...)."""

    name: str
    overall_accuracy: float
    position_accuracy: dict[int, float]  # col_idx -> accuracy
    carry_accuracy: float | None = None  # how often carries correct when needed
    num_problems: int = 0


# ═══════════════════════════════════════════════════════════════════════
#  Problem generators
# ═══════════════════════════════════════════════════════════════════════


def _generate_carry_problem(max_digits: int = 4) -> tuple[str, str]:
    """Generate an addition problem that *requires* carry propagation.

    E.g. 56+78 → cross-column carry (units 6+8≥10, tens 5+7+1≥10).
    """
    force_carry = True
    while True:
        a_digits = random.randint(2, max_digits)
        b_digits = random.randint(2, max_digits)
        a = random.randint(10 ** (a_digits - 1), 10**a_digits - 1)
        b = random.randint(10 ** (b_digits - 1), 10**b_digits - 1)
        ans = a + b
        str_ans = str(ans)
        # Check at least one column produces a carry
        a_s = str(a).zfill(max(len(str(a)), len(str(b))))
        b_s = str(b).zfill(len(a_s))
        carries = False
        for da, db in zip(reversed(a_s), reversed(b_s)):
            if int(da) + int(db) >= 10:
                carries = True
                break
        if carries or not force_carry:
            return f"{a}+{b}", str_ans


def _generate_ood_problem(
    train_max_digits: int, ood_digits: int
) -> tuple[str, str]:
    """Generate a problem at OOD digit length."""
    return generate_problem(train_max_digits + 1, ood_digits)


def _generate_compositional(
    depth: int = 2, max_digits: int = 2
) -> tuple[str, str]:
    """Generate a chained expression like (12+34)+(56-78)."""
    # Build sub-expressions
    sub_results = []
    sub_strs = []
    for _ in range(depth):
        expr, ans = generate_problem(1, max_digits)
        sub_strs.append(f"({expr})")
        sub_results.append(int(ans))
    # Combine
    full_expr = "+".join(sub_strs)
    total = sum(sub_results)
    return full_expr, str(total)


def _generate_robustness_pairs(
    base_expr: str, base_ans: str, num_variants: int = 3
) -> list[tuple[str, str, str]]:
    """Generate perturbed variants of a problem.

    Returns list of (variant_expr, expected_ans, perturbation_type).
    """
    pairs: list[tuple[str, str, str]] = []
    # Extract operands
    m = re.match(r"(\d+)([+\-*/])(\d+)", base_expr)
    if not m:
        return pairs
    a_str, op, b_str = m.group(1), m.group(2), m.group(3)
    a, b = int(a_str), int(b_str)

    # 1. Digit reversal (123 -> 321)
    rev_a = int(a_str[::-1])
    rev_b = int(b_str[::-1])
    if op == "+":
        pairs.append((f"{rev_a}+{rev_b}", str(rev_a + rev_b), "digit_reversal"))
    elif op == "-":
        if rev_a >= rev_b:
            pairs.append((f"{rev_a}-{rev_b}", str(rev_a - rev_b), "digit_reversal"))

    # 2. Add noise tokens (insert spaces)
    noisy = f"{a_str} + {b_str}"
    if op == "+":
        pairs.append((noisy, str(a + b), "spaced"))

    # 3. Leading zeros
    if len(a_str) > 1:
        leading = f"0{a_str}{op}{b_str}"
        expected = str(a + b) if op == "+" else str(a - b) if a >= b else str(b - a)
        if op in ("+", "-"):
            pairs.append((leading, expected, "leading_zeros"))
    return pairs[:num_variants]


# ═══════════════════════════════════════════════════════════════════════
#  Eval harness
# ═══════════════════════════════════════════════════════════════════════


def _solve(
    model: MathTransformer, tok: MathTokenizer, expr: str, max_new: int = 16
) -> str:
    """Run model on a single expression, return predicted answer string."""
    prompt = f"{expr}=" if "=" not in expr else expr
    generated = model.generate(
        tok, prompt, max_new_tokens=max_new, temperature=0.0, top_k=1
    )
    if "=" in generated:
        pred = generated.split("=")[-1].strip()
        pred = "".join(c for c in pred if c.isdigit() or c == "-")
        return pred
    return ""


def _check(pred: str, expected: str) -> bool:
    """Check if prediction matches expected."""
    return pred == expected


def _make_carry_template(expr: str, ans: str) -> dict:
    """Decompose a + expression into digit columns for carry analysis."""
    m = re.match(r"(\d+)\+(\d+)", expr)
    if not m:
        return {}
    a_s, b_s = m.group(1), m.group(2)
    ans_s = ans
    max_len = max(len(a_s), len(b_s), len(ans_s))
    a_s = a_s.zfill(max_len)
    b_s = b_s.zfill(max_len)
    ans_s = ans_s.zfill(max_len)

    columns = []
    carry_in = 0
    for col in range(max_len - 1, -1, -1):
        da, db = int(a_s[col]), int(b_s[col])
        digit_sum = da + db + carry_in
        expected_digit = digit_sum % 10
        expected_carry = 1 if digit_sum >= 10 else 0
        actual_ans_digit = int(ans_s[col])
        columns.append(
            {
                "col": max_len - 1 - col,  # 0=units, 1=tens, ...
                "a_digit": da,
                "b_digit": db,
                "carry_in": carry_in,
                "digit_sum": digit_sum,
                "expected_digit": expected_digit,
                "expected_carry": expected_carry,
                "actual_ans_digit": actual_ans_digit,
            }
        )
        carry_in = expected_carry
    return {"columns": columns}


# ═══════════════════════════════════════════════════════════════════════
#  Individual benchmarks
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_accuracy(
    model: MathTransformer,
    tok: MathTokenizer,
    num_problems: int = 200,
    max_digits: int = 4,
    verbose: bool = True,
    op: str | None = None,
) -> EvalResult:
    """Evaluate raw accuracy on random arithmetic problems.

    Args:
        model: Trained transformer.
        tok: Tokenizer.
        num_problems: Sample size.
        max_digits: Maximum operand digits.
        verbose: Print failure examples.
        op: If set (e.g. '+', '-'), only test that operation.

    Returns:
        EvalResult with accuracy and per-problem breakdown.
    """
    correct = 0
    details = []

    for i in range(num_problems):
        expr, ans = generate_problem(1, max_digits)
        if op and expr.split("+")[0] if "+" in expr else expr.split("-")[0]:
            # Re-roll if not matching requested op
            while True:
                expr, ans = generate_problem(1, max_digits)
                if any(c in expr for c in op or "+-*/"):
                    # Simple check: last char before digit = operator
                    op_in_expr = re.search(r"[+\-*/]", expr)
                    if op_in_expr and op_in_expr.group() == op:
                        break

        pred = _solve(model, tok, expr)
        is_correct = pred == ans
        if is_correct:
            correct += 1
        details.append(
            {"expr": expr, "expected": ans, "predicted": pred, "correct": is_correct}
        )
        if verbose and not is_correct and i < 15:
            print(f"  FAIL: {expr}={ans} | pred: {pred}")

    acc = correct / num_problems * 100
    return EvalResult(
        name=f"Random {max_digits}-digit",
        accuracy=acc,
        num_problems=num_problems,
        details=details,
    )


@torch.no_grad()
def evaluate_per_position(
    model: MathTransformer,
    tok: MathTokenizer,
    num_problems: int = 200,
    max_digits: int = 4,
    verbose: bool = True,
) -> PerPositionResult:
    """Evaluate accuracy per digit column on addition problems only.

    Reports:
        - Overall accuracy
        - Accuracy at each digit column (units, tens, hundreds, ...)
        - Carry accuracy: how often the carry digit is correct in columns
          where a carry is expected
    """
    # Generate addition-only problems with carries
    add_problems: list[tuple[str, str]] = []
    for _ in range(num_problems):
        # Mix carry and non-carry
        if random.random() < 0.4:
            expr, ans = _generate_carry_problem(max_digits)
        else:
            while True:
                expr, ans = generate_problem(1, max_digits)
                if "+" in expr:
                    break
        add_problems.append((expr, ans))

    # Per-column stats: count of columns at each position
    col_correct: dict[int, int] = {}
    col_total: dict[int, int] = {}
    carry_correct = 0
    carry_total = 0
    overall_correct = 0

    for expr, expected in add_problems:
        pred = _solve(model, tok, expr)
        if pred == expected:
            overall_correct += 1

        # Column analysis (only for addition)
        carry_info = _make_carry_template(expr, expected)
        if carry_info:
            columns = carry_info["columns"]
            # Pad prediction to same length
            pred_padded = pred.zfill(len(columns))
            for c in columns:
                col = c["col"]
                col_total[col] = col_total.get(col, 0) + 1
                pred_digit = pred_padded[col] if col < len(pred_padded) else ""
                if pred_digit.isdigit() and int(pred_digit) == c["expected_digit"]:
                    col_correct[col] = col_correct.get(col, 0) + 1
                # Carry check
                if c["expected_carry"]:
                    carry_total += 1
                    # Correct if the final next-column digit accounts for carry
                    # We check: was the predicted digit in this column correct?
                    if pred_digit.isdigit() and int(pred_digit) == c["expected_digit"]:
                        carry_correct += 1

    overall = overall_correct / len(add_problems) * 100
    pos_acc = {
        col: (col_correct.get(col, 0) / col_total[col] * 100)
        for col in sorted(col_total.keys())
    }
    carry_acc = (carry_correct / carry_total * 100) if carry_total > 0 else None

    if verbose:
        print(f"\n  Per-position accuracy ({len(add_problems)} addition problems):")
        for col, acc in sorted(pos_acc.items()):
            label = ["units", "tens", "hundreds", "thousands", "10K", "100K"][
                min(col, 5)
            ]
            print(f"    Column {col} ({label}): {acc:.1f}%")
        if carry_acc is not None:
            print(f"    Carry-digit accuracy: {carry_acc:.1f}%")

    return PerPositionResult(
        name=f"Per-position (add, {max_digits}-digit)",
        overall_accuracy=overall,
        position_accuracy=pos_acc,
        carry_accuracy=carry_acc,
        num_problems=len(add_problems),
    )


@torch.no_grad()
def evaluate_ood(
    model: MathTransformer,
    tok: MathTokenizer,
    train_max_digits: int = 4,
    num_problems: int = 100,
    verbose: bool = True,
) -> EvalResult:
    """Evaluate out-of-distribution generalization on harder problems.

    Tests the model on digit lengths beyond what it was trained on.
    """
    ood_digits = train_max_digits + 2  # 2 digits beyond training
    correct = 0
    details = []

    for _ in range(num_problems):
        expr, ans = _generate_ood_problem(train_max_digits, ood_digits)
        pred = _solve(model, tok, expr)
        is_correct = pred == ans
        if is_correct:
            correct += 1
        details.append(
            {"expr": expr, "expected": ans, "predicted": pred, "correct": is_correct}
        )
        if verbose and not is_correct and len([d for d in details if not d["correct"]]) <= 10:
            print(f"  OOD FAIL: {expr}={ans} | pred: {pred}")

    acc = correct / num_problems * 100
    return EvalResult(
        name=f"OOD ({train_max_digits+1}-{ood_digits}-digit)",
        accuracy=acc,
        num_problems=num_problems,
        details=details,
    )


@torch.no_grad()
def evaluate_compositional(
    model: MathTransformer,
    tok: MathTokenizer,
    num_problems: int = 50,
    max_digits: int = 2,
    depth: int = 2,
    verbose: bool = True,
) -> EvalResult:
    """Evaluate on multi-step chained arithmetic.

    Tests whether the model can compose multiple operations, e.g.
    (12+34)+(56-78)=
    """
    correct = 0
    details = []

    for _ in range(num_problems):
        expr, ans = _generate_compositional(depth, max_digits)
        pred = _solve(model, tok, expr, max_new=32)
        is_correct = pred == ans
        if is_correct:
            correct += 1
        details.append(
            {"expr": expr, "expected": ans, "predicted": pred, "correct": is_correct}
        )
        if verbose and not is_correct and len([d for d in details if not d["correct"]]) <= 8:
            print(f"  COMP FAIL: {expr}={ans} | pred: {pred}")

    acc = correct / num_problems * 100
    return EvalResult(
        name=f"Compositional (depth={depth}, {max_digits}-digit)",
        accuracy=acc,
        num_problems=num_problems,
        details=details,
    )


@torch.no_grad()
def evaluate_robustness(
    model: MathTransformer,
    tok: MathTokenizer,
    num_problems: int = 50,
    max_digits: int = 3,
    verbose: bool = True,
) -> EvalResult:
    """Evaluate robustness to input perturbations.

    Tests digit reversal, spacing, and leading zeros to see
    if the model relies on surface pattern matching.
    """
    correct = 0
    total = 0
    details = []

    for _ in range(num_problems):
        expr, ans = generate_problem(1, max_digits)
        variants = _generate_robustness_pairs(expr, ans)
        for v_expr, v_ans, ptype in variants:
            pred = _solve(model, tok, v_expr)
            is_correct = pred == v_ans
            if is_correct:
                correct += 1
            total += 1
            details.append(
                {
                    "expr": v_expr,
                    "expected": v_ans,
                    "predicted": pred,
                    "correct": is_correct,
                    "perturbation": ptype,
                }
            )
            if verbose and not is_correct and len([d for d in details if not d["correct"]]) <= 8:
                print(f"  ROBUST FAIL [{ptype}]: {v_expr}={v_ans} | pred: {pred}")

    acc = correct / total * 100 if total > 0 else 0.0
    return EvalResult(
        name=f"Robustness ({max_digits}-digit, {total} variants)",
        accuracy=acc,
        num_problems=total,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Full benchmark runner
# ═══════════════════════════════════════════════════════════════════════


# ── Adversarial test cases ──────────────────────────────────────────

BOUNDARY_PROBLEMS: list[tuple[str, str]] = [
    # Zero boundaries
    ("0+0", "0"), ("0+1", "1"), ("1+0", "1"), ("0-0", "0"),
    ("0*5", "0"), ("5*0", "0"), ("0/1", "0"),
    # Carry cascade
    ("9+1", "10"), ("99+1", "100"), ("999+1", "1000"), ("9999+1", "10000"),
    ("1+9", "10"), ("1+99", "100"), ("1+999", "1000"),
    # Borrow cascade
    ("10-1", "9"), ("100-1", "99"), ("1000-1", "999"), ("10000-1", "9999"),
    # All nines
    ("9+9", "18"), ("99+99", "198"), ("999+999", "1998"),
    ("99-9", "90"), ("999-99", "900"),
    # Large carry chains
    ("45+55", "100"), ("55+45", "100"),
    ("123+877", "1000"), ("876+124", "1000"),
    # Negatives / subtraction edge
    ("5-5", "0"), ("10-10", "0"),
    ("100-100", "0"), ("123-123", "0"),
]

ADVERSARIAL_FORMATS: list[tuple[str, str, str]] = [
    # (format_name, transform_fn_repr)
    ("unicode_digits", "unicode"),        # 1 2 3 → ₁₂₃
    ("hex_input", "hex"),                 # interpret as hex
    ("binary_input", "binary"),           # interpret as binary
    ("reversed", "reverse"),              # 12+34 → 43+21
    ("spaces_everywhere", "spaces"),      # 1 2 + 3 4
    ("newlines", "newlines"),             # 12\n+\n34
    ("tab_separated", "tabs"),            # 12\t+\t34
]


def _make_boundary_input(expr: str, fmt: str) -> str:
    """Apply a format transformation to an expression."""
    import re
    if fmt == "reverse":
        # Reverse digit strings while preserving operator position
        m = re.match(r"(\d+)([+\-*/])(\d+)", expr)
        if m:
            return m.group(1)[::-1] + m.group(2) + m.group(3)[::-1]
    if fmt == "spaces":
        return " ".join(expr)
    if fmt == "newlines":
        return "\n".join(expr)
    if fmt == "tabs":
        return "\t".join(expr)
    if fmt == "unicode":
        # Map digits to subscript
        sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return expr.translate(sub)
    return expr


@torch.no_grad()
def evaluate_adversarial(
    model: MathTransformer,
    tok: MathTokenizer,
    num_boundary: int = 20,
    num_noise: int = 30,
    num_fgsm: int = 20,
    verbose: bool = True,
) -> dict[str, EvalResult]:
    """Comprehensive adversarial robustness evaluation.

    Tests four categories of adversarial vulnerability:

    1. **Boundary cases** — hand-picked edge cases (carry cascade, borrow cascade, zeros)
    2. **Format attacks** — unicode, reversed, spaced, newline, tab-separated
    3. **Noise injection** — random token swaps in the prompt
    4. **Gradient-based (FGSM)** — small perturbations to input embeddings

    Args:
        model: The transformer model.
        tok: Tokenizer.
        num_boundary: Number of boundary problems to test.
        num_noise: Number of noise-injected problems to test.
        num_fgsm: Number of FGSM-attack problems to test.
        verbose: Print failure examples.

    Returns:
        Dict ``{"boundary": EvalResult, "format_attacks": EvalResult,
                 "noise": EvalResult, "fgsm": EvalResult}``
    """
    results: dict[str, EvalResult] = {}

    # ── 1. Boundary cases ──────────────────────────────────────────
    correct = 0
    details = []
    problems = BOUNDARY_PROBLEMS[:num_boundary]
    for expr, expected in problems:
        pred = _solve(model, tok, expr)
        is_correct = pred == expected
        if is_correct:
            correct += 1
        details.append({"expr": expr, "expected": expected, "predicted": pred, "correct": is_correct})
        if verbose and not is_correct and len([d for d in details if not d["correct"]]) <= 8:
            print(f"  BOUNDARY FAIL: {expr}={expected} | pred: {pred}")

    acc = correct / len(problems) * 100 if problems else 0.0
    results["boundary"] = EvalResult(
        name=f"Boundary cases ({len(problems)} problems)",
        accuracy=acc,
        num_problems=len(problems),
        details=details,
    )

    # ── 2. Format attacks ─────────────────────────────────────────
    format_correct = 0
    format_total = 0
    format_details = []
    for _ in range(num_boundary // 2):
        expr, ans = generate_problem(1, 3)
        for fmt_name, fmt_code in ADVERSARIAL_FORMATS:
            adv_expr = _make_boundary_input(expr, fmt_code)
            if adv_expr == expr:
                continue
            pred = _solve(model, tok, adv_expr)
            is_correct = pred == ans
            if is_correct:
                format_correct += 1
            format_total += 1
            format_details.append({
                "expr": adv_expr, "expected": ans, "predicted": pred,
                "correct": is_correct, "format": fmt_name,
            })
            if verbose and not is_correct and len([d for d in format_details if not d["correct"]]) <= 8:
                print(f"  FORMAT FAIL [{fmt_name}]: {adv_expr}={ans} | pred: {pred}")

    results["format_attacks"] = EvalResult(
        name=f"Format attacks ({format_total} variants)",
        accuracy=format_correct / format_total * 100 if format_total > 0 else 0.0,
        num_problems=format_total,
        details=format_details,
    )

    # ── 3. Noise injection ────────────────────────────────────────
    noise_correct = 0
    noise_total = 0
    noise_details = []
    for _ in range(num_noise):
        expr, ans = generate_problem(1, 3)
        # Swap two adjacent digits in one operand
        import random
        m = __import__('re').match(r"(\d+)([+\-*/])(\d+)", expr)
        if not m:
            continue
        a_str, op, b_str = m.group(1), m.group(2), m.group(3)
        # 50% chance: perturb operand A, 50% operand B
        target = a_str if random.random() < 0.5 else b_str
        if len(target) < 2:
            continue
        idx = random.randint(0, len(target) - 2)
        perturbed = target[:idx] + target[idx + 1] + target[idx] + target[idx + 2:]
        if target == a_str:
            noisy_expr = perturbed + op + b_str
        else:
            noisy_expr = a_str + op + perturbed
        expected = str(eval(expr))  # safe: only digits and basic ops
        pred = _solve(model, tok, noisy_expr)
        is_correct = pred == expected
        if is_correct:
            noise_correct += 1
        noise_total += 1
        noise_details.append({
            "expr": noisy_expr, "expected": expected, "predicted": pred,
            "correct": is_correct, "original": expr,
        })
        if verbose and not is_correct and len([d for d in noise_details if not d["correct"]]) <= 5:
            print(f"  NOISE FAIL: {noisy_expr}={expected} (orig: {expr}) | pred: {pred}")

    results["noise"] = EvalResult(
        name=f"Noise injection ({noise_total} variants)",
        accuracy=noise_correct / noise_total * 100 if noise_total > 0 else 0.0,
        num_problems=noise_total,
        details=noise_details,
    )

    # ── 4. FGSM-style gradient attack ──────────────────────────────
    # Perturb input embeddings in the direction of the gradient w.r.t.
    # the cross-entropy loss. If accuracy drops significantly, the model
    # is relying on surface patterns rather than true reasoning.
    fgsm_correct = 0
    fgsm_total = 0
    fgsm_details = []
    epsilon = 0.1
    device = next(model.parameters()).device

    for _ in range(num_fgsm):
        expr, ans = generate_problem(1, 3)
        prompt = f"{expr}="

        # Tokenize
        input_ids = tok.encode(prompt, add_special_tokens=True)
        if len(input_ids) > tok.max_seq_len:
            input_ids = input_ids[: tok.max_seq_len]
        inp = torch.tensor([input_ids], device=device, dtype=torch.long)

        # Target: next-token prediction of the correct answer
        target_ids = tok.encode(ans, add_special_tokens=False)
        if not target_ids:
            continue
        target = torch.tensor([target_ids[0]], device=device)

        # FGSM: get gradient of loss w.r.t. input token embeddings
        # Temporarily enable gradients, then disable after
        model.zero_grad()
        adv_embed = None
        with torch.enable_grad():
            # Detach to ensure leaf tensor for grad
            inp_embed = model.token_embedding(inp).detach().requires_grad_(True)

            # Run through layers
            h = inp_embed
            for layer in model.layers:
                h, _ = layer(h)
            h = model.norm(h)
            logits = model.lm_head(h)
            last_logits = logits[0, -1]
            loss = torch.nn.functional.cross_entropy(last_logits.unsqueeze(0), target)
            loss.backward()

            if inp_embed.grad is not None:
                grad_sign = inp_embed.grad.sign()
                adv_embed = inp_embed + epsilon * grad_sign

        if adv_embed is None:
            continue
        h_adv = adv_embed
        for layer in model.layers:
            h_adv, _ = layer(h_adv)
        h_adv = model.norm(h_adv)
        adv_logits = model.lm_head(h_adv)

        # Greedy decode
        adv_pred_ids = adv_logits[0, -1].argmax().item()
        adv_pred = tok.decode([adv_pred_ids], skip_special=True) if hasattr(tok, "decode") else str(adv_pred_ids)

        is_correct = adv_pred == ans
        if is_correct:
            fgsm_correct += 1
        fgsm_total += 1
        fgsm_details.append({
            "expr": expr, "expected": ans, "predicted": adv_pred,
            "correct": is_correct, "epsilon": epsilon,
        })
        if verbose and not is_correct and len([d for d in fgsm_details if not d["correct"]]) <= 5:
            print(f"  FGSM FAIL: {expr}={ans} | adv_pred: {adv_pred} (ε={epsilon})")

        # Clean up graph
        inp_embed.grad = None

    results["fgsm"] = EvalResult(
        name=f"FGSM attack ({fgsm_total} problems, ε={epsilon})",
        accuracy=fgsm_correct / fgsm_total * 100 if fgsm_total > 0 else 0.0,
        num_problems=fgsm_total,
        details=fgsm_details,
    )

    return results


@torch.no_grad()
def full_benchmark(
    model: MathTransformer,
    tok: MathTokenizer,
    train_max_digits: int = 4,
    verbose: bool = True,
    adversarial_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run all benchmarks and return structured results.

    This is the primary entry point for systematic evaluation.

    Args:
        model: Trained transformer.
        tok: Tokenizer.
        train_max_digits: Digit length the model was trained at.
        verbose: Print progress.

    Returns:
        Dict of benchmark_name -> EvalResult or PerPositionResult.
    """
    results: dict[str, Any] = {}

    print("=" * 60)
    print("  TABULA RASA — FULL BENCHMARK SUITE")
    print(f"  Model params: {count_parameters(model):,}")
    print(f"  Train max digits: {train_max_digits}")
    print("=" * 60)

    # Default adversarial kwargs (FGSM is slow on CPU, keep low)
    adv_kwargs = adversarial_kwargs or {"num_boundary": min(24, len(BOUNDARY_PROBLEMS)),
                                         "num_noise": 20, "num_fgsm": 10}

    # 1. Basic accuracy (same-distribution)
    print("\n[1/6] Basic accuracy...")
    results["accuracy"] = evaluate_accuracy(
        model, tok, num_problems=200, max_digits=train_max_digits, verbose=verbose
    )
    print(f"  → {results['accuracy'].accuracy:.1f}%")

    # 2. Per-position analysis (addition only)
    print("\n[2/6] Per-position analysis...")
    results["per_position"] = evaluate_per_position(
        model, tok, num_problems=200, max_digits=min(4, train_max_digits), verbose=verbose
    )
    print(f"  → Overall: {results['per_position'].overall_accuracy:.1f}%")

    # 3. OOD generalization
    print("\n[3/6] Out-of-distribution...")
    results["ood"] = evaluate_ood(
        model, tok, train_max_digits=train_max_digits, num_problems=100, verbose=verbose
    )
    print(f"  → {results['ood'].accuracy:.1f}%")

    # 4. Compositional
    print("\n[4/6] Compositional (multi-step)...")
    results["compositional"] = evaluate_compositional(
        model, tok, num_problems=50, max_digits=2, depth=2, verbose=verbose
    )
    print(f"  → {results['compositional'].accuracy:.1f}%")

    # 5. Robustness
    print("\n[5/8] Robustness (perturbations)...")
    results["robustness"] = evaluate_robustness(
        model, tok, num_problems=30, max_digits=min(3, train_max_digits), verbose=verbose
    )
    print(f"  → {results['robustness'].accuracy:.1f}%")

    # 6. Adversarial testing (boundary, format attacks, noise, FGSM)
    print("\n[6/8] Adversarial testing...")
    adv_results = evaluate_adversarial(
        model, tok, verbose=verbose, **adv_kwargs,
    )
    for k, v in adv_results.items():
        results[f"adversarial/{k}"] = v
        print(f"  → {v.name}: {v.accuracy:.1f}%")

    # 7. Scratchpad correctness (if model generates scratchpad)
    print("\n[7/8] Scratchpad analysis...")
    # Quick check: does the model produce scratchpad-like output?
    test_expr, test_ans = generate_problem(1, 3)
    test_out = model.generate(tok, f"{test_expr}=", max_new_tokens=48, temperature=0.0, top_k=1)
    has_scratchpad = "|" in test_out or "carry" in test_out.lower()
    print(f"  Scratchpad tokens detected: {has_scratchpad}")
    print(f"  Sample output: {repr(test_out[:80])}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  {'Benchmark':<40} {'Accuracy':<10}")
    print(f"  {'-'*40} {'-'*10}")
    print(f"  {'Basic accuracy':<40} {results['accuracy'].accuracy:>6.1f}%")
    print(f"  {'Per-position (overall)':<40} {results['per_position'].overall_accuracy:>6.1f}%")
    pos_strs = [
        f"col{k}={v:.0f}%"
        for k, v in sorted(results["per_position"].position_accuracy.items())
    ]
    print(f"  {'  Per-column':<40} {', '.join(pos_strs)}")
    if results["per_position"].carry_accuracy is not None:
        print(f"  {'  Carry accuracy':<40} {results['per_position'].carry_accuracy:>6.1f}%")
    print(f"  {'OOD generalization':<40} {results['ood'].accuracy:>6.1f}%")
    print(f"  {'Compositional':<40} {results['compositional'].accuracy:>6.1f}%")
    print(f"  {'Robustness':<40} {results['robustness'].accuracy:>6.1f}%")
    # Adversarial sub-results
    for adv_name in ["boundary", "format_attacks", "noise", "fgsm"]:
        key = f"adversarial/{adv_name}"
        if key in results:
            label = f"  Adversarial/{adv_name}"
            print(f"  {label:<40} {results[key].accuracy:>6.1f}%")
    print("=" * 60)

    return results


@torch.no_grad()
def evaluate_feature_attack(
    model: MathTransformer,
    tok: MathTokenizer,
    num_problems: int = 20,
    epsilon: float = 0.5,
    layer_idx: int = -1,
    verbose: bool = True,
) -> EvalResult:
    """Feature-space adversarial attack — disrupts internal representations.

    Reference [13] from the PDF: "Task-Agnostic Attacks Against Vision
    Foundation Models". Instead of perturbing inputs (like FGSM), this
    finds perturbations that maximally disrupt the model's HIDDEN STATES.

    How it works:
    1. Forward pass: record hidden state at target layer
    2. Compute gradient of hidden state norm w.r.t. input embeddings
    3. Perturb input embeddings in that direction
    4. Measure accuracy drop

    A large accuracy drop indicates the model relies on fragile internal
    representations rather than robust reasoning.

    Args:
        model: The transformer model.
        tok: Tokenizer.
        num_problems: Number of problems to attack.
        epsilon: Perturbation magnitude.
        layer_idx: Which layer to attack (-1 = last).
        verbose: Print failure examples.

    Returns:
        EvalResult with attack accuracy.
    """
    correct = 0
    total = 0
    details = []
    device = next(model.parameters()).device
    model.eval()

    for _ in range(num_problems):
        expr, ans = generate_problem(1, 3)
        prompt = f"{expr}="

        # Tokenize
        input_ids = tok.encode(prompt, add_special_tokens=True)
        if len(input_ids) > tok.max_seq_len:
            input_ids = input_ids[: tok.max_seq_len]
        inp = torch.tensor([input_ids], device=device, dtype=torch.long)

        # Get clean hidden state at target layer
        with torch.enable_grad():
            emb = model.token_embedding(inp).detach().requires_grad_(True)
            h = emb
            target_layer = layer_idx if layer_idx >= 0 else len(model.layers) + layer_idx
            for i, layer in enumerate(model.layers):
                h, _ = layer(h)
                if i == target_layer:
                    # Maximize the L2 norm of hidden states
                    target = h.norm()  # maximize this
                    break

            # Gradient of hidden state norm w.r.t. input embeddings
            grad = torch.autograd.grad(target, emb)[0]
            if grad is None:
                continue

            # Perturb in the direction that maximally changes hidden states
            grad_sign = grad.sign()
            adv_emb = emb + epsilon * grad_sign

        # Forward pass on perturbed embeddings
        h_adv = adv_emb
        for layer in model.layers:
            h_adv, _ = layer(h_adv)
        h_adv = model.norm(h_adv)
        adv_logits = model.lm_head(h_adv)

        # Greedy decode
        adv_pred_ids = adv_logits[0, -1].argmax().item()
        adv_pred = tok.decode([adv_pred_ids], skip_special=True) if hasattr(tok, "decode") else str(adv_pred_ids)

        is_correct = adv_pred == ans
        if is_correct:
            correct += 1
        total += 1
        details.append({
            "expr": expr, "expected": ans, "predicted": adv_pred,
            "correct": is_correct, "epsilon": epsilon,
        })
        if verbose and not is_correct and len([d for d in details if not d["correct"]]) <= 5:
            print(f"  FEATURE-ATTACK FAIL: {expr}={ans} | adv_pred: {adv_pred} (ε={epsilon})")

    acc = correct / total * 100 if total > 0 else 0.0
    return EvalResult(
        name=f"Feature-space attack ({total} problems, ε={epsilon})",
        accuracy=acc,
        num_problems=total,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Legacy API — backward-compatible wrappers
# ═══════════════════════════════════════════════════════════════════════


MULTI_STEP_PROBLEMS: list[tuple[str, str]] = [
    ("2+3+4=", "9"),
    ("10-3-2=", "5"),
    ("12+34=", "46"),
    ("56-18=", "38"),
    ("99+1=", "100"),
]


@torch.no_grad()
def solve_problems(
    model: MathTransformer, tok: MathTokenizer, problems: list[tuple[str, str]] | None = None
) -> None:
    """Legacy compatibility wrapper — prints a table of problem results."""
    if problems is None:
        problems = MULTI_STEP_PROBLEMS
    print(f'{"Problem":<20} {"Expected":<12} {"Predicted":<12} {"Result":<8}')
    print("-" * 52)
    for expr, expected in problems:
        prompt = expr if expr.endswith("=") else expr + "="
        generated = model.generate(tok, prompt, max_new_tokens=10, temperature=0.3, top_k=3)
        if "=" in generated:
            pred = generated.split("=")[-1].strip()
            pred = "".join(c for c in pred if c.isdigit() or c == "-")
        else:
            pred = ""
        result = "PASS" if pred == expected else "FAIL"
        print(f"{expr:<20} {expected:<12} {pred:<12} {result:<8}")


# ═══════════════════════════════════════════════════════════════════════
#  Model loading & CLI
# ═══════════════════════════════════════════════════════════════════════


def load_model(checkpoint_path: str | Path) -> tuple[MathTransformer, MathTokenizer]:
    """Load a trained model and matching tokenizer from a checkpoint."""
    cfg = Config()
    tok = MathTokenizer.load(str(Path(checkpoint_path).parent / "tokenizer.json"))
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, tok


if __name__ == "__main__":
    import sys
    import json

    ckpt = Path("checkpoints/best.pt")
    train_max_digits = 4
    save_path = None
    args = sys.argv[1:]

    # Parse --save flag
    if "--save" in args:
        idx = args.index("--save")
        save_path = args[idx + 1] if idx + 1 < len(args) else None
        args = args[:idx] + (args[idx + 2:] if idx + 2 < len(args) else [])

    if args:
        ckpt = Path(args[0])
    if len(args) > 1:
        train_max_digits = int(args[1])

    if not ckpt.exists():
        ckpt = Path("checkpoints/final.pt")
    if not ckpt.exists():
        print("No checkpoint found. Train first: python3 train.py")
        sys.exit(1)

    print(f"Loading model from {ckpt}")
    model, tok = load_model(ckpt)
    results = full_benchmark(model, tok, train_max_digits=train_max_digits)

    if save_path:
        summary = {}
        for k, v in results.items():
            if hasattr(v, "accuracy"):
                summary[k] = round(v.accuracy, 1)
            elif hasattr(v, "overall_accuracy"):
                summary[k] = round(v.overall_accuracy, 1)
            elif hasattr(v, "position_accuracy"):
                for col, acc in v.position_accuracy.items():
                    summary[f"per_pos_col{col}"] = round(acc, 1)
                if v.carry_accuracy is not None:
                    summary["per_pos_carry"] = round(v.carry_accuracy, 1)
        for adv_key in ["adversarial/boundary", "adversarial/format_attacks",
                         "adversarial/noise", "adversarial/fgsm"]:
            if adv_key in results:
                summary[adv_key.replace("/", "_")] = round(results[adv_key].accuracy, 1)

        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nBenchmark saved to {save_path}")
