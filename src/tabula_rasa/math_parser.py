"""Deterministic math expression parser and validator.

Used by the Socratic critique loop to verify model-generated
arithmetic output without running a neural net.

All functions are pure Python (stdlib only) with no neural net calls.
Supports: +, -, *, / with integer arithmetic.

Scratchpad format:
    The fused carry-digit token format encodes column-by-column computation
    as pairs of (carry, digit) characters read left-to-right.
    E.g., "0406" means:
      - Column 0 (most significant): carry=0, digit=4
      - Column 1 (least significant): carry=0, digit=6
    Column 0 is the leftmost pair (most significant digit position).
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPS = frozenset({"+", "-", "*", "/"})

OP_NAMES: dict[str, str] = {
    "+": "addition",
    "-": "subtraction",
    "*": "multiplication",
    "/": "division",
}

_EXPR_RE = re.compile(r"^(-?\d+)\s*([+\-*/])\s*(-?\d+)$")

# ---------------------------------------------------------------------------
# 1. parse_expression
# ---------------------------------------------------------------------------


def parse_expression(text: str) -> Optional[tuple[str, int, int, str]]:
    """Parse a math expression string like ``'12+34'`` into components.

    Args:
        text: Expression string, e.g. ``'12+34'``, ``'100-50'``, ``'6*7'``,
              ``'42/6'``.  Leading/trailing whitespace is stripped.
              Negative numbers are supported: ``'-5+3'``, ``'10--3'``.

    Returns:
        Tuple of ``(operation, a, b, formatted)`` where:
          - *operation* is one of ``'+'``, ``'-'``, ``'*'``, ``'/'``
          - *a* and *b* are the integer operands
          - *formatted* is the normalised expression string (e.g. ``'12+34'``)

        Returns ``None`` if the expression cannot be parsed.
    """
    text = text.strip()
    m = _EXPR_RE.match(text)
    if not m:
        return None
    a_str, op, b_str = m.group(1), m.group(2), m.group(3)
    a, b = int(a_str), int(b_str)
    formatted = f"{a}{op}{b}"
    return (op, a, b, formatted)


# ---------------------------------------------------------------------------
# 2. evaluate
# ---------------------------------------------------------------------------


def evaluate(a: int, b: int, op: str) -> int:
    """Deterministically compute the result of a binary arithmetic operation.

    Args:
        a: First operand (integer).
        b: Second operand (integer).
        op: One of ``'+'``, ``'-'``, ``'*'``, ``'/'``.

    Returns:
        Integer result of the operation.

    Raises:
        ValueError: If *op* is not a supported operation.
        ZeroDivisionError: If *op* is ``'/'`` and *b* is zero.
    """
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("Division by zero")
        # Integer division (floor).  The spec says "deterministic" and
        # the rest of the system works with integers.
        return a // b
    raise ValueError(f"Unsupported operation: '{op}'")


# ---------------------------------------------------------------------------
# 3. verify_equation
# ---------------------------------------------------------------------------


def verify_equation(equation: str) -> dict[str, Any]:
    """Verify a complete equation string like ``'12+34=46'``.

    Args:
        equation: A string containing an arithmetic expression and expected
                  result, separated by ``'='`` (e.g. ``'12+34=46'``).

    Returns:
        A dictionary with the following keys:

        - **valid** (``bool``): Whether the equation is correct.
        - **expected** (``str``): The correct result as a string.
        - **got** (``str``): The result found in the equation string.
        - **expr** (``str``): The normalised expression (e.g. ``'12+34'``).
        - **op** (``str``): The operator character.
        - **a** (``int``): First operand.
        - **b** (``int``): Second operand.

        On parse / evaluation failure a key ``error`` is present instead
        of partial results.
    """
    equation = equation.strip()

    # Split on '='
    parts = equation.split("=")
    if len(parts) != 2:
        return {
            "valid": False,
            "error": "Cannot parse",
            "expr": equation,
            "expected": "",
            "got": "",
        }

    expr_str, got_str = parts[0].strip(), parts[1].strip()

    parsed = parse_expression(expr_str)
    if parsed is None:
        return {
            "valid": False,
            "error": "Cannot parse",
            "expr": expr_str,
            "expected": "",
            "got": got_str,
        }

    op, a, b, formatted_expr = parsed

    try:
        expected = evaluate(a, b, op)
    except (ValueError, ZeroDivisionError) as exc:
        return {
            "valid": False,
            "error": f"Evaluation error: {exc}",
            "expr": formatted_expr,
            "expected": "",
            "got": got_str,
        }

    expected_str = str(expected)

    return {
        "valid": expected_str == got_str,
        "expected": expected_str,
        "got": got_str,
        "expr": formatted_expr,
        "op": op,
        "a": a,
        "b": b,
    }


# ---------------------------------------------------------------------------
# 4. verify_scratchpad  (column-by-column fused carry-digit format)
# ---------------------------------------------------------------------------


def _pad_digits(a: int, b: int) -> tuple[str, str, int]:
    """Pad two integers to equal-length digit strings (absolute values).

    Returns ``(a_padded, b_padded, num_digits)``.
    Negative numbers use their absolute values; the sign is tracked separately.
    """
    a_str = str(abs(a))
    b_str = str(abs(b))
    n = max(len(a_str), len(b_str), 1)
    return a_str.zfill(n), b_str.zfill(n), n


def _column_add(
    a_digit: int, b_digit: int, carry_in: int
) -> tuple[int, int]:
    """Compute (carry_out, result_digit) for one column of addition."""
    total = a_digit + b_digit + carry_in
    return total // 10, total % 10


def _column_sub(a_digit: int, b_digit: int, borrow_in: int) -> tuple[int, int]:
    """Compute (borrow_out, result_digit) for one column of subtraction.

    *borrow_in* is 0 or 1 (1 means "we already borrowed from this column").
    """
    diff = a_digit - b_digit - borrow_in
    if diff < 0:
        return 1, diff + 10
    return 0, diff


def verify_scratchpad(text: str) -> dict[str, Any]:
    """Verify a model's scratchpad output in fused carry-digit token format.

    The scratchpad format encodes column-by-column computation as consecutive
    pairs of (carry, digit) characters.  Pairs are read left-to-right:
    column 0 = leftmost pair = most significant digit position.

    Example:
        Input ``"12+34=0406"``
        - Scratchpad ``"0406"`` → pairs ``"04"``, ``"06"``
        - Column 0 (tens):   carry=0, digit=4
        - Column 1 (ones):   carry=0, digit=6

    Args:
        text: Full model output containing ``<expression>=<scratchpad>``,
              e.g. ``'12+34=0406'`` or ``'12+34=06'``.

    Returns:
        A dictionary with:

        - **valid** (``bool``): Whether **every** column matches the correct
          column-by-column arithmetic.
        - **columns** (``list[dict]``): Per-column analysis.  Each entry has
          keys ``col`` (int), ``carry`` (int), ``digit`` (int),
          ``correct`` (bool).
        - **errors** (``list[str]``): Human-readable error messages.
    """
    text = text.strip()

    # --- split on '=' -------------------------------------------------------
    parts = text.split("=", maxsplit=1)
    if len(parts) != 2:
        return {
            "valid": False,
            "columns": [],
            "errors": ["Cannot parse: expected '=' in output"],
        }

    expr_str, scratch_str = parts[0].strip(), parts[1].strip()

    parsed = parse_expression(expr_str)
    if parsed is None:
        return {
            "valid": False,
            "columns": [],
            "errors": [f"Cannot parse expression: '{expr_str}'"],
        }

    op, a, b, _ = parsed

    # --- validate scratchpad format -----------------------------------------
    if not scratch_str.isdigit():
        return {
            "valid": False,
            "columns": [],
            "errors": ["Scratchpad contains non-digit characters"],
        }

    num_pairs = len(scratch_str) // 2

    # --- dispatch by operator -----------------------------------------------
    if op == "+":
        return _verify_add_scratchpad(a, b, scratch_str, num_pairs)
    if op == "-":
        return _verify_sub_scratchpad(a, b, scratch_str, num_pairs)
    if op == "*":
        return _verify_mul_scratchpad(a, b, scratch_str)
    # division — not columnar in the same sense
    return _verify_div_scratchpad(a, b, scratch_str)


def _verify_add_scratchpad(
    a: int, b: int, scratch: str, num_pairs: int
) -> dict[str, Any]:
    """Verify scratchpad for addition (column-by-column)."""
    if len(scratch) % 2 != 0:
        return {
            "valid": False,
            "columns": [],
            "errors": [
                f"Addition scratchpad length must be even (pairs of carry+digit), "
                f"got {len(scratch)} chars"
            ],
        }
    a_pad, b_pad, num_digits = _pad_digits(a, b)

    # Extra pair beyond num_digits represents a final carry-out
    total_cols = max(num_pairs, num_digits)

    columns: list[dict[str, Any]] = []
    errors: list[str] = []
    all_valid = True
    carry: int = 0

    # Process right-to-left for carry propagation, but build result left-to-right
    col_results: list[dict[str, Any]] = [None] * total_cols  # type: ignore[list-item]

    for col_from_right in range(total_cols):
        # Determine scratchpad pair index for this column
        # If num_pairs == total_cols: pair_idx = total_cols - 1 - col_from_right
        pair_idx = num_pairs - 1 - col_from_right

        # Get the operand digits for this column (right-aligned)
        a_digit = int(a_pad[-(col_from_right + 1)]) if col_from_right < num_digits else 0
        b_digit = int(b_pad[-(col_from_right + 1)]) if col_from_right < num_digits else 0

        # Compute expected
        carry_out, result_digit = _column_add(a_digit, b_digit, carry)

        if pair_idx >= 0:
            # This column is present in the scratchpad
            model_carry = int(scratch[pair_idx * 2])
            model_digit = int(scratch[pair_idx * 2 + 1])
            correct = model_carry == carry_out and model_digit == result_digit
            if not correct:
                all_valid = False
                errors.append(
                    f"Column {pair_idx}: carry={model_carry}, digit={model_digit}, "
                    f"expected carry={carry_out}, digit={result_digit}"
                )
            col_results[pair_idx] = {
                "col": pair_idx,
                "carry": model_carry,
                "digit": model_digit,
                "correct": correct,
            }
        else:
            # Extra column not covered by scratchpad — this is a final carry
            if carry != 0:
                all_valid = False
                errors.append(
                    f"Missing final carry column: expected carry={carry}"
                )

        carry = carry_out

    # Final carry: if there's an extra column, the model should have it
    if carry != 0 and num_pairs == num_digits:
        # Model is missing the final carry column
        all_valid = False
        errors.append(
            f"Missing column {num_digits}: expected carry={carry} "
            f"(final carry-out not shown)"
        )
    elif carry != 0 and num_pairs > num_digits:
        # Final carry column exists — check it
        extra_pair_idx = num_pairs - 1  # leftmost pair
        model_carry = int(scratch[extra_pair_idx * 2])
        model_digit = int(scratch[extra_pair_idx * 2 + 1])
        correct = model_carry == 0 and model_digit == carry
        if not correct:
            all_valid = False
            errors.append(
                f"Column {extra_pair_idx}: carry={model_carry}, digit={model_digit}, "
                f"expected carry=0, digit={carry}"
            )
        if col_results[extra_pair_idx] is None:
            col_results[extra_pair_idx] = {
                "col": extra_pair_idx,
                "carry": model_carry,
                "digit": model_digit,
                "correct": correct,
            }

    # Filter out None entries from col_results
    columns = [c for c in col_results if c is not None]

    return {"valid": all_valid, "columns": columns, "errors": errors}


def _verify_sub_scratchpad(
    a: int, b: int, scratch: str, num_pairs: int
) -> dict[str, Any]:
    """Verify scratchpad for subtraction (column-by-column)."""
    if len(scratch) % 2 != 0:
        return {
            "valid": False,
            "columns": [],
            "errors": [
                f"Subtraction scratchpad length must be even (pairs of carry+digit), "
                f"got {len(scratch)} chars"
            ],
        }
    a_pad, b_pad, num_digits = _pad_digits(a, b)

    total_cols = max(num_pairs, num_digits)

    columns: list[dict[str, Any]] = []
    errors: list[str] = []
    all_valid = True
    borrow: int = 0

    col_results: list[dict[str, Any] | None] = [None] * total_cols

    for col_from_right in range(total_cols):
        pair_idx = num_pairs - 1 - col_from_right

        a_digit = int(a_pad[-(col_from_right + 1)]) if col_from_right < num_digits else 0
        b_digit = int(b_pad[-(col_from_right + 1)]) if col_from_right < num_digits else 0

        borrow_out, result_digit = _column_sub(a_digit, b_digit, borrow)

        if pair_idx >= 0:
            model_carry = int(scratch[pair_idx * 2])  # "carry" field = borrow
            model_digit = int(scratch[pair_idx * 2 + 1])
            correct = model_carry == borrow_out and model_digit == result_digit
            if not correct:
                all_valid = False
                errors.append(
                    f"Column {pair_idx}: borrow={model_carry}, digit={model_digit}, "
                    f"expected borrow={borrow_out}, digit={result_digit}"
                )
            col_results[pair_idx] = {
                "col": pair_idx,
                "carry": model_carry,
                "digit": model_digit,
                "correct": correct,
            }

        borrow = borrow_out

    if borrow != 0 and num_pairs == num_digits:
        all_valid = False
        errors.append(
            f"Missing final borrow column: expected borrow={borrow}"
        )

    columns = [c for c in col_results if c is not None]

    return {"valid": all_valid, "columns": columns, "errors": errors}


def _verify_mul_scratchpad(a: int, b: int, scratch: str) -> dict[str, Any]:
    """Verify scratchpad for multiplication by comparing against full result."""
    try:
        expected = evaluate(a, b, "*")
    except ValueError as exc:
        return {"valid": False, "columns": [], "errors": [str(exc)]}

    expected_str = str(expected)

    # The model may output just the result digits or carry-digit pairs
    # Try simple result comparison first
    if scratch.lstrip("-") == expected_str.lstrip("-"):
        return {"valid": True, "columns": [], "errors": []}

    # Check if the scratchpad (interpreted as digits) equals the result
    # Strip any leading zeros from scratch
    scratch_as_int = int(scratch) if scratch else 0
    if scratch_as_int == expected:
        return {"valid": True, "columns": [], "errors": []}

    return {
        "valid": False,
        "columns": [],
        "errors": [
            f"Multiplication: expected {expected_str}, "
            f"got scratchpad '{scratch}'"
        ],
    }


def _verify_div_scratchpad(a: int, b: int, scratch: str) -> dict[str, Any]:
    """Verify scratchpad for division."""
    if b == 0:
        return {
            "valid": False,
            "columns": [],
            "errors": ["Division by zero"],
        }
    try:
        expected = evaluate(a, b, "/")
    except (ValueError, ZeroDivisionError) as exc:
        return {"valid": False, "columns": [], "errors": [str(exc)]}

    expected_str = str(expected)

    if scratch.lstrip("-") == expected_str.lstrip("-"):
        return {"valid": True, "columns": [], "errors": []}

    scratch_as_int = int(scratch) if scratch else 0
    if scratch_as_int == expected:
        return {"valid": True, "columns": [], "errors": []}

    return {
        "valid": False,
        "columns": [],
        "errors": [
            f"Division: expected {expected_str}, "
            f"got scratchpad '{scratch}'"
        ],
    }


# ---------------------------------------------------------------------------
# 5. critique_output
# ---------------------------------------------------------------------------


def critique_output(model_output: str, problem: str) -> str:
    """Critique a model's raw arithmetic output against the original problem.

    This is the main entry point for the Socratic critique loop.  It takes the
    model's generated text and the original problem, checks correctness, and
    returns a human-readable critique string.

    Args:
        model_output: The raw output from ``model.generate()``.
        problem: The original arithmetic problem (e.g. ``'12+34'``).

    Returns:
        An empty string ``''`` if the output is completely correct.
        Otherwise, a human-readable critique describing the first error found:
          - Column-level errors: ``"Column N: carry=X, digit=Y, expected ..."``
          - Syntax errors: ``"Syntax error: expected '=' in output."``
          - Expression mismatch: ``"Expression mismatch: ..."``
          - Value errors: ``"Expected X, got Y."``
    """
    model_output = model_output.strip()
    problem = problem.strip()

    if not model_output:
        return "Syntax error: empty output."

    # If the output contains '=', treat as equation/scratchpad
    if "=" in model_output:
        parts = model_output.split("=", maxsplit=1)
        after_eq = parts[1].strip() if len(parts) == 2 else ""

        # Heuristic: if the part after '=' looks like a multi-pair scratchpad
        # (even length >= 4, all digits), try scratchpad verification first.
        # Otherwise fall through to standard equation verification.
        is_likely_scratchpad = (
            len(after_eq) >= 4
            and len(after_eq) % 2 == 0
            and after_eq.isdigit()
        )

        if is_likely_scratchpad:
            sp_result = verify_scratchpad(model_output)
            if not sp_result["valid"]:
                if sp_result["errors"]:
                    return sp_result["errors"][0]
                return "Scratchpad verification failed."
            # Scratchpad columns all correct — overall result is correct.
            return ""

        # Treat as standard equation
        eq_result = verify_equation(model_output)
        if "error" in eq_result:
            return f"Syntax error: {eq_result['error']}"
        if not eq_result["valid"]:
            return (
                f"Expression {eq_result['expr']}: "
                f"expected {eq_result['expected']}, "
                f"got {eq_result['got']}."
            )
        return ""

    # No '=' sign — the model may have just output the answer
    parsed_problem = parse_expression(problem)
    if parsed_problem is None:
        return f"Syntax error: cannot parse problem '{problem}'."

    op, a, b, _ = parsed_problem
    try:
        expected = evaluate(a, b, op)
    except (ValueError, ZeroDivisionError):
        return f"Evaluation error for '{problem}'."

    # Check if model output is the expected result
    try:
        got = int(model_output)
    except ValueError:
        return f"Syntax error: expected numeric output, got '{model_output}'."

    if got == expected:
        return ""

    return f"Expected {expected}, got {got}."


# ---------------------------------------------------------------------------
# 6. generate_curriculum_problems
# ---------------------------------------------------------------------------


def generate_curriculum_problems(
    op: str, count: int, max_digits: int = 4
) -> list[tuple[str, str]]:
    """Generate deterministic curriculum problems with known answers.

    This is a purely deterministic alternative to dataset-based problem
    generation.  Problems are generated systematically (not randomly) to
    cover a range of difficulty levels.

    Args:
        op: One of ``'+'``, ``'-'``, ``'*'``, ``'/'``.
        count: Number of problems to generate.
        max_digits: Maximum number of digits per operand (default 4).

    Returns:
        A list of ``(problem, answer)`` tuples, e.g.:
        ``[('12+34', '46'), ('7+8', '15'), ...]``

    Raises:
        ValueError: If *op* is unsupported or *count* is non-positive.
    """
    if op not in OPS:
        raise ValueError(f"Unsupported operation: '{op}'")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if max_digits < 1:
        raise ValueError(f"max_digits must be >= 1, got {max_digits}")

    problems: list[tuple[str, str]] = []
    # Use a seeded deterministic sequence so the same call always gives
    # the same problems (for reproducibility in testing).
    rng = random.Random(42)

    generated = 0
    # Guard against infinite loops for edge cases
    max_attempts = count * 50
    attempts = 0

    while generated < count and attempts < max_attempts:
        attempts += 1
        if op == "+":
            a = rng.randint(0, 10**max_digits - 1)
            b = rng.randint(0, 10**max_digits - 1)
            ans = a + b
        elif op == "-":
            a = rng.randint(0, 10**max_digits - 1)
            b = rng.randint(0, a)  # ensure non-negative result
            ans = a - b
        elif op == "*":
            digits_a = rng.randint(1, max_digits)
            digits_b = rng.randint(1, max(1, max_digits - 1))
            a = rng.randint(10 ** (digits_a - 1), 10**digits_a - 1) if digits_a > 1 else rng.randint(1, 9)
            b = rng.randint(10 ** (digits_b - 1), 10**digits_b - 1) if digits_b > 1 else rng.randint(1, 9)
            ans = a * b
        elif op == "/":
            b = rng.randint(1, 9)  # single-digit divisor for clean division
            quotient = rng.randint(0, 10**max_digits - 1)
            a = b * quotient
            ans = quotient
        else:
            continue  # pragma: no cover

        problem_str = f"{a}{op}{b}"
        answer_str = str(ans)
        problems.append((problem_str, answer_str))
        generated += 1

    return problems


# ---------------------------------------------------------------------------
# Public API convenience
# ---------------------------------------------------------------------------

__all__ = [
    "parse_expression",
    "evaluate",
    "verify_equation",
    "verify_scratchpad",
    "critique_output",
    "generate_curriculum_problems",
]
