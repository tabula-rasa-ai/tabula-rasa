"""
Multiplication scratchpad for Tabula Rasa.

Generates a step-by-step multiplication trace showing partial products and
their summation. Designed for training specialist models with Chain-of-Thought
reasoning for multiplication problems.

Format: "12*4=48<STEP>12*30=360<STEP>48+360=408<END>408"

Each multiplication step shows the partial product with correct place-value
alignment. The final summation step adds partial products. <END> precedes
the final answer.

Supports operands up to 3 digits each.
"""

from __future__ import annotations


def generate_mult_scratchpad(op1: int, op2: int) -> tuple[str, int]:
    """Generate a multiplication scratchpad trace.

    Takes two operands (up to 3 digits each) and produces a scratchpad string
    showing each partial product and the final summation, using ``<STEP>``
    separators between steps and ``<END>`` before the final answer.

    Args:
        op1: First operand (multiplicand).
        op2: Second operand (multiplier).

    Returns:
        A tuple ``(scratchpad_string, answer)`` where the scratchpad is
        formatted like ``'12*4=48<STEP>12*30=360<STEP>48+360=408<END>408'``.

    Raises:
        ValueError: If either operand has more than 3 digits.
    """
    if op1 > 9999 or op2 > 9999:
        raise ValueError(f"Operands limited to 4 digits, got {op1} and {op2}")
    if op1 < 0 or op2 < 0:
        raise ValueError(f"Negative operands not supported, got {op1} and {op2}")

    answer = op1 * op2
    steps: list[str] = []

    # Break the multiplier into individual digits with their place values
    op2_str = str(op2)
    partial_products: list[tuple[int, int]] = []  # (partial_product, place_value)

    for i, digit_char in enumerate(reversed(op2_str)):
        digit = int(digit_char)
        place = 10 ** i  # ones=1, tens=10, hundreds=100
        partial = op1 * digit * place

        if i == 0:
            # Ones place: multiply op1 * digit
            steps.append(f"{op1}*{digit}={op1 * digit}")
        else:
            # Tens/hundreds place: show with explicit trailing zero
            # e.g. 12*30 or 12*300
            shifted_digit = digit * place
            steps.append(f"{op1}*{shifted_digit}={partial}")

        partial_products.append((partial, place))

    # Summation step: add all partial products
    if len(partial_products) > 1:
        terms = [str(pp[0]) for pp in partial_products]
        sum_expr = "+".join(terms)
        steps.append(f"{sum_expr}={answer}")

    # Build the final scratchpad string
    scratchpad = "<STEP>".join(steps) + f"<END>{answer}"
    return scratchpad, answer


def parse_mult_scratchpad(output: str) -> str | None:
    """Extract the final answer from a multiplication scratchpad output.

    Looks for ``<END>`` followed by digits. Falls back to extracting the
    last number found in the output.

    Args:
        output: The model's generated output string.

    Returns:
        The final answer as a string, or ``None`` if no answer was found.
    """
    text = output.replace("<EOS>", "").replace("<PAD>", "").replace("<BOS>", "").strip()

    if "<END>" in text:
        return text.split("<END>")[-1].strip()

    # Fallback: look for the last number in the output
    import re
    numbers = re.findall(r"\d+", text)
    if numbers:
        return numbers[-1]
    return None


if __name__ == "__main__":
    # Demo
    examples = [
        (12, 34),
        (7, 8),
        (123, 45),
        (99, 99),
        (5, 9),
        (8, 7),
    ]
    for a, b in examples:
        sp, ans = generate_mult_scratchpad(a, b)
        print(f"\n{a} * {b} = {ans}")
        print(f"  Scratchpad: {sp}")
        extracted = parse_mult_scratchpad(sp)
        print(f"  Parsed answer: {extracted}")
