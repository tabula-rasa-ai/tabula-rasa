"""Hard-Negative Mining for auto_train.py — identifies failure patterns and generates edge cases.

After each training round, evaluates specific problems to find failure patterns,
then generates similar edge cases to compose a targeted curriculum.
"""

import random
from typing import Optional


def evaluate_find_failures(model, tokenizer, cfg, op: str, num: int = 100) -> list[str]:
    """Evaluate a specialist and return the specific problem strings that failed.

    Args:
        model: MathTransformer model.
        tokenizer: MathTokenizer instance.
        cfg: Config object.
        op: Operation string ('add', 'sub', 'mul').
        num: Number of problems to test.

    Returns:
        List of (problem, correct_answer, model_output) tuples for failures.
    """
    from train_specialist import evaluate_detailed

    try:
        return evaluate_detailed(model, tokenizer, cfg, op, num=num)
    except (ImportError, AttributeError):
        # Fallback: basic evaluation
        return _basic_failure_scan(model, tokenizer, cfg, op, num)


def _basic_failure_scan(model, tokenizer, cfg, op: str, num: int) -> list[str]:
    """Basic failure scanner when evaluate_detailed is not available."""
    import torch
    from train_specialist import generate_problem, OPS

    failures = []
    op_fn = OPS[op]
    for _ in range(num):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        if op == "sub" and a < b:
            a, b = b, a
        prompt = f"{a}{OPS[op]}{b}="
        correct = op_fn(a, b)

        full = model.generate(tokenizer, prompt, max_new_tokens=10, temperature=0.0, top_k=5)
        output = full.split("=")[-1].replace("<EOS>", "").replace("<PAD>", "").strip()

        # Parse scratchpad
        if len(output) >= 2 and all(c in "0123456789" for c in output):
            digits = output[1::2]  # every 2nd char = digit values
            answer = digits[::-1]  # reverse LSD-first
            answer_str = answer.lstrip("0") or "0"
        else:
            answer_str = output

        try:
            model_ans = int(answer_str)
        except ValueError:
            model_ans = -1

        if model_ans != correct:
            failures.append((prompt, str(correct), output))

    return failures


def generate_hard_negatives(failures: list[tuple[str, str, str]], max_variants: int = 5) -> list[str]:
    """Generate similar edge cases from failure patterns.

    For each failing problem, analyzes the operation and digits, then
    generates variants with the same pattern (e.g., 8-3 -> 7-4, 9-2, 6-5, 12-7, 15-9).

    Args:
        failures: List of (problem, correct_answer, model_output) tuples.
        max_variants: Max variants per failure.

    Returns:
        List of problem strings (like "7-4=") for retraining.
    """
    variants = []
    seen = set()

    for problem, correct, _ in failures:
        if len(variants) >= max_variants * len(failures):
            break

        # Parse problem: find operator position
        op_pos = -1
        op_char = None
        for c in "+-*/":
            pos = problem.find(c)
            if pos >= 0:
                op_pos = pos
                op_char = c
                break

        if op_pos < 0:
            continue

        left_str = problem[:op_pos]
        right_str = problem[op_pos + 1:].rstrip("=")

        try:
            a = int(left_str)
            b = int(right_str)
        except ValueError:
            continue

        # Generate variants
        for _ in range(max_variants):
            if op_char == "+":
                delta = random.randint(-3, 3)
                na, nb = a + delta, b - delta
                if na >= 0 and nb >= 0 and na + nb < 1000:
                    variant = f"{na}+{nb}="
            elif op_char == "-":
                delta = random.randint(-3, 3)
                na, nb = a + delta, b + delta
                if na >= 0 and nb >= 0 and na >= nb:
                    # Also try different borrow patterns
                    variant = f"{na}-{nb}="
                else:
                    # Generate a problem with similar borrow complexity
                    diff = a - b
                    if diff > 0:
                        new_a = random.randint(diff + 1, max(diff + 5, 9))
                        new_b = new_a - diff
                        variant = f"{new_a}-{new_b}="
                    else:
                        continue
            elif op_char == "*":
                delta_a = random.randint(-2, 2)
                delta_b = random.randint(-2, 2)
                na = max(0, a + delta_a)
                nb = max(0, b + delta_b)
                if na * nb < 1000:
                    variant = f"{na}*{nb}="
                else:
                    continue
            else:
                continue

            if variant not in seen and variant != problem:
                seen.add(variant)
                variants.append(variant)

        # Also add the original failure
        if problem not in seen:
            seen.add(problem)
            variants.append(problem)

    return variants


def curriculum_with_hard_negatives(base_curriculum: list, failures: list[tuple]) -> list:
    """Create a curriculum that prioritizes hard-negative edge cases.

    Args:
        base_curriculum: Original curriculum phases list [(steps, max_digits), ...].
        failures: List of failing problems.

    Returns:
        Modified curriculum with hard-negative boot camp prepended.
    """
    if not failures:
        return base_curriculum

    # Prepend a "hard negative boot camp" phase: same digits as failures, targeted
    hard_neg_steps = min(2000, len(failures) * 100)
    curriculum = [(hard_neg_steps, 0)]  # Phase 0: hard negative targeted training

    return curriculum + base_curriculum
