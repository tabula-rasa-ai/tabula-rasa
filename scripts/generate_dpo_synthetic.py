"""Generate synthetic DPO preference pairs for testing the pipeline.

Creates realistic (chosen, rejected) pairs for 1-digit arithmetic
where the rejected trace has a column error and the chosen trace is correct.
"""

import json
import random
import sys
from pathlib import Path


def _gen_problem(max_digits=1):
    a = random.randint(0, 10**max_digits - 1)
    b = random.randint(0, 10**max_digits - 1)
    op = random.choice(["+", "-"])
    if op == "-":
        a, b = max(a, b), min(a, b)  # ensure non-negative result
    expr = f"{a}{op}{b}"
    return expr, str(eval(expr)), op, a, b


def _correct_scratchpad(a, b, op):
    """Generate correct fused carry-digit scratchpad."""
    carry = 0
    sp = ""
    ra, rb = str(a)[::-1], str(b)[::-1]
    max_len = max(len(ra), len(rb))
    for i in range(max_len):
        da = int(ra[i]) if i < len(ra) else 0
        db = int(rb[i]) if i < len(rb) else 0
        if op == "+":
            total = da + db + carry
            carry = total // 10
            digit = total % 10
        else:
            if da < db + carry:
                da += 10
            total = da - db - carry
            carry = 0
            digit = total
        sp += f"{carry}{digit}"
    if carry and op == "+":
        sp += f"0{carry}"
    return sp


def _wrong_scratchpad(a, b, op):
    """Generate a wrong scratchpad by corrupting one column."""
    correct = _correct_scratchpad(a, b, op)
    if len(correct) < 2:
        # Too short to corrupt meaningfully; swap digits
        return correct[::-1] if len(correct) >= 2 else str(int(correct) ^ 1)
    # Pick a random column and flip one digit
    col_idx = random.randrange(0, len(correct), 2)
    wrong = list(correct)
    digit_pos = col_idx + random.choice([0, 1])
    wrong_digit = str((int(wrong[digit_pos]) + random.randint(1, 9)) % 10)
    wrong[digit_pos] = wrong_digit
    wrong_str = "".join(wrong)
    return wrong_str if wrong_str != correct else correct[::-1]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic DPO preference pairs"
    )
    parser.add_argument("-n", "--num-pairs", type=int, default=50,
                        help="Number of pairs to generate (default: 50)")
    parser.add_argument("-o", "--output", type=str, default="dpo_pairs_synthetic.jsonl",
                        help="Output path (default: dpo_pairs_synthetic.jsonl)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    pairs = []

    for _ in range(args.num_pairs):
        expr, answer, op, a, b = _gen_problem(max_digits=1)
        chosen = _correct_scratchpad(a, b, op)
        rejected = _wrong_scratchpad(a, b, op)

        pairs.append({
            "prompt": f"{expr}=",
            "chosen": chosen,
            "rejected": rejected,
            "expression": expr,
            "expected": answer,
            "op": op,
        })

    with open(args.output, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Stats
    correct_count = sum(1 for p in pairs if p["chosen"] == _correct_scratchpad(
        int(p["expression"].split(p["op"])[0]),
        int(p["expression"].split(p["op"])[1]),
        p["op"]
    ))
    print(f"Generated {len(pairs)} DPO preference pairs")
    print(f"Output: {args.output}")
    print(f"Valid chosen traces: {correct_count}/{len(pairs)}")
    print(f"\nSample:")
    print(json.dumps(pairs[0], indent=2))


if __name__ == "__main__":
    main()
