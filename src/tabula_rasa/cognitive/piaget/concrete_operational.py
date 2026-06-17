"""Concrete Operational Stage — Reversibility.

Tests whether the model has developed reversible mental operations:
if it learned a+b=c, can it infer c-b=a without explicit training?

This is the hallmark of Piaget's Concrete Operational stage:
operations become "reversible" — the child can mentally undo an action.

Key metric: **transfer accuracy** — performance on reverse problems
without any training on them.
"""


def run_concrete_operational(model, tokenizer, verbose: bool = False) -> dict:
    """Run Concrete Operational benchmark tests.

    Test categories:
    1. **Addition reversibility** — if model learned 12+34=46,
       can it solve 46-34=12 or 46-12=34?
    2. **Multiplication reversibility** — 3*4=12 → 12/4=3
    3. **Chained operation reversibility** — (2+3)*4=20 → 20/4-3=2
    """
    results = []

    # --- Test 1: Addition <-> Subtraction (inverse operations) -----
    add_sub_pairs = [
        ("12+34=", "46", "46-34=", "12"),
        ("56+78=", "134", "134-78=", "56"),
        ("100+200=", "300", "300-200=", "100"),
        ("5+5=", "10", "10-5=", "5"),
        ("99+1=", "100", "100-1=", "99"),
    ]

    forward_correct = 0
    reverse_correct = 0
    total_pairs = len(add_sub_pairs)

    for forward_expr, forward_ans, reverse_expr, reverse_ans in add_sub_pairs:
        # Forward test (should be trained)
        out_fwd = model.generate(tokenizer, forward_expr, max_new_tokens=8, temperature=0.0)
        pred_fwd = ''.join(c for c in out_fwd.split('=')[-1].strip() if c.isdigit() or c == '-') if '=' in out_fwd else ''
        if pred_fwd == forward_ans:
            forward_correct += 1

        # Reverse test (transfer — should not be explicitly trained)
        out_rev = model.generate(tokenizer, reverse_expr, max_new_tokens=8, temperature=0.0)
        pred_rev = ''.join(c for c in out_rev.split('=')[-1].strip() if c.isdigit() or c == '-') if '=' in out_rev else ''
        if pred_rev == reverse_ans:
            reverse_correct += 1

        if verbose:
            fwd_ok = 'PASS' if pred_fwd == forward_ans else 'FAIL'
            rev_ok = 'PASS' if pred_rev == reverse_ans else 'FAIL'
            print(f'  [Concrete]   {forward_expr:>8s} -> {pred_fwd:>4s} ({fwd_ok})  |  '
                  f'{reverse_expr:>8s} -> {pred_rev:>4s} ({rev_ok})')

    # --- Test 2: Multiplication <-> Division -----------------------
    mul_div_pairs = [
        ("3*4=", "12", "12/4=", "3"),
        ("5*7=", "35", "35/7=", "5"),
        ("2*50=", "100", "100/50=", "2"),
    ]

    mul_forward_correct = 0
    div_reverse_correct = 0
    mul_total = len(mul_div_pairs)

    for fwd_expr, fwd_ans, rev_expr, rev_ans in mul_div_pairs:
        out_fwd = model.generate(tokenizer, fwd_expr, max_new_tokens=8, temperature=0.0)
        pred_fwd = ''.join(c for c in out_fwd.split('=')[-1].strip() if c.isdigit() or c == '-') if '=' in out_fwd else ''
        if pred_fwd == fwd_ans:
            mul_forward_correct += 1

        out_rev = model.generate(tokenizer, rev_expr, max_new_tokens=8, temperature=0.0)
        pred_rev = ''.join(c for c in out_rev.split('=')[-1].strip() if c.isdigit() or c == '-') if '=' in out_rev else ''
        if pred_rev == rev_ans:
            div_reverse_correct += 1

        if verbose:
            print(f'  [Concrete]   {fwd_expr:>8s} -> {pred_fwd:>4s}  |  {rev_expr:>8s} -> {pred_rev:>4s}')

    add_rev_pct = (reverse_correct / total_pairs) * 100 if total_pairs > 0 else 0
    add_fwd_pct = (forward_correct / total_pairs) * 100 if total_pairs > 0 else 0
    mul_rev_pct = (div_reverse_correct / mul_total) * 100 if mul_total > 0 else 0
    mul_fwd_pct = (mul_forward_correct / mul_total) * 100 if mul_total > 0 else 0

    forward_overall = forward_correct + mul_forward_correct
    reverse_overall = reverse_correct + div_reverse_correct
    reversibility_ratio = reverse_overall / max(forward_overall, 1)

    return {
        'stage': 'Concrete Operational',
        'tests': {
            'addition_reversibility': {
                'score': add_rev_pct,
                'forward_score': add_fwd_pct,
                'passed': reverse_correct,
                'total': total_pairs,
                'description': 'If a+b=c learned, can model solve c-b=a?',
            },
            'multiplication_reversibility': {
                'score': mul_rev_pct,
                'forward_score': mul_fwd_pct,
                'passed': div_reverse_correct,
                'total': mul_total,
                'description': 'If a*b=c learned, can model solve c/b=a?',
            },
        },
        'reversibility_ratio': reversibility_ratio,
        'overall_score': (add_rev_pct + mul_rev_pct) / 2,
    }
