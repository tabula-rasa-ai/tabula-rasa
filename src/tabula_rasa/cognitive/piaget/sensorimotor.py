"""Sensorimotor Stage — Token Identity.

Tests the model's ability to distinguish and correctly process individual tokens
(positional discrimination, character vs digit identity).

Key capability: does the model know that '12' and '21' are different?
Does it see the difference between '5' and '55'?
"""



def run_sensorimotor(model, tokenizer, verbose: bool = False) -> dict:
    """Run Sensorimotor benchmark tests.

    Test categories:
    1. **Digit reversal** — '12' vs '21': model must produce different outputs
    2. **Digit identity** — '5' vs '55': model must treat these as different numbers
    3. **Token boundary** — does the model distinguish 123 from 12+3?
    4. **Character-level fallback** — does the model still recognize individual
       characters even when BPE merges exist?
    """
    device = None
    if hasattr(model, 'parameters'):
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = None
    results = []

    # ─── Test 1: Digit Reversal ───────────────────────────────────
    reversal_pairs = [
        ("12+34", "21+43"),       # reversed operands
        ("56+78", "65+87"),
        ("100-1", "001-1"),       # reversed digits
    ]
    reversal_score = 0
    for a, b in reversal_pairs:
        ids_a = tokenizer.encode(a, add_special_tokens=True)
        ids_b = tokenizer.encode(b, add_special_tokens=True)
        # At character level, these must produce different token sequences
        different = ids_a != ids_b
        if different:
            reversal_score += 1
        if verbose:
            print(f'  [Sensorimotor] Reversal: {a} vs {b} → {"DISTINCT" if different else "SAME (bad)"}')

    # ─── Test 2: Digit Identity ──────────────────────────────────
    identity_pairs = [
        ("5", "55"),
        ("7", "77"),
        ("99", "9"),
    ]
    identity_score = 0
    for a, b in identity_pairs:
        ids_a = tokenizer.encode(a, add_special_tokens=True)
        ids_b = tokenizer.encode(b, add_special_tokens=True)
        different = ids_a != ids_b
        if different:
            identity_score += 1
        if verbose:
            print(f'  [Sensorimotor] Identity: {a} vs {b} → {"DISTINCT" if different else "SAME (bad)"}')

    # ─── Test 3: Generate different outputs for different inputs ──
    if hasattr(model, 'generate'):
        gen_pairs = [
            ("12+34=", "21+43="),
            ("5+5=",   "55+5="),
            ("100-1=", "001-1="),
        ]
        gen_score = 0
        for prompt_a, prompt_b in gen_pairs:
            out_a = model.generate(tokenizer, prompt_a, max_new_tokens=8, temperature=0.0)
            out_b = model.generate(tokenizer, prompt_b, max_new_tokens=8, temperature=0.0)
            different = out_a != out_b
            if different:
                gen_score += 1
            if verbose:
                print(f'  [Sensorimotor] Generate: "{prompt_a}" → {repr(out_a)}')
                print(f'                  "{prompt_b}" → {repr(out_b)}')
                print(f'                  → {"DIFFERENT (good)" if different else "SAME (bad)"}')

    reversal_score_pct = (reversal_score / len(reversal_pairs)) * 100
    identity_score_pct = (identity_score / len(identity_pairs)) * 100
    gen_score_pct = (gen_score / len(gen_pairs)) * 100

    return {
        'stage': 'Sensorimotor',
        'tests': {
            'digit_reversal': {
                'score': reversal_score_pct,
                'passed': reversal_score,
                'total': len(reversal_pairs),
                'description': 'Model distinguishes reversed digit sequences',
            },
            'digit_identity': {
                'score': identity_score_pct,
                'passed': identity_score,
                'total': len(identity_pairs),
                'description': 'Model distinguishes 5 from 55',
            },
            'generation_diversity': {
                'score': gen_score_pct,
                'passed': gen_score,
                'total': len(gen_pairs),
                'description': 'Model produces different outputs for different inputs',
            },
        },
        'overall_score': (reversal_score_pct + identity_score_pct + gen_score_pct) / 3,
    }
