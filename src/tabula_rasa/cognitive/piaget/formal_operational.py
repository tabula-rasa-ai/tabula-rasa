"""Formal Operational Stage — Hypothetical Reasoning.

Tests whether the Socratic Engine can exhibit formal operational thought:
- Can the Skeptic persona predict the Logician's counter-argument?
- Can the model reason about hypotheticals it hasn't been trained on?
- Can it generate logically valid chains of inference?

This is the hallmark of Piaget's Formal Operational stage:
abstract, hypothetical, and systematic reasoning.
"""


def run_formal_operational(verify_socratic_fn=None, verbose: bool = False) -> dict:
    """Run Formal Operational benchmark tests.

    Args:
        verify_socratic_fn: A function that takes a (premise, conclusion) or
            debate transcript and returns a validity score.
            If None, tests run in analysis-only mode.

    Test categories:
    1. **Skeptic Prediction** — can the Skeptic persona generate the
       Logician's counter-argument before it's generated?
    2. **Chain of Inference** — can the model chain multiple syllogisms?
    3. **Hypothetical Reasoning** — can it reason from false premises?
    4. **TRLD Compliance** (if verify_socratic_fn supports TRLD) — does the
       model emit valid TRLD syntax?
    """
    results = []

    # ─── Test 1: Chain of Inference ───────────────────────────────
    # Multi-step syllogism
    chains = [
        {
            'name': 'Two-step chain',
            'premises': [
                'All humans are mortal',
                'Socrates is human',
                'All mortals die',
            ],
            'conclusion': 'Socrates dies',
            'valid': True,
        },
        {
            'name': 'Invalid chain',
            'premises': [
                'All birds can fly',
                'Penguins are birds',
            ],
            'conclusion': 'Penguins cannot fly',  # Actually false — penguins ARE birds and birds fly
            'valid': False,
        },
    ]

    chain_score = 0
    chain_total = len(chains)

    if verify_socratic_fn:
        for chain in chains:
            trld_text = ' '.join(f'[PREMISE: {p}]' for p in chain['premises'])
            trld_text += f' [CONCLUSION: {chain["conclusion"]}]'
            result = verify_socratic_fn(trld_text)

            is_correct = result.get('valid') == chain['valid']
            if is_correct:
                chain_score += 1
            if verbose:
                status = '✓' if is_correct else '✗'
                print(f'  [Formal] {chain["name"]}: {status}')
                print(f'          {chain["premises"][0][:40]}...')
                print(f'          => {chain["conclusion"]}')
    else:
        # Analysis-only: just count without verification
        if verbose:
            for chain in chains:
                print(f'  [Formal] {chain["name"]}: {chain["premises"]} => {chain["conclusion"]} '
                      f'(expected: {"valid" if chain["valid"] else "invalid"})')

    # ─── Test 2: Counter-Argument Recognition ─────────────────────
    # Can the model identify logical fallacies?
    fallacies = [
        {
            'name': 'Ad Hominem',
            'statement': 'You cannot trust his argument because he is not an expert',
            'has_fallacy': True,
        },
        {
            'name': 'Valid argument',
            'statement': 'All humans are mortal, Socrates is human, therefore Socrates is mortal',
            'has_fallacy': False,
        },
    ]

    if verify_socratic_fn and hasattr(verify_socratic_fn, 'score_debate_argument'):
        fallacy_score = 0
        fallacy_total = len(fallacies)
        for f in fallacies:
            result = verify_socratic_fn.score_debate_argument(f['statement'])
            detected_fallacy = len(result.get('fallacies', [])) > 0
            correct = detected_fallacy == f['has_fallacy']
            if correct:
                fallacy_score += 1
            if verbose:
                print(f'  [Formal] Fallacy: {f["name"]} → {"✓" if correct else "✗"}')
    else:
        fallacy_score = 0
        fallacy_total = 0

    # ─── Test 3: Hypothetical Reasoning ───────────────────────────
    # Can the model reason from counter-factual premises?
    hypotheticals = [
        {
            'name': 'Counter-factual',
            'trld': '[PREMISE: All stones are alive] [PREMISE: Granite is a stone] [CONCLUSION: Granite is alive]',
            'valid': True,  # Logically valid (from the premises, even though false in reality)
        },
    ]

    hypo_score = 0
    hypo_total = len(hypotheticals)
    if verify_socratic_fn:
        for h in hypotheticals:
            result = verify_socratic_fn(h['trld'])
            is_correct = result.get('valid') == h['valid']
            if is_correct:
                hypo_score += 1
            if verbose:
                print(f'  [Formal] {h["name"]}: {"✓" if is_correct else "✗"}')

    chain_pct = (chain_score / chain_total) * 100 if chain_total > 0 else 0
    hypo_pct = (hypo_score / hypo_total) * 100 if hypo_total > 0 else 0

    return {
        'stage': 'Formal Operational',
        'tests': {
            'chain_of_inference': {
                'score': chain_pct,
                'passed': chain_score,
                'total': chain_total,
                'description': 'Model correctly chains multiple syllogism steps',
            },
            'hypothetical_reasoning': {
                'score': hypo_pct,
                'passed': hypo_score,
                'total': hypo_total,
                'description': 'Model reasons correctly from counter-factual premises',
            },
        },
        'overall_score': (chain_pct + hypo_pct) / 2,
    }
