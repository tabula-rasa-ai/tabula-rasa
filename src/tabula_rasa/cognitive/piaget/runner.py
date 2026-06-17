"""Piaget Benchmark Runner — orchestrates all four stages and logs results.

Produces a "Cognitive Growth Chart" tracked over time as a JSONL file.
Each run appends a row: timestamp, per-stage scores, overall cognitive score.

Usage:
    from tabula_rasa.cognitive.piaget.runner import run_piaget_benchmark
    results = run_piaget_benchmark(model, tokenizer, specialist_map=...)
    # results contains per-stage dicts + overall_cognitive_score
"""

import json
import time
from pathlib import Path

# Growth chart log — append-only, one JSON object per run
GROWTH_CHART = Path('egefalos/piaget/growth_chart.jsonl')


def _load_growth_chart() -> list:
    """Load historical growth chart entries."""
    if not GROWTH_CHART.exists():
        return []
    entries = []
    with open(GROWTH_CHART) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _append_to_growth_chart(entry: dict):
    """Append one growth chart entry."""
    GROWTH_CHART.parent.mkdir(parents=True, exist_ok=True)
    with open(GROWTH_CHART, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def run_piaget_benchmark(model, tokenizer, specialist_map: dict = None,
                         verify_socratic_fn=None, verbose: bool = False) -> dict:
    """Run the full Piaget Benchmark across all four stages.

    Args:
        model: The trained model (must have .generate() method).
        tokenizer: The model's tokenizer.
        specialist_map: Optional dict of {domain_name: (model, tokenizer)}.
        verify_socratic_fn: Optional TRLD verification function
            (e.g., LogicVerifier.verify_trld).
        verbose: Print per-test details.

    Returns:
        dict with per-stage results and overall_cognitive_score.
    """
    from tabula_rasa.cognitive.piaget.sensorimotor import run_sensorimotor
    from tabula_rasa.cognitive.piaget.preoperational import run_preoperational
    from tabula_rasa.cognitive.piaget.concrete_operational import run_concrete_operational
    from tabula_rasa.cognitive.piaget.formal_operational import run_formal_operational

    print(f'\n{"=" * 60}')
    print(f'  Piaget Cognitive Benchmark')
    print(f'  Measuring developmental stages of Tabula Rasa')
    print(f'{"=" * 60}')
    print()

    # Stage 1: Sensorimotor
    print('  [Stage 1/4] Sensorimotor — Token Identity')
    stage1 = run_sensorimotor(model, tokenizer, verbose=verbose)
    print(f'    Score: {stage1["overall_score"]:.1f}%')

    # Stage 2: Preoperational
    print('  [Stage 2/4] Preoperational — Symbolic Routing')
    stage2 = run_preoperational(model, tokenizer, specialist_map=specialist_map, verbose=verbose)
    print(f'    Score: {stage2["overall_score"]:.1f}%')

    # Stage 3: Concrete Operational
    print('  [Stage 3/4] Concrete Operational — Reversibility')
    stage3 = run_concrete_operational(model, tokenizer, verbose=verbose)
    print(f'    Score: {stage3["overall_score"]:.1f}%')
    print(f'    Reversibility Ratio: {stage3.get("reversibility_ratio", 0):.2f} '
          f'(1.0 = perfect transfer)')

    # Stage 4: Formal Operational
    print('  [Stage 4/4] Formal Operational — Hypothetical Reasoning')
    stage4 = run_formal_operational(verify_socratic_fn=verify_socratic_fn, verbose=verbose)
    print(f'    Score: {stage4["overall_score"]:.1f}%')

    # ─── Overall Cognitive Score ──────────────────────────────────
    scores = [stage1["overall_score"], stage2["overall_score"],
              stage3["overall_score"], stage4["overall_score"]]
    overall = sum(scores) / len(scores)

    # Weight concrete operational slightly higher (hardest to achieve)
    weighted = (stage1["overall_score"] * 0.20 +
                stage2["overall_score"] * 0.20 +
                stage3["overall_score"] * 0.35 +
                stage4["overall_score"] * 0.25)

    print()
    print(f'{"=" * 60}')
    print(f'  Cognitive Profile')
    print(f'{"─" * 60}')
    print(f'  Sensorimotor:        {stage1["overall_score"]:5.1f}%  '
          f'{"✓" if stage1["overall_score"] > 80 else "○"}')
    print(f'  Preoperational:      {stage2["overall_score"]:5.1f}%  '
          f'{"✓" if stage2["overall_score"] > 60 else "○"}')
    print(f'  Concrete Operational: {stage3["overall_score"]:5.1f}%  '
          f'{"✓" if stage3["overall_score"] > 50 else "○"}')
    print(f'  Formal Operational:  {stage4["overall_score"]:5.1f}%  '
          f'{"✓" if stage4["overall_score"] > 30 else "○"}')
    print(f'{"─" * 60}')
    print(f'  Unweighted Score:    {overall:5.1f}%')
    print(f'  Weighted Score:      {weighted:5.1f}%')
    print(f'  (Scores >80/60/50/30 suggest stage transition)')
    print(f'{"=" * 60}')

    result = {
        'timestamp': time.time(),
        'datetime': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'stages': {
            'sensorimotor': stage1,
            'preoperational': stage2,
            'concrete_operational': stage3,
            'formal_operational': stage4,
        },
        'overall_cognitive_score': overall,
        'weighted_cognitive_score': weighted,
        'stage_transitions': {
            'sensorimotor': stage1["overall_score"] > 80,
            'preoperational': stage2["overall_score"] > 60,
            'concrete_operational': stage3["overall_score"] > 50,
            'formal_operational': stage4["overall_score"] > 30,
        },
    }

    # Append to growth chart
    _append_to_growth_chart(result)

    # Show growth trajectory
    history = _load_growth_chart()
    if len(history) > 1:
        prev = history[-2].get('overall_cognitive_score', 0)
        delta = overall - prev
        print(f'\n  Growth since last run: {delta:+.1f}% '
              f'({"↑ improving" if delta > 0 else "↓ declining" if delta < 0 else "→ stable"})')

    return result


def print_growth_chart():
    """Print the historical growth chart."""
    entries = _load_growth_chart()
    if not entries:
        print('  [Piaget] No growth chart data yet. Run the benchmark first.')
        return

    print(f'\n  Cognitive Growth Chart ({len(entries)} runs)')
    print(f'  {"Run":<5s} {"Date":<20s} {"Overall":>8s} {"Sens":>5s} {"Preop":>5s} '
          f'{"ConcOp":>6s} {"FormOp":>6s}')
    print(f'  {"─" * 56}')
    for i, entry in enumerate(entries):
        s = entry.get('stages', {})
        date = entry.get('datetime', '?')[:16]
        overall = entry.get('overall_cognitive_score', 0)
        sens = s.get('sensorimotor', {}).get('overall_score', 0)
        preop = s.get('preoperational', {}).get('overall_score', 0)
        conc = s.get('concrete_operational', {}).get('overall_score', 0)
        form = s.get('formal_operational', {}).get('overall_score', 0)
        print(f'  #{i+1:<3d} {date:<20s} {overall:>7.1f}% {sens:>4.0f}% '
              f'{preop:>4.0f}% {conc:>5.0f}% {form:>5.0f}%')


# Quick test (if run directly)
if __name__ == '__main__':
    print('Piaget Benchmark Runner')
    print('Load a model and run:')
    print('  from tabula_rasa.cognitive.piaget.runner import run_piaget_benchmark')
    print('  results = run_piaget_benchmark(model, tokenizer)')
    print()
    print_growth_chart()
