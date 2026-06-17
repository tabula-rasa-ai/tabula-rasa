"""Neuro-Symbolic Tool Use — <MATH_CALL> token system.

When the Language specialist encounters a quantitative problem,
it emits a <MATH_CALL> token. The system intercepts this, routes
the math to the Math specialist, and injects the answer back.

This prevents the language model from guessing at arithmetic
and preserves the specialist architecture's honesty.
"""

import re, json
from pathlib import Path

MATH_CALL_TOKEN = '<MATH_CALL>'
MATH_RESULT_TOKEN = '<MATH_RESULT>'


def detect_math_call(text: str) -> list[dict]:
    """Detect math expressions in text that need solving.

    Returns list of {expression, position} dicts.
    """
    problems = []

    # Pattern: "X + Y", "X - Y", "X * Y", "X / Y"
    patterns = [
        (r'(\d+)\s*\+\s*(\d+)', lambda a, b: str(int(a) + int(b))),
        (r'(\d+)\s*-\s*(\d+)', lambda a, b: str(int(a) - int(b))),
        (r'(\d+)\s*\*\s*(\d+)', lambda a, b: str(int(a) * int(b))),
        (r'(\d+)\s*/\s*(\d+)', lambda a, b: str(int(a) // int(b)) if int(b) != 0 else None),
        # Natural language patterns
        (r'(\d+)\s*(apples?|oranges?|dollars?|items?|students?|people?|times?|groups?|books?)\s+(and|plus|add|then|more|remaining)\s+(\d+)',
         lambda a, _, __, b: str(int(a) + int(b))),
        (r'(\d+)\s*(apples?|oranges?|dollars?|items?|students?|people?|times?|groups?|books?)\s+(and|minus|less|eat|take|remove|lose)\s+(\d+)',
         lambda a, _, __, b: str(int(a) - int(b))),
        (r'(\d+)\s+(times|multiplied by|groups? of|each|per)\s+(\d+)',
         lambda a, _, b: str(int(a) * int(b))),
        (r'(\d+)\s+(divided by|split into|per|each|among)\s+(\d+)',
         lambda a, _, b: str(int(a) // int(b)) if int(b) != 0 else None),
        (r'total\s+(?:of|cost|is)?\s*(\d+)\s*(?:\*|x|times)\s*(\d+)',
         lambda a, b: str(int(a) * int(b))),
    ]

    for pattern, solver in patterns:
        for m in re.finditer(pattern, text):
            expr = m.group(0)
            answer = solver(m.group(1), m.group(2))
            if answer is not None:
                problems.append({
                    'expression': expr,
                    'answer': answer,
                    'start': m.start(),
                    'end': m.end(),
                })

    # Deduplicate by position
    seen = set()
    unique = []
    for p in problems:
        key = (p['start'], p['end'])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def solve_problems(text: str) -> tuple[str, list[dict]]:
    """Replace math expressions in text with <MATH_CALL> + answers.

    Returns (processed_text, solved_problems)
    """
    problems = detect_math_call(text)
    if not problems:
        return text, []

    # Process in reverse order to preserve positions
    result = text
    solved = []
    for p in reversed(problems):
        call = f'{MATH_CALL_TOKEN}{p["expression"]}{MATH_RESULT_TOKEN}{p["answer"]}'
        result = result[:p['start']] + call + result[p['end']:]
        solved.append(p)

    return result, solved


def extract_and_solve(math_call_text: str, math_specialist=None, tokenizer=None) -> str:
    """Process text with <MATH_CALL> tokens through a math specialist.

    If no specialist provided, uses built-in Python eval for simple math.
    """
    result = math_call_text

    # Find all MATH_CALL...MATH_RESULT blocks
    pattern = re.escape(MATH_CALL_TOKEN) + r'(.+?)' + re.escape(MATH_RESULT_TOKEN) + r'(\d+)'
    for m in re.finditer(pattern, result):
        expr = m.group(1)
        computed = m.group(2)

        # If we have a math specialist, use it for verification
        if math_specialist and tokenizer:
            try:
                from tabula_rasa.model import MathTransformer
                prompt = f'{expr}='
                out = math_specialist.generate(tokenizer, prompt, max_new_tokens=10,
                                               temperature=0.1, top_k=3)
                specialist_answer = ''.join(c for c in (out.split('=')[-1] if '=' in out else '')
                                            if c.isdigit() or c == '-')
                if specialist_answer:
                    computed = specialist_answer
            except:
                pass

        # Replace with just the answer
        full_pattern = re.escape(MATH_CALL_TOKEN) + re.escape(expr) + re.escape(MATH_RESULT_TOKEN) + re.escape(computed)
        result = re.sub(full_pattern, computed, result, count=1)

    return result


def math_call_train_data() -> list[tuple]:
    """Generate training data for the <MATH_CALL> token pattern."""
    examples = [
        ("I have five apples and eat two. How many remain?",
         f"I have 5 apples and eat 2. How many remain? {MATH_CALL_TOKEN}5-2{MATH_RESULT_TOKEN}3"),
        ("If there are twelve students and four groups.",
         f"If there are 12 students and 4 groups. {MATH_CALL_TOKEN}12/4{MATH_RESULT_TOKEN}3"),
        ("Buy three items at five dollars each.",
         f"Buy 3 items at 5 dollars each. {MATH_CALL_TOKEN}3*5{MATH_RESULT_TOKEN}15"),
    ]
    return examples


if __name__ == '__main__':
    print('Neuro-Symbolic Tool Use — <MATH_CALL>\n')

    # Test detection
    tests = [
        "I have 5 apples and eat 2. How many remain?",
        "If there are 12 students divided into 4 groups.",
        "Buy 3 items at 5 dollars each. Total cost?",
    ]

    for text in tests:
        problems = detect_math_call(text)
        processed, solved = solve_problems(text)
        print(f'Input:  {text}')
        print(f'Solved: {len(solved)} problems')
        for p in solved:
            print(f'  {p["expression"]} = {p["answer"]}')
        print(f'Output: {processed}')
        print()

    # Test training data
    print('Training data samples:')
    for inp, out in math_call_train_data():
        print(f'  Input:  {inp}')
        print(f'  Output: {out}')
        print()
