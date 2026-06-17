"""
grammar_specialist.py — Stage 5: The Realm of Syntax.

Grammar as "Game Rules" — exactly like AlphaZero treats Chess rules.
The CFG parser (grammar_engine.py) is the "Environment" that evaluates moves.

Curriculum:
  1. Subject-Verb Agreement: "The dogs runs" → -1 (illegal move)
  2. Determiner-Noun Agreement: "a apples" → -1
  3. Sentence Structure: must have NP + VP (+1 for valid parse tree)
  4. Tense Consistency: "Yesterday I go" → -1 (tense violation)
  5. Long-Distance Dependencies: "Although... , ..." → +1 if properly closed

The Value Head learns "Structural Intuition" — it inherently feels when
a sentence is grammatically incomplete before finishing generation.

Tokenizer: a-z + A-Z + punctuation
Graduation: 98%+ grammaticality accuracy
"""

import random

import torch
from torch.optim import AdamW

from tabula_rasa.cognitive.grammar_engine import (
    grammar_score,
)

# ═════════════════════════════════════════════════════════════════════
# GRAMMAR PROBLEM GENERATORS
# ═════════════════════════════════════════════════════════════════════

def generate_sv_agreement_problem(rng=None):
    """Generate a subject-verb agreement problem.

    Correct: "The dog runs" / "The dogs run"
    Incorrect: "The dog run" / "The dogs runs"
    """
    if rng is None:
        rng = random.Random()

    subj_singular = rng.choice(['The dog', 'A cat', 'My friend', 'This child', 'The bird'])
    subj_plural = rng.choice(['The dogs', 'The cats', 'My friends', 'The children', 'The birds'])
    verb_singular = rng.choice(['runs', 'walks', 'eats', 'sings', 'flies'])
    verb_plural = rng.choice(['run', 'walk', 'eat', 'sing', 'fly'])
    obj = rng.choice(['fast.', 'outside.', 'now.', 'quietly.', 'daily.'])

    if rng.random() > 0.5:
        sentence = f'{subj_singular} {verb_singular} {obj}'
        expected = 'correct'
    else:
        sentence = f'{subj_singular} {verb_plural} {obj}'
        expected = 'incorrect'

    return f'grammar({sentence})', expected


def generate_det_noun_problem(rng=None):
    """Generate a determiner-noun agreement problem.

    Correct: "an apple" / "the apples"
    Incorrect: "a apples" / "an dogs"
    """
    if rng is None:
        rng = random.Random()

    nouns_singular = ['apple', 'dog', 'cat', 'bird', 'tree', 'book', 'car']
    nouns_plural = ['apples', 'dogs', 'cats', 'birds', 'trees', 'books', 'cars']

    if rng.random() > 0.5:
        noun = rng.choice(nouns_singular)
        det = rng.choice(['a', 'an', 'the', 'this', 'that', 'my'])
        if det == 'an' and noun[0] not in 'aeiou':
            det = 'a'
        expected = 'correct'
    else:
        noun = rng.choice(nouns_plural)
        det = rng.choice(['a', 'an', 'this', 'that'])
        if det in ('a', 'an'):
            det = rng.choice(['these', 'those', 'the'])
            if rng.random() > 0.5:
                det = 'a'  # deliberately wrong
                expected = 'incorrect'
            else:
                expected = 'correct'
        else:
            expected = 'correct' if det in ('these', 'those', 'the', 'my', 'some', 'any') else 'incorrect'

    verb = rng.choice(['is', 'are', 'was', 'were'])
    adj = rng.choice(['red', 'big', 'small', 'fast', 'tall'])

    if random.Random().random() > 0.7:
        sentence = f'{det} {adj} {noun} {verb} here.'
    else:
        sentence = f'{det} {noun} {verb} here.'

    return f'grammar({sentence})', expected


def generate_structure_problem(rng=None):
    """Generate a sentence structure problem.

    Must have at minimum NP + VP.
    """
    if rng is None:
        rng = random.Random()

    if rng.random() > 0.5:
        # Well-formed: NP VP
        subj = rng.choice(['The sun', 'A bird', 'My mother', 'The children'])
        verb = rng.choice(['shines', 'flies', 'cooks', 'play'])
        obj = rng.choice(['brightly.', 'high.', 'dinner.', 'outside.'])
        sentence = f'{subj} {verb} {obj}'
        expected = 'correct'
    else:
        # Malformed: Fragment
        sentence = rng.choice([
            'The red apple',
            'Running quickly',
            'In the garden',
            'Because of the rain',
            'A',
            '',
            'The',
        ])
        expected = 'incorrect'

    return f'grammar({sentence})', expected


def generate_long_distance_problem(rng=None):
    """Generate a long-distance dependency problem.

    Correct: "Although it rained, we went outside."
    Incorrect: "Although it rained we went outside." (missing comma)
    """
    if rng is None:
        rng = random.Random()

    subordinate = rng.choice([
        'Although the weather was cold',
        'Because it was raining',
        'While the sun was shining',
        'When the bell rang',
        'If you study hard',
    ])
    main = rng.choice([
        'the children played outside',
        'we stayed inside',
        'they went for a walk',
        'everyone left the room',
        'you will succeed',
    ])

    if rng.random() > 0.5:
        sentence = f'{subordinate}, {main}.'
        expected = 'correct'
    else:
        sentence = f'{subordinate} {main}.'  # Missing comma
        expected = 'incorrect'

    return f'grammar({sentence})', expected


# ═════════════════════════════════════════════════════════════════════
# GRAMMAR GENERATORS MAP
# ═════════════════════════════════════════════════════════════════════

GRAMMAR_GENERATORS = [
    ('sv_agreement', generate_sv_agreement_problem),
    ('det_noun', generate_det_noun_problem),
    ('structure', generate_structure_problem),
    ('long_distance', generate_long_distance_problem),
]


def generate_grammar_problem(rng=None):
    """Generate a random grammar problem."""
    if rng is None:
        rng = random.Random()
    _, generator = rng.choice(GRAMMAR_GENERATORS)
    return generator(rng)


def generate_dataset(count: int, seed: int = 42) -> list:
    """Generate a dataset of grammar problems."""
    rng = random.Random(seed)
    return [generate_grammar_problem(rng) for _ in range(count)]


# ═════════════════════════════════════════════════════════════════════
# GRAMMAR GAME — The "Environment" evaluates each move
# ═════════════════════════════════════════════════════════════════════

def grammar_environment(sentence: str) -> float:
    """The Grammar "Environment" — evaluates a sentence move.

    Returns reward -1 to +1 based on:
    - Subject-Verb agreement
    - Determiner-Noun agreement
    - Sentence structure
    - Punctuation
    - Long-distance dependencies

    This is the AlphaZero Rules Engine for grammar.
    """
    return grammar_score(sentence)


# ═════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════

def train_grammar_specialist(model, tokenizer,
                              train_steps: int = 5000,
                              batch_size: int = 64,
                              learning_rate: float = 5e-4,
                              device: str = 'cpu') -> dict:
    """Train the Grammar Specialist."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    step = 0

    while step < train_steps:
        problems = generate_dataset(batch_size, seed=step)

        batch_inputs = []
        batch_targets = []

        for inp, expected in problems:
            text = f'{inp}={expected}'
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > model.config.max_seq_len:
                ids = ids[:model.config.max_seq_len]
            padded = ids + [tokenizer.pad_id] * (model.config.max_seq_len - len(ids))
            batch_inputs.append(padded[:-1])
            batch_targets.append(padded[1:])

        x = torch.tensor(batch_inputs, dtype=torch.long, device=device)
        y = torch.tensor(batch_targets, dtype=torch.long, device=device)
        y[y == tokenizer.pad_id] = -100

        logits, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 500 == 0:
            predictions = logits.argmax(dim=-1)
            mask = y != -100
            match = (predictions == y) & mask
            acc = match.sum().item() / max(mask.sum().item(), 1)
            print(f'  [Grammar] Step {step:>5d}/{train_steps} | '
                  f'loss={total_loss/step:.4f} | acc={acc*100:.1f}%')

    model.eval()
    return {
        'stage': 'grammar',
        'steps': train_steps,
        'final_loss': total_loss / max(train_steps, 1),
    }


# ═════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_grammar(model, tokenizer, num_problems: int = 200,
                      device: str = 'cpu') -> float:
    """Evaluate grammar judgment accuracy."""
    model.eval()
    model = model.to(device)
    problems = generate_dataset(num_problems, seed=999)
    correct = 0
    for inp, expected in problems:
        prompt = f'{inp}='
        generated = model.generate(tokenizer, prompt, max_new_tokens=10, temperature=0.0)
        answer = generated.split('=')[-1].strip() if '=' in generated else ''
        if answer.lower() == expected.lower():
            correct += 1
    return correct / max(num_problems, 1)


# ═════════════════════════════════════════════════════════════════════
# DEMO
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== Stage 5: Grammar Game ===\n')

    # Test the grammar environment on sample sentences
    tests = [
        ('The dog runs fast.', 'correct'),
        ('The dogs runs fast.', 'incorrect'),
        ('a apples is here.', 'incorrect'),
        ('The children play outside.', 'correct'),
        ('Although it rained, we stayed inside.', 'correct'),
        ('Although it rained we stayed inside.', 'incorrect'),
    ]

    for sentence, expected in tests:
        score = grammar_environment(sentence)
        judgment = 'correct' if score > 0 else 'incorrect'
        match = '✓' if judgment == expected else '✗'
        print(f'  {match} "{sentence}"')
        print(f'     Score: {score:+.2f} | Expected: {expected} | Got: {judgment}')

    # Show random problems
    print('\n=== Random Problems ===')
    rng = random.Random(42)
    for name, gen in GRAMMAR_GENERATORS:
        inp, exp = gen(rng)
        print(f'  {name:15s}: {inp} = {exp}')
