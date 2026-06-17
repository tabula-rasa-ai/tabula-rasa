"""
pattern_specialist.py — Stage 2: The Realm of Patterns.

Before learning words, the brain must learn that symbols can represent
patterns, not just quantities. This is the bridge from Math to Language.

Curriculum:
  1. String Reversal: "hello" → "olleh" (long-distance dependencies)
  2. Palindrome Detection: "racecar" → True (symmetry recognition)
  3. Symbol Counting: "aabbbcccc" → "a:2,b:3,c:4" (frequency tracking)
  4. Sequence Completion: "a-b-a-b" → "c-d-c-d" (pattern extrapolation)
  5. Alternation: "abab" → "cdcd" (rule discovery)

Tokenizer: a-z + 0-9 (no uppercase, no punctuation — pure pattern space)

Graduation: 98%+ on held-out pattern test set
"""

import random

import torch
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

# ═════════════════════════════════════════════════════════════════════
# PATTERN PROBLEM GENERATORS
# ═════════════════════════════════════════════════════════════════════

LETTERS = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'


def generate_reversal_problem(rng=None):
    """String reversal: learn long-distance dependencies.

    Example:    reverse(hello)
    Expected:   olleh

    Challenges the transformer's attention to track position from end.
    """
    if rng is None:
        rng = random.Random()
    length = rng.randint(3, 15)
    chars = ''.join(rng.choice(LETTERS) for _ in range(length))
    return f'reverse({chars})', chars[::-1]


def generate_palindrome_problem(rng=None):
    """Palindrome detection: symmetry recognition.

    Example:    is_palindrome(racecar)
    Expected:   True

    The model must compare first/last, second/second-last, etc.
    """
    if rng is None:
        rng = random.Random()

    if rng.random() > 0.5:
        # Generate a palindrome
        mid = ''.join(rng.choice(LETTERS) for _ in range(rng.randint(2, 8)))
        s = mid + mid[::-1]
        return f'is_palindrome({s})', 'True'
    else:
        # Generate a non-palindrome (guaranteed)
        s = ''.join(rng.choice(LETTERS) for _ in range(rng.randint(4, 12)))
        # Ensure it's not a palindrome by construction
        if s == s[::-1]:
            s = s + 'x'
        return f'is_palindrome({s})', 'False'


def generate_counting_problem(rng=None):
    """Symbol counting: frequency tracking.

    Example:    count(aabbbcccc)
    Expected:   a:2,b:3,c:4

    The model must track frequencies of each unique symbol.
    """
    if rng is None:
        rng = random.Random()

    # Pick 3-5 symbols
    n_symbols = rng.randint(2, 5)
    symbols = rng.sample(LETTERS, n_symbols)

    # Build string with varying counts
    parts = []
    counts = {}
    for s in symbols:
        count = rng.randint(1, 9)
        parts.append(s * count)
        counts[s] = count

    rng.shuffle(parts)
    s = ''.join(parts)

    # Expected: comma-separated symbol:count pairs (sorted by appearance)
    # Read back in original shuffled order within each symbol
    expected_parts = []
    seen = set()
    for c in s:
        if c not in seen:
            expected_parts.append(f'{c}:{counts[c]}')
            seen.add(c)

    return f'count({s})', ','.join(expected_parts)


def generate_sequence_problem(rng=None):
    """Sequence completion: pattern extrapolation.

    Example:    next(a-b-a-b)
    Expected:   c-d

    The model must discover the alternation pattern and project it forward.
    """
    if rng is None:
        rng = random.Random()

    mode = rng.choice(['alternating', 'incrementing', 'double'])

    if mode == 'alternating':
        # a-b-a-b -> next = c-d
        letters = rng.sample(LETTERS, 2)
        repeats = rng.randint(2, 4)
        pattern = f'{letters[0]}-{letters[1]}-' * repeats
        pattern = pattern.rstrip('-')
        # Next pair: shift both letters by 2 positions
        next_a = chr(ord(letters[0]) + 2) if ord(letters[0]) + 2 <= ord('z') else 'a'
        next_b = chr(ord(letters[1]) + 2) if ord(letters[1]) + 2 <= ord('z') else 'b'
        expected = f'{next_a}-{next_b}'
    elif mode == 'incrementing':
        # a-c-e -> next = g
        start_idx = LETTERS.index(rng.choice(LETTERS[:20]))
        step = rng.choice([2, 3, 4])
        n = rng.randint(2, 4)
        pattern_parts = []
        for i in range(n):
            idx = start_idx + i * step
            if idx >= len(LETTERS):
                break
            pattern_parts.append(LETTERS[idx])
        pattern = '-'.join(pattern_parts)
        next_idx = start_idx + n * step
        next_char = LETTERS[next_idx] if next_idx < len(LETTERS) else pattern_parts[-1]
        expected = next_char
    else:
        # double pattern: aa-bb-cc -> next = dd
        chars = rng.sample(LETTERS, rng.randint(2, 4))
        pattern = '-'.join(c * 2 for c in chars)
        next_char = chr(ord(chars[-1]) + 1) if ord(chars[-1]) + 1 <= ord('z') else 'a'
        expected = f'{next_char}{next_char}'

    return f'next({pattern})', expected


def generate_alternation_problem(rng=None):
    """Letter alternation: rule discovery.

    Example:    abab -> cdcd

    Discover the rule "advance each letter by 2 positions" and apply it.
    """
    if rng is None:
        rng = random.Random()

    n = rng.randint(2, 5)
    shift = rng.choice([1, 2, 3])
    chars = rng.sample(LETTERS[:20], n)
    pattern = ''.join(chars)

    shifted = ''.join(chr(ord(c) + shift) if ord(c) + shift <= ord('z') else 'a' for c in chars)

    # Repeat the pattern n times
    full_pattern = pattern * rng.randint(2, 3)
    full_shifted = shifted * (len(full_pattern) // n)

    return f'alternate({full_pattern})', full_shifted


# ═════════════════════════════════════════════════════════════════════
# PATTERN PROBLEM GENERATORS MAP
# ═════════════════════════════════════════════════════════════════════

PROBLEM_GENERATORS = [
    ('reversal', generate_reversal_problem),
    ('palindrome', generate_palindrome_problem),
    ('counting', generate_counting_problem),
    ('sequence', generate_sequence_problem),
    ('alternation', generate_alternation_problem),
]


def generate_pattern_problem(rng=None):
    """Generate a random pattern problem from any category."""
    if rng is None:
        rng = random.Random()
    _, generator = rng.choice(PROBLEM_GENERATORS)
    return generator(rng)


def generate_dataset(count: int, seed: int = 42) -> list:
    """Generate a dataset of pattern problems."""
    rng = random.Random(seed)
    return [generate_pattern_problem(rng) for _ in range(count)]


# ═════════════════════════════════════════════════════════════════════
# PATTERN TRAINING SESSION
# ═════════════════════════════════════════════════════════════════════

def train_pattern_specialist(model, tokenizer,
                              train_steps: int = 5000,
                              batch_size: int = 64,
                              learning_rate: float = 5e-4,
                              device: str = 'cpu') -> dict:
    """Train the Pattern Specialist on string manipulation tasks.

    The model learns to attend to long-distance dependencies
    (reversal needs end-to-start attention) and symbol frequencies.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    step = 0

    while step < train_steps:
        # Generate batch
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

        # Forward
        logits, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        # Evaluate
        if step % 500 == 0:
            predictions = logits.argmax(dim=-1)
            mask = y != -100
            match = (predictions == y) & mask
            acc = match.sum().item() / max(mask.sum().item(), 1)
            avg_loss = total_loss / step

            print(f'  [Pattern] Step {step:>5d}/{train_steps} | '
                  f'loss={avg_loss:.4f} | acc={acc*100:.1f}%')

    model.eval()
    return {
        'stage': 'patterns',
        'steps': train_steps,
        'final_loss': total_loss / max(train_steps, 1),
    }


# ═════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_patterns(model, tokenizer, num_problems: int = 200,
                       device: str = 'cpu') -> float:
    """Evaluate the Pattern Specialist on a held-out test set."""
    model.eval()
    model = model.to(device)

    problems = generate_dataset(num_problems, seed=999)
    correct = 0

    for inp, expected in problems:
        prompt = f'{inp}='
        generated = model.generate(tokenizer, prompt, max_new_tokens=20, temperature=0.0)
        answer = generated.split('=')[-1].strip() if '=' in generated else ''
        if answer == expected:
            correct += 1

    return correct / max(num_problems, 1)


# ═════════════════════════════════════════════════════════════════════
# DEMO
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer

    cfg = Config()
    cfg.use_value_head = True
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.max_seq_len = 128

    tok = MathTokenizer()
    # Add a-z and 0-9 for patterns
    for c in 'abcdefghijklmnopqrstuvwxyz0123456789-:,':
        if c not in tok.char_to_id:
            tok.add_token(c)
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    print(f'Pattern Specialist: {count_parameters(model):,} params')
    print(f'Tokenizer: {tok.vocab_size} tokens')

    print('\n=== Sample Problems ===')
    rng = random.Random(42)
    for name, generator in PROBLEM_GENERATORS:
        inp, exp = generator(rng)
        print(f'  {name:15s}: {inp} = {exp}')

    print('\n=== Training (demo: 200 steps) ===')
    results = train_pattern_specialist(model, tok, train_steps=200, batch_size=32)

    print('\n=== Evaluation ===')
    acc = evaluate_patterns(model, tok, num_problems=50)
    print(f'  Accuracy: {acc*100:.1f}%')
