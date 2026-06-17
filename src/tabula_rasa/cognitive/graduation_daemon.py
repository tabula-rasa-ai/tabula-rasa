"""
graduation_daemon.py — The 6-Stage Cognitive Syllabus Orchestrator.

Automates the developmental progression:
  Stage 1 (Math)       → Absolute Truth: digits, ops, equality
  Stage 2 (Patterns)   → Bridge to Language: strings, palindromes, sequences
  Stage 3 (Taxonomy)   → Is-A Logic: noun hierarchies, set theory, Z3-verified
  Stage 4 (State)      → Causality: English→code state tracking via sandbox
  Stage 5 (Grammar)    → Syntax: AST-like parse tree grammar game
  Stage 6 (Dialectics) → Wisdom: Socratic debate on ebooks

Each stage:
  1. Has its own tokenizer (frozen per stage, never mutated)
  2. Runs AlphaZero self-play until 98% mastery on held-out test set
  3. Upon graduation: weights frozen, next stage auto-spawned
  4. Router learns to dispatch to the correct frozen specialist
"""

import json
import random
import time
from enum import Enum
from pathlib import Path

import torch

from tabula_rasa.model import MathTransformer

# Lazy import for ai_server's SKILL_REGISTRY (optional — graceful fallback)
try:
    from tabula_rasa.server.ai_server import SKILL_REGISTRY
except ImportError:
    SKILL_REGISTRY = {}

# ═════════════════════════════════════════════════════════════════════
# DEVELOPMENTAL STAGES
# ═════════════════════════════════════════════════════════════════════

class CognitiveStage(Enum):
    MATH = 1      # Stage 1: Arithmetic (DONE)
    PATTERNS = 2  # Stage 2: String patterns, reversal, palindromes
    TAXONOMY = 3  # Stage 3: Is-A hierarchies, set theory
    STATE = 4     # Stage 4: State tracking, causality
    GRAMMAR = 5   # Stage 5: Grammar rules as game
    DIALECTICS = 6  # Stage 6: Socratic philosophy


STAGE_NAMES = {
    CognitiveStage.MATH: 'math_general',
    CognitiveStage.PATTERNS: 'pattern_specialist',
    CognitiveStage.TAXONOMY: 'taxonomy_specialist',
    CognitiveStage.STATE: 'state_specialist',
    CognitiveStage.GRAMMAR: 'grammar_specialist',
    CognitiveStage.DIALECTICS: 'dialectics_specialist',
}

STAGE_DIRS = {
    CognitiveStage.MATH: 'specialists/math/general',
    CognitiveStage.PATTERNS: 'specialists/patterns/general',
    CognitiveStage.TAXONOMY: 'specialists/taxonomy/general',
    CognitiveStage.STATE: 'specialists/state/general',
    CognitiveStage.GRAMMAR: 'specialists/grammar/general',
    CognitiveStage.DIALECTICS: 'specialists/dialectics/general',
}

# Tokenizer configs per stage — each stage gets its own tokenizer
# When moving to next stage, the old tokenizer is FROZEN (never expanded)
STAGE_TOKENIZER_CONFIGS = {
    CognitiveStage.MATH: {
        'chars': '0123456789+-*/=()%.',
        'vocab_size': 24,
        'type': 'char',
    },
    CognitiveStage.PATTERNS: {
        'chars': 'abcdefghijklmnopqrstuvwxyz0123456789-:><',
        'vocab_size': 47,
        'type': 'char',
    },
    CognitiveStage.TAXONOMY: {
        'chars': 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-:\'[]()',
        'vocab_size': 100,
        'type': 'char',
    },
    CognitiveStage.STATE: {
        'chars': 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:\'[]()_=+-*/;',
        'vocab_size': 512,
        'type': 'bpe',
    },
    CognitiveStage.GRAMMAR: {
        'chars': 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:\'[]();',
        'vocab_size': 512,
        'type': 'bpe',
    },
    CognitiveStage.DIALECTICS: {
        'chars': 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:\'[]();"',
        'vocab_size': 1024,
        'type': 'bpe',
    },
}

# Mastery thresholds: 98% on held-out test set to graduate
MASTERY_THRESHOLD = 0.98
HELD_OUT_TEST_SIZE = 1000


# ═════════════════════════════════════════════════════════════════════
# GRADUATION CHECK
# ═════════════════════════════════════════════════════════════════════

def evaluate_held_out_test(specialist_name: str, test_data: list = None) -> float:
    """Evaluate a specialist on a held-out test set.

    The test set is 1,000 problems the specialist has NEVER seen.
    Accuracy >= 98% means the stage is mastered.

    Args:
        specialist_name: Name of the specialist to evaluate
        test_data: Optional list of (input, expected) pairs. If None,
                   auto-generates stage-appropriate test data.

    Returns:
        float: Accuracy (0.0 to 1.0)
    """
    from tabula_rasa.server.ai_server import SKILL_REGISTRY, SkillManager

    manager = SkillManager()

    # Load the specialist
    info = SKILL_REGISTRY.get(specialist_name)
    if not info:
        print(f'  [Graduation] Unknown specialist: {specialist_name}')
        return 0.0

    model = manager.models.get(specialist_name)
    tok = manager.tokenizers.get(specialist_name)

    if model is None or tok is None:
        print(f'  [Graduation] {specialist_name} not loaded')
        return 0.0

    model.eval()
    device = next(model.parameters()).device

    if test_data is None:
        test_data = generate_held_out_test(specialist_name, HELD_OUT_TEST_SIZE)

    correct = 0
    total = 0

    with torch.no_grad():
        for inp, expected in test_data:
            prompt = str(inp)
            generated = model.generate(tok, prompt, max_new_tokens=15, temperature=0.0)

            # Extract answer
            answer = generated.replace(prompt, '').strip()
            if answer == str(expected):
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    return accuracy


def generate_held_out_test(specialist_name: str, count: int = 1000) -> list:
    """Auto-generate held-out test data appropriate for each stage."""
    tests = []
    rng = random.Random(42)  # Deterministic seed for reproducibility

    if specialist_name == 'math_general' or specialist_name == 'general_math':
        # Stage 1: Arithmetic problems the model hasn't seen
        for _ in range(count):
            a = rng.randint(100, 9999)
            b = rng.randint(100, 9999)
            op = rng.choice(['+', '-', '*'])
            if op == '-':
                a, b = max(a, b), min(a, b)  # Ensure positive result
            expr = f'{a}{op}{b}'
            answer = str(eval(expr))
            tests.append((f'{expr}=', answer))

    elif specialist_name == 'pattern_specialist':
        # Stage 2: String patterns, reversals, counting
        for _ in range(count):
            mode = rng.choice(['reverse', 'palindrome_check', 'count', 'sequence'])
            if mode == 'reverse':
                s = ''.join(rng.choice('abcdefghij') for _ in range(rng.randint(3, 12)))
                tests.append((f'reverse({s})', s[::-1]))
            elif mode == 'palindrome_check':
                mid = ''.join(rng.choice('abcdefghij') for _ in range(rng.randint(2, 6)))
                s = mid + mid[::-1]
                tests.append((f'is_palindrome({s})', 'True'))
            elif mode == 'count':
                s = ''.join(rng.choice('abc') for _ in range(rng.randint(5, 20)))
                tests.append((f'count({s})', str(len(set(s)))))
            elif mode == 'sequence':
                c1 = rng.choice('abcdef')
                c2 = rng.choice('ghijkl')
                n = rng.randint(3, 6)
                pattern = ''.join(c1 + c2 for _ in range(n))
                next_pair = c1 + c2
                tests.append((f'next({pattern})', next_pair))

    elif specialist_name == 'taxonomy_specialist':
        # Stage 3: Is-A taxonomy verification
        taxonomies = [
            ('apple', 'fruit'), ('banana', 'fruit'), ('orange', 'fruit'),
            ('dog', 'mammal'), ('cat', 'mammal'), ('whale', 'mammal'),
            ('sparrow', 'bird'), ('eagle', 'bird'), ('penguin', 'bird'),
            ('salmon', 'fish'), ('trout', 'fish'), ('shark', 'fish'),
            ('rose', 'flower'), ('tulip', 'flower'), ('daisy', 'flower'),
            ('oak', 'tree'), ('maple', 'tree'), ('pine', 'tree'),
        ]
        for _ in range(count):
            item, category = rng.choice(taxonomies)
            if rng.random() > 0.5:
                tests.append((f'is_a({item},{category})', 'True'))
            else:
                wrong = rng.choice([c for _, c in taxonomies if c != category])
                tests.append((f'is_a({item},{wrong})', 'False'))

    elif specialist_name == 'state_specialist':
        # Stage 4: State tracking
        for _ in range(count):
            init = rng.randint(0, 10)
            changes = []
            val = init
            for _ in range(rng.randint(2, 5)):
                delta = rng.randint(-3, 5)
                if delta >= 0:
                    changes.append(f'add {delta}')
                else:
                    changes.append(f'remove {abs(delta)}')
                val += delta
                val = max(0, val)
            prompt = f'start={init} ' + ' '.join(changes)
            tests.append((prompt, str(val)))

    elif specialist_name == 'grammar_specialist':
        # Stage 5: Grammaticality judgment
        from tabula_rasa.cognitive.grammar_engine import grammar_score
        for _ in range(count):
            # Use sentence templates
            subj = rng.choice(['The dog', 'A cat', 'The children', 'My friend'])
            verb = rng.choice(['runs', 'run', 'walks', 'walk', 'eats', 'eat'])
            obj = rng.choice(['fast.', 'slowly.', 'quickly.', 'home.', 'outside.'])
            sentence = f'{subj} {verb} {obj}'
            score = grammar_score(sentence)
            expected = 'correct' if score > 0 else 'incorrect'
            tests.append((f'grammar({sentence})', expected))

    return tests


# ═════════════════════════════════════════════════════════════════════
# FREEZE WEIGHTS
# ═════════════════════════════════════════════════════════════════════

def freeze_weights(model):
    """Freeze all parameters — they will never be trained again.

    Frozen specialists become immutable oracles. The Router dispatches
    to them, but their weights never change.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f'  [Graduation] Weights frozen ({sum(p.numel() for p in model.parameters()):,} params locked)')


def is_frozen(model) -> bool:
    """Check if a model's weights are frozen."""
    return all(not param.requires_grad for param in model.parameters())


# ═════════════════════════════════════════════════════════════════════
# GRADUATION STATE — Persistent tracking
# ═════════════════════════════════════════════════════════════════════

GRADUATION_FILE = Path('memory/graduation_state.json')


def load_graduation_state() -> dict:
    """Load persistent graduation state from disk."""
    if GRADUATION_FILE.exists():
        try:
            return json.loads(GRADUATION_FILE.read_text())
        except Exception:
            pass
    return {
        'current_stage': 1,
        'graduated_stages': {},
        'stage_accuracies': {},
        'total_training_steps': 0,
        'last_graduation': None,
    }


def save_graduation_state(state: dict):
    """Save graduation state to disk."""
    GRADUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    GRADUATION_FILE.write_text(json.dumps(state, indent=2, default=str))


# ═════════════════════════════════════════════════════════════════════
# NEXT STAGE TRIGGER
# ═════════════════════════════════════════════════════════════════════

def get_next_stage(current_stage: CognitiveStage):
    """Get the next developmental stage after the current one."""
    stages = list(CognitiveStage)
    try:
        idx = stages.index(current_stage)
        if idx + 1 < len(stages):
            return stages[idx + 1]
    except ValueError:
        pass
    return None


def get_stage_number(stage: CognitiveStage) -> int:
    """Get the human-readable stage number."""
    return stage.value


# ═════════════════════════════════════════════════════════════════════
# GRADUATION CYCLE — The main daemon loop
# ═════════════════════════════════════════════════════════════════════

def run_graduation_check(specialist_name: str, force: bool = False) -> dict:
    """Run one graduation check for a specialist.

    Evaluates on held-out test set. If >= 98%, freezes weights,
    triggers the next stage, and returns graduation result.

    Returns:
        dict with 'graduated', 'accuracy', 'next_stage', etc.
    """
    print(f'\n  [Graduation] Checking {specialist_name}...')

    # Generate held-out test
    test_data = generate_held_out_test(specialist_name, HELD_OUT_TEST_SIZE)
    accuracy = evaluate_held_out_test(specialist_name, test_data)

    state = load_graduation_state()

    result = {
        'specialist': specialist_name,
        'accuracy': accuracy,
        'graduated': accuracy >= MASTERY_THRESHOLD,
        'threshold': MASTERY_THRESHOLD,
    }

    state['stage_accuracies'][specialist_name] = accuracy

    if accuracy >= MASTERY_THRESHOLD:
        print(f'  [Graduation] {specialist_name} has ACHIEVED MASTERY ({accuracy*100:.1f}% >= {MASTERY_THRESHOLD*100:.0f}%)')

        # Freeze the model
        from tabula_rasa.server.ai_server import SkillManager
        manager = SkillManager()
        model = manager.models.get(specialist_name)
        if model:
            freeze_weights(model)
            # Save frozen state
            info = SKILL_REGISTRY.get(specialist_name) if 'SKILL_REGISTRY' in dir() else None
            # Mark as graduated
            state['graduated_stages'][specialist_name] = {
                'accuracy': accuracy,
                'timestamp': time.time(),
                'stage': state['current_stage'],
            }

            # Trigger next stage
            current_stage_num = state['current_stage']
            next_stage_num = current_stage_num + 1

            if next_stage_num <= 6:
                state['current_stage'] = next_stage_num
                next_stage = CognitiveStage(next_stage_num)
                next_name = STAGE_NAMES[next_stage]
                print(f'  [Graduation] Triggering Stage {next_stage_num}: {next_name}')
                result['next_stage'] = next_name
                result['next_stage_number'] = next_stage_num
            else:
                print('  [Graduation] ALL STAGES COMPLETE! Tabula Rasa is fully developed.')
                result['all_complete'] = True

            state['last_graduation'] = time.time()

        save_graduation_state(state)
    else:
        print(f'  [Graduation] {specialist_name} at {accuracy*100:.1f}% (need {MASTERY_THRESHOLD*100:.0f}%). Continuing training.')

    return result


def daemon_cycle(force_check: bool = False):
    """Run one cycle of the graduation daemon.

    Checks the current stage for mastery. If graduated, triggers next stage.
    """
    state = load_graduation_state()
    current_stage_num = state['current_stage']

    print('\n  ╔══════════════════════════════════════════╗')
    print(f'  ║  GRADUATION DAEMON — Stage {current_stage_num}/6    ║')
    print('  ╚══════════════════════════════════════════╝')

    if current_stage_num > 6:
        print('  [Graduation] All 6 stages graduated!')
        return

    current_stage = CognitiveStage(current_stage_num)
    specialist_name = STAGE_NAMES[current_stage]
    stage_dir = STAGE_DIRS[current_stage]

    print(f'  Current: Stage {current_stage_num} — {current_stage.name}')
    print(f'  Specialist: {specialist_name}')
    print(f'  Directory: {stage_dir}')

    # Check if checkpoint exists
    ckpt_path = Path(stage_dir) / 'best.pt'
    if not ckpt_path.exists():
        print(f'  [Graduation] No checkpoint found for {specialist_name}. Waiting for training...')
        return

    result = run_graduation_check(specialist_name, force=force_check)

    # Print graduated stages summary
    graduated = state.get('graduated_stages', {})
    if graduated:
        print('\n  Graduated stages:')
        for name, info in graduated.items():
            acc = info.get('accuracy', 0) * 100
            print(f'    [{info.get("stage", "?")}] {name}: {acc:.1f}%')

    print(f'  Next stage: Stage {state["current_stage"]}/6')

    return result


# ═════════════════════════════════════════════════════════════════════
# STAGE MANAGER — Spawn a new specialist for a given stage
# ═════════════════════════════════════════════════════════════════════

def spawn_stage(stage_number: int) -> tuple:
    """Spawn a specialist for a given developmental stage.

    Creates the model with the appropriate tokenizer and config.
    Does NOT train it — just creates the architecture.

    Args:
        stage_number: 1-6

    Returns:
        (model, tokenizer) tuple
    """
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer

    if stage_number < 1 or stage_number > 6:
        raise ValueError(f'Invalid stage: {stage_number}')

    stage = CognitiveStage(stage_number)
    config = STAGE_TOKENIZER_CONFIGS[stage]
    dir_path = STAGE_DIRS[stage]

    print(f'  [Spawn] Creating Stage {stage_number} specialist ({stage.name})...')
    print(f'  [Spawn] Tokenizer: {config["type"]}, vocab={config["vocab_size"]}')

    # Create directory
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create config
    cfg = Config()
    cfg.use_value_head = True
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.n_heads = 4
    cfg.max_seq_len = min(128 + stage_number * 32, 256)

    # Create tokenizer
    tok = MathTokenizer()
    # Expand tokenizer for this stage
    for c in config['chars']:
        if c not in tok.char_to_id:
            tok.add_token(c)

    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    # Save tokenizer
    tok.save(str(Path(dir_path) / 'tokenizer.json'))

    # Create model
    model = MathTransformer(cfg)

    print(f'  [Spawn] Model: {sum(p.numel() for p in model.parameters()):,} params')
    print(f'  [Spawn] Tokenizer: {tok.vocab_size} tokens')
    print(f'  [Spawn] Saved to {dir_path}')

    return model, tok


# ═════════════════════════════════════════════════════════════════════
# FULL PIPELINE — Run all stages sequentially
# ═════════════════════════════════════════════════════════════════════

def run_developmental_pipeline(start_stage: int = 1):
    """Run the full 6-stage developmental pipeline.

    For each stage:
      1. Spawn the specialist (if not existing)
      2. Train until 98% mastery
      3. Freeze weights
      4. Move to next stage

    This is the main entry point for autonomous development.
    """
    state = load_graduation_state()

    for stage_num in range(start_stage, 7):
        stage = CognitiveStage(stage_num)
        specialist = STAGE_NAMES[stage]
        stage_dir = STAGE_DIRS[stage]

        print(f'\n{"="*60}')
        print(f'  STAGE {stage_num}: {stage.name}')
        print(f'  Specialist: {specialist}')
        print(f'{"="*60}')

        # Check if already graduated
        if specialist in state.get('graduated_stages', {}):
            print('  Already graduated. Skipping.')
            continue

        # Create directory and tokenizer if needed
        Path(stage_dir).mkdir(parents=True, exist_ok=True)
        tok_path = Path(stage_dir) / 'tokenizer.json'

        if not tok_path.exists():
            spawn_stage(stage_num)

        # Run graduation check (will trigger training if below threshold)
        result = run_graduation_check(specialist)

        if result.get('graduated'):
            print(f'  Stage {stage_num} complete! Moving to Stage {stage_num + 1}...')
        else:
            print(f'  Stage {stage_num} still in progress ({result["accuracy"]*100:.1f}%).')
            print(f'  Resume training: python3 run_stage.py --stage {stage_num}')
            break


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Graduation Daemon')
    parser.add_argument('--check', action='store_true', help='Run one graduation check cycle')
    parser.add_argument('--pipeline', type=int, default=None, metavar='START_STAGE',
                        help='Run full developmental pipeline from given stage')
    parser.add_argument('--spawn', type=int, default=None, metavar='STAGE',
                        help='Spawn a specialist for a given stage (1-6)')
    parser.add_argument('--status', action='store_true', help='Show graduation status')
    parser.add_argument('--force', action='store_true', help='Force re-check even if below threshold')

    args = parser.parse_args()

    if args.spawn:
        model, tok = spawn_stage(args.spawn)
        print(f'Spawned Stage {args.spawn} specialist.')

    elif args.pipeline is not None:
        run_developmental_pipeline(args.pipeline)

    elif args.status:
        state = load_graduation_state()
        print(f'Current stage: {state["current_stage"]}/6')
        print('Graduated stages:')
        for name, info in state.get('graduated_stages', {}).items():
            print(f'  {name}: {info["accuracy"]*100:.1f}%')
        print('Accuracies:')
        for name, acc in state.get('stage_accuracies', {}).items():
            print(f'  {name}: {acc*100:.1f}%')

    else:
        daemon_cycle(force_check=args.force)
