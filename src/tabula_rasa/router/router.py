"""
router.py — The Neural Router: dispatches prompts to the correct frozen specialist.

Tabula Rasa uses a Mixture-of-Experts architecture where each cognitive stage
has its own specialist with a frozen, immutable tokenizer. The Router is the
gateway that decides which specialist handles each incoming prompt.

Routing Rules (deterministic, no learned model needed):
  Math (Stage 1):     Contains digits + operators (+-*/=)
  Patterns (Stage 2): Contains 'reverse', 'palindrome', 'count', 'next'
  Taxonomy (Stage 3): Contains 'is_a', 'chain', 'common', 'is_not'
  State (Stage 4):    Contains 'I have', 'How many', inventory words
  Grammar (Stage 5):  Contains 'grammar(' prefix
  Dialectics (Stage 6): Everything else (Socratic debate, philosophy)

The Router maintains a dispatch table and confidence scores.
"""

import re
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════
# ROUTING RULES — Pattern-based dispatch
# ═════════════════════════════════════════════════════════════════════

class RoutingRule:
    """A routing rule that maps prompt patterns to specialists."""

    def __init__(self, specialist: str, stage: int, patterns: list,
                 priority: int = 0, description: str = ''):
        self.specialist = specialist
        self.stage = stage
        self.patterns = [re.compile(p, re.IGNORECASE) if isinstance(p, str) else p for p in patterns]
        self.priority = priority
        self.description = description

    def match(self, prompt: str) -> bool:
        """Check if the prompt matches any of this rule's patterns."""
        return any(p.search(prompt) for p in self.patterns)


# ═════════════════════════════════════════════════════════════════════
# ROUTING TABLE
# ═════════════════════════════════════════════════════════════════════
# Ordered by priority (higher = checked first)

ROUTING_TABLE = [
    # Stage 1: Math
    RoutingRule('math_general', 1, [
        r'^\d+\s*[+\-*/=]\s*\d',    # 12+34=
        r'^calculate\b', r'^solve\b',
        r'^what is\s+\d+', r'^compute\b',
        r'^evaluate\b',
        r'\d+\s*[+\-*/]\s*\d+\s*[=]?$',
    ], priority=10, description='Arithmetic expressions'),

    # Stage 2: Patterns
    RoutingRule('pattern_specialist', 2, [
        r'^reverse\b', r'^palindrome\b',
        r'^count\(', r'^next\(',
        r'^alternate\b', r'^pattern\b',
        r'^sequence\b', r'^is_palindrome\b',
        r'^sort_string\b',
    ], priority=9, description='String patterns and manipulation'),

    # Stage 3: Taxonomy
    RoutingRule('taxonomy_specialist', 3, [
        r'^is_a\(', r'^is_not\b',
        r'^chain\(', r'^common\b',
        r'^taxonomy\b', r'^category\b',
        r'\bis a\s+\w+', r'\bis an?\s+\w+',
        r'^what is\s+(a|an)\s+\w+',
        r'^hierarchy\b',
    ], priority=8, description='Is-A taxonomy and categorization'),

    # Stage 4: State tracking
    RoutingRule('state_specialist', 4, [
        r'^I have\b', r'^How many\b',
        r'\b(apple|orange|banana|book|pen|coin|ticket|cookie|candy|marble)s?\b',
        r'\b(buy|eat|give|find|lose|sell|collect|receive)\b',
        r'\b(position|speed|distance|height|temperature)\b',
        r'\bincreases?\b', r'\bdecreases?\b',
    ], priority=7, description='State tracking and causality'),

    # Stage 5: Grammar
    RoutingRule('grammar_specialist', 5, [
        r'^grammar\(', r'^is_grammatical\b',
        r'^check_grammar\b', r'^correct_sentence\b',
        r'^subject_verb\b',
    ], priority=6, description='Grammar verification'),

    # Stage 6: Language (Dialectics) — catch-all for natural language
    RoutingRule('language_general', 6, [
        r'.*',  # Matches everything (catch-all)
    ], priority=0, description='General language and dialectics'),
]


# ═════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════

class Router:
    """Routes prompts to the correct cognitive specialist.

    The Router is a deterministic pattern-matcher that:
    1. Checks the prompt against priority-ordered rules
    2. Returns the best-matching specialist + confidence
    3. Falls through to the general language specialist for unknown prompts
    """

    def __init__(self):
        self.rules = sorted(ROUTING_TABLE, key=lambda r: -r.priority)
        self.stats = {'routed': 0, 'by_specialist': {}}

    def route(self, prompt: str) -> dict:
        """Route a prompt to the most appropriate specialist.

        Returns:
            {
                'specialist': str (name of specialist),
                'stage': int (1-6),
                'confidence': float (0-1),
                'rule': str (description of match rule),
            }
        """
        self.stats['routed'] += 1

        for rule in self.rules:
            if rule.match(prompt):
                confidence = min(1.0, rule.priority / 10.0 + 0.5)

                # Track stats
                self.stats['by_specialist'][rule.specialist] = \
                    self.stats['by_specialist'].get(rule.specialist, 0) + 1

                return {
                    'specialist': rule.specialist,
                    'stage': rule.stage,
                    'confidence': confidence,
                    'rule': rule.description,
                }

        # Catch-all: language general (Stage 6)
        self.stats['by_specialist']['language_general'] = \
            self.stats['by_specialist'].get('language_general', 0) + 1

        return {
            'specialist': 'language_general',
            'stage': 6,
            'confidence': 0.3,
            'rule': 'General language (catch-all)',
        }

    def route_batch(self, prompts: list) -> list:
        """Route multiple prompts."""
        return [self.route(p) for p in prompts]

    def get_stats(self) -> dict:
        """Get routing statistics."""
        return dict(self.stats)

    def reset_stats(self):
        """Reset routing statistics."""
        self.stats = {'routed': 0, 'by_specialist': {}}


# ═════════════════════════════════════════════════════════════════════
# ROUTER INTEGRATION — Connect to SkillManager
# ═════════════════════════════════════════════════════════════════════

def route_and_answer(prompt: str, manager=None) -> dict:
    """Route a prompt to the right specialist and get an answer.

    This is the main entry point for the Tabula Rasa AI.
    It replaces the simple router in tabula_rasa.py's detect_skill().

    Args:
        prompt: User's question or command
        manager: SkillManager instance (from tabula_rasa)

    Returns:
        dict with 'answer', 'specialist', 'stage', 'confidence'
    """
    router = Router()
    route_result = router.route(prompt)

    specialist_name = route_result['specialist']
    stage = route_result['stage']
    confidence = route_result['confidence']

    if manager is None:
        return {
            'answer': None,
            'specialist': specialist_name,
            'stage': stage,
            'confidence': confidence,
            'message': f'Routing to Stage {stage}: {specialist_name}',
        }

    # Try the routed specialist first
    if specialist_name in manager.models:
        # Generate answer
        model = manager.models[specialist_name]
        tok = manager.tokenizers[specialist_name]

        # Use the specialist's generate method
        import time
        t0 = time.time()

        if hasattr(model, 'generate'):
            answer = model.generate(tok, prompt, max_new_tokens=20, temperature=0.3)
        else:
            # Fallback: use the first model that works
            for name, m in manager.models.items():
                if hasattr(m, 'generate'):
                    tok = manager.tokenizers[name]
                    answer = m.generate(tok, prompt, max_new_tokens=20, temperature=0.3)
                    break
            else:
                answer = "I don't know about that yet."

        elapsed = time.time() - t0

        return {
            'answer': answer,
            'specialist': specialist_name,
            'stage': stage,
            'confidence': confidence,
            'time_ms': round(elapsed * 1000),
            'status': 'answered',
        }

    # Fallback: try any loaded specialist
    for name, model in manager.models.items():
        if hasattr(model, 'generate'):
            tok = manager.tokenizers.get(name)
            if tok:
                import time
                t0 = time.time()
                answer = model.generate(tok, prompt, max_new_tokens=20, temperature=0.3)
                elapsed = time.time() - t0
                return {
                    'answer': answer,
                    'specialist': name,
                    'stage': 'fallback',
                    'confidence': 0.3,
                    'time_ms': round(elapsed * 1000),
                    'status': 'answered',
                }

    return {
        'answer': None,
        'specialist': specialist_name,
        'stage': stage,
        'confidence': 0.0,
        'message': f"I don't have a specialist for this. Needs Stage {stage}: {specialist_name}",
        'status': 'unknown',
    }


# ═════════════════════════════════════════════════════════════════════
# ROUTING DEMO
# ═════════════════════════════════════════════════════════════════════

DEFAULT_ROUTER = Router()


def route(prompt: str) -> dict:
    """Convenience function: route a prompt."""
    return DEFAULT_ROUTER.route(prompt)


# ═════════════════════════════════════════════════════════════════════
# DEMO
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== Neural Router Demo ===\n')

    prompts = [
        '12+34=',
        'reverse(hello)',
        'is_a(apple,fruit)',
        'I have 5 apples. I eat 2. How many do I have?',
        'grammar(The dogs runs fast)',
        'What is the meaning of life?',
        'count(aabbbcccc)',
        '100*250=',
        'chain(dog,animal)',
        'The sun shines brightly.',
    ]

    router = Router()

    for prompt in prompts:
        result = router.route(prompt)
        print(f'  [{result["specialist"]:25s}] (Stage {result["stage"]}, '
              f'conf={result["confidence"]:.2f}) {prompt[:50]}')

    print(f'\nStats: {router.get_stats()}')
