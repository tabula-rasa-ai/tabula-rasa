"""
AutoDatasetGenerator — Synthetic data pipeline for blank-slate specialists.

Generates a full training dataset from a user's seed prompt using only
existing project resources (no external LLMs, no internet):

  1. TEMPLATE AUGMENTATION — paraphrase variants (e.g. "What is X?" → "Define X")
  2. HIPPOCAMPUS MINING — similar past Q&A from SQLite memory
  3. SOCRATIC SELF-PLAY — existing trained models generate answers
  4. RETRIEVAL BACKFILL — best word-overlap match from similar intents

Usage:
    from tabula_rasa.dataset_generator import AutoDatasetGenerator
    gen = AutoDatasetGenerator(intent, seed_prompt, manager)
    dataset = gen.generate()
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Optional

# ─── Template Augmentation ─────────────────────────────────────────

# Intent-specific template generators
TEMPLATE_GENERATORS = {
    'definition_question': {
        'templates': [
            "What is {topic}?",
            "What are {topic}?",
            "Define {topic}",
            "Explain {topic}",
            "Tell me about {topic}",
            "Describe {topic}",
            "What does {topic} mean?",
            "Give me the definition of {topic}",
            "What is meant by {topic}?",
            "Can you explain {topic}?",
        ],
        'topic_extract': [
            r'what is\s+(.+?)\??\s*$',
            r'what are\s+(.+?)\??\s*$',
            r'define\s+(.+?)\??\s*$',
            r'explain\s+(.+?)\??\s*$',
            r'what does\s+(.+?)\s+mean\??\s*$',
            r'tell me about\s+(.+?)\??\s*$',
            r'describe\s+(.+?)\??\s*$',
        ],
    },
    'explanation_question': {
        'templates': [
            "Why does {topic}?",
            "How does {topic}?",
            "Where does {topic}?",
            "When does {topic}?",
            "Explain why {topic}",
            "Tell me why {topic}",
            "What causes {topic}?",
            "How come {topic}?",
        ],
        'topic_extract': [
            r'why\s+(.+?)\??\s*$',
            r'how\s+(.+?)\??\s*$',
            r'where\s+(.+?)\??\s*$',
            r'when\s+(.+?)\??\s*$',
            r'explain\s+(.+?)\??\s*$',
        ],
    },
    'capability_question': {
        'templates': [
            "What can you do?",
            "What do you know?",
            "What are your skills?",
            "Tell me your capabilities",
            "What tasks can you perform?",
            "What are you good at?",
            "How can you help me?",
            "What features do you have?",
            "List your abilities",
            "What do you specialize in?",
        ],
        'topic_extract': [],
    },
    'greeting': {
        'templates': [
            "Hello",
            "Hi",
            "Hey",
            "Hi there",
            "Hello there",
            "Good morning",
            "Good evening",
            "Hey there",
            "Greetings",
            "Howdy",
        ],
        'topic_extract': [],
    },
    'conversation': {
        'templates': [
            "Tell me a joke",
            "Tell me a story",
            "Tell me something interesting",
            "What's new?",
            "How are you?",
            "What's up?",
            "Tell me something fun",
            "Give me a fun fact",
            "Entertain me",
            "Surprise me",
        ],
        'topic_extract': [],
    },
    'question': {
        'templates': [
            "What is {topic}?",
            "Who is {topic}?",
            "When did {topic}?",
            "Where is {topic}?",
            "Why does {topic}?",
            "How does {topic}?",
        ],
        'topic_extract': [
            r'what\s+(?:is|are|was|were)\s+(.+?)\??\s*$',
            r'who\s+(?:is|are|was|were)\s+(.+?)\??\s*$',
            r'when\s+(?:is|are|was|were|did|does)\s+(.+?)\??\s*$',
            r'where\s+(?:is|are|was|were|did|does)\s+(.+?)\??\s*$',
            r'why\s+(?:is|are|was|were|did|does)\s+(.+?)\??\s*$',
            r'how\s+(?:is|are|was|were|did|does)\s+(.+?)\??\s*$',
        ],
    },
    'unknown': {
        'templates': [
            "{topic}",
            "Tell me about {topic}",
            "What is {topic}?",
            "Explain {topic}",
        ],
        'topic_extract': [],
    },
}

WORD_REPLACEMENTS = {
    'AI': ['Artificial Intelligence', 'machine learning', 'neural networks'],
    'machine learning': ['AI', 'deep learning', 'ML', 'artificial intelligence'],
    'neural network': ['neural net', 'ANN', 'network', 'deep learning model'],
    'transformer': ['transformer model', 'neural transformer', 'attention model'],
    'pytorch': ['torch', 'PyTorch framework'],
    'python': ['Python programming', 'Python language', 'Python code'],
    'algorithm': ['method', 'approach', 'technique', 'procedure'],
    'function': ['method', 'routine', 'procedure', 'operation'],
}


class AutoDatasetGenerator:
    """Generates synthetic training data using only blank-slate resources."""

    def __init__(self, intent: str, seed_prompt: str, seed_answer: str,
                 skill_manager=None, max_pairs: int = 20):
        self.intent = intent
        self.seed = (seed_prompt, seed_answer)
        self.manager = skill_manager
        self.max_pairs = max_pairs
        self.pairs: list[tuple[str, str]] = []
        self.generated_from = []  # tracking for curriculum ordering
        self._topic = self._extract_topic(seed_prompt)

    def _extract_topic(self, text: str) -> str:
        """Extract the 'topic' from a question using regex patterns."""
        config = TEMPLATE_GENERATORS.get(self.intent, {})
        for pattern in config.get('topic_extract', []):
            m = re.search(pattern, text.lower().strip())
            if m:
                topic = m.group(1).strip()
                # Strip trailing punctuation that got captured
                topic = topic.rstrip('?!.,;:"\' ')
                return topic
        return text.strip().rstrip('?!.,;:"\' ')

    # ─── Stage 1: Template Augmentation ────────────────────────

    def generate_templates(self) -> list[tuple[str, str]]:
        """Generate template variations of the seed question."""
        config = TEMPLATE_GENERATORS.get(self.intent, {})
        templates = config.get('templates', [])
        if not templates or not self._topic:
            return []

        topic = self._topic
        pairs = []

        for tmpl in templates:
            try:
                question = tmpl.format(topic=topic)
            except KeyError:
                continue
            # Don't duplicate exact seed
            if question.lower().strip() == self.seed[0].lower().strip():
                continue
            pairs.append((question, self.seed[1]))  # same answer

        # Also generate synonym variants by swapping words
        extra = []
        for q, a in pairs:
            for old_word, replacements in WORD_REPLACEMENTS.items():
                if old_word in q.lower():
                    for new_word in replacements:
                        variant = re.sub(
                            re.escape(old_word), new_word, q, flags=re.IGNORECASE
                        )
                        if variant != q and (variant, a) not in extra:
                            extra.append((variant, a))
        pairs.extend(extra)

        random.shuffle(pairs)
        self.generated_from.append('template')
        return pairs[:min(len(pairs), self.max_pairs // 2)]

    # ─── Stage 2: Hippocampus Mining ───────────────────────────

    def mine_hippocampus(self) -> list[tuple[str, str]]:
        """Query SQLite hippocampus for semantically similar past interactions."""
        try:
            from egefalos.hippocampus import get_recent, get_old_memories
            from pathlib import Path

            db_path = Path('memory/hippocampus.db')
            if not db_path.exists():
                return []

            experiences = get_recent(limit=200, tier='active')
            old = get_old_memories(limit=100, tier='longterm')
            experiences.extend(old)

            if not experiences:
                return []

            seed_words = set(self.seed[0].lower().split())
            scored = []
            for exp in experiences:
                input_text = exp.get('input_text', '') or ''
                output_text = exp.get('output_text', '') or exp.get('response', '') or ''
                if not input_text or not output_text:
                    continue
                if len(output_text) < 3:
                    continue
                exp_words = set(input_text.lower().split())
                if not exp_words:
                    continue
                overlap = len(seed_words & exp_words) / max(len(exp_words), 1)
                if overlap >= 0.2:  # at least 20% word overlap
                    scored.append((overlap, input_text, output_text))

            scored.sort(key=lambda x: -x[0])
            pairs = []
            seen_questions = {self.seed[0].lower().strip()}
            for _, q, a in scored[:15]:
                if q.lower().strip() not in seen_questions:
                    pairs.append((q, a))
                    seen_questions.add(q.lower().strip())

            self.generated_from.append('hippocampus')
            return pairs
        except Exception as e:
            print(f'  [*] Hippocampus mining skipped: {e}')
            return []

    # ─── Stage 3: Socratic Self-Play ───────────────────────────

    def generate_by_self_play(self) -> list[tuple[str, str]]:
        """Use existing trained models to generate answers for synthetic questions."""
        if not self.manager or not hasattr(self.manager, 'models'):
            return []
        if not self.manager.models:
            return []

        pairs = []
        seen_questions = {self.seed[0].lower().strip()}

        # Try each existing model to generate answers for template questions
        templates = self.generate_templates()[:5]  # Only generate 5 via self-play
        for question, _ in templates:
            if question.lower().strip() in seen_questions:
                continue
            seen_questions.add(question.lower().strip())

            answer = self._try_all_models(question)
            if answer and len(answer) > 3 and not self._is_garbage(answer):
                pairs.append((question, answer))

        self.generated_from.append('self_play')
        return pairs

    def _try_all_models(self, question: str) -> Optional[str]:
        """Try all loaded models to answer a question, return best answer."""
        if not self.manager:
            return None

        question_with_eq = question if question.endswith('=') else question + '='
        answers = []

        for skill_name, model in self.manager.models.items():
            if skill_name not in self.manager.tokenizers:
                continue
            tok = self.manager.tokenizers[skill_name]
            try:
                from egefalos.tabula_rasa import scale_config
                sc = scale_config(skill_name, self.manager.skill_levels.get(skill_name, 0))
                full = model.generate(
                    tok, question_with_eq,
                    max_new_tokens=sc.get('max_tokens', 40),
                    temperature=0.3,
                    top_k=5,
                )
                if '=' in full:
                    ans = full.split('=', 1)[1].strip()
                else:
                    ans = full.strip()
                # Clean special tokens
                ans = ans.replace('<EOS>', '').replace('<PAD>', '').strip()
                if len(ans) >= 3 and not self._is_garbage(ans):
                    answers.append((len(ans), ans))
            except Exception:
                continue

        # Sort by length (longer = more contentful)
        answers.sort(key=lambda x: -x[0])
        return answers[0][1] if answers else None

    @staticmethod
    def _is_garbage(text: str) -> bool:
        """Check if generated text is garbage (repetitive, keyboard-smash, or too short)."""
        if len(text) < 3:
            return True
        # Check for excessive repetition
        if len(set(text)) <= 2 and len(text) >= 5:
            return True
        # Check for same token repeated
        words = text.split()
        if len(words) >= 3 and all(w == words[0] for w in words[:5]):
            return True
        # Check for non-alphanumeric junk
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.3:
            return True
        # Check for keyboard-smash: high consonant clusters, no vowels in long strings
        vowels = set('aeiouyAEIOUY')
        has_vowel = any(c in vowels for c in text)
        if len(text) >= 8 and not has_vowel:
            return True
        # Check for all-caps smash (GWDGDWWDG type)
        if len(text) >= 6:
            upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            lower_ratio = sum(1 for c in text if c.islower()) / max(len(text), 1)
            if upper_ratio > 0.8 and lower_ratio < 0.1:
                return True
        # Check for low unique chars (repetitive garbage like "ABABABABAB")
        if len(text) >= 10 and len(set(text)) <= 4:
            return True
        return False

    # ─── Stage 4: Retrieval Backfill ───────────────────────────

    def backfill_from_similar_intents(self) -> list[tuple[str, str]]:
        """Pull the closest-matching Q&A from existing datasets of related intents."""
        if not self.manager:
            return []

        pairs = []
        seed_words = set(self.seed[0].lower().split())

        # Scan all existing datasets for related content
        datasets_dir = Path('datasets')
        if not datasets_dir.exists():
            return []

        seen_questions = {self.seed[0].lower().strip()}

        for ds_file in datasets_dir.glob('*.json'):
            if ds_file.stem == self.intent:  # Skip our own intent
                continue
            try:
                data = json.loads(ds_file.read_text(encoding='utf-8'))
                for q, a in data:
                    if q.lower().strip() in seen_questions:
                        continue
                    q_words = set(q.lower().split())
                    if not q_words:
                        continue
                    overlap = len(seed_words & q_words) / max(len(q_words), 1)
                    if overlap >= 0.25 and len(a) > 5:
                        pairs.append((q, a))
                        seen_questions.add(q.lower().strip())
            except Exception:
                continue

        pairs.sort(key=lambda x: -len(set(x[0].lower().split()) & seed_words))
        self.generated_from.append('backfill')
        return pairs[:max(0, self.max_pairs - len(self.pairs) - 1)]

    # ─── Generation Pipeline ───────────────────────────────────

    def generate(self) -> list[tuple[str, str]]:
        """Run the full 4-stage generation pipeline. Returns deduplicated pairs."""

        # Stage 0: Seed pair (always included)
        self.pairs = [self.seed]

        # Stage 1: Template augmentation
        templates = self.generate_templates()
        for q, a in templates:
            if len(self.pairs) >= self.max_pairs:
                break
            if q.lower().strip() != self.seed[0].lower().strip():
                if not any(q.lower().strip() == p[0].lower().strip() for p in self.pairs):
                    self.pairs.append((q, a))

        # Stage 2: Hippocampus mining
        mined = self.mine_hippocampus()
        for q, a in mined:
            if len(self.pairs) >= self.max_pairs:
                break
            if not any(q.lower().strip() == p[0].lower().strip() for p in self.pairs):
                self.pairs.append((q, a))

        # Stage 3: Socratic self-play (generates fresh answers)
        self_played = self.generate_by_self_play()
        for q, a in self_played:
            if len(self.pairs) >= self.max_pairs:
                break
            if a and not any(q.lower().strip() == p[0].lower().strip() for p in self.pairs):
                self.pairs.append((q, a))

        # Stage 4: Backfill from similar intents
        backfilled = self.backfill_from_similar_intents()
        for q, a in backfilled:
            if len(self.pairs) >= self.max_pairs:
                break
            if not any(q.lower().strip() == p[0].lower().strip() for p in self.pairs):
                self.pairs.append((q, a))

        # Curate: remove exact duplicates
        seen = set()
        deduped = []
        for q, a in self.pairs:
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append((q, a))
        self.pairs = deduped

        return self.pairs


def generate_dataset(intent: str, seed_prompt: str, seed_answer: str,
                     skill_manager=None, max_pairs: int = 20,
                     output_path: Optional[str] = None) -> list[tuple[str, str]]:
    """Convenience function: create and run AutoDatasetGenerator, save result."""
    gen = AutoDatasetGenerator(intent, seed_prompt, seed_answer, skill_manager, max_pairs)
    pairs = gen.generate()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'  [*] AutoDatasetGenerator: {len(pairs)} pairs from {gen.generated_from}')
    return pairs
