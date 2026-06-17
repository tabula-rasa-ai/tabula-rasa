"""
AutoDatasetGenerator — Synthetic data pipeline for blank-slate specialists.

Generates a full training dataset from a user's seed prompt using only
existing project resources (no external LLMs, no internet).

Stages:
1. TEMPLATE AUGMENTATION — paraphrase variants (e.g. "What is X?" → "Define X")
2. RETRIEVAL BACKFILL — best semantic match from similar intents
3. HIPPOCAMPUS MINING — similar past Q&A from SQLite memory
4. SOCRATIC SELF-PLAY — existing trained models generate answers
"""
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set
from collections import Counter

# ─── Configuration & Constants ─────────────────────────────────────

STOPWORDS: Set[str] = {
    'what', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'to', 'of', 'in', 
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under', 
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    'can', 'will', 'just', 'should', 'now', 'do', 'does', 'did', 'tell', 'me', 
    'about', 'give', 'you', 'your', 'i', 'my', 'mine', 'it', 'its', 'we', 'our'
}

TEMPLATE_GENERATORS = {
    'definition_question': {
        'templates': [
            "What is {topic}?", "What are {topic}?", "Define {topic}.", 
            "Explain {topic}.", "Tell me about {topic}.", "Describe {topic}.", 
            "What does {topic} mean?", "Give me the definition of {topic}.", 
            "What is meant by {topic}?", "Can you explain {topic}?"
        ],
        'topic_extract': [r'what is\s+(.+)', r'define\s+(.+)', r'explain\s+(.+)', r'describe\s+(.+)'],
    },
    'explanation_question': {
        'templates': [
            "Why does {topic}?", "How does {topic}?", "Where does {topic}?", 
            "When does {topic}?", "Explain why {topic}.", "What causes {topic}?"
        ],
        'topic_extract': [r'why\s+(.+)', r'how\s+(.+)', r'explain\s+(.+)'],
    },
    'capability_question': {
        'templates': [
            "What can you do?", "What do you know?", "What are your skills?", 
            "Tell me your capabilities.", "What tasks can you perform?"
        ],
        'topic_extract': [],
    },
    'greeting': {
        'templates': ["Hello.", "Hi.", "Hey.", "Hi there.", "Good morning.", "Greetings."],
        'topic_extract': [],
    },
    'conversation': {
        'templates': [
            "Tell me a joke.", "Tell me a story.", "Tell me something interesting.", 
            "What's new?", "How are you?", "Entertain me."
        ],
        'topic_extract': [],
    },
    'unknown': {
        'templates': ["{topic}.", "Tell me about {topic}.", "What is {topic}?"],
        'topic_extract': [],
    }
}

WORD_REPLACEMENTS = {
    'AI': ['Artificial Intelligence', 'machine learning', 'neural networks'],
    'machine learning': ['AI', 'deep learning', 'ML'],
    'neural network': ['neural net', 'ANN', 'deep learning model'],
    'transformer': ['transformer model', 'attention model'],
}


class AutoDatasetGenerator:
    """Generates synthetic training data using only blank-slate resources."""
    
    def __init__(self, intent: str, seed_prompt: str, seed_answer: str,
                 skill_manager=None, max_pairs: int = 20):
        self.intent = intent
        self.seed = (seed_prompt, seed_answer)
        self.manager = skill_manager
        self.max_pairs = max_pairs
        self.pairs: List[Tuple[str, str, int]] = []  # (question, answer, difficulty)
        self._topic = self._extract_topic(seed_prompt)

    def _extract_topic(self, text: str) -> str:
        """Extract the 'topic' from a question using robust regex."""
        config = TEMPLATE_GENERATORS.get(self.intent, {})
        text_clean = re.sub(r'[?.!]+$', '', text.lower().strip())
        
        for pattern in config.get('topic_extract', []):
            m = re.search(pattern, text_clean)
            if m:
                return m.group(1).strip()
        return text_clean

    def _calculate_semantic_score(self, text1: str, text2: str) -> float:
        """Calculates a stopword-filtered, TF-IDF-lite overlap score."""
        words1 = [w for w in re.findall(r'\w+', text1.lower()) if w not in STOPWORDS and len(w) > 1]
        words2 = [w for w in re.findall(r'\w+', text2.lower()) if w not in STOPWORDS and len(w) > 1]
        
        if not words1 or not words2: return 0.0
            
        counts1 = Counter(words1)
        counts2 = Counter(words2)
        
        common = set(counts1.keys()) & set(counts2.keys())
        if not common: return 0.0
            
        score = sum(min(counts1[w], counts2[w]) for w in common)
        norm = math.sqrt(len(words1) * len(words2))
        return score / norm if norm > 0 else 0.0

    # ─── Stage 1: Template Augmentation ────────────────────────

    def generate_templates(self, kb_answer: str = "") -> List[Tuple[str, str, int]]:
        config = TEMPLATE_GENERATORS.get(self.intent, {})
        templates = config.get('templates', [])
        if not templates or not self._topic: return []

        topic = self._topic
        pairs = []

        for tmpl in templates:
            try:
                question = tmpl.format(topic=topic)
            except KeyError:
                continue
            
            if question.lower().strip() == self.seed[0].lower().strip():
                continue
                
            # Prefer kb_answer for templates, fall back to seed
            ans = kb_answer if kb_answer and len(kb_answer) > 10 else self.seed[1]
            if len(ans) < 10 and '{topic}' not in ans:
                ans = f"Regarding {topic}, {ans}"
                
            pairs.append((question, ans, 1))  # Difficulty 1

        # Synonym variants
        extra = []
        for q, a, d in pairs:
            for old_word, replacements in WORD_REPLACEMENTS.items():
                if old_word in q.lower():
                    for new_word in replacements:
                        variant = re.sub(re.escape(old_word), new_word, q, flags=re.IGNORECASE)
                        if variant != q:
                            extra.append((variant, a, d))
        pairs.extend(extra)
        
        random.shuffle(pairs)
        return pairs[:min(len(pairs), self.max_pairs // 2)]

    # ─── Stage 2: Hippocampus Mining ───────────────────────────

    def mine_hippocampus(self) -> List[Tuple[str, str, int]]:
        try:
            from tabula_rasa.memory.hippocampus import get_recent, get_old_memories
            db_path = Path('memory/hippocampus.db')
            if not db_path.exists(): return []

            experiences = get_recent(limit=200, tier='active')
            experiences.extend(get_old_memories(limit=100, tier='longterm'))
            if not experiences: return []

            scored = []
            for exp in experiences:
                input_text = exp.get('input_text', '') or ''
                output_text = exp.get('output_text', '') or exp.get('response', '') or ''
                if not input_text or not output_text or len(output_text) < 3: continue
                
                overlap = self._calculate_semantic_score(self.seed[0], input_text)
                if overlap >= 0.15:  # Adjusted threshold for TF-IDF lite
                    scored.append((overlap, input_text, output_text))

            scored.sort(key=lambda x: -x[0])
            return [(q, a, 3) for _, q, a in scored[:15]]  # Difficulty 3
        except Exception as e:
            print(f'  [*] Hippocampus mining skipped: {e}')
            return []

    # ─── Stage 3: Socratic Self-Play ───────────────────────────

    def generate_by_self_play(self) -> List[Tuple[str, str, int]]:
        if not self.manager or not hasattr(self.manager, 'models') or not self.manager.models:
            return []

        pairs = []
        templates = self.generate_templates()[:5]
        
        for question, _, _ in templates:
            answer = self._try_all_models(question)
            if answer and len(answer) > 3 and not self._is_garbage(answer):
                pairs.append((question, answer, 4))  # Difficulty 4 (Hardest)

        return pairs

    def _try_all_models(self, question: str) -> Optional[str]:
        if not self.manager: return None
        question_with_eq = question if question.endswith('=') else question + '='
        answers = []

        for skill_name, model in self.manager.models.items():
            if skill_name not in self.manager.tokenizers: continue
            tok = self.manager.tokenizers[skill_name]
            try:
                from tabula_rasa.server.ai_server import scale_config
                sc = scale_config(skill_name, self.manager.skill_levels.get(skill_name, 0))
                full = model.generate(
                    tok, question_with_eq,
                    max_new_tokens=sc.get('max_tokens', 40),
                    temperature=0.3, top_k=5,
                )
                ans = full.split('=', 1)[1].strip() if '=' in full else full.strip()
                ans = ans.replace('<EOS>', '').replace('<PAD>', '').strip()
                
                if len(ans) >= 3 and not self._is_garbage(ans):
                    answers.append((len(ans), ans))
            except Exception:
                continue

        answers.sort(key=lambda x: -x[0])
        return answers[0][1] if answers else None

    @staticmethod
    def _is_garbage(text: str) -> bool:
        if len(text) < 3: return True
        if len(set(text)) <= 3 and len(text) >= 4: return True
        
        # Check for repeating n-grams (e.g., "I am I am I am")
        words = text.split()
        if len(words) >= 4:
            for n in range(1, 4):
                ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
                if len(set(ngrams)) <= len(ngrams) * 0.3:
                    return True
                    
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.3: return True
        
        vowels = set('aeiouyAEIOUY')
        if len(text) >= 8 and not any(c in vowels for c in text): return True
        
        return False

    # ─── Stage 4: Retrieval Backfill ───────────────────────────

    def backfill_from_similar_intents(self) -> List[Tuple[str, str, int]]:
        if not self.manager: return []
        
        pairs = []
        
        # Priority 1: Knowledge base (bundled real-world Q&A, highest quality)
        kb_path = Path('datasets/knowledge_base.json')
        if kb_path.exists():
            try:
                kb_data = json.loads(kb_path.read_text(encoding='utf-8'))
                # First check our own intent category
                if self.intent in kb_data:
                    for q, a in kb_data[self.intent]:
                        overlap = self._calculate_semantic_score(self.seed[0], q)
                        if overlap >= 0.20:
                            pairs.append((q, a, 1))  # Difficulty 1 (high quality, low difficulty)
                # Then check other intent categories
                for cat, entries in kb_data.items():
                    if cat == self.intent: continue
                    for q, a in entries:
                        overlap = self._calculate_semantic_score(self.seed[0], q)
                        if overlap >= 0.20:
                            pairs.append((q, a, 2))  # Difficulty 2
            except Exception as e:
                print(f'  [*] Knowledge base load failed: {e}')
        
        # Priority 2: Existing datasets (lower quality, might be user-created)
        datasets_dir = Path('datasets')
        if datasets_dir.exists():
            for ds_file in datasets_dir.glob('*.json'):
                if ds_file.stem == self.intent: continue
                try:
                    data = json.loads(ds_file.read_text(encoding='utf-8'))
                    for q, a in data:
                        if len(a) <= 5: continue
                        overlap = self._calculate_semantic_score(self.seed[0], q)
                        if overlap >= 0.20:
                            pairs.append((q, a, 3))  # Difficulty 3 (lower quality)
                except Exception:
                    continue

        pairs.sort(key=lambda x: x[2])  # Sort by difficulty (lowest first = best matches)
        return pairs[:max(0, self.max_pairs - len(self.pairs) - 1)]

    # ─── Generation Pipeline ───────────────────────────────────

    def generate(self) -> List[Tuple[str, str]]:
        # Try to find a real answer from knowledge base first
        seed_qa = (self.seed[0], self.seed[1])  # default: placeholder
        kb_path = Path('datasets/knowledge_base.json')
        if kb_path.exists():
            try:
                kb_data = json.loads(kb_path.read_text(encoding='utf-8'))
                best_match = None
                best_score = 0.0
                for cat, entries in kb_data.items():
                    for q, a in entries:
                        score = self._calculate_semantic_score(self.seed[0], q)
                        if score > best_score and len(a) > 10:
                            best_score = score
                            best_match = (q, a)
                if best_match and best_score >= 0.20:
                    # Use the kb answer instead of placeholder
                    seed_qa = (self.seed[0], best_match[1])
            except Exception:
                pass
        
        # Stage 0: Seed pair
        self.pairs = [(seed_qa[0], seed_qa[1], 0)]

        # Extract kb_answer for templates (prefer real answer over placeholder)
        kb_answer = seed_qa[1] if 'learning about' not in seed_qa[1] else ""

        # Stage 1: Template augmentation
        self.pairs.extend(self.generate_templates(kb_answer))

        # Stage 2: Backfill from similar intents
        self.pairs.extend(self.backfill_from_similar_intents())

        # Stage 3: Hippocampus mining
        self.pairs.extend(self.mine_hippocampus())

        # Stage 4: Socratic self-play
        self.pairs.extend(self.generate_by_self_play())

        # CURRICULUM SORT: Sort by difficulty, then by question length
        self.pairs.sort(key=lambda x: (x[2], len(x[0])))

        # Deduplicate and strip difficulty tag
        seen = set()
        deduped = []
        for q, a, _ in self.pairs:
            key = q.lower().strip()
            if key not in seen and key != self.seed[0].lower().strip():
                seen.add(key)
                deduped.append((q, a))
                
        # Always include seed at the very beginning
        final_pairs = [(seed_qa[0], seed_qa[1])] + deduped
        return final_pairs[:self.max_pairs]


def generate_dataset(intent: str, seed_prompt: str, seed_answer: str,
                     skill_manager=None, max_pairs: int = 20,
                     output_path: Optional[str] = None) -> List[Tuple[str, str]]:
    """Convenience function: create and run AutoDatasetGenerator, save result."""
    gen = AutoDatasetGenerator(intent, seed_prompt, seed_answer, skill_manager, max_pairs)
    pairs = gen.generate()
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'  [*] AutoDatasetGenerator: {len(pairs)} pairs generated.')
    return pairs
