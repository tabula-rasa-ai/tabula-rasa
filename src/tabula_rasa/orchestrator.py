"""
SpecialistOrchestrator — decomposes complex questions into sub-questions
and routes each to the best-trained specialist, combining results.

Handles:
- Multi-part questions ("what is X and how does Y work?")
- Comparison questions ("difference between X and Y")
- Compound questions spanning multiple intent types
"""

import re
from typing import Optional, List, Tuple

# Question type patterns
SUB_QUESTION_SPLITTERS = [
    r'\band\b(?=\s+\w+(?:\s|$))',   # "what is X and how does Y work"
    r'\bbut\b',                      # "what is X but what about Y?"
    r'\b(vs|versus|compared to)\b',  # "difference between X and Y"
    r'[.;]\s*',                      # "what is X. How does Y work?"
]

# Known specialist types and their detection patterns
SPECIALIST_DETECTORS = [
    ('definition_question', [r'\bwhat (is|are|was|were)\b', r'\bdefine\b', r'\bexplain\b', r'\bmean[s]?\b']),
    ('explanation_question', [r'\bhow\b', r'\bwhy\b', r'\bwhere\b', r'\bwhen\b', r'\bcauses?\b']),
    ('math', [r'\d+\s*[+\-*/]\s*\d+', r'\b(?:add|sum|plus|minus|times|multiply|divide)\b']),
    ('conversation', [r'\btell me\b', r'\bjoke\b', r'\bstory\b']),
    ('capability_question', [r'\bwhat can you\b', r'\bwhat do you\b', r'\bcapabilities?\b']),
    ('greeting', [r'\b(?:hi|hello|hey|greetings|howdy)\b']),
]


def decompose_question(question: str) -> List[str]:
    """Split a complex question into sub-questions."""
    q = question.strip()
    if not q.endswith('?'):
        q += '?'
    parts = [q]

    for splitter in SUB_QUESTION_SPLITTERS:
        split = []
        for p in parts:
            pieces = re.split(splitter, p, maxsplit=1)
            if len(pieces) == 2 and len(pieces[0].strip()) > 5 and len(pieces[1].strip()) > 5:
                a = pieces[0].strip().rstrip('.,; ')
                b = pieces[1].strip().rstrip('.,; ')
                if not a.endswith('?'): a += '?'
                if not b.endswith('?'): b += '?'
                split.append(a)
                split.append(b)
            else:
                split.append(p)
        parts = split
        if len(parts) >= 3:
            break

    return parts


def detect_specialist_for_subquestion(subq: str) -> str:
    """Route a sub-question to the best specialist type."""
    subq_lower = subq.lower()
    for specialty, patterns in SPECIALIST_DETECTORS:
        for pat in patterns:
            if re.search(pat, subq_lower):
                return specialty
    return 'definition_question'


class SpecialistOrchestrator:
    """Routes complex questions across multiple specialists."""

    def __init__(self, skill_manager):
        self.manager = skill_manager

    def answer(self, question: str) -> dict:
        """Answer a question, potentially routing sub-questions to multiple specialists."""
        sub_questions = decompose_question(question)

        if len(sub_questions) <= 1:
            return None  # Signal to use normal ask flow

        answers = []
        all_known = True
        for subq in sub_questions:
            specialty = detect_specialist_for_subquestion(subq)
            model_name = None
            if specialty == 'math':
                for s in ['addition', 'multiplication', 'division', 'general_math']:
                    if s in self.manager.models:
                        model_name = s
                        break
            elif specialty in self.manager.models:
                model_name = specialty
            else:
                for s in self.manager.known_skills:
                    if s not in ['addition', 'multiplication', 'division', 'general_math']:
                        model_name = s
                        break

            if model_name:
                # Try retrieval first (instant, accurate)
                retrieved = self.manager._retrieve_answer(specialty, subq)
                if retrieved and len(retrieved[0]) > 5 and 'learning about' not in retrieved[0]:
                    answers.append((subq, retrieved[0]))
                    continue

                # Fallback to neural generation
                subq_with_eq = subq if subq.endswith('=') else subq + '='
                model = self.manager.models[model_name]
                tok = self.manager.tokenizers[model_name]
                from egefalos.tabula_rasa import scale_config
                sc = scale_config(model_name, self.manager.skill_levels.get(model_name, 0))
                try:
                    full = model.generate(
                        tok, subq_with_eq,
                        max_new_tokens=sc.get('max_tokens', 50),
                        temperature=0.3, top_k=5,
                    )
                    if '=' in full:
                        ans = full.split('=', 1)[1].strip()
                    else:
                        ans = full.strip()
                    ans = ans.replace('<EOS>', '').replace('<PAD>', '').strip()
                    if len(ans) >= 3:
                        answers.append((subq, ans))
                        continue
                except Exception:
                    pass
            # Last resort (neural generation also failed)
            answers.append((subq, f'I need to learn about {subq} first.'))
            all_known = False

        if len(answers) == 1:
            combined = answers[0][1]
        else:
            parts = []
            for subq, ans in answers:
                parts.append(f"{subq}: {ans}")
            combined = ' | '.join(parts)

        return {
            'answer': combined,
            'knows': all_known,
            'sub_questions': [s for s, _ in answers],
            'status': 'answered',
            'time_ms': 0,
            'training_info': {'mode': 'orchestrated'},
        }
