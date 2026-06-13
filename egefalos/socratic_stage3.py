"""Socratic Engine — Stage 3: Dialectical Self-Play.

Full Socratic dialogues between specialized personas. A judge scores
the debate. Winning logical paths become high-IQ training data that
gets consolidated during the sleep cycle.

This is the "IQ supernova" — self-generated reasoning chains that
train the model to think logically, not just pattern-match.
"""

import sys, json, random, time, math
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters


PERSONAS = {
    'logician': "You are a strict logician. Use formal logic and evidence.",
    'philosopher': "You are a philosopher. Consider ethics, meaning, and first principles.",
    'skeptic': "You are a skeptic. Question every assumption and find counterexamples.",
    'scientist': "You are a scientist. Use empirical evidence and the scientific method.",
    'mathematician': "You are a mathematician. Use formal proofs and precise definitions.",
}


class DialecticalEngine:
    """Multi-persona debate with judge scoring."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def speak(self, persona: str, context: str, max_tokens=50) -> str:
        """Generate a statement from a persona with context."""
        prompt = f'{PERSONAS[persona]}\n\nContext: {context}\n\nResponse:'
        out = self.model.generate(self.tok, prompt,
                                  max_new_tokens=max_tokens,
                                  temperature=0.7, top_k=10)
        return out

    def debate(self, topic: str, persona_a='logician', persona_b='philosopher',
               turns=5) -> dict:
        """Run a full Socratic dialogue between two personas."""
        transcript = []
        context_a = f'Topic: {topic}\nWe are debating this topic. State your position.'
        context_b = f'Topic: {topic}\nYour opponent just spoke. Respond with a counterargument.'

        for turn in range(turns):
            # Persona A speaks
            a_response = self.speak(persona_a, context_a)
            transcript.append({
                'turn': turn + 1,
                'persona': persona_a,
                'text': a_response,
            })

            # Persona B responds
            b_response = self.speak(persona_b, f'{context_b}\n\nOpponent said: {a_response}')
            transcript.append({
                'turn': turn + 1,
                'persona': persona_b,
                'text': b_response,
            })

            # Update contexts
            context_a = f'{context_a}\n\n{persona_b}: {b_response}\n\nYour response:'
            context_b = f'{context_b}\n\n{persona_a}: {a_response}\n\nYour response:'

        return {
            'topic': topic,
            'persona_a': persona_a,
            'persona_b': persona_b,
            'transcript': transcript,
            'turns': turns,
        }


class ValueJudge:
    """Scores debate arguments for logical soundness."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def score_argument(self, statement: str, context: str = '') -> dict:
        """Score an argument's logical quality (0-100)."""
        prompt = f'Rate this argument\'s logical quality from 0-100.\n\nArgument: {statement}\n\nScore:'
        score_text = self.model.generate(self.tok, prompt, max_new_tokens=5,
                                         temperature=0.1, top_k=3)

        # Extract numeric score
        score = 50  # default
        for word in score_text.split():
            try:
                s = float(word.strip(',.'))
                if 0 <= s <= 100:
                    score = s
                    break
            except:
                continue

        return {
            'statement': statement,
            'score': score,
        }

    def judge_debate(self, debate: dict) -> dict:
        """Score a complete debate and determine the winner."""
        results = []
        for entry in debate['transcript']:
            result = self.score_argument(entry['text'], debate['topic'])
            result['persona'] = entry['persona']
            results.append(result)

        # Average scores per persona
        scores = {}
        for r in results:
            p = r['persona']
            if p not in scores:
                scores[p] = []
            scores[p].append(r['score'])

        avg_scores = {p: sum(s) / len(s) for p, s in scores.items()}
        winner = max(avg_scores, key=avg_scores.get) if avg_scores else None

        return {
            'debate': debate,
            'scores': avg_scores,
            'winner': winner,
            'winning_arguments': [
                r for r in results
                if r['persona'] == winner and r['score'] >= 70
            ],
        }


def consolidate_winners(model, tokenizer, judge_results: list[dict], epochs=3):
    """Train the model on winning logical arguments only."""
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    model.train()

    # Extract winning arguments
    training_pairs = []
    for result in judge_results:
        for arg in result.get('winning_arguments', []):
            text = f"Logical argument: {arg['statement']}"
            training_pairs.append(text)

    if not training_pairs:
        print('  No winning arguments to consolidate')
        return 0

    print(f'  Consolidating {len(training_pairs)} winning arguments...')

    total_pairs = 0
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for text in training_pairs:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > 64:
                ids = ids[:64]
            padded = ids + [tokenizer.pad_id] * (64 - len(ids))
            x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
            total_pairs += 1

        if n > 0:
            print(f'  Epoch {epoch+1}: loss={total_loss/n:.4f}')

    return total_pairs


def full_socratic_session(model, tokenizer, topic: str, turns=3):
    """Run a complete Socratic session: debate → judge → consolidate."""
    print(f'\n{"=" * 60}')
    print(f'  Socratic Session')
    print(f'  Topic: {topic}')
    print(f'  Personas: Logician vs Philosopher ({turns} turns)')
    print(f'{"=" * 60}')

    # Phase 1: Debate
    engine = DialecticalEngine(model, tokenizer)
    debate = engine.debate(topic, 'logician', 'philosopher', turns=turns)
    print(f'\n  Debate complete: {len(debate["transcript"])} exchanges')

    # Phase 2: Judge
    judge = ValueJudge(model, tokenizer)
    result = judge.judge_debate(debate)
    print(f'\n  Scores:')
    for p, s in result['scores'].items():
        print(f'    {p}: {s:.1f}')
    print(f'  Winner: {result["winner"]}')
    print(f'  Winning arguments: {len(result["winning_arguments"])}')

    # Phase 3: Consolidate
    total = consolidate_winners(model, tokenizer, [result], epochs=2)
    print(f'\n  Consolidated {total} training pairs into memory')

    return result


if __name__ == '__main__':
    print('=' * 60)
    print('  Socratic Engine — Stage 3: Dialectical Self-Play')
    print('=' * 60)

    model, tok = None, None
    for path in ['specialists/math/general', 'specialists/math/add', 'checkpoints']:
        d = Path(path)
        ckpt = d / 'best.pt'
        if ckpt.exists():
            tok = MathTokenizer.load(str(d / 'tokenizer.json'))
            cfg = Config()
            cfg.vocab_size = tok.vocab_size
            tok.max_seq_len = cfg.max_seq_len
            model = MathTransformer(cfg)
            state = torch.load(ckpt, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            print(f'  Loaded model from {path}')
            break

    if model is None:
        print('[!] No model found')
        sys.exit(1)

    # Run a Socratic session
    full_socratic_session(model, tok,
                          topic='The ethics of artificial intelligence',
                          turns=2)
