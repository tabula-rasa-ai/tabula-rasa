"""Socratic Engine — Stage 3: Dialectical Self-Play with Arithmetic Reasoning.

Full Socratic dialogues between specialized personas. A judge scores
the debate. Winning logical paths become high-IQ training data that
gets consolidated during the sleep cycle.

Supports both philosophical debate (original) and arithmetic reasoning
(self-critique of mathematical scratchpads).
"""

import sys, json, random, time, math
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.math_parser import verify_scratchpad, verify_equation, evaluate as math_eval


PERSONAS = {
    'logician': "You are a strict logician. Use formal logic and evidence.",
    'philosopher': "You are a philosopher. Consider ethics, meaning, and first principles.",
    'skeptic': "You are a skeptic. Question every assumption and find counterexamples.",
    'scientist': "You are a scientist. Use empirical evidence and the scientific method.",
    'mathematician': "You are a mathematician. Use formal proofs and precise definitions.",
}

# Arithmetic-specific personas for debate over scratchpad correctness
ARITH_PERSONAS = {
    'generator': "You are solving a math problem. Show your work step by step.",
    'critic': "You are a strict math teacher. Check every step for errors. "
              "Identify the exact column where the carry or digit is wrong.",
    'verifier': "You are a verifier. Compare the proposed solution against "
                "the correct answer and list each discrepancy.",
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
            a_response = self.speak(persona_a, context_a)
            transcript.append({
                'turn': turn + 1,
                'persona': persona_a,
                'text': a_response,
            })
            b_response = self.speak(persona_b, f'{context_b}\n\nOpponent said: {a_response}')
            transcript.append({
                'turn': turn + 1,
                'persona': persona_b,
                'text': b_response,
            })
            context_a = f'{context_a}\n\n{persona_b}: {b_response}\n\nYour response:'
            context_b = f'{context_b}\n\n{persona_a}: {a_response}\n\nYour response:'

        return {
            'topic': topic,
            'persona_a': persona_a,
            'persona_b': persona_b,
            'transcript': transcript,
            'turns': turns,
        }

    def debate_arithmetic(self, expression: str, model_trace: str,
                          correct_trace: str, turns=3) -> dict:
        """Debate the correctness of an arithmetic scratchpad solution.

        Personas debate whether a particular column calculation is correct
        by examining carries, digits, and column sums.

        Args:
            expression: The math expression (e.g. "12+34").
            model_trace: The model's scratchpad (e.g. "0406").
            correct_trace: The correct scratchpad (e.g. "0406" or different).
            turns: Number of debate rounds.

        Returns:
            Debate result with transcript and learned correction.
        """
        transcript = []
        context = (
            f"Problem: {expression}=\n"
            f"Proposed solution trace: {model_trace}\n"
            f"Expected trace: {correct_trace}\n\n"
            f"Column-by-column analysis:\n"
        )

        # Build column analysis
        columns = []
        max_len = max(len(model_trace), len(correct_trace)) // 2
        for col in range(max_len):
            m_carry = model_trace[col*2] if col*2 < len(model_trace) else '?'
            m_digit = model_trace[col*2+1] if col*2+1 < len(model_trace) else '?'
            c_carry = correct_trace[col*2] if col*2 < len(correct_trace) else '?'
            c_digit = correct_trace[col*2+1] if col*2+1 < len(correct_trace) else '?'
            match = "CORRECT" if (m_carry == c_carry and m_digit == c_digit) else "MISMATCH"
            columns.append({
                'col': col, 'model_carry': m_carry, 'model_digit': m_digit,
                'correct_carry': c_carry, 'correct_digit': c_digit, 'match': match,
            })

        context += '\n'.join(
            f"  Column {c['col']}: model=({c['model_carry']},{c['model_digit']}) "
            f"expected=({c['correct_carry']},{c['correct_digit']}) [{c['match']}]"
            for c in columns
        )

        for turn in range(turns):
            # Generator proposes
            gen_prompt = (
                f"You are the generator. You produced trace '{model_trace}' "
                f"for {expression}=. Defend your answer column by column.\n"
                f"Context:\n{context}\n\nResponse:"
            )
            gen_response = self.model.generate(
                self.tok, gen_prompt,
                max_new_tokens=40, temperature=0.7
            )
            transcript.append({'turn': turn + 1, 'role': 'generator', 'text': gen_response})

            # Critic challenges
            crit_prompt = (
                f"You are the critic. The generator claims '{model_trace}' is correct "
                f"for {expression}= but the expected trace is '{correct_trace}'.\n"
                f"Point out the exact column where the logic fails.\n"
                f"Response:"
            )
            crit_response = self.model.generate(
                self.tok, crit_prompt,
                max_new_tokens=40, temperature=0.5
            )
            transcript.append({'turn': turn + 1, 'role': 'critic', 'text': crit_response})

            # Update context with latest exchange
            context += f"\n\nGenerator: {gen_response}\nCritic: {crit_response}"

        return {
            'expression': expression,
            'model_trace': model_trace,
            'correct_trace': correct_trace,
            'columns': columns,
            'transcript': transcript,
            'turns': turns,
            'correct': model_trace == correct_trace,
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
        score = 50
        for word in score_text.split():
            try:
                s = float(word.strip(',.'))
                if 0 <= s <= 100:
                    score = s
                    break
            except:
                continue
        return {'statement': statement, 'score': score}

    def judge_debate(self, debate: dict) -> dict:
        """Score a complete debate and determine the winner."""
        results = []
        for entry in debate['transcript']:
            result = self.score_argument(entry['text'])
            result['role'] = entry.get('role', entry.get('persona', 'unknown'))
            results.append(result)

        scores = {}
        for r in results:
            p = r.get('role', 'unknown')
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
                if r.get('role') == winner and r['score'] >= 70
            ],
        }

    def judge_arithmetic(self, debate: dict) -> dict:
        """Judge an arithmetic debate using the verifier (deterministic).

        Unlike philosophical debate which uses model-based scoring,
        arithmetic debates can be scored deterministically by comparing
        the model's trace against the correct trace column by column.

        Returns:
            Dict with deterministic score (proportion of correct columns).
        """
        columns = debate.get('columns', [])
        if not columns:
            return {'score': 0.0, 'correct_columns': 0, 'total_columns': 1}

        total = len(columns)
        correct = sum(1 for c in columns if c['match'] == 'CORRECT')
        score = correct / total * 100

        return {
            'expression': debate.get('expression', ''),
            'score': score,
            'correct_columns': correct,
            'total_columns': total,
            'all_correct': correct == total,
        }


def consolidate_winners(model, tokenizer, judge_results: list[dict], epochs=3):
    """Train the model on winning logical arguments only."""
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    model.train()

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


def run_arithmetic_session(
    model, tokenizer, cfg,
    num_problems: int = 20,
    turns_per_problem: int = 3,
    op: str = 'add',
    max_digits: int = 1,
) -> dict:
    """Run arithmetic debate and training for multiple problems.

    For each problem:
    1. Generate a problem (prompt + correct answer)
    2. Generate model's scratchpad
    3. Generate correct scratchpad
    4. Debate the correctness
    5. Judge deterministically via column comparison
    6. Train on corrections

    Args:
        model: The model.
        tokenizer: MathTokenizer.
        cfg: Config with operation settings.
        num_problems: Number of problems to debate.
        turns_per_problem: Debate rounds per problem.
        op: Operation ('add', 'sub', 'mul').
        max_digits: Max digits per operand.

    Returns:
        Dict with results.
    """
    from train_specialist import generate_problem, _correct_scratchpad

    engine = DialecticalEngine(model, tokenizer)
    judge = ValueJudge(model, tokenizer)

    total_correct = 0
    corrections = []
    debates = []

    print(f'\n  Arithmetic Socratic Session: {num_problems} problems')
    print(f'  Operation: {op} | Max digits: {max_digits}')

    for prob_idx in range(num_problems):
        # Generate problem
        expr, answer = generate_problem(op, max_digits, max_digits,
                                         reversed=True, scratchpad=True)
        prompt = f'{expr}='
        expected_int = int(answer) if answer.lstrip('-').isdigit() else 0

        # Generate model's trace
        model_output = model.generate(
            tokenizer, prompt,
            max_new_tokens=max_digits * 4 + 4,
            temperature=0.3,
        )

        # Parse the model's scratchpad from the output
        gen_trace = model_output.replace(prompt, '').strip()
        # Remove special tokens
        gen_trace = gen_trace.replace('<EOS>', '').replace('<PAD>', '').strip()

        # Generate correct trace using math_parser
        # Parse expression to get a, b, op
        expr_clean = expr.replace(' ', '')
        if '+' in expr_clean:
            parts = expr_clean.split('+')
            a, b = int(parts[0]), int(parts[1])
            correct_trace = _generate_correct_trace(a, b, '+')
        elif '-' in expr_clean:
            parts = expr_clean.split('-')
            a, b = int(parts[0]), int(parts[1])
            correct_trace = _generate_correct_trace(a, b, '-')
        else:
            correct_trace = answer

        # Debate the correctness
        debate = engine.debate_arithmetic(
            expr_clean, gen_trace, correct_trace,
            turns=turns_per_problem,
        )
        debates.append(debate)

        # Judge deterministically
        verdict = judge.judge_arithmetic(debate)
        is_correct = gen_trace == correct_trace
        if is_correct:
            total_correct += 1

        if not is_correct and verdict.get('score', 0) > 0:
            corrections.append({
                'expression': expr_clean,
                'trace': gen_trace,
                'correct_trace': correct_trace,
                'correct_columns': verdict.get('correct_columns', 0),
                'total_columns': verdict.get('total_columns', 1),
            })

        if (prob_idx + 1) % 5 == 0:
            print(f'    Problem {prob_idx+1}/{num_problems}: '
                  f'{total_correct}/{prob_idx+1} correct')

    # Train on corrections
    if corrections:
        print(f'  Training on {len(corrections)} corrections...')
        optimizer = AdamW(model.parameters(), lr=1e-5)
        for corr in corrections:
            prompt = f'{corr["expression"]}='
            text = f'{prompt}{corr["correct_trace"]}'
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

    return {
        'num_problems': num_problems,
        'correct': total_correct,
        'accuracy': total_correct / max(1, num_problems) * 100,
        'corrections': len(corrections),
        'debates': debates,
    }


def _generate_correct_trace(a: int, b: int, op: str) -> str:
    """Generate correct fused carry-digit scratchpad."""
    carry = 0
    sp = ''
    ra, rb = str(a)[::-1], str(b)[::-1]
    max_len = max(len(ra), len(rb))
    for i in range(max_len):
        da = int(ra[i]) if i < len(ra) else 0
        db = int(rb[i]) if i < len(rb) else 0
        if op == '+':
            total = da + db + carry
            carry = total // 10
            digit = total % 10
        else:
            if da < db + carry:
                da += 10
            total = da - db - carry
            carry = 0
            digit = total
        sp += f'{carry}{digit}'
    if carry and op == '+':
        sp += f'0{carry}'
    return sp


def full_socratic_session(model, tokenizer, topic: str, turns=3):
    """Run a complete Socratic session: debate -> judge -> consolidate."""
    print(f'\n{"=" * 60}')
    print(f'  Socratic Session')
    print(f'  Topic: {topic}')
    print(f'  Personas: Logician vs Philosopher ({turns} turns)')
    print(f'{"=" * 60}')

    engine = DialecticalEngine(model, tokenizer)
    debate = engine.debate(topic, 'logician', 'philosopher', turns=turns)
    print(f'\n  Debate complete: {len(debate["transcript"])} exchanges')

    judge = ValueJudge(model, tokenizer)
    result = judge.judge_debate(debate)
    print(f'\n  Scores:')
    for p, s in result['scores'].items():
        print(f'    {p}: {s:.1f}')
    print(f'  Winner: {result["winner"]}')
    print(f'  Winning arguments: {len(result["winning_arguments"])}')

    total = consolidate_winners(model, tokenizer, [result], epochs=2)
    print(f'\n  Consolidated {total} training pairs into memory')
    return result


if __name__ == '__main__':
    print('=' * 60)
    print('  Socratic Engine — Stage 3: Dialectical Self-Play')
    print('=' * 60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['philosophy', 'arithmetic'], default='philosophy')
    parser.add_argument('--problems', type=int, default=10)
    parser.add_argument('--turns', type=int, default=3)
    parser.add_argument('--op', default='add')
    args = parser.parse_args()

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

    if args.mode == 'arithmetic':
        run_arithmetic_session(model, tok, cfg,
                               num_problems=args.problems,
                               turns_per_problem=args.turns,
                               op=args.op)
    else:
        full_socratic_session(model, tok,
                              topic='The ethics of artificial intelligence',
                              turns=args.turns)
