"""Socratic Engine — Stage 2: Generator vs Critic.

Two specialists debate: one proposes, the other attacks. The generator
learns to anticipate counter-arguments and self-correct.

This mirrors Plato's dialectic method: thesis → antithesis → synthesis.
"""

import sys
from pathlib import Path

import torch
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer

CRITIC_PROMPT = "Find the logical flaw, missing premise, or contradiction in the following statement:"


def load_model(checkpoint_dir: str) -> tuple:
    """Load a model and tokenizer from a specialist directory."""
    d = Path(checkpoint_dir)
    ckpt = d / 'best.pt'
    if not ckpt.exists():
        ckpt = d / 'final.pt'
    if not ckpt.exists():
        return None, None
    tok = MathTokenizer.load(str(d / 'tokenizer.json'))
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tok


class GeneratorCritic:
    """Generator (proposes) vs Critic (attacks) debate loop."""

    def __init__(self, generator, critic, tokenizer):
        self.gen = generator
        self.crit = critic
        self.tok = tokenizer
        self.device = next(generator.parameters()).device

    @torch.no_grad()
    def generate(self, model, prompt: str, max_tokens=30, temperature=0.5):
        """Generate text from a model."""
        out = model.generate(self.tok, prompt,
                             max_new_tokens=max_tokens,
                             temperature=temperature, top_k=5)
        return out

    def debate(self, premise: str, turns=3) -> list[dict]:
        """Run a debate between generator and critic.

        Returns the debate history.
        """
        history = []
        current_position = premise

        print(f'\n  Debate on: "{premise[:60]}..."')
        print(f'  {"=" * 50}')

        for turn in range(turns):
            # Generator proposes/revises
            gen_prompt = f"Based on the premise: {premise}\nPropose a conclusion:"
            gen_response = self.generate(self.gen, gen_prompt, max_tokens=30)
            history.append({
                'turn': turn + 1,
                'role': 'generator',
                'text': gen_response,
            })
            print(f'\n  [Generator] {gen_response[:80]}...')

            # Critic attacks
            crit_prompt = f"{CRITIC_PROMPT}\n\nStatement: {gen_response}\n\nCritique:"
            crit_response = self.generate(self.crit, crit_prompt, max_tokens=30)
            history.append({
                'turn': turn + 1,
                'role': 'critic',
                'text': crit_response,
            })
            print(f'  [Critic]    {crit_response[:80]}...')

            # Generator revises based on critique
            revision_prompt = f"{CRITIC_PROMPT}\n\nCritique: {crit_response}\n\nRevised statement:"
            revision = self.generate(self.gen, revision_prompt, max_tokens=30)
            history.append({
                'turn': turn + 1,
                'role': 'revision',
                'text': revision,
            })
            print(f'  [Revised]   {revision[:80]}...')

            current_position = revision

        return history

    @torch.no_grad()
    def self_correct(self, statement: str) -> dict:
        """Single self-correction cycle. Used for training data generation."""
        # Critic finds flaws
        crit_prompt = f"{CRITIC_PROMPT}\n\nStatement: {statement}\n\nCritique:"
        critique = self.generate(self.crit, crit_prompt, max_tokens=20)

        # Generator revises
        rev_prompt = f"{CRITIC_PROMPT}\n\nCritique: {critique}\n\nRevised statement:"
        revision = self.generate(self.gen, rev_prompt, max_tokens=20)

        return {
            'original': statement,
            'critique': critique,
            'revised': revision,
        }

    def self_correct_and_log(self, statement: str) -> dict:
        """Self-correct and log the result to verified monologue."""
        result = self.self_correct(statement)
        if result['revised'] and len(result['revised']) > 5:
            try:
                from bpe_tokenizer import log_verified_output
                log_verified_output(result['revised'])
            except ImportError:
                pass
        return result


def generate_training_data(engine: GeneratorCritic, num_examples=10) -> list[dict]:
    """Generate synthetic high-IQ training data through debate."""
    premises = [
        "All swans are white. This bird is a swan.",
        "If it rains, the ground gets wet. The ground is wet.",
        "Every human is mortal. Socrates is a human.",
        "Adding 0 to any number gives that number. 5 + 0 = ?",
        "Multiplying by 1 gives the same number. 7 x 1 = ?",
    ]

    data = []
    for premise in premises[:num_examples]:
        try:
            result = engine.self_correct(premise)
            data.append(result)
        except Exception as e:
            print(f'  Error on premise "{premise}": {e}')

    return data


def train_on_debates(model, tokenizer, debate_data: list[dict], epochs=3):
    """Fine-tune model on debate outcomes (generator learns from critic)."""
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    model.train()

    total_pairs = 0
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for d in debate_data:
            # Train on: original → revised (the improvement path)
            for pair in [(d['original'], d['revised'])]:
                text = f'Statement: {pair[0]} Revised: {pair[1]}'
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
            print(f'  Epoch {epoch+1}: loss={total_loss/n:.4f} ({total_pairs} pairs)')

    return total_pairs


if __name__ == '__main__':
    print('=' * 60)
    print('  Socratic Engine — Stage 2: Generator vs Critic')
    print('=' * 60)

    # Use generalist as both generator and critic (same architecture, different prompts)
    gen, tok = load_model('specialists/math/general')
    if gen is None:
        print('[!] No model found. Train a generalist first.')
        sys.exit(1)

    crit, _ = load_model('specialists/math/general')  # Same model, different prompt

    engine = GeneratorCritic(gen, crit, tok)

    # Run debate
    history = engine.debate("If you have 5 apples and eat 2, how many remain?", turns=2)
    print(f'\n  Debate complete: {len(history)} exchanges')

    # Generate training data
    print('\n  Generating synthetic training data...')
    data = generate_training_data(engine, num_examples=3)
    print(f'  Generated {len(data)} debate examples')

    # Train generator on improvements
    print('\n  Training generator on debate outcomes...')
    total = train_on_debates(gen, tok, data, epochs=2)
    print(f'\n  Training complete: {total} pairs learned')
