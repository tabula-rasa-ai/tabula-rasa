"""Self-improvement loop: generate → solve → verify → train → repeat.

This implements ReST (Reinforced Self-Training): the model generates
solutions to math problems, a verifier checks correctness, and the
model trains on its own correct outputs. Each cycle lifts "IQ."

Includes Hard Negative Mining — entropy-based blind-spot detection
that biases the curriculum toward the model's hardest examples.
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import math
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.dataset import generate_problem, format_math_sample
from tabula_rasa.model import MathTransformer


# ─── Hard Negative Miner ─────────────────────────────────────────────

class HardNegativeMiner:
    """Identifies model blind spots via softmax entropy scoring.

    Generates seed problems, runs a forward pass, measures per-token
    entropy (uncertainty), then biases new generation toward the
    model's failure modes. 60% hard-seed mutations + 40% fresh random
    for exploration stability.
    """

    def __init__(self, model: MathTransformer, tokenizer: MathTokenizer,
                 config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = torch.device(config.device)

    @torch.no_grad()
    def score_samples(self, samples: list[list[int]]) -> list[float]:
        """Score each sample by mean softmax entropy over answer tokens.

        Higher score = model is more uncertain = harder example.
        """
        self.model.eval()
        if not samples:
            return []

        max_len = max(len(s) for s in samples)
        padded = [s + [self.tokenizer.pad_id] * (max_len - len(s))
                  for s in samples]
        batch = torch.tensor(padded, dtype=torch.long).to(self.device)

        logits, _, _ = self.model(batch)  # (B, T, V)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

        scores = []
        for i, sample in enumerate(samples):
            valid = min(len(sample) - 1, entropy.size(1))
            scores.append(entropy[i, :valid].mean().item())
        return scores

    def generate_curriculum(self, num_samples: int,
                            min_digits: int = 1,
                            max_digits: int = 4) -> list[tuple[str, str]]:
        """Produce num_samples × (expr, answer) with hard-negative bias.

        Phase 1 — Seed scoring: generate ~500 candidates, score by
        entropy, keep the hardest 20% as mutation seeds.
        Phase 2 — Curriculum: 60% hard-seed mutations, 40% fresh random.
        """
        # Phase 1: seed scoring
        num_seeds = min(500, num_samples)
        seeds: list[tuple[str, str, list[int]]] = []
        attempts = 0
        while len(seeds) < num_seeds and attempts < num_seeds * 5:
            expr, ans = generate_problem(min_digits, max_digits)
            text = format_math_sample(expr, ans)
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= self.cfg.max_seq_len:
                seeds.append((expr, ans, ids))
            attempts += 1

        scored = list(zip(seeds, self.score_samples([s[2] for s in seeds])))
        scored.sort(key=lambda x: -x[1])  # hardest first
        hard_seeds = [(s[0][0], s[0][1]) for s in scored[:max(1, len(scored) // 5)]]

        # Phase 2: build curriculum
        curriculum: list[tuple[str, str]] = []

        # 60% hard-seed mutations
        mutation_count = int(num_samples * 0.6)
        for _ in range(mutation_count):
            if not hard_seeds:
                break
            seed = random.choice(hard_seeds)
            mutated = self._mutate(seed)
            if mutated:
                curriculum.append(mutated)

        # 40% fresh random for exploration stability
        fill = num_samples - len(curriculum)
        for _ in range(fill):
            expr, ans = generate_problem(min_digits, max_digits)
            curriculum.append((expr, ans))

        random.shuffle(curriculum)
        return curriculum

    @staticmethod
    def _mutate(seed: tuple[str, str]) -> tuple[str, str] | None:
        """Return a single mutated problem similar to the seed."""
        expr, _ = seed
        for op in ['+', '-', '*', '/']:
            if op not in expr:
                continue
            parts = expr.split(op)
            if len(parts) != 2:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
                delta_a = random.randint(-max(1, a // 3), max(1, a // 3))
                delta_b = random.randint(-max(1, b // 3), max(1, b // 3))
                na, nb = max(1, a + delta_a), max(1, b + delta_b)

                if op == '+':
                    return (f'{na}+{nb}', str(na + nb))
                elif op == '-':
                    na, nb = (max(na, nb), min(na, nb))
                    return (f'{na}-{nb}', str(na - nb))
                elif op == '*':
                    return (f'{na}*{nb}', str(na * nb))
                elif op == '/':
                    # Keep division exact
                    nb = max(1, nb)
                    ans = na // nb
                    if ans == 0:
                        ans = 1
                    na = nb * ans
                    return (f'{na}/{nb}', str(ans))
            except (ValueError, ZeroDivisionError):
                pass
            break
        return None


# ─── Verifier ───────────────────────────────────────────────────────

class MathVerifier:
    """Verifies math expressions by evaluating them in Python."""

    @staticmethod
    def verify(expr: str, predicted_answer: str) -> bool:
        """Check if predicted answer matches actual computation."""
        try:
            actual = eval(expr)
            expected = str(actual)
            return predicted_answer.strip() == expected.strip()
        except:
            return False

    @staticmethod
    def compute_answer(expr: str) -> str | None:
        """Compute the actual answer to an expression."""
        try:
            return str(eval(expr))
        except:
            return None


# ─── Difficulty Scheduler ───────────────────────────────────────────

class DifficultyScheduler:
    """Gradually increases problem difficulty based on accuracy."""

    def __init__(self, initial_max_digits=2):
        self.max_digits = initial_max_digits
        self.acc_history = []

    def update(self, accuracy: float):
        self.acc_history.append(accuracy)
        # If accuracy > 70% over last 3 cycles, increase difficulty
        if len(self.acc_history) >= 3:
            recent = self.acc_history[-3:]
            if all(a > 70.0 for a in recent):
                self.max_digits = min(self.max_digits + 1, 6)
                print(f'  [*] Difficulty increased to {self.max_digits}-digit problems!')

    def generate_problem(self):
        return generate_problem(1, self.max_digits)


# ─── Self-Training Dataset ──────────────────────────────────────────

class SelfTrainDataset(Dataset):
    """Dataset built from self-generated correct solutions."""

    def __init__(self, tokenizer: MathTokenizer, samples: list[list[int]],
                 max_seq_len: int = 64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        padded = ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(ids))
        x = padded[:-1]
        y = padded[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ─── Self-Improvement Loop ──────────────────────────────────────────

class SelfImprovementLoop:
    """Self-training loop: generate → solve → verify → fine-tune."""

    def __init__(self, model: MathTransformer, tokenizer: MathTokenizer,
                 config: Config, checkpoint_dir: str = 'self_train'):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = torch.device(config.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.verifier = MathVerifier()
        self.difficulty = DifficultyScheduler()
        self.miner = HardNegativeMiner(model, tokenizer, config)

    @torch.no_grad()
    def generate_solutions(self, num_problems: int = 5000,
                           use_hard_negative: bool = True) -> list[tuple]:
        """Generate solutions using hard-negative-biased curriculum.

        When use_hard_negative=True, problems are biased toward the
        model's blind spots. When False (or on first cycle before the
        model has any training), uses pure random generation.
        """
        correct_samples = []
        total_attempts = 0
        stats = {'correct': 0, 'wrong': 0, 'parse_errors': 0}

        print(f'  Generating {num_problems} solutions...')

        # Build curriculum (hard-negative biased or pure random)
        if use_hard_negative:
            print(f'  [*] Using hard-negative curriculum (entropy-guided)')
            curriculum = self.miner.generate_curriculum(
                num_problems,
                min_digits=1,
                max_digits=self.difficulty.max_digits,
            )
        else:
            curriculum = None

        while len(correct_samples) < num_problems:
            if curriculum:
                idx = total_attempts % len(curriculum)
                expr, expected = curriculum[idx]
            else:
                expr, expected = self.difficulty.generate_problem()
            prompt = f'{expr}='

            # Model generates solution
            generated = self.model.generate(
                self.tokenizer, prompt,
                max_new_tokens=10,
                temperature=0.5,  # Some randomness for diversity
                top_k=5,
            )
            total_attempts += 1

            # Extract predicted answer
            if '=' in generated:
                pred = generated.split('=')[-1].strip()
                pred_clean = ''.join(c for c in pred if c.isdigit() or c == '-')
            else:
                pred_clean = ''
                stats['parse_errors'] += 1
                continue

            # Verify
            if pred_clean == expected:
                text = f'{expr}={expected}'
                ids = self.tokenizer.encode(text, add_special_tokens=True)
                if len(ids) <= self.cfg.max_seq_len:
                    correct_samples.append(ids)
                stats['correct'] += 1
            else:
                stats['wrong'] += 1

            if total_attempts % 500 == 0:
                ratio = stats['correct'] / max(1, total_attempts) * 100
                print(f'    Progress: {len(correct_samples)}/{num_problems} correct '
                      f'({ratio:.0f}% hit rate)')

        accuracy = stats['correct'] / max(1, total_attempts) * 100
        print(f'  Generated {len(correct_samples)} correct solutions '
              f'(hit rate: {accuracy:.1f}%)')
        return correct_samples, accuracy

    def train_on_solutions(self, samples: list[list[int]], num_epochs: int = 3):
        """Fine-tune model on self-generated correct solutions."""
        dataset = SelfTrainDataset(self.tokenizer, samples, self.cfg.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size,
                            shuffle=True, num_workers=0)

        optimizer = AdamW(self.model.parameters(),
                          lr=self.cfg.learning_rate / 2,  # Lower LR for fine-tuning
                          weight_decay=self.cfg.weight_decay)

        self.model.train()
        total_steps = len(loader) * num_epochs
        step = 0
        total_loss = 0.0

        print(f'  Fine-tuning on {len(samples)} samples for {num_epochs} epochs '
              f'({total_steps} steps)...')

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                _, loss, _ = self.model(x, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                step += 1

            avg_epoch_loss = epoch_loss / max(1, n_batches)
            total_loss += epoch_loss
            print(f'    Epoch {epoch + 1}: loss={avg_epoch_loss:.4f}')

        avg_loss = total_loss / max(1, step)
        print(f'  Fine-tuning complete (avg loss: {avg_loss:.4f})')

    @torch.no_grad()
    def evaluate(self, num_problems: int = 200) -> float:
        """Evaluate current accuracy."""
        from tabula_rasa.eval import evaluate_accuracy
        return evaluate_accuracy(
            self.model, self.tokenizer,
            num_problems=num_problems,
            max_digits=self.difficulty.max_digits,
            verbose=False,
        )

    def run_cycle(self, cycle: int, num_new_problems: int = 5000,
                  train_epochs: int = 3,
                  use_hard_negative: bool = True):
        """Run one self-improvement cycle."""
        print(f'\n{"="*60}')
        print(f'  SELF-IMPROVEMENT CYCLE {cycle}')
        print(f'  Difficulty: {self.difficulty.max_digits}-digit problems'
              f'  |  Hard-negative: {"ON" if use_hard_negative else "OFF"}')
        print(f'{"="*60}')

        # Step 1: Generate and verify (with hard-negative bias)
        samples, accuracy = self.generate_solutions(
            num_new_problems, use_hard_negative=use_hard_negative)

        # Step 2: Fine-tune on correct solutions
        self.train_on_solutions(samples, train_epochs)

        # Step 3: Evaluate
        eval_acc = self.evaluate()
        print(f'  Post-cycle accuracy: {eval_acc:.1f}%')

        # Step 4: Update difficulty
        self.difficulty.update(eval_acc)

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'cycle_{cycle}.pt'
        torch.save({
            'cycle': cycle,
            'model_state_dict': self.model.state_dict(),
            'accuracy': eval_acc,
            'difficulty': self.difficulty.max_digits,
        }, checkpoint_path)
        print(f'  Checkpoint saved to {checkpoint_path}')

        return eval_acc

    def run(self, num_cycles: int = 5, problems_per_cycle: int = 5000,
            train_epochs: int = 3, use_hard_negative: bool = True):
        """Run multiple self-improvement cycles."""
        print(f'\nStarting self-improvement: {num_cycles} cycles')
        print(f'{problems_per_cycle} problems per cycle')
        print(f'{train_epochs} training epochs per cycle'
              f'  |  Hard-negative: {"ON" if use_hard_negative else "OFF"}')
        print()

        accuracies = []
        for cycle in range(1, num_cycles + 1):
            t0 = time.time()
            acc = self.run_cycle(cycle, problems_per_cycle, train_epochs,
                                 use_hard_negative=(cycle > 1 and use_hard_negative))
            elapsed = time.time() - t0
            accuracies.append(acc)
            print(f'  Cycle {cycle}: {elapsed:.0f}s, accuracy={acc:.1f}%')

        print(f'\n{"="*60}')
        print(f'  SELF-IMPROVEMENT COMPLETE')
        print(f'  Starting accuracy: {accuracies[0]:.1f}%' if accuracies else '')
        print(f'  Final accuracy: {accuracies[-1]:.1f}%' if accuracies else '')
        print(f'  Best accuracy: {max(accuracies):.1f}%' if accuracies else '')
        for i, acc in enumerate(accuracies):
            print(f'    Cycle {i+1}: {acc:.1f}%')
        print(f'{"="*60}')

        return accuracies


def load_model_for_self_train(checkpoint_path='checkpoints/best.pt'):
    """Load trained model for self-improvement."""
    if not Path(checkpoint_path).exists():
        checkpoint_path = 'checkpoints/final.pt'
    if not Path(checkpoint_path).exists():
        print(f'No checkpoint found at {checkpoint_path}')
        return None, None

    cfg = Config()
    tok = MathTokenizer.load(str(Path(checkpoint_path).parent / 'tokenizer.json'))
    cfg.vocab_size = tok.vocab_size  # type: ignore[misc]
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tok


if __name__ == '__main__':
    import sys

    print('Loading base model...')
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best.pt'
    model, tok = load_model_for_self_train(ckpt)
    if model is None:
        sys.exit(1)

    print(f'Model loaded. Starting self-improvement...')
    print(f'Device: {model.device}')

    # Move model to device
    device = torch.device(model.device)
    model = model.to(device)

    cfg = Config()
    cfg.device = model.device
    cfg.vocab_size = tok.vocab_size

    loop = SelfImprovementLoop(model, tok, cfg)

    num_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    problems_per_cycle = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    loop.run(num_cycles=num_cycles, problems_per_cycle=problems_per_cycle)
