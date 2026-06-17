"""Evolution Strategies trainer — black-box optimization for MathTransformer.

Reference [53] from the PDF: "Evolution Strategies at the Hyperscale" (arXiv:2511.16652).
ES replaces gradient descent with population-based evolution:
  No backprop, no autograd, no GPU needed — just forward passes + parameter perturbations.

How it works:
  1. Take current params θ
  2. Sample noise ε₁, ε₂, ... εₙ ~ N(0, I)  (population of N individuals)
  3. For each εᵢ: create perturbed params θ + σ·εᵢ
  4. Evaluate each perturbed model on training data → fitness score Fᵢ
  5. Rank-normalize fitness scores
  6. Update: θ ← θ + α/(N·σ) · Σᵢ(Fᵢ · εᵢ)

Benefits:
  - No gradient computation (works on CPU)
  - Naturally parallelizable (each perturbation is independent)
  - Can handle non-differentiable objectives
  - Good for escaping sharp local minima

Usage:
    python3 -m tabula_rasa.evolution_strategies --steps 1000 --population 64 --sigma 0.02
    python3 -m tabula_rasa.evolution_strategies --resume checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


def _flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _unflatten_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Set model parameters from a flattened vector."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view(p.shape))
        idx += n


def _count_params(model: nn.Module) -> int:
    """Count total number of parameters."""
    return sum(p.numel() for p in model.parameters())


@dataclass
class ESResult:
    """Result of one ES training run."""

    final_loss: float
    best_loss: float
    steps: int
    population_size: int
    sigma: float
    lr: float
    elapsed_s: float
    params_count: int
    loss_history: list[float] = field(default_factory=list)


class EvolutionStrategies:
    """Evolution Strategies optimizer.

    Implements the OpenAI ES algorithm:
        θ ← θ + α/(N·σ) · Σᵢ(Fᵢ · εᵢ)

    Where:
        θ = model parameters (flattened)
        N = population size
        σ = noise standard deviation
        α = learning rate
        εᵢ = sampled noise vectors
        Fᵢ = fitness scores (higher = better)

    Args:
        model: PyTorch model to optimize.
        fitness_fn: Function that takes model and returns a scalar loss (lower = better).
        population_size: Number of perturbations per step (default 32).
        sigma: Noise standard deviation (default 0.02).
        lr: Learning rate for ES update (default 0.01).
        weight_decay: L2 regularization coefficient (default 0.001).
        antithetic: Use antithetic sampling (ε and -ε pairs, default True).
        rank_normalize: Rank-normalize fitness before update (default True).
    """

    def __init__(
        self,
        model: nn.Module,
        fitness_fn: Callable[[nn.Module], float],
        population_size: int = 32,
        sigma: float = 0.02,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        antithetic: bool = True,
        rank_normalize: bool = True,
    ):
        self.model = model
        self.fitness_fn = fitness_fn
        self.population_size = population_size if not antithetic else population_size // 2
        self.sigma = sigma
        self.lr = lr
        self.weight_decay = weight_decay
        self.antithetic = antithetic
        self.rank_normalize = rank_normalize

        self.params = _flatten_params(model)
        self.num_params = self.params.numel()
        self.device = self.params.device

        # For noise: cache on CPU, move to device for operations
        self.noise_device = self.device
        self.fitness_history: list[float] = []

    def _sample_noise(self) -> torch.Tensor:
        """Sample Gaussian noise for the population.

        Returns:
            Noise tensor (population_size, num_params).
        """
        noise = torch.randn(self.population_size, self.num_params, device=self.noise_device)
        return noise

    def _compute_fitness(self, flat_params: torch.Tensor) -> float:
        """Evaluate a single parameter vector.

        Args:
            flat_params: Flattened parameter vector.

        Returns:
            Fitness score (lower = better).
        """
        _unflatten_params(self.model, flat_params)
        self.model.eval()
        with torch.no_grad():
            loss = self.fitness_fn(self.model)
        return float(loss)

    def _rank_normalize(self, fitness: torch.Tensor) -> torch.Tensor:
        """Convert fitness scores to rank-based weights.

        Uses the standard ES rank transformation:
            Fᵢ = max(0, log(N/2 + 1) - log(rank(i)))

        This clips outliers and provides a smooth gradient signal.
        """
        N = fitness.size(0)
        # Lower loss = better → higher rank
        _, indices = torch.sort(fitness)  # ascending: best first
        ranks = torch.zeros(N, device=fitness.device)
        ranks[indices] = torch.arange(N, device=fitness.device, dtype=torch.float)
        # Rank weights: log(N/2 + 1) - log(rank + 1)
        weights = torch.log(torch.tensor(N / 2.0 + 1.0, device=fitness.device)) - torch.log(ranks + 1.0)
        weights = torch.clamp(weights, min=0.0)
        # Normalize to zero mean
        weights = weights - weights.mean()
        if weights.std() > 0:
            weights = weights / weights.std()
        return weights

    def step(self) -> dict:
        """Perform one ES step.

        Returns:
            Dict with 'loss', 'best_loss', 'noise_std'.
        """
        # 1. Sample noise
        noise = self._sample_noise()  # (population_size, num_params)

        # 2. Evaluate each perturbed model
        fitness = torch.zeros(self.population_size, device=self.noise_device)

        for i in range(self.population_size):
            perturbed = self.params + self.sigma * noise[i]
            fitness[i] = self._compute_fitness(perturbed)

            # Antithetic: also evaluate the negative perturbation
            if self.antithetic:
                perturbed_neg = self.params - self.sigma * noise[i]
                f_neg = self._compute_fitness(perturbed_neg)
                # Fitness for this direction is the average
                fitness[i] = (fitness[i] + f_neg) / 2.0

        # 3. Rank-normalize fitness
        if self.rank_normalize:
            weights = self._rank_normalize(fitness)
        else:
            # Z-score normalization
            weights = -(fitness - fitness.mean())
            if fitness.std() > 0:
                weights = weights / fitness.std()

        # 4. Compute gradient estimate
        # g = 1/(N·σ) · Σᵢ(wᵢ · εᵢ)
        weighted_noise = noise.t() @ weights  # (num_params,)
        gradient = weighted_noise / (self.population_size * self.sigma)

        # 5. L2 weight decay
        if self.weight_decay > 0:
            gradient = gradient - self.weight_decay * self.params

        # 6. Update parameters
        self.params = self.params + self.lr * gradient

        # 7. Apply to model
        _unflatten_params(self.model, self.params)

        # 8. Compute current loss
        current_loss = self._compute_fitness(self.params)
        self.fitness_history.append(current_loss)

        return {
            "loss": current_loss,
            "best_loss": min(self.fitness_history),
            "noise_std": self.sigma,
        }

    def train(self, num_steps: int = 100, verbose: bool = True) -> ESResult:
        """Run ES training for a given number of steps.

        Args:
            num_steps: Number of ES iterations.
            verbose: Print progress.

        Returns:
            ESResult with training summary.
        """
        best_loss = float("inf")
        start = time.time()

        for step in range(num_steps):
            info = self.step()
            best_loss = min(best_loss, info["loss"])

            if verbose and (step % max(1, num_steps // 10) == 0 or step == num_steps - 1):
                elapsed = time.time() - start
                print(f"  Step {step:>4d}/{num_steps} | loss={info['loss']:.4f} | best={best_loss:.4f} | σ={info['noise_std']:.4f} | {elapsed:.0f}s")

        elapsed = time.time() - start

        return ESResult(
            final_loss=self.fitness_history[-1] if self.fitness_history else 0.0,
            best_loss=best_loss,
            steps=num_steps,
            population_size=self.population_size,
            sigma=self.sigma,
            lr=self.lr,
            elapsed_s=elapsed,
            params_count=self.num_params,
            loss_history=self.fitness_history,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Fitness function builders
# ═══════════════════════════════════════════════════════════════════════


def make_math_fitness(
    tokenizer: Any,
    num_problems: int = 50,
    max_digits: int = 4,
    device: str = "cpu",
) -> Callable[[nn.Module], float]:
    """Create a fitness function for math accuracy.

    The fitness is 1 - accuracy (we minimize loss = maximize accuracy).
    Uses greedy decoding to evaluate the model.

    Args:
        tokenizer: MathTokenizer.
        num_problems: Number of problems per evaluation.
        max_digits: Max operand digits.
        device: Device string.

    Returns:
        Fitness function f(model) → loss (lower = better).
    """
    from tabula_rasa.dataset import generate_problem

    def fitness(model: nn.Module) -> float:
        model.eval()
        correct = 0
        total = num_problems

        for _ in range(total):
            expr, ans = generate_problem(1, max_digits)
            prompt = f"{expr}="
            try:
                generated = model.generate(tokenizer, prompt, max_new_tokens=12, temperature=0.0, top_k=1)
                if "=" in generated:
                    pred = generated.split("=")[-1].strip()
                    pred = "".join(c for c in pred if c.isdigit() or c == "-")
                    if pred == ans:
                        correct += 1
            except Exception:
                pass

        accuracy = correct / total
        # Return loss: we want to minimize this (lower = better)
        return 1.0 - accuracy

    return fitness


def make_cross_entropy_fitness(
    tokenizer: Any,
    num_samples: int = 100,
    max_digits: int = 4,
    device: str = "cpu",
) -> Callable[[nn.Module], float]:
    """Create a fitness function based on cross-entropy loss.

    More sample-efficient than accuracy-based fitness, especially in early training.

    Args:
        tokenizer: MathTokenizer.
        num_samples: Number of training samples per evaluation.
        max_digits: Max operand digits.
        device: Device string.

    Returns:
        Fitness function f(model) → loss (lower = better).
    """
    from torch.utils.data import DataLoader

    from tabula_rasa.dataset import MathDataset

    ds = MathDataset(tokenizer, num_samples=num_samples, min_digits=1, max_digits=max_digits, seed=42)
    dl = DataLoader(ds, batch_size=min(32, num_samples), shuffle=False)

    def fitness(model: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                _, loss, _ = model(x, y)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
        return total_loss / max(count, 1)

    return fitness


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolution Strategies Trainer")
    parser.add_argument("--steps", type=int, default=200, help="ES iterations")
    parser.add_argument("--population", type=int, default=32, help="Population size")
    parser.add_argument("--sigma", type=float, default=0.02, help="Noise std")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--preset", type=str, default="1M", choices=["1M", "5M"], help="Model preset")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--fitness", type=str, default="accuracy", choices=["accuracy", "xentropy"],
                        help="Fitness function")
    parser.add_argument("--digits", type=int, default=2, help="Max digits")
    parser.add_argument("--eval-problems", type=int, default=50, help="Problems per eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer, count_parameters
    from tabula_rasa.tokenizer import MathTokenizer

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  EVOLUTION STRATEGIES — Hyperscale Training")
    print("  Reference [53]: 'Evolution Strategies at the Hyperscale'")
    print("=" * 60)
    print()
    print(f"  Population: {args.population}")
    print(f"  Sigma:      {args.sigma}")
    print(f"  LR:         {args.lr}")
    print(f"  Steps:      {args.steps}")
    print(f"  Preset:     {args.preset}")
    print(f"  Digits:     {args.digits}")
    print()

    # Create model
    tok = MathTokenizer()
    cfg = Config()
    cfg.apply_preset(args.preset)
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    cfg.max_digits = args.digits

    model = MathTransformer(cfg)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"  Resumed from {args.resume}")

    n_params = count_parameters(model)
    print(f"  Model: {n_params:,} params")

    # Create fitness function
    if args.fitness == "accuracy":
        fitness_fn = make_math_fitness(tok, args.eval_problems, args.digits, "cpu")
    else:
        fitness_fn = make_cross_entropy_fitness(tok, args.eval_problems, args.digits, "cpu")

    # Create ES optimizer
    es = EvolutionStrategies(
        model=model,
        fitness_fn=fitness_fn,
        population_size=args.population,
        sigma=args.sigma,
        lr=args.lr,
        antithetic=True,
        rank_normalize=True,
    )

    print(f"  Params to optimize: {es.num_params:,}")
    print(f"  Evaluations per step: {args.population}")
    print(f"  Total evaluations: {args.population * args.steps:,}")
    print()

    # Train
    result = es.train(num_steps=args.steps)

    # Summary
    print()
    print("=" * 60)
    print("  ES TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final loss:     {result.final_loss:.4f}")
    print(f"  Best loss:      {result.best_loss:.4f}")
    print(f"  Steps:           {result.steps}")
    print(f"  Time:            {result.elapsed_s:.0f}s ({result.elapsed_s/60:.1f} min)")
    print(f"  Eval/step:       {result.population_size}")
    print(f"  Total evals:     {result.population_size * result.steps:,}")
    print("=" * 60)

    # Save checkpoint
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    torch.save({
        "step": result.steps,
        "model_state_dict": model.state_dict(),
        "loss": result.final_loss,
        "best_loss": result.best_loss,
    }, save_dir / "es_final.pt")
    print(f"  Saved: {save_dir / 'es_final.pt'}")
