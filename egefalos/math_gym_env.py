"""MathGymEnv — Gymnasium-style environment for arithmetic self-play.

A reinforcement learning environment where an agent solves arithmetic problems
by generating tokens one at a time. Compatible with the existing Micro-MCTS
engine in ``egefalos/mcts.py`` and with standard RL libraries (SB3, etc.).

State:
  - A math problem is generated on reset() (e.g. "12+34")
  - The agent outputs token IDs one at a time (the scratchpad answer)
  - Episode ends when the agent emits <EOS> or hits max_seq_len

Reward:
  +1  → Final answer is correct (passes verify_scratchpad)
  -1  → Final answer has a syntax error (non-digit chars in scratchpad)
   0  → Intermediate steps (episode not yet complete)

Observation:
  - Token IDs generated so far, left-padded to max_seq_len
  - Or: the raw token sequence (for models that handle variable length)

Usage:
    import gymnasium as gym
    from egefalos.math_gym_env import MathGymEnv

    env = MathGymEnv(op='+', max_digits=2, tokenizer=tok, model=model)
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # or from your policy
        obs, reward, terminated, truncated, info = env.step(action)
"""

import math
import random
from typing import Any, Optional

import torch
import torch.nn.functional as F

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer
from tabula_rasa.math_parser import (
    parse_expression,
    evaluate,
    verify_equation,
    verify_scratchpad,
    OPS,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


# ═══════════════════════════════════════════════════════════════════
# Math Problem Generator
# ═══════════════════════════════════════════════════════════════════


def _gen_problem(op: str, max_digits: int, force_carry: bool = True) -> tuple[str, str, int]:
    """Generate a math problem. Returns (expr, expected_str, expected_int)."""
    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)

    if op == "+" and force_carry and random.random() < 0.5:
        a = random.randint(10 ** (d1 - 1), 10**d1 - 1) if d1 > 1 else random.randint(1, 9)
        b = random.randint(10 ** (d2 - 1), 10**d2 - 1) if d2 > 1 else random.randint(1, 9)
        if a + b == a:
            b = random.randint(1, max(1, 10 ** min(d1, d2) - 1) if min(d1, d2) > 1 else 5)
    elif op == "-" and force_carry and random.random() < 0.5:
        b_val = random.randint(10 ** (d2 - 1), 10**d2 - 1) if d2 > 1 else random.randint(1, 9)
        a_val = b_val + random.randint(1, 10**max(d1, d2) - 1)
        a, b = a_val, b_val
    else:
        a = random.randint(0, 10**d1 - 1) if d1 > 1 else random.randint(0, 9)
        b = random.randint(0, 10**d2 - 1) if d2 > 1 else random.randint(0, 9)

    if op == "/" and b == 0:
        b = random.randint(1, 9)

    expr = f"{a}{op}{b}"
    expected = evaluate(a, b, op)
    return expr, str(expected), expected


# ═══════════════════════════════════════════════════════════════════
# MathGymEnv
# ═══════════════════════════════════════════════════════════════════


class MathGymEnv:
    """Gymnasium-style environment for arithmetic self-play.

    The agent learns to generate scratchpad output token by token.
    Compatible with both Gymnasium's API and direct use with the MCTS engine.

    Args:
        op: Operation — one of '+', '-', '*', '/'.
        max_digits: Maximum digits per operand (1-4).
        tokenizer: MathTokenizer for encode/decode.
        model: Optional MathTransformer for policy/value queries.
        max_steps: Max tokens per episode (default: 20).
        force_carry: Bias toward carry/borrow problems.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        op: str = "+",
        max_digits: int = 4,
        tokenizer: Optional[MathTokenizer] = None,
        model: Optional[MathTransformer] = None,
        max_steps: int = 20,
        force_carry: bool = True,
        seed: Optional[int] = None,
    ):
        self.op = op
        self.max_digits = max_digits
        self.tok = tokenizer or MathTokenizer()
        self.model = model
        self.max_steps = max_steps
        self.force_carry = force_carry
        self.device = torch.device("cpu")

        if self.model is not None:
            self.device = next(model.parameters()).device

        if seed is not None:
            random.seed(seed)

        # Gymnasium spaces (for compatibility)
        self.vocab_size = self.tok.vocab_size
        if HAS_GYM:
            self.action_space = spaces.Discrete(self.vocab_size)
            self.observation_space = spaces.Box(
                low=0, high=self.vocab_size - 1,
                shape=(max_steps + 10,),  # problem + scratchpad tokens
                dtype=int,
            )

        # Internal state
        self._reset_state()

    def _reset_state(self):
        """Clear episode-specific state."""
        self._expr = ""
        self._expected_str = ""
        self._expected_int = 0
        self._tokens: list[int] = []  # Token IDs generated so far (after '=')
        self._prompt_tokens: list[int] = []  # Token IDs for the expression
        self._step = 0
        self._done = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """Reset the environment and generate a new problem.

        Args:
            seed: Optional seed.
            options: Optional dict with 'problem' key to set a specific problem.

        Returns:
            (observation, info) tuple.
        """
        if seed is not None:
            random.seed(seed)

        self._reset_state()

        if options and "problem" in options:
            problem_str = options["problem"]
            parsed = parse_expression(problem_str)
            if parsed is None:
                raise ValueError(f"Invalid problem: {problem_str}")
            op, a, b, formatted = parsed
            self._expr = formatted
            self._expected_int = evaluate(a, b, op)
            self._expected_str = str(self._expected_int)
        else:
            self._expr, self._expected_str, self._expected_int = _gen_problem(
                self.op, self.max_digits, self.force_carry
            )

        # Encode the prompt (expression + '=')
        prompt_text = f"{self._expr}="
        self._prompt_tokens = self.tok.encode(prompt_text, add_special_tokens=True)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple:
        """Take a step by generating a token.

        Provides **dense rewards**:
        - +0.1 per valid token (digit or carry-digit pair)
        - +0.5 per correctly calculated column
        - +1.0 for the final correct answer
        - -0.1 for invalid tokens (non-digit when digit expected)

        Args:
            action: Token ID to append.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("Episode already terminated. Call reset() first.")

        self._tokens.append(action)
        self._step += 1

        # Check termination conditions
        terminated = False
        truncated = False
        reward = 0.0

        # Compute intermediate dense reward for this step
        decoded = self.tok.decode(self._tokens, skip_special=True).strip()

        # Dense reward: +0.1 for any valid digit/carry-digit token
        if decoded and decoded[-1].isdigit():
            reward += 0.1

        # Dense reward: +0.5 for a correctly completed column (pair of carry+digit)
        if len(decoded) >= 2 and len(decoded) % 2 == 0:
            # We have a complete column pair — check correctness
            expected_scratchpad = self._generate_expected_scratchpad()
            if expected_scratchpad and decoded == expected_scratchpad[:len(decoded)]:
                reward += 0.5

        # <EOS> token → episode complete, evaluate final answer
        if action == self.tok.eos_id or self._step >= self.max_steps:
            terminated = True
            reward = self._compute_reward()  # +1 or -1

        obs = self._get_obs()

        if terminated:
            self._done = True

        return obs, reward, terminated, truncated, self._get_info()

    def _generate_expected_scratchpad(self) -> str:
        """Generate the correct scratchpad for the current problem.

        Uses fused carry-digit format matching the model's training.
        Returns the expected scratchpad string (e.g. '0406' for 12+34).
        """
        parts = self._expr.split(self.op, 1)
        if len(parts) != 2:
            return ""
        a_str, b_str = parts
        try:
            a = int(a_str)
            b = int(b_str)
        except ValueError:
            return ""
        ans = evaluate(a, b, self.op)
        if self.op in ("+", "-"):
            carry = 0
            sp = ""
            ra, rb = str(a)[::-1], str(b)[::-1]
            max_len = max(len(ra), len(rb))
            for i in range(max_len):
                da = int(ra[i]) if i < len(ra) else 0
                db = int(rb[i]) if i < len(rb) else 0
                if self.op == "+":
                    total = da + db + carry
                    carry = total // 10
                    digit = total % 10
                else:  # subtraction
                    if da < db + carry:
                        da += 10
                    total = da - db - carry
                    carry = 1 if da >= 10 else 0
                    digit = total % 10 if total >= 0 else (total + 10) % 10
                sp += f"{carry}{digit}"
            if carry and self.op == "+":
                sp += f"0{carry}"
            return sp
        return str(ans)

    def _compute_reward(self) -> float:
        """Evaluate the token sequence and return the reward.

        +1 for correct answer.
        -1 for syntax error / wrong answer.
         0 for incomplete (shouldn't happen here — only called on termination).
        """
        # Decode the generated tokens
        scratchpad = self.tok.decode(self._tokens, skip_special=True).strip()

        # Check for invalid characters
        if not scratchpad.isdigit():
            return -1.0

        # Verify as scratchpad format
        full_equation = f"{self._expr}={scratchpad}"
        result = verify_scratchpad(full_equation)

        if result.get("valid", False):
            return 1.0

        # Also try plain equation verification
        eq_result = verify_equation(full_equation)
        if eq_result.get("valid", False):
            return 1.0

        return -1.0

    def _get_obs(self) -> torch.Tensor:
        """Build the current observation tensor.

        For Gymnasium API: returns the full token sequence
        (prompt + generated so far) as a tensor.
        """
        full_ids = self._prompt_tokens + self._tokens
        return torch.tensor([full_ids], dtype=torch.long, device=self.device)

    def _get_info(self) -> dict:
        """Return diagnostic info about the current episode."""
        return {
            "expression": self._expr,
            "expected": self._expected_str,
            "step": self._step,
            "max_steps": self.max_steps,
            "tokens": list(self._tokens),
            "scratchpad": self.tok.decode(self._tokens, skip_special=True).strip(),
        }

    def render(self, mode: str = "human"):
        """Print the current state."""
        scratchpad = self.tok.decode(self._tokens, skip_special=True)
        print(f"Problem: {self._expr}=")
        print(f"Tokens: {self._tokens}")
        print(f"Scratchpad: '{scratchpad}'")
        print(f"Step: {self._step}/{self.max_steps}")

    def close(self):
        """Clean up resources."""
        pass

    # ── MCTS-Compatible Interface ──────────────────────────────────

    def get_legal_actions(self, input_ids: torch.Tensor) -> list[int]:
        """Get legal actions at the current state.

        For the MCTS engine: returns all vocabulary token IDs.
        Subclasses can override to add constraints (e.g. grammar rules).

        Args:
            input_ids: Current input tensor (1, seq_len).

        Returns:
            List of valid token IDs.
        """
        return list(range(self.vocab_size))

    def is_terminal(self, input_ids: torch.Tensor) -> bool:
        """Check if the current state is terminal.

        Args:
            input_ids: Current input tensor.

        Returns:
            True if the last token is <EOS> or max length reached.
        """
        if input_ids.size(1) >= self.max_steps + 10:
            return True
        last_token = input_ids[0, -1].item()
        return last_token == self.tok.eos_id

    def evaluate_state(self, input_ids: torch.Tensor) -> float:
        """Score a completed state for MCTS value backup.

        Args:
            input_ids: Completed sequence tensor.

        Returns:
            Score in [-1, 1] — +1 for correct, -1 otherwise.
        """
        # Decode the scratchpad portion (after '=')
        full_text = self.tok.decode(input_ids[0].tolist(), skip_special=True)
        if "=" in full_text:
            _, scratchpad = full_text.split("=", 1)
            result = verify_scratchpad(f"{self._expr}={scratchpad}")
            if result.get("valid", False):
                return 1.0
            eq_result = verify_equation(f"{self._expr}={scratchpad}")
            if eq_result.get("valid", False):
                return 1.0
        return -1.0

    # ── Self-Play Training Support ──────────────────────────────────

    def generate_training_data(
        self,
        model: MathTransformer,
        num_games: int = 10,
        max_rollout_tokens: int = 8,
        temperature: float = 0.8,
    ) -> list[dict]:
        """Generate self-play training data using Micro-MCTS.

        For each game:
        1. Reset the environment → get a problem
        2. Use MCTS to select actions
        3. Record (state, MCTS policy, outcome) for training

        Args:
            model: The policy+value network.
            num_games: Number of self-play games.
            max_rollout_tokens: Max tokens per MCTS rollout.
            temperature: Sampling temperature.

        Returns:
            List of training examples: {state, policy, value, expression, scratchpad}
        """
        try:
            from egefalos.mcts import micro_mcts_search, MCTSNode, select, expand, rollout, backup
        except ImportError:
            print("[!] egefalos.mcts not available — install or check PYTHONPATH")
            return []

        examples = []

        for game_idx in range(num_games):
            obs, info = self.reset()
            expr = info["expression"]

            # Build initial MCTS state from the prompt
            prompt_ids = obs[0]  # (seq_len,) tensor
            root = MCTSNode(token_ids=prompt_ids.tolist())

            game_examples = []
            game_done = False

            while not game_done:
                # Run MCTS from root
                mcts_result = micro_mcts_search(
                    model,
                    self.tok,
                    prompt_ids.unsqueeze(0),  # (1, seq_len)
                    concept=expr,
                    temperature=temperature,
                    top_k=5,
                    simulations=min(16, len(self.tok.stoi)),
                    max_rollout_tokens=max_rollout_tokens,
                    c_puct=1.0,
                )

                # The MCTS returns the best action
                action_id = mcts_result.get("action", -1)
                if action_id < 0 or action_id >= self.vocab_size:
                    action_id = random.randrange(self.vocab_size)

                # Build MCTS policy from visit counts
                policy = {}
                for tok_id, child in root.children.items():
                    policy[tok_id] = child.visit_count

                # Normalize policy
                total_visits = sum(policy.values()) if policy else 1
                policy_probs = {
                    tok_id: visits / total_visits
                    for tok_id, visits in policy.items()
                }

                # Record training example
                game_examples.append({
                    "state": prompt_ids.clone(),
                    "policy": policy_probs,
                    "expression": expr,
                })

                # Step in the environment
                obs, reward, terminated, truncated, info = self.step(action_id)
                game_done = terminated or truncated

                # Update input_ids for next MCTS iteration
                prompt_ids = obs[0]

                if game_done:
                    outcome = reward  # +1 or -1
                    for ex in game_examples:
                        ex["value"] = outcome
                    examples.extend(game_examples)

            if (game_idx + 1) % 5 == 0:
                print(f"  [*] Game {game_idx + 1}/{num_games} — {expr} — "
                      f"outcome={'✓' if outcome > 0 else '✗'}")

        return examples


# ═══════════════════════════════════════════════════════════════════
# Gymnasium Wrapper (provides the official Gymnasium interface)
# ═══════════════════════════════════════════════════════════════════


if HAS_GYM:

    class GymMathGymEnv(MathGymEnv, gym.Env):
        """Gymnasium-compatible wrapper for MathGymEnv.

        Provides the official ``gym.Env`` interface with:
        - ``observation_space``: ``Box(0, vocab_size-1, (max_seq_len,), int)``
        - ``action_space``: ``Discrete(vocab_size)``
        - ``reset()`` returns ``(obs, info)``
        - ``step()`` returns ``(obs, reward, terminated, truncated, info)``
        """

        metadata = {"render_modes": ["human"]}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Already set in parent if HAS_GYM is True at import time,
            # but safe to re-set for the gym.Env subclass
            self.action_space = spaces.Discrete(self.vocab_size)
            self.observation_space = spaces.Box(
                low=0, high=self.vocab_size - 1,
                shape=(self.max_steps + 10,),
                dtype=int,
            )

        def reset(self, *, seed=None, options=None):
            return super().reset(seed=seed, options=options)

        def step(self, action):
            return super().step(action)


# ═══════════════════════════════════════════════════════════════════
# CLI / Quick Test
# ═══════════════════════════════════════════════════════════════════


def main():
    """Quick-test the MathGymEnv with a trained model or random policy.

    Usage:
        python3 -m egefalos.math_gym_env --random        # Random policy
        python3 -m egefalos.math_gym_env --greedy         # Greedy policy
        python3 -m egefalos.math_gym_env --mcts           # MCTS self-play
    """
    import argparse

    parser = argparse.ArgumentParser(description="MathGymEnv quick test")
    parser.add_argument("--random", action="store_true", help="Random policy")
    parser.add_argument("--greedy", action="store_true", help="Greedy argmax policy")
    parser.add_argument("--mcts", action="store_true", help="MCTS self-play")
    parser.add_argument("--op", default="+", help="Operation")
    parser.add_argument("--games", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-digits", type=int, default=2, help="Max digits")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    from pathlib import Path
    import torch
    from tabula_rasa.config import Config

    # Load model
    ckpt = Path("specialists/math/add/best.pt")
    tok = MathTokenizer.load("specialists/math/add/tokenizer.json") if (Path("specialists/math/add/tokenizer.json")).exists() else MathTokenizer()
    model = None

    if ckpt.exists():
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        tok.max_seq_len = cfg.max_seq_len
        tmp_model = MathTransformer(cfg)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        tmp_model.load_state_dict(state["model_state_dict"])
        tmp_model.eval()
        model = tmp_model
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    else:
        print("No checkpoint found. Use --random to test without model.")
        if not args.random:
            return

    env = MathGymEnv(
        op=args.op,
        max_digits=args.max_digits,
        tokenizer=tok,
        model=model,
        seed=args.seed,
    )

    if args.random:
        print(f"=== Random Policy ({args.op}, {args.max_digits}-digit) ===")
        correct = 0
        for i in range(args.games):
            obs, info = env.reset()
            done = False
            while not done:
                action = random.randrange(tok.vocab_size)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            symbol = "✓" if reward > 0 else "✗"
            if reward > 0:
                correct += 1
            print(f"  [{i + 1}/{args.games}] {info['expression']}={info['expected']} "
                  f"-> '{info['scratchpad']}' {symbol}")

        print(f"\n  Result: {correct}/{args.games} correct ({correct/args.games*100:.0f}%)")

    elif args.greedy and model:
        print(f"=== Greedy Policy ({args.op}, {args.max_digits}-digit) ===")
        correct = 0
        for i in range(args.games):
            obs, info = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    logits, _, _ = model.forward(obs)
                    action = logits[0, -1].argmax().item()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            symbol = "✓" if reward > 0 else "✗"
            if reward > 0:
                correct += 1
            print(f"  [{i + 1}/{args.games}] {info['expression']}={info['expected']} "
                  f"-> '{info['scratchpad']}' {symbol}")

        print(f"\n  Result: {correct}/{args.games} correct ({correct/args.games*100:.0f}%)")

    elif args.mcts and model:
        print(f"=== MCTS Self-Play ({args.op}, {args.max_digits}-digit) ===")
        data = env.generate_training_data(
            model=model,
            num_games=args.games,
            temperature=0.8,
        )
    else:
        print("No policy selected. Use --random, --greedy, or --mcts.")


if __name__ == "__main__":
    main()
