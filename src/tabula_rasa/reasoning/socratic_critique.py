"""Socratic Critique Loop — Multi-turn self-correction with deterministic critic.

Generates math problems, feeds them to the model, runs a deterministic parser
to find errors, constructs targeted hints, lets the model revise, and tracks
how prediction error drops across rounds.

Flow:
  Round 1: Generate trace → Critic → Hint
  Round 2: Revise with hint → Critic → Hint (if needed)
  ...
  Round N: Trace passes critic → Log telemetry

Telemetry logged to socratic_telemetry.json for analysis.
"""

import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.math_parser import (
    verify_scratchpad,
    verify_equation,
    parse_expression,
    evaluate,
    OPS,
)


# ═══════════════════════════════════════════════════════════════════
# Telemetry
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RoundTelemetry:
    """Prediction metrics for one critique round."""

    round: int
    prediction_loss: float  # Cross-entropy loss on the scratchpad tokens
    prediction_perplexity: float  # exp(loss)
    max_logit: float  # Max output logit (confidence proxy)
    entropy: float  # Entropy of the output distribution
    num_correct_columns: int
    num_total_columns: int
    critic_errors: list[str]
    trace: str  # The model's raw scratchpad output


@dataclass
class ProblemTelemetry:
    """Full telemetry for one problem across all rounds."""

    expression: str = ""
    expected: str = ""
    n_rounds: int = 0
    passed: bool = False  # Did the model get it right by the final round?
    rounds: list[RoundTelemetry] = field(default_factory=list)
    total_time_ms: float = 0.0
    first_round_loss: float = 0.0
    last_round_loss: float = 0.0
    loss_drop_pct: float = 0.0  # improvement(%)


TELEMETRY_FILE = Path("memory/socratic_telemetry.json")


def _load_telemetry() -> list[dict]:
    if TELEMETRY_FILE.exists():
        try:
            return json.loads(TELEMETRY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_telemetry(entries: list[dict]):
    TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    TELEMETRY_FILE.write_text(json.dumps(entries, indent=2))


# ═══════════════════════════════════════════════════════════════════
# Problem Generator
# ═══════════════════════════════════════════════════════════════════


def generate_problem(
    op: str = "+",
    max_digits: int = 4,
    force_carry: bool = True,
) -> tuple[str, str, int]:
    """Generate a random math problem and its expected answer.

    Args:
        op: One of '+', '-', '*', '/'.
        max_digits: Maximum digits per operand (1-4).
        force_carry: If True, bias toward carry/borrow problems.

    Returns:
        (expression, expected_str, expected_int)
        e.g. ('24+18', '42', 42)
    """
    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)

    if op == "+" and force_carry and random.random() < 0.6:
        # Force a carry by ensuring column sums >= 10 in at least one position
        # Pick digits that produce carries
        a = random.randint(10 ** (d1 - 1), 10**d1 - 1) if d1 > 1 else random.randint(1, 9)
        b = random.randint(10 ** (d2 - 1), 10**d2 - 1) if d2 > 1 else random.randint(1, 9)
        # Ensure at least one column has carry
        # Simple approach: make the sum's digits require carries
        result = a + b
        if result == a:  # b was 0, retry
            b = random.randint(1, max(1, 10 ** min(d1, d2) - 1) if min(d1, d2) > 1 else 5)
            result = a + b
    elif op == "-" and force_carry and random.random() < 0.6:
        # Force borrow: make lower digit > upper digit in at least one column
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
# Hint Constructor
# ═══════════════════════════════════════════════════════════════════


CRITIC_TO_HINT = {
    "carry": "Incorrect carry. Remember: when two digits plus previous carry sum to 10 or more, you carry 1 to the next column.",
    "borrow": "Incorrect borrow. When subtracting, if the top digit is smaller than the bottom digit, you must borrow 1 from the next column.",
    "digit": "Incorrect result digit. Double-check the column-by-column arithmetic.",
    "missing_column": "Missing columns. Ensure every digit position has a corresponding carry-digit pair.",
    "extra_column": "Too many columns. The answer should have fewer or equal columns than the operands (plus possible final carry).",
    "non_digit": "Scratchpad contains non-digit characters. Only digits 0-9 are valid in carry and digit positions.",
}


def _classify_error(error: str) -> str:
    """Classify an error string into a hint category."""
    err_lower = error.lower()
    if "carry" in err_lower and "expected" in err_lower:
        return "carry"
    if "borrow" in err_lower and "expected" in err_lower:
        return "borrow"
    if "digit" in err_lower and "expected" in err_lower:
        return "digit"
    if "missing" in err_lower:
        return "missing_column"
    if "too many" in err_lower:
        return "extra_column"
    if "non-digit" in err_lower:
        return "non_digit"
    return "digit"


def construct_hint(
    problem: str,
    trace: str,
    critic_result: dict,
) -> str:
    """Convert the deterministic critic's output into a targeted revision hint.

    The hint tells the model exactly which column had the error and what
    the expected value should be, but does NOT give the full answer.

    Args:
        problem: The math expression (e.g. "24+18")
        trace: The model's scratchpad output (e.g. "24+18=312")
        critic_result: Output from verify_scratchpad()

    Returns:
        A hint string that can be appended to the prompt for revision.
    """
    errors = critic_result.get("errors", [])
    if not errors:
        return ""  # No hint needed — already correct

    # Pick the first error (most significant column error)
    first_error = errors[0]

    # Extract column number if present
    col_match = __import__("re").search(r"Column (\d+)", first_error)
    col_info = f" at column {col_match.group(1)}" if col_match else ""

    category = _classify_error(first_error)
    general_hint = CRITIC_TO_HINT.get(category, "Check your column-by-column arithmetic carefully.")

    # Build a targeted hint: specific error + general guidance
    hint_parts = [
        f"[CRITIQUE: {first_error}{col_info}]",
        general_hint,
        f"Problem: {problem}=",
    ]

    return " ".join(hint_parts)


# ═══════════════════════════════════════════════════════════════════
# Prediction Error Computation
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_prediction_loss(
    model: MathTransformer,
    tokenizer: MathTokenizer,
    prompt: str,
    target_scratchpad: str,
) -> RoundTelemetry:
    """Compute the model's prediction loss on a specific scratchpad output.

    Encodes the full sequence (prompt + scratchpad) and computes
    cross-entropy loss only on the scratchpad tokens (everything after '='),
    excluding prompt tokens via ignore_index=-100.

    Args:
        model: The transformer model.
        tokenizer: Matching tokenizer.
        prompt: Input prompt (e.g. "24+18=").
        target_scratchpad: Expected scratchpad output (e.g. "0406").

    Returns:
        A RoundTelemetry with loss, perplexity, entropy, and max_logit.
    """
    full_text = prompt + target_scratchpad
    ids = tokenizer.encode(full_text)
    input_ids = torch.tensor([ids])
    targets = torch.tensor([ids])

    # Mask everything up to and including '=' with -100 (ignore_index)
    eq_id = tokenizer.stoi.get("=", -1)
    if eq_id >= 0:
        eq_positions = (targets == eq_id).nonzero(as_tuple=True)
        if len(eq_positions[0]) > 0:
            eq_pos = eq_positions[0][0].item()
            targets[0, : eq_pos + 1] = -100

    logits, loss, _ = model(input_ids, targets=targets)

    # Compute per-token metrics on the scratchpad portion
    scratchpad_start = eq_pos + 1 if eq_id >= 0 else len(ids) // 2
    scratchpad_logits = logits[0, scratchpad_start - 1 : -1, :]  # shifted for next-token pred
    scratchpad_targets = targets[0, scratchpad_start:]

    # Filter out ignored positions
    valid_mask = scratchpad_targets != -100
    valid_logits = scratchpad_logits[valid_mask]
    valid_targets = scratchpad_targets[valid_mask]

    if len(valid_targets) == 0:
        return RoundTelemetry(
            round=0,
            prediction_loss=0.0,
            prediction_perplexity=1.0,
            max_logit=0.0,
            entropy=0.0,
            num_correct_columns=0,
            num_total_columns=0,
            critic_errors=[],
            trace=target_scratchpad,
        )

    # Perplexity
    loss_val = loss.item()
    perplexity = math.exp(loss_val)

    # Entropy on the last valid position
    last_logits = scratchpad_logits[
        valid_mask.nonzero(as_tuple=True)[0][-1].item(), :
    ]
    probs = F.softmax(last_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_logit = last_logits.max().item()

    return RoundTelemetry(
        round=0,
        prediction_loss=loss_val,
        prediction_perplexity=perplexity,
        max_logit=max_logit,
        entropy=entropy,
        num_correct_columns=0,
        num_total_columns=0,
        critic_errors=[],
        trace=target_scratchpad,
    )


# ═══════════════════════════════════════════════════════════════════
# Socratic Critique Loop
# ═══════════════════════════════════════════════════════════════════


class SocraticCritiqueLoop:
    """Multi-turn critique loop: Generate → Critic → Hint → Revise → Telemetry.

    Args:
        model: A trained MathTransformer.
        tokenizer: Matching MathTokenizer.
        max_rounds: Maximum critique rounds per problem (default: 5).
        max_new_tokens: Tokens to generate per call (default: 15).
        temperature: Sampling temperature for generation (default: 0.0 = greedy).
    """

    def __init__(
        self,
        model: MathTransformer,
        tokenizer: MathTokenizer,
        max_rounds: int = 5,
        max_new_tokens: int = 15,
        temperature: float = 0.0,
    ):
        self.model = model
        self.tok = tokenizer
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = next(model.parameters()).device

    # ── Batch Problem Generation ──────────────────────────────────

    def generate_problems(
        self,
        op: str = "+",
        count: int = 50,
        max_digits: int = 4,
        force_carry: bool = True,
        seed: Optional[int] = None,
    ) -> list[tuple[str, str, int]]:
        """Generate a batch of problems for evaluation.

        Returns list of (expr, expected_str, expected_int).
        """
        if seed is not None:
            random.seed(seed)
        problems = []
        for _ in range(count):
            problems.append(generate_problem(op, max_digits, force_carry))
        return problems

    # ── Single Problem Critique ───────────────────────────────────

    def critique_problem(self, problem: str, expected: str) -> ProblemTelemetry:
        """Run the full critique loop on one math problem.

        Args:
            problem: Math expression (e.g. "24+18")
            expected: Correct answer string (e.g. "42")

        Returns:
            ProblemTelemetry with round-by-round data.
        """
        prompt = f"{problem}="
        t0 = time.time()
        telemetry = ProblemTelemetry(expression=problem, expected=expected)

        current_trace = ""
        first_loss = None

        for round_num in range(1, self.max_rounds + 1):
            # --- Generate trace ---
            full_prompt = prompt
            if current_trace and telemetry.rounds:
                last_hint = construct_hint(
                    problem, current_trace, {"errors": telemetry.rounds[-1].critic_errors}
                )
                full_prompt = last_hint

            raw = self.model.generate(
                self.tok,
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=5 if self.temperature > 0 else 0,
            )

            # Extract the scratchpad portion (after '=')
            if "=" in raw:
                current_trace = raw.split("=", 1)[-1].strip()
            else:
                current_trace = raw.strip()

            # Strip EOS/PAD
            current_trace = (
                current_trace.replace("<EOS>", "")
                .replace("<PAD>", "")
                .replace("<BOS>", "")
                .strip()
            )

            # Limit scratchpad to sane length to prevent infinite-repeat generation.
            # At most 2 * (max_digits + 1) chars for add/sub (carry+digit per column).
            max_scratchpad_len = 2 * (4 + 2)  # 12 chars
            if len(current_trace) > max_scratchpad_len:
                # Try find shortest valid prefix — reuse the same logic as model.py
                valid_trimmed = current_trace
                for end in range(max_scratchpad_len, 1, -2):  # step by 2 (pairs)
                    candidate = current_trace[:end]
                    v = verify_scratchpad(f"{problem}={candidate}")
                    if v.get("valid", False):
                        valid_trimmed = candidate
                        break
                if len(valid_trimmed) <= max_scratchpad_len:
                    current_trace = valid_trimmed
                else:
                    current_trace = current_trace[:max_scratchpad_len]

            # --- Deterministic Critic ---
            full_equation = f"{problem}={current_trace}"
            critic_result = verify_scratchpad(full_equation)

            # Also try verify_equation as fallback
            if not critic_result.get("errors"):
                eq_result = verify_equation(full_equation)
                if eq_result.get("valid"):
                    critic_result["valid"] = True

            errors = critic_result.get("errors", [])

            # --- Prediction metrics ---
            prompt_part = f"{problem}="
            round_telemetry = compute_prediction_loss(
                self.model, self.tok, prompt_part, current_trace
            )
            round_telemetry.round = round_num
            round_telemetry.critic_errors = errors

            if critic_result.get("columns"):
                cols = critic_result["columns"]
                round_telemetry.num_correct_columns = sum(1 for c in cols if c.get("correct"))
                round_telemetry.num_total_columns = len(cols)

            round_telemetry.trace = current_trace
            telemetry.rounds.append(round_telemetry)

            if first_loss is None:
                first_loss = round_telemetry.prediction_loss

            # --- Early exit on success ---
            if critic_result.get("valid", False) or not errors:
                telemetry.passed = True
                break

            # --- Early exit on no improvement (trace unchanged across 2 rounds) ---
            if len(telemetry.rounds) >= 3:
                prev_trace = telemetry.rounds[-2].trace
                if prev_trace == current_trace:
                    # Model is stuck — stop
                    break

        # --- Aggregate telemetry ---
        telemetry.total_time_ms = (time.time() - t0) * 1000
        telemetry.n_rounds = len(telemetry.rounds)
        telemetry.first_round_loss = first_loss or 0.0
        telemetry.last_round_loss = (
            telemetry.rounds[-1].prediction_loss if telemetry.rounds else 0.0
        )
        if first_loss and first_loss > 0:
            telemetry.loss_drop_pct = (
                (first_loss - telemetry.last_round_loss) / first_loss * 100
            )

        return telemetry

    # ── Batch Critique ────────────────────────────────────────────

    def run_benchmark(
        self,
        op: str = "+",
        count: int = 50,
        max_digits: int = 4,
        force_carry: bool = True,
        seed: int = 42,
        quiet: bool = False,
    ) -> list[dict]:
        """Run the critique loop on a batch of problems and log telemetry.

        Args:
            op: Operation to benchmark.
            count: Number of problems to test.
            max_digits: Max digits per operand.
            force_carry: Bias toward carry problems.
            seed: RNG seed for reproducibility.
            quiet: If True, don't print progress.

        Returns:
            List of telemetry dicts (one per problem).
        """
        problems = self.generate_problems(op, count, max_digits, force_carry, seed)
        results = []

        for i, (expr, expected_str, expected_int) in enumerate(problems):
            if not quiet:
                status = (
                    f"  [{i + 1}/{count}] {expr}={expected_str}..."
                )
                print(status, end="")

            telemetry = self.critique_problem(expr, expected_str)
            results.append(asdict(telemetry))

            if not quiet:
                symbol = "✓" if telemetry.passed else "✗"
                rounds = telemetry.n_rounds
                loss_drop = telemetry.loss_drop_pct
                print(f" {symbol} ({rounds}r, loss -{loss_drop:.1f}%)")

        # Compute aggregate stats
        solved = sum(1 for r in results if r["passed"])
        avg_rounds = sum(r["n_rounds"] for r in results) / len(results) if results else 0
        avg_loss_drop = sum(r["loss_drop_pct"] for r in results) / len(results) if results else 0
        avg_time_ms = sum(r["total_time_ms"] for r in results) / len(results) if results else 0

        summary = {
            "config": {
                "op": op,
                "problems": count,
                "max_digits": max_digits,
                "force_carry": force_carry,
                "max_rounds": self.max_rounds,
                "temperature": self.temperature,
            },
            "summary": {
                "solved": solved,
                "total": count,
                "solved_pct": round(solved / count * 100, 1) if count > 0 else 0.0,
                "avg_rounds": round(avg_rounds, 2),
                "avg_loss_drop_pct": round(avg_loss_drop, 1),
                "avg_time_ms": round(avg_time_ms, 1),
            },
            "results": results,
        }

        # Save telemetry
        existing = _load_telemetry()
        existing.append(summary)
        _save_telemetry(existing)

        if not quiet:
            print()
            print(f"  {'=' * 40}")
            print(f"  Solved: {solved}/{count} ({summary['summary']['solved_pct']}%)")
            print(f"  Avg rounds: {avg_rounds:.2f}")
            print(f"  Avg loss drop: {avg_loss_drop:.1f}%")
            print(f"  Avg time: {avg_time_ms:.1f}ms")
            print(f"  Telemetry saved to {TELEMETRY_FILE}")

        return results


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════


def main():
    """Run the Socratic Critique Loop from the command line.

    Usage:
        python3 -m egefalos.socratic_critique [op] [--count N] [--max-digits N]
                                              [--rounds N] [--temperature T] [--seed S]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Socratic Critique Loop — multi-turn self-correction benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("op", nargs="?", default="+", help="Operation: + - * / (default: +)")
    parser.add_argument("--count", type=int, default=50, help="Number of problems")
    parser.add_argument("--max-digits", type=int, default=4, help="Max digits per operand")
    parser.add_argument("--rounds", type=int, default=5, help="Max critique rounds")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to .pt checkpoint")
    parser.add_argument("--no-carry", action="store_true", help="Disable force-carry bias")
    args = parser.parse_args()

    # Load model
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Auto-detect best available checkpoint
        candidates = [
            Path("specialists/math/add/best.pt"),
            Path("specialists/math/add/final.pt"),
            Path("specialists/math/general/best.pt"),
            Path("specialists/math/general/final.pt"),
            Path("checkpoints/best.pt"),
            Path("checkpoints/final.pt"),
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            print("No checkpoint found. Train first: python3 train_specialist.py add --quick")
            return

    tokenizer_path = ckpt_path.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        # Try the mul tokenizer as fallback (we saw it exists)
        mul_tok = Path("specialists/math/mul/tokenizer.json")
        if mul_tok.exists():
            tokenizer_path = mul_tok
        else:
            print(f"Tokenizer not found at {tokenizer_path}")
            return

    print(f"Loading model from {ckpt_path}")
    tok = MathTokenizer.load(str(tokenizer_path))
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    params = count_parameters(model)
    print(f"Model params: {params:,}")
    print(f"Tokenizer vocab: {tok.vocab_size}")
    print(f"Max rounds: {args.rounds}")
    print()

    loop = SocraticCritiqueLoop(
        model=model,
        tokenizer=tok,
        max_rounds=args.rounds,
        temperature=args.temperature,
    )

    loop.run_benchmark(
        op=args.op,
        count=args.count,
        max_digits=args.max_digits,
        force_carry=not args.no_carry,
        seed=args.seed,
        quiet=False,
    )


if __name__ == "__main__":
    main()
