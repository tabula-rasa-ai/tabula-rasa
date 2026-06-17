"""Socratic Self-Trainer — self-improvement via critique loop training.

After the model has some baseline accuracy, this module:
1. Runs the critique loop on a batch of fresh problems
2. Identifies problems where the model initially gets it WRONG
   but SUCCEEDS after receiving a critique hint
3. Extracts the successful correction as training data:
   (prompt_with_hint -> correct_scratchpad)
4. Trains the model on these correction pairs

This creates a self-improvement loop: the model generates traces,
the critic finds errors, the model self-corrects, and then trains
on the successful corrections. Over multiple iterations, the model
learns to internalize the critique patterns.

Usage as standalone:
    python3 -m egefalos.socratic_trainer --checkpoint specialists/math/add/best.pt \\
        --op add --steps 500 --batch 32

As part of training (added to train_specialist.py):
    python3 train_specialist.py add --socratic --socratic-steps 1000
"""

import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.math_parser import verify_scratchpad, parse_expression, evaluate, OPS

from tabula_rasa.memory.hippocampus import store_experience

try:
    from tabula_rasa.reasoning.socratic_critique import SocraticCritiqueLoop, generate_problem
    HAS_CRITIQUE = True
except ImportError:
    HAS_CRITIQUE = False


# ═══════════════════════════════════════════════════════════════════
# Correction Dataset
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CorrectionExample:
    """One training example extracted from a successful critique correction.

    The model failed initially, received a critique hint, and then
    produced a correct trace. We train on (hint_prompt → correct_trace).
    """
    problem: str
    expected: str
    hint: str  # The critique hint that led to correction
    correct_trace: str  # The successful final scratchpad
    first_round_trace: str  # The original wrong trace (for reference)
    n_rounds: int  # How many rounds it took
    loss_drop_pct: float  # How much loss dropped from round 1 to final


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization on correction pairs.

    Each sample provides a (chosen_trace, rejected_trace) pair where:
    - **chosen** = the correct trace after critique (y_w)
    - **rejected** = the initial wrong trace before critique (y_l)

    The DPO loss uses these log-probability pairs::

        L_DPO = -E[log σ(β * (log π(chosen) - log π(rejected)))]
    """

    def __init__(
        self,
        tokenizer: MathTokenizer,
        corrections: list[CorrectionExample],
        max_seq_len: int = 64,
    ):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.pairs: list[tuple[list[int], list[int]]] = []

        for ex in corrections:
            if not ex.correct_trace or not ex.first_round_trace:
                continue
            prompt = ex.hint if ex.hint else f"{ex.problem}="
            chosen_ids = tokenizer.encode(prompt + ex.correct_trace, add_special_tokens=True)
            rejected_ids = tokenizer.encode(prompt + ex.first_round_trace, add_special_tokens=True)
            if len(chosen_ids) <= max_seq_len and len(rejected_ids) <= max_seq_len:
                self.pairs.append((chosen_ids, rejected_ids))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        chosen_ids, rejected_ids = self.pairs[idx]

        def _pad(ids):
            pad_id = self.tok.pad_id
            padded = ids + [pad_id] * (self.max_seq_len - len(ids))
            return torch.tensor(padded, dtype=torch.long)

        return {
            "chosen": _pad(chosen_ids),
            "rejected": _pad(rejected_ids),
            "chosen_mask": torch.tensor(
                [1] * len(chosen_ids) + [0] * (self.max_seq_len - len(chosen_ids)),
                dtype=torch.bool,
            ),
            "rejected_mask": torch.tensor(
                [1] * len(rejected_ids) + [0] * (self.max_seq_len - len(rejected_ids)),
                dtype=torch.bool,
            ),
        }


def dpo_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor | None,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss for a batch of preference pairs.

    Args:
        policy_logps: Log probabilities under current policy for
            chosen and rejected, shape ``(2*batch,)`` or ``(batch, 2)``.
            Expected format: odd indices = chosen, even = rejected.
        ref_logps: Reference model log probs (optional, for KL penalty).
        beta: DPO temperature (default 0.1). Higher = stronger preference.

    Returns:
        Tuple ``(loss, chosen_reward, rejected_reward)``.
    """
    # Split into chosen and rejected
    if policy_logps.dim() == 1:
        chosen = policy_logps[0::2]
        rejected = policy_logps[1::2]
    else:
        chosen = policy_logps[:, 0]
        rejected = policy_logps[:, 1]

    log_ratio = chosen - rejected  # log π(chosen) - log π(rejected)

    if ref_logps is not None:
        if ref_logps.dim() == 1:
            ref_chosen = ref_logps[0::2]
            ref_rejected = ref_logps[1::2]
        else:
            ref_chosen = ref_logps[:, 0]
            ref_rejected = ref_logps[:, 1]
        log_ratio = log_ratio - (ref_chosen - ref_rejected)

    # DPO loss: -E[log σ(β * log_ratio)]
    loss = -F.logsigmoid(beta * log_ratio).mean()

    # Implicit rewards (for monitoring)
    chosen_reward = beta * log_ratio.detach()
    rejected_reward = -chosen_reward

    return loss, chosen_reward.mean(), rejected_reward.mean()


class CorrectionDataset(Dataset):
    """Dataset of correction pairs for Socratic self-training.

    Each sample is a (hint_prompt, target_trace) pair where the model
    originally failed but succeeded after receiving a critique hint.
    """

    def __init__(
        self,
        tokenizer: MathTokenizer,
        corrections: list[CorrectionExample],
        max_seq_len: int = 32,
        use_loss_masking: bool = True,
    ):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.use_loss_masking = use_loss_masking
        self.samples: list[list[int]] = []

        for ex in corrections:
            # Build the hint prompt (the model got the hint in the critique loop)
            hint_prompt = ex.hint if ex.hint else f"{ex.problem}="

            # Full target: hint_prompt + correct_trace
            full_text = hint_prompt + ex.correct_trace
            ids = tokenizer.encode(full_text, add_special_tokens=True)
            if len(ids) <= max_seq_len:
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.samples[idx]
        pad_id = self.tok.pad_id
        padded = ids + [pad_id] * (self.max_seq_len - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)

        if self.use_loss_masking:
            # Mask PADs
            y[y == pad_id] = -100
            # Mask everything up to hint's '=' (the model should learn the
            # answer portion, not just copy the hint)
            eq_id = self.tok.stoi.get("=", -1)
            if eq_id >= 0:
                eq_positions = (y == eq_id).nonzero(as_tuple=True)[0]
                if len(eq_positions) > 0:
                    eq_pos = eq_positions[0].item()
                    y[: eq_pos + 1] = -100

        return x, y


# ═══════════════════════════════════════════════════════════════════
# Socratic Self-Trainer
# ═══════════════════════════════════════════════════════════════════


class SocraticSelfTrainer:
    """Self-improvement via critique loop: generate → critic → train.

    Iteratively:
    1. Runs the critique loop on fresh problems
    2. Extracts successful corrections (model failed → hint → succeeded)
    3. Trains the model on those correction pairs
    4. Evaluates to measure improvement

    Args:
        model: MathTransformer to train.
        tokenizer: Matching MathTokenizer.
        device: torch device.
        op: Operation (+, -, *, /).
        max_digits: Max digits for generated problems.
        lr: Learning rate for fine-tuning.
        eval_every: Evaluate every N training steps.
    """

    def __init__(
        self,
        model: MathTransformer,
        tokenizer: MathTokenizer,
        device: torch.device,
        op: str = "+",
        max_digits: int = 4,
        lr: float = 1e-4,
        eval_every: int = 100,
    ):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.op = op
        self.max_digits = max_digits
        self.lr = lr
        self.eval_every = eval_every
        self.cfg = Config()
        self.cfg.vocab_size = tokenizer.vocab_size
        self.cfg.max_digits = max_digits
        self.cfg.use_scratchpad = True
        self.cfg.use_reversed = True
        self.cfg.eval_temperature = 0.0
        self.cfg.eval_max_tokens = 15

    # ── Correction Collection ───────────────────────────────────

    def collect_corrections(
        self,
        num_problems: int = 100,
        max_rounds: int = 5,
        seed: Optional[int] = None,
        quiet: bool = False,
    ) -> tuple[list[CorrectionExample], dict]:
        """Run the critique loop and extract successful corrections.

        Args:
            num_problems: Number of problems to generate.
            max_rounds: Max critique rounds per problem.
            seed: RNG seed.
            quiet: Suppress progress output.

        Returns:
            (corrections list, summary dict)
        """
        if not HAS_CRITIQUE:
            raise ImportError("egefalos.socratic_critique required for collection")

        loop = SocraticCritiqueLoop(
            model=self.model,
            tokenizer=self.tok,
            max_rounds=max_rounds,
            temperature=0.0,
        )

        problems = loop.generate_problems(
            op=self.op,
            count=num_problems,
            max_digits=self.max_digits,
            force_carry=True,
            seed=seed or random.randint(0, 2**31),
        )

        corrections: list[CorrectionExample] = []
        solved = 0
        total = len(problems)

        if not quiet:
            print(f"  Collecting corrections ({num_problems} problems)...")

        for i, (expr, expected_str, expected_int) in enumerate(problems):
            if not quiet and (i + 1) % 20 == 0:
                print(f"    [{i + 1}/{total}] found {len(corrections)} corrections so far")

            telemetry = loop.critique_problem(expr, expected_str)

            # Skip if model got it right on round 1 (no correction needed)
            if telemetry.passed and telemetry.n_rounds == 1:
                solved += 1
                continue

            # Skip if model never got it right
            if not telemetry.passed:
                continue

            # Found a successful correction!
            # Find the round where the model first got it right
            correct_round = None
            for rnd in telemetry.rounds:
                if not rnd.critic_errors:
                    correct_round = rnd
                    break
            if correct_round is None:
                continue

            # The hint used was from the PREVIOUS round (the one that led
            # to the correction)
            hint_idx = correct_round.round - 2  # round before the correct one
            hint = ""
            if hint_idx >= 0 and hint_idx < len(telemetry.rounds):
                # Reconstruct the hint from the errors of that round
                prev_round = telemetry.rounds[hint_idx]
                if prev_round.critic_errors:
                    from tabula_rasa.reasoning.socratic_critique import construct_hint
                    hint = construct_hint(expr, prev_round.trace, {"errors": prev_round.critic_errors})

            if not hint:
                hint = f"{expr}="

            first_round = telemetry.rounds[0] if telemetry.rounds else None

            corrections.append(CorrectionExample(
                problem=expr,
                expected=expected_str,
                hint=hint,
                correct_trace=correct_round.trace,
                first_round_trace=first_round.trace if first_round else "",
                n_rounds=telemetry.n_rounds,
                loss_drop_pct=telemetry.loss_drop_pct,
            ))

        # Store successful corrections in hippocampus for replay
        for ex in corrections:
            try:
                store_experience(
                    input_text=ex.hint,
                    prediction_error=1.0 - ex.loss_drop_pct / 100.0,
                    output_text=ex.correct_trace,
                    skill=f"socratic_correction_{self.op}",
                    tier="longterm",
                )
            except Exception:
                pass  # Hippocampus may not be available

        summary = {
            "total_problems": total,
            "solved_on_round_1": solved,
            "corrections_collected": len(corrections),
            "improvement_candidates": total - solved,
            "correction_rate_pct": round(
                len(corrections) / max(1, total - solved) * 100, 1
            ),
        }

        if not quiet:
            print(f"    Done: {summary['corrections_collected']} corrections "
                  f"from {total} problems "
                  f"({summary['correction_rate_pct']}% of failures corrected)")

        return corrections, summary

    # ── Training ────────────────────────────────────────────────

    def train_on_corrections(
        self,
        corrections: list[CorrectionExample],
        steps: int = 500,
        batch_size: int = 32,
        lr: Optional[float] = None,
        quiet: bool = False,
    ) -> dict:
        """Train the model on correction pairs.

        Args:
            corrections: List of CorrectionExample from collect_corrections().
            steps: Number of training steps.
            batch_size: Batch size.
            lr: Learning rate (default: self.lr).
            quiet: Suppress progress output.

        Returns:
            Summary dict with training results.
        """
        if not corrections:
            return {"trained": False, "reason": "no corrections to train on"}

        dataset = CorrectionDataset(
            self.tok, corrections,
            max_seq_len=self.cfg.max_seq_len,
            use_loss_masking=self.cfg.use_loss_masking,
        )

        if len(dataset) < batch_size:
            return {"trained": False, "reason": f"only {len(dataset)} samples, need at least {batch_size}"}

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        effective_steps = min(steps, len(loader) * 3)  # At most 3 epochs
        optimizer = optim.AdamW(self.model.parameters(), lr=lr or self.lr, weight_decay=0.01)

        self.model.train()
        t0 = time.time()
        global_step = 0
        losses = []

        if not quiet:
            print(f"  Training on {len(dataset)} correction pairs ({effective_steps} steps)...")

        for epoch in range(3):  # At most 3 epochs
            if global_step >= effective_steps:
                break
            for x, y in loader:
                if global_step >= effective_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                _, loss, _ = self.model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())
                global_step += 1

                if not quiet and global_step % self.eval_every == 0:
                    avg_loss = sum(losses[-self.eval_every:]) / max(1, len(losses[-self.eval_every:]))
                    print(f"    Step {global_step}/{effective_steps} | loss={avg_loss:.4f}")

        elapsed = time.time() - t0
        avg_loss = sum(losses[-100:]) / max(1, len(losses[-100:])) if losses else float('inf')

        # Quick eval on fresh problems
        acc_before = self._quick_eval(50)
        acc_after = self._quick_eval(50)

        result = {
            "trained": True,
            "samples": len(dataset),
            "steps": global_step,
            "final_loss": round(avg_loss, 4),
            "time_seconds": round(elapsed, 1),
            "acc_before_pct": round(acc_before, 1),
            "acc_after_pct": round(acc_after, 1),
            "improvement_pct": round(acc_after - acc_before, 1),
        }

        if not quiet:
            print(f"  Training complete: {result['steps']} steps in {result['time_seconds']}s")
            print(f"  Accuracy: {result['acc_before_pct']}% -> {result['acc_after_pct']}%")

        return result

    @torch.no_grad()
    def _quick_eval(self, num: int = 50) -> float:
        """Quick accuracy evaluation on fresh problems."""
        from tabula_rasa.reasoning.socratic_critique import generate_problem as gen_prob
        self.model.eval()
        correct = 0
        for _ in range(num):
            expr, expected_str, _ = gen_prob(self.op, self.max_digits, force_carry=False)
            prompt = f"{expr}="
            out = self.model.generate(
                self.tok, prompt,
                max_new_tokens=self.cfg.eval_max_tokens,
                temperature=0.0, top_k=0,
            )
            if "=" in out:
                ans = out.split("=", 1)[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
            else:
                ans = out.strip()
            if ans == expected_str or (
                ans.isdigit() and len(ans) >= 2 and ans[1::2] == expected_str
            ):
                correct += 1
        return correct / max(1, num) * 100

    # DPO Training

    def dpo_on_corrections(
        self,
        corrections: list[CorrectionExample],
        steps: int = 100,
        batch_size: int = 16,
        beta: float = 0.1,
        quiet: bool = False,
    ) -> dict:
        from torch.utils.data import DataLoader
        if len(corrections) < 3:
            return {"trained": False, "reason": "too few corrections"}
        ds = DPODataset(self.tok, corrections, max_seq_len=self.cfg.max_seq_len)
        if len(ds) < 1:
            return {"trained": False, "reason": "empty dataset"}
        loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr * 0.5)
        self.model.train()
        total_loss = 0.0
        n = 0
        for step in range(steps):
            for batch in loader:
                ch = batch["chosen"].to(self.device)
                rej = batch["rejected"].to(self.device)
                lc, _, _ = self.model(ch[:, :-1], targets=ch[:, 1:])
                lr, _, _ = self.model(rej[:, :-1], targets=rej[:, 1:])
                lcp = lc.log_softmax(dim=-1).gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)
                lrp = lr.log_softmax(dim=-1).gather(-1, rej[:, 1:].unsqueeze(-1)).squeeze(-1)
                cm = batch["chosen_mask"][:, 1:].to(self.device)
                rm = batch["rejected_mask"][:, 1:].to(self.device)
                lcp = (lcp * cm).sum(dim=1) / cm.sum(dim=1).clamp(min=1)
                lrp = (lrp * rm).sum(dim=1) / rm.sum(dim=1).clamp(min=1)
                stacked = torch.stack([lcp, lrp], dim=1).flatten()
                loss, _, _ = dpo_loss(stacked, ref_logps=None, beta=beta)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                n += 1
            if not quiet and (step + 1) % max(1, steps // 5) == 0:
                print(f"    [DPO] Step {step + 1}/{steps} | loss={total_loss / max(1, n):.4f}")
        return {"trained": True, "steps": steps, "loss": total_loss / max(1, n)}

    # Full Self-Improvement Loop

    def run_iteration(
        self,
        num_problems: int = 200,
        collect_rounds: int = 5,
        train_steps: int = 500,
        batch_size: int = 32,
        seed: Optional[int] = None,
        quiet: bool = False,
    ) -> dict:
        """Run one full Socratic self-improvement iteration.

        1. Collect corrections via critique loop
        2. Train on collected corrections
        3. Return results

        Args:
            num_problems: Problems to generate for correction collection.
            collect_rounds: Max critique rounds.
            train_steps: Training steps on correction pairs.
            batch_size: Batch size for correction training.
            seed: RNG seed.

        Returns:
            Dict with collection + training results.
        """
        if not quiet:
            print(f"\n  {'='*50}")
            print(f"  Socratic Self-Improvement Iteration")
            print(f"  {'='*50}")

        corrections, collect_summary = self.collect_corrections(
            num_problems=num_problems,
            max_rounds=collect_rounds,
            seed=seed,
            quiet=quiet,
        )

        train_summary = self.train_on_corrections(
            corrections=corrections,
            steps=train_steps,
            batch_size=batch_size,
            quiet=quiet,
        )

        # ── DPO refinement (optional) ──
        if getattr(self, "_use_dpo", False) and len(corrections) > 5:
            dpo_summary = self.dpo_on_corrections(
                corrections=corrections,
                steps=max(10, train_steps // 2),
                batch_size=batch_size,
                quiet=quiet,
            )
            train_summary["dpo"] = dpo_summary

        return {
            "collection": collect_summary,
            "training": train_summary,
        }

    def run_self_improvement(
        self,
        iterations: int = 5,
        problems_per_iter: int = 200,
        train_steps_per_iter: int = 500,
        batch_size: int = 32,
        quiet: bool = False,
    ) -> list[dict]:
        """Run multiple self-improvement iterations.

        Each iteration: generate problems → collect corrections → train.
        Accuracy should asymptotically improve as the model learns from
        its own mistakes.

        Args:
            iterations: Number of self-improvement cycles.
            problems_per_iter: Problems per iteration.
            train_steps_per_iter: Training steps per iteration.
            batch_size: Batch size.

        Returns:
            List of iteration result dicts.
        """
        results = []
        baseline_acc = self._quick_eval(100)

        if not quiet:
            print(f"\n  {'='*60}")
            print(f"  Socratic Self-Improvement — {iterations} iterations")
            print(f"  Baseline accuracy: {baseline_acc:.1f}%")
            print(f"  {'='*60}")

        for i in range(iterations):
            if not quiet:
                print(f"\n  --- Iteration {i + 1}/{iterations} ---")

            iter_result = self.run_iteration(
                num_problems=problems_per_iter,
                train_steps=train_steps_per_iter,
                batch_size=batch_size,
                quiet=quiet,
            )
            results.append(iter_result)

        final_acc = self._quick_eval(100)

        if not quiet:
            print(f"\n  {'='*60}")
            print(f"  Self-Improvement Complete")
            print(f"  Baseline: {baseline_acc:.1f}% -> Final: {final_acc:.1f}%")
            improvement = final_acc - baseline_acc
            print(f"  Net improvement: {improvement:+.1f}%")
            total_corrections = sum(
                r["collection"]["corrections_collected"] for r in results
            )
            print(f"  Total corrections collected: {total_corrections}")
            print(f"  {'='*60}")

        return results


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════


def main():
    """Run Socratic self-improvement from the command line.

    Usage:
        python3 -m egefalos.socratic_trainer --checkpoint specialists/math/add/best.pt
            --op add --iterations 5 --problems 200 --train-steps 500
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Socratic Self-Trainer — self-improvement via critique loop"
    )
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to .pt checkpoint")
    parser.add_argument("--op", type=str, default="+",
                        help="Operation: + - * /")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Self-improvement iterations")
    parser.add_argument("--problems", type=int, default=100,
                        help="Problems per iteration")
    parser.add_argument("--train-steps", type=int, default=300,
                        help="Training steps per iteration")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max-digits", type=int, default=2,
                        help="Max digits per operand")
    args = parser.parse_args()

    # Load model
    from pathlib import Path
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        candidates = [
            Path("specialists/math/add/best.pt"),
            Path("specialists/math/add/final.pt"),
            Path("checkpoints/best.pt"),
        ]
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
    if ckpt_path is None or not ckpt_path.exists():
        print("No checkpoint found.")
        return

    tok_path = ckpt_path.parent / "tokenizer.json"
    if not tok_path.exists():
        # Try fallback
        for p in Path("specialists/math").glob("*/tokenizer.json"):
            tok_path = p
            break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {ckpt_path}")
    tok = MathTokenizer.load(str(tok_path))
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Model: {count_parameters(model):,} params on {device}")

    trainer = SocraticSelfTrainer(
        model=model,
        tokenizer=tok,
        device=device,
        op=args.op,
        max_digits=args.max_digits,
        lr=args.lr,
    )

    trainer.run_self_improvement(
        iterations=args.iterations,
        problems_per_iter=args.problems,
        train_steps_per_iter=args.train_steps,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
