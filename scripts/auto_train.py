"""Autonomous training loop — tests all operations, trains the weakest.

Usage:
    python3 auto_train.py                    # Train until all ops reach 50%
    python3 auto_train.py --target 70        # Train until all ops reach 70%
    python3 auto_train.py --budget 10000     # Max 10K total training steps
    python3 auto_train.py --rounds 3         # Re-evaluate up to 3 times
    python3 auto_train.py --ops add sub      # Only train addition and subtraction

How it works:
    1. Load each specialist checkpoint (if exists)
    2. Quick eval (50 problems) to measure current accuracy
    3. Find the weakest operation below target
    4. Train it with curriculum learning
    5. Repeat until all ops pass or budget exhausted
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import torch
from egefalos.hard_negative_mining import (
    evaluate_find_failures,
    generate_hard_negatives,
)

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer
from train_specialist import (
    _INTERRUPTED,
    OP_NAMES,
    OPS,
    _signal_handler,
    evaluate,
    train_specialist,
)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _load_specialist(op: str, tokenizer):
    """Load a specialist's best checkpoint. Returns (model, cfg, state) or None."""
    op_dir = Path("specialists") / "math" / op
    for ckpt_name in ["best.pt", "final.pt", "checkpoint.pt"]:
        ckpt_path = op_dir / ckpt_name
        if ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                sd = state.get("model_state_dict", None)
                if sd is None:
                    continue
                # Auto-detect architecture from checkpoint
                ckpt_d_model = sd["token_embedding.weight"].shape[1]
                ckpt_n_layers = len(
                    [
                        k
                        for k in sd
                        if k.startswith("layers.") and k.endswith(".attention.wq.weight")
                    ]
                )

                cfg = Config()
                cfg.vocab_size = tokenizer.vocab_size
                cfg.d_model = ckpt_d_model
                cfg.n_layers = ckpt_n_layers
                if ckpt_n_layers > 0:
                    cfg.d_ff = sd["layers.0.feed_forward.w1.weight"].shape[0]
                cfg.n_heads = {128: 4, 64: 2, 96: 4, 256: 8}.get(
                    ckpt_d_model, max(1, ckpt_d_model // 32)
                )

                model = MathTransformer(cfg)
                model.load_state_dict(sd, strict=False)
                return model, cfg, state
            except Exception as e:
                print(f"  [!] Failed to load {op}/{ckpt_name}: {e}")
    return None, None, None


def _checkpoint_matches_config(op: str, cfg: Config) -> bool:
    """Check if existing checkpoint architecture matches current Config."""
    op_dir = Path("specialists") / "math" / op
    for ckpt_name in ["checkpoint.pt", "best.pt"]:
        ckpt_path = op_dir / ckpt_name
        if ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                sd = state.get("model_state_dict", {})
                ckpt_d = sd.get("token_embedding.weight", torch.zeros(1, 1)).shape[1]
                ckpt_L = len(
                    [
                        k
                        for k in sd
                        if k.startswith("layers.") and k.endswith(".attention.wq.weight")
                    ]
                )
                return ckpt_d == cfg.d_model and ckpt_L == cfg.n_layers
            except Exception:
                pass
    return True  # No checkpoint = matches anything


def checkpoint_exists(op: str) -> bool:
    """Check if any checkpoint exists for an operation."""
    op_dir = Path("specialists") / "math" / op
    return any((op_dir / name).exists() for name in ["checkpoint.pt", "best.pt", "final.pt"])


def quick_eval_all(ops, tokenizer, num_problems=30):
    """Quick eval on all operations. Returns {op: accuracy%}.

    Operations with no checkpoint get 0%.
    """
    results = {}
    tok = tokenizer

    for op in ops:
        model, cfg, state = _load_specialist(op, tok)
        if model is None:
            results[op] = 0.0
            print(f"  {OP_NAMES[op]:>14}: NO CHECKPOINT (0%)")
            continue

        model.eval()
        # Use stored accuracy if available (avoids expensive eval)
        stored_acc = state.get("acc", None)
        if stored_acc is not None and stored_acc > 0:
            results[op] = stored_acc
            print(
                f"  {OP_NAMES[op]:>14}: {stored_acc:.1f}% (cached) | d={cfg.d_model} L={cfg.n_layers}"
            )
        else:
            # Run quick eval
            tok.max_seq_len = cfg.max_seq_len
            acc = evaluate(model, tok, cfg, op, num=num_problems)
            results[op] = acc
            print(
                f"  {OP_NAMES[op]:>14}: {acc:.1f}% (evaluated live) | d={cfg.d_model} L={cfg.n_layers}"
            )

        del model  # Free memory

    return results


def auto_train(ops=None, target=50.0, budget=50000, rounds=5, deep=False, test_hard=False):
    """Autonomous training loop.

    Args:
        ops: List of operations to train (default: all 4)
        target: Target accuracy % for each operation
        budget: Maximum total training steps across all operations
        rounds: Max re-evaluation rounds
        deep: Use deeper model architecture
        test_hard: Test on max_digits+1
    """

    if ops is None:
        ops = list(OPS.keys())

    tok = MathTokenizer()
    total_steps_used = 0
    history = []  # List of {round, accuracies, trained_op, steps_used}

    print(f"\n{"="*60}")
    print(f"  AUTONOMOUS TRAINING — Target: {target}% each")
    print(f'  Operations: {", ".join(OPS[o] for o in ops)}')
    print(f"  Budget: {budget:,} steps | Rounds: {rounds}")
    print(f"  Deep model: {deep} | Hard eval: {test_hard}")
    print(f"{"="*60}\n")

    for round_num in range(1, rounds + 1):
        if _INTERRUPTED:
            print("\n  Interrupted by user. Stopping.")
            break

        if total_steps_used >= budget:
            print(f"\n  Budget exhausted ({total_steps_used:,}/{budget:,} steps). Stopping.")
            break

        print(f"\n--- Round {round_num}/{rounds} ---")

        # Evaluate all operations
        accuracies = quick_eval_all(ops, tok)

        # Find weakest operation below target
        weak_ops = {op: acc for op, acc in accuracies.items() if acc < target}

        if not weak_ops:
            print("\n  ALL OPERATIONS AT OR ABOVE TARGET!")
            break

        # Sort by ascending accuracy — train weakest first
        sorted_weak = sorted(weak_ops.items(), key=lambda x: x[1])
        weakest_op, weakest_acc = sorted_weak[0]

        print(f"\n  Weakest: {OP_NAMES[weakest_op]} at {weakest_acc:.1f}%")
        print(f"  Gap to target: {target - weakest_acc:.1f}%")

        # ── Hard-Negative Mining ─────────────────────────────────
        # Find specific failing problems and generate edge-case variants
        hard_negatives = []
        try:
            model, cfg, _ = _load_specialist(weakest_op, tok)
            if model is not None:
                failures = evaluate_find_failures(model, tok, cfg, weakest_op, num=50)
                if failures:
                    hard_negatives = generate_hard_negatives(failures, max_variants=3)
                    if hard_negatives:
                        print(f"  Hard negatives: {len(failures)} failures → {len(hard_negatives)} edge-case variants")
                        # Display a sample
                        for hn in hard_negatives[:3]:
                            print(f"    Variant: {hn}")
        except Exception as e:
            print(f"  [HardNegative] Skipped: {e}")

        # Decide training budget for this operation
        remaining_budget = budget - total_steps_used
        # Use adaptive steps: more if far from target, less if close
        gap = target - weakest_acc
        if gap > 40:
            steps_for_op = min(15000, remaining_budget)
        elif gap > 20:
            steps_for_op = min(10000, remaining_budget)
        elif gap > 10:
            steps_for_op = min(5000, remaining_budget)
        else:
            steps_for_op = min(3000, remaining_budget)

        print(f"  Training {weakest_op} for {steps_for_op:,} steps...")

        # Check if checkpoint exists for resume
        # Decide whether to resume
        has_checkpoint = checkpoint_exists(weakest_op)
        if has_checkpoint:
            # Check if checkpoint architecture matches current config
            matches = _checkpoint_matches_config(weakest_op, Config())
            if not matches:
                print("  [!] Checkpoint architecture mismatch — starting fresh")
                resume = False
            else:
                resume = weakest_acc > 5  # Only resume if it learned something
        else:
            resume = False

        # Train
        train_start = time.time()
        train_specialist(
            weakest_op,
            steps=steps_for_op,
            resume=resume,
            test_hard=test_hard,
            deep=deep,
        )
        train_time = time.time() - train_start

        total_steps_used += steps_for_op

        # Record history
        history.append(
            {
                "round": round_num,
                "trained_op": weakest_op,
                "before_acc": weakest_acc,
                "steps_used": steps_for_op,
                "total_steps": total_steps_used,
                "time_seconds": round(train_time),
                "all_accuracies": accuracies,
            }
        )

        # Print progress
        print(f"\n  Total steps used: {total_steps_used:,}/{budget:,}")
        print(f"  Remaining budget: {remaining_budget - steps_for_op:,}")

    # ── Final evaluation ──
    print(f"\n{"="*60}")
    print("  FINAL ACCURACIES")
    print(f"{"="*60}\n")

    final_acc = quick_eval_all(ops, tok)

    passed = sum(1 for acc in final_acc.values() if acc >= target)
    total = len(final_acc)

    print(f"\n  Result: {passed}/{total} operations at or above {target}% target")
    print(f"  Total training steps: {total_steps_used:,}")

    # Print training history
    if history:
        print("\n  Training History:")
        for h in history:
            op_name = OP_NAMES[h["trained_op"]]
            print(
                f'    Round {h["round"]}: {op_name} ({h["before_acc"]:.0f}% -> trained {h["steps_used"]:,} steps in {h["time_seconds"]}s)'
            )

    # Save history to file
    try:
        hist_path = Path("auto_train_history.json")
        hist_data = {
            "target": target,
            "budget": budget,
            "deep": deep,
            "final_accuracies": {op: round(acc, 1) for op, acc in final_acc.items()},
            "passed": passed,
            "total": total,
            "history": history,
        }
        hist_path.write_text(json.dumps(hist_data, indent=2))
        print(f"\n  History saved to {hist_path}")
    except Exception:
        pass

    return final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous training — tests and trains the weakest operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 auto_train.py                     # Train all ops to 50%
  python3 auto_train.py --target 70         # Higher target
  python3 auto_train.py --deep              # Use deeper model
  python3 auto_train.py --ops add sub       # Only add + sub
  python3 auto_train.py --budget 5000       # Quick test budget
        """,
    )
    parser.add_argument(
        "--target",
        type=float,
        default=50.0,
        help="Target accuracy %% for each operation (default: 50)",
    )
    parser.add_argument(
        "--budget", type=int, default=50000, help="Max total training steps (default: 50000)"
    )
    parser.add_argument(
        "--rounds", type=int, default=5, help="Max re-evaluation rounds (default: 5)"
    )
    parser.add_argument("--deep", action="store_true", help="Use deeper model (d=64 L=8 ff=256)")
    parser.add_argument("--test-hard", action="store_true", help="Also test on max_digits+1")
    parser.add_argument("--ops", nargs="+", default=None, help="Operations to train (default: all)")

    args = parser.parse_args()

    ops = args.ops if args.ops else list(OPS.keys())

    # Validate ops
    for op in ops:
        if op not in OPS:
            print(f"Unknown operation: {op}. Use: add, sub, mul, div")
            sys.exit(1)

    auto_train(
        ops=ops,
        target=args.target,
        budget=args.budget,
        rounds=args.rounds,
        deep=args.deep,
        test_hard=args.test_hard,
    )
