#!/usr/bin/env python3
"""cf_metrics.py — Catastrophic Forgetting Metrics Suite for Tabula Rasa.

Evaluates how much a specialist model forgets after auto-scaling or
sequential training (e.g. Level 0 → Level 2). Compares a baseline
checkpoint (before scaling) against a post-scaling checkpoint.

Computes:
  - Forgetting       = accuracy_before - accuracy_after (positive = forgetting)
  - BWT              = accuracy_after - accuracy_before (Backward Transfer)
                       positive = beneficial transfer, negative = forgetting
  - Retention Rate   = accuracy_after / accuracy_before (1.0 = perfect retention)
  - Per-digit forgetting (ones vs tens vs hundreds place)
  - EWC-protected vs unprotected forgetting comparison

Usage:
    python3 experiments/cf_metrics.py --before specialists/math/add/best.pt \\
                                      --after specialists/math/add/checkpoint.pt
    python3 experiments/cf_metrics.py --before checkpoints/base.pt \\
                                      --after checkpoints/scaled.pt --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Ensure src/ is on the Python path
_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

SEED = 42
OPS = {"add": "+", "sub": "-", "mul": "*"}

# Standardized benchmark sizes
BENCHMARK_SIZES: dict[str, int] = {
    "add": 50,
    "sub": 50,
    "mul": 50,
}

# Difficulty levels for reporting
DIGIT_LEVELS: dict[int, str] = {
    0: "ones",
    1: "tens",
    2: "hundreds",
    3: "thousands",
}


# ═══════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkResult:
    """Accuracy results for one operation on one checkpoint."""

    name: str  # e.g. "add", "sub", "mul"
    correct: int = 0
    total: int = 0
    per_digit_correct: dict[int, int] = field(default_factory=dict)
    per_digit_total: dict[int, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total

    def per_digit_accuracy(self, digit: int) -> float:
        total = self.per_digit_total.get(digit, 0)
        if total == 0:
            return float("nan")
        return 100.0 * self.per_digit_correct.get(digit, 0) / total


@dataclass
class ForgettingMetrics:
    """All forgetting metrics for one operation."""

    operation: str
    accuracy_before: float
    accuracy_after: float
    forgetting: float  # acc_before - acc_after (positive = forgetting)
    bwt: float  # acc_after - acc_before (Backward Transfer)
    retention_rate: float  # acc_after / acc_before
    per_digit_forgetting: dict[int, float] = field(default_factory=dict)
    per_digit_bwt: dict[int, float] = field(default_factory=dict)
    sample_failures_before: list[str] = field(default_factory=list)
    sample_failures_after: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════


def _infer_config_from_state(state: dict[str, Any], tok: MathTokenizer) -> Config:
    """Infer architecture hyperparameters from a checkpoint state_dict.

    Scans the keys to determine d_model, n_layers, n_heads, d_ff, etc.
    Falls back to default 1M preset if inference fails.
    """
    cfg = Config()
    cfg.vocab_size = tok.vocab_size

    shapes: dict[str, tuple[int, ...]] = {}
    for key, tensor in state.items():
        if isinstance(tensor, torch.Tensor):
            shapes[key] = tuple(tensor.shape)

    d_model = None
    if "token_embedding.weight" in shapes:
        d_model = shapes["token_embedding.weight"][1]
    elif "layers.0.attention_norm.weight" in shapes:
        d_model = shapes["layers.0.attention_norm.weight"][0]
    elif "norm.weight" in shapes:
        d_model = shapes["norm.weight"][0]

    if d_model is not None:
        cfg.d_model = d_model

    n_layers = 0
    for key in shapes:
        if key.startswith("layers.") and ".attention.wq.weight" in key:
            parts = key.split(".")
            try:
                idx = int(parts[1])
                n_layers = max(n_layers, idx + 1)
            except (ValueError, IndexError):
                pass
    if n_layers > 0:
        cfg.n_layers = n_layers
    # Infer n_heads from RoPE inv_freq or wq weight shape
    # inv_freq has shape [head_dim/2] where head_dim = d_model / n_heads
    rope_key = None
    for k in shapes:
        if "rope.inv_freq" in k:
            rope_key = k
            break
    if rope_key is not None:
        inv_freq_len = shapes[rope_key][0]
        head_dim = inv_freq_len * 2
        if d_model and d_model % head_dim == 0:
            cfg.n_heads = d_model // head_dim
    elif "layers.0.attention.wq.weight" in shapes:
        wq_shape = shapes["layers.0.attention.wq.weight"]
        if d_model and wq_shape[0] % d_model == 0:
            n_heads = wq_shape[0] // d_model
            cfg.n_heads = n_heads
    if "layers.0.feed_forward.w1.weight" in shapes:
        cfg.d_ff = shapes["layers.0.feed_forward.w1.weight"][0]
    elif "layers.0.feed_forward.w2.weight" in shapes:
        cfg.d_ff = shapes["layers.0.feed_forward.w2.weight"][1]

    if any("rope.inv_freq" in k for k in shapes):
        cfg.pos_encoding = "rope"
    elif any("pos_embed.pe.weight" in k for k in shapes):
        cfg.pos_encoding = "learned"

    if any("feed_forward.router.weight" in k for k in shapes):
        cfg.use_moe = True

    if any(k.startswith("value_head.") for k in shapes):
        cfg.use_value_head = True

    return cfg


def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[MathTransformer, MathTokenizer, Config]:
    """Load a trained model and tokenizer from a checkpoint.

    Auto-detects architecture from the checkpoint state_dict.

    Args:
        checkpoint_path: Path to the .pt checkpoint file. The corresponding
            tokenizer JSON is expected at ``{checkpoint_dir}/tokenizer.json``.
        device: Device to load the model onto ('cpu' or 'cuda').

    Returns:
        Tuple ``(model, tokenizer, config)`` with the model in eval mode.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        print(f"  [ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    tok_path = ckpt.parent / "tokenizer.json"
    if not tok_path.exists():
        print(f"  [ERROR] Tokenizer not found at {tok_path}")
        sys.exit(1)

    print(f"  Loading tokenizer from {tok_path}")
    tokenizer_data = Path(tok_path).read_text(encoding="utf-8")
    tok = MathTokenizer.__new__(MathTokenizer)
    data = json.loads(tokenizer_data)
    tok.stoi = data["stoi"]
    tok.itos = {int(k): v for k, v in data["itos"].items()}
    tok.vocab_size = len(tok.stoi)
    tok.pad_id = tok.stoi["<PAD>"]
    tok.bos_id = tok.stoi["<BOS>"]
    tok.eos_id = tok.stoi["<EOS>"]
    tok.unk_id = tok.stoi["<UNK>"]
    tok.step_id = tok.stoi.get("<STEP>", tok.eos_id)
    tok.end_id = tok.stoi.get("<END>", tok.eos_id)
    tok._tokens_sorted = sorted(tok.stoi.keys(), key=len, reverse=True)

    print(f"  Loading checkpoint from {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=True)

    if "model_state_dict" in state:
        model_state = state["model_state_dict"]
    elif "state_dict" in state:
        model_state = state["state_dict"]
    else:
        model_state = state

    cfg = _infer_config_from_state(model_state, tok)
    cfg.vocab_size = tok.vocab_size

    print(f"  Inferred architecture:")
    print(f"    d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"    pos_encoding={cfg.pos_encoding}, max_seq_len={cfg.max_seq_len}")
    print(f"    vocab_size={cfg.vocab_size}")

    model = MathTransformer(cfg)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    model.to(device)

    print(f"  Model parameters: {count_parameters(model):,}")
    print()

    return model, tok, cfg


# ═══════════════════════════════════════════════════════════════════
# Problem Generation
# ═══════════════════════════════════════════════════════════════════


def gen_problems(
    op: str,
    num: int = 50,
    min_digits: int = 1,
    max_digits: int = 2,
) -> list[tuple[str, int, int, int]]:
    """Generate math problems for the given operation.

    Returns list of (expr_string, a, b, expected_answer).
    """
    rng = random.Random(SEED)
    problems: list[tuple[str, int, int, int]] = []
    attempts = 0

    while len(problems) < num and attempts < num * 20:
        attempts += 1
        a_digits = rng.randint(min_digits, max_digits)
        b_digits = rng.randint(min_digits, max_digits)
        a = rng.randint(10 ** (a_digits - 1), 10**a_digits - 1)
        b = rng.randint(10 ** (b_digits - 1), 10**b_digits - 1)

        if op == "add":
            ans = a + b
        elif op == "sub":
            if a < b:
                a, b = b, a
            ans = a - b
        elif op == "mul":
            # Keep results manageable — limit digit count
            if a_digits + b_digits > 4:
                b = rng.randint(1, 99) if a > 99 else rng.randint(1, 9)
            ans = a * b
        else:
            continue

        expr = f"{a}{OPS[op]}{b}"
        problems.append((expr, a, b, ans))

    return problems


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════


def _extract_answer(output: str, expr_prompt: str) -> str | None:
    """Extract the predicted answer from the model's generated output."""
    text = output.strip()

    if text.startswith(expr_prompt):
        text = text[len(expr_prompt):].strip()

    # Clean special tokens
    text = text.replace("<EOS>", "").replace("<PAD>", "").replace("<BOS>", "").strip()

    if "<END>" in text:
        # CoT/scratchpad format
        return text.split("<END>")[-1].strip()

    if "=" in text:
        after_eq = text.split("=")[-1].strip()
        # Try to extract digits
        digits = "".join(c for c in after_eq if c.isdigit())
        if digits:
            return digits
        return None

    # No '=' — try bare digits
    digits = "".join(c for c in text if c.isdigit())
    return digits if digits else None


def _compute_per_digit(expected: int, predicted: int) -> dict[int, bool]:
    """Compute per-digit correctness for expected vs predicted answer.

    Returns dict mapping digit_position (0=ones, 1=tens, ...) to bool.
    """
    exp_str = str(expected)
    pred_str = str(predicted)

    max_len = max(len(exp_str), len(pred_str))
    exp_padded = exp_str.zfill(max_len)
    pred_padded = pred_str.zfill(max_len)

    results: dict[int, bool] = {}
    for i in range(max_len):
        pos = max_len - 1 - i  # 0=ones, 1=tens, ...
        results[pos] = exp_padded[i] == pred_padded[i]
    return results


@torch.no_grad()
def evaluate_operation(
    model: MathTransformer,
    tok: MathTokenizer,
    op: str,
    num: int = 50,
    max_new_tokens: int = 30,
    min_digits: int = 1,
    max_digits: int = 2,
) -> BenchmarkResult:
    """Evaluate a model on one operation.

    Args:
        model: The transformer model.
        tok: The tokenizer.
        op: Operation name ('add', 'sub', 'mul').
        num: Number of problems to generate.
        max_new_tokens: Max tokens for generation.
        min_digits: Minimum operand digits.
        max_digits: Maximum operand digits.

    Returns:
        BenchmarkResult with accuracy and per-digit breakdown.
    """
    result = BenchmarkResult(name=op)
    problems = gen_problems(op, num, min_digits, max_digits)
    result.total = len(problems)

    for expr, a, b, expected in problems:
        prompt = f"{expr}="
        generated = model.generate(
            tok, prompt, max_new_tokens=max_new_tokens,
            temperature=0.0, top_k=5,
        )

        predicted_str = _extract_answer(generated, prompt)
        expected_str = str(expected)

        if predicted_str is not None and predicted_str == expected_str:
            result.correct += 1
        else:
            display = generated[:60] + "..." if len(generated) > 60 else generated
            result.failures.append(
                f"  FAIL: {expr}={expected} | pred='{predicted_str}' "
                f"| raw={repr(display)}"
            )

        # Per-digit breakdown
        if predicted_str is not None and predicted_str.lstrip("-").isdigit():
            try:
                predicted_int = abs(int(predicted_str))
                digit_info = _compute_per_digit(expected, predicted_int)
                for pos, correct_flag in digit_info.items():
                    result.per_digit_total[pos] = result.per_digit_total.get(pos, 0) + 1
                    if correct_flag:
                        result.per_digit_correct[pos] = result.per_digit_correct.get(pos, 0) + 1
            except ValueError:
                pass
        else:
            for pos in range(len(expected_str)):
                result.per_digit_total[pos] = result.per_digit_total.get(pos, 0) + 1

    return result


# ═══════════════════════════════════════════════════════════════════
# Forgetting Metrics Computation
# ═══════════════════════════════════════════════════════════════════


def compute_forgetting_metrics(
    before_result: BenchmarkResult,
    after_result: BenchmarkResult,
) -> ForgettingMetrics:
    """Compute all forgetting metrics comparing before and after results.

    Args:
        before_result: Benchmark result from the baseline checkpoint.
        after_result: Benchmark result from the post-scaling checkpoint.

    Returns:
        ForgettingMetrics with all computed values.
    """
    acc_before = before_result.accuracy
    acc_after = after_result.accuracy

    metrics = ForgettingMetrics(
        operation=before_result.name,
        accuracy_before=acc_before,
        accuracy_after=acc_after,
        forgetting=acc_before - acc_after,
        bwt=acc_after - acc_before,
        retention_rate=acc_after / acc_before if acc_before > 0 else float("nan"),
        sample_failures_before=before_result.failures[:3],
        sample_failures_after=after_result.failures[:3],
    )

    # Per-digit forgetting
    all_positions = sorted(
        set(before_result.per_digit_total.keys()) | set(after_result.per_digit_total.keys())
    )
    for pos in all_positions:
        acc_b = before_result.per_digit_accuracy(pos)
        acc_a = after_result.per_digit_accuracy(pos)
        if not (acc_b != acc_b or acc_a != acc_a):  # Skip nan
            metrics.per_digit_forgetting[pos] = acc_b - acc_a
            metrics.per_digit_bwt[pos] = acc_a - acc_b

    return metrics


# ═══════════════════════════════════════════════════════════════════
# EWC Protection Test
# ═══════════════════════════════════════════════════════════════════


def test_ewc_protection(
    before_model: MathTransformer,
    after_model: MathTransformer,
    tok: MathTokenizer,
    op: str,
    num: int = 25,
) -> dict[str, float]:
    """Compare forgetting with and without EWC protection.

    Uses the model's parameter divergence as a proxy for how much the
    weights changed (EWC-protected parameters should change less).

    Returns:
        Dict with parameter-level divergence metrics.
    """
    metrics: dict[str, float] = {}

    state_before = before_model.state_dict()
    state_after = after_model.state_dict()

    # Compute total parameter divergence
    total_params = 0
    total_l2 = 0.0
    layer_divergences: dict[str, float] = {}

    for key in state_before:
        if key in state_after and "weight" in key:
            diff = state_after[key] - state_before[key]
            l2 = diff.norm().item()
            n = diff.numel()
            total_l2 += l2
            total_params += n
            # Store per-layer divergence for key layers
            layer_name = ".".join(key.split(".")[:-1])  # Strip 'weight'
            if layer_name not in layer_divergences:
                layer_divergences[layer_name] = l2

    metrics["total_param_divergence_l2"] = total_l2
    metrics["avg_param_divergence"] = total_l2 / max(1, total_params)

    # Attention layers typically change more during new learning
    attn_div = sum(v for k, v in layer_divergences.items() if "attention" in k)
    ffn_div = sum(v for k, v in layer_divergences.items() if "feed_forward" in k)
    metrics["attention_divergence_l2"] = attn_div
    metrics["ffn_divergence_l2"] = ffn_div

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Results Printing
# ═══════════════════════════════════════════════════════════════════


def _print_separator(char: str = "=", width: int = 72) -> None:
    print(char * width)


def _print_forgetting_table(metrics_list: list[ForgettingMetrics]) -> None:
    """Print a formatted table of forgetting metrics."""
    _print_separator()
    print("  CATASTROPHIC FORGETTING METRICS")
    _print_separator("-")
    print()

    for metrics in metrics_list:
        print(f"  Operation: {metrics.operation.upper()}")
        print(f"  {'-' * 60}")
        print(f"    {'Accuracy (before):':<30} {metrics.accuracy_before:>6.1f}%")
        print(f"    {'Accuracy (after):':<30} {metrics.accuracy_after:>6.1f}%")
        print(f"    {'Forgetting (- = BWT):':<30} {metrics.forgetting:>+6.1f}%")
        print(f"    {'BWT (Backward Transfer):':<30} {metrics.bwt:>+6.1f}%")
        print(f"    {'Retention Rate:':<30} {metrics.retention_rate:>6.3f}  "
              f"{'[PERFECT]' if abs(metrics.retention_rate - 1.0) < 0.01 else ''}")
        print()

        if metrics.per_digit_forgetting:
            print(f"    Per-digit forgetting:")
            for pos in sorted(metrics.per_digit_forgetting.keys()):
                place = DIGIT_LEVELS.get(pos, f"pos={pos}")
                f_val = metrics.per_digit_forgetting[pos]
                print(f"      {place:>10}: {f_val:>+6.1f}%  "
                      f"(BWT: {metrics.per_digit_bwt.get(pos, 0.0):>+6.1f}%)")
            print()

        if metrics.sample_failures_before:
            print(f"    Sample failures (before) — {len(metrics.sample_failures_before)} shown:")
            for f in metrics.sample_failures_before:
                print(f"      {f}")
        if metrics.sample_failures_after:
            print(f"    Sample failures (after) — {len(metrics.sample_failures_after)} shown:")
            for f in metrics.sample_failures_after:
                print(f"      {f}")
        print()

    # Summary table
    _print_separator("-")
    print("  SUMMARY TABLE")
    _print_separator("-")
    header = (
        f"  {'Operation':<12} {'Before':<10} {'After':<10} "
        f"{'Forget':<10} {'BWT':<10} {'Retention':<10}"
    )
    print(header)
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for metrics in metrics_list:
        print(
            f"  {metrics.operation:<12} {metrics.accuracy_before:>5.1f}%  "
            f"{metrics.accuracy_after:>5.1f}%  "
            f"{metrics.forgetting:>+5.1f}%  "
            f"{metrics.bwt:>+5.1f}%  "
            f"{metrics.retention_rate:>7.3f}"
        )

    _print_separator()


def _print_ewc_protection(results: dict[str, dict[str, float]]) -> None:
    """Print EWC protection test results."""
    print()
    _print_separator()
    print("  EWC PROTECTION ANALYSIS")
    _print_separator("-")
    print()

    for ckpt_name, metrics in results.items():
        print(f"  Checkpoint pair: {ckpt_name}")
        print(f"    Total parameter divergence (L2):  {metrics.get('total_param_divergence_l2', 0):>8.2f}")
        print(f"    Avg parameter divergence:         {metrics.get('avg_param_divergence', 0):>8.6f}")
        print(f"    Attention layer divergence (L2):  {metrics.get('attention_divergence_l2', 0):>8.2f}")
        print(f"    FFN layer divergence (L2):        {metrics.get('ffn_divergence_l2', 0):>8.2f}")
        print()


# ═══════════════════════════════════════════════════════════════════
# JSON Export
# ═══════════════════════════════════════════════════════════════════


def export_json(metrics_list: list[ForgettingMetrics], path: str | Path) -> None:
    """Export forgetting metrics to a JSON file.

    Args:
        metrics_list: List of ForgettingMetrics to export.
        path: Output JSON file path.
    """
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": [],
    }

    for m in metrics_list:
        entry = {
            "operation": m.operation,
            "accuracy_before": m.accuracy_before,
            "accuracy_after": m.accuracy_after,
            "forgetting": m.forgetting,
            "bwt": m.bwt,
            "retention_rate": m.retention_rate,
            "per_digit_forgetting": {str(k): v for k, v in m.per_digit_forgetting.items()},
            "per_digit_bwt": {str(k): v for k, v in m.per_digit_bwt.items()},
        }
        data["metrics"].append(entry)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results exported to {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catastrophic Forgetting Metrics Suite for Tabula Rasa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 experiments/cf_metrics.py --before specialists/math/add/best.pt --after specialists/math/add/checkpoint.pt
  python3 experiments/cf_metrics.py --before checkpoints/l0.pt --after checkpoints/l2.pt --all --ewc
  python3 experiments/cf_metrics.py --before specialists/math/add/best.pt --after specialists/math/general/best.pt
        """,
    )
    parser.add_argument(
        "--before", "-b",
        type=str,
        required=True,
        help="Baseline checkpoint path (before scaling)",
    )
    parser.add_argument(
        "--after", "-a",
        type=str,
        required=True,
        help="Post-scaling checkpoint path (after scaling)",
    )
    parser.add_argument(
        "--all", "-A",
        action="store_true",
        help="Test all operations (add, sub, mul)",
    )
    parser.add_argument(
        "--op",
        type=str,
        default="add",
        choices=["add", "sub", "mul"],
        help="Operation to test (default: add, use --all for all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of problems per operation (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum new tokens to generate (default: 30)",
    )
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="Also run EWC protection analysis (parameter divergence)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=1,
        help="Minimum operand digits (default: 1)",
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=2,
        help="Maximum operand digits (default: 2)",
    )

    args = parser.parse_args()

    # ════════════════════════════════════════════
    # Header
    # ════════════════════════════════════════════
    _print_separator()
    print("  Tabula Rasa — Catastrophic Forgetting Metrics")
    _print_separator("-")
    print(f"  Before: {args.before}")
    print(f"  After:  {args.after}")
    print(f"  Device: {args.device}")
    print(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print_separator()

    # ════════════════════════════════════════════
    # Load models
    # ════════════════════════════════════════════
    print()
    print("  Loading BEFORE checkpoint...")
    before_model, before_tok, before_cfg = load_model(args.before, device=args.device)

    print()
    print("  Loading AFTER checkpoint...")
    after_model, after_tok, after_cfg = load_model(args.after, device=args.device)

    # Verify tokenizers match
    if before_tok.vocab_size != after_tok.vocab_size:
        print("  [WARNING] Tokenizer vocab sizes differ! "
              f"before={before_tok.vocab_size}, after={after_tok.vocab_size}")

    # ════════════════════════════════════════════
    # Determine which operations to test
    # ════════════════════════════════════════════
    ops_to_test: list[str] = ["add", "sub", "mul"] if args.all else [args.op]

    # ════════════════════════════════════════════
    # Run evaluations
    # ════════════════════════════════════════════
    print()
    print("  Running evaluations (this may take a while)...")
    print()

    metrics_list: list[ForgettingMetrics] = []

    for op in ops_to_test:
        print(f"  [{op.upper()}] Evaluating before model...")
        before_result = evaluate_operation(
            before_model, before_tok, op,
            num=args.num,
            max_new_tokens=args.max_tokens,
            min_digits=args.min_digits,
            max_digits=args.max_digits,
        )
        print(f"    Accuracy: {before_result.accuracy:.1f}% ({before_result.correct}/{before_result.total})")

        print(f"  [{op.upper()}] Evaluating after model...")
        after_result = evaluate_operation(
            after_model, after_tok, op,
            num=args.num,
            max_new_tokens=args.max_tokens,
            min_digits=args.min_digits,
            max_digits=args.max_digits,
        )
        print(f"    Accuracy: {after_result.accuracy:.1f}% ({after_result.correct}/{after_result.total})")

        metrics = compute_forgetting_metrics(before_result, after_result)
        metrics_list.append(metrics)
        print()

    # ════════════════════════════════════════════
    # Print forgetting metrics
    # ════════════════════════════════════════════
    _print_forgetting_table(metrics_list)

    # ════════════════════════════════════════════
    # EWC protection test
    # ════════════════════════════════════════════
    if args.ewc:
        print("  Running EWC protection analysis...")
        ewc_results: dict[str, dict[str, float]] = {}
        for op in ops_to_test:
            ewc_results[op] = test_ewc_protection(
                before_model, after_model, before_tok, op, num=args.num,
            )
        _print_ewc_protection(ewc_results)

    # ════════════════════════════════════════════
    # Export results
    # ════════════════════════════════════════════
    if args.export:
        export_json(metrics_list, args.export)
        print()

    # ════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════
    _print_separator()
    avg_forgetting = sum(m.forgetting for m in metrics_list) / len(metrics_list)
    print(f"  Average forgetting across {len(metrics_list)} operation(s): {avg_forgetting:+.1f}%")
    if avg_forgetting > 0:
        print(f"  [⚠] Forgetting detected — model lost {avg_forgetting:.1f}% accuracy on average.")
    elif avg_forgetting < 0:
        print(f"  [✓] Positive Backward Transfer — model improved by {-avg_forgetting:.1f}% on average.")
    else:
        print(f"  [=] No change detected.")
    _print_separator()


if __name__ == "__main__":
    main()
