"""ProofGrid-compatible benchmark export — formats Tabula-Rasa results for external math reasoning evaluation.

ProofGrid is "a live benchmark for evaluating LLMs on research-level mathematical problems"
(recommended by the PDF). This module formats our benchmark results into ProofGrid-compatible
JSON and also exports a standalone problem set for external math reasoning evaluation.

Usage:
    python3 -m tabula_rasa.benchmark_export checkpoints/best.pt
    python3 -m tabula_rasa.benchmark_export checkpoints/best.pt --format proofgrid
    python3 -m tabula_rasa.benchmark_export checkpoints/best.pt --format mlebench
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


def generate_proofgrid_problems(num_problems: int = 100, max_digits: int = 4) -> list[dict]:
    """Generate problems in ProofGrid-compatible format.

    ProofGrid expects: {
        "problem_id": str,
        "problem": str,       # The math expression
        "expected": str,      # The correct answer
        "domain": str,        # e.g. "arithmetic", "algebra"
        "difficulty": int,    # 1-5
        "type": str,          # "free_response" or "multiple_choice"
    }

    Returns:
        List of problem dicts.
    """
    from tabula_rasa.dataset import generate_problem

    problems = []
    for i in range(num_problems):
        expr, ans = generate_problem(1, max_digits)

        # Determine difficulty based on digits and operation
        import re
        m = re.match(r"(\d+)([+\-*/])(\d+)", expr)
        if m:
            a_digits = len(m.group(1))
            b_digits = len(m.group(2))
            op = m.group(3)
            difficulty = min(5, 1 + (a_digits + b_digits) // 2)
            if op in ("*", "/"):
                difficulty = min(5, difficulty + 1)

            problems.append({
                "problem_id": f"tr_{i:04d}",
                "problem": f"{expr}=",
                "expected": ans,
                "domain": "arithmetic",
                "difficulty": difficulty,
                "type": "free_response",
                "source": "tabula_rasa",
            })

    return problems


def evaluate_proofgrid(
    model: Any, tokenizer: Any, problems: list[dict], verbose: bool = False
) -> list[dict]:
    """Evaluate model on ProofGrid-formatted problems.

    Args:
        model: MathTransformer.
        tokenizer: MathTokenizer.
        problems: List of problem dicts (from generate_proofgrid_problems).
        verbose: Print per-problem results.

    Returns:
        List of result dicts with model predictions.
    """
    model.eval()
    results = []
    correct = 0

    for p in problems:
        expr = p["problem"].rstrip("=")
        expected = p["expected"]

        prompt = f"{expr}="
        generated = model.generate(tokenizer, prompt, max_new_tokens=12, temperature=0.0, top_k=1)

        if "=" in generated:
            pred = generated.split("=")[-1].strip()
            pred = "".join(c for c in pred if c.isdigit() or c == "-")
        else:
            pred = ""

        is_correct = pred == expected
        if is_correct:
            correct += 1

        result = {
            "problem_id": p["problem_id"],
            "problem": p["problem"],
            "expected": expected,
            "predicted": pred,
            "correct": is_correct,
            "difficulty": p["difficulty"],
            "domain": p["domain"],
        }
        results.append(result)

        if verbose and not is_correct:
            print(f"  FAIL [{p['problem_id']}] {p['problem']} → pred={pred} expected={expected}")

    return results


def export_mlebench(results: list[dict], path: str | Path) -> None:
    """Export results in MLE-bench compatible format.

    MLE-bench expects: {"task_id": str, "pass": bool, "score": float}
    """
    mle_results = []
    for r in results:
        mle_results.append({
            "task_id": r["problem_id"],
            "pass": r["correct"],
            "score": 1.0 if r["correct"] else 0.0,
            "domain": r.get("domain", "arithmetic"),
        })

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["correct"]),
        "accuracy": round(sum(1 for r in results if r["correct"]) / max(len(results), 1) * 100, 1),
        "by_difficulty": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Per-difficulty breakdown
    for d in sorted(set(r["difficulty"] for r in results)):
        subset = [r for r in results if r["difficulty"] == d]
        correct = sum(1 for r in subset if r["correct"])
        summary["by_difficulty"][f"level_{d}"] = {
            "total": len(subset),
            "passed": correct,
            "accuracy": round(correct / max(len(subset), 1) * 100, 1),
        }

    output = {
        "framework": "tabula_rasa_benchmark_export",
        "summary": summary,
        "results": mle_results,
    }

    path = Path(path)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Exported {len(results)} results to {path}")
    print(f"  Accuracy: {summary['accuracy']:.1f}% ({summary['passed']}/{summary['total']})")


def export_proofgrid(results: list[dict], path: str | Path) -> None:
    """Export results in ProofGrid-compatible format."""
    from datetime import datetime

    proofgrid = {
        "benchmark": {
            "name": "Tabula-Rasa Arithmetic Benchmark",
            "version": "1.0",
            "description": "Arithmetic evaluation generated by Tabula-Rasa",
        },
        "model": {
            "name": "MathTransformer",
            "params": "1M-10M",
        },
        "results": [],
        "summary": {
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "accuracy": round(
                sum(1 for r in results if r["correct"]) / max(len(results), 1) * 100, 2
            ),
            "generated_at": datetime.now().isoformat(),
        },
    }

    for r in results:
        proofgrid["results"].append({
            "id": r["problem_id"],
            "expression": r["problem"],
            "expected": r["expected"],
            "predicted": r["predicted"],
            "correct": r["correct"],
            "difficulty": r["difficulty"],
        })

    path = Path(path)
    with open(path, "w") as f:
        json.dump(proofgrid, f, indent=2)
    print(f"  ProofGrid export: {path}")
    print(f"  Accuracy: {proofgrid['summary']['accuracy']:.1f}%")


def full_export(model: Any, tokenizer: Any, output_dir: str | Path = "exports",
                num_problems: int = 200, max_digits: int = 4) -> dict:
    """Generate and evaluate on ProofGrid problems, export in all formats.

    Args:
        model: MathTransformer.
        tokenizer: MathTokenizer.
        output_dir: Output directory.
        num_problems: Number of problems to generate.
        max_digits: Max digits per operand.

    Returns:
        Dict of export paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BENCHMARK EXPORT — ProofGrid + MLE-bench formats")
    print("=" * 60)

    # Generate problems
    print(f"\n  Generating {num_problems} problems (max {max_digits}-digit)...")
    problems = generate_proofgrid_problems(num_problems, max_digits)
    print(f"  Generated {len(problems)} problems")

    # Evaluate
    print(f"  Evaluating model...")
    t0 = time.time()
    results = evaluate_proofgrid(model, tokenizer, problems, verbose=False)
    elapsed = time.time() - t0
    correct = sum(1 for r in results if r["correct"])
    print(f"  Accuracy: {correct}/{len(results)} = {correct/max(len(results),1)*100:.1f}% ({elapsed:.1f}s)")

    # Export
    print(f"\n  Exporting...")
    exports = {}

    pg_path = output_dir / "proofgrid_results.json"
    export_proofgrid(results, pg_path)
    exports["proofgrid"] = str(pg_path)

    mb_path = output_dir / "mlebench_results.json"
    export_mlebench(results, mb_path)
    exports["mlebench"] = str(mb_path)

    # Full detail
    detail_path = output_dir / "full_results.json"
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2)
    exports["full"] = str(detail_path)

    print(f"\n{'='*60}")
    print(f"  Export complete — {len(exports)} files")
    for fmt, path in exports.items():
        print(f"    {fmt}: {path}")
    print(f"{'='*60}")

    return exports


if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    fmt = "--format" in sys.argv and sys.argv[sys.argv.index("--format") + 1] if "--format" in sys.argv else "all"
    num = int(sys.argv[sys.argv.index("--num") + 1]) if "--num" in sys.argv else 100

    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer

    print(f"Loading model from {ckpt}")
    tok = MathTokenizer()
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]

    # Handle vocab mismatch
    ckpt_vocab = sd["token_embedding.weight"].shape[0]
    if ckpt_vocab != cfg.vocab_size:
        old_w = sd["token_embedding.weight"]
        new_w = torch.randn(cfg.vocab_size, cfg.d_model) * 0.02
        new_w[:ckpt_vocab] = old_w
        sd["token_embedding.weight"] = new_w
        old_w = sd["lm_head.weight"]
        new_w = torch.randn(cfg.vocab_size, cfg.d_model) * 0.02
        new_w[:ckpt_vocab] = old_w
        sd["lm_head.weight"] = new_w
        print(f"  Extended embeddings: {ckpt_vocab} → {cfg.vocab_size}")

    model.load_state_dict(sd, strict=False)
    model.eval()

    if fmt == "proofgrid":
        problems = generate_proofgrid_problems(num)
        results = evaluate_proofgrid(model, tok, problems)
        export_proofgrid(results, "exports/proofgrid_results.json")
    elif fmt == "mlebench":
        problems = generate_proofgrid_problems(num)
        results = evaluate_proofgrid(model, tok, problems)
        export_mlebench(results, "exports/mlebench_results.json")
    else:
        full_export(model, tok, "exports", num_problems=num)
