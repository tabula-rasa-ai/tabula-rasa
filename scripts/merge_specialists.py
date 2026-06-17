"""Post-Hoc Specialist Merging — Task Arithmetic / TIES.

Creates a single multi-task model from isolated specialist checkpoints,
bypassing continual learning entirely. Supports two merging strategies:

1. **Task Arithmetic** (default)::
       θ_merged = θ_base + Σ_i λ_i · (θ_i - θ_base)

2. **TIES-Merging** (when ``--ties`` is set): Trims conflicting sign
   changes before merging, keeping only the top-k% of task vector
   components by magnitude.

Usage:
    python3 scripts/merge_specialists.py add sub --lambdas 1.0 1.0
    python3 scripts/merge_specialists.py add sub mul --lambdas 1.0 0.8 0.7 --ties
    python3 scripts/merge_specialists.py add sub --eval-only
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer


def load_specialist(op: str, device: str = "cpu") -> dict:
    """Load a trained specialist checkpoint.

    Returns:
        Dict with keys ``'model_state_dict'``, ``'acc'``, ``'config'``.
    """
    path = ROOT / "specialists" / "math" / op / "best.pt"
    if not path.exists():
        path = ROOT / "specialists" / "math" / op / "final.pt"
    if not path.exists():
        print(f"  [X] No checkpoint found for '{op}' at {path.parent}")
        return None

    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config", {})
    acc = ckpt.get("acc", 0.0)
    print(f"  Loaded '{op}' — accuracy: {acc:.1f}%, params: {sum(p.numel() for p in sd.values()):,}")
    return {"state_dict": sd, "acc": acc, "config": cfg}


def build_base_model(vocab_size: int) -> MathTransformer:
    """Build a fresh base model for merging."""
    cfg = Config()
    cfg.vocab_size = vocab_size
    model = MathTransformer(cfg)
    return model


def task_arithmetic_merge(
    specialist_sds: list[dict],
    lambdas: list[float],
    base_model: MathTransformer,
    trim_topk: float = 0.0,
) -> dict:
    """Merge specialist state dicts using Task Arithmetic or TIES.

    Args:
        specialist_sds: List of ``{'state_dict': sd, 'acc': acc}`` dicts.
        lambdas: Per-specialist scaling coefficients (same length as sds).
        base_model: Freshly initialized model (provides θ_base).
        trim_topk: If > 0, TIES-mode: keep only top-k fraction of
            task vector components by magnitude. 0.2 = keep top 20%.

    Returns:
        Merged state dict.
    """
    base_sd = base_model.state_dict()
    merged_sd = {k: v.clone() for k, v in base_sd.items()}
    n_tasks = len(specialist_sds)

    for task_idx, spec in enumerate(specialist_sds):
        lam = lambdas[task_idx]
        sd = spec["state_dict"]
        task_vec = {}

        for k in merged_sd:
            if k not in sd:
                continue
            # Handle size mismatches (e.g. tokenizer vocab changes)
            if sd[k].shape != merged_sd[k].shape:
                # Try to pad or truncate
                if sd[k].numel() < merged_sd[k].numel():
                    # Pad with zeros
                    padded = torch.zeros_like(merged_sd[k])
                    flat = sd[k].view(-1)
                    padded.view(-1)[:flat.numel()] = flat
                    delta = padded - base_sd[k]
                else:
                    # Truncate
                    flat = sd[k].view(-1)
                    delta = flat[:merged_sd[k].numel()].view(merged_sd[k].shape) - base_sd[k]
            else:
                delta = sd[k] - base_sd[k]
            task_vec[k] = delta

        # TIES: trim conflicting signs
        if trim_topk > 0 and task_idx > 0:
            # For each parameter, find sign agreement across tasks
            for k in list(task_vec.keys()):
                # Keep only top-k% by magnitude
                flat = task_vec[k].abs().view(-1)
                threshold_idx = max(1, int(flat.numel() * (1 - trim_topk)))
                vals, _ = torch.topk(flat, threshold_idx)
                threshold = vals[-1].item()
                mask = task_vec[k].abs() < threshold
                task_vec[k][mask] = 0.0

        # Apply task vector: θ += λ · τ
        for k in task_vec:
            merged_sd[k] = merged_sd[k] + lam * task_vec[k]

    return merged_sd


def evaluate_merged(model: MathTransformer, tokenizer: MathTokenizer, ops: list[str],
                    num: int = 100) -> dict:
    """Evaluate the merged model on multiple operations."""
    from train_specialist import generate_problem, parse_cot_answer

    model.eval()
    results = {}
    for op in ops:
        correct = 0
        for _ in range(num):
            expr, ans_raw = generate_problem(
                op, min_digits=1, max_digits=2,
                reversed=(op in ("add", "sub")),
                scratchpad=(op in ("add", "sub")),
                cot=(op in ("add", "sub")),
            )
            prompt = f"{expr}="
            out = model.generate(tokenizer, prompt, max_new_tokens=30, temperature=0.0)
            if op in ("add", "sub"):
                pred = parse_cot_answer(out)
                true_answer = str(ans_raw).split("<END>")[-1].strip() if "<END>" in ans_raw else ans_raw
                if pred == true_answer:
                    correct += 1
            else:
                if "=" in out:
                    pred = out.split("=")[-1].strip()
                    if pred == ans_raw:
                        correct += 1
        results[op] = correct / num * 100
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Merge trained specialists via Task Arithmetic / TIES"
    )
    parser.add_argument("ops", nargs="+", help="Operations to merge (e.g. add sub mul)")
    parser.add_argument(
        "--lambdas", type=float, nargs="*", default=None,
        help="Per-specialist scaling coefficients (default: 1.0 for all)"
    )
    parser.add_argument(
        "--ties", type=float, default=0.0, const=0.2, nargs="?",
        help="TIES-Merging: keep top-k fraction of task vector (default: 0.2). "
             "Pass --ties without value to use 0.2."
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip merge, just evaluate existing merged model"
    )
    parser.add_argument(
        "--output", type=str, default="specialists/math/merged",
        help="Output directory (default: specialists/math/merged)"
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate after merge")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all specialists
    specialists = []
    for op in args.ops:
        spec = load_specialist(op, device)
        if spec is None:
            sys.exit(1)
        specialists.append(spec)

    # Default lambdas: uniform
    lambdas = args.lambdas or [1.0] * len(args.ops)
    if len(lambdas) != len(args.ops):
        print(f"  [X] Got {len(lambdas)} lambdas for {len(args.ops)} ops")
        sys.exit(1)

    # Build base model from the tokenizer vocabulary
    tok = MathTokenizer()
    base_model = build_base_model(tok.vocab_size)

    if not args.eval_only:
        print(f"\n{'='*60}")
        print("  Task Arithmetic Merge")
        print(f"  Operations: {', '.join(args.ops)}")
        print(f"  Lambdas:    {lambdas}")
        print(f"  TIES top-k: {args.ties or 'disabled'}")
        print(f"  Base model: {count_parameters(base_model):,} params")
        print(f"{'='*60}\n")

        # Merge
        merged_sd = task_arithmetic_merge(
            specialists, lambdas, base_model, trim_topk=args.ties
        )

        # Save
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": merged_sd,
            "config": {
                "d_model": base_model.config.d_model,
                "n_layers": base_model.config.n_layers,
                "n_heads": base_model.config.n_heads,
                "d_ff": base_model.config.d_ff,
                "merged_ops": args.ops,
                "merged_lambdas": lambdas,
            },
        }, out_dir / "best.pt")
        tok.save(str(out_dir / "tokenizer.json"))
        print(f"  Merged model saved to {out_dir / 'best.pt'}")

    # Evaluate
    if args.eval or args.eval_only:
        merged_model = build_base_model(tok.vocab_size)
        ckpt = torch.load(Path(args.output) / "best.pt", map_location=device, weights_only=True)
        merged_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        merged_model.to(device)

        print("\n  Evaluating merged model...")
        results = evaluate_merged(merged_model, tok, args.ops, num=50)
        for op, acc in results.items():
            print(f"    {op}: {acc:.1f}%")
        avg = sum(results.values()) / len(results)
        print(f"    Average: {avg:.1f}%")


if __name__ == "__main__":
    main()
