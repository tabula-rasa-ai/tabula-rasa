"""Specialist Consolidation — Model Soup for LoRA adapters.

Implements three strategies for managing conversational specialist LoRA adapters:

1. **Model Soup** (``model_soup``): Weighted average of multiple adapters'
   A and B matrices into a single merged adapter.

2. **Similar Specialist Consolidation** (``consolidate_similar_specialists``):
   Groups adapters by pairwise cosine similarity of their weight vectors,
   then merges each group into one consolidated adapter. Original adapters
   are renamed to ``lora.pt.bak`` to mark them as consolidated.

3. **Stale Adapter Pruning** (``prune_stale_adapters``): Removes adapters
   that haven't been accessed recently, keeping only the top-K most recent.

All operations work on LoRA checkpoint files (``specialists/<name>/lora.pt``)
— the lightweight adapter files, not full model checkpoints.

Usage:
    from egefalos.specialist_consolidation import (
        model_soup,
        consolidate_similar_specialists,
        prune_stale_adapters,
    )

    # Merge multiple adapters with uniform weights
    model_soup(
        ["specialists/greeting/lora.pt", "specialists/question/lora.pt"],
        "specialists/consolidated/chat/lora.pt",
    )

    # Auto-discover and group similar adapters
    consolidate_similar_specialists(similarity_threshold=0.8)

    # Keep only the 10 most recently used adapters
    prune_stale_adapters(keep_top_k=10, max_age_days=30)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ─── Constants ────────────────────────────────────────────────────

SPECIALISTS_ROOT = Path("specialists")
CONSOLIDATED_ROOT = SPECIALISTS_ROOT / "consolidated"
ACCESS_LOG = SPECIALISTS_ROOT / ".adapter_access.json"


# ─── Helpers ──────────────────────────────────────────────────────


def _load_lora_state(path: Path | str) -> dict[str, torch.Tensor]:
    """Load a LoRA adapter state dict from disk.

    Args:
        path: Path to a ``.pt`` file saved by ``save_lora_adapters``.

    Returns:
        The full state dict as loaded by ``torch.load``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {path}")
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _extract_weight_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all trainable LoRA matrices into a single 1-D vector.

    Concatenates ``lora_{i}_A`` and ``lora_{i}_B`` tensors for every
    layer *i* found in *state*.

    Args:
        state: A LoRA state dict from ``save_lora_adapters``.

    Returns:
        A 1-D ``torch.Tensor`` of concatenated weights.
    """
    vectors: list[torch.Tensor] = []
    i = 0
    while True:
        key_a = f"lora_{i}_A"
        key_b = f"lora_{i}_B"
        if key_a in state:
            vectors.append(state[key_a].flatten())
        if key_b in state:
            vectors.append(state[key_b].flatten())
        if key_a not in state and key_b not in state:
            if i == 0:
                # No LoRA keys found — empty state
                break
            # We've exhausted all layers
            break
        i += 1
    if not vectors:
        return torch.zeros(0)
    return torch.cat(vectors)


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute cosine similarity between two flattened weight vectors.

    Args:
        vec_a: First weight vector (1-D).
        vec_b: Second weight vector (1-D).

    Returns:
        Cosine similarity in ``[-1, 1]``. Returns 0.0 if either vector
        is zero-norm.
    """
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return 0.0
    # Handle size mismatches gracefully (e.g. different LoRA ranks)
    min_len = min(vec_a.numel(), vec_b.numel())
    a = vec_a[:min_len].float()
    b = vec_b[:min_len].float()
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a.item() < 1e-10 or norm_b.item() < 1e-10:
        return 0.0
    return float(torch.dot(a, b) / (norm_a * norm_b))


def _discover_adapter_paths() -> list[Path]:
    """Find all active LoRA adapter files (excluding consolidated and backup).

    Returns:
        Sorted list of ``Path`` objects pointing to ``lora.pt`` files.
    """
    paths = sorted(SPECIALISTS_ROOT.glob("*/lora.pt"))
    # Exclude internal / consolidated directories
    filtered = [
        p
        for p in paths
        if not p.parent.name.startswith("_")
        and not p.parent.name.startswith("consolidated")
        and not p.parent.name.startswith(".")
    ]
    return filtered


def _get_adapter_name(lora_path: Path) -> str:
    """Return the adapter name (parent directory basename)."""
    return lora_path.parent.name


def _log_access(adapter_name: str) -> None:
    """Record that an adapter was accessed (for staleness tracking).

    Args:
        adapter_name: Name of the adapter (directory basename).
    """
    try:
        log = _read_access_log()
        log[adapter_name] = time.time()
        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        ACCESS_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    except Exception:
        pass


def _read_access_log() -> dict[str, float]:
    """Read the adapter access timestamp log.

    Returns:
        Dict mapping adapter name -> Unix timestamp of last access.
    """
    if ACCESS_LOG.exists():
        try:
            return json.loads(ACCESS_LOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ─── Main API ─────────────────────────────────────────────────────


def model_soup(
    adapter_paths: list[Path | str],
    output_path: Path | str,
    weights: Optional[list[float]] = None,
) -> dict:
    """Merge multiple LoRA adapters via weighted average (Model Soup).

    Loads each adapter's ``A`` and ``B`` matrices, computes a weighted
    average across all adapters, and saves the result.

    .. math::
        \\theta_{\\text{soup}} = \\sum_i w_i \\cdot \\theta_i

    Args:
        adapter_paths: List of paths to ``lora.pt`` files.
        output_path: Destination path for the merged adapter.
        weights: Per-adapter weights. If ``None``, uniform weighting is
            used (``1/N``). If provided, must have the same length as
            *adapter_paths* and will be normalised to sum to 1.

    Returns:
        Dict with keys:
            - ``'adapter_names'``: List of adapter names (directory names).
            - ``'weights'``: Normalised weights used.
            - ``'output_path'``: Where the merged adapter was saved.
            - ``'total_layers'``: Number of LoRA layers merged.
            - ``'total_params'``: Total parameter count per layer.

    Raises:
        ValueError: If *adapter_paths* is empty, or if *weights* length
            mismatches.
        FileNotFoundError: If any adapter path does not exist.
    """
    paths = [Path(p) for p in adapter_paths]
    if not paths:
        raise ValueError("adapter_paths must contain at least one path")

    n = len(paths)
    if weights is not None:
        if len(weights) != n:
            raise ValueError(
                f"Expected {n} weights, got {len(weights)}"
            )
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()  # normalise
    else:
        w = np.ones(n, dtype=np.float64) / n

    # Load all adapter states
    states: list[dict] = []
    names: list[str] = []
    for p in paths:
        states.append(_load_lora_state(p))
        names.append(_get_adapter_name(p))

    # Determine the layer count from the first adapter
    first_state = states[0]
    i = 0
    layer_keys_a: list[str] = []
    layer_keys_b: list[str] = []
    while True:
        ka = f"lora_{i}_A"
        kb = f"lora_{i}_B"
        has_a = ka in first_state
        has_b = kb in first_state
        if not has_a and not has_b:
            break
        if has_a:
            layer_keys_a.append(ka)
        if has_b:
            layer_keys_b.append(kb)
        i += 1

    if i == 0:
        raise ValueError("No LoRA layers found in adapter state")

    # Build merged state dict
    merged: dict = {}
    for ka in layer_keys_a:
        tensors = torch.stack([s[ka] for s in states])
        # Weighted sum: Σ w_i * tensor_i
        merged[ka] = (tensors * torch.tensor(w, dtype=tensors.dtype).view(-1, 1, 1)).sum(
            dim=0
        )

    for kb in layer_keys_b:
        tensors = torch.stack([s[kb] for s in states])
        merged[kb] = (tensors * torch.tensor(w, dtype=tensors.dtype).view(-1, 1, 1)).sum(
            dim=0
        )

    # Copy metadata from first adapter
    for key in first_state:
        if key.startswith("lora_") and key not in merged:
            # Copy metadata fields like rank, alpha, in/out features
            if any(k in key for k in ("_rank", "_alpha", "_in_features", "_out_features")):
                merged[key] = first_state[key]

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(output))

    # Compute stats
    total_params = sum(
        merged[k].numel()
        for k in merged
        if k.endswith("_A") or k.endswith("_B")
    )

    result: dict = {
        "adapter_names": names,
        "weights": w.tolist(),
        "output_path": str(output),
        "total_layers": len(layer_keys_a),
        "total_params": total_params,
    }
    return result


def consolidate_similar_specialists(
    similarity_threshold: float = 0.8,
) -> list[dict]:
    """Group and merge similar LoRA adapters by cosine similarity.

    Loads all active ``lora.pt`` files from ``specialists/*/``,
    computes a pairwise cosine similarity matrix of their flattened
    weight vectors, groups adapters whose similarity exceeds
    *similarity_threshold*, and merges each group into a consolidated
    adapter saved under ``specialists/consolidated/<group_name>/``.

    After consolidation, original adapter files are renamed to
    ``lora.pt.bak`` to mark them as consolidated (they can be restored
    by removing the ``.bak`` suffix).

    Args:
        similarity_threshold: Minimum cosine similarity to consider two
            adapters "similar". Must be in ``(0, 1]``. Default 0.8.

    Returns:
        List of dicts, one per consolidated group, with keys:
            - ``'group_name'``: Generated group name.
            - ``'members'``: List of adapter names.
            - ``'mean_similarity'``: Average within-group similarity.
            - ``'output_path'``: Path to the saved consolidated adapter.
            - ``'num_layers'``: Number of LoRA layers.

    Raises:
        ValueError: If *similarity_threshold* is out of range.
    """
    if not 0 < similarity_threshold <= 1:
        raise ValueError(
            f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
        )

    adapter_paths = _discover_adapter_paths()
    if len(adapter_paths) < 2:
        return []  # Nothing to consolidate

    # Load weight vectors
    names: list[str] = []
    vectors: list[torch.Tensor] = []
    states: dict[str, dict] = {}
    for p in adapter_paths:
        name = _get_adapter_name(p)
        state = _load_lora_state(p)
        vec = _extract_weight_vector(state)
        names.append(name)
        vectors.append(vec)
        states[name] = state

    # Build similarity matrix (upper triangle)
    n = len(names)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # Greedy grouping: start with unassigned nodes, form groups
    assigned: set[int] = set()
    groups: list[list[int]] = []

    for i in range(n):
        if i in assigned:
            continue
        # Find all unassigned nodes similar to i
        group = [i]
        assigned.add(i)
        for j in range(n):
            if j in assigned:
                continue
            if sim_matrix[i, j] >= similarity_threshold:
                group.append(j)
                assigned.add(j)
        groups.append(group)

    results: list[dict] = []
    CONSOLIDATED_ROOT.mkdir(parents=True, exist_ok=True)

    for idx, group in enumerate(groups):
        if len(group) < 2:
            continue  # Skip singleton groups (no consolidation needed)

        member_names = [names[i] for i in group]
        group_name = f"group_{idx:03d}__" + "_".join(member_names)
        output_path = CONSOLIDATED_ROOT / group_name / "lora.pt"

        # Collect adapter paths for this group
        group_paths = [adapter_paths[i] for i in group]

        # Merge with uniform weights
        merge_result = model_soup(
            adapter_paths=group_paths,
            output_path=output_path,
            weights=None,
        )

        # Compute mean within-group similarity
        group_sims: list[float] = []
        for i in group:
            for j in group:
                if i < j:
                    group_sims.append(float(sim_matrix[i, j]))
        mean_sim = float(np.mean(group_sims)) if group_sims else 0.0

        # Mark originals as consolidated (rename lora.pt -> lora.pt.bak)
        for p in group_paths:
            bak_path = p.with_suffix(".pt.bak")
            if p.exists():
                shutil.move(str(p), str(bak_path))

        results.append(
            {
                "group_name": group_name,
                "members": member_names,
                "mean_similarity": round(mean_sim, 4),
                "output_path": str(output_path),
                "num_layers": merge_result["total_layers"],
            }
        )

    return results


def prune_stale_adapters(
    keep_top_k: int = 10,
    max_age_days: int = 30,
) -> dict:
    """Remove stale LoRA adapters, keeping only the most recently used.

    An adapter is considered "stale" if:
    1. It is not among the *keep_top_k* most recently accessed adapters
       (by access timestamp in the access log), **and**
    2. Its last access was more than *max_age_days* ago.

    Stale adapters are moved to a ``specialists/_stale/<name>/``
    directory rather than permanently deleted, allowing recovery.

    This function respects consolidated adapters — it does not prune
    adapters under ``specialists/consolidated/``.

    Args:
        keep_top_k: Minimum number of adapters to keep regardless of age
            (default 10). Only adapters beyond this count are eligible
            for pruning.
        max_age_days: Age threshold in days (default 30). An adapter
            must be older than this to be pruned.

    Returns:
        Dict with keys:
            - ``'pruned'``: List of pruned adapter names.
            - ``'kept'``: List of kept adapter names.
            - ``'stale_dir'``: Path where pruned adapters were moved.
    """
    access_log = _read_access_log()
    adapter_paths = _discover_adapter_paths()
    now = time.time()
    cutoff = now - max_age_days * 86400

    # Score each adapter: use access log timestamp, or file mtime as fallback
    scored: list[tuple[float, Path]] = []
    for p in adapter_paths:
        name = _get_adapter_name(p)
        ts = access_log.get(name, None)
        if ts is None:
            # Fallback to file modification time
            try:
                ts = os.path.getmtime(p)
            except OSError:
                ts = 0.0
        scored.append((ts, p))

    # Sort by access time descending (most recent first)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep the top K
    kept_paths: list[Path] = [p for _, p in scored[:keep_top_k]]
    candidates: list[Path] = [p for _, p in scored[keep_top_k:]]

    # Prune candidates that are older than max_age_days
    pruned_names: list[str] = []
    kept_names: list[str] = [_get_adapter_name(p) for p in kept_paths]

    stale_dir = SPECIALISTS_ROOT / "_stale"
    for p in candidates:
        name = _get_adapter_name(p)
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0.0

        if now - mtime < cutoff:
            # Not old enough — keep it
            kept_names.append(name)
            kept_paths.append(p)
            continue

        # Prune: move to _stale directory
        dest = stale_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        for src_file in p.parent.iterdir():
            if src_file.is_file():
                shutil.move(str(src_file), str(dest / src_file.name))
        pruned_names.append(name)

    return {
        "pruned": pruned_names,
        "kept": kept_names,
        "stale_dir": str(stale_dir),
    }


# ─── Convenience CLI ──────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for consolidation operations.

    Usage:
        python3 -m egefalos.specialist_consolidation soup \\
            specialists/greeting/lora.pt specialists/question/lora.pt \\
            --output specialists/merged/greeting_question/lora.pt

        python3 -m egefalos.specialist_consolidation consolidate \\
            --threshold 0.75

        python3 -m egefalos.specialist_consolidation prune \\
            --keep 5 --age 14
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Specialist Consolidation — Model Soup for LoRA adapters"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Soup subcommand
    soup_parser = subparsers.add_parser(
        "soup", help="Merge multiple LoRA adapters via Model Soup"
    )
    soup_parser.add_argument(
        "adapters", nargs="+", help="Paths to lora.pt files"
    )
    soup_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for merged adapter",
    )
    soup_parser.add_argument(
        "--weights",
        "-w",
        type=float,
        nargs="*",
        default=None,
        help="Per-adapter weights (default: uniform)",
    )

    # Consolidate subcommand
    cons_parser = subparsers.add_parser(
        "consolidate", help="Group and merge similar adapters"
    )
    cons_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Cosine similarity threshold (default: 0.8)",
    )

    # Prune subcommand
    prune_parser = subparsers.add_parser(
        "prune", help="Remove stale adapters"
    )
    prune_parser.add_argument(
        "--keep",
        "-k",
        type=int,
        default=10,
        help="Minimum number of adapters to keep (default: 10)",
    )
    prune_parser.add_argument(
        "--age",
        "-a",
        type=int,
        default=30,
        help="Max age in days before pruning (default: 30)",
    )

    args = parser.parse_args()

    if args.command == "soup":
        result = model_soup(args.adapters, args.output, args.weights)
        print(f"  [*] Model Soup complete:")
        print(f"      Adapters: {', '.join(result['adapter_names'])}")
        print(f"      Weights:  {[round(w, 3) for w in result['weights']]}")
        print(f"      Output:   {result['output_path']}")
        print(f"      Layers:   {result['total_layers']}")
        print(f"      Params:   {result['total_params']:,}")

    elif args.command == "consolidate":
        results = consolidate_similar_specialists(args.threshold)
        if not results:
            print(
                "  [*] No similar adapter groups found (need at least 2 adapters "
                f"with similarity >= {args.threshold})"
            )
        else:
            print(f"  [*] Consolidated {len(results)} groups:")
            for r in results:
                print(
                    f"      Group '{r['group_name']}': "
                    f"{r['members']} "
                    f"(sim={r['mean_similarity']:.3f})"
                )
                print(f"        -> {r['output_path']}")

    elif args.command == "prune":
        result = prune_stale_adapters(
            keep_top_k=args.keep, max_age_days=args.age
        )
        print(f"  [*] Pruning complete:")
        print(f"      Kept:   {len(result['kept'])} adapters")
        print(f"      Pruned: {len(result['pruned'])} adapters")
        if result["pruned"]:
            print(f"      Moved to: {result['stale_dir']}/")
            for name in result["pruned"]:
                print(f"        - {name}")


if __name__ == "__main__":
    main()
