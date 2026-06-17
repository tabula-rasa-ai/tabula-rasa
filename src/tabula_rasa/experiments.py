"""Experiment tracker — logs every training run with config + benchmark results.

Each run is saved as a timestamped JSON in experiments/runs/.
Compare runs, track what worked, and never lose a result.

Usage:
    from tabula_rasa.experiments import log_run, list_runs, compare_runs

    # After training and benchmarking:
    log_run(
        name="moe_rasa_10M",
        config={"preset": "10M", "use_moe": True, "use_rasa": True},
        metrics={"accuracy": 89.5, "ood": 56.0},
        checkpoint="checkpoints/best.pt",
        notes="MoE + RASA, 30K steps",
    )

    # Later:
    runs = list_runs()
    compare_runs(runs[-2], runs[-1])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentRun:
    """A single experiment run with config, metrics, and results."""

    name: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    timestamp: str = ""
    duration_s: float = 0.0
    checkpoint: str = ""
    notes: str = ""
    git_hash: str = ""
    run_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.run_id:
            self.run_id = f"{self.timestamp[:19].replace('T', '_').replace(':', '')}_{self.name[:20]}"


def log_run(
    name: str,
    config: dict | None = None,
    metrics: dict | None = None,
    checkpoint: str = "",
    notes: str = "",
    duration_s: float = 0.0,
) -> ExperimentRun:
    """Log an experiment run to experiments/runs/.

    Args:
        name: Human-readable name (e.g. 'moe_rasa_10M_30K').
        config: Dict of config key-value pairs used for this run.
        metrics: Dict of metric name -> value (e.g. {'accuracy': 89.5}).
        checkpoint: Path to best checkpoint.
        notes: Free-text notes about the run.
        duration_s: Training duration in seconds.

    Returns:
        The ExperimentRun object that was saved.
    """
    # Auto-capture config from tabula_rasa.config if not provided
    if config is None:
        try:
            from tabula_rasa.config import Config as _Cfg
            cfg = _Cfg()
            # Capture key architectural settings
            config = {
                "preset": cfg.config_preset,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "d_ff": cfg.d_ff,
                "max_seq_len": cfg.max_seq_len,
                "use_moe": cfg.use_moe,
                "num_experts": cfg.num_experts,
                "top_k": cfg.top_k,
                "use_rasa": cfg.use_rasa,
                "use_lora": cfg.use_lora,
                "lora_rank": cfg.lora_rank,
                "use_streambp": cfg.use_streambp,
                "use_entropy_curriculum": cfg.use_entropy_curriculum,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "max_steps": cfg.max_steps,
            }
        except ImportError:
            config = {}

    # Try to get git hash
    git_hash = ""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=RUNS_DIR.parent,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    run = ExperimentRun(
        name=name,
        config=config or {},
        metrics=metrics or {},
        checkpoint=checkpoint,
        notes=notes,
        duration_s=duration_s,
        git_hash=git_hash,
    )

    # Save
    path = RUNS_DIR / f"{run.run_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(run), f, indent=2, default=str)
    print(f"[Experiment] Logged {name} → {path}")
    return run


def list_runs(limit: int = 20) -> list[ExperimentRun]:
    """List recent experiment runs, newest first.

    Args:
        limit: Max runs to return.

    Returns:
        List of ExperimentRun objects.
    """
    paths = sorted(RUNS_DIR.glob("*.json"), reverse=True)
    runs = []
    for p in paths[:limit]:
        try:
            with open(p) as f:
                data = json.load(f)
            runs.append(ExperimentRun(**data))
        except Exception as e:
            print(f"  [!] Failed to load {p.name}: {e}")
    return runs


def print_runs(runs: list[ExperimentRun] | None = None):
    """Pretty-print experiment runs as a table."""
    if runs is None:
        runs = list_runs()

    if not runs:
        print("No experiments logged yet.")
        return

    print(f"{'Run':<30} {'Name':<25} {'Accuracy':<10} {'Preset':<6} {'Features':<30} {'Notes'}")
    print(f"{'─'*30} {'─'*25} {'─'*10} {'─'*6} {'─'*30} {'─'*20}")
    for r in runs:
        name_short = r.name[:24]
        acc = r.metrics.get("accuracy", r.metrics.get("basic_accuracy", ""))
        acc_str = f"{acc:.1f}%" if isinstance(acc, (int, float)) else "—"
        preset = r.config.get("preset", r.config.get("config_preset", ""))
        features = []
        if r.config.get("use_moe"): features.append("MoE")
        if r.config.get("use_rasa"): features.append("RASA")
        if r.config.get("use_lora"): features.append("LoRA")
        if r.config.get("use_streambp", True): features.append("StreamBP")
        feat_str = "+".join(features) if features else "base"
        print(f"{r.run_id[:28]:<30} {name_short:<25} {acc_str:<10} {preset:<6} {feat_str:<30} {r.notes[:18]}")


def compare_runs(run_a: ExperimentRun, run_b: ExperimentRun) -> dict[str, float]:
    """Compare metrics between two runs.

    Args:
        run_a: First run (baseline).
        run_b: Second run (comparison).

    Returns:
        Dict of metric_name -> delta (b - a).
    """
    deltas = {}
    all_keys = set(run_a.metrics.keys()) | set(run_b.metrics.keys())
    for key in sorted(all_keys):
        va = run_a.metrics.get(key)
        vb = run_b.metrics.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            deltas[key] = round(vb - va, 2)

    print(f"{'='*60}")
    print(f"  COMPARISON: {run_a.name} vs {run_b.name}")
    print(f"{'='*60}")
    if deltas:
        print(f"  {'Metric':<30} {'A':>8} {'B':>8} {'Δ':>8}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")
        for key, delta in deltas.items():
            va = run_a.metrics.get(key, 0)
            vb = run_b.metrics.get(key, 0)
            print(f"  {key:<30} {va:>8.2f} {vb:>8.2f} {delta:>+8.2f}")
    else:
        print("  No numeric metrics to compare.")
    print(f"{'='*60}")
    return deltas


def export_csv(path: str | Path = "experiments/runs.csv"):
    """Export all runs to CSV for analysis in spreadsheets."""
    runs = list_runs(limit=1000)
    if not runs:
        print("No runs to export.")
        return

    import csv
    path = Path(path)
    with open(path, "w", newline="") as f:
        # Collect all possible keys
        all_config_keys = set()
        all_metric_keys = set()
        for r in runs:
            all_config_keys.update(r.config.keys())
            all_metric_keys.update(r.metrics.keys())
        config_keys = sorted(all_config_keys)
        metric_keys = sorted(all_metric_keys)
        fieldnames = ["run_id", "name", "timestamp", "git_hash", "duration_s", "notes"] + config_keys + metric_keys

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            row = {
                "run_id": r.run_id,
                "name": r.name,
                "timestamp": r.timestamp,
                "git_hash": r.git_hash,
                "duration_s": r.duration_s,
                "notes": r.notes,
            }
            row.update(r.config)
            row.update(r.metrics)
            writer.writerow(row)
    print(f"Exported {len(runs)} runs to {path}")


def clear_runs():
    """Delete all logged experiment runs (use with caution)."""
    count = len(list(RUNS_DIR.glob("*.json")))
    for p in RUNS_DIR.glob("*.json"):
        p.unlink()
    print(f"Deleted {count} experiment logs.")


if __name__ == "__main__":
    import sys

    if "--list" in sys.argv:
        runs = list_runs()
        print_runs(runs)

    elif "--export" in sys.argv:
        idx = sys.argv.index("--export")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "experiments/runs.csv"
        export_csv(path)

    elif "--clear" in sys.argv:
        clear_runs()

    else:
        print(f"Experiment tracker — {len(list(RUNS_DIR.glob('*.json')))} runs logged")
        print(f"  Runs dir: {RUNS_DIR}")
        print(f"  Commands: --list, --export [path], --clear")
