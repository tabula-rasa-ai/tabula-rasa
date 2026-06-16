"""Socratic vs Standard Training Benchmark.

Compares standard training vs Socratic self-improvement across multiple seeds.
Generates a results table with accuracy before/after Socratic refinement.

Usage:
    python3 scripts/socratic_benchmark.py              # Quick: 2 seeds, 5000 steps
    python3 scripts/socratic_benchmark.py --full       # Full: 5 seeds, 30000 steps
    python3 scripts/socratic_benchmark.py --seeds 3 --steps 10000
"""

import subprocess, sys, json, time, re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EXPERIMENTS = BASE / 'experiments.json'
RESULTS = BASE / 'RESULTS.md'


def _detect_gpu() -> dict:
    """Detect GPU and return optimal training config."""
    import torch
    if not torch.cuda.is_available():
        return {'gpu': False, 'batch': 128, 'amp': False, 'time_per_step': 0.9}

    n_gpu = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

    # Batch and AMP based on GPU class
    if mem > 20:  # RTX 4090/5090 tier
        batch, amp = 2048, True
    elif mem > 10:  # RTX 3090/4070 tier
        batch, amp = 1024, True
    elif mem > 6:  # RTX 3060/4060 tier
        batch, amp = 512, True
    else:  # T4 / small GPU
        batch, amp = 256, True

    # Estimate steps/sec for 1M param model
    # RTX 5090: ~5000+ st/s with batch=2048+AMP
    # T4: ~200 st/s with batch=256+AMP
    time_per_step = max(0.0002, 1.0 / (max(200, mem * 200)))

    return {
        'gpu': True,
        'gpu_name': name,
        'gpu_count': n_gpu,
        'vram_gb': round(mem, 1),
        'batch': batch,
        'amp': amp,
        'time_per_step': time_per_step,
    }


def run_experiment(label: str, op: str, steps: int, gpu_config: dict, extra_args: list[str] = None) -> dict:
    """Run a single training experiment and return its results."""
    print(f"\n{'='*70}")
    print(f"  [{label}] Training {op} — {steps} steps")
    if gpu_config['gpu']:
        print(f"  GPU: {gpu_config['gpu_name']} ({gpu_config['vram_gb']}GB) | "
              f"Batch: {gpu_config['batch']} | AMP: {'ON' if gpu_config['amp'] else 'OFF'}")
    else:
        print(f"  CPU | Batch: {gpu_config['batch']}")
    print(f"{'='*70}")

    cmd = [sys.executable, 'train_specialist.py', op,
           '--steps', str(steps),
           '--batch', str(gpu_config['batch'])]
    if gpu_config['amp']:
        cmd.append('--amp')
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True, timeout=72000)
    elapsed = time.time() - t0

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    # Parse accuracy from stdout
    acc = 0.0
    for line in result.stdout.split('\n'):
        m = re.search(r'Best:\s*([\d.]+)%', line)
        if m:
            acc = max(acc, float(m.group(1)))
        m = re.search(r'best:\s*([\d.]+)%', line)
        if m:
            acc = max(acc, float(m.group(1)))

    print(f"  [{label}] Done in {elapsed/60:.1f} min — Best accuracy: {acc:.1f}%")
    return {
        'label': label,
        'op': op,
        'steps': steps,
        'accuracy': acc,
        'time_seconds': elapsed,
        'extra_args': extra_args or [],
        'exit_code': result.returncode,
    }


def compare_socratic(seeds: int, steps: int, ops: list[str] = None, gpu_config: dict = None):
    """Run standard vs Socratic comparison across seeds."""
    if ops is None:
        ops = ['add']

    if gpu_config is None:
        gpu_config = _detect_gpu()

    all_results = []

    for op in ops:
        print(f"\n{'#'*70}")
        print(f"  Benchmarking: {op.upper()} — {seeds} seeds × {steps} steps")
        if gpu_config['gpu']:
            print(f"  Hardware: {gpu_config['gpu_name']} ({gpu_config['vram_gb']}GB) × {gpu_config['gpu_count']}")
            print(f"  Config: batch={gpu_config['batch']}, AMP={'ON' if gpu_config['amp'] else 'OFF'}")
        print(f"{'#'*70}")

        for seed in range(1, seeds + 1):
            seed_label = f"Seed {seed}/{seeds}"

            # Standard training
            std_result = run_experiment(
                f"{seed_label} Standard",
                op, steps, gpu_config,
                extra_args=[] if seed == 1 else ['--steps', str(steps)]
            )
            std_result['variant'] = 'standard'
            std_result['seed'] = seed
            all_results.append(std_result)

            # Socratic training (post-training refinement)
            soc_result = run_experiment(
                f"{seed_label} Socratic",
                op, steps, gpu_config,
                extra_args=['--socratic', '--socratic-steps', '500', '--socratic-problems', '200']
            )
            soc_result['variant'] = 'socratic'
            soc_result['seed'] = seed
            all_results.append(soc_result)

    # Print results table
    print(f"\n{'='*70}")
    print(f"  RESULTS: Socratic vs Standard Training Comparison")
    print(f"{'='*70}")
    print(f"  {'Seed':<8} {'Variant':<12} {'Op':<6} {'Accuracy':<12} {'Time':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*6} {'-'*12} {'-'*12}")

    for r in all_results:
        mins = r['time_seconds'] / 60
        print(f"  {r['seed']:<8} {r['variant']:<12} {r['op']:<6} {r['accuracy']:<10.1f}% {mins:<10.0f}m")

    # Summarize
    std_accs = [r['accuracy'] for r in all_results if r['variant'] == 'standard']
    soc_accs = [r['accuracy'] for r in all_results if r['variant'] == 'socratic']

    if std_accs and soc_accs:
        avg_std = sum(std_accs) / len(std_accs)
        avg_soc = sum(soc_accs) / len(soc_accs)
        print(f"\n  Summary:")
        print(f"  Standard average: {avg_std:.1f}%")
        print(f"  Socratic average: {avg_soc:.1f}%")
        print(f"  Improvement: {avg_soc - avg_std:+.1f}%")

    # Save results JSON
    results_path = BASE / 'results' / 'socratic_comparison.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        'config': {'seeds': seeds, 'steps': steps, 'ops': ops},
        'results': all_results,
        'summary': {
            'standard_avg': round(sum(std_accs) / len(std_accs), 1) if std_accs else None,
            'socratic_avg': round(sum(soc_accs) / len(soc_accs), 1) if soc_accs else None,
            'improvement': round(sum(soc_accs) / len(soc_accs) - sum(std_accs) / len(std_accs), 1) if std_accs and soc_accs else None,
        }
    }, open(results_path, 'w'), indent=2)
    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Socratic vs Standard benchmark')
    parser.add_argument('--full', action='store_true', help='Full benchmark (5 seeds, 30000 steps)')
    parser.add_argument('--seeds', type=int, default=2, help='Number of seeds (default: 2)')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps per run (default: 5000)')
    parser.add_argument('--op', type=str, default='add', help='Operation to benchmark (default: add)')
    args = parser.parse_args()

    seeds = 5 if args.full else args.seeds
    steps = 30000 if args.full else args.steps

    gpu = _detect_gpu()
    sps = gpu['time_per_step']
    est_hours = seeds * 2 * steps / sps / 3600

    print(f"Socratic Benchmark — {seeds} seeds, {steps} steps, op={args.op}")
    if gpu['gpu']:
        print(f"  GPU: {gpu['gpu_name']} ({gpu['vram_gb']}GB) | "
              f"Batch: {gpu['batch']} | AMP: {'ON' if gpu['amp'] else 'OFF'}")
        print(f"  Estimated time: {est_hours:.1f} hours ({est_hours*60:.0f} min)")
        print(f"  ~{sps*steps:.0f} steps/s per run\n")
    else:
        print(f"  CPU (no GPU detected) | Estimated time: {est_hours:.1f} hours\n")

    compare_socratic(seeds, steps, [args.op], gpu)
