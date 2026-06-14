"""Generate lambda sweep Pareto frontier plot from results JSON."""

import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: python3 -m pip install matplotlib")
    sys.exit(1)

# Load results
results_path = Path("experiments/ablation_lambda_sweep_results.json")
if not results_path.exists():
    print(f"No results found at {results_path}")
    print("Run experiments/run_ablation_lambda_sweep.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

lambdas = sorted([int(k) for k in data.keys()])
add_retention = [data[str(lam)]["addition_final_1d"] for lam in lambdas]
sub_learning = [data[str(lam)]["subtraction_final_1d"] for lam in lambdas]
add_baseline = [data[str(lam)]["addition_baseline_1d"] for lam in lambdas]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Individual curves
ax1.axhline(
    y=sum(add_baseline) / len(add_baseline),
    color="green",
    linestyle=":",
    alpha=0.4,
    label=f"Baseline avg ({sum(add_baseline)/len(add_baseline):.0f}%)",
)
ax1.plot(lambdas, add_retention, "g-o", linewidth=2.5, markersize=10, label="Addition (preserve)")
ax1.plot(lambdas, sub_learning, "r-s", linewidth=2.5, markersize=10, label="Subtraction (learn)")
ax1.set_xlabel("λ (EWC penalty weight)", fontsize=12)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("Two-Task Trade-off Curve", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11)
ax1.set_ylim(90, 102)
ax1.grid(True, alpha=0.3)
ax1.set_xscale("log")
ax1.set_xticks(lambdas)
ax1.set_xticklabels([str(l) for l in lambdas])

# Right: Pareto frontier
ax2.plot(add_retention, sub_learning, "b-o", linewidth=2.5, markersize=10)
for i, lam in enumerate(lambdas):
    ax2.annotate(
        f"λ={lam}",
        (add_retention[i], sub_learning[i]),
        fontsize=10,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )
ax2.set_xlabel("Addition Retention (%)", fontsize=12)
ax2.set_ylabel("Subtraction Learning (%)", fontsize=12)
ax2.set_title("Pareto Frontier (2 tasks)", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(93, 100)
ax2.set_ylim(93, 102)

fig.suptitle(
    "Lambda EWC Sensitivity: Pareto Frontier is Flat for Two Tasks",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    r"C:\Users\Admin\tabula-rasa\experiments\lambda_sweep_pareto.png", dpi=150, bbox_inches="tight"
)
print("SAVED: experiments/lambda_sweep_pareto.png")

# Print summary table
print()
print("=== LAMBDA SWEEP SUMMARY ===")
print(f'  {"λ":>6} | {"Baseline":>9} | {"Add Final":>10} | {"Sub Final":>10}')
print(f'  {"-"*6}-+-{"-"*9}-+-{"-"*10}-+-{"-"*10}')
for i, lam in enumerate(lambdas):
    print(
        f"  {lam:>6} | {add_baseline[i]:>8.0f}% | {add_retention[i]:>9.0f}% | {sub_learning[i]:>9.0f}%"
    )

print()
flat = max(sub_learning) - min(sub_learning) < 5
print(f'Pareto frontier is {"FLAT ✅" if flat else "HAS A CURVE"}')
print(f"  Add retention range: {min(add_retention):.0f}%-{max(add_retention):.0f}%")
print(f"  Sub learning range:  {min(sub_learning):.0f}%-{max(sub_learning):.0f}%")
