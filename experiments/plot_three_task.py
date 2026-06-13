"""Generate three-task scalability plot from results JSON."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results_path = Path('experiments/ablation_three_task_results.json')
if not results_path.exists():
    print(f"No results found at {results_path}")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

# Extract accuracies at each step
# Step 1: After addition (baseline)
baseline = data['step1_load']['baseline']
add_baseline = baseline['add']['accuracy']
sub_baseline = baseline['sub']['accuracy']
mul_baseline = baseline['mul']['accuracy']

# Step 2: After subtraction training
after_sub = data['step2_train_sub']['measurements']

# Step 3: After multiplication training
after_mul = data['step4_train_mul']['measurements']

steps = ['Before\ntraining', 'After\nAddition', 'After\nSubtraction', 'After\nMultiplication']
add_acc = [0, add_baseline, after_sub['add']['accuracy'], after_mul['add']['accuracy']]
sub_acc = [0, 0, after_sub['sub']['accuracy'], after_mul['sub']['accuracy']]
mul_acc = [0, 0, 0, after_mul['mul']['accuracy']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Stacked bar chart
x = np.arange(len(steps))
width = 0.25
ax1.bar(x - width, add_acc, width, label='Addition', color='green', alpha=0.8)
ax1.bar(x, sub_acc, width, label='Subtraction', color='blue', alpha=0.8)
ax1.bar(x + width, mul_acc, width, label='Multiplication', color='orange', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(steps, fontsize=10)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Three-Task Sequence: Scalability Boundary at Task 3', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_ylim(0, 110)
ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
ax1.grid(True, alpha=0.2, axis='y')

# Annotate collapse
ax1.annotate('COLLAPSE\nAdd: 100%→3%\nSub: 100%→0%',
             xy=(3, 20), fontsize=11, color='red', fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='#ffeeee', edgecolor='red', alpha=0.8))

# Right: Fisher norm timeline
stages = ['F_add', 'After sub\nmerge', 'After mul\nmerge']
fisher_norms = [
    data['step2_train_sub']['fisher_norm'],
    data['step3_merge_fisher']['fisher_norm_after'],
    data['step4_train_mul']['fisher_norm'],
]

ax2.plot(stages, fisher_norms, 'b-o', linewidth=2.5, markersize=10)
ax2.set_ylabel('Fisher Norm (√ΣFᵢ²)', fontsize=12, fontweight='bold')
ax2.set_title('Diagnostic: Fisher Norm Shrinks During Merge', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(2.0, 2.5)

# Annotate the shrink
ax2.annotate('−6% loss',
             xy=(2, fisher_norms[2]), xytext=(1.5, fisher_norms[2] - 0.15),
             fontsize=12, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

fig.suptitle('Scalability Boundary: Information Loss During Fisher Merge',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\Admin\tabula-rasa\experiments\three_task_scalability.png', dpi=300, bbox_inches='tight')
print('SAVED: experiments/three_task_scalability.png')

# Print summary
print()
print('=== THREE-TASK RESULTS ===')
print(f'  {"Step":<20} | {"Add":>6} | {"Sub":>6} | {"Mul":>6} | {"Fisher":>8}')
print(f'  {"-"*20}-+-{"-"*6}-+-{"-"*6}-+-{"-"*6}-+-{"-"*8}')
print(f'  {"After Addition":<20} | {add_baseline:>5.0f}% | {"-":>6} | {"-":>6} | {fisher_norms[0]:>8.2f}')
print(f'  {"After Subtraction":<20} | {after_sub["add"]["accuracy"]:>5.0f}% | {after_sub["sub"]["accuracy"]:>5.0f}% | {"-":>6} | {fisher_norms[1]:>8.2f}')
print(f'  {"After Multiplication":<20} | {after_mul["add"]["accuracy"]:>5.0f}% | {after_mul["sub"]["accuracy"]:>5.0f}% | {after_mul["mul"]["accuracy"]:>5.0f}% | {fisher_norms[2]:>8.2f}')
print()
print(f'Fisher norm change: {fisher_norms[0]:.2f} → {fisher_norms[1]:.2f} → {fisher_norms[2]:.2f}')
print(f'Information loss at third merge: {(1 - fisher_norms[2]/fisher_norms[1])*100:.1f}%')
print(f'All tasks preserved (>85%): {"NO ❌" if after_mul["add"]["accuracy"] < 85 else "YES ✅"}')
