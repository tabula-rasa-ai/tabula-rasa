"""Fix hardcoded API ports in dashboard HTML files to be dynamic."""
import re
from pathlib import Path

dash_dir = Path("Dashboard")
views_dir = dash_dir / "views"
core_dir = dash_dir / "core"

files_to_fix = [
    # (path, pattern, replacement)
    *[(core_dir / "dashboard.html", r"'http://localhost:8002'",
       "window.location.protocol + '//' + window.location.hostname + ':' + window.location.port"),
      ],
    *[(views_dir / f, r"'http://localhost:8000'",
       "window.location.protocol + '//' + window.location.hostname + ':' + window.location.port")
      for f in [
          "attention_inspector.html", "checkpoint_manager.html",
          "gpu_training.html", "log_viewer.html", "loss_curve_analyzer.html",
          "mcts_tree.html", "model_config.html",
      ]],
]

# Special case: experiment_comparator.html already has a smart pattern
for path, pattern, replacement in files_to_fix:
    if not path.exists():
        print(f"  SKIP {path} (not found)")
        continue
    text = path.read_text(encoding="utf-8")
    old_count = len(re.findall(re.escape(pattern), text))
    if old_count == 0:
        print(f"  SKIP {path.name} (pattern not found)")
        continue
    new_text = text.replace(pattern, replacement)
    path.write_text(new_text, encoding="utf-8")
    print(f"  FIX  {path.name}: {old_count} occurrence(s)")

print("\nDone.")
