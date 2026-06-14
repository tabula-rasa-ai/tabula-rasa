# Good First Issues

Welcome! These are small, well-defined tasks perfect for your first contribution to Tabula Rasa. Each takes 1-3 hours and helps you learn the codebase.

## How to Claim an Issue

1. Pick a task below
2. Open a new issue with the task title and `good first issue` label
3. Comment on it to let others know you're working on it
4. Submit a PR following CONTRIBUTING.md

---

## Task 1: Add Exponentiation Operation

**Difficulty:** Beginner | **Files:** 4 | **Estimate:** 2 hours

Add a `pow` (exponentiation `^`) operation following the guide in `CONTRIBUTING_EXTENDING.md`.

**What to do:**
- Add `'^'` to `OPERATIONS` in `src/tabula_rasa/dataset.py`
- Add `'^'` to `MATH_CHARS` in `src/tabula_rasa/tokenizer.py`
- Add `'pow': '^'` to `OPS` in `train_specialist.py`
- Add `'pow': '^'` to `OPS` in `specialist_router.py`
- Train: `python train_specialist.py pow --quick`

**Verification:** `curl -X POST http://localhost:8000/generate -d '{"prompt":"2^3="}'` returns `"8"`

**Reference:** `CONTRIBUTING_EXTENDING.md` (complete factorial tutorial)

---

## Task 2: Add Unit Test for a Module

**Difficulty:** Beginner | **Files:** 1-2 | **Estimate:** 1 hour

Pick a module without full test coverage and add tests.

**Modules needing tests:**
- `src/tabula_rasa/generate.py` — test `load_model()` with mock checkpoint
- `egefalos/sleep_cycle.py` — test consolidation loop logic
- `egefalos/neocortex.py` — test memory consolidation functions
- `specialist_router.py` — test `detect_operation()` and `SpecialistManager`

**What to do:**
- Add a new test file in `tests/` (e.g., `tests/test_generate.py`)
- Test 3-5 behaviors using pytest
- Run: `pytest tests/test_*.py -v`

**Reference:** Existing tests in `tests/test_tokenizer.py`, `tests/test_model.py`

---

## Task 3: Add Dashboard Visualization

**Difficulty:** Intermediate | **Files:** 1-2 | **Estimate:** 2 hours

Add a new visualization to the web dashboard. The dashboard is at `Dashboard/views/`.

**Ideas:**
- **Loss landscape view** — 2D contour plot of loss across model dimensions
- **Learning rate schedule viewer** — visualize the cosine/warmup schedule
- **Specialist comparison radar chart** — compare accuracy across digit lengths

**What to do:**
- Create `Dashboard/views/new_view.html` (look at existing views for template style)
- Add a nav link in `Dashboard/core/dashboard.html`
- The view calls `API + '/api/config'` and `/training-progress` for data

**Reference:** `Dashboard/views/loss_curve_analyzer.html`, `Dashboard/views/experiment_comparator.html`

---

## Task 4: Add Scheduler Visualizer to Dashboard

**Difficulty:** Intermediate | **Files:** 1 | **Estimate:** 1.5 hours

Create a dashboard view that plots the LR schedule before training starts.

**What to do:**
- Read `config.py` to get `learning_rate`, `warmup_steps`, `max_steps`, `lr_schedule`
- Plot the scheduled LR values across all training steps
- Show separate curves for cosine, linear, and constant schedules
- Let the user adjust parameters and see the curve update live

---

## Task 5: Add --dry-run Flag to train_specialist.py

**Difficulty:** Beginner | **Files:** 1 | **Estimate:** 1 hour

Add a `--dry-run` flag that prints what would happen without actually training.

**What to do:**
- Add `--dry-run` to argparse
- When set, print the model architecture, dataset size, and training plan
- Exit without training

**Verification:** `python train_specialist.py add --quick --dry-run` prints plan and exits

---

## Task 6: Profile Training Speed

**Difficulty:** Beginner | **Files:** 1 | **Estimate:** 1 hour

Create a script that profiles training speed across different configs.

**What to do:**
- Create `experiments/profile_training.py`
- Test training speed at d_model=[32, 64, 128] and n_layers=[1, 2, 4]
- Print a table of steps/second for each config
- Use `torch.cuda.Event` or `time.perf_counter()` for timing

**Reference:** `experiments/run_benchmarks.py`

---

## Task 7: Add MPS (Apple Silicon) Device Support

**Difficulty:** Intermediate | **Files:** 2-3 | **Estimate:** 2 hours

Add `mps` as a recognized device option for Apple Silicon Macs.

**What to do:**
- Update `config.py` `device` property: try `torch.backends.mps.is_available()` after CUDA
- Update `train_specialist.py` device detection: `'mps' if torch.backends.mps.is_available() else 'cpu'`
- Test on an Apple Silicon Mac

---

## Task 8: Export Benchmark Results as Markdown

**Difficulty:** Beginner | **Files:** 1 | **Estimate:** 1 hour

Add a `--md` flag to `experiments/run_benchmarks.py` that outputs results as a markdown table suitable for pasting into README.md or RESULTS.md.

**What to do:**
- Add argparse `--md` flag
- When set, format the JSON results as a markdown table
- Print to stdout

---

## Task 9: Fix a TODO in the Codebase

**Difficulty:** Varies | **Files:** 1-3 | **Estimate:** 1-2 hours

Search the codebase for `# TODO` or `# FIXME` comments and fix one.

**To find them:**
```bash
grep -rn "TODO\|FIXME" src/ egefalos/ experiments/ --include="*.py"
```

---

## Task 10: Add a New Dashboard Theme

**Difficulty:** Beginner | **Files:** 1 | **Estimate:** 1 hour

Add a light theme toggle to the dashboard.

**What to do:**
- In `Dashboard/core/dashboard.html`, add a CSS variables swap
- Create a light color scheme: `--bg: #ffffff`, `--text: #1e293b`, etc.
- Add a toggle button in the status bar
- Save preference to `localStorage`
