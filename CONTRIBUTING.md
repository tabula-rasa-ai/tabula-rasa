# Contributing to Tabula Rasa AI

First off, thank you for considering contributing! Tabula Rasa is an experimental research project exploring how small transformers can learn from scratch. Every contribution — from bug reports to new specialist operations — moves the project forward.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style & Standards](#code-style--standards)
- [Adding a New Operation](#adding-a-new-operation)
- [Running Experiments](#running-experiments)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)

---

## Code of Conduct

This project is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md). All participants are expected to uphold its standards.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/tabula-rasa-ai/tabula-rasa.git
cd tabula-rasa

# Install dependencies
pip install torch numpy tqdm

# Train a specialist (smoke test — ~2 minutes on CPU)
python train_specialist.py add --quick

# Start the dashboard
python api_server.py
# Open http://localhost:8000
```

## Development Environment

### Requirements
- Python 3.9+
- PyTorch 2.0+ (CPU is fine for development)
- Git

### Recommended setup
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dev dependencies
pip install torch numpy tqdm
```

### Windows note
Double-click `start_tabula_rasa.bat` to start both servers and open the dashboard.

## Code Style & Standards

### Python
- **Formatting**: 4-space indentation, ~100 char lines
- **Imports**: stdlib → third-party → local, separated by blank lines
- **Types**: Type hints encouraged but not required for research code
- **Docstrings**: Google-style docstrings for public functions
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants

### Research code conventions
- **Reproducibility**: Always set `torch.manual_seed(42)` (or pass seed as arg)
- **Determinism**: Use `torch.use_deterministic_algorithms(True)` for core experiments
- **Logging**: Log to both stdout and a log file with timestamps
- **Results**: Save evaluation results as JSON with full experiment metadata

### Configuration
All hyperparameters live in `config.py`. Don't hardcode values in training scripts — derive from `Config()`:

```python
# Good
from config import Config
cfg = Config()
model = Transformer(cfg)

# Bad
model = Transformer(d_model=128, n_layers=4, ...)
```

## Adding a New Operation

The best way to contribute a new arithmetic operation is to follow the [Extending Tabula Rasa](CONTRIBUTING_EXTENDING.md) tutorial. In brief:

1. **Add dataset generation** in `dataset.py` (or a new file)
2. **Update the tokenizer** if your operation needs new tokens
3. **Train a specialist**: `python train_specialist.py myop`
4. **Add to the router**: register in `specialist_router.py`
5. **Verify**: run evaluation and save results as JSON

See `CONTRIBUTING_EXTENDING.md` for a complete walkthrough (factorial example).

## Running Experiments

### Smoke test
```bash
python train_specialist.py add --quick  # 500 steps, 1-2 minutes on CPU
```

### Full training
```bash
python train_specialist.py add           # 30K steps with curriculum
python train_specialist.py sub --ewc     # Train subtraction with EWC
python train_specialist.py mul --ewc     # Train multiplication with EWC
```

### Benchmarks
```bash
# Sequential retention test
python experiments/run_sequential_retention.py

# Ablation: no EWC control
python experiments/run_ablation_no_ewc.py

# Ablation: lambda sensitivity sweep
python experiments/run_ablation_lambda_sweep.py

# Ablation: three-task scalability
python experiments/run_ablation_three_task.py
```

### Auto-train loop
```bash
python auto_train.py                       # Train weakest until all >= 50%
python auto_train.py --target 70 --budget 10000
```

## Testing

Tests are in `test_partial.py` and individual experiment scripts under `experiments/`.

```bash
# Run existing tests
python test_partial.py

# Run an experiment as a test
python experiments/run_ablation_no_ewc.py
```

When adding new functionality, include:
- A self-contained test script or experiment runner
- Expected results documented in the script header
- JSON results output for cross-validation

### Writing a test
```python
"""Test for my new feature."""
import torch
from model import Transformer

def test_feature():
    model = Transformer.from_config()
    # ... test logic ...
    assert result == expected, f"Expected {expected}, got {result}"
```

## Pull Request Process

### Small fixes (typos, docs, 1-file changes)
1. Create a branch: `git checkout -b fix/my-fix`
2. Make your change
3. Commit: `git commit -m "fix: what you fixed"`
4. Push and open a PR

### New features or experiments
1. **Open an issue** first to discuss the approach
2. Create a branch: `git checkout -b feat/my-feature`
3. Implement with TDD where possible
4. Update docs (README, CONTRIBUTING if applicable)
5. Run at least one validation experiment
6. Commit with descriptive messages
7. Open a PR against `main`

### PR checklist
- [ ] Code follows style guidelines
- [ ] New code includes tests or experiment scripts
- [ ] RESULTS.md updated if claims change
- [ ] Documentation updated (README, CONTRIBUTING, etc.)
- [ ] All existing experiments still pass
- [ ] License headers added to new Python files (optional but appreciated)

### Commit style
```
type: short description

Longer explanation if needed. Use imperative mood.
```
Types: `feat`, `fix`, `docs`, `experiment`, `refactor`, `perf`, `test`, `chore`

## Project Structure

```
tabula-rasa/
├── api_server.py              # REST API + dashboard host
├── auto_train.py              # Autonomous training loop
├── model.py                   # Transformer model architecture
├── train_specialist.py        # Specialist trainer with EWC support
├── tokenizer.py               # Math tokenizer (carry-digit tokens)
├── config.py                  # All hyperparameters in one place
├── dataset.py                 # Problem generation
├── eval.py                    # Standalone evaluation
├── generate.py                # Standalone inference
│
├── egefalos/                  # Developmental AI system
│   ├── tabula_rasa.py         # Main entry point
│   ├── sleep_cycle.py         # Consolidation daemon
│   ├── mcts.py                # Monte Carlo Tree Search
│   ├── online_ewc.py          # Online Elastic Weight Consolidation
│   ├── hippocampus.py         # SQLite experience replay
│   ├── neocortex.py           # Neocortex module
│   ├── socratic_stage1.py     # Socratic dialogue (stage 1)
│   ├── socratic_stage2.py     # Socratic dialogue (stage 2)
│   ├── socratic_stage3.py     # Socratic dialogue (stage 3)
│   └── ...                    # Grammar, code, pattern specialists
│
├── Dashboard/                 # Web UI
│   ├── views/                 # Dashboard view templates
│   └── serve.py               # Dashboard server
│
├── experiments/               # Experiment runners + results
│   ├── run_ablation_no_ewc.py
│   ├── run_ablation_lambda_sweep.py
│   ├── run_ablation_three_task.py
│   └── ...
│
├── specialist_network.py      # Specialist lifecycle management
├── specialist_router.py       # Query routing
├── serve.py                   # Dashboard server
├── self_improve.py            # Self-play loop
│
├── start_tabula_rasa.bat      # Windows launcher
├── requirements.txt           # Dependencies
├── whitepaper.md              # Technical whitepaper
├── RESULTS.md                 # Validation results
├── ROADMAP.md                 # Project roadmap
├── CONTRIBUTING.md            # This file
├── LICENSE                    # MIT license
└── README.md                  # Project overview
```

## Key Design Decisions

### Why 1M parameters?
Enough to learn arithmetic, small enough to iterate fast on CPU. The architecture scales: increase `d_model`, `n_layers`, or `n_heads` in `config.py`.

### Why fused carry-digit tokens?
Each column's carry and digit are encoded as a single token (e.g., "04" = carry=0, digit=4). This gives the transformer aligned positional information, making carry propagation learnable at 1M params.

### Why Online EWC instead of standard EWC?
Online EWC merges Fisher matrices with exponential decay (γ=0.9), so the penalty term stays O(1) regardless of task count. Standard EWC stores one Fisher per task and sums them — O(N) memory. Online EWC is optimal within model capacity.

### Why CPU-first?
Accessibility. Tabula Rasa should be runnable by anyone with a laptop. GPU support exists (`--device cuda`) but all core claims are validated on CPU.

## Adding a New Tool to the Tool-Use System

Specialists can call external tools via `[[tool_name(args)]]` syntax. This is a great first contribution.

### Step 1: Write the tool function

In `egefalos/tool_use.py`, add a function:

```python
def get_weather(city: str) -> str:
    \"\"\"Get current weather for a city. Usage: [[get_weather(London)]]\"\"\"
    # Your implementation here
    return f"Sunny, 22°C in {city}"
```

### Step 2: Register it

Add your function to the `TOOL_REGISTRY` dict in the same file:

```python
TOOL_REGISTRY = {
    'calculator': calculator,
    'python': execute_python,
    'today': get_today,
    'random': get_random,
    'help': tool_help,
    'weather': get_weather,     # <-- your tool
}
```

### Step 3: Test it

```python
from egefalos.tool_use import ToolUse
tools = ToolUse()
result = tools.execute("What's the weather in [[weather(Tokyo)]]?")
print(result)  # "What's the weather in Sunny, 22°C in Tokyo?"
```

### Rules for tools
1. **Pure functions** — no side effects, same input = same output
2. **Short execution** — under 1 second (tools run during generation)
3. **Safe defaults** — never call destructive APIs without confirmation
4. **String in, string out** — tools receive and return strings (parsed from the `[[tool(args)]]` syntax)
5. **Nested parentheses** — automatically handled by the balanced-paren parser. `[[python(print(sum(range(10))))]]` works correctly

### Reference
- Tool parser: `egefalos/tool_use.py` — `_find_tool_calls()` uses balanced-paren parsing
- Tool example: `calculator` evaluates math expressions via `math_parser`
- Tool example: `python` runs code in a sandboxed subprocess

---

**Questions?** Open an issue or join [Telegram](https://t.me/TabulaRasaAi).
