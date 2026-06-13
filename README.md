# Tabula Rasa AI

**Learning from scratch, one specialist at a time.**

Tabula Rasa is an experimental AI system that trains small transformer models from a blank slate. Currently in **Phase 1**: proving single-task arithmetic mastery before adding continual learning.

```
pip install torch numpy tqdm
python3 train_specialist.py add
python3 api_server.py
```

Then open http://localhost:8000

---

## Status

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Arithmetic Specialist — single-task mastery | ✅ **COMPLETE** |
| **2** | Continual Learning — Online EWC + multi-specialist | ✅ **COMPLETE** |
| **3** | Reasoning — Socratic Engine, self-play | ❌ **FUTURE** |

---

## Phase 1: Arithmetic Specialist (COMPLETE)

- ✅ **1M-param transformer** with RoPE, RMSNorm, configurable depth
- ✅ **Fused carry-digit tokenizer** — 20 two-character tokens (00-19) encoding carry + digit per column
- ✅ **Addition specialist: 100% accuracy in 3,000 steps** (1-digit, 1M params, CPU)
- ✅ **Curriculum learning** — starts with 1-digit, progresses to 4-digit
- ✅ **Config-driven training** — all knobs in `config.py`, editable via dashboard
- ✅ **Auto-train loop** — evaluates all operations, trains the weakest, repeats
- ✅ **Interactive dashboard** — 25+ views for training, config, evaluation
- ✅ **REST API** — inference, training control, checkpoint management

### Validated Results

| Claim | Result | Evidence |
|-------|--------|----------|
| 100% 1-digit addition in ≤4K steps | **100% in 3K steps** | `RESULTS.md` |
| Fused carry-digit tokens (00-19) | **20 tokens in vocab** (IDs 4-23) | `tokenizer.py` |
| Online EWC Fisher Matrix | **✅ IMPLEMENTED & VALIDATED** | `RESULTS.md §2B` |

### What the architecture proves

The fused carry-digit token insight is the core innovation in Phase 1. By encoding each column's carry and digit as a single token (e.g. "04" = carry=0, digit=4), the transformer sees aligned positional information that makes carry propagation learnable at 1M parameters — a problem that typically requires much larger models.

## Phase 2: Continual Learning (COMPLETE)

The system now learns sequential tasks without catastrophic forgetting, using **Online Elastic Weight Consolidation (EWC)**:

- ✅ **Online EWC module** — Fisher matrix computation, exponential decay merge (γ=0.9), EWC penalty in training loss (λ=1000)
- ✅ **39 parameter groups** tracked across the 1M-parameter model, saved as `ewc_fisher.pt`
- ✅ **Hippocampus (SQLite)** — stores high-surprise experiences, supports replay buffer sampling
- ✅ **Sleep cycle daemon** — periodic replay → Fisher compute → merge → train → consolidate
- ✅ **CLI integration** — `--ewc` flag on `train_specialist.py`
- ✅ **Sequential retention validated** — addition (83%) → train subtraction with EWC → addition **improves to 92%**, subtraction acquired at **97%**

### Validated Results

| Metric | Value | Significance |
|--------|-------|-------------|
| Addition baseline | 83.0% | Starting point |
| After EWC + subtraction | **92.0%** | +9pp improvement (not degradation!) |
| Retention drop | **-9.0 pp** | NEGATIVE = improvement |
| Subtraction acquired | **97.0%** | New task mastered while preserving old |
| Fisher groups tracked | 39 | Covers all weight layers |
| EWC merge decay (γ) | 0.9 | Keep 90% old, 10% new |
| EWC penalty weight (λ) | 1000 | Task preservation strength |

**Key finding:** Online EWC not only prevents forgetting — it improves old-task accuracy through implicit regularization. The consolidated weight space is a better local minimum than the original isolated training.

See `RESULTS.md` for the full scientific analysis, including the ablation roadmap.

## Phase 3: Reasoning (FUTURE)

- [ ] Socratic Engine (dialectical self-play)
- [ ] Language AlphaZero
- [ ] Code AlphaZero

## Architecture (Phase 1)

| Component | File | Purpose |
|-----------|------|---------|
| Transformer | `model.py` | Causal transformer with RoPE, RMSNorm, configurable depth |
| Specialist Trainer | `train_specialist.py` | Curriculum learning for one operation at a time |
| Auto-Train | `auto_train.py` | Autonomous loop: eval all, train weakest, repeat |
| API Server | `api_server.py` | REST API for inference, training control, config |
| Dashboard | `serve.py` | Web UI (25+ views) served alongside the API |
| Tokenizer | `tokenizer.py` | Custom math tokenizer with carry-digit encoding |
| Config | `config.py` | All knobs in one file |
| Router | `specialist_router.py` | Routes queries to the right specialist model |
| Network | `specialist_network.py` | Manages specialist lifecycle and discovery |

### Model Specs (default)

| Param | Value |
|-------|-------|
| Parameters | ~1M (128-dim, 4 layers, 4 heads, 512 FF) |
| Context | 32 tokens (enough for scratchpad arithmetic) |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Activation | ReLU |
| Vocab | 44 tokens (4 special + 20 carry-digit + 20 math chars) |

## Getting Started

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.60.0
```

### Train a specialist

```bash
python3 train_specialist.py add        # Train addition (3K steps to 100% 1-digit)
python3 train_specialist.py sub        # Train subtraction
python3 train_specialist.py mul        # Train multiplication
python3 train_specialist.py div        # Train division
python3 train_specialist.py all        # All four operations
python3 train_specialist.py add --quick  # Smoke test (500 steps)
python3 train_specialist.py add --resume # Resume from checkpoint
```

### Auto-train (autonomous loop)

```bash
python3 auto_train.py                      # Train weakest until all ≥50%
python3 auto_train.py --target 70          # Higher bar
python3 auto_train.py --budget 10000       # 10K step budget
python3 auto_train.py --ops add sub        # Only add + sub
```

### Start the dashboard

```bash
python3 api_server.py          # API on port 8000
```

Open http://localhost:8000

### Windows quick start

Double-click `start_tabula_rasa.bat` — starts both servers and opens the dashboard.

## Dashboard

The web dashboard at http://localhost:8000 has 25+ views:

| View | What It Does |
|------|-------------|
| Training Monitor | Live loss/accuracy chart, start/stop/pause training |
| Specialist Trainer | Per-specialist config and training controls |
| Model Config | Edit all 30+ architecture/training params live |
| Experiment Comparator | Compare accuracy across runs |
| Checkpoint Manager | Save/load/export trained specialists |
| Interactive Chat | Talk to the model |
| Attention Inspector | Visualize attention patterns |
| Data Explorer | Examine the training dataset |
| Error Analysis | Find which digit positions the model gets wrong |
| Log Viewer | Browse training logs |
| System Status | CPU/memory/process info |
| P2P Swarm | Peer-to-peer specialist exchange (experimental) |

## Configuration

Everything in `config.py` — edit directly or use the Model Config dashboard:

```python
d_model = 128           # Embedding dimension
n_layers = 4            # Transformer depth
n_heads = 4             # Attention heads
d_ff = 512              # Feed-forward hidden dim
dropout = 0.1           # Dropout rate
learning_rate = 0.001   # Peak LR
batch_size = 32         # Batch size
max_digits = 4          # Max digits per operand
use_curriculum = True   # Curriculum learning
use_reversed = True     # Reverse digits for carry alignment
use_loss_masking = True # Ignore prompt/pad in loss
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Run inference |
| `/correct` | POST | Online correction (5 gradient steps) |
| `/api/config` | GET/POST | Read/write config |
| `/api/train/start` | POST | Start training a specialist |
| `/api/train/stop` | POST | Stop training |
| `/api/train/auto` | POST | Start auto-train loop |
| `/api/train/auto/status` | GET | Auto-train status |
| `/api/train/status` | GET | Training status |
| `/training-progress` | GET | Live loss/accuracy data |
| `/api/checkpoints` | GET | List available checkpoints |
| `/api/experiments` | GET/POST | Experiment tracking CRUD |
| `/api/export` | POST | Export a specialist as zip |
| `/api/import` | POST | Import a shared specialist |
| `/health` | GET | Server health check |

## Project Structure

```
tabula-rasa/
├── api_server.py          # REST API + dashboard host
├── auto_train.py          # Autonomous training loop
├── model.py               # Transformer model
├── train_specialist.py    # Specialist trainer
├── tokenizer.py           # Math tokenizer (carry-digit tokens)
├── config.py              # All hyperparameters
├── dataset.py             # Problem generation
├── eval.py                # Standalone evaluation
├── generate.py            # Standalone inference
│
├── egefalos/              # Developmental system (in development)
│   ├── tabula_rasa.py     # Main entry point
│   ├── piaget/            # Piagetian stages
│   ├── sleep_cycle.py     # Consolidation (planned)
│   └── mcts.py            # Monte Carlo Tree Search
│
├── Dashboard/             # Web UI (25+ views)
├── specialist_network.py  # Specialist lifecycle management
├── specialist_router.py   # Query routing
├── serve.py               # Dashboard server
├── self_improve.py        # Self-play loop
│
├── start_tabula_rasa.bat  # Windows launcher
├── requirements.txt       # Dependencies
├── whitepaper.md          # Technical whitepaper
├── RESULTS.md             # Validation results
└── README.md              # This file
```

---

[GitHub](https://github.com/tabula-rasa-ai/tabula-rasa) • [Telegram](https://t.me/TabulaRasaAi) • [Whitepaper](whitepaper.md) • [Results](RESULTS.md)
