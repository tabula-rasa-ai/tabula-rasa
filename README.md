# Tabula Rasa AI

**Learning from scratch, one specialist at a time.**

Tabula Rasa is an experimental AI system that trains small transformer models from a blank slate to master arithmetic, language, and reasoning skills. Each skill is a dedicated specialist model trained independently, with a router that knows what it knows and says "I don't know" for what it doesn't.

```
pip install torch numpy tqdm
python3 train_specialist.py add
python3 api_server.py
```

Then open http://localhost:8000

---

## Core Philosophy

Most AI systems are giant monolithic models trained on everything at once. Tabula Rasa takes the opposite approach:

- **Specialize first** — each skill gets its own dedicated transformer (~1M params)
- **Graduate, don't accumulate** — specialists are frozen once they pass 98% accuracy
- **Route by capability** — a neural router dispatches problems to the right specialist

This is loosely inspired by Piaget's stages of cognitive development: infants learn object permanence before counting, counting before arithmetic, arithmetic before algebra.

## Architecture

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

- **Parameters**: ~1M (128-dim, 4 layers, 4 heads, 512 FF)
- **Context**: 32 tokens (enough for scratchpad arithmetic)
- **Position encoding**: RoPE
- **Normalization**: RMSNorm
- **Activation**: ReLU

## What It Can Do

### Arithmetic (mature)

Trainable specialists for:
- **Addition** — multi-digit with carry propagation
- **Subtraction** — multi-digit with borrowing
- **Multiplication** — multi-digit
- **Division** — integer division

Key training techniques:
- **Reversed digits** — aligns carry positions for easier learning
- **Loss masking** — ignores prompt/pad tokens during training
- **Curriculum learning** — starts with 1-digit, progresses to 4-digit
- **Scratchpad** — model shows carry steps in output
- **Hard negatives** — focuses training on problems the model gets wrong

### AlphaZero Self-Play (experimental)

- **Language AlphaZero** (`run_language_az.py`) — self-play language acquisition via MCTS
- **Code AlphaZero** (`run_code_az.py`) — self-play programming in a Python sandbox

### Cognitive Stages (egefalos/)

The Piaget-inspired developmental system builds skills in stages:

| Stage | Skills | 
|-------|--------|
| **Infancy (S1-2)** | Math, Pattern recognition |
| **Toddlerhood (S3-4)** | Taxonomy, State tracking |
| **Childhood (S5)** | Grammar, Syntax |
| **Adulthood (S6)** | Dialogue, Complex reasoning |

Source: `egefalos/` directory — `tabula_rasa.py` is the main entry point for the full system.

## Getting Started

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.60.0
```

### Train a specialist

```bash
python3 train_specialist.py add        # Train addition
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

The auto-trainer evaluates all operations, finds the weakest below target, trains it, and repeats.

### Start the dashboard

```bash
python3 api_server.py          # API on port 8000
```

Then open http://localhost:8000 in your browser.

### Windows quick start

Double-click `start_tabula_rasa.bat` — it starts both servers and opens the dashboard.

### Full system

```bash
python3 egefalos/tabula_rasa.py                # Port 8002
python3 egefalos/tabula_rasa.py --learn biology # Queue a new skill
```

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
| P2P Swarm | Peer-to-peer specialist exchange |
| Full list | `Dashboard/views/` |

## Configuration

Everything is in `config.py` — edit directly or use the Model Config dashboard:

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

The dashboard's Model Config view lets you change any parameter via the API (`GET/POST /api/config`) without touching the file.

## API

The API server runs on port 8000:

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
| `/api/registry` | GET | P2P registry |
| `/api/chain-status` | GET | Chain training status |
| `/api/system-resources` | GET | CPU/memory/process stats |

## Project Structure

```
tabula-rasa/
├── api_server.py          # REST API + dashboard host
├── auto_train.py          # Autonomous training loop
├── model.py               # Transformer model
├── train_specialist.py    # Specialist trainer
├── tokenizer.py           # Math tokenizer
├── bpe_tokenizer.py       # BPE tokenizer (experimental)
├── config.py              # All hyperparameters
├── config.py.bak          # Config backup
├── dataset.py             # Problem generation
├── eval.py                # Standalone evaluation
├── generate.py            # Standalone inference
│
├── egefalos/              # Cognitive stage system
│   ├── tabula_rasa.py     # Main entry point
│   ├── piaget/            # Piagetian stages
│   │   ├── sensorimotor.py
│   │   ├── preoperational.py
│   │   ├── concrete_operational.py
│   │   └── formal_operational.py
│   ├── sleep_cycle.py     # Consolidation (memory replay)
│   ├── mcts.py            # Monte Carlo Tree Search
│   ├── language_az.py     # Language AlphaZero
│   ├── code_curriculum.py
│   ├── code_sandbox.py    # Python sandbox for self-play
│   └── ...                # Specialist modules
│
├── Dashboard/             # Web UI
│   ├── dashboard.html     # Main layout
│   ├── core/              # Shared JS/CSS
│   └── views/             # 25+ view panels
│
├── specialist_network.py  # Specialist lifecycle management
├── specialist_router.py   # Query routing
├── export_specialist.py   # Export/import flow
│
├── serve.py               # Standalone dashboard server
├── self_improve.py        # Self-play improvement loop
├── scripts/               # Utility scripts
├── train.py / train_gpu.py / train_optimized.py  # Training variants
├── run_stage.py           # Stage runner
├── run_language_az.py     # Language AlphaZero runner
├── run_code_az.py         # Code AlphaZero runner
│
├── start_tabula_rasa.bat  # Windows launcher
├── setup_gpu.sh           # GPU setup
├── requirements.txt       # Dependencies
└── whitepaper.md          # Technical whitepaper
```

## Development

Tabula Rasa is an active research project. Areas of exploration:

- **Scaling** — larger models, more operations, harder curriculum
- **Transfer learning** — can addition help with multiplication?
- **Sleep cycles** — offline consolidation via memory replay
- **P2P specialist exchange** — share trained specialists between instances
- **Self-play** — AlphaZero-style improvement without human data
- **Graduation** — 98% accuracy = frozen, never overwritten

---

[GitHub](https://github.com/tabula-rasa-ai/tabula-rasa) &bull; [Telegram](https://t.me/TabulaRasaAi) &bull; [Whitepaper](whitepaper.md)
