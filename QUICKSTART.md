# Quickstart — Tabula Rasa

A step-by-step guide for someone who has **never used this project before**.
Every command includes the expected output so you know it worked.

---

## Prerequisites

- Python 3.9 or newer
- Git
- 4GB free disk space
- No GPU required (CPU is fine)

**Check your Python version:**

```bash
python3 --version
# Expected: Python 3.9.x or higher
```

---

## Step 1: Get the Code

```bash
git clone https://github.com/tabula-rasa-ai/tabula-rasa.git
cd tabula-rasa
```

You should now be in a directory called `tabula-rasa/`.

---

## Step 2: Install Dependencies

```bash
pip install torch numpy tqdm
```

If you have a CUDA-capable GPU and want GPU training:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 3: Train Your First Specialist (30 seconds)

This trains a tiny addition model on 1-digit problems to verify everything works:

```bash
python3 train_specialist.py add --quick
```

**Expected output (last lines):**
```
  Step    200/200 | loss=0.0023 | lr=0.000001 | 0.8 st/s | ETA: 0m P1
    Eval step 200: 95.0% (best: 100.0%) | 1d:100% (acc<50%, skip per-digit)

  Done! 38s (0.6 min) | Best: 100.0% | Steps: 200
```

If you see `Done!` and a `Best` accuracy above 0%, it worked.

---

## Step 4: Test the Trained Model

```bash
python3 -c "
from pathlib import Path
import torch
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer

# Load the model you just trained
tok = MathTokenizer.load('specialists/math/add/tokenizer.json')
cfg = Config()
cfg.vocab_size = tok.vocab_size
model = MathTransformer(cfg)
state = torch.load('specialists/math/add/best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state['model_state_dict'])
model.eval()

# Ask it some problems
for expr in ['2+2=', '5+7=', '9+8=']:
    out = model.generate(tok, expr, max_new_tokens=10, temperature=0.0)
    print(f'{expr} -> {out}')
"
```

**Expected output:**
```
2+2= -> 2+2=04
5+7= -> 5+7=12
9+8= -> 9+8=17
```

The format `04` means carry=0, digit=4 → answer is 4.
The format `12` means carry=1, digit=2 → answer is 12.

---

## Step 5: Run the Dashboard (Optional)

```bash
python3 api_server.py
```

Open http://localhost:8000 in your browser. You'll see the Tabula Rasa dashboard
with training monitor, model config editor, experiment comparator, and more.

---

## What Just Happened?

1. A **transformer with random weights** was created (~1 million parameters)
2. It trained on **2,000 addition problems** using a special scratchpad format
3. The scratchpad decomposes each column into (carry, digit) pairs — the model
   learns carry propagation instead of just memorizing answers
4. After 200 steps (~30 seconds), it reached **100% accuracy on 1-digit addition**
5. The trained model and tokenizer are saved to `specialists/math/add/`

---

## Next Steps

| Goal | Command |
|------|---------|
| Train a full model (30K steps) | `python3 train_specialist.py add` |
| Train subtraction | `python3 train_specialist.py sub` |
| Train all 4 operations | `python3 train_specialist.py all` |
| Train with EWC (no forgetting) | `python3 train_specialist.py add --ewc` |
| Train with Socratic self-improvement | `python3 train_specialist.py add --socratic` |
| Quick ablation (no digit reversal) | `python3 train_specialist.py add --quick --no-reversed` |
| Run all tests | `python3 -m pytest tests/ -q` |
| View benchmarks | See [GPU_BENCHMARKS.md](GPU_BENCHMARKS.md) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Run: `pip install torch`

### `FileNotFoundError: No checkpoint found`

You skipped Step 3. Run:
```bash
python3 train_specialist.py add --quick
```

### Training is very slow

This is normal on CPU. A full 30K-step run takes ~9 hours on a 4-core laptop.
Use `--quick` for testing (200 steps, ~30 seconds). For GPU training, install
PyTorch with CUDA support (see Step 2).

### `AssertionError: d_model not divisible by n_heads`

Your config has incompatible model dimensions. Fix in `src/tabula_rasa/config.py`
or use the Model Config dashboard. Default values are correct.

---

## Need Help?

- [Telegram](https://t.me/TabulaRasaAi) — chat with the community
- [GitHub Issues](https://github.com/tabula-rasa-ai/tabula-rasa/issues) — report bugs
- [CONTRIBUTING.md](CONTRIBUTING.md) — developer setup guide
