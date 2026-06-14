---
title: Tabula Rasa — Math Solver
emoji: 🧮
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 5.0
app_file: app.py
pinned: true
license: mit
short_description: A transformer that learned arithmetic from scratch
---

# Tabula Rasa — Interactive Math Demo

A transformer trained from a **blank slate** to solve arithmetic.
No pretraining. No transfer learning. No symbolic algebra. Just pure gradient descent.

## How to Deploy

### Option 1: One-Click (HF Spaces)

1. Push the `app.py` and `hf_space/requirements.txt` to a Hugging Face Space
2. Set `HF_MODEL_REPO` to your pushed hub repo (e.g. `tabula-rasa-ai/add-specialist`)
3. The Space auto-builds and deploys

### Option 2: Local

```bash
pip install gradio torch huggingface_hub
python app.py
```

Opens at `http://localhost:7860`

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_MODEL_REPO` | Load model from HF Hub (e.g. `tabula-rasa-ai/add-specialist`) |
| `LOCAL_CKPT` | Path to a local .pt checkpoint (falls back to auto-detect) |

## Usage

1. Type or paste a math expression: `12+34=`, `56-78=`, `9*8=`, `100/5=`
2. Press Enter or click **Solve**
3. See the result with timing metadata
4. Click **Random Problem** to test the model

Toggle **Show Scratchpad** to see the model's internal carry-digit trace
(for specialists trained with scratchpad format).
