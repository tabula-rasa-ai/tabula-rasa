# GPU Benchmarks — Tabula Rasa

Performance benchmarks for Tabula Rasa specialist models across different
hardware configurations. Run these before and after architecture changes to
quantify impact.

## Current Architecture (Default)

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_layers | 4 |
| n_heads | 4 |
| d_ff | 512 |
| Parameters | ~1,060,992 |
| Vocabulary | 44 tokens (fused carry-digit) |
| Max seq len | 32 |

## Benchmark Results

### CPU (Intel N150, 4 cores)

| Config | Steps/s | Time to 30K steps | Notes |
|--------|---------|-------------------|-------|
| RoPE + ReLU (default) | ~0.9 | ~9 hours | Baseline, threads=6 |
| RoPE + SwiGLU | ~0.9 | ~9 hours | Same throughput |
| Learned pos encoding | ~0.2 | ~42 hours | Learned embedding overhead |
| batch_size=128 | ~0.9 | ~9 hours | Fewer DataLoader resets |
| batch_size=32 | ~0.9 | ~9 hours | Same throughput, more epochs |
| torch.compile() | ~0.5 | ~17 hours | 0.83x slowdown on CPU |

### CUDA GPU (Expected — run for your hardware)

| GPU | Config | Steps/s | Time to 30K | Notes |
|-----|--------|---------|-------------|-------|
| RTX 4090 | batch=256, AMP | ~2,000 | ~15s | FlashAttention supported |
| RTX 3090 | batch=256 | ~800 | ~38s | |
| RTX 3060 | batch=128 | ~400 | ~75s | |
| T4 (Colab) | batch=128 | ~200 | ~2.5min | |

### Apple Silicon (MPS)

| Chip | Config | Steps/s | Notes |
|------|--------|---------|-------|
| M1 | batch=32 | ~8 | MPS backend |
| M2 | batch=64 | ~15 | |
| M3 Max | batch=128 | ~40 | |

## How to Run Benchmarks

### Full training benchmark

```bash
# CPU benchmark (default)
python3 train_specialist.py add --steps 5000 --log-level INFO

# GPU benchmark
CUDA_VISIBLE_DEVICES=0 python3 train_specialist.py add --steps 5000 \
    --batch 256 --amp --log-level INFO
```

### Quick throughput test (100 steps)

```bash
python3 -c "
import sys, time, torch; sys.path.insert(0, 'src')
from tabula_rasa.model import MathTransformer
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer

cfg = Config(); cfg.vocab_size = 44
model = MathTransformer(cfg)
tok = MathTokenizer()
tok.max_seq_len = cfg.max_seq_len

x = torch.randint(0, 44, (cfg.batch_size, 16))
y = torch.randint(0, 44, (cfg.batch_size, 16))

# Warmup
for _ in range(5): model(x, y)

# Benchmark
t0 = time.time()
for _ in range(100): model(x, y)
elapsed = time.time() - t0
print(f'Throughput: {100/elapsed:.1f} steps/s (batch={cfg.batch_size})')
print(f'Memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB' if torch.cuda.is_available() else '')
"
```

### Eval-only benchmark

```bash
python3 -c "
import sys, time; sys.path.insert(0, 'src')
from evaluate import evaluate
from tabula_rasa.generate import load_model

model, tok = load_model('specialists/math/add/best.pt')
from tabula_rasa.config import Config
cfg = Config(); cfg.vocab_size = tok.vocab_size

t0 = time.time()
for _ in range(10): acc = evaluate(model, tok, cfg, 'add', num=100)
elapsed = time.time() - t0
print(f'Eval throughput: {1000/elapsed:.0f} samples/s (avg {(elapsed/10)*1000:.0f}ms per 100-sample eval)')
"
```

## FlashAttention Support

FlashAttention is used automatically when:
1. `torch>=2.0` with CUDA is available (PyTorch's native SDPA uses it)
2. The model runs on CUDA with compatible GPU (compute capability 7.5+)
3. `mask=None` is passed (enables `is_causal=True` in SDPA)

To verify FlashAttention is active:

```python
import torch.backends.cuda
print(f'FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}')
print(f'Mem-efficient attention: {torch.backends.cuda.mem_efficient_sdp_enabled()}')
```

## MPS (Apple Silicon) Notes

- MPS backend supports the full model: forward, backward, generation
- AMP (mixed precision) is disabled on MPS (GradScaler not supported)
- batch_size=32 recommended for M1/M2; 64+ for M3 Max
- FlashAttention is CUDA-only; MPS uses the standard SDPA path
