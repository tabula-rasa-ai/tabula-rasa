# Tabula Rasa

A transformer trained from a **blank slate** — no pretraining, no transfer learning,
just gradient descent from random initialization.

- **1M-parameter transformer** learns 1-digit addition to 100% in ~3,000 steps on CPU
- **Operation-specific scratchpads** for carry/borrow propagation and partial-products
- **Continual learning** via Online EWC, MAS, OGD — no catastrophic forgetting
- **Multi-specialist architecture** with neural routing

## Key Results

| Operation | 1-digit | 2-digit | 4-digit |
|-----------|---------|---------|---------|
| Addition  | 100%    | ~95%    | ~51%    |
| Subtraction | ~85%  | ~60%    | ~5%     |
| Multiplication | ~70% | ~30%  | ~3%     |

## Installation

```bash
pip install tabula-rasa-ai
```

## Quickstart

```bash
# Train an addition specialist
tabula-rasa train add --quick

# Run inference
tabula-rasa serve
curl -X POST http://localhost:8002/generate -d '{"prompt":"12+34="}'
```
