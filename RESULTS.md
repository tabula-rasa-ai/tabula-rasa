# Validation Results — Phase 1

> Automated validation of core claims in Tabula Rasa AI.
> Date: 2026-06-13 | Environment: CPU-only (Windows 11)

---

## 1A. Addition Specialist Training

**Claim:** "100% accuracy in 4,000 steps on 1-digit addition"

**Result: ✅ VALIDATED — 100% in 3,000 steps**

| Metric | Value |
|--------|-------|
| Steps to 93% | 1,000 |
| Steps to 100% | **3,000** |
| Final 1-digit accuracy (200 eval) | **100.0%** |
| 12+5=17 | ❌ (2-digit, not trained yet — expected) |
| Model params | 1,060,992 |
| Architecture | d=128, L=4, h=4, ff=512 |
| Device | CPU (no GPU available) |

**Note:** The overall eval accuracy shows ~7% because it averages all digit lengths (1d-4d), but the model has only been trained on phase 1 (max_digits=1 at that point). The 1-digit column **1d:100%** confirms the claim is sound.

**Training details:**
- Reversed digits: enabled
- Loss masking: enabled
- Scratchpad: enabled
- Curriculum: phase 1/4 (max_digits=1)
- Optimizer: AdamW, cosine LR schedule
- Dataset: 5,000 samples, 50% forced carry

---

## 1B. Fused Carry-Digit Tokens

**Claim:** "20 fused carry-digit tokens (f'{carry}{digit}') in the tokenizer"

**Result: ✅ VALIDATED**

20 carry-digit tokens confirmed in `MathTokenizer` vocabulary:

```
Token  ID
"00"   4
"01"   5
...
"09"   13
"10"   14
"11"   15
...
"19"   23
```

- Tokenizer file: `tokenizer.py` (line 18: `CARRY_TOKENS = [f"{c}{d}" for c in range(2) for d in range(10)]`)
- Total vocab: 44 tokens (4 special + 20 carry-digit + 20 math chars)
- Tokenizer uses longest-match-first encoding strategy
- Model sees raw IDs — carry-digit fusion happens at the tokenizer level

---

### 1C. Online EWC Fisher Matrix Persistence

**Claim:** EWC Fisher matrix saved at `specialists/math/addition/ewc_fisher.pt`

**Initial Result: ❌ NOT IMPLEMENTED** (as of initial validation)

**Current Status: ✅ IMPLEMENTED** (as of 2026-06-13)

The following components were built:

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Online EWC module | `egefalos/online_ewc.py` | ~175 | ✅ Tested |
| EWC integration in trainer | `train_specialist.py` | +50 | ✅ Tested |
| Hippocampus (SQLite) | `egefalos/hippocampus.py` | ~180 | ✅ Pre-existing |
| Sleep cycle daemon | `egefalos/sleep_cycle.py` | ~220 | ✅ Tested |

**Verified functionality:**
- [x] Fisher matrix computation on data samples
- [x] Merge with exponential decay: F_combined = γ·F_old + (1-γ)·F_new
- [x] EWC penalty term in training loss: (λ/2)·ΣF·(θ-θ*)²
- [x] Save/load `ewc_fisher.pt` (39 param groups saved)
- [x] Anchor weight save/restore
- [x] Sleep cycle: replay → Fisher compute → merge → train → consolidate
- [x] `--ewc` CLI flag on `train_specialist.py`
- [x] Hippocampus SQLite: store, sample, mark consolidated

---

## Environment Notes

- No GPU available on this machine (all training runs on CPU)
- CPU training speed: ~2.7-10 st/s for 1M param model
- 4000 steps takes approximately 25-30 minutes on CPU (with eval at step 3000)
- GitHub: https://github.com/tabula-rasa-ai/tabula-rasa
