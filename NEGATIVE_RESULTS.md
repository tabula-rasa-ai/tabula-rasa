# Negative Results & Lessons Learned

This document systematically catalogs approaches that **failed**, hypotheses that were
**rejected**, and engineering dead-ends encountered during Tabula Rasa's development.
Documenting negative results is as important as positive ones — it saves future
researchers from repeating the same mistakes and reveals fundamental constraints.

---

## 1. Continual Learning Failures

### 1.1 Three-Task Collapse (1M Parameter Limit)

**Hypothesis:** Online EWC (merged Fisher, γ=0.9) can protect 3+ structurally
divergent arithmetic tasks in a 1M-parameter transformer.

**Result:** **Rejected.** Both Online EWC and standard (non-merged) EWC fail
identically at 3 tasks:

| Metric | 1M (Online EWC) | 1M (Standard EWC) | 5M (Fisher Archive) |
|--------|-----------------|-------------------|---------------------|
| Add after sub | 40% | 38% | **100%** |
| Sub after mul | 2% | 0% | N/A (killed early) |
| Mul acquired | 14% | 12% | N/A |

**Root cause:** The 1M-param transformer lacks the representational capacity to
encode three structurally distinct arithmetic operations (addition, subtraction,
multiplication) simultaneously. The Fisher matrices for add and sub overlap by
~0.977 (cosine similarity), meaning EWC cannot distinguish between the two.
Multiplication's structural divergence causes the merged Fisher to protect
neither.

**Resolution:** Scaling to 5.27M params (d=256, L=5) restores 100% addition
retention and 93% subtraction acquisition. The boundary is strictly capacity,
not algorithm.

### 1.2 Adaptive Gamma Rejection

**Hypothesis:** Dynamically tuning EWC's gamma (Fisher merge decay) based on
task overlap can improve multi-task retention.

**Result:** **Rejected.** Fisher overlap between addition and subtraction is
0.977 — tasks are not divergent enough for adaptive blending to matter. Adaptive
gamma collapses to the same fixed value regardless of input. The Pareto frontier
of lambda (100-2000) is flat for two-task scenarios: all values achieve >95%
on both tasks.

### 1.3 EWC Lambda Sweep — No Trade-off Found

**Hypothesis:** Higher EWC lambda preserves old tasks at the expense of new-task
learning (the standard EWC trade-off).

**Result:** **Partially rejected.** In the 2-task scenario (add → sub), there
is no trade-off. λ ∈ {100, 200, 500, 1000, 2000} all achieve >95% on both
tasks with near-zero variance. The Pareto frontier is flat. This is a
best-case finding but means lambda tuning is uninformative for 2 structurally
similar tasks. A trade-off may exist for 3+ divergent tasks but cannot be
isolated from the capacity collapse.

---

## 2. Training Failures

### 2.1 Batch Size Collapse (batch=2048)

**Symptom:** Training loss drops to 0.0000 within 500 steps, but exact-match
accuracy stays at 0% for the entire run. GPU utilization is 90%+ but model
produces random outputs.

**Root cause:** With a 1M-param model and batch=2048, each gradient step averages
2048 samples. Loss masking means only ~4 answer tokens per sample contribute
real gradient signal (~8192 meaningful tokens per step). The gradient variance
is abnormally low — the model memorizes each mini-batch rather than learning
the underlying algorithm.

**Safe upper bounds:**
| Model | Max Batch | Gradient Signal |
|-------|-----------|-----------------|
| 1M | 256 | ~1024 answer tokens/step |
| 5M | 512 | ~2048 answer tokens/step |
| 10M | 1024 | ~4096 answer tokens/step |

**Detection:** If training loss approaches 0 but accuracy stays at 0%, the batch
is too large. Reduce by 4x and verify accuracy climbs above 50%.

### 2.2 Curriculum Evaluation Misalignment

**Symptom:** Training with curriculum phases (e.g., 1-digit → 2-digit → 3-digit)
produces eval accuracy of 12-20% even when the model is 100% accurate on the
current phase. The early-stop triggers prematurely and kills the run.

**Root cause:** `cfg.max_digits = 4` in config.py sets the eval to average across
ALL digit levels (1d, 2d, 3d, 4d). During phase 1 (1-digit), the eval reports:

```
Eval: 1d=100%, 2d=0%, 3d=0%, 4d=0% → reported ~25%
```

The early stopper sees 5 evals with "no improvement" (25% → 25% → 25%) and kills
the run at step 5,000 when it should have trained for 30,000.

**Fix:** Always align `cfg.max_digits` with the curriculum:

```python
# Phase 1: 1-digit training
cfg.max_digits = 1
cfg.curriculum_phases = [(30000, 1)]
```

### 2.3 EWC Penalty Autograd Growth

**Symptom:** Training starts fast (1-2s per eval) but progressively slows to
60s+ per eval after ~2000 steps. The slowdown is linear with step count.

**Root cause:** The EWC penalty computation used in-place `+=` to accumulate
penalty terms:

```python
# BAD — O(N) graph depth
task_penalty = torch.tensor(0.0)
for name, param in model.named_parameters():
    task_penalty += (f * (param - theta_star) ** 2).sum()  # creates 1 graph node per step
```

After 3000 steps × 48 parameter groups = 144,000 autograd graph nodes, the
backward pass traverses the entire graph, causing the slowdown.

**Fix:** Use `list + sum()` for O(1) graph depth:

```python
# CORRECT — O(1) graph depth
terms = []
for name, param in model.named_parameters():
    terms.append((f * (param - theta_star) ** 2).sum())
return sum(terms)
```

### 2.4 Loss-Accuracy Divergence

**Finding:** Loss and accuracy are nearly uncorrelated for the fused carry-digit
task.

| Step | Loss | Accuracy |
|------|------|----------|
| 200 | 1.06 | 0% |
| 500 | 0.84 | 8% |
| 1000 | 0.61 | 18% |
| 2000 | 0.12 | 62% |
| 3000 | 0.04 | 98% |

Loss drops to near-zero before accuracy exceeds 50%. Watching loss as the
primary training signal leads to false conclusions — the model appears to be
learning (loss decreasing) when it is actually just minimizing a noisy loss
function without understanding the task.

**Resolution:** Accuracy is the primary signal. Loss is informational but should
never be used for early stopping or run-kill decisions. Added `--eval-interval`
flag to make accuracy the default monitoring metric.

### 2.5 5M Model Convergence Sensitivity

**Finding:** The 5.27M model (d=256, L=5) fails to converge with the same
hyperparameters that work for the 1M model (batch=256, LR=0.001, train_samples=50000).

| Config | 5M Result | Root Cause |
|--------|-----------|------------|
| batch=256, samples=50000 | 0% accuracy | Too many unique samples per epoch (195 batches) |
| batch=128, samples=5000 | 100% by step 500 | Matches 1M recipe — 39 batches/epoch, 77 passes |

**Lesson:** Larger models need **more repetitions per example**, not more unique
examples. The 5M model with 50000 samples sees each example ~13 times vs ~77
times for the 1M model with 5000 samples. Parameter count alone doesn't explain
learning dynamics — data repetition rate matters.

---

## 3. Engineering Failures

### 3.1 Causal Mask Silently Disabled

**Bug:** The `_causal_mask()` tensor was computed and passed to every attention
layer. Inside `Attention.forward()`, the check `use_causal = (mask is None and
past_kv is None)` evaluated to `False` because `mask` was not `None`. Result:
SDPA received `is_causal=False + attn_mask=None` = fully bidirectional attention.

**Impact:** Training for months with bidirectional attention on a causal
(left-to-right) task. The model was "cheating" by looking at future tokens.
Accuracy was inflated by ~15-20% compared to proper causal training.

**Detection:** The `test_causal_masking_active` test compares logits for
sequences identical except for the last token. With causal masking, the first
N-1 positions must produce identical logits. This test was created as a
regression guard.

**Fix:** Remove mask computation in `forward()` and pass `mask=None`:

```python
# Before:
mask = self._causal_mask(seq_len, device)
for layer in self.layers:
    x, layer_kv = layer(x, mask)

# After:
for layer in self.layers:
    x, layer_kv = layer(x, mask=None)
```

### 3.2 DataLoader Exhaustion in Experiment Scripts

**Bug:** Standalone experiment scripts (`experiments/run_*.py`) create a
`DataLoader` and iterate with `for x, y in loader:`. With dataset_size=5000
and batch=128, the loader exhausts after ~39 iterations. The experiment
reports "done" after 39 steps instead of the requested 3000.

**Fix:** Always use the `_infinite_batches()` generator pattern when writing
standalone scripts:

```python
def _infinite_batches(ds, bs, dev):
    while True:
        for x, y in DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True):
            yield x.to(dev), y.to(dev)
```

### 3.3 Hippocampus Connection Reopening

**Bug:** Every call to `store_experience()`, `get_unconsolidated()`, etc. opened
a fresh SQLite connection, which also re-ran `CREATE TABLE`, `ALTER TABLE`, and
`CREATE INDEX` on every invocation. The sleep cycle was spending 80% of its time
in schema initialization.

**Fix:** Module-level persistent connection with WAL mode and `_schema_initialized`
flag. The connection is opened once and reused.

---

## 4. Rejected Hypotheses

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| MoE + global EWC fixes 3-task collapse | Rejected | Router collapses to 1-2 experts without aux loss |
| ExpertEWC + MoE fixes 3-task collapse | Rejected | Per-expert Fishers protect but shared params still collapse |
| Adaptive gamma mitigates Fisher norm shrinkage | Rejected | Fisher overlap=0.977 → adaptive gamma is flat |
| Larger batch sizes train faster for small models | Rejected | batch=2048 → 0% accuracy on 1M model |
| Loss is a good early stopping signal | Rejected | Loss drops to 0 while accuracy is 0% |
| More training samples help larger models | Rejected | 50000 samples → worse than 5000 for 5M model |
| Reverse digits is optional for causal attention | Rejected | Without reversal, carry propagation fails on MSD-first |
| LoRA matches full fine-tuning for arithmetic | Partially | Works for fine-tuning, not for learning from scratch |

---

## 5. Hardware Requirements & Costs

### 5.1 Training Time (1M model, 1-digit addition, 30000 steps)

| Hardware | Steps/sec | Total Time | Cost (on-demand) |
|----------|-----------|------------|------------------|
| CPU (4-core) | 0.9 | ~9 hours | $0 (local) |
| CPU (16-core) | 2.1 | ~4 hours | $0 (local) |
| Apple M1/M2 | 15 | ~33 min | $0 (local) |
| RTX 3060 (12GB) | 80 | ~6 min | ~$0.02 (RunPod) |
| RTX 4080 (16GB) | 200 | ~2.5 min | ~$0.01 (RunPod) |
| RTX 5090 (32GB) | 300 | ~1.7 min | ~$0.01 (RunPod) |

### 5.2 Memory

| Model | Params | Checkpoint Size | Fisher Size (per task) | Peak VRAM (batch=128) |
|-------|--------|-----------------|----------------------|----------------------|
| 1M | 1,061,504 | 4.2 MB | 4.0 MB | ~256 MB |
| 5M | 5,269,248 | 21 MB | 20 MB | ~1.2 GB |
| 10M | 9,864,000 | 40 MB | 38 MB | ~2.4 GB |

### 5.3 Minimum Viable Hardware

| Task | Minimum RAM | Min VRAM | Storage | Estimated Time |
|------|------------|----------|---------|---------------|
| 1-digit add (1M) | 4 GB | None (CPU OK) | 50 MB | 9 hours (CPU) |
| 1-digit add (1M, quick) | 2 GB | None | 10 MB | 3 min |
| 3-task ablation (1M) | 4 GB | None | 100 MB | 27 hours (CPU) |
| 3-task ablation (5M) | 8 GB | 2 GB | 200 MB | ~12 hours (CPU), ~15 min (GPU) |
| Socratic self-play | 8 GB | 4 GB | 500 MB | ~2 hours (GPU) |
| PPO training | 8 GB | 4 GB | 1 GB | ~4 hours (GPU) |

---

## 6. Summary of Failures by Category

### Most Costly (wasted GPU hours)
1. **Batch=2048** — weeks of GPU time producing 0% accuracy before the bug was
   identified
2. **Causal mask bug** — months of training with bidirectional attention;
   all results before the fix are invalid for causal tasks
3. **EWC autograd growth** — progressively slower training killed multi-day runs

### Most Surprising
1. **Loss-accuracy divergence** — violates standard ML assumptions about loss
   being a reliable training signal
2. **5M model instability** — larger models are more sensitive to hyperparameters,
   not less

### Most Important for Future Work
1. **Capacity boundary at 1M** — the 3-task collapse is architectural, not
   algorithmic. Scaling laws for continual learning are an open question.
2. **Fisher overlap near 1.0** — if two tasks' Fishers are nearly identical,
   EWC cannot distinguish between them regardless of lambda tuning
