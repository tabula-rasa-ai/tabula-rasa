# Extending Tabula Rasa: Adding a New Operation

This tutorial walks through adding a new arithmetic operation — **factorial** (`!`) — to Tabula Rasa. By the end, you'll have trained a factorial specialist and routed queries to it via the API.

**Time:** ~30 minutes (CPU) | **Difficulty:** Intermediate

---

## Table of Contents

1. [Overview of What You'll Modify](#1-overview-of-what-youll-modify)
2. [Step 1: Add Dataset Generation](#2-step-1-add-dataset-generation)
3. [Step 2: Update the Tokenizer](#3-step-2-update-the-tokenizer)
4. [Step 3: Register the Operation in the Trainer](#4-step-3-register-the-operation-in-the-trainer)
5. [Step 4: Train Your Specialist](#5-step-4-train-your-specialist)
6. [Step 5: Register the Operation in the Router](#6-step-5-register-the-operation-in-the-router)
7. [Step 6: Test End-to-End](#7-step-6-test-end-to-end)
8. [Step 7: Evaluate and Document](#8-step-7-evaluate-and-document)
9. [Checklist for Adding Your Own Operation](#9-checklist-for-adding-your-own-operation)

---

## 1. Overview of What You'll Modify

| File | Change |
|------|--------|
| `dataset.py` | Add `generate_factorial_problem()` function and register `'!'` in `OPERATIONS` |
| `tokenizer.py` | Add `!` to `MATH_CHARS` |
| `train_specialist.py` | Add `'fac'` to `OPS` dict and `OP_NAMES` dict, implement factorial problem logic |
| `specialist_router.py` | Add `'!': 'fac'` to `OPS` |
| Dashboard | No changes needed — dashboard auto-discovers new operations via the API |

**Why factorial?** It's structurally different from add/sub/mul/div — it's a unary operation (one operand) with exponential growth. This tests whether the system can handle operations outside the binary-arithmetic paradigm.

---

## 2. Step 1: Add Dataset Generation

Edit `dataset.py` to add factorial support:

### 2a. Add `'!'` to OPERATIONS

```python
# Before
OPERATIONS = ['+', '-', '*', '/']

# After
OPERATIONS = ['+', '-', '*', '/', '!']
```

### 2b. Add the factorial generation function

Add this function after `generate_problem`:

```python
def generate_factorial_problem(min_digits=1, max_digits=2):
    """Generate a factorial problem.

    Factorial grows fast: 5! = 120, 8! = 40320.
    We cap at 8 to keep results manageable (< 6 digits).
    """
    n = random.randint(1, min(8, 10**max_digits - 1))

    # Compute factorial
    result = 1
    for i in range(2, n + 1):
        result *= i

    expr = f'{n}!'
    answer = str(result)
    return expr, answer
```

### 2c. Update `generate_problem` to handle `'!'`

Modify `generate_problem` to detect factorial:

```python
def generate_problem(min_digits=1, max_digits=4):
    """Generate an arithmetic problem and its answer."""
    op = random.choice(OPERATIONS)

    # Handle factorial (unary operation)
    if op == '!':
        return generate_factorial_problem(min_digits, max_digits)

    # ... rest of existing binary-op logic ...
```

---

## 3. Step 2: Update the Tokenizer

Edit `tokenizer.py`:

```python
# Before
MATH_CHARS = list('0123456789+-*/=().% ')

# After
MATH_CHARS = list('0123456789+-*/=().% !')
```

That's it — `!` is now a recognized token. The tokenizer uses longest-match-first encoding, so `!` in a prompt like `5!=120` will correctly tokenize as `5`, `!`, `=`, `1`, `2`, `0`.

**Verify:**
```python
from tokenizer import MathTokenizer
tok = MathTokenizer()
ids = tok.encode('5!=120')
print(tok.decode(ids))  # Should print: 5!=120
assert '!' in tok.stoi
```

---

## 4. Step 3: Register the Operation in the Trainer

Edit `train_specialist.py`:

### 4a. Add to OPS and OP_NAMES

```python
# Before
OPS = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
OP_NAMES = {'add': 'Addition', 'sub': 'Subtraction', 'mul': 'Multiplication', 'div': 'Division'}

# After
OPS = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'fac': '!'}
OP_NAMES = {'add': 'Addition', 'sub': 'Subtraction', 'mul': 'Multiplication',
            'div': 'Division', 'fac': 'Factorial'}
```

### 4b. Add factorial logic to `generate_problem`

In the `generate_problem` function inside `train_specialist.py`, add:

```python
if op == 'fac':
    n = random.randint(1, min(8, 10**max_digits - 1))
    result = 1
    for i in range(2, n + 1):
        result *= i
    ans = result
```

### 4c. Update scratchpad logic (if used)

If scratchpad is enabled, factorial's scratchpad is simpler (no carries):

```python
# In the scratchpad formatting section
if op == 'fac':
    scratchpad = f'{n}!={ans}'
```

---

## 5. Step 4: Train Your Specialist

```bash
# Quick smoke test (500 steps)
python train_specialist.py fac --quick

# Full training (30K steps with curriculum)
python train_specialist.py fac
```

**Expected output:**
```
Training Factorial — steps=30000, device=cpu
Step   500 | loss=2.34 | acc=12% | lr=4.5e-4
Step  1000 | loss=1.89 | acc=34% | lr=8.2e-4
Step  3000 | loss=1.12 | acc=72% | lr=9.8e-4
...
Step 30000 | loss=0.08 | acc=100% | lr=0.0e-0
```

Training checkpoints are saved to `specialists/fac/`.

---

## 6. Step 5: Register the Operation in the Router

Edit `specialist_router.py`:

```python
# Before
OPS = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}

# After
OPS = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '!': 'fac'}
```

Also update `detect_operation` if needed — factorial is detected by `!` in the prompt.

And update the `SpecialistManager.__init__` list:

```python
# Before
for op in ['add', 'sub', 'mul', 'div']:

# After
for op in ['add', 'sub', 'mul', 'div', 'fac']:
```

---

## 7. Step 6: Test End-to-End

### Start the router
```bash
python specialist_router.py
```

### Send a factorial query
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"5!="}'
```

**Expected response:**
```json
{
  "prompt": "5!=",
  "result": "5!=120",
  "specialist": "fac",
  "operation": "!",
  "time_ms": 45
}
```

### Test via the dashboard
Open http://localhost:8000 and use the Interactive Chat view.

---

## 8. Step 7: Evaluate and Document

### Run evaluation
```bash
python eval.py --op fac
```

### Save results
```python
# In your experiment script
results = {
    'operation': 'factorial',
    'accuracy': 100.0,
    'steps': 30000,
    'params': 1060992,
}
import json
with open('experiments/factorial_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Update docs
- Add factorial to the operations table in `README.md`
- Optionally update `RESULTS.md` with validation data

---

## 9. Checklist for Adding Your Own Operation

- [ ] `dataset.py`: Add operation symbol to `OPERATIONS`, implement problem generation
- [ ] `tokenizer.py`: Add any new characters to `MATH_CHARS`
- [ ] `train_specialist.py`: Add to `OPS` and `OP_NAMES`, implement computation logic
- [ ] Train a specialist: `python train_specialist.py myop --quick`
- [ ] `specialist_router.py`: Add to `OPS` dict and `SpecialistManager` loop
- [ ] Test: `curl -X POST ... -d '{"prompt":"..."}'`
- [ ] Evaluate: `python eval.py --op myop`
- [ ] Document: Update README and RESULTS.md

### Common pitfalls

| Pitfall | Solution |
|---------|----------|
| **Unary vs binary ops** | Trainer assumes two operands — handle factorial-like ops separately |
| **Result too large** | Clip operands so result fits in model's context (max_seq_len=32) |
| **New token not registered** | Must add to both `tokenizer.py` AND the trainer's local logic |
| **Tokenizer can't decode result** | Check `MATH_CHARS` includes all characters in the answer |
| **Router can't find specialist** | Update the `for op in [...]` loop in `SpecialistManager.__init__` |
| **Dashboard doesn't show new op** | This is usually auto-discovered, but restart `api_server.py` |

---

## Advanced: Adding a Non-Arithmetic Operation

If you want to go beyond arithmetic (e.g., string reversal, palindrome detection):

1. **Tokenizer:** You may need a general text tokenizer (`bpe_tokenizer.py` exists as a reference)
2. **Dataset:** Generate text pairs instead of number pairs
3. **Model:** The existing transformer handles any token-to-token mapping — no architecture changes needed
4. **Training:** Use the same `train_specialist.py` framework, just pass different data

---

**Questions?** Open an issue or join [Telegram](https://t.me/TabulaRasaAi).
