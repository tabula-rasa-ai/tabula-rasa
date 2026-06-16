#!/usr/bin/env bash
# Train Subtraction and Multiplication Specialists
# Run this on the GPU machine
set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo " Training Subtraction Specialist (sub)"
echo " Target: >= 98% 1-digit accuracy"
echo " Steps: 4000"
echo "========================================"
python3 scripts/train_specialist.py sub --steps 4000 --batch 64
echo "Sub training done"

echo ""
echo "========================================"
echo " Training Multiplication Specialist (mul)"
echo " Target: >= 90% 1-digit accuracy"
echo " Steps: 4000"
echo "========================================"
python3 scripts/train_specialist.py mul --steps 4000 --batch 64
echo "Mul training done"

echo ""
echo "========================================"
echo " Updating RESULTS.md"
echo "========================================"
python3 -c "
from pathlib import Path
import subprocess, sys

results = Path('RESULTS.md')
text = results.read_text(encoding='utf-8')

# Check specialist accuracy
def get_acc(op):
    try:
        ckpt = Path(f'specialists/math/{op}/best.pt')
        if not ckpt.exists():
            ckpt = Path(f'specialists/math/{op}/final.pt')
        if not ckpt.exists():
            return None
        from tabula_rasa.config import Config
        from tabula_rasa.tokenizer import MathTokenizer
        from tabula_rasa.model import MathTransformer
        import torch
        tok = MathTokenizer.load(str(ckpt.parent / 'tokenizer.json'))
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        tok.max_seq_len = cfg.max_seq_len
        model = MathTransformer(cfg)
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model_state_dict'], strict=False)
        model.eval()
        from tabula_rasa.dataset import generate_problem
        correct = total = 0
        for _ in range(200):
            expr, ans = generate_problem(1, op)
            prompt = f'{expr}='
            out = model.generate(tok, prompt, max_new_tokens=10, temperature=0.0)
            pred = out.split('=')[-1].strip() if '=' in out else ''
            pred = ''.join(c for c in pred if c.isdigit() or c == '-')
            if pred == ans:
                correct += 1
            total += 1
        return correct / total * 100
    except Exception as e:
        return f'error: {e}'

sub_acc = get_acc('sub')
mul_acc = get_acc('mul')

print(f'Sub accuracy: {sub_acc:.1f}%' if sub_acc else 'Sub: no checkpoint')
print(f'Mul accuracy: {mul_acc:.1f}%' if mul_acc else 'Mul: no checkpoint')

# Update table in RESULTS.md
import re
if sub_acc:
    text = re.sub(r'(?<=sub )\d+\.?\d*(?=%| -)', f'{sub_acc:.1f}', text)
if mul_acc:
    text = re.sub(r'(?<=mul )\d+\.?\d*(?=%| -)', f'{mul_acc:.1f}', text)
results.write_text(encoding='utf-8')
print('RESULTS.md updated')
"

echo ""
echo "========================================"
echo " All done!"
echo " Check RESULTS.md for updated accuracies"
echo "========================================"
