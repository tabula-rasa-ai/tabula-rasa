"""Train a 10M chat specialist on GPU with BPE tokenizer.
Usage: python3 train_chat_gpu.py chat_data.jsonl

Learns subword tokens from the data first (num_merges=200),
so "Hello" becomes ~2 tokens instead of 5 characters.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.chat_dataset import encode_chat_sample
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Tokenizer with BPE ──
tok = BPETokenizer()

# Learn BPE merges from the training data
data_path = sys.argv[1] if len(sys.argv) > 1 else "chat_data.jsonl"
pairs = []
with open(data_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        item = json.loads(line)
        p = item.get("prompt", item.get("instruction", ""))
        r = item.get("response", item.get("output", ""))
        if p and r: pairs.append((p, r))

print(f"Loaded {len(pairs)} pairs, learning BPE merges...")
all_texts = [p for pair in pairs for p in pair]
tok.learn_bpe_from_texts(all_texts, num_merges=200, verbose=False)
print(f"Vocab: {tok.vocab_size} tokens ({tok.bpe_vocab_size} BPE merges)")

# ── 10M model ──
cfg = Config()
cfg.apply_preset("10M")
cfg.vocab_size = tok.vocab_size
cfg.max_seq_len = 128
model = MathTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")

os.makedirs("specialists/chat", exist_ok=True)
tok.save("specialists/chat/tokenizer.json")

# ── Dataset ──
num_samples = 50000
max_len = 64
all_x, all_y = [], []
for i in range(num_samples):
    p, r = pairs[i % len(pairs)]
    prompt = p[:max_len // 2 - 1] + "="
    response = r[:max_len // 2 - 1]
    x, y = encode_chat_sample(prompt, response, tok, cfg.max_seq_len, True)
    all_x.append(x.unsqueeze(0))
    all_y.append(y.unsqueeze(0))
X = torch.cat(all_x).to(device)
Y = torch.cat(all_y).to(device)
print(f"Data: {X.shape}")

# ── Train ──
batch_size = 256
steps = 5000
opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
t0 = time.time()

for step in range(steps):
    idx = torch.randperm(num_samples)[:batch_size].to(device)
    _, loss, _ = model(X[idx], Y[idx])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 500 == 0:
        sps = step / max(1, time.time() - t0)
        elapsed = time.time() - t0
        eta = (steps - step) / max(sps, 0.01) / 60
        print(f"S{step}/{steps} loss={loss.item():.4f} {sps:.1f}st/s eta={eta:.0f}m")
        torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": step},
                   "specialists/chat/checkpoint.pt")

torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": steps},
           "specialists/chat/best.pt")
print(f"DONE in {(time.time()-t0)/60:.1f}m")
