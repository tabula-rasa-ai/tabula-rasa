"""Train a 10M chat specialist on JSONL data.
Uses base character-level tokenizer (no BPE merges — skips slow learning phase).
"""
import os, sys, time, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.chat_dataset import encode_chat_sample
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer

os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

# ── Character-level tokenizer (fast, no BPE learning) ──
tok = BPETokenizer()
print(f"Vocab: {tok.vocab_size} tokens (no BPE)")

# ── 10M model ──
cfg = Config()
cfg.apply_preset("10M")
cfg.vocab_size = tok.vocab_size
cfg.max_seq_len = 96
model = MathTransformer(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")

os.makedirs("specialists/chat", exist_ok=True)
tok.save("specialists/chat/tokenizer.json")

# ── Load JSONL pairs ──
chat_pairs = []
data_path = sys.argv[1] if len(sys.argv) > 1 else "chat_data.jsonl"
with open(data_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        item = json.loads(line)
        p = item.get("prompt", item.get("instruction", ""))
        r = item.get("response", item.get("output", ""))
        if p and r:
            chat_pairs.append((p, r))
print(f"Loaded {len(chat_pairs)} unique pairs")

# ── Build dataset ──
num_samples = 50000
max_len = 48
t0 = time.time()
all_x, all_y = [], []
for i in range(num_samples):
    p, r = chat_pairs[i % len(chat_pairs)]
    pl = min(len(p), max_len // 2 - 1)
    rl = min(len(r), max_len // 2 - 1)
    prompt = p[:pl] + "="
    response = r[:rl]
    x, y = encode_chat_sample(prompt, response, tok, cfg.max_seq_len, True)
    all_x.append(x.unsqueeze(0))
    all_y.append(y.unsqueeze(0))
X = torch.cat(all_x)
Y = torch.cat(all_y)
print(f"Data: {X.shape}, built in {time.time()-t0:.0f}s")

# ── Train ──
batch_size = 64
steps = 5000
opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
t0 = time.time()

for step in range(steps):
    idx = torch.randperm(num_samples)[:batch_size]
    _, loss, _ = model(X[idx], Y[idx])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 500 == 0 and step > 0:
        sps = step / max(1, time.time() - t0)
        elapsed = time.time() - t0
        eta = (steps - step) / max(sps, 0.01) / 3600
        print(f"S{step}/{steps} loss={loss.item():.4f} {sps:.1f}st/s eta={eta:.1f}h", flush=True)
        torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": step},
                   "specialists/chat/checkpoint.pt")

torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": steps},
           "specialists/chat/best.pt")
torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": steps},
           "specialists/chat/final.pt")
print(f"DONE in {(time.time()-t0)/3600:.1f}h", flush=True)
