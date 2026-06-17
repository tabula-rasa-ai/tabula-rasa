"""Learn BPE merges from chat text and train a chat specialist with subword tokens."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.chat_dataset import ChatDataset, encode_chat_sample
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer

# Learn BPE merges
tok = BPETokenizer()
print(f"Base vocab: {tok.vocab_size}")

chat_texts = [
    "Hello!=", "Hi there! I'm Tabula Rasa, a helpful AI assistant.",
    "How are you?=", "I'm an AI, so I don't have feelings, but I'm fully operational and ready to help!",
    "What is 2+2?=", "4",
    "Tell me a joke.", "Why did the math book look sad? Because it had too many problems.",
    "What's your name?=", "I'm Tabula Rasa, a transformer trained from scratch to learn and assist.",
    "Who created you?=", "I was created by the Tabula Rasa AI research project - learning from scratch, one specialist at a time.",
    "What can you do?=", "I can solve math problems, answer questions, and learn new skills through continual learning.",
    "Explain gravity.", "Gravity is a natural force that attracts objects with mass toward one another.",
    "What is the capital of France?=", "The capital of France is Paris.",
    "What is machine learning?=", "Machine learning is a branch of AI where systems learn patterns from data.",
    "Tell me a fun fact.", "Did you know? Octopuses have three hearts, and two stop when they swim!",
    "What is Python?=", "Python is a high-level, interpreted programming language known for its readability.",
    "How do transformers work?=", "Transformers use self-attention to process sequences in parallel.",
    "What is deep learning?=", "Deep learning uses multi-layer neural networks to learn hierarchical representations.",
    "Thanks!=", "You're welcome! Feel free to ask me anything.",
    "Goodbye!=", "Goodbye! Come back anytime.",
    "What is the meaning of life?=", "42 - at least according to Douglas Adams.",
    "Tell me something deep.", "Every atom in your body was forged in the heart of a star.",
    "What is a neural network?=", "A neural network is composed of layers of interconnected nodes (neurons).",
    "Hi there!", "Hello! How can I help you today?",
    "What is AI?=", "AI is the field of creating machines that can perform tasks requiring human intelligence.",
]

tok.learn_bpe_from_texts(chat_texts, num_merges=100, verbose=True)
print(f"Vocab: {tok.vocab_size} tokens ({tok.bpe_vocab_size} BPE merges)")

# Train model
os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

cfg = Config()
cfg.apply_preset("1M")
cfg.vocab_size = tok.vocab_size
cfg.max_seq_len = 96
model = MathTransformer(cfg)
print(f"Model: {sum(p.numel() for p in model.parameters()):,}")

os.makedirs("specialists/chat", exist_ok=True)
tok.save("specialists/chat/tokenizer.json")

ds = ChatDataset(num_samples=1000, max_len=48)
all_x, all_y = [], []
for i in range(1000):
    p, r = ds[i]
    x, y = encode_chat_sample(p, r, tok, 96, True)
    all_x.append(x.unsqueeze(0))
    all_y.append(y.unsqueeze(0))
X = torch.cat(all_x)
Y = torch.cat(all_y)
print(f"Data: {X.shape}")

opt = torch.optim.AdamW(model.parameters(), lr=0.001)
t0 = time.time()
for step in range(800):
    idx = torch.randperm(1000)[:32]
    _, loss, _ = model(X[idx], Y[idx])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 200 == 0:
        sps = step / max(1, time.time() - t0)
        print(f"S{step} loss={loss.item():.4f} {sps:.1f}st/s")
        torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": step}, "specialists/chat/checkpoint.pt")

torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": 800}, "specialists/chat/best.pt")
torch.save({"model_state_dict": model.state_dict(), "acc": 0, "global_step": 800}, "specialists/chat/final.pt")
print(f"DONE in {time.time()-t0:.0f}s")
