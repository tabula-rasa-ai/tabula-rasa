"""Download quality chat dataset for Tabula Rasa.
Usage: python3 download_dataset.py [num_samples=5000]

Downloads Alpaca-cleaned from Hugging Face and converts to our JSONL format.
No HF datasets library needed — uses direct HTTP download.
"""
import sys, json, random, os, gzip, io

NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

# ── Option 1: Alpaca-cleaned (simple instruction format) ──
print("Downloading alpaca-cleaned dataset...")

# Try HF dataset viewer API (no datasets library needed)
import urllib.request
import urllib.error

url = "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json"
try:
    with urllib.request.urlopen(url, timeout=30) as f:
        data = json.loads(f.read().decode())
    print(f"Downloaded {len(data)} raw examples")
except Exception as e:
    print(f"Alpaca download failed: {e}")
    print("Trying alternative: Dolly 15k...")
    # Fallback: Dolly 15k
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    data = []
    with urllib.request.urlopen(url, timeout=30) as f:
        for line in f.read().decode().strip().split("\n"):
            if line:
                data.append(json.loads(line))
    print(f"Downloaded {len(data)} Dolly examples")

random.shuffle(data)
pairs = []
for item in data[:NUM_SAMPLES]:
    if "instruction" in item and "output" in item:
        # Alpaca format
        prompt = item["instruction"].strip()
        inp = item.get("input", "").strip()
        output = item["output"].strip()
        if inp and inp != "<noinput>":
            prompt = f"{prompt}\n{inp}"
        if prompt and output:
            pairs.append({"prompt": prompt, "response": output})
    elif "prompt" in item and "response" in item:
        pairs.append({"prompt": item["prompt"], "response": item["response"]})

print(f"Converted {len(pairs)} pairs")

# Shuffle and save
random.shuffle(pairs)
out_path = "chat_data.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\nSaved {len(pairs)} pairs to {out_path}")
if pairs:
    print(f"Sample:\n  Q: {pairs[0]['prompt'][:80]}\n  A: {pairs[0]['response'][:80]}")
print(f"\nTo train: python3 train_chat_gpu.py {out_path}")
