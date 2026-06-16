"""GPU Training Runner — trains all specialists on a rented GPU.

Usage on a cloud GPU (RunPod, Vast.ai, etc.):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    python train_gpu.py

Trains: generalist (all 4 ops) + each individual specialist.
Saves: portable checkpoint zips + training logs to /output/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

# ─── Detect hardware ───────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    print("  [!] No GPU detected. Running on CPU (slow).")
print(f"  Torch: {torch.__version__}")
print()

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmap_dataset import get_loader
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer
from train_optimized import quick_eval

# ─── Output directory ──────────────────────────────────────────────
OUT = Path("/output") if os.name != "nt" else Path("gpu_output")
OUT.mkdir(parents=True, exist_ok=True)


def train_one(name: str, data_dir: str, save_dir: Path, steps=20000, batch_size=512, lr=3e-4):
    """Train one specialist on GPU and save checkpoint."""

    # Tokenizer
    tok = MathTokenizer()
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    # Model on GPU
    model = MathTransformer(cfg).to(DEVICE)
    params = count_parameters(model)
    print(f'\n{"="*60}')
    print(f"  Training: {name}")
    print(f"  Params:  {params:,} | Device: {DEVICE}")
    print(f"  Batch:   {batch_size} | Steps: {steps} | LR: {lr:.0e}")
    print(f'{"="*60}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    # Load mmap data (may be CPU-mapped, that's fine)
    loader = get_loader(data_dir, batch_size=batch_size)
    total_batches = len(loader)

    # Warmup + cosine LR
    warmup = 200

    def get_lr(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    global_step = 0
    best_acc = 0.0
    t0 = time.time()
    losses = []
    step_times = []

    while global_step < steps:
        for x, y in loader:
            if global_step >= steps:
                break

            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss, _ = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for pg in optimizer.param_groups:
                pg["lr"] = lr * get_lr(global_step)

            losses.append(loss.item())
            global_step += 1
            step_times.append(time.time())

            # Progress
            if global_step % max(200, steps // 20) == 0:
                now = time.time()
                elapsed = now - t0
                recent = step_times[-20:] if len(step_times) >= 20 else step_times
                rate = (
                    (len(recent) - 1) / max(0.001, recent[-1] - recent[0]) if len(recent) > 1 else 0
                )
                remaining = (steps - global_step) / max(rate, 0.001)
                avg_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1)

                bar = "#" * int(30 * global_step / steps) + "-" * (
                    30 - int(30 * global_step / steps)
                )
                print(
                    f"  [{bar}] {global_step:>6d}/{steps} | loss={avg_loss:.4f} | {rate:.1f} st/s | ETA {int(remaining//60):02d}:{int(remaining%60):02d}"
                )

            # Accuracy eval
            if global_step % max(1000, steps // 5) == 0:
                op_name = name.split()[0].lower()
                op = op_name if op_name in ("add", "sub", "mul", "div") else None
                acc = quick_eval(model, tok, op=op, num=200)
                print(f'  {"":>32} Acc: {acc:.1f}% (best: {max(best_acc, acc):.1f}%)')
                if acc > best_acc:
                    best_acc = acc
                    _save_checkpoint(model, tok, save_dir, global_step, best_acc, params, cfg)
                    _export_portable(save_dir, name)

    # Final save
    _save_checkpoint(model, tok, save_dir, global_step, best_acc, params, cfg, final=True)
    _export_portable(save_dir, name)

    elapsed = time.time() - t0
    print(f"  Done! {elapsed:.0f}s ({elapsed/60:.1f} min) Best: {best_acc:.1f}%")

    # Hard Negative Mining: fine-tune on the model's blind spots
    if name != "Generalist":  # only for specialists
        print(f"\n  --- Hard Negative Mining for {name} ---")
        hnm_acc = hard_negative_finetune(model, tok, save_dir, num_rounds=3)
        print(f"  HNM done! Final acc: {hnm_acc:.1f}%")
        _save_checkpoint(
            model, tok, save_dir, steps, max(best_acc, hnm_acc), params, cfg, final=True
        )
        _export_portable(save_dir, name)

    return model, tok


def _save_checkpoint(model, tok, save_dir, step, acc, params, cfg, final=False):
    """Save model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_dir / "tokenizer.json"))
    name = "final.pt" if final else "best.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "acc": acc,
            "config": {
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "d_ff": cfg.d_ff,
                "params": params,
            },
        },
        save_dir / name,
    )
    print(f"  Saved: {save_dir / name}")


def _export_portable(save_dir: Path, name: str):
    """Package as portable .zip (weights + tokenizer)."""
    safe_name = name.lower().replace(" ", "_")
    zip_path = OUT / f"specialist_{safe_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["best.pt", "tokenizer.json"]:
            p = save_dir / f
            if p.exists():
                z.write(p, f)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Exported: {zip_path} ({size_mb:.1f} MB)")


def hard_negative_finetune(
    model, tok, save_dir, num_rounds=3, hard_ratio=0.3, ft_steps=500, lr=1e-4
):
    """Entropy-guided hard negative fine-tuning on GPU.

    Each round:
    1. Score a batch of training data by softmax entropy
    2. Select the hardest `hard_ratio` fraction
    3. Fine-tune on those high-entropy samples
    """
    print(f"  Hard-neg rounds: {num_rounds} | ratio: {hard_ratio} | steps: {ft_steps}")

    # Load full training data for this specialist
    op_name = str(save_dir).split("/")[-1]
    data_dir = f"specialists/math/{op_name}/data"
    from mmap_dataset import get_loader

    full_loader = get_loader(data_dir, batch_size=512, shuffle=False)

    # Collect all samples into one tensor
    all_x, all_y = [], []
    for x, y in full_loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    n_total = all_x.size(0)
    n_hard = max(1, int(n_total * hard_ratio))
    print(f"  Total samples: {n_total} | Hard samples/round: {n_hard}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model.config.weight_decay)
    best_acc = 0.0

    for rnd in range(num_rounds):
        # ── Score by entropy ──
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(all_x.to(DEVICE))  # (N, T, V)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # (N, T)

            # Per-sample: mean entropy over non-masked positions
            mask = (all_y != -100).to(DEVICE)
            sample_entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Select hardest samples
        _, hard_idx = torch.topk(sample_entropy, n_hard)
        hard_x = all_x[hard_idx.cpu()]
        hard_y = all_y[hard_idx.cpu()]

        print(f"  Round {rnd+1}: mean entropy of hard set = {sample_entropy[hard_idx].mean():.4f}")

        # ── Fine-tune on hard samples ──
        model.train()
        for step in range(ft_steps):
            perm = torch.randperm(n_hard)
            for i in range(0, n_hard, 128):
                idx = perm[i : i + 128]
                bx, by = hard_x[idx].to(DEVICE), hard_y[idx].to(DEVICE)
                _, loss = model(bx, by)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Evaluate
        from train_optimized import quick_eval

        op = op_name if op_name in ("add", "sub", "mul", "div") else None
        acc = quick_eval(model, tok, op=op, num=200)
        print(f"  Round {rnd+1}: acc={acc:.1f}%")
        if acc > best_acc:
            best_acc = acc

    return best_acc


def main():
    prep_ok = all((Path("data") / f).exists() for f in ["inputs.pt", "targets.pt"])
    if not prep_ok:
        print("  Preparing mmap dataset...")
        from prepare_dataset import prepare_dataset

        prepare_dataset()  # generalist
        for op in ["add", "sub", "mul", "div"]:
            prepare_dataset(op)

    # ── Generalist ──
    train_one("Generalist", "data", OUT / "generalist", steps=20000, batch_size=256, lr=3e-4)

    # ── Individual specialists ──
    for op in ["add", "sub", "mul", "div"]:
        data_dir = f"specialists/math/{op}/data"
        if not (Path(data_dir) / "inputs.pt").exists():
            print(f"  Skipping {op} — no data at {data_dir}")
            continue
        is_rev = op in ("add", "sub")
        train_one(
            f'{op} Specialist{" (reversed)" if is_rev else ""}',
            data_dir,
            OUT / op,
            steps=10000,
            batch_size=256,
            lr=3e-4,
        )

    # ── Summary ──
    print(f'\n{"="*60}')
    print(f"  All training complete!")
    print(f"  Output: {OUT}")
    print(f"  Files:")
    for f in sorted(OUT.iterdir()):
        if f.suffix == ".zip":
            print(f"    {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
    print(f'{"="*60}')


if __name__ == "__main__":
    main()
