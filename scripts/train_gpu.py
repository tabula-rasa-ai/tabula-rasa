"""GPU Training Runner — uses all enhancements: LoRA, AMP, MoE, RASA, new benchmarks.

Usage on cloud GPU (RunPod, Vast.ai, etc.):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    python scripts/train_gpu.py                          # defaults
    python scripts/train_gpu.py --moe --num-experts 4    # MoE
    python scripts/train_gpu.py --lora --rank 16         # LoRA fine-tune
    python scripts/train_gpu.py --rasa                    # RASA attention
    python scripts/train_gpu.py --preset 10M --steps 50000
"""

import argparse
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

# ─── Hardware detection ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name}  ({mem_gb:.1f} GB)")
else:
    print("  [!] No GPU detected. Running on CPU (very slow).")
print(f"  Torch: {torch.__version__}")
print()

# Lazy imports — these modules are in the project root, loaded inside functions
# to avoid import-order issues when this script is imported vs run directly.
# from mmap_dataset import get_loader
# from tabula_rasa.config import Config
# from tabula_rasa.lora import apply_lora_to_model, save_lora_adapters, set_lora_trainable
# from tabula_rasa.model import MathTransformer, count_parameters
# from tabula_rasa.tokenizer import MathTokenizer

OUT = Path("/output") if os.name != "nt" else Path("gpu_output")
OUT.mkdir(parents=True, exist_ok=True)


def create_model(cfg, use_lora=False, lora_rank=8, lora_alpha=1.0, quantize_bits=0):
    """Create model with optional LoRA and quantization. Move to GPU immediately."""
    from tabula_rasa.lora import apply_lora_to_model, set_lora_trainable
    model = MathTransformer(cfg).to(DEVICE)

    # Quantize if requested (requires bitsandbytes on CUDA)
    if quantize_bits in (4, 8) and DEVICE == "cuda":
        try:
            model.quantize_(bits=quantize_bits)
            print(f"  [*] Quantized to {quantize_bits}-bit")
        except ImportError as e:
            print(f"  [!] Quantization skipped: {e}")
    elif quantize_bits in (4, 8):
        print("  [!] Quantization requires CUDA. Skipping.")

    # StreamBP: selective attention checkpointing (built into TransformerBlock)
    if getattr(cfg, "use_streambp", True):
        print("  [*] StreamBP enabled (attention checkpointing)")

    # LoRA
    lora_layers = []
    if use_lora:
        targets = getattr(cfg, "lora_target_modules", "wq,wk,wv,wo").split(",")
        lora_layers = apply_lora_to_model(model, rank=lora_rank, alpha=lora_alpha,
                                          target_modules=targets)
        set_lora_trainable(model, trainable=True)
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [*] LoRA: rank={lora_rank}, alpha={lora_alpha}, trainable={n_lora:,}")

    return model, lora_layers


def get_lr_schedule(step, warmup, total_steps, base_lr):
    """Cosine LR with linear warmup."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, tok, op=None, num=200, max_digits=4):
    """Quick accuracy eval — uses greedy decoding, reports accuracy."""
    from tabula_rasa.dataset import generate_problem
    model.eval()
    correct = 0
    for _ in range(num):
        expr, ans = generate_problem(1, max_digits)
        if op and op not in expr:
            continue
        prompt = f"{expr}="
        gen = model.generate(tok, prompt, max_new_tokens=12, temperature=0.0, top_k=1)
        if "=" in gen:
            pred = gen.split("=")[-1].strip()
            pred = "".join(c for c in pred if c.isdigit() or c == "-")
            if pred == ans:
                correct += 1
    acc = correct / num * 100
    model.train()
    return acc


def run_full_benchmark(model, tok, name, max_digits=4):
    """Run the full 8-dimension benchmark suite and save results."""
    print(f"\n  --- Full benchmark: {name} ---")
    try:
        from tabula_rasa.eval import full_benchmark
        results = full_benchmark(model, tok, train_max_digits=max_digits, verbose=True)

        summary = {}
        for k, v in results.items():
            if hasattr(v, "accuracy"):
                summary[k] = round(v.accuracy, 1)
            elif hasattr(v, "overall_accuracy"):
                summary[k] = round(v.overall_accuracy, 1)
        for adv_key in ["adversarial/boundary", "adversarial/format_attacks", "adversarial/noise", "adversarial/fgsm"]:
            if adv_key in results:
                summary[adv_key.replace("/", "_")] = round(results[adv_key].accuracy, 1)

        bench_path = OUT / f"benchmark_{name.lower().replace(' ', '_')}.json"
        with open(bench_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Benchmark saved: {bench_path}")
        return summary
    except Exception as e:
        print(f"  Benchmark skipped: {e}")
        return {}


def train_one(name, data_dir, save_dir, steps=20000, batch_size=512, lr=3e-4,
              use_lora=False, lora_rank=8, lora_alpha=1.0, quantize_bits=0):
    """Train one specialist on GPU with all enhancements."""
    from mmap_dataset import get_loader

    from tabula_rasa.config import Config
    from tabula_rasa.model import count_parameters
    from tabula_rasa.tokenizer import MathTokenizer
    tok = MathTokenizer()
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model, lora_layers = create_model(cfg, use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha, quantize_bits=quantize_bits)
    params = count_parameters(model)

    # Op detection
    op_map = {"generalist": None, "add": "add", "sub": "sub", "mul": "mul", "div": "div"}
    op = None
    for k, v in op_map.items():
        if k in name.lower():
            op = v
            break

    print(f'\n{"="*60}')
    print(f"  Training: {name}")
    print(f"  Params:  {params:,} | Device: {DEVICE}")
    print(f"  Batch:   {batch_size} | Steps: {steps} | LR: {lr:.0e}")
    print(f"  Config:  d={cfg.d_model} L={cfg.n_layers} h={cfg.n_heads} ff={cfg.d_ff}")
    print(f"  Features: LoRA={use_lora} MoE={cfg.use_moe} RASA={cfg.use_rasa}")
    print(f'{"="*60}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    # AMP scaler (only on CUDA)
    use_amp = DEVICE == "cuda" and getattr(cfg, "use_amp", True)
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=use_amp) if use_amp else None

    # Data
    loader = get_loader(data_dir, batch_size=batch_size)
    warmup = min(200, steps // 10)

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
            lr_current = get_lr_schedule(global_step, warmup, steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_current

            # Mixed precision forward
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast(device_type=DEVICE):
                    _, loss, _ = model(x, y)
                    moe_aux = model.get_moe_aux_loss() * 0.01 if cfg.use_moe else 0.0
                    total_loss = loss + moe_aux
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss, _ = model(x, y)
                moe_aux = model.get_moe_aux_loss() * 0.01 if cfg.use_moe else 0.0
                total_loss = loss + moe_aux
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            losses.append(loss.item())
            global_step += 1
            step_times.append(time.time())

            # Progress
            if global_step % max(200, steps // 20) == 0:
                now = time.time()
                elapsed = now - t0
                recent_times = step_times[-20:] if len(step_times) >= 20 else step_times
                rate = (len(recent_times) - 1) / max(0.001, recent_times[-1] - recent_times[0]) if len(recent_times) > 1 else 0
                remaining = (steps - global_step) / max(rate, 0.001)
                avg_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1)
                bar = "#" * int(30 * global_step / steps) + "-" * (30 - int(30 * global_step / steps))
                print(f"  [{bar}] {global_step:>6d}/{steps} | loss={avg_loss:.4f} | {rate:.1f} st/s | LR {lr_current:.2e} | ETA {int(remaining//60):02d}:{int(remaining%60):02d}")

            # Accuracy eval
            if global_step % max(1000, steps // 5) == 0:
                acc = evaluate(model, tok, op=op, num=200, max_digits=cfg.max_digits)
                print(f'  {"":>32} Acc: {acc:.1f}% (best: {max(best_acc, acc):.1f}%)')
                if acc > best_acc:
                    best_acc = acc
                    _save_checkpoint(model, tok, save_dir, global_step, best_acc, params, cfg, lora_layers)
                    _export_portable(save_dir, name)

    # Final save
    _save_checkpoint(model, tok, save_dir, global_step, best_acc, params, cfg, lora_layers, final=True)
    _export_portable(save_dir, name)

    elapsed = time.time() - t0
    print(f"  Done! {elapsed:.0f}s ({elapsed/60:.1f} min) Best: {best_acc:.1f}%")

    # Hard Negative Mining
    if op:
        print(f"\n  --- Hard Negative Mining for {name} ---")
        hnm_acc = hard_negative_finetune(model, tok, save_dir, op=op, num_rounds=3)
        print(f"  HNM done! Final acc: {hnm_acc:.1f}%")
        _save_checkpoint(model, tok, save_dir, steps, max(best_acc, hnm_acc), params, cfg, lora_layers, final=True)
        _export_portable(save_dir, name)
        best_acc = max(best_acc, hnm_acc)

    # Full benchmark
    bench_results = run_full_benchmark(model, tok, name, max_digits=cfg.max_digits)

    # Auto-log experiment
    try:
        from tabula_rasa.experiments import log_run
        log_run(
            name=name.lower().replace(" ", "_"),
            config={
                "preset": cfg.config_preset,
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads, "d_ff": cfg.d_ff,
                "max_seq_len": cfg.max_seq_len,
                "use_moe": cfg.use_moe, "num_experts": cfg.num_experts,
                "use_rasa": cfg.use_rasa,
                "use_lora": use_lora, "lora_rank": lora_rank,
                "use_streambp": cfg.use_streambp,
                "batch_size": batch_size, "learning_rate": lr, "max_steps": steps,
            },
            metrics=bench_results,
            checkpoint=str(save_dir / "best.pt"),
            notes=f"GPU train: {name}",
            duration_s=elapsed,
        )
    except Exception as e:
        print(f"  [!] Experiment logging skipped: {e}")

    return model, tok


def _save_checkpoint(model, tok, save_dir, step, acc, params, cfg, lora_layers=None, final=False):
    """Save model checkpoint + optional LoRA adapters."""
    from tabula_rasa.lora import save_lora_adapters
    save_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_dir / "tokenizer.json"))
    name = "final.pt" if final else "best.pt"
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "acc": acc,
        "config": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "d_ff": cfg.d_ff,
            "params": params, "use_moe": cfg.use_moe, "use_rasa": cfg.use_rasa,
        },
    }
    if lora_layers:
        lora_name = "lora_final.pt" if final else "lora_best.pt"
        save_lora_adapters(lora_layers, str(save_dir / lora_name))
    torch.save(state, save_dir / name)
    print(f"  Saved: {save_dir / name}")


def _export_portable(save_dir, name):
    """Package as portable .zip (weights + tokenizer + optional LoRA)."""
    safe_name = name.lower().replace(" ", "_")
    zip_path = OUT / f"specialist_{safe_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["best.pt", "tokenizer.json", "lora_best.pt", "lora_final.pt"]:
            p = save_dir / f
            if p.exists():
                z.write(p, f)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Exported: {zip_path} ({size_mb:.1f} MB)")


def hard_negative_finetune(model, tok, save_dir, op=None, num_rounds=3, hard_ratio=0.3, ft_steps=500, lr=1e-4):
    """Entropy-guided hard negative fine-tuning on GPU."""
    from mmap_dataset import get_loader
    print(f"  Hard-neg rounds: {num_rounds} | ratio: {hard_ratio} | steps: {ft_steps}")
    op_name = str(save_dir).split("/")[-1] if op is None else op
    data_dir = f"specialists/math/{op_name}/data"
    if not (Path(data_dir) / "inputs.pt").exists():
        print(f"  No data at {data_dir}, skipping HNM")
        return 0.0

    full_loader = get_loader(data_dir, batch_size=512, shuffle=False)
    all_x, all_y = [], []
    for x, y in full_loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    n_total = all_x.size(0)
    n_hard = max(1, int(n_total * hard_ratio))
    print(f"  Total: {n_total} | Hard/round: {n_hard}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model.config.weight_decay)
    best_acc = 0.0

    for rnd in range(num_rounds):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(all_x.to(DEVICE))
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)
            mask = (all_y != -100).to(DEVICE)
            sample_entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        _, hard_idx = torch.topk(sample_entropy, n_hard)
        hard_x, hard_y = all_x[hard_idx.cpu()], all_y[hard_idx.cpu()]
        print(f"  Round {rnd+1}: entropy={sample_entropy[hard_idx].mean():.4f}")

        model.train()
        for step in range(ft_steps):
            perm = torch.randperm(n_hard)
            for i in range(0, n_hard, 128):
                idx = perm[i:i+128]
                bx, by = hard_x[idx].to(DEVICE), hard_y[idx].to(DEVICE)
                _, loss, _ = model(bx, by)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        acc = evaluate(model, tok, op=op_name, num=200)
        print(f"  Round {rnd+1}: acc={acc:.1f}%")
        if acc > best_acc:
            best_acc = acc
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="GPU Training Runner")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--preset", type=str, default="1M", choices=["1M", "5M", "10M"],
                        help="Architecture preset")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--moe", action="store_true", help="Enable MoE layers")
    parser.add_argument("--num-experts", type=int, default=4, help="MoE expert count")
    parser.add_argument("--top-k", type=int, default=2, help="MoE top-K routing")
    parser.add_argument("--rasa", action="store_true", help="Enable RASA attention bias")
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                        help="Quantize model to 4-bit or 8-bit (requires bitsandbytes on GPU)")
    parser.add_argument("--generalist-only", action="store_true", help="Train generalist only")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip full benchmark")
    args = parser.parse_args()

    if hasattr(args, 'history') and args.history:
        from tabula_rasa.experiments import list_runs, print_runs
        print_runs(list_runs(limit=30))
        sys.exit(0)

    # Apply preset to Config
    cfg = Config()
    cfg.apply_preset(args.preset)
    if args.moe:
        cfg.use_moe = True
        cfg.num_experts = args.num_experts
        cfg.top_k = args.top_k
    if args.rasa:
        cfg.use_rasa = True
    # Save config for model creation
    cfg.batch_size = args.batch
    cfg.learning_rate = args.lr
    cfg.max_steps = args.steps

    # Prepare datasets if needed
    prep_ok = all((Path("data") / f).exists() for f in ["inputs.pt", "targets.pt"])
    if not prep_ok:
        print("  Preparing mmap dataset...")
        from prepare_dataset import prepare_dataset
        prepare_dataset()
        for op in ["add", "sub", "mul", "div"]:
            prepare_dataset(op)

    # ── Generalist ──
    train_one("Generalist", "data", OUT / "generalist",
              steps=args.steps, batch_size=args.batch, lr=args.lr,
              use_lora=args.lora, lora_rank=args.lora_rank,
              quantize_bits=args.quantize)

    if not args.generalist_only:
        for op in ["add", "sub", "mul", "div"]:
            data_dir = f"specialists/math/{op}/data"
            if not (Path(data_dir) / "inputs.pt").exists():
                print(f"  Skipping {op} — no data at {data_dir}")
                continue
            train_one(f"{op} Specialist", data_dir, OUT / op,
                      steps=args.steps // 2, batch_size=args.batch, lr=args.lr,
                      use_lora=args.lora, lora_rank=args.lora_rank,
                      quantize_bits=args.quantize)

    # ── Summary ──
    print(f'\n{"="*60}')
    print("  All training complete!")
    print(f"  Output: {OUT}")
    print(f"  Config: preset={args.preset} MoE={args.moe} RASA={args.rasa} LoRA={args.lora}")
    for f in sorted(OUT.iterdir()):
        if f.suffix == ".zip":
            print(f"    {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
    print(f'{"="*60}')


if __name__ == "__main__":
    main()
