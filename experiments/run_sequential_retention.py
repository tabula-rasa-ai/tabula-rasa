"""EWC Sequential Retention Test — uses the proven train_specialist pipeline.

1. Uses existing addition best.pt (100% 1-digit)
2. Runs sleep cycle to compute Fisher
3. Loads addition model, trains on subtraction WITH EWC
4. Measures addition retention
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from egefalos.hippocampus import clear as clear_hippo
from egefalos.hippocampus import store_experience
from egefalos.online_ewc import OnlineEWC
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer
from train_specialist import (
    SpecialistDataset,
    _get_lr,
    _make_optimizer,
    evaluate,
    evaluate_per_digit,
    generate_problem,
)


def main():
    print("=" * 60)
    print("  EWC SEQUENTIAL RETENTION VALIDATION")
    print("  Using proven addition checkpoint (best.pt)")
    print("=" * 60)

    device = "cpu"
    tok = MathTokenizer()
    add_dir = Path("specialists/math/add")
    results = {}

    # ─── Load addition model ───
    print("\n--- Loading addition checkpoint ---")
    checkpoint = torch.load(add_dir / "best.pt", map_location="cpu", weights_only=True)
    sd = checkpoint.get("model_state_dict", checkpoint)

    d_model = sd["token_embedding.weight"].shape[1]
    n_layers = len(
        [k for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight")]
    )
    d_ff = sd["layers.0.feed_forward.w1.weight"].shape[0] if n_layers > 0 else 256
    n_heads = {128: 4, 64: 2, 96: 4, 256: 8}.get(d_model, max(1, d_model // 32))

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.d_ff = d_ff
    cfg.n_heads = n_heads
    cfg.max_seq_len = 32
    cfg.use_reversed = True
    cfg.use_scratchpad = True
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    model.load_state_dict(sd, strict=False)
    model.train()  # Put in train mode for continued training

    # ─── Phase 1: Baseline accuracy ───
    print("\n--- Phase 1: Baseline ---")
    cfg.max_digits = 1
    cfg.min_digits = 1
    add_baseline = evaluate(model, tok, cfg, "add", num=100)
    results["addition_baseline_1d"] = add_baseline
    print(f"  1-digit addition: {add_baseline:.1f}%")

    per_digit = evaluate_per_digit(model, tok, cfg, "add", per_digit_samples=20)
    digit_str = " ".join(f"{d}d:{v:.0f}%" for d, v in per_digit.items())
    print(f"  Per digit: {digit_str}")

    # ─── Phase 2: Compute Fisher on addition ───
    print("\n--- Phase 2: Compute Fisher ---")
    clear_hippo()

    # Store addition problems in hippocampus for sleep cycle
    cfg.max_digits = 1
    cfg.min_digits = 1
    model.eval()
    with torch.no_grad():
        for i in range(100):
            expr, ans = generate_problem("add", 1, 1, reversed=True, scratchpad=True)
            full = f"{expr}={ans}"
            ids = tok.encode(full, add_special_tokens=True)
            ids_t = torch.tensor([ids])
            model_cpu = model.cpu() if device != "cpu" else model
            _, loss, _ = model(ids_t[:, :-1], ids_t[:, 1:])
            store_experience(full, loss.item(), ans, "math")
    model.train()

    # Use the sleep cycle to compute and merge Fisher
    from egefalos.hippocampus import get_stats
    from egefalos.sleep_cycle import consolidate_sleep_cycle

    print(f"  Hippocampus: {get_stats()}")

    consolidate_sleep_cycle(
        model_path=add_dir / "best.pt",
        tokenizer=tok,
        config=cfg,
        num_samples=100,
        epochs=2,
        lambda_ewc=1000.0,
        gamma=0.9,
        output_dir=add_dir,
        max_seq_len=cfg.max_seq_len,
    )

    # ─── Phase 3: Train on subtraction with EWC ───
    print("\n--- Phase 3: Train Subtraction WITH EWC ---")

    # Reload the addition model fresh (sleep cycle saved best.pt)
    checkpoint = torch.load(add_dir / "best.pt", map_location="cpu", weights_only=True)
    sd2 = checkpoint.get("model_state_dict", checkpoint)
    model2 = MathTransformer(cfg)
    model2.load_state_dict(sd2, strict=False)
    model2.train()

    # Load EWC state
    ewc = OnlineEWC(model2, gamma=0.9)
    ewc_path = add_dir / "ewc_fisher.pt"
    if ewc_path.exists():
        ewc.load(ewc_path)
        print(f"  EWC loaded: {ewc.task_count} tasks, {len(ewc.fisher_dict)} fisher params")
    ewc.save_anchor_weights()

    # Train on subtraction
    optimizer = torch.optim.AdamW(
        model2.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
    )

    steps = 2000
    t_start = time.time()

    # Build subtraction dataset
    from train_specialist import SpecialistDataset as SpecDS

    max_digits_orig = cfg.max_digits
    cfg.max_digits = 1
    cfg.min_digits = 1
    sub_ds = SpecDS(tok, "sub", cfg)
    cfg.max_digits = max_digits_orig
    loader = DataLoader(sub_ds, batch_size=32, shuffle=True, drop_last=True)

    global_step = 0
    for epoch in range(20):
        for x, y in loader:
            if global_step >= steps:
                break

            _, task_loss, _ = model2(x, y)
            ewc_pen = ewc.compute_ewc_penalty(lambda_ewc=1000.0)
            total_loss = task_loss + ewc_pen

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
            optimizer.step()

            for pg in optimizer.param_groups:
                pg["lr"] = 0.001 * _get_lr(global_step, 200, steps, "cosine")

            global_step += 1

        if epoch % 2 == 0:
            model2.eval()
            add_a = evaluate(model2, tok, cfg, "add", num=50)
            sub_a = evaluate(model2, tok, cfg, "sub", num=50)
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1}: add={add_a:.1f}% sub={sub_a:.1f}% [{elapsed:.0f}s]")
            model2.train()

    elapsed = time.time() - t_start
    print(f"  Training done: {elapsed:.0f}s, {global_step} steps")

    # ─── Phase 4: Final measurements ───
    print("\n--- Phase 4: Results ---")
    model2.eval()

    add_final = evaluate(model2, tok, cfg, "add", num=100)
    sub_final = evaluate(model2, tok, cfg, "sub", num=100)
    per_digit_final = evaluate_per_digit(model2, tok, cfg, "add", per_digit_samples=20)
    digit_str_final = " ".join(f"{d}d:{v:.0f}%" for d, v in per_digit_final.items())

    results["addition_final_1d"] = add_final
    results["subtraction_final_1d"] = sub_final
    results["retention_drop_pp"] = add_baseline - add_final
    results["retention_preserved"] = (add_baseline - add_final) < 5.0
    results["per_digit_final"] = {str(k): v for k, v in per_digit_final.items()}

    print(f"  Addition baseline:          {add_baseline:.1f}%")
    print(f"  After EWC + subtraction:    {add_final:.1f}%")
    print(f"  Retention drop:             {add_baseline - add_final:.1f} pp")
    print(f"  Subtraction acquired:       {sub_final:.1f}%")
    print(f"  Per digit:                  {digit_str_final}")
    print(
        f"  Retention preserved (<5pp): {'YES ✅' if (add_baseline - add_final) < 5.0 else 'NO ❌'}"
    )

    # ─── Save ───
    output_path = Path("experiments/sequential_retention_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {output_path}")


if __name__ == "__main__":
    main()
