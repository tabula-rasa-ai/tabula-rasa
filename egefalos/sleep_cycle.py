"""Sleep cycle: consolidate experiences into long-term memory via Online EWC.

During sleep:
1. Sample high-surprise experiences from hippocampus
2. Compute Fisher information on replay data
3. Merge into running EWC total
4. Train with EWC penalty to consolidate without forgetting
"""
import sys
from pathlib import Path
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Optional
import argparse
import time

from tabula_rasa.model import MathTransformer
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from egefalos.online_ewc import OnlineEWC
from egefalos.hippocampus import get_unconsolidated, get_stats, mark_consolidated, decay_memories


def consolidate_sleep_cycle(
    model_path: Path,
    tokenizer: MathTokenizer,
    config: Config,
    num_samples: int = 2048,
    epochs: int = 3,
    lambda_ewc: float = 1000.0,
    gamma: float = 0.9,
    output_dir: Optional[Path] = None,
    max_seq_len: int = 32,
):
    """Run one sleep cycle: replay + EWC consolidation.

    Args:
        model_path: Path to the specialist's best.pt checkpoint
        tokenizer: MathTokenizer instance
        config: Model configuration
        num_samples: Max replay experiences to consume
        epochs: Training epochs over replay buffer
        lambda_ewc: EWC penalty strength
        gamma: Fisher merge decay rate
        output_dir: Specialist directory (default: parent of model_path)
        max_seq_len: Max token sequence length
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = get_stats()

    print(f"\n{'='*60}")
    print(f"  SLEEP CYCLE — Consolidation")
    print(f"  Device: {device}")
    print(f"  Hippocampus: {stats['total_experiences']} total, {stats['unconsolidated']} unconsolidated")
    print(f"  Avg prediction error: {stats['avg_error']}")
    print(f"{'='*60}")

    # Load model
    if not model_path.exists():
        print(f"  [Sleep] Model not found: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    sd = checkpoint.get('model_state_dict', checkpoint)

    # Auto-detect architecture
    d_model = sd['token_embedding.weight'].shape[1]
    n_layers = len([k for k in sd if k.startswith('layers.') and k.endswith('.attention.wq.weight')])
    d_ff = sd['layers.0.feed_forward.w1.weight'].shape[0] if n_layers > 0 else 256
    n_heads = {128: 4, 64: 2, 96: 4, 256: 8}.get(d_model, max(1, d_model // 32))

    cfg = Config()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.d_ff = d_ff
    cfg.n_heads = n_heads
    cfg.max_seq_len = max_seq_len

    model = MathTransformer(cfg)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.train()

    if output_dir is None:
        output_dir = model_path.parent

    # Initialize EWC
    ewc = OnlineEWC(model, gamma=gamma)
    ewc_path = output_dir / 'ewc_fisher.pt'
    if ewc_path.exists():
        ewc.load(ewc_path)
        print(f"  [Sleep] Loaded EWC from {ewc_path} ({ewc.task_count} prior tasks)")

    # Get unconsolidated experiences
    experiences = get_unconsolidated(limit=num_samples)
    if not experiences:
        print("  [Sleep] No unconsolidated experiences. Nothing to do.")
        return

    print(f"  [Sleep] Loaded {len(experiences)} high-surprise experiences")

    # Tokenize experiences into training batches
    input_ids_list = []
    targets_list = []

    for exp in experiences:
        text = exp.get('input', '') or ''
        try:
            ids = tokenizer.encode(text, add_special_tokens=True)
        except Exception:
            continue

        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        # Create targets: -100 for prompt tokens (ignored in loss)
        # The answer is everything after the last separator or '='
        # Simple heuristic: last third of tokens is the answer
        split_point = max(1, len(ids) * 2 // 3)
        targets = [-100] * split_point + ids[split_point:]

        # Pad to max_seq_len
        pad_len = max_seq_len - len(ids)
        padded_ids = ids + [tokenizer.pad_id] * pad_len
        padded_targets = targets + [-100] * pad_len

        input_ids_list.append(padded_ids[:max_seq_len])
        targets_list.append(padded_targets[:max_seq_len])

    if not input_ids_list:
        print("  [Sleep] No valid experiences after tokenization.")
        return

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)

    dataset = TensorDataset(input_ids_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=min(32, len(input_ids_list)), shuffle=True)

    # Phase 1: Compute Fisher on replay data
    print("  [Sleep] Phase 1: Computing Fisher Information Matrix...")
    new_fisher = ewc.compute_fisher(dataloader, num_samples=len(dataloader))
    print(f"  [Sleep] Fisher computed across {len(new_fisher)} parameter groups")

    # Phase 2: Merge Fisher
    print("  [Sleep] Phase 2: Merging into running Fisher total...")
    ewc.merge_fisher(new_fisher)
    ewc.save_anchor_weights()

    # Phase 3: Train with EWC penalty
    print(f"  [Sleep] Phase 3: Training with EWC penalty (λ={lambda_ewc})...")
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_penalty = 0.0
        num_batches = 0

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward
            _, task_loss, _ = model(input_ids, targets=targets)

            # EWC penalty
            ewc_penalty = ewc.compute_ewc_penalty(lambda_ewc=lambda_ewc)

            # Total loss
            total_loss = task_loss + ewc_penalty

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += task_loss.item()
            epoch_penalty += ewc_penalty.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_penalty = epoch_penalty / max(num_batches, 1)
        print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  ewc_penalty={avg_penalty:.4f}")

    # Phase 4: Save
    print("  [Sleep] Phase 4: Saving consolidated model and EWC state...")

    # Save consolidated checkpoint as best.pt
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': cfg.d_model, 'n_layers': cfg.n_layers,
            'd_ff': cfg.d_ff, 'n_heads': cfg.n_heads,
            'max_seq_len': cfg.max_seq_len,
        },
        'consolidation_step': ewc.consolidation_steps,
        'ewc_task_count': ewc.task_count,
    }, model_path)

    # Save EWC state
    ewc.consolidation_steps += 1
    ewc.save(ewc_path)

    # Mark experiences as consolidated
    exp_ids = [e['id'] for e in experiences if 'id' in e]
    if exp_ids:
        mark_consolidated(exp_ids)

    # Synaptic decay — demote/forget unused memories, clean up tiers
    decay_result = decay_memories(min_access=3, max_age_days=7,
                                  demote_working=True, cleanup_active=True)
    print(f'  Synaptic decay: {decay_result}')

    print(f"  [Sleep] Consolidation complete.")
    print(f"  [Sleep] Saved to {output_dir}")
    print(f"  [Sleep] Consolidated {len(exp_ids)} experiences")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sleep cycle consolidation')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to specialist checkpoint (best.pt)')
    parser.add_argument('--num-samples', type=int, default=2048,
                        help='Max replay samples to consume')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs over replay buffer')
    parser.add_argument('--lambda-ewc', type=float, default=1000.0,
                        help='EWC penalty strength')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Fisher merge decay rate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Specialist output directory')
    parser.add_argument('--max-seq-len', type=int, default=32,
                        help='Max token sequence length')
    args = parser.parse_args()

    config = Config()
    tok = MathTokenizer()
    tok.max_seq_len = args.max_seq_len

    output_dir = Path(args.output_dir) if args.output_dir else None

    consolidate_sleep_cycle(
        model_path=Path(args.model),
        tokenizer=tok,
        config=config,
        num_samples=args.num_samples,
        epochs=args.epochs,
        lambda_ewc=args.lambda_ewc,
        gamma=args.gamma,
        output_dir=output_dir,
        max_seq_len=args.max_seq_len,
    )
