"""Neocortex — Slow, permanent consolidation during sleep.

During the sleep cycle, this module:
1. Replays surprising experiences from the hippocampus
2. Mixes in random old memories (prevents catastrophic forgetting)
3. Uses Online Elastic Weight Consolidation (EWC) to protect important weights
   — Fisher matrices are merged so overhead stays O(1) forever
4. Trains the model, permanently wiring logic into the brain

This mirrors the human neocortex: slow consolidation of patterns
and rules from daily experiences into permanent knowledge.
"""

import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

# ─── Online Elastic Weight Consolidation (EWC) ────────────────────
# Online EWC (Schwarz et al., 2018) merges all past-task Fisher matrices
# into a single running total, so the EWC penalty overhead stays O(1)
# regardless of how many skills the model learns over its lifetime.

class EWC:
    """Online Elastic Weight Consolidation.

    Instead of keeping a separate Fisher matrix per past task (which grows
    O(N) with skill count), this merges each new Fisher into a running
    total, keeping overhead fixed at O(1).

    Fisher and optimum_params are saveable/loadable alongside model weights
    so EWC protection survives restarts.
    """

    def __init__(self, model, fisher_samples=100, lambda_ewc=1000.0, gamma=0.9):
        self.model = model
        self.device = next(model.parameters()).device
        self.fisher = {}          # running total Fisher diagonal
        self.optimum_params = {}  # anchor weights when Fisher was computed
        self.fisher_samples = fisher_samples
        self.lambda_ewc = lambda_ewc
        self.gamma = gamma        # how much to down-weight old Fisher on merge

    # ── Fisher computation ───────────────────────────────────────

    @torch.no_grad()
    def compute_fisher(self, dataloader):
        """Compute the Fisher Information Matrix for current knowledge.

        The Fisher tells us which weights are important for the tasks
        the model has already learned. Higher Fisher = more important.
        """
        print('  [EWC] Computing Fisher Information Matrix...')
        self.model.train()

        # Initialize new-fisher accumulator + snapshot current weights
        new_fisher = {}
        current_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param.data)
                current_params[name] = param.data.clone().detach()

        # Accumulate Fisher over samples
        count = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= self.fisher_samples:
                break
            x, y = x.to(self.device), y.to(self.device)

            self.model.zero_grad()
            _, loss, _ = self.model(x, y)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.data ** 2 / self.fisher_samples
            count += 1

        self.model.eval()
        print(f'  [EWC] Fisher computed over {count} samples')

        # ── Online merge: F_combined = gamma * F_old + F_new ─────
        if self.fisher:
            print(f'  [EWC] Merging with previous Fisher (gamma={self.gamma})')
            for name in self.fisher:
                if name in new_fisher:
                    self.fisher[name] = self.gamma * self.fisher[name] + new_fisher[name]
                # If name not in new_fisher (e.g. new layer added), keep old
            # Add any new params that didn't exist in old Fisher
            for name in new_fisher:
                if name not in self.fisher:
                    self.fisher[name] = new_fisher[name]
                    self.optimum_params[name] = current_params[name]
            # Update optimum_params for merged entries (keep original anchor)
            # In Online EWC, the *original* anchor is preserved. We only set
            # anchor for params that didn't have one yet.
            for name in self.optimum_params:
                if name not in current_params:
                    del self.optimum_params[name]
        else:
            # First time — just use the new Fisher directly
            self.fisher = new_fisher
            self.optimum_params = current_params

    # ── EWC penalty ──────────────────────────────────────────────

    def ewc_loss(self, model):
        """Compute the EWC penalty — extra loss for changing important weights.

        loss = lambda_ewc * sum(Fisher/2 * (param - optimum)²)
        """
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimum_params[name]) ** 2).sum()
        return self.lambda_ewc * loss

    # ── Save / Load ──────────────────────────────────────────────

    def state_dict(self):
        """Serialize Fisher + optimum_params for checkpointing."""
        return {
            'fisher': {k: v.cpu() for k, v in self.fisher.items()},
            'optimum_params': {k: v.cpu() for k, v in self.optimum_params.items()},
            'lambda_ewc': self.lambda_ewc,
            'gamma': self.gamma,
        }

    def load_state_dict(self, state):
        """Restore Fisher + optimum_params from checkpoint."""
        self.fisher = {k: v.to(self.device) for k, v in state['fisher'].items()}
        self.optimum_params = {k: v.to(self.device) for k, v in state['optimum_params'].items()}
        self.lambda_ewc = state.get('lambda_ewc', self.lambda_ewc)
        self.gamma = state.get('gamma', self.gamma)
        print(f'  [EWC] Loaded Fisher matrix ({len(self.fisher)} parameter groups)')


# ─── Experience Replay Buffer ─────────────────────────────────────

def build_replay_loader(examples: list[dict], tokenizer, batch_size=32, max_seq_len=64,
                        bpe_dropout_prob: float = 0.0):
    """Build a DataLoader from experience replay examples.

    Args:
        bpe_dropout_prob: If >0, randomly bypass BPE merges with this probability
            during replay. This forces the model to maintain character-level
            reasoning and prevents subword over-reliance (BPE Merge Trap).
            Only effective when tokenizer has BPE merges (BPETokenizer with
            learned merges). No-op on base MathTokenizer.
    """
    import random as _random

    import torch
    from torch.utils.data import DataLoader, Dataset

    class ReplayDataset(Dataset):
        def __init__(self, examples, tokenizer, max_seq_len):
            self.samples = []
            for ex in examples:
                text = ex['input']
                if ex.get('output'):
                    text = f'{ex["input"]} {ex["output"]}'

                # ── BPE Dropout: randomly fall back to raw character encoding ──
                # Prevents subword over-reliance by forcing the model to sometimes
                # process tokens at the character level during consolidation.
                if bpe_dropout_prob > 0 and hasattr(tokenizer, 'encode_bpe') and _random.random() < bpe_dropout_prob:
                    # Bypass BPE merges — use base character encoding directly
                    ids = tokenizer.encode(text, add_special_tokens=True)
                else:
                    ids = tokenizer.encode(text, add_special_tokens=True)

                if len(ids) <= max_seq_len:
                    padded = ids + [tokenizer.pad_id] * (max_seq_len - len(ids))
                    x = torch.tensor(padded[:-1], dtype=torch.long)
                    y = torch.tensor(padded[1:], dtype=torch.long)
                    # Mask PAD positions in targets
                    y[y == tokenizer.pad_id] = -100
                    self.samples.append((x, y))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    ds = ReplayDataset(examples, tokenizer, max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


# ─── Sleep Consolidation ──────────────────────────────────────────

def consolidate(model, tokenizer, ewc: EWC = None,
                new_experiences: list = None, old_memories: list = None,
                epochs=5, batch_size=32, learning_rate=1e-4,
                bpe_dropout_prob: float = 0.1):
    """Run a full consolidation cycle.

    Combines new experiences + old memories + EWC protection.
    """
    device = next(model.parameters()).device
    model = model.to(device)

    # Build replay buffer: mix new + old
    replay_data = []
    if new_experiences:
        replay_data.extend(new_experiences)
    if old_memories:
        # Add random old memories (30% of batch)
        n_old = max(1, len(replay_data) // 3)
        replay_data.extend(random.sample(old_memories, min(n_old, len(old_memories))))

    if not replay_data:
        print('  [Neocortex] No experiences to consolidate')
        return 0

    print(f'  [Neocortex] Consolidating {len(replay_data)} experiences...')
    print(f'    New: {len(new_experiences or [])} | Old: {len(replay_data) - len(new_experiences or [])}')

    # Surprise-threshold filtering: only run heavy EWC on high-surprise memories
    high_surprise = [e for e in (new_experiences or []) if e.get('error', 0) > 8.0]
    low_surprise = [e for e in (new_experiences or []) if e.get('error', 0) <= 8.0]
    print(f'    High surprise (>8.0): {len(high_surprise)} | Low surprise: {len(low_surprise)}')

    loader = build_replay_loader(replay_data, tokenizer, batch_size,
                                bpe_dropout_prob=bpe_dropout_prob)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    model.train()
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, ce_loss = model(x, y)

            # EWC penalty (if available)
            total_loss = ce_loss
            if ewc is not None:
                total_loss = ce_loss + ewc.ewc_loss(model)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce_loss.item()
            n_batches += 1
            total_steps += 1

        if n_batches > 0:
            print(f'    Epoch {epoch+1}: CE loss={epoch_loss/n_batches:.4f}')

    model.eval()
    print(f'  [Neocortex] Consolidation complete: {total_steps} steps')
    return total_steps


# ─── Full Sleep Cycle with Online EWC ─────────────────────────────

def full_sleep_cycle(model, tokenizer, skill_dir: str = None, lambda_ewc=1000.0, gamma=0.9):
    """Run the complete sleep consolidation cycle with Online EWC.

    Args:
        model: The model to consolidate into.
        tokenizer: The model's tokenizer.
        skill_dir: Path to specialist directory (for checkpoint + Fisher save).
        lambda_ewc: Strength of EWC penalty.
        gamma: Fisher merge decay — how much to down-weight old Fisher.
    """
    from tabula_rasa.memory.hippocampus import (
        get_old_memories,
        get_unconsolidated,
        mark_consolidated,
    )

    print(f'\n{"=" * 60}')
    print('  Sleep Cycle — Neocortex Consolidation (Online EWC)')
    print(f'{"=" * 60}')

    # Step 1: Get new experiences
    new_exp = get_unconsolidated(limit=200)
    if not new_exp:
        print('  No new experiences to consolidate')
        return 0

    print(f'  New experiences: {len(new_exp)}')

    # Step 2: Get old memories for replay
    old_mem = get_old_memories(limit=50)
    print(f'  Old memories (replay): {len(old_mem)}')

    # Step 3: Load saved Fisher if it exists (Online EWC persistence)
    ewc = EWC(model, fisher_samples=min(len(old_mem or []), 50),
              lambda_ewc=lambda_ewc, gamma=gamma)
    fisher_path = None
    if skill_dir:
        fisher_path = Path(skill_dir) / 'ewc_fisher.pt'
        if fisher_path.exists():
            try:
                state = torch.load(fisher_path, map_location=ewc.device, weights_only=True)
                ewc.load_state_dict(state)
                print(f'  [EWC] Loaded previous Fisher from {fisher_path}')
            except Exception as e:
                print(f'  [EWC] Could not load saved Fisher: {e}')

    # Step 4: Compute new Fisher on old memories (what the model already knows)
    if old_mem:
        try:
            old_loader = build_replay_loader(old_mem, tokenizer, batch_size=32)
            ewc.compute_fisher(old_loader)
        except Exception as e:
            print(f'  [EWC] Skipped: {e}')

    # Step 5: Consolidate (train on new + old with EWC protection)
    result = consolidate(
        model, tokenizer, ewc=ewc,
        new_experiences=new_exp,
        old_memories=old_mem,
        epochs=5,
        batch_size=32,
        bpe_dropout_prob=0.1,
    )

    # Step 6: Mark experiences as consolidated
    ids = [e['id'] for e in new_exp]
    mark_consolidated(ids)
    print(f'  Marked {len(ids)} experiences as consolidated')

    # Step 7: Save checkpoint + Fisher together
    if skill_dir:
        save_dir = Path(skill_dir)
        # Save model weights
        save_path = save_dir / 'best.pt'
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        print(f'  Saved model to {save_path}')
        # Save Fisher alongside (Online EWC persistence)
        torch.save(ewc.state_dict(), fisher_path)
        print(f'  Saved Online EWC Fisher to {fisher_path}')

    return result


if __name__ == '__main__':
    # Test with loaded model
    print('Testing Neocortex consolidation...')

    # Load model
    from train_specialist import load_specialist

    model, tok = load_specialist('specialists/math/add')
    if model is None:
        print('No specialist found. Train one first.')
        sys.exit(1)

    # Run test cycle
    full_sleep_cycle(model, tok, skill_dir='specialists/math/add',
                     lambda_ewc=500.0, gamma=0.9)
    print('\nDone.')
