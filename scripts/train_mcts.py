"""Online MCTS training loop — self-play via reward signal.

Replaces static dataset training with outcome-based reinforcement learning:
  Generate → MCTS explore → Reward (+1/-1) → Train → Repeat

This is GRPO / RLVR — the same approach behind DeepSeek-R1's math performance.
The value head (config.use_value_head) learns to predict the outcome.

Usage:
    python3 scripts/train_mcts.py add --steps 1000                  # Online training
    python3 scripts/train_mcts.py add --steps 2000 --preset 10M     # 10M-param model
    python3 scripts/train_mcts.py add --steps 500 --moe             # With MoE layer
"""

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from egefalos.math_gym_env import MathGymEnv
from egefalos.mcts import MicroMCTS
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer


class MCTSTrainer:
    """Online MCTS training loop with outcome-based reward.

    Each step: generate problem → MCTS explore → get reward → train → repeat.
    """

    def __init__(self, model, tokenizer, config, device, op='add',
                 mcts_simulations=16, max_digits=2):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.op = op
        self.max_digits = max_digits

        self.env = MathGymEnv(
            tokenizer=tokenizer,
            op=op,
            max_digits=max_digits,
            max_seq_len=config.max_seq_len,
        )

        self.mcts = MicroMCTS(
            model=model,
            tokenizer=tokenizer,
            num_simulations=mcts_simulations,
            c_puct=1.0,
            discount=0.99,
            device=device,
        )

    def train_step(self, optimizer, criterion) -> dict:
        """One training step: generate problem, MCTS search, train.

        Returns metrics dict.
        """
        # Reset environment → get a fresh problem
        obs, info = self.env.reset()
        problem_text = info.get('problem', '')
        expected = info.get('answer', '')

        # Run MCTS to find the best trajectory
        input_ids = obs['input_ids'].unsqueeze(0).to(self.device)
        trajectory, values = self.mcts.search(input_ids)

        # Decode trajectory to get model's answer
        if trajectory:
            decoded = self.tokenizer.decode(trajectory, skip_special=True)
        else:
            decoded = ''

        # Compute reward: +1 if correct, -1 if wrong, 0 if partial
        reward = 1.0 if decoded.strip() == expected.strip() else -1.0

        # Train on this trajectory using reward as target for value head
        self.model.train()
        self.model.zero_grad()

        # Forward pass
        traj_tensor = torch.tensor([trajectory], device=self.device) if trajectory else input_ids
        logits, ce_loss, extras = self.model(traj_tensor, targets=traj_tensor)

        # Value head loss: MSE against reward
        value_loss = torch.tensor(0.0, device=self.device)
        if self.model.value_head is not None:
            # Get value from model's forward (uses last token's hidden state internally)
            _, _, value_out = self.model(traj_tensor, targets=traj_tensor)
            if value_out is not None:
                pred_value = value_out.squeeze(-1)
                target_value = torch.tensor([reward], device=self.device)
                value_loss = nn.MSELoss()(pred_value, target_value)

        # Combined loss: cross-entropy + value head + aux load-balancing (if MoE)
        total_loss = ce_loss + 0.5 * value_loss

        # Add MoE auxiliary loss if applicable
        for layer in self.model.layers:
            ff = layer.feed_forward
            if hasattr(ff, 'last_aux_loss') and ff.last_aux_loss != 0:
                total_loss = total_loss + 0.01 * ff.last_aux_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return {
            'loss': ce_loss.item(),
            'value_loss': value_loss.item(),
            'reward': reward,
            'decoded': decoded,
            'expected': expected,
            'correct': reward == 1.0,
        }


def main():
    parser = argparse.ArgumentParser(description='Online MCTS training loop')
    parser.add_argument('op', type=str, nargs='?', default='add',
                        help='Operation to train')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--preset', type=str, default='1M',
                        choices=['1M', '10M'], help='Architecture preset')
    parser.add_argument('--moe', action='store_true', help='Enable MoE layers')
    parser.add_argument('--max-digits', type=int, default=2)
    parser.add_argument('--mcts-sims', type=int, default=16,
                        help='MCTS simulations per search')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.steps = 100
        args.eval_every = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config
    cfg = Config().apply_preset(args.preset)
    cfg.max_seq_len = max(cfg.max_seq_len, 64)
    cfg.use_value_head = True  # Need value head for MCTS
    cfg.use_moe = args.moe
    cfg.use_reversed = False
    cfg.use_loss_masking = True

    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size

    model = MathTransformer(cfg).to(device)
    print(f'\n{"="*60}')
    print('  ONLINE MCTS TRAINING')
    print(f'  Op: {args.op} | Preset: {args.preset} | MoE: {args.moe}')
    print(f'  Device: {device} | Params: {count_parameters(model):,}')
    print(f'  MCTS sims: {args.mcts_sims} | Steps: {args.steps} | LR: {args.lr}')
    print(f'  Max digits: {args.max_digits}')
    print(f'{"="*60}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = MCTSTrainer(
        model=model, tokenizer=tok, config=cfg, device=device,
        op=args.op, mcts_simulations=args.mcts_sims,
        max_digits=args.max_digits,
    )

    # Training loop
    history = []
    t0 = time.time()
    best_reward = -1.0

    def sigint_handler(sig, frame):
        print(f'\n  [Interrupt] Saving at step {step}...')
        Path('specialists/mcts').mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'global_step': step,
            'best_reward': best_reward,
        }, ROOT / 'specialists/mcts' / 'checkpoint.pt')
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    for step in range(1, args.steps + 1):
        result = trainer.train_step(optimizer, criterion)
        history.append(result)
        best_reward = max(best_reward, result['reward'])

        if step % args.eval_every == 0 or step == args.steps:
            elapsed = time.time() - t0
            recent = history[-args.eval_every:]
            avg_reward = sum(r['reward'] for r in recent) / len(recent)
            correct = sum(1 for r in recent if r['correct'])
            sps = step / max(1, elapsed)
            print(f'  Step {step:>6d} | loss={result["loss"]:.4f} | '
                  f'val_loss={result["value_loss"]:.4f} | '
                  f'avg_reward={avg_reward:+.3f} | '
                  f'acc={correct}/{len(recent)} ({correct/max(1,len(recent))*100:.0f}%) | '
                  f'{sps:.1f} st/s')

    # Save final
    Path('specialists/mcts').mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'global_step': args.steps,
        'best_reward': best_reward,
        'history': history[-100:],
    }, ROOT / 'specialists/mcts' / 'best.pt')

    elapsed = time.time() - t0
    final_acc = sum(1 for r in history[-100:] if r['correct']) / max(1, min(100, len(history))) * 100
    print(f'\n{"="*60}')
    print(f'  DONE! {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'  Final accuracy (last 100): {final_acc:.0f}%')
    print('  Model saved: specialists/mcts/best.pt')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
