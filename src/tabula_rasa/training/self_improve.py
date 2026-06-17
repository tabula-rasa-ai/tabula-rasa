"""Self-Improvement Daemon — autonomous self-play improvement cycle.

Runs the Socratic self-improvement loop continuously:
  Generate → Critique → Correct → Train → Save → Repeat

Each cycle:
1. Generates N fresh problems at current difficulty
2. Runs the 5-round critique loop on each
3. Trains on successful correction pairs
4. Logs accuracy improvement and saves checkpoint
5. Advances difficulty when mastery threshold is met

Usage:
    python3 egefalos/self_improve.py                              # Auto-detect latest checkpoint
    python3 egefalos/self_improve.py --checkpoint specialists/math/add/best.pt
    python3 egefalos/self_improve.py --daemon --interval 3600     # Run hourly
    python3 egefalos/self_improve.py --iterations 10 --problems 200 --train-steps 500
"""

from __future__ import annotations

import sys, time, argparse, json
from pathlib import Path

# Ensure project root and src are on path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / 'src'))
sys.path.insert(0, str(_root))

import torch
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.reasoning.socratic_trainer import SocraticSelfTrainer


def _find_latest_checkpoint() -> Path | None:
    """Find the most recently trained specialist checkpoint."""
    candidates = sorted(Path('specialists/math').glob('*/best.pt'))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    fallback = list(Path('checkpoints').glob('*.pt'))
    if fallback:
        return max(fallback, key=lambda p: p.stat().st_mtime)
    return None


def _find_tokenizer(checkpoint_path: Path) -> Path | None:
    """Find tokenizer.json alongside or nearby the checkpoint."""
    tok = checkpoint_path.parent / 'tokenizer.json'
    if tok.exists():
        return tok
    for p in Path('specialists/math').glob('*/tokenizer.json'):
        return p
    return None


def load_specialist(checkpoint_path: str | None = None) -> tuple:
    """Load a trained specialist model and tokenizer.

    Returns:
        (model, tokenizer, config, device) tuple.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find checkpoint
    ckpt_path = Path(checkpoint_path) if checkpoint_path else _find_latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        print('  [SelfImprove] No checkpoint found.')
        print('  Train one first: python3 train_specialist.py add --quick')
        return None, None, None, None

    # Find tokenizer
    tok_path = _find_tokenizer(ckpt_path)
    if tok_path is None or not tok_path.exists():
        print(f'  [SelfImprove] No tokenizer found alongside {ckpt_path}')
        return None, None, None, None

    # Load tokenizer
    tok = MathTokenizer.load(str(tok_path))

    # Load model
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = state.get('model_state_dict', state)
    model.load_state_dict(sd, strict=False)
    model.eval()

    print(f'  [SelfImprove] Loaded: {ckpt_path}')
    print(f'  [SelfImprove] Model: {count_parameters(model):,} params on {device}')
    print(f'  [SelfImprove] Tokenizer: {tok.vocab_size} tokens')

    return model, tok, cfg, device


def run_improvement_cycle(model, tok, cfg, device, args, cycle_num: int = 1) -> dict:
    """Run one self-improvement cycle and return metrics."""
    op_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
    op = op_map.get(args.op, args.op)

    print(f'\n  ── Self-Improvement Cycle #{cycle_num} ──')
    print(f'  Operation: {op} | Max digits: {args.max_digits}')
    print(f'  Problems: {args.problems} | Train steps: {args.train_steps} | LR: {args.lr}')

    trainer = SocraticSelfTrainer(
        model=model,
        tokenizer=tok,
        device=device,
        op=op,
        max_digits=args.max_digits,
        lr=args.lr,
    )

    results = trainer.run_self_improvement(
        iterations=args.iterations,
        problems_per_iter=args.problems,
        train_steps_per_iter=args.train_steps,
        batch_size=args.batch,
    )

    return {
        'cycle': cycle_num,
        'results': results,
    }


def save_improvement_log(history: list[dict], path: Path | None = None):
    """Save improvement history to JSON."""
    if path is None:
        path = Path('memory') / 'self_improve_history.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f'  [SelfImprove] Log saved to {path}')


def main():
    parser = argparse.ArgumentParser(
        description='Self-Improvement Daemon — autonomous self-play'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt checkpoint (auto-detects latest if omitted)')
    parser.add_argument('--op', type=str, default='add',
                        choices=['add', 'sub', 'mul', 'div'],
                        help='Operation to improve')
    parser.add_argument('--max-digits', type=int, default=2,
                        help='Max digits per operand')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Socratic iterations per cycle')
    parser.add_argument('--problems', type=int, default=100,
                        help='Problems per iteration')
    parser.add_argument('--train-steps', type=int, default=300,
                        help='Training steps per iteration')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size for correction training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for correction training')
    parser.add_argument('--daemon', action='store_true',
                        help='Run continuously in daemon mode')
    parser.add_argument('--interval', type=int, default=3600,
                        help='Seconds between cycles in daemon mode (default: 3600 = 1h)')
    parser.add_argument('--cycles', type=int, default=0,
                        help='Max cycles (0 = unlimited in daemon mode)')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'  SELF-IMPROVEMENT DAEMON')
    print(f'  {"="*50}')
    print(f'  Op: {args.op} | Max digits: {args.max_digits}')
    print(f'  Iterations per cycle: {args.iterations}')
    print(f'  Problems per iteration: {args.problems}')
    print(f'  Training steps: {args.train_steps} | Batch: {args.batch} | LR: {args.lr}')
    if args.daemon:
        print(f'  Mode: DAEMON (every {args.interval}s, {"unlimited" if args.cycles == 0 else args.cycles + " cycles"})')
    else:
        print(f'  Mode: SINGLE CYCLE')
    print(f'{"="*60}')

    # Load model
    model, tok, cfg, device = load_specialist(args.checkpoint)
    if model is None:
        return

    # Run improvement cycles
    history = []
    cycle = 0
    max_cycles = args.cycles if args.cycles > 0 else (1 if not args.daemon else 999999)

    while cycle < max_cycles:
        cycle += 1
        result = run_improvement_cycle(model, tok, cfg, device, args, cycle)
        history.append(result)
        save_improvement_log(history)

        if not args.daemon or cycle >= max_cycles:
            break

        print(f'\n  [SelfImprove] Sleeping {args.interval}s until next cycle...')
        time.sleep(args.interval)

    print(f'\n{"="*60}')
    print(f'  SELF-IMPROVEMENT COMPLETE')
    print(f'  Cycles: {cycle}')
    print(f'  History saved to: memory/self_improve_history.json')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
