"""Tabula Rasa CLI — train, serve, bench, auto-train.

Usage:
    tabula-rasa serve [--port PORT] [--host HOST]
    tabula-rasa train <op> [--quick] [--resume] [--steps N] [--batch N] [--lr LR]
    tabula-rasa bench <checkpoint> [max_digits]
    tabula-rasa auto-train [--target PCT] [--ops OP [OP ...]]
"""

import os
import sys

# Ensure src/ and project root are on path
_src_dir = os.path.dirname(os.path.abspath(__file__))  # src/tabula_rasa/
_src = os.path.dirname(_src_dir)  # src/
_project_root = os.path.dirname(_src)  # project root
for p in [_src, _project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == 'serve':
        _serve(args)
    elif cmd == 'train':
        _train(args)
    elif cmd == 'bench':
        _bench(args)
    elif cmd == 'auto-train':
        _auto_train(args)
    elif cmd in ('-h', '--help'):
        print(__doc__)
    else:
        print(f'Unknown command: {cmd}')
        print(__doc__)
        sys.exit(1)


def _serve(args):
    """Start the Tabula Rasa AI server (port 8002)."""
    # Import here so the package can be installed without all deps at import time
    from tabula_rasa.server.ai_server import main as server_main

    # Build a fresh argv for the server's own parser
    sys.argv = ['ai_server.py'] + args
    server_main()


def _train(args):
    """Train a math specialist. Delegates to scripts/train_specialist.py."""
    from scripts.train_specialist import train_specialist

    op = args[0] if args else 'add'
    kwargs = {'quick': False, 'resume': False, 'steps': 0, 'batch_size': 0, 'lr': 0}
    i = 1
    while i < len(args):
        if args[i] == '--quick':
            kwargs['quick'] = True
        elif args[i] == '--resume':
            kwargs['resume'] = True
        elif args[i] == '--steps':
            i += 1
            kwargs['steps'] = int(args[i])
        elif args[i] == '--batch':
            i += 1
            kwargs['batch_size'] = int(args[i])
        elif args[i] == '--lr':
            i += 1
            kwargs['lr'] = float(args[i])
        i += 1

    train_specialist(op, **kwargs)


def _bench(args):
    """Run full benchmark on a checkpoint."""
    from tabula_rasa.config import Config
    from tabula_rasa.eval import full_benchmark
    from tabula_rasa.eval import load_model as eval_load_model
    from tabula_rasa.tokenizer import MathTokenizer

    if not args:
        print('Usage: tabula-rasa bench <checkpoint.pt> [max_digits]')
        sys.exit(1)

    ckpt_path = args[0]
    max_digits = int(args[1]) if len(args) > 1 else 4

    cfg = Config()
    tok = MathTokenizer()
    model, _ = eval_load_model(ckpt_path, cfg, tok)

    results = full_benchmark(model, tok, train_max_digits=max_digits)

    for name, result in results.items():
        if hasattr(result, 'accuracy'):
            print(f'{name}: {result.accuracy:.1f}%')
        elif hasattr(result, 'position_accuracy'):
            print(f'{name}: {result.position_accuracy}')


def _auto_train(args):
    """Auto-train weakest specialists."""
    import subprocess
    cmd = [sys.executable, 'scripts/auto_train.py'] + args
    subprocess.run(cmd, cwd=_project_root)


if __name__ == '__main__':
    main()
