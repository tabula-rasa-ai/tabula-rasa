"""
run_code_az.py — Launch Code AlphaZero self-play from dashboard or CLI.

Usage:
    python3 run_code_az.py --syntax 100 --fuzzing 50 --algorithm 30

Called by the Specialist Trainer dashboard's "Start Code Self-Play" button.
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description='Code AlphaZero self-play')
    parser.add_argument('--syntax', type=int, default=100, help='Stage 1: Syntax games')
    parser.add_argument('--fuzzing', type=int, default=50, help='Stage 2: Fuzzing games')
    parser.add_argument('--algorithm', type=int, default=30, help='Stage 3: Algorithm games')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--stage', type=str, default=None, choices=['syntax', 'fuzzing', 'algorithm'],
                        help='Run only a specific stage')
    args = parser.parse_args()

    print(f'[CodeAZ] Starting Code AlphaZero self-play', flush=True)
    print(f'[CodeAZ] Syntax={args.syntax} Fuzzing={args.fuzzing} Algorithm={args.algorithm}', flush=True)

    from egefalos.code_specialist import CodeAlphaZero
    from egefalos.code_curriculum import (
        run_syntax_session, run_fuzzing_session, run_algorithm_session,
        full_code_training_session,
    )
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer

    cfg = Config()
    cfg.use_value_head = True
    cfg.max_seq_len = 256

    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = CodeAlphaZero(cfg)
    print(f'[CodeAZ] Model created: {sum(p.numel() for p in model.parameters()):,} params', flush=True)

    if args.stage:
        # Run a single stage
        if args.stage == 'syntax':
            results = run_syntax_session(model, tok, args.syntax, args.lr, 'cpu')
        elif args.stage == 'fuzzing':
            results = run_fuzzing_session(model, tok, args.fuzzing, args.lr, 'cpu')
        elif args.stage == 'algorithm':
            results = run_algorithm_session(model, tok, args.algorithm, args.lr, 'cpu')
    else:
        # Run through all stages
        results = full_code_training_session(
            model, tok,
            num_syntax=args.syntax,
            num_fuzzing=args.fuzzing,
            num_algorithm=args.algorithm,
            learning_rate=args.lr,
            device='cpu',
        )

    print(f'[CodeAZ] Self-play complete!', flush=True)

    # Flatten results for JSON output
    if isinstance(results, dict):
        for stage_key, stage_results in results.items() if 'stage' not in results else [(None, results)]:
            if isinstance(stage_results, dict):
                for k, v in stage_results.items():
                    if k != 'stage':
                        print(f'[CodeAZ_RESULT] {stage_key}.{k}: {v}', flush=True)

    # Save checkpoint
    import torch
    from pathlib import Path
    save_path = Path('specialists/code/general')
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
    }, save_path / 'best.pt')
    print(f'[CodeAZ] Model saved to {save_path / "best.pt"}', flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
