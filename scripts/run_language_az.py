"""
run_language_az.py — Launch Language AlphaZero self-play from dashboard or CLI.

Usage:
    python3 run_language_az.py --games 200 --difficulty medium --mcts --grammar --value-head

This script is called by the Specialist Trainer dashboard's "Start Self-Play" button.
Avoids inline escaping issues by being a proper script file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Language AlphaZero self-play")
    parser.add_argument("--games", type=int, default=200, help="Number of self-play games")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Concept difficulty level",
    )
    parser.add_argument("--mcts", action="store_true", help="Enable Micro-MCTS")
    parser.add_argument("--grammar", action="store_true", help="Enable grammar rules")
    parser.add_argument("--value-head", action="store_true", help="Enable value head training")
    parser.add_argument(
        "--mcts-simulations", type=int, default=16, help="MCTS simulations per search"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    print(f"[LanguageAZ] Starting self-play: {args.games} games, {args.difficulty}", flush=True)
    print(
        f"[LanguageAZ] MCTS={args.mcts} Grammar={args.grammar} ValueHead={args.value_head}",
        flush=True,
    )

    from egefalos.grammar_engine import create_grammar_rules
    from egefalos.language_az import LanguageAlphaZero
    from egefalos.mcts import create_mcts_fn
    from egefalos.semantic_game import alphazero_gymnasium_session
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer

    cfg = Config()
    cfg.use_value_head = args.value_head
    cfg.language_max_seq_len = 128

    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = LanguageAlphaZero(cfg)
    print(
        f"[LanguageAZ] Model created: {sum(p.numel() for p in model.parameters()):,} params",
        flush=True,
    )

    grammar_rules = create_grammar_rules() if args.grammar else None
    mcts_fn = create_mcts_fn(model, tok, simulations=args.mcts_simulations) if args.mcts else None

    print(f"[LanguageAZ] Running gymnasium session...", flush=True)

    results = alphazero_gymnasium_session(
        model,
        tok,
        num_games=args.games,
        difficulty=args.difficulty,
        mcts_fn=mcts_fn,
        grammar_rules=grammar_rules,
        learning_rate=args.lr,
        device="cpu",
    )

    print(f"[LanguageAZ] Self-play complete!", flush=True)
    print(f'[LanguageAZ] Games: {results["games_played"]}', flush=True)
    print(f'[LanguageAZ] Avg reward: {results["avg_reward"]:.3f}', flush=True)
    print(f'[LanguageAZ] Perfect rate: {results["perfect_rate"]:.1f}%', flush=True)
    print(f'[LanguageAZ] Training loss: {results["training_loss"]:.4f}', flush=True)

    # Save the model checkpoint
    from pathlib import Path

    import torch

    save_path = Path("specialists/language/general")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gym_results": results,
        },
        save_path / "best.pt",
    )
    print(f'[LanguageAZ] Model saved to {save_path / "best.pt"}', flush=True)

    # Output JSON for dashboard consumption
    print(f"[LanguageAZ_RESULT] {json.dumps(results)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
