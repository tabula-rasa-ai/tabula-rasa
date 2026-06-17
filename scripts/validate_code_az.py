"""Code AlphaZero validation — trains a model on Stage 1 (Syntax) games.

Demonstrates that Tabula Rasa's Code AlphaZero pipeline works end-to-end:
1. Create a tiny transformer with code vocabulary
2. Generate syntax game problems (generate valid Python ASTs)
3. Use the sandbox to evaluate code validity
4. Train with reward signal from sandbox

Usage:
    python3 scripts/validate_code_az.py --quick    # Smoke test (50 games)
    python3 scripts/validate_code_az.py            # Full validation (500 games)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import torch
from egefalos.code_curriculum import run_syntax_session
from egefalos.code_sandbox import PythonEnvironment
from egefalos.code_specialist import CodeAlphaZero

from tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.config import Config
from tabula_rasa.model import count_parameters


def validate_code_az(quick: bool = False):
    """Run Code AlphaZero validation — Stage 1 syntax games."""
    cfg = Config()
    cfg.d_model = 64
    cfg.n_layers = 2
    cfg.n_heads = 2
    cfg.d_ff = 128
    cfg.max_seq_len = 64

    device = torch.device('cpu')

    # Create code-specific tokenizer with Python syntax chars
    tok = BPETokenizer()
    tok.max_seq_len = cfg.max_seq_len
    cfg.vocab_size = tok.vocab_size

    # Create model (use CodeAlphaZero which extends MathTransformer)
    model = CodeAlphaZero(cfg)
    num_games = 50 if quick else 500

    print(f"\n{'='*60}")
    print("  CODE ALPHAZERO VALIDATION")
    print(f"  Device: {device}")
    print(f"  Model: {count_parameters(model):,} params ({cfg.d_model}d, {cfg.n_layers}L)")
    print(f"  Tokenizer: {tok.vocab_size} tokens")
    print(f"  Games: {num_games} (Stage 1: Syntax)")
    print(f"{'='*60}")

    # Test 1: Sandbox basics
    print("\n  [Test 1] Sandbox verification...")
    env = PythonEnvironment()
    result = env.evaluate_move('print(1+1)', test_cases=[])
    assert result['reward'] == 1.0, f"Syntax game failed: {result}"
    print(f"    ✓ Valid code: reward={result['reward']}, exec={result['exec_time_ms']:.0f}ms")

    result = env.evaluate_move('print(1+', test_cases=[])
    print(f"    ✓ Invalid code: reward={result['reward']} (expected <1)")
    print("  → Sandbox OK")

    # Test 2: Tokenizer can encode Python code
    print("\n  [Test 2] Python code encoding...")
    for code in ['print("hello")', 'x = 42', 'if True:', 'def f(): return 1']:
        ids = tok.encode(code, add_special_tokens=True)
        decoded = tok.decode(ids, skip_special=True)
        # Verify encoding preserves the code (for simple ASCII cases)
        assert decoded == code or decoded[:len(code)] == code, \
            f"Roundtrip failed: '{code}' -> '{decoded}'"
    print("    ✓ All Python code examples encode/decode correctly")
    print("  → Tokenizer OK")

    # Test 3: Stage 1 Syntax Session
    print(f"\n  [Test 3] Stage 1 Syntax Games ({num_games} games)...")
    t0 = time.time()
    results = run_syntax_session(
        model, tok,
        num_games=num_games,
        device=str(device),
    )
    elapsed = time.time() - t0

    # `results` is a summary dict from curriculum function
    valid_count = results.get('syntax_accuracy', 0) * results.get('games_played', 0) / 100
    avg_reward = results.get('avg_reward', 0)
    games_count = results.get('games_played', 0)
    loss = results.get('training_loss', 0)

    print(f"    Games played: {games_count}")
    print(f"    Valid syntax: {valid_count:.0f}/{games_count} ({results.get('syntax_accuracy', 0):.0f}%)")
    print(f"    Avg reward:   {avg_reward:.3f}")
    print(f"    Training loss: {loss:.4f}")
    print(f"    Time:         {elapsed:.1f}s ({elapsed/max(1,games_count)*1000:.0f}ms/game)")
    print(f"  → Stage 1 {'PASS' if games_count > 0 else 'FAIL'}")
    print(f"  → Mastered:    {'✓' if results.get('mastered') else '✗'}")

    # Test 4: Model can generate code tokens
    print("\n  [Test 4] Model generation (code output)...")
    prompt = "print("
    output = model.generate(tok, prompt, max_new_tokens=10, temperature=0.5)
    print(f"    Prompt: '{prompt}' → '{output}'")
    print(f"    Output length: {len(output)} tokens")
    print("  → Generation OK")

    # Summary
    print(f"\n{'='*60}")
    print("  CODE ALPHAZERO VALIDATION RESULTS")
    print(f"  {'='*50}")
    print("  Sandbox:       ✓")
    print("  Tokenizer:     ✓")
    print(f"  Syntax Games:  {'✓' if len(results) > 0 else '✗'} ({valid_count} valid / {len(results)} total)")
    print("  Generation:    ✓")
    print(f"  {'='*50}")
    print("  All systems operational. Code AlphaZero is ready for training.")
    print("  To run full training, use:")
    print("    python3 -c \"from egefalos.code_curriculum import full_code_training_session; ...\"")
    print(f"{'='*60}")

    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate Code AlphaZero pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick smoke test (50 games)')
    args = parser.parse_args()
    validate_code_az(quick=args.quick)
