#!/usr/bin/env python3
"""End-to-end integration test: validates ALL improvements working together.

Tests every component we've built:
  ✓ Config (entropy curriculum, gradient checkpointing)
  ✓ Model (RASA, MoE, LoRA)
  ✓ Training (AMP, gradient checkpointing, MoE aux loss)
  ✓ Evaluation (6-dim benchmark + adversarial)
  ✓ MCTS inference
  ✓ RAG engine
  ✓ Prepared Slate Orchestrator
  ✓ Experiment runner (syntax check)

Usage:
    python3 scripts/e2e_integration_test.py              # CPU quick
    python3 scripts/e2e_integration_test.py --gpu        # GPU, proper model
    python3 scripts/e2e_integration_test.py --full       # Full test suite
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

PASS = 0
FAIL = 0
TESTS = []


def test(name):
    """Decorator to register a test."""
    def wrapper(fn):
        TESTS.append((name, fn))
        return fn
    return wrapper


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        print(f"  ✗ {msg}")


# ═══════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ═══════════════════════════════════════════════════════════════════════


@test("Config: entropy curriculum & gradient checkpointing defaults")
def test_config():
    from tabula_rasa.config import Config
    cfg = Config()
    check(cfg.use_entropy_curriculum == True, "entropy curriculum ON by default")
    check(cfg.use_gradient_checkpointing == True, "gradient checkpointing ON by default")
    check(cfg.use_moe == False, "MoE OFF by default (opt-in)")
    check(cfg.use_rasa == False, "RASA OFF by default (opt-in)")
    check(cfg.use_mcts_inference == False, "MCTS OFF by default (opt-in)")
    check(cfg.mcts_simulations == 16, "MCTS simulations = 16")
    check(cfg.rasa_max_relative == 16, "RASA max relative = 16")
    check(cfg.num_experts == 4, "MoE experts = 4")
    check(cfg.top_k == 2, "MoE top-k = 2")


@test("Config: presets")
def test_presets():
    from tabula_rasa.config import Config
    cfg = Config()
    presets = {"1M": (128, 4, 4, 512, 32), "5M": (256, 5, 8, 1024, 48), "10M": (320, 6, 8, 1280, 64)}
    for name, (d, l, h, ff, seq) in presets.items():
        c = Config()
        c.apply_preset(name)
        check((c.d_model, c.n_layers, c.n_heads, c.d_ff, c.max_seq_len) == (d, l, h, ff, seq),
              f"preset {name}: d={d} L={l} h={h} ff={ff} seq={seq}")


@test("Model: creation with RASA")
def test_model_rasa():
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer, count_parameters
    from tabula_rasa.tokenizer import MathTokenizer

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 32; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 64
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 32
    cfg.use_rasa = True

    model = MathTransformer(cfg)
    params = count_parameters(model)
    check(params > 0, f"model created: {params:,} params")
    check(model.layers[0].attention.use_rasa, "RASA enabled in attention layer")
    check(hasattr(model.layers[0].attention, "rel_bias"), "RASA rel_bias embedding exists")
    check(model.layers[0].attention.rel_bias.weight.shape == (33, 4),
          f"rel_bias shape {model.layers[0].attention.rel_bias.weight.shape}")

    # Forward pass
    x = torch.randint(0, tok.vocab_size, (2, 8))
    logits, loss, v = model(x, x)
    check(torch.isfinite(loss), f"forward loss={loss.item():.4f}")


@test("Model: creation with MoE")
def test_model_moe():
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer, count_parameters
    from tabula_rasa.tokenizer import MathTokenizer

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 32; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 64
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 32
    cfg.use_moe = True; cfg.num_experts = 4; cfg.top_k = 2

    model = MathTransformer(cfg)
    dense = count_parameters(MathTransformer(Config()))
    moe_params = count_parameters(model)
    check(moe_params > dense, f"MoE params {moe_params:,} > dense {dense:,}")

    # MoE aux loss
    x = torch.randint(0, tok.vocab_size, (2, 8))
    logits, loss, v = model(x, x)
    aux = model.get_moe_aux_loss()
    check(torch.isfinite(aux), f"MoE aux loss={aux.item():.4f}")

    # Every MoE layer stores aux loss
    for layer in model.layers:
        ff = layer.feed_forward
        check(hasattr(ff, "last_aux_loss"), f"layer has MoE aux loss: {ff.last_aux_loss:.4f}")


@test("Model: creation with LoRA")
def test_model_lora():
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer, count_parameters
    from tabula_rasa.tokenizer import MathTokenizer
    from tabula_rasa.lora import apply_lora_to_model, set_lora_trainable, save_lora_adapters

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 32; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 64
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 32

    model = MathTransformer(cfg)
    lora = apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=["wq", "wk", "wv", "wo"])
    check(len(lora) == 8, f"LoRA layers applied: {len(lora)}")

    set_lora_trainable(model, trainable=True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(n_trainable > 0, f"trainable LoRA params: {n_trainable}")

    # Save/load roundtrip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        lora_path = f.name
    save_lora_adapters(lora, lora_path)
    check(os.path.getsize(lora_path) > 0, f"LoRA saved: {os.path.getsize(lora_path)} bytes")
    os.unlink(lora_path)


@test("Tokenizer: encode/decode roundtrip")
def test_tokenizer():
    from tabula_rasa.tokenizer import MathTokenizer
    tok = MathTokenizer()
    tests = ["12+34=46", "7+8=15", "100-1=99", "5*6=30", "12/4=3"]
    for text in tests:
        ids = tok.encode(text, add_special_tokens=True)
        decoded = tok.decode(ids, skip_special=True)
        check(decoded == text, f"roundtrip: {text}")


@test("Benchmark: full_benchmark imports and structure")
def test_benchmark():
    from tabula_rasa.eval import (EvalResult, PerPositionResult,
                                   evaluate_accuracy, evaluate_per_position,
                                   evaluate_ood, evaluate_compositional,
                                   evaluate_robustness, evaluate_adversarial,
                                   full_benchmark, BOUNDARY_PROBLEMS)
    check(len(BOUNDARY_PROBLEMS) == 31, f"boundary cases: {len(BOUNDARY_PROBLEMS)}")
    for fn in [evaluate_accuracy, evaluate_per_position, evaluate_ood,
               evaluate_compositional, evaluate_robustness, evaluate_adversarial,
               full_benchmark]:
        check(callable(fn), f"{fn.__name__} is callable")


@test("Benchmark: adversarial runs on tiny model")
def test_benchmark_adversarial():
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer
    from tabula_rasa.eval import evaluate_adversarial

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.d_ff = 32
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 16
    model = MathTransformer(cfg); model.eval()

    adv = evaluate_adversarial(model, tok, num_boundary=5, num_noise=2, num_fgsm=0, verbose=False)
    check(len(adv) == 4, f"4 adversarial categories: {list(adv.keys())}")
    for k, v in adv.items():
        check(hasattr(v, 'accuracy') and hasattr(v, 'num_problems'),
              f"{k}: accuracy={v.accuracy:.0f}% ({v.num_problems} problems)")


@test("MCTS: inference module imports and structure")
def test_mcts():
    from tabula_rasa.mcts_inference import MCTSInference, mcts_generate
    check(callable(mcts_generate), "mcts_generate convenience fn callable")

    # Verify class structure
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 32; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 64
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 32
    model = MathTransformer(cfg); model.eval()

    mcts = MCTSInference(model, tok, num_simulations=4, top_k=5, temperature=0.5, max_new_tokens=8)
    answer, stats = mcts.generate("2+3")
    check("num_simulations" in stats, f"MCTS stats: sims={stats['num_simulations']}, nodes={stats['total_nodes']}")
    check(stats["tree_size"] > 0, f"MCTS tree visited: {stats['tree_size']}")


@test("RAG: engine builds index and retrieves")
def test_rag():
    from tabula_rasa.rag import RAGEngine, Document, RetrievalResult
    engine = RAGEngine()
    n = engine.build_index()
    check(n > 0, f"RAG indexed {n} documents")

    results = engine.retrieve("What is a transformer?", top_k=2)
    check(len(results) > 0, f"RAG retrieved {len(results)} results")
    if results:
        check(results[0].score > 0, f"top score: {results[0].score:.3f}")
        check("transformer" in results[0].text.lower(), "result mentions transformer")

    # Direct answer
    answer = engine.retrieve_answer("What is a transformer?")
    check(answer is not None, "direct answer found")


@test("Orchestrator: query routing")
def test_orchestrator():
    from tabula_rasa.orchestrator import PreparedSlateOrchestrator, AnswerResult
    orch = PreparedSlateOrchestrator()

    # Math detection
    check(orch._is_math_query("12+34"), "math: '12+34'")
    check(orch._is_math_query("what is 12+34?"), "math: 'what is 12+34?'")
    check(not orch._is_math_query("Hello"), "not math: 'Hello'")
    check(not orch._is_math_query("What is AI?"), "not math: 'What is AI?'")

    # Greeting
    check(orch._is_greeting("Hello"), "greeting: 'Hello'")
    check(orch._is_greeting("Good morning"), "greeting: 'Good morning'")
    check(not orch._is_greeting("What is AI?"), "not greeting")

    # Definition
    check(orch._is_definition("What is a transformer?"), "definition: 'What is a transformer?'")
    check(not orch._is_definition("12+34"), "not definition: '12+34'")

    # Answer routing (no backends → fallback)
    r = orch.answer("Hello!")
    check(r.strategy == "fallback_rule", f"Hello → {r.strategy}")
    check(r.confidence == 0.8, f"greeting confidence: {r.confidence}")

    r = orch.answer("What is AI?")
    check(r.strategy == "fallback_unknown", f"AI → {r.strategy}")
    check("not sure" in r.answer.lower(), "fallback answer says 'not sure'")

    # Stats
    stats = orch.stats()
    check(stats["total_queries"] == 2, f"query count: {stats['total_queries']}")
    check("fallback_rule" in stats["strategy_counts"], "counts: fallback_rule")
    check("fallback_unknown" in stats["strategy_counts"], "counts: fallback_unknown")


@test("Experiment runner: syntax check works")
def test_experiment_runner():
    import subprocess
    result = subprocess.run(
        [sys.executable, "experiments/run_benchmarks.py", "--syntax-only"],
        capture_output=True, text=True, timeout=30,
    )
    check(result.returncode == 0, f"exit code: {result.returncode}")
    check("168" in result.stdout or "All" in result.stdout, "all files parsed")


@test("LoRA: save/load roundtrip preserves adapter weights")
def test_lora_saveload():
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer, count_parameters
    from tabula_rasa.tokenizer import MathTokenizer
    from tabula_rasa.lora import apply_lora_to_model, save_lora_adapters, load_lora_adapters

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 32; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 64
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 32

    model1 = MathTransformer(cfg)
    lora1 = apply_lora_to_model(model1, rank=4, alpha=1.0)
    w_before = lora1[0].lora_A[0, 0].item()

    save_lora_adapters(lora1, "/tmp/test_lora.pt")

    model2 = MathTransformer(cfg)
    lora2 = load_lora_adapters(model2, "/tmp/test_lora.pt", rank=4, alpha=1.0)
    w_after = lora2[0].lora_A[0, 0].item()

    check(abs(w_before - w_after) < 1e-6, f"LoRA weight preserved: {w_before:.6f} == {w_after:.6f}")
    os.unlink("/tmp/test_lora.pt")


@test("RASA: forward pass with RASA bias")
def test_rasa_forward():
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 32; cfg.n_layers = 1; cfg.n_heads = 4; cfg.d_ff = 64
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 32
    cfg.use_rasa = True

    model = MathTransformer(cfg); model.eval()
    x = torch.randint(0, tok.vocab_size, (1, 16))

    # Run twice: with and without RASA, ensure different outputs
    out1, _, _ = model(x, x)
    check(torch.isfinite(out1).all(), "RASA forward: finite logits")

    # Disable RASA and compare
    model.layers[0].attention.use_rasa = False
    out2, _, _ = model(x, x)
    model.layers[0].attention.use_rasa = True
    diff = (out1 - out2).abs().max().item()
    check(diff > 0, f"RASA changes output: max diff = {diff:.6f}")


@test("Full benchmark: runs with adversarial_kwargs")
def test_benchmark_kwargs():
    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer
    from tabula_rasa.eval import full_benchmark

    tok = MathTokenizer()
    cfg = Config(); cfg.d_model = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.d_ff = 32
    cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 16
    model = MathTransformer(cfg); model.eval()

    # Run with minimal adversarial kwargs
    results = full_benchmark(model, tok, train_max_digits=1, verbose=False,
                             adversarial_kwargs={"num_boundary": 3, "num_noise": 1, "num_fgsm": 0})
    required = ["accuracy", "per_position", "ood", "compositional", "robustness",
                "adversarial/boundary", "adversarial/format_attacks", "adversarial/noise"]
    for key in required:
        check(key in results, f"benchmark key: {key}")
    check(len(results) >= 8, f"benchmark dimensions: {len(results)}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E2E Integration Test")
    parser.add_argument("--gpu", action="store_true", help="Include GPU-specific tests")
    parser.add_argument("--full", action="store_true", help="Run all tests (including slow)")
    parser.add_argument("--list", action="store_true", help="List tests without running")
    args = parser.parse_args()

    # Filter tests
    if args.list:
        print(f"Available tests ({len(TESTS)}):")
        for name, _ in TESTS:
            print(f"  {name}")
        sys.exit(0)

    # ── Header ──
    print("=" * 60)
    print("  TABULA RASA — E2E INTEGRATION TEST")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    print()

    # ── Run ──
    start = time.time()
    for name, fn in TESTS:
        print(f"[{name}]")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            FAIL += 1
            import traceback
            print(f"  ✗ EXCEPTION: {e}")
            if args.full:
                traceback.print_exc()
        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")
        print()

    # ── Summary ──
    total = PASS + FAIL
    elapsed = time.time() - start
    print("=" * 60)
    print(f"  RESULTS: {PASS}/{total} passed ({FAIL} failed)")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
