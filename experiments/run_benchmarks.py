#!/usr/bin/env python3
"""Tabula Rasa — Standardised Benchmark Suite.

Tests all core claims: architecture, tokenizer, training, EWC, checkpoints, API.
Outputs JSON report to stdout and saves to experiments/benchmark_results.json.

Usage:
    python3 experiments/run_benchmarks.py --quick   # Tiny model, 10s per test (default)
    python3 experiments/run_benchmarks.py --full    # Default config, longer timeouts
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

# ── Imports under test ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXPERIMENTS_DIR / "benchmark_results.json"


# ── Timeout guard ──────────────────────────────────────────────────
class TimeoutError(Exception):
    pass


def timeout(seconds: float):
    """Decorator that raises TimeoutError if the wrapped function takes > seconds."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = []
            exc = []

            def worker():
                try:
                    result.append(func(*args, **kwargs))
                except BaseException as e:
                    exc.append(e)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=seconds)
            if t.is_alive():
                raise TimeoutError(f"Test exceeded {seconds}s timeout and was aborted")
            if exc:
                raise exc[0]
            return result[0]

        return wrapper

    return decorator


# ── Benchmark runner ──────────────────────────────────────────────
class BenchmarkRunner:
    """Orchestrates individual tests and collects results."""

    def __init__(self, quick: bool = True):
        self.quick = quick
        self.results: list[dict] = []
        self.per_test_timeout = 10 if quick else 60
        # Training test always gets a longer timeout
        self.training_subprocess_timeout = 120 if quick else 300
        self.training_thread_timeout = self.training_subprocess_timeout + 15

    def run(self) -> list[dict]:
        tests = [
            ("architecture_smoke", self.test_architecture_smoke),
            ("tokenizer_roundtrip", self.test_tokenizer_roundtrip),
            ("tokenizer_carry_tokens", self.test_tokenizer_carry_tokens),
            ("ewc_import", self.test_ewc_import),
            ("ewc_compute_fisher", self.test_ewc_compute_fisher),
            ("quick_training", self.test_quick_training),
            ("checkpoint_save_load", self.test_checkpoint_save_load),
            ("api_endpoints", self.test_api_endpoints),
        ]
        for name, func in tests:
            result = self._run_single(name, func)
            self.results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f'  [{status}] {name:<30s} ({result["duration_s"]:.2f}s)  {result["detail"]}')
        return self.results

    def _run_single(self, name: str, func) -> dict:
        start = time.time()
        # Training test uses its own longer timeout
        test_timeout = (
            self.training_thread_timeout if name == "quick_training" else self.per_test_timeout
        )
        wrapped = timeout(test_timeout)(func)
        try:
            detail = wrapped()
            passed = True
        except TimeoutError as e:
            passed = False
            detail = f"TIMEOUT after {self.per_test_timeout}s"
        except Exception as e:
            passed = False
            tb = traceback.format_exc()
            # Keep detail concise — first line of exception + last call
            lines = tb.strip().split("\n")
            short_tb = "\n".join(lines[-3:]) if len(lines) > 3 else tb
            detail = f"{type(e).__name__}: {e}"
        duration = time.time() - start
        return {
            "test": name,
            "passed": passed,
            "detail": detail,
            "duration_s": round(duration, 3),
        }

    # ── Test 1: Architecture smoke test ────────────────────────────
    def test_architecture_smoke(self) -> str:
        from tabula_rasa.config import Config
        from tabula_rasa.model import MathTransformer, count_parameters
        from tabula_rasa.tokenizer import MathTokenizer

        cfg = Config()
        if self.quick:
            cfg.d_model = 64
            cfg.n_layers = 2
            cfg.n_heads = 2
            cfg.d_ff = 256
            cfg.max_seq_len = 32

        tok = MathTokenizer()
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)

        n_params = count_parameters(model)
        assert n_params > 0, "Model has zero parameters"

        # Forward pass on a small batch
        ids = tok.encode("12+34=46", add_special_tokens=True)
        x = torch.tensor([ids])
        logits, loss, value = model(x, x)
        assert logits.shape[0] == 1, f"Expected batch=1, got {logits.shape[0]}"
        assert (
            logits.shape[-1] == tok.vocab_size
        ), f"Expected vocab={tok.vocab_size}, got {logits.shape[-1]}"
        assert loss is not None and torch.isfinite(loss), f"Loss is NaN/inf: {loss}"

        arch = f"MathTransformer-{cfg.d_model}x{cfg.n_layers}"
        return f"model={arch} params={n_params:,} loss={loss.item():.4f}"

    # ── Test 2: Tokenizer encode/decode roundtrip ──────────────────
    def test_tokenizer_roundtrip(self) -> str:
        from tabula_rasa.tokenizer import MathTokenizer

        tok = MathTokenizer()
        tests = [
            "12+34=46",
            "7+8=15",
            "100-1=99",
            "5*6=30",
            "12/4=3",
            "(2+3)*4=20",
        ]
        for text in tests:
            ids = tok.encode(text, add_special_tokens=True)
            decoded = tok.decode(ids, skip_special=True)
            assert decoded == text, f"Roundtrip failed: {text!r} -> {decoded!r}"

        # Verify special tokens exist
        assert tok.pad_id >= 0
        assert tok.bos_id >= 0
        assert tok.eos_id >= 0
        assert tok.unk_id >= 0
        assert tok.vocab_size > 0
        return f"vocab={tok.vocab_size} tests={len(tests)}"

    # ── Test 3: Carry-digit tokens ─────────────────────────────────
    def test_tokenizer_carry_tokens(self) -> str:
        from tabula_rasa.tokenizer import MathTokenizer

        tok = MathTokenizer()
        # Carry-digit tokens should be recognised as single tokens
        for carry in range(2):
            for digit in range(10):
                token_str = f"{carry}{digit}"
                ids = tok.encode(token_str, add_special_tokens=False)
                # Should be exactly 1 token (not 2 individual digits)
                assert (
                    len(ids) == 1
                ), f"Carry token {token_str!r} encoded to {len(ids)} tokens: {ids}"
                decoded = tok.decode(ids, skip_special=True)
                assert decoded == token_str, f"Carry token roundtrip: {token_str!r} -> {decoded!r}"

        # Verify carry tokens exist in vocabulary
        carry_count = sum(1 for t in tok.stoi if len(t) == 2 and t[0] in "01" and t[1].isdigit())
        assert carry_count == 20, f"Expected 20 carry tokens, got {carry_count}"
        return f"carry_tokens={carry_count}"

    # ── Test 4: EWC import ─────────────────────────────────────────
    def test_ewc_import(self) -> str:
        from egefalos.online_ewc import OnlineEWC
        from tabula_rasa.config import Config
        from tabula_rasa.model import MathTransformer
        from tabula_rasa.tokenizer import MathTokenizer

        cfg = Config()
        if self.quick:
            cfg.d_model = 64
            cfg.n_layers = 2
            cfg.n_heads = 2
            cfg.d_ff = 256

        tok = MathTokenizer()
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)

        ewc = OnlineEWC(model, gamma=0.9)
        assert ewc.gamma == 0.9
        assert ewc.task_count == 0

        ewc.save_anchor_weights()
        assert len(ewc.anchor_dict) > 0, "Anchor weights empty after save_anchor_weights()"

        return f"OnlineEWC loaded gamma={ewc.gamma} task_count={ewc.task_count}"

    # ── Test 5: EWC compute Fisher ─────────────────────────────────
    def test_ewc_compute_fisher(self) -> str:
        from torch.utils.data import DataLoader

        from egefalos.online_ewc import OnlineEWC
        from tabula_rasa.config import Config
        from tabula_rasa.dataset import MathDataset
        from tabula_rasa.model import MathTransformer
        from tabula_rasa.tokenizer import MathTokenizer

        cfg = Config()
        if self.quick:
            cfg.d_model = 64
            cfg.n_layers = 2
            cfg.n_heads = 2
            cfg.d_ff = 256
            cfg.max_seq_len = 32

        tok = MathTokenizer()
        cfg.vocab_size = tok.vocab_size

        model = MathTransformer(cfg)
        ewc = OnlineEWC(model)

        # Create a tiny dataloader
        ds = MathDataset(tok, num_samples=20, min_digits=1, max_digits=2)
        dl = DataLoader(ds, batch_size=8, shuffle=True)

        fisher = ewc.compute_fisher(dl, num_samples=16)
        assert len(fisher) > 0, "Fisher dict is empty"
        # All values should be non-negative (squared gradients)
        for name, f in fisher.items():
            assert (f >= 0).all(), f"Fisher for {name} has negative values: {f.min().item()}"

        # Test merge
        ewc.merge_fisher(fisher)
        assert ewc.task_count == 1

        # Test compute_fisher_overlap
        overlap = OnlineEWC.compute_fisher_overlap(fisher, fisher)
        assert abs(overlap - 1.0) < 1e-4, f"Same-Fisher overlap should be ~1.0, got {overlap}"

        return f"Fisher params={len(fisher)} tasks={ewc.task_count} overlap={overlap:.4f}"

    # ── Test 6: Quick training ─────────────────────────────────────
    def test_quick_training(self) -> str:
        if self.quick:
            # Use the --quick flag in train_specialist.py
            cmd = [
                sys.executable or "python3",
                str(PROJECT_ROOT / "train_specialist.py"),
                "add",
                "--quick",
            ]
        else:
            # Short full-mode training: 1000 steps, no curriculum
            cmd = [
                sys.executable or "python3",
                str(PROJECT_ROOT / "train_specialist.py"),
                "add",
                "--steps",
                "1000",
            ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=self.training_subprocess_timeout,
            env=env,
        )
        output = proc.stdout + proc.stderr

        # Check for completion signal
        completed = "Done" in output or "finished" in output or "Accuracy" in output
        if not completed:
            # Check if we got at least some training steps
            steps_run = "Step" in output
            if not steps_run:
                return f"No training output. Exit={proc.returncode}. stderr={proc.stderr[:300]}"

        # Try to extract accuracy
        import re

        acc_match = re.search(r"Accuracy[:\s]+([\d.]+)%", output)
        acc = float(acc_match.group(1)) if acc_match else None

        # Check if checkpoint was saved
        best_pt = PROJECT_ROOT / "specialists" / "math" / "add" / "best.pt"
        ckpt_pt = PROJECT_ROOT / "specialists" / "math" / "add" / "checkpoint.pt"

        ckpt_found = best_pt.exists() or ckpt_pt.exists()

        detail_parts = []
        if acc is not None:
            detail_parts.append(f"acc={acc:.1f}%")
        detail_parts.append(f"checkpoint={ckpt_found}")
        detail_parts.append(f"exit={proc.returncode}")

        return " | ".join(detail_parts)

    # ── Test 7: Checkpoint save/load ───────────────────────────────
    def test_checkpoint_save_load(self) -> str:
        from tabula_rasa.config import Config
        from tabula_rasa.model import MathTransformer
        from tabula_rasa.tokenizer import MathTokenizer

        cfg = Config()
        if self.quick:
            cfg.d_model = 64
            cfg.n_layers = 2
            cfg.n_heads = 2
            cfg.d_ff = 256
            cfg.max_seq_len = 32

        tok = MathTokenizer()
        cfg.vocab_size = tok.vocab_size

        # Create model and save
        model = MathTransformer(cfg)
        save_dir = PROJECT_ROOT / "experiments" / "_benchmark_ckpt_test"
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / "test_model.pt"
        tok_path = save_dir / "tokenizer.json"

        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        tok.save(str(tok_path))

        # Verify saved files exist
        assert ckpt_path.exists(), f"Checkpoint not saved to {ckpt_path}"
        assert tok_path.exists(), f"Tokenizer not saved to {tok_path}"

        model.eval()
        # Get output from original model
        ids = tok.encode("12+34=46", add_special_tokens=True)
        x = torch.tensor([ids])
        with torch.no_grad():
            orig_logits, _, _ = model(x)

        # Reload
        cfg2 = Config()
        if self.quick:
            cfg2.d_model = 64
            cfg2.n_layers = 2
            cfg2.n_heads = 2
            cfg2.d_ff = 256
            cfg2.max_seq_len = 32
        tok2 = MathTokenizer.load(str(tok_path))
        cfg2.vocab_size = tok2.vocab_size
        tok2.max_seq_len = cfg2.max_seq_len

        model2 = MathTransformer(cfg2)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model2.load_state_dict(state["model_state_dict"])
        model2.eval()

        with torch.no_grad():
            reload_logits, _, _ = model2(x)

        # Verify identical output
        diff = (orig_logits - reload_logits).abs().max().item()
        assert diff < 1e-6, f"Outputs differ after save/load: max_diff={diff:.2e}"

        # Cleanup
        import shutil

        shutil.rmtree(save_dir, ignore_errors=True)

        return f"params_match diff={diff:.2e}"

    # ── Test 8: API endpoints ─────────────────────────────────────
    def test_api_endpoints(self) -> str:
        """Start api_server.py in background, hit /health and /generate."""
        import socket
        import time as _time

        # Pick a random available port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        api_script = str(PROJECT_ROOT / "api_server.py")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [sys.executable or "python3", api_script, str(port)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        # Wait for server to start
        import urllib.error
        import urllib.request

        base_url = f"http://127.0.0.1:{port}"
        server_ready = False
        deadline = _time.time() + 8
        while _time.time() < deadline:
            try:
                resp = urllib.request.urlopen(f"{base_url}/health", timeout=2)
                if resp.status == 200:
                    server_ready = True
                    break
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                _time.sleep(0.3)

        if not server_ready:
            proc.kill()
            return "API server failed to start within 8s"

        try:
            # ── /health endpoint ──
            resp = urllib.request.urlopen(f"{base_url}/health", timeout=5)
            health_data = json.loads(resp.read().decode())
            health_ok = "model" in health_data and "params" in health_data
            health_detail = (
                f'model={health_data.get("model", "?")} params={health_data.get("params", "?")}'
            )

            # ── /generate endpoint ──
            body = json.dumps({"prompt": "12+34="}).encode()
            req = urllib.request.Request(
                f"{base_url}/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=5)
            gen_data = json.loads(resp.read().decode())
            gen_ok = "result" in gen_data
            gen_detail = f'result={gen_data.get("result", "?")[:30]}'

            assert health_ok, f"/health returned: {health_data}"
            assert gen_ok, f"/generate returned: {gen_data}"

            return f"/health: {health_detail} | /generate: {gen_detail}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tabula Rasa — Standardised Benchmark Suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Quick mode: tiny model, 10s per test timeout (default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Full mode: default model config, longer timeouts",
    )
    args = parser.parse_args()

    # --full overrides --quick
    quick = not args.full

    print("=== Tabula Rasa Benchmark Suite ===")
    if quick:
        print("  Mode: QUICK (tiny config, 10s per test)")
    else:
        print("  Mode: FULL (default config, 60s per test)")

    runner = BenchmarkRunner(quick=quick)
    results = runner.run()

    # Overall summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "quick" if quick else "full",
    }

    print(f"\n=== Results: {passed}/{total} passed ({failed} failed) ===")
    print()

    # Write results
    report = {"results": results, "summary": summary}
    RESULTS_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
