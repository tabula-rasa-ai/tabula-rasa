"""Router server — dispatches math problems to specialist models.

Usage:
    python3 specialist_router.py

Then:
    curl http://localhost:8001/generate -d '{"prompt":"12+34="}'
    curl http://localhost:8001/health
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

OPS = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


def detect_operation(prompt: str) -> str:
    """Detect which operation from the prompt."""
    for sym, name in OPS.items():
        if sym in prompt:
            return name
    return "add"  # fallback


class SpecialistManager:
    """Loads and manages all 4 specialist models."""

    def __init__(self, specialists_dir="specialists"):
        self.dir = Path(specialists_dir)
        self.models = {}
        self.tokenizers = {}
        self.device = "cpu"

        for op in ["add", "sub", "mul", "div"]:
            op_dir = self.dir / op
            ckpt = op_dir / "best.pt"
            if not ckpt.exists():
                ckpt = op_dir / "final.pt"
            if not ckpt.exists():
                print(f"  [!] No checkpoint for {op}. Train first.")
                continue

            tok = MathTokenizer.load(str(op_dir / "tokenizer.json"))
            cfg = Config()
            cfg.vocab_size = tok.vocab_size
            tok.max_seq_len = cfg.max_seq_len

            model = MathTransformer(cfg)
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            model.to(self.device)

            self.models[op] = model
            self.tokenizers[op] = tok
            params = count_parameters(model)
            acc = state.get("acc", "?")
            print(f"  [*] {op}: {params:,} params, best_acc={acc}%")

        print(f"  Loaded {len(self.models)} specialists")

    @property
    def loaded_ops(self):
        return list(self.models.keys())

    @torch.no_grad()
    def generate(self, prompt: str, temperature=0.3, top_k=5, max_new_tokens=15):
        op = detect_operation(prompt)
        if op not in self.models:
            return {"error": f'No specialist for operation "{op}"', "detected": op}

        model = self.models[op]
        tok = self.tokenizers[op]
        model.eval()

        # ── Reversed inference for add/sub ─────────────────────
        # Training uses reversed digits so carries align with
        # left-to-right generation. At inference, we reverse the
        # input operands, feed to model, and reverse the output.
        needs_reverse = op in ("add", "sub")
        processed_prompt = prompt
        if needs_reverse:
            # Extract operands: "12+34=" → reverse digits → "21+43="
            import re as _re

            nums = _re.findall(r"\d+", prompt)
            if len(nums) >= 2:
                a_rev = nums[0][::-1]
                b_rev = nums[1][::-1]
                op_sym = "+" if op == "add" else "-"
                processed_prompt = f"{a_rev}{op_sym}{b_rev}="

        t0 = time.time()
        if not processed_prompt.endswith("="):
            processed_prompt += "="

        full = model.generate(
            tok,
            processed_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        elapsed = time.time() - t0

        # Reverse the answer back for add/sub
        result = full
        if needs_reverse and "=" in full:
            ans_part = full.split("=")[-1].strip()
            ans_rev = ans_part[::-1]
            result = f'{prompt.rstrip("=")}={ans_rev}'

        return {
            "prompt": prompt,
            "result": result,
            "specialist": op,
            "reversed": needs_reverse,
            "operation": {"+": "add", "-": "sub", "*": "mul", "/": "div"}.get(
                next((c for c in prompt if c in "+-*/"), "?"), "?"
            ),
            "time_ms": round(elapsed * 1000),
        }


manager = None


class RouterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"  [*] {args[0]} {args[1]} {args[2]}\n")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "specialists": manager.loaded_ops,
                    "count": len(manager.loaded_ops),
                }
            )
        else:
            self._send_json(
                {
                    "endpoints": {
                        "GET /health": "Status and loaded specialists",
                        "POST /generate": "Solve math (auto-routes to specialist)",
                    }
                }
            )

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/generate":
            self._send_json({"error": "Not found"}, 404)
            return

        cl = int(self.headers.get("Content-Length", 0))
        if cl == 0:
            self._send_json({"error": "Empty body"}, 400)
            return

        try:
            data = json.loads(self.rfile.read(cl))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        prompt = data.get("prompt", "").strip()
        if not prompt:
            self._send_json({"error": "Missing prompt"}, 400)
            return

        temperature = float(data.get("temperature", 0.3))
        top_k = int(data.get("top_k", 5))

        try:
            result = manager.generate(prompt, temperature, top_k)
            self._send_json(result)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def main():
    global manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--dir", type=str, default="specialists")
    args = parser.parse_args()

    print(f"[*] Loading specialists from {args.dir}/")
    manager = SpecialistManager(args.dir)

    if not manager.loaded_ops:
        print("[!] No specialists found. Train them first:")
        print("    python3 train_specialist.py add")
        print("    python3 train_specialist.py sub")
        print("    python3 train_specialist.py mul")
        print("    python3 train_specialist.py div")
        sys.exit(1)

    print(f"[*] Starting router on http://{args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), RouterHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
