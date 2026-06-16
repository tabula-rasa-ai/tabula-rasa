"""Router server — dispatches to math or chat specialist models.

Auto-detects if a query is math or chat, routes accordingly.
Math queries go to operation-specific specialists (add/sub/mul/div).
Chat queries go to the chat specialist.

Usage:
    python3 specialist_router.py

Then:
    curl http://localhost:8001/generate -d '{"prompt":"12+34="}'
    curl http://localhost:8001/generate -d '{"prompt":"Hello!"}'
    curl http://localhost:8001/health
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import argparse
import json
import re
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse

import torch

from tabula_rasa.bpe_tokenizer import BPETokenizer
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

# Neural router (Neocortex)
from egefalos.router_model import RouterModel, RouterConfig, INTENT_NAMES, vector_to_intents

# Remote chat server config (GPU machine or local)
REMOTE_CHAT_URL = os.environ.get("CHAT_SERVER_URL", "http://127.0.0.1:8201")

OPS = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

# Math-like patterns: contains digits close to arithmetic ops
MATH_RE = re.compile(r"\d+\s*[+\-*/]\s*\d+")

# Questions with "=" already — likely chat prompts formatted for the model
# (like "Hello!=" or "What is 2+2?=")
SEPARATOR_EQUALS = re.compile(r".+=$")


def detect_intent(prompt: str) -> str:
    """Detect intent using neural router (Neocortex), with regex fallback.
    
    Uses the trained NeuralRouterManager when available, falls back
    to regex pattern matching if the router isn't loaded.
    
    Returns 'math' | 'chat' | 'code' | 'memory'.
    """
    # Use neural router if loaded
    global router_manager
    if router_manager is not None and router_manager.loaded:
        result = router_manager.classify(prompt)
        intent = result["primary"]
        # Map router intents to routing labels
        mapping = {"MATH": "math", "CODE": "code", "MEMORY": "memory", "CHAT": "chat"}
        return mapping.get(intent, "chat")
    
    # Regex fallback
    p = prompt.rstrip("=").strip()
    if MATH_RE.search(p):
        return "math"
    if re.match(r"^\d+\s*[+\-*/]\s*\d+$", p):
        return "math"
    has_math_op = any(op in p for op in OPS)
    has_digits = bool(re.search(r"\d", p))
    if has_math_op and has_digits:
        return "math"
    return "chat"


def detect_operation(prompt: str) -> str:
    """Detect which math operation from the prompt."""
    for sym, name in OPS.items():
        if sym in prompt:
            return name
    return "add"  # fallback


class MathSpecialistManager:
    """Loads and manages math specialist models (add/sub/mul/div)."""

    def __init__(self, specialists_dir="specialists"):
        self.dir = Path(specialists_dir)
        self.math_dir = self.dir / "math"
        self.models = {}
        self.tokenizers = {}
        self.device = "cpu"

        math_dir = self.math_dir if self.math_dir.exists() else self.dir
        print(f"  [DEBUG] math_dir={math_dir.resolve()}, exists={math_dir.exists()}")
        for op in ["add", "sub", "mul", "div"]:
            op_dir = math_dir / op
            tok_path = op_dir / "tokenizer.json"
            print(f"  [DEBUG] {op}: op_dir={op_dir.resolve()}, tok_exists={tok_path.exists()}")
            ckpt = op_dir / "best.pt"
            if not ckpt.exists():
                ckpt = op_dir / "final.pt"
            if not ckpt.exists():
                print(f"  [!] No checkpoint for math/{op}. Train first.")
                continue

            tok = MathTokenizer.load(str(op_dir / "tokenizer.json"))
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            sd = state.get("model_state_dict", state)
            actual_vocab = sd["token_embedding.weight"].shape[0] if isinstance(sd, dict) else tok.vocab_size

            cfg = Config()
            cfg.vocab_size = actual_vocab
            tok.max_seq_len = cfg.max_seq_len

            model = MathTransformer(cfg)
            model.load_state_dict(sd)
            model.eval()
            model.to(self.device)

            self.models[op] = model
            self.tokenizers[op] = tok
            params = count_parameters(model)
            acc = state.get("acc", "?")
            print(f"  [*] math/{op}: {params:,} params, best_acc={acc}%")

        print(f"  Math specialists: {len(self.models)} loaded")

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

        needs_reverse = op in ("add", "sub")
        processed_prompt = prompt
        if needs_reverse:
            nums = re.findall(r"\d+", prompt)
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

        result = full
        if needs_reverse and "=" in full:
            ans_part = full.split("=")[-1].strip()
            ans_rev = ans_part[::-1]
            result = f'{prompt.rstrip("=")}={ans_rev}'

        return {
            "prompt": prompt,
            "result": result,
            "specialist": op,
            "type": "math",
            "reversed": needs_reverse,
            "operation": next((c for c in prompt if c in "+-*/"), "?"),
            "time_ms": round(elapsed * 1000),
        }


class ChatSpecialistManager:
    """Loads and manages the chat specialist model."""

    def __init__(self, specialists_dir="specialists"):
        self.dir = Path(specialists_dir)
        self.chat_dir = self.dir / "chat"
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        ckpt = self.chat_dir / "best.pt"
        if not ckpt.exists():
            ckpt = self.chat_dir / "final.pt"
        if not ckpt.exists():
            print("  [!] No chat specialist checkpoint found. Train first:")
            print("      python3 train_chat_specialist.py")
            return

        # Load tokenizer
        tok_path = self.chat_dir / "tokenizer.json"
        if tok_path.exists():
            tok = BPETokenizer.load(str(tok_path))
        else:
            tok = BPETokenizer()

        state = torch.load(ckpt, map_location=self.device, weights_only=True)
        sd = state.get("model_state_dict", state)
        actual_vocab = sd["token_embedding.weight"].shape[0] if isinstance(sd, dict) else tok.vocab_size

        # Auto-detect architecture from checkpoint
        cfg = Config()
        cfg.vocab_size = actual_vocab
        cfg.use_reversed = False
        cfg.use_scratchpad = False
        ckpt_d_model = sd["token_embedding.weight"].shape[1]
        ckpt_n_layers = len([k for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight")])
        cfg.d_model = ckpt_d_model
        cfg.n_layers = ckpt_n_layers
        if ckpt_n_layers > 0:
            cfg.d_ff = sd["layers.0.feed_forward.w1.weight"].shape[0]
            # Detect head_dim from RoPE inv_freq (half of head_dim)
            inv_freq = sd.get("layers.0.attention.rope.inv_freq",
                              sd.get("layers.0.attention.rope.inv_freq", None))
            if inv_freq is not None:
                head_dim = inv_freq.shape[0] * 2
                cfg.n_heads = ckpt_d_model // head_dim
            else:
                cfg.n_heads = max(1, ckpt_d_model // 64)
        tok.max_seq_len = cfg.max_seq_len

        model = MathTransformer(cfg)
        model.load_state_dict(sd)
        model.eval()
        model.to(self.device)

        self.model = model
        self.tokenizer = tok
        self.params = count_parameters(model)
        self.acc = state.get("acc", "?")

        print(f"  [*] chat: {self.params:,} params, best_acc={self.acc}%")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def generate(self, prompt: str, temperature=0.5, top_k=10, max_new_tokens=64):
        model = self.model
        tok = self.tokenizer

        t0 = time.time()

        # Don't append = if it's a chat prompt — the prompt itself may not have it
        # Chat prompts end with = (training format: "Hello!=")
        if not prompt.endswith("="):
            processed = prompt + "="
        else:
            processed = prompt

        full = model.generate(
            tok,
            processed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            clean_output=False,
            repetition_penalty=1.3,
        )
        elapsed = time.time() - t0

        # Extract the response (everything after the '=' separator)
        if "=" in full:
            response = full.split("=", 1)[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
        else:
            response = full.replace("<EOS>", "").replace("<PAD>", "").strip()

        return {
            "prompt": prompt,
            "result": full,
            "response": response,
            "specialist": "chat",
            "type": "chat",
            "time_ms": round(elapsed * 1000),
        }


class RemoteChatManager:
    """Forwards chat queries to a GPU-hosted LLM server.
    
    Set CHAT_SERVER_URL env var to the remote server address.
    Example: CHAT_SERVER_URL=http://192.168.1.100:8200
    """

    def __init__(self):
        self.url = REMOTE_CHAT_URL
        if self.url:
            print(f"  [*] Remote chat: {self.url}")
            # Test connection
            try:
                urllib.request.urlopen(f"{self.url}/", timeout=3)
                print(f"  [*] Remote chat: connected")
                self._connected = True
            except Exception as e:
                print(f"  [!] Remote chat: {e}")
                self._connected = False
        else:
            print("  [*] Remote chat: disabled (set CHAT_SERVER_URL)")
            self._connected = False

    @property
    def loaded(self) -> bool:
        return self._connected

    def generate(self, prompt: str, temperature=0.7, top_k=10, max_new_tokens=30):
        t0 = time.time()
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.url}/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            response = data.get("response", "")
        except Exception as e:
            response = f"[Chat server error: {e}]"
        elapsed = time.time() - t0
        return {
            "prompt": prompt,
            "result": response,
            "response": response,
            "specialist": "remote-llm",
            "type": "chat",
            "time_ms": round(elapsed * 1000),
        }


# ── Global managers ──────────────────────────────────────────────
math_manager = None
chat_manager = None
router_manager = None
remote_chat = None


class NeuralRouterManager:
    """Loads and manages the trained Neocortex Router Transformer.
    
    A tiny 541K-param classifier that routes prompts to the correct
    specialist (MATH, CODE, MEMORY, CHAT) with 99.3% accuracy.
    Uses mean pooling over all tokens for robust classification.
    """

    def __init__(self, router_dir="checkpoints/router"):
        self.device = "cpu"
        self.model = None
        self.tokenizer = None

        ckpt = Path(router_dir) / "best.pt"
        tok_path = Path(router_dir) / "tokenizer.json"
        if not ckpt.exists() or not tok_path.exists():
            print(f"  [!] No neural router checkpoint at {router_dir}")
            return

        # Load tokenizer
        self.tokenizer = BPETokenizer.load(str(tok_path))

        # Load model
        self.model = RouterModel.load(str(ckpt))
        self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        print(f"  [*] Neural router: {params:,} params, {len(INTENT_NAMES)} intents")

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @torch.no_grad()
    def classify(self, prompt: str, threshold: float = 0.5) -> dict:
        """Classify a prompt into intent category.
        
        Returns:
            dict with 'primary' (str), 'intents' (list), 'confidence' (float)
        """
        if not self.loaded:
            return {"primary": "CHAT", "intents": ["CHAT"], "confidence": 0.0}

        # Encode
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) > self.model.config.max_seq_len:
            ids = ids[:self.model.config.max_seq_len - 1] + [self.tokenizer.eos_id]

        input_ids = torch.tensor([ids], device=self.device)

        # Predict
        logits = self.model(input_ids)
        result = vector_to_intents(
            logits[0], 
            multi_label=self.model.config.multi_label,
            threshold=threshold,
        )
        return result


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
            math_ops = math_manager.loaded_ops if math_manager else []
            chat_loaded = chat_manager is not None and chat_manager.loaded
            self._send_json({
                "status": "ok",
                "math_specialists": math_ops,
                "chat_loaded": chat_loaded,
                "endpoints": {
                    "GET /health": "Status and loaded specialists",
                    "GET /skills": "List available skills",
                    "POST /generate": "Auto-routes to math or chat specialist",
                    "POST /chat": "Force route to chat specialist",
                    "POST /math": "Force route to math specialist",
                    "POST /ask": "Ask a question (dashboard-compatible)",
                },
            })
        elif path == "/skills":
            skills = []
            if math_manager and math_manager.loaded_ops:
                for op in math_manager.loaded_ops:
                    skills.append({"name": f"math/{op}", "status": "ready", "description": f"{op.capitalize()} arithmetic"})
            if chat_manager and chat_manager.loaded:
                skills.append({"name": "chat", "status": "ready", "description": "General chat and Q&A"})
            self._send_json({"skills": skills})
        elif path == "/docs":
            self._send_json({
                "endpoints": {
                    "GET /health": "Status and loaded specialists",
                    "GET /skills": "List available skills",
                    "POST /generate": "Auto-route: detects math vs chat",
                    "POST /ask": "Ask a question (dashboard-compatible)",
                    "POST /chat": "Force chat generation",
                    "POST /math": "Force math generation",
                },
                "usage": {
                    "math": 'curl -X POST -H "Content-Type: application/json" '
                            '-d \'{"prompt":"12+34="}\' http://localhost:8001/generate',
                    "chat": 'curl -X POST -H "Content-Type: application/json" '
                            '-d \'{"question":"Hello!"}\' http://localhost:8001/ask',
                },
            })
        else:
            self._send_json({"error": "Not found", "endpoints": ["/health", "/skills", "/generate", "/ask", "/chat", "/math"]}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

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
        question = data.get("question", "").strip()
        if not prompt and not question:
            self._send_json({"error": "Missing prompt or question"}, 400)
            return

        # Allow question field for /ask endpoint
        effective_prompt = question or prompt

        temperature = float(data.get("temperature", 0.5))
        top_k = int(data.get("top_k", 10))
        max_new_tokens = int(data.get("max_tokens", 64))

        try:
            if path == "/chat":
                # Force chat routing
                if chat_manager is None or not chat_manager.loaded:
                    self._send_json({"error": "Chat specialist not loaded. Train it first: python3 train_chat_specialist.py"}, 503)
                    return
                result = chat_manager.generate(effective_prompt, temperature, top_k, max_new_tokens)
                self._send_json(result)

            elif path == "/math":
                # Force math routing
                if math_manager is None or not math_manager.loaded_ops:
                    self._send_json({"error": "No math specialists loaded"}, 503)
                    return
                result = math_manager.generate(effective_prompt, temperature, top_k, max_new_tokens)
                self._send_json(result)

            elif path == "/generate":
                # Auto-detect using neural router
                intent = detect_intent(effective_prompt)
                if intent == "math" and math_manager and math_manager.loaded_ops:
                    result = math_manager.generate(effective_prompt, temperature, top_k, max_new_tokens)
                elif intent in ("chat", "code", "memory"):
                    # Try remote LLM first, fall back to local chat specialist
                    if remote_chat and remote_chat.loaded:
                        result = remote_chat.generate(effective_prompt, temperature, top_k, max_new_tokens)
                    elif chat_manager and chat_manager.loaded:
                        result = chat_manager.generate(effective_prompt, temperature, top_k, max_new_tokens)
                    else:
                        result = {"error": "No chat specialist available"}
                elif chat_manager and chat_manager.loaded:
                    result = chat_manager.generate(effective_prompt, temperature, top_k, max_new_tokens)
                elif math_manager and math_manager.loaded_ops:
                    result = math_manager.generate(effective_prompt, temperature, top_k, max_new_tokens)
                else:
                    self._send_json({"error": "No specialists available"}, 503)
                    return
                result["detected_intent"] = intent
                self._send_json(result)

            elif path == "/ask":
                # Dashboard-compatible endpoint — check intent first
                if not question:
                    self._send_json({"error": "Missing question"}, 400)
                    return

                # Use neural router to detect intent
                intent = detect_intent(question)
                
                # Math queries go to math specialist
                if intent == "math" and math_manager and math_manager.loaded_ops:
                    result = math_manager.generate(question, temperature, top_k, max_new_tokens)
                    self._send_json({
                        "knows": True,
                        "answer": result.get("result", ""),
                        "skill": f"math/{result.get('specialist', '?')}",
                        "confidence": 90,
                        "time_ms": result.get("time_ms", 0),
                    })
                    return
                
                # Chat queries go to remote LLM or local chat specialist
                if remote_chat and remote_chat.loaded:
                    result = remote_chat.generate(question, temperature, top_k, max_new_tokens)
                    self._send_json({
                        "knows": True,
                        "answer": result["response"],
                        "skill": "llm",
                        "confidence": 90,
                        "time_ms": result["time_ms"],
                    })
                elif chat_manager and chat_manager.loaded:
                    result = chat_manager.generate(question, temperature, top_k, max_new_tokens)
                    self._send_json({
                        "knows": True,
                        "answer": result["response"],
                        "skill": "chat",
                        "confidence": 50,
                        "time_ms": result["time_ms"],
                        "analysis": None,
                        "suggested_skills": [],
                    })
                elif math_manager and math_manager.loaded_ops:
                    result = math_manager.generate(question, temperature, top_k, max_new_tokens)
                    self._send_json({
                        "knows": True,
                        "answer": result.get("result", ""),
                        "skill": f"math/{result.get('specialist', '?')}",
                        "confidence": 50,
                        "time_ms": result.get("time_ms", 0),
                    })
                else:
                    self._send_json({
                        "knows": False,
                        "answer": "I don't know about that yet.",
                        "skill": "",
                        "confidence": 0,
                        "analysis": "No specialist is trained for this topic.",
                        "suggested_skills": ["chat"],
                    })

            else:
                self._send_json({"error": "Not found", "endpoints": ["/generate", "/chat", "/math"]}, 404)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def main():
    global math_manager, chat_manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--dir", type=str, default="specialists")
    parser.add_argument("--no-chat", action="store_true", help="Skip loading chat specialist")
    parser.add_argument("--no-math", action="store_true", help="Skip loading math specialists")
    args = parser.parse_args()

    print(f"[*] Loading specialists from {args.dir}/")

    if not args.no_math:
        print("  [*] Loading math specialists...")
        math_manager = MathSpecialistManager(args.dir)
    else:
        print("  [*] Math specialists disabled (--no-math)")

    if not args.no_chat:
        print("  [*] Loading chat specialist...")
        chat_manager = ChatSpecialistManager(args.dir)
    else:
        print("  [*] Chat specialist disabled (--no-chat)")

    print("  [*] Loading neural router (Neocortex)...")
    global router_manager, remote_chat
    router_manager = NeuralRouterManager()

    # Try local LLM server
    remote_chat = RemoteChatManager()

    if (math_manager is None or not math_manager.loaded_ops) and \
       (chat_manager is None or not chat_manager.loaded):
        print("[!] No specialists available. Train them first:")
        print("    python3 train_specialist.py add     # Math")
        print("    python3 train_chat_specialist.py     # Chat")
        sys.exit(1)

    print(f"[*] Starting router on http://{args.host}:{args.port}")

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ThreadedHTTPServer((args.host, args.port), RouterHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
