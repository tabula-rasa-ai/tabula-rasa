"""HTTP API server for the math transformer model.

Usage:
    python3 serve.py                   # Start server (default port 8000)
    python3 serve.py --port 8080       # Custom port
    python3 serve.py --checkpoint path # Custom checkpoint

Then:
    curl http://localhost:8000/generate -d '{"prompt":"12+34="}'
    curl http://localhost:8000/train    -d '{"pairs":[["12+34","46"]]}'
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import argparse
import json
import mimetypes
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

# SSE client registry: list of (queue, last_event_id) for streaming
SSE_CLIENTS: list[list] = []  # [[queue, last_id], ...]


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — handles concurrent requests."""

    allow_reuse_address = True
    daemon_threads = True


import torch
import torch.nn as nn
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer


class ModelHandler:
    """Loads and runs the math transformer model."""

    def __init__(self, checkpoint_path: str):
        self.device = "cpu"
        self.cfg = Config()
        self.config_path = Path(checkpoint_path).parent
        tokenizer_path = self.config_path / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = Path("checkpoints/tokenizer.json")

        self.tokenizer = MathTokenizer.load(str(tokenizer_path))
        self.cfg.vocab_size = self.tokenizer.vocab_size  # type: ignore[misc]
        self.tokenizer.max_seq_len = self.cfg.max_seq_len

        self.model = MathTransformer(self.cfg)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.params = count_parameters(self.model)
        self.name = f"MathTransformer-{self.params // 1000}K"
        self.checkpoint_path = checkpoint_path
        self.total_corrections = 0
        self.optimizer = None

    def _get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        return self.optimizer

    @torch.no_grad()
    def generate(
        self, prompt: str, temperature: float = 0.3, top_k: int = 5, max_new_tokens: int = 15
    ) -> dict:
        """Generate completion for a prompt."""
        self.model.eval()
        t0 = time.time()

        if not prompt.endswith("="):
            prompt += "="

        full = self.model.generate(
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        elapsed = time.time() - t0
        # Parse scratchpad: extract digit values from carry-digit pairs
        display = full
        if "=" in full:
            raw = full.split("=", 1)[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
            if len(raw) >= 2:
                # scratchpad format: carry-digit per column, LSD-first
                digits = raw[1::2]  # every 2nd char starting at 1 = digit values
                carries = raw[0::2]  # every 2nd char starting at 0 = carry values
                # Reverse LSD-first -> normal order
                normal = digits[::-1]
                # If last column had a carry, prepend '1'
                if carries and carries[-1] == "1":
                    display_str = "1" + normal
                else:
                    display_str = normal.lstrip("0") or "0"
            else:
                display_str = raw
            display = display_str
        return {
            "prompt": prompt,
            "result": full,
            "display": display,
            "tokens": len(self.tokenizer.encode(full)),
            "time_ms": round(elapsed * 1000),
            "model": self.name,
        }

    def train_step(self, pairs: list[tuple[str, str]], epochs: int = 5) -> dict:
        """Fine-tune on (expression, answer) pairs immediately."""
        self.model.train()
        opt = self._get_optimizer()

        # Tokenize all pairs
        input_ids_list = []
        target_ids_list = []
        for expr, answer in pairs:
            text = f"{expr}={answer}"
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > self.cfg.max_seq_len:
                ids = ids[: self.cfg.max_seq_len]
            padded = ids + [self.tokenizer.pad_id] * (self.cfg.max_seq_len - len(ids))
            x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0)
            input_ids_list.append(x)
            target_ids_list.append(y)

        x_batch = torch.cat(input_ids_list, dim=0)
        y_batch = torch.cat(target_ids_list, dim=0)

        losses = []
        for epoch in range(epochs):
            opt.zero_grad()
            _, loss, _ = self.model(x_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        self.total_corrections += len(pairs)

        # Save checkpoint
        ckpt_dir = Path(self.checkpoint_path).parent
        ckpt_path = ckpt_dir / "finetuned.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "corrections": self.total_corrections,
                "loss": losses[-1],
            },
            ckpt_path,
        )

        self.model.eval()
        return {
            "status": "ok",
            "pairs": len(pairs),
            "epochs": epochs,
            "final_loss": losses[-1],
            "losses": losses,
            "total_corrections": self.total_corrections,
            "checkpoint": str(ckpt_path),
        }

    def train_on_own_mistakes(self, prompt: str, correct_answer: str, epochs: int = 5) -> dict:
        """Generate a wrong prediction, then train on the correct answer."""
        expr = prompt.rstrip("=")

        # Train on (expr, correct_answer)
        result = self.train_step([(expr, correct_answer)], epochs)

        return {
            **result,
            "corrected": f"{expr}={correct_answer}",
        }


# ─── Editable config fields (name → (line_regex, type_fn)) ──────────
CONFIG_FIELDS = {
    # Architecture
    "d_model": (r"^    d_model(?::\s*\w+)?\s*=", int),
    "n_layers": (r"^    n_layers(?::\s*\w+)?\s*=", int),
    "n_heads": (r"^    n_heads(?::\s*\w+)?\s*=", int),
    "d_ff": (r"^    d_ff(?::\s*\w+)?\s*=", int),
    "max_seq_len": (r"^    max_seq_len(?::\s*\w+)?\s*=", int),
    "dropout": (r"^    dropout(?::\s*\w+)?\s*=", float),
    "activation": (r"^    activation(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "norm_type": (r"^    norm_type(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "pos_encoding": (r"^    pos_encoding(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "weight_init": (r"^    weight_init(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "init_std": (r"^    init_std(?::\s*\w+)?\s*=", float),
    # Training
    "batch_size": (r"^    batch_size(?::\s*\w+)?\s*=", int),
    "learning_rate": (r"^    learning_rate(?::\s*\w+)?\s*=", float),
    "weight_decay": (r"^    weight_decay(?::\s*\w+)?\s*=", float),
    "warmup_steps": (r"^    warmup_steps(?::\s*\w+)?\s*=", int),
    "max_steps": (r"^    max_steps(?::\s*\w+)?\s*=", int),
    "optimizer": (r"^    optimizer(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "lr_schedule": (r"^    lr_schedule(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "adam_beta1": (r"^    adam_beta1(?::\s*\w+)?\s*=", float),
    "adam_beta2": (r"^    adam_beta2(?::\s*\w+)?\s*=", float),
    "grad_clip_norm": (r"^    grad_clip_norm(?::\s*\w+)?\s*=", float),
    "label_smoothing": (r"^    label_smoothing(?::\s*\w+)?\s*=", float),
    "use_reversed": (r"^    use_reversed(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "use_loss_masking": (r"^    use_loss_masking(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "use_hard_negative": (r"^    use_hard_negative(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "tie_embeddings": (r"^    tie_embeddings(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "eval_temperature": (r"^    eval_temperature(?::\s*\w+)?\s*=", float),
    "eval_max_tokens": (r"^    eval_max_tokens(?::\s*\w+)?\s*=", int),
    # Data
    "train_samples": (r"^    train_samples(?::\s*\w+)?\s*=", lambda s: int(s.replace("_", ""))),
    "eval_samples": (r"^    eval_samples(?::\s*\w+)?\s*=", lambda s: int(s.replace("_", ""))),
    "min_digits": (r"^    min_digits(?::\s*\w+)?\s*=", int),
    "max_digits": (r"^    max_digits(?::\s*\w+)?\s*=", int),
}


def _config_path():
    """Resolve config.py location (moved to src/tabula_rasa/ in refactor)."""
    p = Path("src/tabula_rasa/config.py")
    if p.exists():
        return p
    return Path("config.py")


def get_config_dict():
    """Read current config values from config.py."""
    text = _config_path().read_text()
    vals = {}
    for name, (pat, caster) in CONFIG_FIELDS.items():
        m = re.search(pat + r"\s*([^#\n]+)", text, re.MULTILINE)
        if m:
            try:
                vals[name] = caster(m.group(1).strip())
            except (ValueError, TypeError):
                vals[name] = None
    # Compute param estimate
    # Use actual vocab from loaded model if available, else estimate from config
    tok_vocab = (
        model_handler.tokenizer.vocab_size
        if model_handler is not None and model_handler.tokenizer is not None
        else 24
    )
    embed = tok_vocab * vals.get("d_model", 0)
    lm_head = vals.get("d_model", 0) * tok_vocab
    d_model = vals.get("d_model", 0)
    n_layers = vals.get("n_layers", 0)
    d_ff = vals.get("d_ff", 0)
    n_heads = vals.get("n_heads", 1)
    head_dim = d_model // n_heads if n_heads else 0
    attn_per = 4 * d_model * d_model
    ffn_per = 3 * d_model * d_ff
    total = embed + lm_head + n_layers * (attn_per + ffn_per)
    vals["_param_estimate"] = total
    vals["_vocab_size"] = tok_vocab
    return vals


def save_config(values: dict) -> list[str]:
    """Update config.py with new values. Returns list of changes made."""
    path = _config_path()
    text = path.read_text()
    changes = []
    for name, value in values.items():
        if name.startswith("_"):
            continue
        entry = CONFIG_FIELDS.get(name)
        if not entry:
            continue
        pat, caster = entry
        m = re.search(pat + r"\s*([^#\n]+)", text, re.MULTILINE)
        if m:
            old_val = m.group(1).strip()
            if isinstance(value, bool):
                new_str = "True" if value else "False"
            elif isinstance(value, float):
                new_str = f"{value:g}"
            elif isinstance(value, str) and name in (
                "optimizer",
                "lr_schedule",
                "activation",
                "norm_type",
                "pos_encoding",
                "weight_init",
            ):
                new_str = f"'{value}'"
            else:
                new_str = str(value)
            if name == "train_samples" or name == "eval_samples":
                new_str = f"{value:_}" if isinstance(value, int) and value >= 10000 else str(value)
            full_line = m.group(0)
            eq_idx = full_line.index("=")
            line_prefix = full_line[: eq_idx + 1]
            old_line = full_line
            new_line = line_prefix + " " + new_str
            text = text.replace(old_line, new_line, 1)
            changes.append(f"{name}: {old_val} → {new_str}")
    path.write_text(text)
    return changes


# ─── Shared model instance ────────────────────────────────────
model_handler: ModelHandler | None = None


def _read_body(handler) -> dict | None:
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length == 0:
        return None
    body = handler.rfile.read(content_length)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler."""

    def log_message(self, format, *args):
        sys.stderr.write(f"  [*] {args[0]} {args[1]} {args[2]}\n")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body.encode())))
            self.end_headers()
            self.wfile.write(body.encode())
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # browser closed tab mid-response — harmless

    def _handle_sse(self):
        """Server-Sent Events endpoint — streams training progress."""
        import queue
        q: queue.Queue = queue.Queue()
        client = [q, 0]
        SSE_CLIENTS.append(client)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            # Send initial keepalive
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    data = q.get(timeout=30)
                    if data is None:  # shutdown signal
                        break
                    msg = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Send keepalive comment every 30s
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            if client in SSE_CLIENTS:
                SSE_CLIENTS.remove(client)


def broadcast_sse(event: str | dict) -> None:
    """Broadcast an event to all connected SSE clients.

    Args:
        event: Event data (dict is JSON-serialized, str sent as-is).
    """
    dead = []
    for client in SSE_CLIENTS:
        try:
            client[0].put_nowait(event)
            client[1] += 1
        except Exception:
            dead.append(client)
    for c in dead:
        if c in SSE_CLIENTS:
            SSE_CLIENTS.remove(c)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ── SSE streaming endpoint ──
        if path == "/events":
            self._handle_sse()
            return

        # ── API endpoints ──
        if path == "/training-progress":
            self._send_json(get_training_progress())
        elif path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "model": model_handler.name,
                    "params": model_handler.params,
                    "vocab_size": model_handler.tokenizer.vocab_size,
                    "corrections": model_handler.total_corrections,
                    "training_running": "yes" if model_handler.optimizer is not None else "no",
                }
            )
            return

        # ── API: config ──
        if path == "/api/config":
            self._send_json(get_config_dict())
            return

        # ── API: list checkpoints ──
        if path == "/api/checkpoints":
            import torch

            base = Path(__file__).parent
            results = []
            for p in sorted(base.rglob("*.pt")):
                rel = str(p.relative_to(base))
                if "__pycache__" in rel or "old_checkpoints" in rel:
                    continue
                if p.name in ("inputs.pt", "targets.pt"):
                    continue
                size = p.stat().st_size
                entry = {"path": rel, "name": p.name, "size": size, "size_kb": size // 1024}
                try:
                    ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
                    if isinstance(ckpt, dict):
                        entry["step"] = ckpt.get("step")
                        entry["acc"] = ckpt.get("acc")
                        entry["loss"] = ckpt.get("loss")
                        sd = ckpt.get("model_state_dict", {})
                        if sd:
                            entry["params"] = sum(v.numel() for v in sd.values())
                            n = sum(1 for k in sd if "attention.wq.weight" in k)
                            entry["layers"] = n
                except Exception:
                    pass
                results.append(entry)
            self._send_json({"checkpoints": results, "total": len(results)})
            return

        # ── GPU deployment package ──
        if path == "/tabula-rasa-gpu.tar.gz":
            gz_path = Path(__file__).parent / "tabula-rasa-gpu.tar.gz"
            if gz_path.exists():
                data = gz_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header(
                    "Content-Disposition", 'attachment; filename="tabula-rasa-gpu.tar.gz"'
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
            else:
                self._send_json(
                    {"error": "Package not found. Build it first from the GPU Training page."}, 404
                )
            return

        # ── static files from Dashboard/ ──
        dash_dir = Path(__file__).parent / "Dashboard"
        if path == "/":
            path = "/dashboard.html"
        # strip leading slash, resolve relative to Dashboard/
        rel = path.lstrip("/")
        file_path = (dash_dir / rel).resolve()
        # safety: must stay inside Dashboard/
        try:
            file_path.relative_to(dash_dir.resolve())
        except ValueError:
            self._send_json({"error": "Forbidden"}, 403)
            return

        if file_path.is_file():
            ext = file_path.suffix.lower()
            ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        data = _read_body(self)
        if data is None:
            self._send_json({"error": "Empty request body"}, 400)
            return

        if path == "/generate":
            if model_handler is None:
                self._send_json({"error": "No model loaded. Train first."}, 503)
                return
            prompt = data.get("prompt", "").strip()
            if not prompt:
                self._send_json({"error": 'Missing "prompt"'}, 400)
                return
            temperature = float(data.get("temperature", 0.3))
            top_k = int(data.get("top_k", 5))
            try:
                result = model_handler.generate(prompt, temperature, top_k)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/train":
            pairs = data.get("pairs", [])
            if not pairs or not isinstance(pairs, list):
                self._send_json({"error": 'Missing "pairs" array of [expr, answer]'}, 400)
                return
            epochs = int(data.get("epochs", 5))
            try:
                result = model_handler.train_step(pairs, epochs)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/correct":
            prompt = data.get("prompt", "").strip()
            answer = data.get("answer", "").strip()
            if not prompt or not answer:
                self._send_json({"error": 'Need "prompt" and "answer"'}, 400)
                return
            epochs = int(data.get("epochs", 5))
            try:
                result = model_handler.train_on_own_mistakes(prompt, answer, epochs)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/config":
            if not data:
                self._send_json({"error": "No config data"}, 400)
                return
            changes = save_config(data)
            self._send_json({"status": "saved", "changes": changes})

        elif path == "/api/exec":
            import subprocess

            cmd = data.get("command", "") if data else ""
            if not cmd:
                self._send_json({"error": "No command provided"}, 400)
                return
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=300
                )
                self._send_json(
                    {"exit_code": result.returncode, "output": result.stdout + result.stderr}
                )
            except subprocess.TimeoutExpired:
                self._send_json({"error": "Command timed out"}, 504)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/create-requirements":
            try:
                req_path = Path(__file__).parent / "requirements.txt"
                req_path.write_text("torch>=2.0.0\nnumpy>=1.24.0\ntqdm>=4.60.0\n")
                self._send_json({"status": "created", "path": str(req_path)})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/file-size":
            file_path = data.get("path", "") if data else ""
            if not file_path:
                self._send_json({"error": "No path provided"}, 400)
                return
            try:
                p = Path(file_path)
                if p.exists():
                    self._send_json({"size": p.stat().st_size, "path": str(p)})
                else:
                    self._send_json({"error": "File not found"}, 404)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/clear-training-log":
            try:
                log_path = Path(__file__).parent / "training.log"
                if log_path.exists():
                    log_path.write_text("")
                self._send_json({"status": "cleared"})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/checkpoints":
            import torch

            base = Path(__file__).parent
            results = []
            for p in sorted(base.rglob("*.pt")):
                rel = str(p.relative_to(base))
                if "__pycache__" in rel or "old_checkpoints" in rel:
                    continue
                # Skip data files (inputs.pt, targets.pt)
                if p.name in ("inputs.pt", "targets.pt"):
                    continue
                size = p.stat().st_size
                entry = {"path": rel, "name": p.name, "size": size, "size_kb": size // 1024}
                try:
                    ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
                    if isinstance(ckpt, dict):
                        entry["step"] = ckpt.get("step")
                        entry["acc"] = ckpt.get("acc")
                        entry["loss"] = ckpt.get("loss")
                        sd = ckpt.get("model_state_dict", {})
                        if sd:
                            entry["params"] = sum(v.numel() for v in sd.values())
                            n = sum(1 for k in sd if "attention.wq.weight" in k)
                            entry["layers"] = n
                except Exception:
                    pass
                results.append(entry)
            self._send_json({"checkpoints": results, "total": len(results)})

        elif path == "/api/train/start":
            import io
            import threading
            import uuid

            cmd = data.get("command", "") if data else ""
            if not cmd:
                self._send_json({"error": "No command provided"}, 400)
                return
            job_id = str(uuid.uuid4())[:8]
            output_buf = io.StringIO()

            def run():
                import subprocess

                try:
                    proc = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    TRAIN_JOBS[job_id]["pid"] = proc.pid
                    for line in iter(proc.stdout.readline, ""):
                        output_buf.write(line)
                        TRAIN_JOBS[job_id]["last_output"] = line
                    proc.wait()
                    TRAIN_JOBS[job_id]["exit_code"] = proc.returncode
                    TRAIN_JOBS[job_id]["done"] = True
                except Exception as e:
                    output_buf.write(f"ERROR: {e}\n")
                    TRAIN_JOBS[job_id]["exit_code"] = -1
                    TRAIN_JOBS[job_id]["done"] = True

            TRAIN_JOBS[job_id] = {
                "cmd": cmd,
                "done": False,
                "exit_code": None,
                "buffer": output_buf,
                "last_output": "",
                "pid": None,
                "paused": False,
            }
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            self._send_json({"job_id": job_id, "status": "started"})

        elif path == "/api/train/pause":
            job_id = data.get("job_id", "") if data else ""
            if not job_id or job_id not in TRAIN_JOBS:
                self._send_json({"error": "Job not found"}, 404)
                return
            job = TRAIN_JOBS[job_id]
            if job["done"]:
                self._send_json({"error": "Job already finished"}, 400)
                return
            try:
                import psutil

                parent = psutil.Process(job["pid"])
                for child in parent.children(recursive=True):
                    child.suspend()
                parent.suspend()
                job["paused"] = True
                self._send_json({"status": "paused"})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/train/resume":
            job_id = data.get("job_id", "") if data else ""
            if not job_id or job_id not in TRAIN_JOBS:
                self._send_json({"error": "Job not found"}, 404)
                return
            job = TRAIN_JOBS[job_id]
            if job["done"]:
                self._send_json({"error": "Job already finished"}, 400)
                return
            try:
                import psutil

                parent = psutil.Process(job["pid"])
                for child in parent.children(recursive=True):
                    child.resume()
                parent.resume()
                job["paused"] = False
                self._send_json({"status": "resumed"})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/train/stop":
            job_id = data.get("job_id", "") if data else ""
            if not job_id or job_id not in TRAIN_JOBS:
                self._send_json({"error": "Job not found"}, 404)
                return
            job = TRAIN_JOBS[job_id]
            if job["done"]:
                self._send_json({"error": "Job already finished"}, 400)
                return
            try:
                import psutil

                parent = psutil.Process(job["pid"])
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                job["done"] = True
                job["exit_code"] = -1
                job["paused"] = False
                self._send_json({"status": "stopped"})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/train/status":
            job_id = data.get("job_id", "") if data else ""
            if not job_id or job_id not in TRAIN_JOBS:
                self._send_json({"error": "Job not found"}, 404)
                return
            job = TRAIN_JOBS[job_id]
            buf = job["buffer"]
            output = buf.getvalue()
            self._send_json(
                {
                    "done": job["done"],
                    "exit_code": job["exit_code"],
                    "output": output[-5000:],
                    "last_line": job["last_output"],
                    "paused": job.get("paused", False),
                }
            )

        else:
            self._send_json({"error": "Not found"}, 404)


TRAIN_JOBS = {}

LOG_FILE = Path("training.log")
SPECIALIST_LOG = Path("specialists") / "math" / "add" / "training.log"


def get_training_progress() -> dict:
    """Parse training.log for latest progress."""
    log_file = LOG_FILE
    if SPECIALIST_LOG.exists() and (
        not LOG_FILE.exists() or SPECIALIST_LOG.stat().st_mtime > LOG_FILE.stat().st_mtime
    ):
        log_file = SPECIALIST_LOG
    if not log_file.exists():
        return {"running": False, "error": "No training log found"}

    text = log_file.read_text()
    info = {
        "running": False,
        "step": 0,
        "max_steps": 50000,
        "loss": None,
        "best_acc": None,
        "latest_acc": None,
        "steps_per_sec": None,
        "eta_seconds": None,
        "epoch": None,
        "loss_series": [],
        "acc_series": [],
    }

    # Check if process is still running (last line has "Step" not "complete")
    # Strip null bytes that can corrupt line parsing
    text = text.replace("\x00", "")
    lines = [l for l in text.strip().split(chr(10)) if l.strip()]
    if not lines:
        return info

    last = lines[-1]

    # Parse step: "Step    500/50000 | loss=2.0701 | lr=5.00e-04 | 1.0 steps/s"
    # or new format: "[##---]  600/10000 | loss=2.1114 | 0.4 st/s | ETA 378:04 | lr=3.00e-04"
    step_line = last
    for line in reversed(lines):
        if re.search(r"(?:Step\s+|\[.*?\])\s*\d+/\d+\s+\|\s+loss=", line):
            step_line = line
            break
    m = re.search(r"(?:Step\s+|\[.*?\])\s*(\d+)/(\d+)\s+\|\s+loss=([\d.]+)", step_line)
    if m:
        info["step"] = int(m.group(1))
        info["max_steps"] = int(m.group(2))
        info["loss"] = float(m.group(3))
        info["running"] = True

        # steps/s — handle both "steps/s" and "st/s"
        m2 = re.search(r"\|\s+([\d.]+)\s+(?:st|steps)/s", step_line)
        if m2:
            info["steps_per_sec"] = float(m2.group(1))
            remaining = info["max_steps"] - info["step"]
            if info["steps_per_sec"] > 0:
                info["eta_seconds"] = int(remaining / info["steps_per_sec"])

    # Parse eval: "[*] Eval accuracy: 4.0% (best: 4.0%)"
    # or new: "Acc: 0.0% (best: 0.0%)"
    for line in lines:
        m = re.search(r"(?:Eval accuracy|Acc):\s+([\d.]+)%\s+\(best:\s+([\d.]+)%", line)
        if m:
            info["latest_acc"] = float(m.group(1))
            info["best_acc"] = float(m.group(2))

    # Check if training is complete
    if "Training complete" in text:
        info["running"] = False
        m = re.search(r"Best eval accuracy: ([\\d.]+)%", text)
        if m:
            info["best_acc"] = float(m.group(1))

    # Collect loss series from all Step lines (current run only)
    seen = {}
    current_max_steps = info["max_steps"]
    for line in lines:
        m = re.search(r"(?:Step\s+|\[.*?\])\s*(\d+)/(\d+)\s+\|\s+loss=([\d.]+)", line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            if total != current_max_steps:
                continue  # skip entries from other runs
            loss = float(m.group(3))
            seen[step] = loss  # last wins (dedup)
    loss_series = sorted([{"step": s, "loss": v} for s, v in seen.items()], key=lambda x: x["step"])

    # Parse eval/acc lines with step lookup
    acc_series = []
    for line in lines:
        m = re.search(r"Eval accuracy:\s+([\d.]+)%", line)
        if m:
            # Find the nearest step before this eval line (from loss_series)
            nearest = loss_series[-1]["step"] if loss_series else 0
            for p in reversed(loss_series):
                if p["step"] <= info.get("step", 0):
                    nearest = p["step"]
                    break
            # Actually find the nearest step from line position
            for p in loss_series:
                if p["loss"] >= 0:
                    nearest = p["step"]
            acc_series.append({"step": nearest, "acc": float(m.group(1))})
    info["loss_series"] = loss_series
    info["acc_series"] = acc_series

    return info


def main():
    global model_handler

    parser = argparse.ArgumentParser(description="Math Transformer API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pt)")

    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint:
        candidates = ["specialists/math/general/best.pt", "specialists/math/general/final.pt"]
        for c in candidates:
            if Path(c).exists():
                checkpoint = c
                break

    if not checkpoint or not Path(checkpoint).exists():
        print(f"[!] No checkpoint found. Starting in log-only mode (no model API).")
        print(f"    Train first: python3 train.py")
        model_handler = None
    else:
        print(f"[*] Loading model from {checkpoint}")
        model_handler = ModelHandler(checkpoint)
        print(f"[*] Model: {model_handler.name} ({model_handler.params:,} params)")
        print(f"[*] Endpoints:")
        print(f"    POST /generate  -  generate answer")
        print(f"    POST /train     -  train on [expr, answer] pairs")
        print(f"    POST /correct   -  correct a mistake")

    print(f"[*] Starting server on http://{args.host}:{args.port}")
    print()

    server = ThreadedHTTPServer((args.host, args.port), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
