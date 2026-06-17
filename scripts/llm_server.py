"""Local LLM inference server using transformers + optional RAG context injection.
Usage: python3 llm_server.py [port] [model_name] [rag_url]
Default: HuggingFaceTB/SmolLM2-360M-Instruct on port 8201
RAG: if rag_url provided, retrieves context before each generation
   POST / with "use_rag": true to enable context injection
"""
import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
from transformers import AutoModelForCausalLM, AutoTokenizer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8201
MODEL = sys.argv[2] if len(sys.argv) > 2 else "HuggingFaceTB/SmolLM2-360M-Instruct"
RAG_URL = sys.argv[3] if len(sys.argv) > 3 else None
if RAG_URL:
    print(f"[RAG] Context retrieval from {RAG_URL}", flush=True)

print(f"Loading {MODEL} on CPU...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True)
model.eval()
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

SYSTEM_PROMPT = "You are Tabula Rasa, a helpful AI assistant."

def _fetch_rag_context(query: str, top_k: int = 2) -> str | None:
    """Query the RAG server for context relevant to the user's question.
    Returns formatted context string or None on failure.
    """
    if not RAG_URL:
        return None
    try:
        payload = json.dumps({"query": query, "top_k": top_k, "format": "prompt"}).encode()
        req = urllib.request.Request(
            f"{RAG_URL}/retrieve",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode())
            context = result.get("context", "")
            return context if context else None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
        print(f"[RAG] Warning: retrieval failed: {e}", flush=True)
        return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
        except:
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self._send({"status": "ok", "model": MODEL, "rag": RAG_URL or False})

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(cl))
        prompt = data.get("prompt", data.get("question", ""))
        messages = data.get("messages", [{"role": "user", "content": prompt}])
        max_tokens = int(data.get("max_tokens", 96))
        temp = float(data.get("temperature", 0.7))
        use_rag = data.get("use_rag", False) and RAG_URL is not None

        # ── RAG context injection ───────────────────────────────────
        rag_context = None
        if use_rag and prompt:
            rag_context = _fetch_rag_context(prompt)
            if rag_context:
                # Inject as system message before user message
                messages = [
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{rag_context}"}
                ] + [m for m in messages if m.get("role") != "system"]

        t0 = time.time()
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if hasattr(inputs, 'input_ids'):
            inputs = inputs.input_ids
        out = model.generate(inputs, max_new_tokens=max_tokens, temperature=temp, do_sample=temp > 0, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        elapsed = round((time.time() - t0) * 1000)

        result = {"response": response, "time_ms": elapsed}
        if rag_context:
            result["rag_used"] = True
            result["rag_time_ms"] = elapsed  # includes RAG time in total
        self._send(result)

print(f"Server on http://localhost:{PORT}", flush=True)
ThreadedHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
