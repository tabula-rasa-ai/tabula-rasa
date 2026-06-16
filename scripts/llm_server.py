"""Local LLM inference server using transformers.
Usage: python3 llm_server.py [port] [model_name]
Default: HuggingFaceTB/SmolLM2-360M-Instruct on port 8201
"""
import sys, json, torch, time, os
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
from transformers import AutoModelForCausalLM, AutoTokenizer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8201
MODEL = sys.argv[2] if len(sys.argv) > 2 else "HuggingFaceTB/SmolLM2-360M-Instruct"

print(f"Loading {MODEL} on CPU...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True)
model.eval()
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

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
        self._send({"status": "ok", "model": MODEL})
    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(cl))
        prompt = data.get("prompt", data.get("question", ""))
        messages = data.get("messages", [{"role": "user", "content": prompt}])
        max_tokens = int(data.get("max_tokens", 96))
        temp = float(data.get("temperature", 0.7))
        t0 = time.time()
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if hasattr(inputs, 'input_ids'):
            inputs = inputs.input_ids
        out = model.generate(inputs, max_new_tokens=max_tokens, temperature=temp, do_sample=temp > 0, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        self._send({"response": response, "time_ms": round((time.time()-t0)*1000)})

print(f"Server on http://localhost:{PORT}", flush=True)
ThreadedHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
