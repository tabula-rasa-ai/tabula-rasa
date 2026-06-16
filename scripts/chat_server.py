"""Chat inference server for GPU machine.
Runs Llama 3.2 1B and exposes a simple API.
Windows router forwards chat queries here.

Usage:
    pip install torch transformers
    python3 chat_server.py
"""
import sys, json, torch, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"Loaded. Device: {model.device}")

class Handler(BaseHTTPRequestHandler):
    def _send(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(cl))
        messages = data.get("messages", [{"role": "user", "content": data.get("prompt", "")}])
        max_tokens = int(data.get("max_tokens", 128))
        temperature = float(data.get("temperature", 0.7))

        t0 = time.time()
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        elapsed = time.time() - t0

        self._send({"response": response, "time_ms": round(elapsed * 1000)})

HTTPServer(("0.0.0.0", 8200), Handler).serve_forever()
