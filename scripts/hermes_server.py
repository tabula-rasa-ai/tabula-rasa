"""Minimal API server — serves router predictions + math via specialist."""
import sys, os, json, time, torch
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch.set_num_threads(4)

# Lazy-loaded models
_router = None
_tok = None

def load_router():
    global _router, _tok
    if _router is not None:
        return _router, _tok
    from src.tabula_rasa.bpe_tokenizer import BPETokenizer
    from egefalos.router_model import RouterModel, INTENT_NAMES
    rpath = Path("checkpoints/router")
    if (rpath / "best.pt").exists() and (rpath / "tokenizer.json").exists():
        _tok = BPETokenizer.load(str(rpath / "tokenizer.json"))
        _router = RouterModel.load(str(rpath / "best.pt"))
        _router.eval()
        print(f"  [*] Router loaded: {sum(p.numel() for p in _router.parameters()):,} params")
    return _router, _tok

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"  [*] {args[0]} {args[1]} {args[2]}\n")

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        if path == "/health":
            router, _ = load_router()
            self._json({
                "status": "ok",
                "router": router is not None,
                "server_start": time.strftime("%H:%M:%S"),
            })
        else:
            self._json({"endpoints": {
                "GET /health": "Status check",
                "POST /route": "Route prompt to intent (router)",
                "POST /generate": "Math calculation",
            }})

    def do_POST(self):
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        cl = int(self.headers.get("Content-Length", 0))
        if cl == 0:
            self._json({"error": "Empty body"}, 400)
            return
        try:
            data = json.loads(self.rfile.read(cl))
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON"}, 400)
            return

        prompt = data.get("prompt", "").strip()
        if not prompt:
            self._json({"error": "Missing prompt"}, 400)
            return

        if path == "/route":
            router, tok = load_router()
            if router is None:
                self._json({"error": "Router not trained. Run train_router.py first."}, 503)
                return
            ids = tok.encode(prompt, add_special_tokens=True)[:64]
            pad = 64 - len(ids)
            pt = torch.tensor([ids + [tok.pad_id] * pad])
            with torch.no_grad():
                logits = router(pt)
                import torch.nn.functional as F
                probs = F.softmax(logits, dim=-1)
                conf, pred = probs.max(dim=-1)
            from egefalos.router_model import INTENT_NAMES
            self._json({
                "prompt": prompt,
                "intent": INTENT_NAMES[pred.item()],
                "confidence": round(conf.item(), 3),
            })

        elif path == "/generate":
            # Fallback: try specialist_router style
            self._json({
                "prompt": prompt,
                "note": "Math specialist needs training. Use /route for router.",
            }, 200)

        else:
            self._json({"error": "Not found"}, 404)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"[*] Starting Hermes API server on port {port}")
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        server.server_close()

if __name__ == "__main__":
    main()
