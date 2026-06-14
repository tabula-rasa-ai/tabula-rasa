#!/usr/bin/env python3
"""
HTTP server for the Tabula Rasa Dashboard.
Serves the Dashboard folder so iframes work (no file:// issues).

Usage:
    python3 serve.py            # port 8080 (default)
    python3 serve.py 9090       # custom port

Then open: http://localhost:8080/dashboard.html

Backend servers needed by each tab:
  - Math Tester  → port 8000 (run from project root: python3 serve.py)
  - Tabula Rasa  → port 8002 (run: python3 -m egefalos.tabula_rasa)
"""

import os
import socketserver
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
DIR = os.path.dirname(os.path.abspath(__file__))


class SilentHandler(SimpleHTTPRequestHandler):
    """Suppress ConnectionAbortedError tracebacks (harmless Windows TCP noise)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # browser closed tab / navigated away mid-response — harmless
            pass
        except Exception:
            # log other errors briefly
            import traceback

            traceback.print_exc()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  [*] {args[0]} {args[1]} {args[2]}\n")


class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


print()
print(f"  Tabula Rasa Dashboard")
print(f"  ─────────────────────")
print(f"  Open:  http://localhost:{PORT}/dashboard.html")
print(f"  Dir:   {DIR}")
print()
print(f"  Backend servers needed:")
print(f"    localhost:8000  — Math Tester API  (root serve.py)")
print(f"    localhost:8002  — Tabula Rasa API  (egefalos/tabula_rasa.py)")
print()
print(f"  Ctrl+C to stop")

ReuseTCPServer.allow_reuse_address = True
httpd = ReuseTCPServer(("", PORT), SilentHandler)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\n  [*] Stopped.")
    httpd.server_close()
