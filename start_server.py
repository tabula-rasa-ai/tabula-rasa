"""Start serve.py as a clean subprocess with proper Windows process management."""
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).parent
port = sys.argv[1] if len(sys.argv) > 1 else "8000"
checkpoint = sys.argv[2] if len(sys.argv) > 2 else "specialists/math/add/best.pt"

proc = subprocess.Popen(
    [sys.executable, "serve.py", "--port", port, "--checkpoint", checkpoint],
    cwd=str(root),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Wait for server to start
for _ in range(30):
    import socket
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", int(port)))
        s.close()
        print(f"Server started on port {port} (PID {proc.pid})")
        break
    except ConnectionRefusedError:
        time.sleep(0.5)
else:
    print("Server failed to start")
    proc.terminate()
    sys.exit(1)

# Stream output until process dies
for line in iter(proc.stdout.readline, b""):
    sys.stdout.write(line.decode("utf-8", errors="replace"))
proc.wait()
