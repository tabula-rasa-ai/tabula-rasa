"""Launch subtraction training in background."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import os
import subprocess
import sys
import time

log_path = os.path.join(os.path.dirname(__file__), "specialists", "math", "sub", "training.log")
with open(log_path, "w") as f:
    f.write(f"Launching at {time.ctime()}\n")

proc = subprocess.Popen(
    [
        sys.executable,
        "-u",
        "train_specialist.py",
        "sub",
        "--steps",
        "50000",
        "--batch",
        "256",
        "--lr",
        "5e-4",
    ],
    cwd=os.path.dirname(__file__),
    stdout=open(log_path, "a"),
    stderr=subprocess.STDOUT,
)
print(f"PID: {proc.pid}")
