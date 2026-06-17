#!/usr/bin/env python3
"""Unified launcher — starts all Tabula Rasa services with new features.

Starts:
  1. RAG server (port 8300) — knowledge base retrieval
  2. Math server with /ask endpoint (port 8002) — full prepared slate
  3. API server / dashboard (port 8000) — web UI
  4. LLM server (port 8201, optional) — chat via HuggingFace model

Usage:
    python3 scripts/launcher.py                    # Full stack
    python3 scripts/launcher.py --no-rag           # Skip RAG server
    python3 scripts/launcher.py --no-llm           # Skip LLM server
    python3 scripts/launcher.py --no-dashboard     # No web UI
    python3 scripts/launcher.py --checkpoint specialists/math/add/best.pt
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _find_python():
    return sys.executable or "python3"


def _wait_for_port(port, timeout=15, name="service"):
    """Wait until a TCP port is accepting connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(("127.0.0.1", port))
            s.close()
            print(f"  ✓ {name} (port {port})")
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    print(f"  ✗ {name} (port {port}) — not ready after {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="Tabula Rasa — Unified Launcher")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to model checkpoint")
    parser.add_argument("--port-math", type=int, default=8002,
                        help="Math/AI server port")
    parser.add_argument("--port-api", type=int, default=8000,
                        help="API/Dashboard server port")
    parser.add_argument("--port-rag", type=int, default=8300,
                        help="RAG server port")
    parser.add_argument("--port-llm", type=int, default=8201,
                        help="LLM server port")
    parser.add_argument("--no-rag", action="store_true",
                        help="Skip RAG server")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM server (chat without RAG)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip web dashboard")
    parser.add_argument("--streamlit", action="store_true",
                        help="Use streamlit dashboard instead of API server")
    args = parser.parse_args()

    python = _find_python()
    procs = []

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║     Tabula Rasa — Prepared Slate Stack      ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    # ── 1. RAG Server ─────────────────────────────────────────────
    if not args.no_rag:
        print("[1/4] Starting RAG server...")
        p = subprocess.Popen(
            [python, "-m", "tabula_rasa.rag", "--port", str(args.port_rag)],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("RAG", p))
        _wait_for_port(args.port_rag, name="RAG")
    else:
        print("[1/4] RAG server: skipped")

    # ── 2. Math/AI Server ─────────────────────────────────────────
    print("[2/4] Starting Math/AI server (prepared slate)...")
    cmd = [python, "serve.py", "--port", str(args.port_math)]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    p = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(("Math", p))
    _wait_for_port(args.port_math, name="Math/AI")

    # ── 3. LLM Server (optional) ──────────────────────────────────
    if not args.no_llm:
        print("[3/4] Starting LLM server...")
        rag_url = f"http://127.0.0.1:{args.port_rag}" if not args.no_rag else None
        cmd = [python, "scripts/llm_server.py", str(args.port_llm),
               "HuggingFaceTB/SmolLM2-360M-Instruct"]
        if rag_url:
            cmd.append(rag_url)
        p = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(("LLM", p))
        _wait_for_port(args.port_llm, name="LLM")
    else:
        print("[3/4] LLM server: skipped")

    # ── 4. Dashboard / API Server ─────────────────────────────────
    if not args.no_dashboard:
        print("[4/4] Starting dashboard...")
        if args.streamlit:
            p = subprocess.Popen(
                [python, "-m", "streamlit", "run", "streamlit_app.py",
                 "--server.port", str(args.port_api)],
                cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            p = subprocess.Popen(
                [python, "scripts/api_server.py", str(args.port_api)],
                cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        procs.append(("Dashboard", p))
        _wait_for_port(args.port_api, name="Dashboard")
    else:
        print("[4/4] Dashboard: skipped")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           All systems running!              ║")
    print("  ║                                            ║")
    print(f"  ║  Math/AI      : http://localhost:{args.port_math:<4}           ║")
    print(f"  ║  Dashboard    : http://localhost:{args.port_api:<4}           ║")
    if not args.no_rag:
        print(f"  ║  RAG          : http://localhost:{args.port_rag:<4}           ║")
    if not args.no_llm:
        print(f"  ║  LLM          : http://localhost:{args.port_llm:<4}           ║")
    print("  ║                                            ║")
    print("  ║  Endpoints:                                ║")
    print(f"  ║    POST /ask   — prepared slate            ║")
    print(f"  ║    POST /generate — raw model               ║")
    print(f"  ║    POST /train  — fine-tune                 ║")
    print("  ║                                            ║")
    print("  ║  Press Ctrl+C to stop all services.        ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    # ── Wait for Ctrl+C ───────────────────────────────────────────
    try:
        while True:
            time.sleep(1)
            # Check for dead processes
            for name, p in procs:
                if p.poll() is not None:
                    print(f"  [!] {name} server died (exit code {p.returncode})")
    except KeyboardInterrupt:
        print("\n  Shutting down all services...")
    finally:
        for name, p in procs:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
            print(f"  ✓ {name} stopped")


if __name__ == "__main__":
    main()
