"""Startup health check for Tabula Rasa AI.
Called by start_tabula_rasa.bat after starting servers."""

import sys
import time
import urllib.request


def test(name, url, timeout=5):
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        data = r.read()
        sys.stderr.write("[OK] %s - %d bytes\n" % (name, len(data)))
    except Exception as e:
        sys.stderr.write("[FAIL] %s - %s\n" % (name, e))
    sys.stderr.flush()


time.sleep(2)
test("Port 8000 /health", "http://127.0.0.1:8000/health")
test("Port 8000 /", "http://127.0.0.1:8000/")
test("Port 8002 /skills", "http://127.0.0.1:8002/skills")
test("Port 8002 /", "http://127.0.0.1:8002/")
