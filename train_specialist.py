#!/usr/bin/env python3
"""
Shim — re-exports scripts/train_specialist.py as if at project root.

All experiment scripts, auto_train, and other tooling import from
'train_specialist' directly. This makes that work by proxying
to the real location at scripts/train_specialist.py.
"""
import os
import sys
import types
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import the real module
from scripts import train_specialist as _real_module  # noqa: E402

# Expose everything the real module exposes
# Use module-level __getattr__ for Python 3.7+ dynamic proxying
_IMPORTED = False


def __getattr__(name):
    """Proxy any attribute access to the real train_specialist module."""
    return getattr(_real_module, name)


def __dir__():
    """Return all names from the real module."""
    return dir(_real_module)


# When run directly, delegate to the real script
if __name__ == "__main__":
    real_script = str(_SCRIPTS_DIR / "train_specialist.py")
    sys.argv[0] = real_script
    # Use exec so the real script's __name__ == "__main__"
    with open(real_script, encoding="utf-8") as f:
        code = compile(f.read(), real_script, "exec")
        exec(code, {"__name__": "__main__", "__file__": real_script, "__doc__": _real_module.__doc__})
