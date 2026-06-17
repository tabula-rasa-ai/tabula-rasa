"""Integration tests for the Tabula Rasa codebase.

These tests validate that the CLI tooling, server endpoints,
and training pipeline wire together correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class TestCLI:
    """Test that CLI entry points exist and run without crashing."""

    def test_tabula_rasa_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tabula_rasa", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_tabula_rasa_cli_serve_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tabula_rasa", "serve", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_console_script_installed(self) -> None:
        result = subprocess.run(
            ["tabula-rasa", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_train_specialist_help(self) -> None:
        """scripts/train_specialist.py --help."""
        cwd = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/train_specialist.py", "--help"],
            capture_output=True, text=True, timeout=30, cwd=str(cwd),
        )
        assert result.returncode == 0

    def test_train_specialist_quick_train(self) -> None:
        """Quick smoke test of the training loop."""
        cwd = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/train_specialist.py",
             "add", "--steps", "50", "--batch", "32"],
            capture_output=True, text=True, timeout=300, cwd=str(cwd),
        )
        assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
        assert "Accuracy" in result.stdout or "accuracy" in result.stdout
