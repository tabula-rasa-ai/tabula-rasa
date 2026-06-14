"""Shared fixtures for Tabula Rasa tests."""

import os
import sys

# Add src/ to path so we can import tabula_rasa package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer


@pytest.fixture(scope="session")
def cfg():
    """Default config."""
    return Config()


@pytest.fixture(scope="session")
def tok():
    """Default tokenizer."""
    return MathTokenizer()


@pytest.fixture(scope="session")
def model(cfg, tok):
    """Small model for testing."""
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.d_model = 64
    cfg.n_layers = 2
    cfg.n_heads = 2
    cfg.d_ff = 128
    return MathTransformer(cfg)
