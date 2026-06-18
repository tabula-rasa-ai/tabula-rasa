"""Property-based tests for Tabula Rasa using Hypothesis.

These tests generate random arithmetic problems and assert invariants
about tokenization, scratchpad generation, and model forward passes.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tabula_rasa.training.mult_scratchpad import generate_mult_scratchpad, parse_mult_scratchpad

# ── Pure-logic tests (no model) ──────────────────────────────────────

@given(st.integers(min_value=0, max_value=999), st.integers(min_value=0, max_value=999))
@settings(max_examples=100)
def test_mul_scratchpad_correct(a: int, b: int) -> None:
    """Multiplication scratchpad answer matches real product."""
    sp, answer = generate_mult_scratchpad(a, b)
    parsed = parse_mult_scratchpad(sp)
    assert parsed is not None
    assert int(parsed) == answer, (
        f"{a}*{b}: scratchpad={sp}, parsed={parsed}, expected={answer}"
    )


@given(st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=6))
@settings(max_examples=50)
def test_digits_encode_decode(digits):
    """Single digits survive tokenizer roundtrip."""
    from tabula_rasa.tokenizer import MathTokenizer
    tok = MathTokenizer()
    expr = "".join(str(d) for d in digits)
    ids = tok.encode(expr)
    decoded = tok.decode(ids)
    assert decoded.strip() == expr


# ── Model forward-pass tests ─────────────────────────────────────────

@pytest.mark.slow
class TestModelForward:
    """Verify model forward pass runs without error for all pos_encoding modes."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch

        from tabula_rasa.config import Config
        from tabula_rasa.tokenizer import MathTokenizer

        self.cfg = Config()
        self.cfg.use_gradient_checkpointing = False
        self.cfg.use_streambp = False
        self.device = torch.device("cpu")
        self.tok = MathTokenizer()
        self.tok.step_id = self.tok.step_id or self.tok.eos_id
        self.tok.end_id = self.tok.end_id or self.tok.eos_id

    def _make_model(self):
        from tabula_rasa.model import MathTransformer
        return MathTransformer(self.cfg).to(self.device)

    def _sample_batch(self):
        import torch
        prompts = ["12+34=", "56+78=", "90+11="]
        ids = [self.tok.encode(p) + [self.tok.eos_id] for p in prompts]
        max_len = max(len(i) for i in ids)
        padded = torch.zeros(len(ids), max_len, dtype=torch.long)
        for idx, i in enumerate(ids):
            padded[idx, :len(i)] = torch.tensor(i)
        return padded

    def test_forward_rope(self):
        self.cfg.pos_encoding = "rope"
        model = self._make_model()
        x = self._sample_batch()
        logits, loss, _ = model(x, x)
        assert logits.shape[0] == x.shape[0]
        assert loss is not None

    def test_forward_abacus(self):
        self.cfg.pos_encoding = "abacus"
        model = self._make_model()
        x = self._sample_batch()
        logits, loss, _ = model(x, x)
        assert logits.shape[0] == x.shape[0]

    def test_forward_none_pe(self):
        self.cfg.pos_encoding = "none"
        model = self._make_model()
        x = self._sample_batch()
        logits, loss, _ = model(x, x)
        assert logits.shape[0] == x.shape[0]

    def test_forward_scratchpad(self):
        import torch

        ids = self.tok.encode("12+34=0406") + [self.tok.eos_id]
        x = torch.tensor([ids])
        model = self._make_model()
        logits, loss, _ = model(x, x)
        assert logits.shape[-1] == self.cfg.vocab_size
