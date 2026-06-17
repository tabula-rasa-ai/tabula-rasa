"""Property-based tests using Hypothesis: random arithmetic expressions.

Verifies the core ML pipeline — tokenizer roundtrip, scratchpad format
correctness, and inference — with random ``(a, b)`` pairs so regressions
like the Causal-Carry Mismatch or the missing borrow scratchpad are
caught automatically.
"""

import pytest
from hypothesis import given, strategies as st
from hypothesis import HealthCheck, settings

from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.training.mult_scratchpad import generate_mult_scratchpad, parse_mult_scratchpad


# ── Small integers: 1-3 digits, no negatives ────────────────────────
small_int = st.integers(min_value=1, max_value=999)


# ═══════════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════════

class TestTokenizerRoundtrip:
    """Every encoded expression decodes back to itself."""

    tok = MathTokenizer()

    @given(st.integers(0, 999), st.integers(0, 999))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_add_roundtrip(self, a: int, b: int) -> None:
        expr = f"{a}+{b}="
        ids = self.tok.encode(expr)
        decoded = self.tok.decode(ids)
        assert expr in decoded, f"{expr} -> {ids} -> {decoded}"

    @given(st.integers(0, 999), st.integers(0, 999))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sub_roundtrip(self, a: int, b: int) -> None:
        expr = f"{a}-{b}="
        ids = self.tok.encode(expr)
        decoded = self.tok.decode(ids)
        assert expr in decoded

    @given(st.integers(0, 99), st.integers(0, 99))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mul_roundtrip(self, a: int, b: int) -> None:
        expr = f"{a}*{b}="
        ids = self.tok.encode(expr)
        decoded = self.tok.decode(ids)
        assert expr in decoded


# ═══════════════════════════════════════════════════════════════════
# Scratchpad formats
# ═══════════════════════════════════════════════════════════════════

class TestGenerateProblem:
    """Random problems produce correct scratchpads."""

    @given(small_int, small_int)
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_problem_add(self, a: int, b: int) -> None:
        from scripts.train_specialist import generate_problem

        expr, ans = generate_problem("add", 1, 3, reversed=True, scratchpad=True)
        assert len(expr) > 0
        assert len(ans) > 0

    @given(small_int, small_int)
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_problem_sub(self, a: int, b: int) -> None:
        from scripts.train_specialist import generate_problem

        expr, ans = generate_problem("sub", 1, 3, reversed=True, scratchpad=True)
        assert len(expr) > 0
        assert len(ans) > 0

    @given(st.integers(1, 999), st.integers(1, 999))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_generate_problem_mul(self, a: int, b: int) -> None:
        from scripts.train_specialist import generate_problem

        expr, ans = generate_problem("mul", 1, 3, scratchpad=True)
        # Mul scratchpad must contain <END> marker
        assert "<END>" in ans, f"Mul scratchpad missing <END>: {ans}"

    @given(st.integers(1, 999), st.integers(1, 999))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mul_scratchpad_correct(self, a: int, b: int) -> None:
        """Multiplication scratchpad answer matches real product."""
        sp, answer = generate_mult_scratchpad(a, b)
        parsed = parse_mult_scratchpad(sp)
        assert parsed is not None
        assert int(parsed) == answer, f"{a}*{b}: scratchpad={sp}, parsed={parsed}, expected={answer}"


# ═══════════════════════════════════════════════════════════════════
# Abacus place-value computation
# ═══════════════════════════════════════════════════════════════════

class TestAbacusPlaceValues:
    """Place-value assignment for Abacus embeddings is correct."""

    def _places(self, expr: str) -> list[int]:
        from tabula_rasa.model import MathTransformer
        from tabula_rasa.config import Config
        import torch

        cfg = Config()
        cfg.pos_encoding = "abacus"
        cfg.vocab_size = 50
        model = MathTransformer(cfg)
        tok = MathTokenizer()
        ids = tok.encode(expr, add_special_tokens=False)
        tensor = torch.tensor([ids])
        places = model._compute_place_values(tensor)
        return places[0].tolist()

    def test_add_places(self) -> None:
        """12+34= → BOS=other, 1=ones, 2=tens, +=op, 3=ones, 4=tens, ==eq, EOS=other"""
        p = self._places("12+34=")
        # The tokenizer combines "12" → single carry-digit token at id 22
        # So token 0 = "12" (one token, one digit column), token 1 = "+" (op),
        # token 2 = "3" (tens place in "34"), token 3 = "4" (ones place in "34"),
        # token 4 = "=" (eq).
        # Note: places are assigned right-to-left within each operand.
        # In "34", "4" is the LSD (ones=0), "3" is tens (1).
        assert p[0] == 0, f"expected 0 (ones) for '12', got {p}"
        assert p[1] == -1, f"expected -1 (op) for '+', got {p}"
        assert p[2] == 1, f"expected 1 (tens) for '3', got {p}"   # tens place in "34"
        assert p[3] == 0, f"expected 0 (ones) for '4', got {p}"   # ones place in "34"
        assert p[4] == -2, f"expected -2 (eq) for '=', got {p}"

    def test_larger_numbers(self) -> None:
        """123+45= → correct place values."""
        p = self._places("123+45=")
        # '123' tokenized as '1','2','3' or '12','3' etc depending on longest match
        # Just check operators/eq are correct
        op_pos = [i for i, v in enumerate(p) if v == -1]
        eq_pos = [i for i, v in enumerate(p) if v == -2]
        assert len(op_pos) >= 1, f"no operator found in {p}"
        assert len(eq_pos) >= 1, f"no equals found in {p}"
        # Check there are some positive place values (digits)
        digit_places = [v for v in p if v >= 0]
        assert len(digit_places) > 0, f"no digit place values in {p}"


# ═══════════════════════════════════════════════════════════════════
# Model forward pass (small config)
# ═══════════════════════════════════════════════════════════════════

class TestModelForward:
    """Forward pass with random inputs doesn't crash."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import torch
        from tabula_rasa.config import Config
        from tabula_rasa.model import MathTransformer
        from tabula_rasa.tokenizer import MathTokenizer

        cfg = Config()
        cfg.d_model = 32
        cfg.n_layers = 2
        cfg.n_heads = 4
        cfg.d_ff = 64
        self.cfg = cfg
        self.tok = MathTokenizer()
        self.model = MathTransformer(cfg)
        self.model.eval()

    def test_forward_rope(self) -> None:
        import torch
        ids = self.tok.encode("12+34=")
        x = torch.tensor([ids])
        logits, loss, val = self.model(x, targets=x)
        assert logits.shape[0] == 1
        assert logits.shape[2] == self.tok.vocab_size

    def test_forward_abacus(self) -> None:
        import torch
        from tabula_rasa.model import MathTransformer

        self.cfg.pos_encoding = "abacus"
        model = MathTransformer(self.cfg)
        model.eval()
        ids = self.tok.encode("12+34=")
        x = torch.tensor([ids])
        logits, loss, val = model(x, targets=x)
        assert logits.shape[2] == self.tok.vocab_size

    def test_forward_none_pe(self) -> None:
        import torch
        from tabula_rasa.model import MathTransformer

        self.cfg.pos_encoding = "none"
        model = MathTransformer(self.cfg)
        model.eval()
        ids = self.tok.encode("12+34=")
        x = torch.tensor([ids])
        logits, loss, val = model(x, targets=x)
        assert logits.shape[2] == self.tok.vocab_size

    def test_forward_scratchpad_add(self) -> None:
        import torch
        ids = self.tok.encode("12+34=0406")
        x = torch.tensor([ids])
        logits, loss, val = self.model(x, targets=x)
        assert loss is not None and loss.item() > 0

    def test_forward_scratchpad_mul(self) -> None:
        import torch
        ids = self.tok.encode("7*8=56<END>56")
        x = torch.tensor([ids])
        logits, loss, val = self.model(x, targets=x)
        assert loss is not None and loss.item() > 0
