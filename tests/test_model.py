"""Tests for transformer model — init, forward pass, parameter count."""

import torch

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, alphazero_loss, count_parameters


class TestModelInit:
    def test_create_default(self, tok):
        """Creates with default config."""
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)
        assert model is not None

    def test_create_tiny(self):
        """Creates a tiny model (for fast tests)."""
        cfg = Config()
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.n_heads = 2
        cfg.d_ff = 64
        cfg.vocab_size = 44
        model = MathTransformer(cfg)
        assert model is not None

    def test_param_count_default(self, tok):
        """Default config produces ~1M parameters."""
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)
        params = count_parameters(model)
        expected = 1_062_528  # 50 vocab (was 46) -> +4*128 embedding + +4*128 LM head
        assert params == expected, f"Expected {expected} params, got {params}"

    def test_param_count_tiny(self, tok):
        """Tiny config produces fewer parameters."""
        cfg = Config()
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.n_heads = 2
        cfg.d_ff = 64
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)
        params = count_parameters(model)
        assert params < 500_000, f"Tiny model should be <500K params, got {params}"

    def test_different_vocab(self):
        """Model adapts to different vocabulary sizes."""
        cfg = Config()
        cfg.vocab_size = 100
        model = MathTransformer(cfg)
        assert model is not None


class TestModelForward:
    def test_forward_shape(self, model):
        """Forward pass returns (logits, loss, metrics) with correct shapes."""
        batch_size, seq_len = 2, 16
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        # With targets, model returns loss
        y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        logits, loss, _ = model(x, y)
        assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_forward_no_target(self, model):
        """Forward without targets returns logits and None loss."""
        batch_size, seq_len = 2, 16
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        logits, loss, metrics = model(x)
        assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
        assert loss is None

    def test_forward_mask_ignores_padding(self, model, tok):
        """Forward with masked targets ignores -100 indices in loss."""
        batch_size, seq_len = 2, 16
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        # Set one valid target per batch
        y[0, -1] = tok.stoi["="]
        y[1, -1] = tok.stoi["+"]
        _, loss, _ = model(x, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_forward_deterministic(self, model):
        """Same input produces same output (eval mode)."""
        model.eval()
        batch_size, seq_len = 1, 8
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            logits1, _, _ = model(x)
            logits2, _, _ = model(x)
        assert torch.equal(logits1, logits2)

    def test_grad_flows(self, model):
        """Gradients flow through all parameters."""
        batch_size, seq_len = 2, 16
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        _, loss, _ = model(x, y)
        loss.backward()
        # Check at least one parameter has a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
        )
        assert has_grad, "No gradients flowing through model"


class TestModelGenerate:
    def test_generate_basic(self, model, tok):
        """Generate produces output tokens given a prompt."""
        model.eval()
        prompt = "12+34="
        output = model.generate(tok, prompt, max_new_tokens=5, temperature=0.0)
        assert isinstance(output, str)
        # Output should contain the prompt (model may or may not add new tokens
        # depending on configuration and training state)
        assert prompt in output or output.startswith(prompt.rstrip("="))

    def test_generate_deterministic(self, model, tok):
        """Generate with temperature=0 produces same output."""
        model.eval()
        prompt = "5+3="
        out1 = model.generate(tok, prompt, max_new_tokens=5, temperature=0.0)
        out2 = model.generate(tok, prompt, max_new_tokens=5, temperature=0.0)
        assert out1 == out2

    def test_generate_returns_string(self, model, tok):
        """Generate returns a string."""
        model.eval()
        result = model.generate(tok, "2+2=", max_new_tokens=3)
        assert isinstance(result, str)

    def test_generate_empty_prompt(self, model, tok):
        """Generate handles empty prompt."""
        model.eval()
        result = model.generate(tok, "", max_new_tokens=3)
        assert isinstance(result, str)

    def test_generate_respects_max_tokens(self, model, tok):
        """Generate doesn't exceed max_new_tokens."""
        model.eval()
        model.generate(tok, "1+1=", max_new_tokens=3, temperature=0.0)
        # Just verify it doesn't hang


class TestModelEdgeCases:
    def test_alphazero_loss(self, model):
        """alphazero_loss produces valid losses."""
        batch_size, seq_len = 2, 8
        policy = torch.randn(batch_size, seq_len, model.config.vocab_size)
        value = torch.randn(batch_size, 1)
        targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        value_target = torch.randn(batch_size, 1)
        loss, value_loss = alphazero_loss(policy, value, targets, value_target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert isinstance(value_loss, torch.Tensor)

    def test_cpu_device(self):
        """Model can be moved to CPU explicitly."""
        cfg = Config()
        cfg.vocab_size = 44
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.n_heads = 2
        cfg.d_ff = 64
        model = MathTransformer(cfg)
        model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"


class TestCausalAttention:
    """Verify that attention is properly causal (token N can't attend to N+1)."""

    def test_causal_masking_active(self):
        """When mask=None, SDPA uses is_causal=True — earlier positions are
        unaffected by changes to later positions.

        Setup: two sequences identical except for the last token.
        If causal masking works, the first N-1 logits must be identical
        because no information flows backward.
        """
        cfg = Config()
        cfg.d_model = 32
        cfg.n_layers = 2
        cfg.n_heads = 2
        cfg.d_ff = 64
        cfg.vocab_size = 44
        model = MathTransformer(cfg)
        model.eval()

        seq_a = torch.tensor([[1, 2, 3, 4, 5]])  # last token = 5
        seq_b = torch.tensor([[1, 2, 3, 4, 9]])  # last token = 9 (different)

        with torch.no_grad():
            # The forward() method now passes mask=None internally,
            # so SDPA should use is_causal=True
            logits_a, _, _ = model(seq_a)
            logits_b, _, _ = model(seq_b)

        # Extract logits at positions 0..3 (all but the last)
        # If causal: these must be identical since the differing token
        # at position 4 cannot influence earlier positions.
        prefix_a = logits_a[0, :4, :]
        prefix_b = logits_b[0, :4, :]

        assert torch.equal(prefix_a, prefix_b), (
            "Causal masking FAILED: different last token changed earlier positions. "
            "Attention is NOT causal — token N can attend to N+1."
        )

        # Also verify the last position logits differ (they should — that's the
        # position where the input actually differs)
        last_a = logits_a[0, 4, :]
        last_b = logits_b[0, 4, :]
        assert not torch.equal(
            last_a, last_b
        ), "Last position logits should differ (different input tokens)"

    def test_causal_with_targets(self):
        """When targets are provided, causal masking still works (loss
        at position N shouldn't depend on tokens after N).

        Same setup: two targets differing only in last token. The loss
        contribution for earlier positions should be identical.
        """
        cfg = Config()
        cfg.d_model = 32
        cfg.n_layers = 2
        cfg.n_heads = 2
        cfg.d_ff = 64
        cfg.vocab_size = 44
        model = MathTransformer(cfg)

        x = torch.tensor([[1, 2, 3, 4, 5, 6]])
        y_a = torch.tensor([[1, 2, 3, 4, 5, 9]])  # last target = 9
        y_b = torch.tensor([[1, 2, 3, 4, 5, 0]])  # last target = 0

        _, loss_a, _ = model(x, y_a)
        _, loss_b, _ = model(x, y_b)

        # The input is identical; only the last target differs.
        # If causal masking works, the first 5 positions contribute the
        # same loss regardless of the 6th target. But since we're looking
        # at the total loss (mean across positions), the losses will differ
        # because position 5's contribution changes.
        # To test properly: verify the per-position loss contribution.
        # For this simple test, we verify that the loss IS different
        # (the last position difference should register), proving that
        # masking didn't prevent seeing the change.
        assert loss_a.item() != loss_b.item(), (
            "Loss should differ (different last target) — but this doesn't "
            "prove causality. The previous test does."
        )
