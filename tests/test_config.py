"""Tests for Config class — defaults, properties, edge cases."""

import torch

from tabula_rasa.config import Config


class TestConfigDefaults:
    def test_architecture_defaults(self, cfg):
        """Architecture defaults are set correctly."""
        assert cfg.d_model == 128
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4
        assert cfg.d_ff == 512
        assert cfg.max_seq_len == 256, f"Expected 256 (bumped for long context), got {cfg.max_seq_len}"
        assert cfg.dropout == 0.1

    def test_training_defaults(self, cfg):
        """Training hyperparameter defaults are set correctly."""
        assert cfg.batch_size == 128
        assert cfg.learning_rate == 0.001
        assert cfg.weight_decay == 0.01
        assert cfg.max_steps == 30000
        assert cfg.warmup_steps == 500

    def test_amp_defaults(self, cfg):
        """AMP and gradient accumulation defaults."""
        assert cfg.use_amp is False
        assert cfg.gradient_accumulation_steps == 1

    def test_data_defaults(self, cfg):
        """Data configuration defaults."""
        assert cfg.train_samples == 5_000
        assert cfg.eval_samples == 100
        assert cfg.min_digits == 1
        assert cfg.max_digits == 4
        assert cfg.force_carry_ratio == 0.5
        assert cfg.use_scratchpad is True

    def test_curriculum_defaults(self, cfg):
        """Curriculum learning defaults."""
        assert cfg.use_curriculum is True
        assert len(cfg.curriculum_phases) == 1
        assert cfg.curriculum_phases[0] == (30000, 1)

    def test_optimizer_defaults(self, cfg):
        """Optimizer defaults."""
        assert cfg.optimizer == "adamw"
        assert cfg.lr_schedule == "cosine"
        assert cfg.adam_beta1 == 0.9
        assert cfg.adam_beta2 == 0.999

    def test_tokenizer_defaults(self, cfg):
        """Tokenizer-related defaults."""
        assert cfg.pad_token == "<PAD>"
        assert cfg.bos_token == "<BOS>"
        assert cfg.eos_token == "<EOS>"
        assert cfg.unk_token == "<UNK>"

    def test_vocab_size_none(self, cfg):
        """vocab_size starts as None, set externally."""
        assert cfg.vocab_size is None


class TestConfigDevice:
    def test_device_property_returns_string(self, cfg):
        """device property returns a string."""
        device = cfg.device
        assert isinstance(device, str)
        assert device in ("cpu", "cuda")

    def test_device_matches_torch(self, cfg):
        """device property matches torch's device detection."""
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert cfg.device == expected


class TestConfigCustomization:
    def test_override_architecture(self):
        """Config fields can be overridden."""
        cfg = Config()
        cfg.d_model = 256
        cfg.n_layers = 8
        assert cfg.d_model == 256
        assert cfg.n_layers == 8

    def test_override_amp(self):
        """AMP flag can be toggled."""
        cfg = Config()
        cfg.use_amp = True
        assert cfg.use_amp is True

    def test_override_grad_accum(self):
        """Gradient accumulation can be changed."""
        cfg = Config()
        cfg.gradient_accumulation_steps = 4
        assert cfg.gradient_accumulation_steps == 4

    def test_paths_default(self, cfg):
        """Path defaults."""
        assert cfg.save_dir == "checkpoints"
        assert cfg.data_dir == "data"
