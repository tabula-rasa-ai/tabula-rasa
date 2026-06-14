"""Tests for LoRA (Low-Rank Adaptation) module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile

import torch
import torch.nn as nn

from tabula_rasa.config import Config
from tabula_rasa.lora import (
    LoRALayer,
    apply_lora_to_model,
    load_lora_adapters,
    save_lora_adapters,
    set_lora_trainable,
    switch_lora_adapters,
)
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer


def test_lora_creation():
    """LoRA layer wraps an nn.Linear and preserves its shape."""
    base = nn.Linear(64, 128)
    lora = LoRALayer(base, rank=8, alpha=2.0)

    # Check structure
    assert lora.lora_A.shape == (64, 8), f"Expected (64, 8), got {lora.lora_A.shape}"
    assert lora.lora_B.shape == (8, 128), f"Expected (8, 128), got {lora.lora_B.shape}"
    assert lora.rank == 8
    assert lora.alpha == 2.0
    assert lora.scaling == 2.0 / 8  # alpha / rank

    # Base is frozen
    assert not base.weight.requires_grad
    assert base.bias is None or not base.bias.requires_grad

    # LoRA params are trainable
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_lora_forward():
    """LoRA layer produces a different output than the bare Linear."""
    base = nn.Linear(32, 16)
    x = torch.randn(4, 32)

    # Output without LoRA
    out_base = base(x)

    # Output with LoRA (A is random, B is zero → output should equal base)
    lora = LoRALayer(base, rank=4, alpha=1.0)
    out_lora_init = lora(x)
    assert torch.allclose(
        out_base, out_lora_init, atol=1e-6
    ), "LoRA output should equal base at init (B is zero)"

    # After tweaking B, output should differ
    with torch.no_grad():
        lora.lora_B.data.fill_(0.1)
    out_lora_tweaked = lora(x)
    assert not torch.allclose(
        out_base, out_lora_tweaked, atol=1e-4
    ), "LoRA output should differ after B is non-zero"


def test_apply_lora_to_model():
    """apply_lora_to_model wraps target linear layers in a real model."""
    cfg = Config()
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    model = MathTransformer(cfg)

    # Count original Linear layers matching target names
    original_linears = sum(
        1
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and any(t in name for t in ["wq", "wk", "wv", "wo"])
    )
    assert original_linears > 0, "Model should have attention linear layers"

    # Apply LoRA
    lora_layers = apply_lora_to_model(model, rank=4, alpha=1.0)
    assert (
        len(lora_layers) == original_linears
    ), f"Expected {original_linears} LoRA layers, got {len(lora_layers)}"

    # Check that those layers are now LoRALayer instances
    lora_count = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALayer))
    assert (
        lora_count == original_linears
    ), f"Expected {original_linears} LoRALayers in model, got {lora_count}"

    # Model should still forward
    ids = tok.encode("12+34=46", add_special_tokens=True)
    x = torch.tensor([ids])
    logits, loss, value = model(x, x)
    assert logits.shape == (
        1,
        x.shape[1],
        cfg.vocab_size,
    ), f"Expected logits shape (1, {x.shape[1]}, {cfg.vocab_size}), got {logits.shape}"


def test_set_lora_trainable():
    """set_lora_trainable freezes everything except LoRA params."""
    cfg = Config()
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    model = MathTransformer(cfg)
    apply_lora_to_model(model, rank=4, alpha=1.0)

    # Freeze all, then set LoRA trainable
    set_lora_trainable(model, trainable=True)

    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_lora_save_load():
    """LoRA adapters can be saved and loaded, preserving behaviour."""
    cfg = Config()
    cfg.dropout = 0.0  # Disable dropout for deterministic comparison
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size

    # Use deterministic seed so both models initialize identically
    torch.manual_seed(42)
    model1 = MathTransformer(cfg)
    lora_layers1 = apply_lora_to_model(model1, rank=4, alpha=1.0)
    with torch.no_grad():
        lora_layers1[0].lora_B.data.fill_(0.05)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        save_lora_adapters(lora_layers1, tmp_path)

        # Model 2: start from same seed → identical base weights
        torch.manual_seed(42)
        model2 = MathTransformer(cfg)
        lora_layers2 = load_lora_adapters(model2, tmp_path, rank=4, alpha=1.0)

        # Compare LoRA params
        for i, (l1, l2) in enumerate(zip(lora_layers1, lora_layers2)):
            assert torch.allclose(l1.lora_A, l2.lora_A, atol=1e-6), f"Layer {i}: lora_A mismatch"
            assert torch.allclose(l1.lora_B, l2.lora_B, atol=1e-6), f"Layer {i}: lora_B mismatch"

        # Forward pass should give identical results
        model1.eval()
        model2.eval()
        ids = tok.encode("12+34=46", add_special_tokens=True)
        x = torch.tensor([ids])
        with torch.no_grad():
            logits1, _, _ = model1(x, x)
            logits2, _, _ = model2(x, x)
        assert torch.allclose(logits1, logits2, atol=1e-5), "Logits mismatch after save/load cycle"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_lora_multiple_adapters():
    """Multiple LoRA adapters can be saved/loaded and swapped."""
    cfg = Config()
    cfg.dropout = 0.0  # Disable dropout for deterministic comparison
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size

    ids = tok.encode("12+34=46", add_special_tokens=True)
    x = torch.tensor([ids])

    # Create two sets of LoRA adapters with different B values on the same base
    torch.manual_seed(42)
    adapters = []
    for seed in [42, 123]:
        torch.manual_seed(42)  # Reset to get same base weights
        model = MathTransformer(cfg)
        lora = apply_lora_to_model(model, rank=4, alpha=1.0)
        torch.manual_seed(seed)
        for layer in lora:
            with torch.no_grad():
                layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1
        adapters.append((model, lora))

    # Forward pass should differ between adapters
    adapters[0][0].eval()
    adapters[1][0].eval()
    with torch.no_grad():
        out0, _, _ = adapters[0][0](x, x)
        out1, _, _ = adapters[1][0](x, x)
    assert not torch.allclose(
        out0, out1, atol=1e-4
    ), "Different adapters should produce different outputs"

    # Save then load swapping should work
    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "adapter_a.pt")
        path_b = os.path.join(tmpdir, "adapter_b.pt")
        save_lora_adapters(adapters[0][1], path_a)
        save_lora_adapters(adapters[1][1], path_b)

        # Swap adapter A into a fresh model with same base weights
        torch.manual_seed(42)
        fresh = MathTransformer(cfg)
        loaded_a = load_lora_adapters(fresh, path_a, rank=4, alpha=1.0)
        fresh.eval()
        with torch.no_grad():
            out_loaded_a, _, _ = fresh(x, x)
        assert torch.allclose(
            out_loaded_a, out0, atol=1e-5
        ), "Loaded adapter A should match original"

        # Swap adapter B in-place using switch_lora_adapters
        switch_lora_adapters(fresh, path_b)
        fresh.eval()
        with torch.no_grad():
            out_loaded_b, _, _ = fresh(x, x)
        assert torch.allclose(
            out_loaded_b, out1, atol=1e-5
        ), "Switched adapter B should match original B"
        # After switching to B, output should NOT match A
        assert not torch.allclose(
            out_loaded_b, out0, atol=1e-4
        ), "After switching to B, output should no longer match A"


def test_lora_edge_cases():
    """Edge cases: rank=1, alpha=0, invalid inputs."""
    base = nn.Linear(16, 16)

    # rank=1 (minimum valid)
    lora = LoRALayer(base, rank=1)
    assert lora.lora_A.shape == (16, 1)
    assert lora.lora_B.shape == (1, 16)

    # alpha=0 (effectively disables LoRA update)
    lora_zero = LoRALayer(base, rank=4, alpha=0.0)
    assert lora_zero.scaling == 0.0
    x = torch.randn(2, 16)
    out_zero = lora_zero(x)
    out_base = base(x)
    assert torch.allclose(
        out_zero, out_base, atol=1e-6
    ), "alpha=0 should give identical output to base"

    # Invalid base type
    try:
        LoRALayer(nn.ReLU(), rank=4)  # type: ignore[arg-type]
        assert False, "Should have raised TypeError for non-Linear base"
    except TypeError:
        pass

    # Invalid rank
    try:
        LoRALayer(base, rank=0)
        assert False, "Should have raised ValueError for rank=0"
    except ValueError:
        pass
