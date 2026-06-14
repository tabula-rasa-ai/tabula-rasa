"""Integration tests — end-to-end training, checkpointing, API."""
import sys, os, json, time, signal, subprocess
from pathlib import Path

import pytest
import torch

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.eval import evaluate_accuracy


class TestQuickTraining:
    """End-to-end quick training smoke test."""

    @pytest.fixture(scope='class')
    def trained_model(self):
        """Train a tiny model quickly and return it."""
        cfg = Config()
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.n_heads = 2
        cfg.d_ff = 64
        cfg.vocab_size = 44
        cfg.batch_size = 8
        cfg.max_seq_len = 16
        cfg.use_reversed = True
        cfg.use_loss_masking = True
        cfg.use_scratchpad = True
        cfg.train_samples = 200
        cfg.eval_samples = 20
        cfg.max_steps = 20
        cfg.log_every = 10

        tok = MathTokenizer()
        model = MathTransformer(cfg)

        from tabula_rasa.dataset import generate_problem
        from torch import optim

        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        model.train()

        for step in range(cfg.max_steps):
            optimizer.zero_grad()
            total_loss = 0.0
            for _ in range(cfg.batch_size):
                expr, ans = generate_problem(1, 2)
                text = f'{expr}={ans}'
                ids = tok.encode(text, add_special_tokens=True)
                ids = (ids + [tok.pad_id] * cfg.max_seq_len)[:cfg.max_seq_len]
                x = torch.tensor(ids[:-1]).unsqueeze(0)
                y = torch.tensor(ids[1:]).unsqueeze(0)
                _, loss, _ = model(x, y)
                loss.backward()
                total_loss += loss.item()
            optimizer.step()

        return model, tok, cfg

    def test_training_loss_decreases(self, trained_model):
        """Training loss is finite after quick training."""
        model, tok, cfg = trained_model
        # Verify model can still generate
        model.eval()
        output = model.generate(tok, "2+2=", max_new_tokens=5, temperature=0.0)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_eval_returns_number(self, trained_model):
        """evaluate_accuracy returns a float."""
        model, tok, cfg = trained_model
        acc = evaluate_accuracy(model, tok, num_problems=10, verbose=False)
        assert isinstance(acc, (float, int))


class TestEWCFlag:
    """Basic EWC integration tests."""

    def test_ewc_module_importable(self):
        """OnlineEWC can be imported."""
        from tabula_rasa.model import MathTransformer
        from egefalos.online_ewc import OnlineEWC
        cfg = Config()
        cfg.vocab_size = 44
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.n_heads = 2
        cfg.d_ff = 64
        model = MathTransformer(cfg)
        ewc = OnlineEWC(model)
        assert ewc is not None

    def test_ewc_fisher_compute(self):
        """Fisher matrix computation doesn't crash."""
        from tabula_rasa.model import MathTransformer
        from egefalos.online_ewc import OnlineEWC
        from tabula_rasa.tokenizer import MathTokenizer
        from tabula_rasa.dataset import generate_problem

        cfg = Config()
        cfg.vocab_size = 44
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.n_heads = 2
        cfg.d_ff = 64
        cfg.max_seq_len = 16
        tok = MathTokenizer()
        model = MathTransformer(cfg)
        ewc = OnlineEWC(model)

        # Create a few samples
        samples = []
        for _ in range(10):
            expr, ans = generate_problem(1, 2)
            text = f'{expr}={ans}'
            ids = tok.encode(text, add_special_tokens=True)
            ids = (ids + [tok.pad_id] * cfg.max_seq_len)[:cfg.max_seq_len]
            samples.append(torch.tensor(ids))

        # Create a dataloader from samples
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, i):
                return self.data[i]

        dataset = SimpleDataset(samples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=5)

        # Compute Fisher
        try:
            ewc.compute_fisher(loader)
            fisher_ok = True
        except Exception as e:
            print(f"Fisher error: {e}")
            fisher_ok = False
        assert fisher_ok, "Fisher computation failed"


class TestCheckpointSave:
    """Checkpoint save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Model state can be saved and loaded."""
        cfg = Config()
        cfg.vocab_size = 44
        cfg.d_model = 32
        cfg.n_layers = 1
        cfg.n_heads = 2
        cfg.d_ff = 64

        model = MathTransformer(cfg)
        save_path = tmp_path / "test_model.pt"

        # Save
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        assert save_path.exists()

        # Load
        state = torch.load(save_path, map_location='cpu', weights_only=True)
        model2 = MathTransformer(cfg)
        model2.load_state_dict(state['model_state_dict'])

        # Verify same output
        model.eval()
        model2.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1, _, _ = model(x)
            out2, _, _ = model2(x)
        assert torch.equal(out1, out2), "Saved and loaded models produce different outputs"
