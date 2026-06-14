"""Push trained specialists to Hugging Face Hub.

Uploads trained specialist model weights, config, and model cards to
Hugging Face Hub repositories.

Usage:
    python scripts/push_to_hub.py add                    # Push addition specialist
    python scripts/push_to_hub.py all                    # Push all 4 specialists
    python scripts/push_to_hub.py add --token hf_xxx     # With custom token
    python scripts/push_to_hub.py add --private          # Private repository
    python scripts/push_to_hub.py add --best             # Use best.pt (default: final.pt)
"""

import os
import sys

# Allow imports from the src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    print("[!] PyTorch is required. Install with: pip install torch>=2.0.0")
    sys.exit(1)

# Try importing huggingface_hub (optional dependency)
try:
    from huggingface_hub import HfApi, create_repo, upload_folder

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

from tabula_rasa.config import Config

# ── Constants ──────────────────────────────────────────────────────

SPECIALISTS_DIR = Path(__file__).resolve().parent.parent / "specialists"
OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
OP_NAMES = {
    "add": "Addition",
    "sub": "Subtraction",
    "mul": "Multiplication",
    "div": "Division",
}
DEFAULT_REPO_PREFIX = "tabula-rasa-ai"
HF_BASE_URL = "https://huggingface.co"


def _load_checkpoint(op: str, use_best: bool = False) -> dict[str, Any]:
    """Load a specialist checkpoint and return its contents.

    Args:
        op: Operation name (add, sub, mul, div).
        use_best: If True, load ``best.pt``; otherwise ``final.pt``.

    Returns:
        Dictionary with keys ``model_state_dict``, ``acc``,
        ``global_step``, ``per_digit``, ``config``.

    Raises:
        FileNotFoundError: If neither checkpoint exists.
    """
    specialist_dir = SPECIALISTS_DIR / "math" / op
    if not specialist_dir.exists():
        raise FileNotFoundError(f"Specialist directory not found: {specialist_dir}")

    ckpt_name = "best.pt" if use_best else "final.pt"
    ckpt_path = specialist_dir / ckpt_name

    if not ckpt_path.exists():
        # Fall back to best.pt if final.pt doesn't exist
        fallback = specialist_dir / "best.pt"
        if fallback.exists():
            print(f"  [*] {ckpt_path.name} not found, using {fallback.name}")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {specialist_dir} " f"(tried {ckpt_name} and best.pt)"
            )

    print(f"  [*] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt


def _load_metadata(op: str) -> dict[str, Any] | None:
    """Load metadata.json from the specialist directory if it exists.

    Args:
        op: Operation name.

    Returns:
        Metadata dict, or None if not found.
    """
    meta_path = SPECIALISTS_DIR / "math" / op / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def _load_tokenizer(op: str) -> dict[str, Any] | None:
    """Load tokenizer.json from the specialist directory if it exists.

    Args:
        op: Operation name.

    Returns:
        Tokenizer dict (stoi/itos) or None.
    """
    tok_path = SPECIALISTS_DIR / "math" / op / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path, "r") as f:
            return json.load(f)
    return None


def _build_config_json(
    ckpt_config: dict[str, Any],
    metadata: dict[str, Any] | None,
    tokenizer_data: dict[str, Any] | None,
    op: str,
) -> dict[str, Any]:
    """Build a Hugging Face-compatible config.json for the specialist.

    Args:
        ckpt_config: The ``config`` dict from the checkpoint.
        metadata: Optional metadata from ``metadata.json``.
        tokenizer_data: Optional tokenizer data (stoi/itos).
        op: Operation name.

    Returns:
        Serialisable config dictionary.
    """
    # Start with the default Config() values and overlay checkpoint values
    base_cfg = Config()

    # Determine vocab size from tokenizer data or checkpoint metadata
    vocab_size = None
    if metadata and "vocab_size" in metadata:
        vocab_size = metadata["vocab_size"]
    elif ckpt_config and "vocab_size" in ckpt_config:
        vocab_size = ckpt_config["vocab_size"]
    elif tokenizer_data and "stoi" in tokenizer_data:
        vocab_size = len(tokenizer_data["stoi"])

    # Fall back to default Config's vocab_size attribute if set
    if vocab_size is None:
        vocab_size = getattr(base_cfg, "vocab_size", None) or 44  # default math vocab
    if metadata and "n_heads" in metadata:
        n_heads = metadata["n_heads"]
    else:
        n_heads = getattr(base_cfg, "n_heads", 4)
    if metadata and "d_model" in metadata:
        d_model = metadata["d_model"]
    else:
        d_model = getattr(base_cfg, "d_model", 128)
    if metadata and "n_layers" in metadata:
        n_layers = metadata["n_layers"]
    else:
        n_layers = getattr(base_cfg, "n_layers", 4)

    # Use checkpoint config values where available
    d_ff = ckpt_config.get("d_ff", getattr(base_cfg, "d_ff", 512))
    use_reversed = ckpt_config.get("use_reversed", getattr(base_cfg, "use_reversed", True))
    use_loss_masking = ckpt_config.get(
        "use_loss_masking", getattr(base_cfg, "use_loss_masking", True)
    )
    use_scratchpad = ckpt_config.get("use_scratchpad", getattr(base_cfg, "use_scratchpad", True))

    activation = (metadata or {}).get("activation", getattr(base_cfg, "activation", "relu"))
    dropout = (metadata or {}).get("config", {}).get("dropout", getattr(base_cfg, "dropout", 0.1))

    config_json = {
        "architectures": ["MathTransformer"],
        "model_type": "tabula_rasa_specialist",
        "operation": op,
        "operation_symbol": OPS[op],
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "dropout": dropout,
        "activation": activation,
        "norm_type": getattr(base_cfg, "norm_type", "rmsnorm"),
        "pos_encoding": getattr(base_cfg, "pos_encoding", "rope"),
        "max_seq_len": getattr(base_cfg, "max_seq_len", 32),
        "use_reversed": use_reversed,
        "use_loss_masking": use_loss_masking,
        "use_scratchpad": use_scratchpad,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_embeddings": getattr(base_cfg, "tie_embeddings", False),
    }
    # Add tokenizer config if available
    if tokenizer_data and "stoi" in tokenizer_data:
        config_json["tokenizer_stoi"] = tokenizer_data["stoi"]
        config_json["tokenizer_itos"] = tokenizer_data["itos"]

    return config_json


def _build_model_card(
    op: str,
    metadata: dict[str, Any] | None,
    ckpt: dict[str, Any],
    config_json: dict[str, Any],
    repo_id: str,
) -> str:
    """Build a model card README.md string.

    Args:
        op: Operation name (add, sub, mul, div).
        metadata: Optional metadata from metadata.json.
        ckpt: The full checkpoint dict.
        config_json: The built config.json dict.
        repo_id: Hugging Face Hub repository ID.

    Returns:
        Markdown content for the model card.
    """
    acc = ckpt.get("acc", 0.0)
    global_step = ckpt.get("global_step", 0)
    per_digit = ckpt.get("per_digit", {})

    op_name = OP_NAMES.get(op, op.capitalize())
    op_symbol = OPS.get(op, "?")

    # Build per-digit accuracy table
    per_digit_lines = ""
    if per_digit:
        per_digit_lines = "| Digits | Accuracy |\n|--------|----------|\n"
        for digits in sorted(per_digit.keys()):
            per_digit_lines += f"| {digits} | {per_digit[digits]:.1f}% |\n"

    trainer_link = "Tabula Rasa"

    # Build citation separately to avoid f-string brace conflicts with bibtex
    citation = (
        "@software{tabula_rasa_2024,\n"
        "  author = {{Tabula Rasa AI}},\n"
        "  title = {{Tabula Rasa}: Learning from Scratch, One Specialist at a Time},\n"
        "  year = {2024},\n"
        "  url = {https://github.com/tabula-rasa-ai/tabula-rasa}\n"
        "}"
    )

    readme = f"""---
language:
  - en
license: mit
library_name: pytorch
tags:
  - tabula-rasa
  - specialist
  - arithmetic
  - {op}
  - transformer
  - continual-learning
pipeline_tag: text-generation
datasets:
  - tabula-rasa-ai/math-{op}
metrics:
  - accuracy

model-index:
  - name: {op}-specialist
    results:
      - task:
          type: text-generation
          name: Arithmetic ({op_name})
        dataset:
          type: tabula-rasa-ai/math-{op}
          name: Math {op_name}
        metrics:
          - type: accuracy
            value: {acc}
            name: Overall Accuracy
---

# {op_name} Specialist — Tabula Rasa

This is a **specialist transformer** trained by the [Tabula Rasa](https://github.com/tabula-rasa-ai/tabula-rasa)
continual learning system to perform **{op_name.lower()}** ({op_symbol}) of multi-digit numbers.

## Model Description

| Property | Value |
|----------|-------|
| **Operation** | {op_name} ({op_symbol}) |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~1M |
| **d_model** | {config_json.get('d_model', '?')} |
| **n_layers** | {config_json.get('n_layers', '?')} |
| **n_heads** | {config_json.get('n_heads', '?')} |
| **d_ff** | {config_json.get('d_ff', '?')} |
| **Activation** | {config_json.get('activation', '?')} |
| **Norm** | {config_json.get('norm_type', '?')} |
| **Position Encoding** | {config_json.get('pos_encoding', '?')} |
| **Vocabulary Size** | {config_json.get('vocab_size', '?')} |
| **Reversed Digits** | {config_json.get('use_reversed', False)} |
| **Scratchpad** | {config_json.get('use_scratchpad', False)} |
| **Training Steps** | {global_step} |
| **Overall Accuracy** | {acc:.1f}% |

## Performance by Digit Length

{per_digit_lines if per_digit_lines else "| Digits | Accuracy |\n|--------|----------|\n| N/A | N/A |\n"}

## Usage

```python
import torch
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.config import Config

# Load config
cfg = Config()
cfg.vocab_size = {config_json.get('vocab_size', 44)}
cfg.d_model = {config_json.get('d_model', 128)}
cfg.n_layers = {config_json.get('n_layers', 4)}
cfg.n_heads = {config_json.get('n_heads', 4)}
cfg.d_ff = {config_json.get('d_ff', 512)}
cfg.activation = {config_json.get('activation', 'relu')!r}
cfg.norm_type = {config_json.get('norm_type', 'rmsnorm')!r}
cfg.pos_encoding = {config_json.get('pos_encoding', 'rope')!r}
cfg.use_reversed = {config_json.get('use_reversed', True)}
cfg.use_scratchpad = {config_json.get('use_scratchpad', True)}

# Load tokenizer
tokenizer = MathTokenizer()

# Load model
model = MathTransformer(cfg)
state = torch.hub.load_state_dict_from_url(
    '{HF_BASE_URL}/{repo_id}/resolve/main/pytorch_model.bin',
    map_location='cpu'
)
model.load_state_dict(state)
model.eval()

# Generate
input_text = '12{op_symbol}34='
input_ids = tokenizer.encode(input_text)
input_tensor = torch.tensor([input_ids])
with torch.no_grad():
    logits = model(input_tensor)[0]
    pred_id = logits[0, -1].argmax().item()
print(tokenizer.decode([pred_id]))
```

## Training Details

- **Trainer**: {trainer_link} continual learning system
- **Online EWC** (Elastic Weight Consolidation) protects previously learned skills
- **Curriculum learning**: gradually increases digit length during training
- **Loss masking** ignores prompt/PAD tokens in loss computation

## Citation

```bibtex
{citation}
```

## License

MIT
"""
    return readme


def push_specialist(
    op: str,
    token: str | None = None,
    private: bool = False,
    use_best: bool = False,
) -> str | None:
    """Push a single trained specialist to Hugging Face Hub.

    Args:
        op: Operation name (``add``, ``sub``, ``mul``, ``div``).
        token: Hugging Face API token. If ``None``, uses cached token.
        private: If ``True``, create a private repository.
        use_best: If ``True``, push ``best.pt`` instead of ``final.pt``.

    Returns:
        The repo ID on success, or ``None`` if an error occurred.
    """
    if not _HF_AVAILABLE:
        print("[!] huggingface_hub is required. Install with:\n" "    pip install huggingface_hub")
        return None

    op = op.strip().lower()
    if op not in OPS:
        print(f"[!] Unknown operation: {op!r}. Choose from: {', '.join(OPS.keys())}")
        return None

    op_name = OP_NAMES[op]
    repo_id = f"{DEFAULT_REPO_PREFIX}/{op}-specialist"
    print(f"\n  [{'='*48}]")
    print(f"  [*] Pushing {op_name} Specialist")
    print(f"  [*] Repo ID: {repo_id}")
    print(f"  [{'='*48}]")

    # ── Load checkpoint ────────────────────────────────────────
    try:
        ckpt = _load_checkpoint(op, use_best=use_best)
    except FileNotFoundError as e:
        print(f"  [!] {e}")
        return None

    # ── Load optional metadata ─────────────────────────────────
    metadata = _load_metadata(op)
    tokenizer_data = _load_tokenizer(op)

    # ── Extract values ─────────────────────────────────────────
    ckpt_config = ckpt.get("config", {})
    model_state_dict = ckpt.get("model_state_dict", ckpt)
    acc = ckpt.get("acc", 0.0)
    global_step = ckpt.get("global_step", 0)

    print(f"  [*] Steps: {global_step} | Accuracy: {acc:.1f}%")
    # Use metadata values when checkpoint config is empty (e.g. final.pt)
    n_layers_display = ckpt_config.get("n_layers") or (metadata and metadata.get("n_layers")) or "?"
    d_model_display = ckpt_config.get("d_model") or (metadata and metadata.get("d_model")) or "?"
    n_heads_display = ckpt_config.get("n_heads") or (metadata and metadata.get("n_heads")) or "?"
    d_ff_display = (
        ckpt_config.get("d_ff") or (metadata and metadata.get("config", {}).get("d_ff")) or "?"
    )
    print(
        f"  [*] Architecture: L={n_layers_display} "
        f"D={d_model_display} "
        f"H={n_heads_display} "
        f"FF={d_ff_display}"
    )

    # ── Build config.json ──────────────────────────────────────
    config_json = _build_config_json(ckpt_config, metadata, tokenizer_data, op)

    # ── Build model card ───────────────────────────────────────
    model_card = _build_model_card(op, metadata, ckpt, config_json, repo_id)

    # ── Prepare upload directory ───────────────────────────────
    upload_dir = Path(f"/tmp/tabula_rasa_push_{op}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = upload_dir / "pytorch_model.bin"
    print(f"  [*] Saving weights ({len(model_state_dict)} params)...")
    torch.save(model_state_dict, model_path)
    weight_size = model_path.stat().st_size / 1_048_576
    print(f"  [*] Weights saved: {weight_size:.1f} MB")

    # Save config
    config_path = upload_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)
    print(f"  [*] Config saved: {config_path}")

    # Save model card
    readme_path = upload_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    # Truncated preview
    card_preview = model_card.split("\n")[:5]
    print(f"  [*] Model card saved ({len(model_card)} chars)")

    # Save tokenizer if available
    if tokenizer_data:
        tok_path = upload_dir / "tokenizer.json"
        with open(tok_path, "w") as f:
            json.dump(tokenizer_data, f)
        print(f"  [*] Tokenizer saved: {tok_path}")

    # ── Upload to Hub ──────────────────────────────────────────
    api = HfApi()
    try:
        # Create the repository (no-op if it already exists)
        repo_url = create_repo(
            repo_id,
            token=token,
            private=private,
            exist_ok=True,
        )
        print(f"  [*] Repo ready: {repo_url}")

        # Upload the entire folder
        print(f"  [*] Uploading to {repo_id}...")
        upload_folder(
            repo_id=repo_id,
            folder_path=str(upload_dir),
            token=token,
            commit_message=f"Push {op} specialist (step {global_step}, acc {acc:.1f}%)",
        )
        print(f"  [*] Upload complete!")
        print(f"  [*] View at: {HF_BASE_URL}/{repo_id}")

        return repo_id

    except Exception as e:
        print(f"  [!] Upload failed: {e}")
        return None

    finally:
        # Clean up temp files
        import shutil

        shutil.rmtree(upload_dir, ignore_errors=True)
        print(f"  [*] Cleaned up temp files")


def push_all(
    token: str | None = None,
    private: bool = False,
    use_best: bool = False,
) -> list[str]:
    """Push all 4 arithmetic specialists to Hugging Face Hub.

    Args:
        token: Hugging Face API token.
        private: If ``True``, create private repositories.
        use_best: If ``True``, push ``best.pt`` instead of ``final.pt``.

    Returns:
        List of successfully pushed repo IDs.
    """
    results: list[str] = []
    for op in ["add", "sub", "mul", "div"]:
        repo_id = push_specialist(op, token=token, private=private, use_best=use_best)
        if repo_id:
            results.append(repo_id)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``operation``, ``token``, ``private``,
        and ``best`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Push trained specialists to Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Operations:\n"
            "  add    Push addition specialist\n"
            "  sub    Push subtraction specialist\n"
            "  mul    Push multiplication specialist\n"
            "  div    Push division specialist\n"
            "  all    Push all 4 specialists\n"
            "\n"
            "Examples:\n"
            "  python scripts/push_to_hub.py add\n"
            "  python scripts/push_to_hub.py all\n"
            "  python scripts/push_to_hub.py add --token hf_xxx\n"
            "  python scripts/push_to_hub.py add --private\n"
            "  python scripts/push_to_hub.py add --best\n"
        ),
    )
    parser.add_argument(
        "operation",
        nargs="?",
        default=None,
        help="Specialist operation: add, sub, mul, div, or 'all'",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (default: use cached token)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Use best.pt checkpoint instead of final.pt",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the push script."""
    args = parse_args()
    op = args.operation

    if op is None:
        print("[!] No operation specified.")
        print(__doc__)
        sys.exit(1)

    if op == "all":
        results = push_all(token=args.token, private=args.private, use_best=args.best)
        if results:
            print(f"\n  [✓] Successfully pushed {len(results)} specialist(s):")
            for r in results:
                print(f"      - {r}")
        else:
            print("\n  [!] No specialists were pushed.")
            sys.exit(1)
    else:
        repo_id = push_specialist(op, token=args.token, private=args.private, use_best=args.best)
        if repo_id is None:
            sys.exit(1)
        print(f"\n  [✓] Done: {HF_BASE_URL}/{repo_id}")


if __name__ == "__main__":
    main()
