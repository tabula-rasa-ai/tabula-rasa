"""Tabula Rasa — Interactive Math Demo (Hugging Face Spaces).

A Gradio web app that loads a trained Tabula Rasa specialist model and
lets users interact with it: type math expressions, get answers, see the
scratchpad, and generate random problems.

Configuration via environment variables:
    HF_MODEL_REPO: Hugging Face Hub repo ID (e.g. "tabula-rasa-ai/add-specialist")
                   Falls back to local checkpoint if not set.
    LOCAL_CKPT:    Path to local .pt checkpoint (default: auto-detect).
"""

import json
import os
import random
import sys
from pathlib import Path

# Ensure src/ is on the path for tabula_rasa imports
SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch

from tabula_rasa.config import Config
from tabula_rasa.math_parser import (
    OPS,
    evaluate,
    parse_expression,
    verify_equation,
    verify_scratchpad,
)
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

# ── Global model (loaded once) ─────────────────────────────────────

_model: MathTransformer | None = None
_tokenizer: MathTokenizer | None = None
_current_op: str = "add"
_current_ckpt_name: str = ""


def load_model() -> tuple[MathTransformer, MathTokenizer]:
    """Load the model once, cache globally."""
    global _model, _tokenizer, _current_ckpt_name

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    model_repo = os.environ.get("HF_MODEL_REPO", "")
    local_ckpt = os.environ.get("LOCAL_CKPT", "")

    if model_repo and not local_ckpt:
        # Load from Hugging Face Hub
        print(f"[*] Loading model from HF Hub: {model_repo}")
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file as st_load

            # Try safetensors first, fall back to .bin
            try:
                ckpt_path = hf_hub_download(repo_id=model_repo, filename="model.safetensors")
                state_dict = st_load(ckpt_path)
            except Exception:
                ckpt_path = hf_hub_download(repo_id=model_repo, filename="pytorch_model.bin")
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            # Load config from hub
            config_path = hf_hub_download(repo_id=model_repo, filename="config.json")
            with open(config_path) as f:
                hub_config = json.load(f)

            # Build tokenizer from config stoi/itos if available
            tok = MathTokenizer.load_hub_config(hub_config)
            cfg = Config()
            cfg.vocab_size = tok.vocab_size
            cfg.d_model = hub_config.get("d_model", 128)
            cfg.n_layers = hub_config.get("n_layers", 4)
            cfg.n_heads = hub_config.get("n_heads", 4)
            cfg.d_ff = hub_config.get("d_ff", 512)
            cfg.activation = hub_config.get("activation", "relu")
            cfg.norm_type = hub_config.get("norm_type", "rmsnorm")
            cfg.pos_encoding = hub_config.get("pos_encoding", "rope")
            cfg.use_reversed = hub_config.get("use_reversed", True)
            cfg.use_scratchpad = hub_config.get("use_scratchpad", True)

            model = MathTransformer(cfg)
            model.load_state_dict(state_dict)
            model.eval()
            _model = model
            _tokenizer = tok
            _current_ckpt_name = model_repo
            return model, tok
        except Exception as e:
            print(f"[!] HF Hub load failed: {e}")
            print("[*] Falling back to local checkpoint...")

    # Fallback: load from local checkpoint
    if local_ckpt:
        candidates = [Path(local_ckpt)]
    else:
        candidates = [
            Path("specialists/math/add/best.pt"),
            Path("specialists/math/add/final.pt"),
            Path("specialists/math/add/best_original.pt"),
            Path("specialists/math/general/best.pt"),
            Path("specialists/math/general/final.pt"),
            Path("checkpoints/best.pt"),
            Path("checkpoints/final.pt"),
        ]

    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Train first: python3 train_specialist.py add --quick\n"
            "Or set HF_MODEL_REPO to load from Hugging Face Hub."
        )

    tok_path = ckpt_path.parent / "tokenizer.json"
    if not tok_path.exists():
        # Search in all specialist dirs
        for p in Path("specialists/math").glob("*/tokenizer.json"):
            tok_path = p
            break

    print(f"[*] Loading checkpoint: {ckpt_path}")
    if tok_path.exists():
        tok = MathTokenizer.load(str(tok_path))
    else:
        tok = MathTokenizer()

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    _model = model
    _tokenizer = tok
    _current_ckpt_name = ckpt_path.name
    return model, tok


def parse_scratchpad(result: str) -> dict:
    """Parse model output into display-friendly format.

    Returns dict with keys: raw, display, scratchpad_used, is_parsed.
    """
    if "=" not in result:
        return {"raw": result, "display": result, "scratchpad_used": False, "is_parsed": False}

    expr_part, ans_part = result.split("=", 1)
    ans_clean = ans_part.replace("<EOS>", "").replace("<PAD>", "").strip()

    # Check for scratchpad format (all digits, even length)
    if ans_clean.isdigit() and len(ans_clean) >= 2 and len(ans_clean) % 2 == 0:
        # Try to parse as scratchpad
        digits = ans_clean[1::2]  # every 2nd char starting at 1
        carries = ans_clean[0::2]  # every 2nd char starting at 0
        normal = digits[::-1]  # reverse LSD-first → normal
        if carries and carries[-1] == "1":
            display = "1" + normal  # final carry
        else:
            display = normal.lstrip("0") or "0"

        # Verify with math parser
        eq_result = verify_scratchpad(f"{expr_part}={ans_clean}")
        is_valid = eq_result.get("valid", False)

        return {
            "raw": ans_clean,
            "display": display,
            "scratchpad_used": True,
            "is_valid": is_valid,
            "digits": digits,
            "carries": carries,
            "normal": normal,
            "parsed_display": display,
        }

    # Plain answer
    return {
        "raw": ans_clean,
        "display": ans_clean,
        "scratchpad_used": False,
        "is_valid": False,
    }


def solve_expression(expression: str, temperature: float = 0.0) -> dict:
    """Solve a math expression and return the result with metadata.

    Args:
        expression: Math expression (e.g. "12+34" or "12+34=").
        temperature: Sampling temperature (0 = greedy).

    Returns:
        dict with keys: expression, result, display, model_raw, confidence,
        time_ms, scratchpad_used, is_valid, error.
    """
    # Normalize: ensure trailing =
    expr = expression.strip()
    if not expr.endswith("="):
        expr += "="

    try:
        model, tok = load_model()
    except FileNotFoundError as e:
        return {"expression": expr, "error": str(e)}

    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    import time

    start = time.time()

    try:
        with torch.no_grad():
            raw = model.generate(
                tok,
                expr,
                max_new_tokens=15,
                temperature=temperature,
                top_k=0 if temperature == 0 else 5,
            )
    except Exception as e:
        return {"expression": expr, "error": f"Generation failed: {e}"}

    elapsed_ms = (time.time() - start) * 1000

    # Parse result
    parsed = parse_scratchpad(raw)

    # Compute expected answer for comparison
    expected = None
    is_correct = False
    expr_no_eq = expr.rstrip("=")
    parsed_expr = parse_expression(expr_no_eq)
    if parsed_expr is not None:
        op, a, b, _ = parsed_expr
        try:
            expected = str(evaluate(a, b, op))
        except (ValueError, ZeroDivisionError):
            expected = "error"
        is_correct = parsed.get("display", "") == expected
    else:
        # Not a parseable expression — show raw output
        pass

    return {
        "expression": expr.rstrip("="),
        "result": parsed.get("display", raw),
        "raw_scratchpad": parsed.get("raw", ""),
        "model_raw": raw,
        "expected": expected,
        "is_correct": is_correct,
        "time_ms": round(elapsed_ms, 1),
        "scratchpad_used": parsed.get("scratchpad_used", False),
    }


# ── Random Problem Generator ──────────────────────────────────────

OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def generate_random_problem(operation: str = "add", max_digits: int = 4) -> str:
    """Generate a random math problem string.

    Args:
        operation: One of "add", "sub", "mul", "div".
        max_digits: Max digits per operand.

    Returns:
        Expression string like "24+18="
    """
    op = operation.lower()
    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)

    if op == "add":
        a = random.randint(0, 10**d1 - 1) if d1 > 1 else random.randint(0, 9)
        b = random.randint(0, 10**d2 - 1) if d2 > 1 else random.randint(0, 9)
    elif op == "sub":
        b = random.randint(0, 10**d2 - 1)
        a = b + random.randint(1, 10 ** max(d1, d2) - 1)  # ensure non-negative
    elif op == "mul":
        a = random.randint(0, 10**d1 - 1)
        b = random.randint(0, 10**d2 - 1)
    elif op == "div":
        b = random.randint(1, 10**d2 - 1)
        a = b * random.randint(0, 10**d1 - 1)  # ensure divisible
    else:
        a, b = random.randint(0, 99), random.randint(0, 99)

    symbol = OP_SYMBOLS.get(op, "+")
    return f"{a}{symbol}{b}="


# ═══════════════════════════════════════════════════════════════════
# Gradio Interface
# ═══════════════════════════════════════════════════════════════════


def build_gradio_interface():
    """Build and return the Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(
        title="Tabula Rasa — Math Solver",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="gray",
        ),
        css="""
        .logo-text { font-size: 1.8em; font-weight: 700; letter-spacing: -0.02em; }
        .subtitle { color: #6b7280; margin-top: -0.5em; }
        .result-box { font-size: 1.5em; padding: 1em; border-radius: 8px; text-align: center; }
        .correct { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .incorrect { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
        .scratchpad-badge { font-size: 0.7em; background: #e0e7ff; color: #3730a3;
                           padding: 0.2em 0.6em; border-radius: 4px; margin-left: 0.5em; }
        .metadata { font-size: 0.85em; color: #6b7280; }
        """,
    ) as demo:
        # ── Header ────────────────────────────────────────────────
        gr.Markdown(
            """
            <div style="text-align: center;">
                <div class="logo-text">Tabula Rasa</div>
                <div class="subtitle">Learning from Scratch — One Specialist at a Time</div>
                <p style="margin-top: 1em;">
                    A transformer trained from a blank slate to solve arithmetic.
                    Type an expression like <code>24+18=</code> and press Enter.
                </p>
            </div>
            """
        )

        # ── Settings Row ──────────────────────────────────────────
        with gr.Row():
            operation = gr.Radio(
                choices=["add", "sub", "mul", "div"],
                value="add",
                label="Operation",
                interactive=True,
            )
            difficulty = gr.Radio(
                choices=["1 digit", "2 digits", "3 digits", "4 digits"],
                value="2 digits",
                label="Difficulty",
                interactive=True,
            )
            show_scratchpad = gr.Checkbox(
                label="Show Scratchpad",
                value=False,
                info="Display the model's internal carry-digit trace",
            )

        # ── Input ─────────────────────────────────────────────────
        with gr.Row():
            expr_input = gr.Textbox(
                label="Math Expression",
                placeholder="Type an expression like 24+18= or click 'Random Problem'",
                scale=4,
            )

        with gr.Row():
            solve_btn = gr.Button("Solve", variant="primary", scale=1)
            random_btn = gr.Button("Random Problem", variant="secondary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        # ── Output ────────────────────────────────────────────────
        result_html = gr.HTML(label="Result")

        with gr.Row():
            with gr.Column(scale=1):
                time_display = gr.Textbox(label="Time", interactive=False)
            with gr.Column(scale=1):
                accuracy_display = gr.Textbox(label="Model Checkpoint", interactive=False)

        scratchpad_display = gr.Textbox(
            label="Raw Scratchpad Trace",
            interactive=False,
            visible=False,
        )

        separator = gr.Markdown("---")

        # ── Model Info (Collapsible) ──────────────────────────────
        with gr.Accordion("Model Info", open=False):
            model_info = gr.JSON(label="Model Configuration")

        # ── Event Handlers ────────────────────────────────────────

        def _digits_to_int(d: str) -> int:
            mapping = {
                "1 digit": 1,
                "2 digits": 2,
                "3 digits": 3,
                "4 digits": 4,
            }
            return mapping.get(d, 2)

        def do_solve(expression: str) -> tuple:
            """Solve and return results."""
            if not expression or not expression.strip():
                return _empty_result()

            result = solve_expression(expression)

            if "error" in result:
                html = f'<div class="result-box incorrect">' f'  Error: {result["error"]}' f"</div>"
                return html, "", "", "", ""

            # Build result HTML
            correct_str = result.get("is_correct")
            if correct_str:
                css_class = "correct"
                status = "✓ Correct!"
            else:
                css_class = "incorrect"
                status = "✗ Incorrect"

            display_val = result.get("result", "")
            expected = result.get("expected", "")
            scratchpad_badge = (
                '<span class="scratchpad-badge">scratchpad</span>'
                if result.get("scratchpad_used")
                else ""
            )

            expected_str = f" | Expected: {expected}" if expected else ""

            html = (
                f'<div class="result-box {css_class}">'
                f'  <div style="font-size: 1.8em; font-weight: 600;">'
                f'    {result["expression"]} = {display_val} {scratchpad_badge}'
                f"  </div>"
                f'  <div style="margin-top: 0.5em;">{status}{expected_str}</div>'
                f"</div>"
            )

            time_str = f"{result['time_ms']}ms"
            accuracy_str = _current_ckpt_name if _current_ckpt_name else "local"
            raw_scratchpad = result.get("raw_scratchpad", "")
            scratchpad_raw = result.get("model_raw", "")

            return html, time_str, accuracy_str, raw_scratchpad, scratchpad_raw

        def _empty_result():
            return (
                '<div class="result-box" style="color: #9ca3af;">'
                "  Enter an expression or click Random Problem"
                "</div>",
                "",
                "",
                "",
                "",
            )

        def do_random(op: str, diff: str) -> tuple:
            """Generate a random problem and solve it."""
            nd = _digits_to_int(diff)
            problem = generate_random_problem(op, nd)
            # Return the problem and trigger solve
            return problem, *do_solve(problem)

        def do_clear() -> tuple:
            """Clear all outputs."""
            return "", *_empty_result()

        # Wire up events
        expr_input.submit(
            fn=do_solve,
            inputs=[expr_input],
            outputs=[result_html, time_display, accuracy_display, scratchpad_display],
        )

        solve_btn.click(
            fn=do_solve,
            inputs=[expr_input],
            outputs=[result_html, time_display, accuracy_display, scratchpad_display],
        )

        random_btn.click(
            fn=do_random,
            inputs=[operation, difficulty],
            outputs=[expr_input, result_html, time_display, accuracy_display, scratchpad_display],
        )

        clear_btn.click(
            fn=do_clear,
            inputs=[],
            outputs=[expr_input, result_html, time_display, accuracy_display, scratchpad_display],
        )

        # Toggle scratchpad visibility
        def toggle_scratchpad(show: bool, current: str) -> dict:
            return gr.update(visible=show)

        show_scratchpad.change(
            fn=toggle_scratchpad,
            inputs=[show_scratchpad, scratchpad_display],
            outputs=scratchpad_display,
        )

        # ── Load model info on startup ────────────────────────────
        demo.load(
            fn=lambda: _load_model_info(),
            inputs=[],
            outputs=[model_info, accuracy_display],
        )

    return demo


def _load_model_info() -> tuple:
    """Load model metadata for display."""
    try:
        model, tok = load_model()
        params = count_parameters(model)
        info = {
            "Parameters": f"{params:,}",
            "Vocabulary Size": tok.vocab_size,
            "d_model": model.config.d_model,
            "Layers": model.config.n_layers,
            "Heads": model.config.n_heads,
            "d_ff": model.config.d_ff,
            "Activation": model.config.activation,
            "Norm": model.config.norm_type,
            "Position Encoding": model.config.pos_encoding,
            "Checkpoint": _current_ckpt_name,
        }
        return info, _current_ckpt_name
    except Exception as e:
        return {"Error": str(e)}, "not loaded"


# ═══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load model on startup to warm up
    print("Tabula Rasa — HF Spaces Demo")
    print("=" * 40)
    try:
        print("Loading model...")
        model, tok = load_model()
        params = count_parameters(model)
        print(f"Model loaded: {params:,} params, {tok.vocab_size} tokens")
        print(f"Checkpoint: {_current_ckpt_name}")
    except Exception as e:
        print(f"[!] Model load failed: {e}")
        print("[*] The app will show errors until a model is available.")

    demo = build_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
