"""
Streamlit Dashboard for Tabula Rasa
====================================
Multi-page dashboard with system status, training monitor, specialist overview,
datasets view, and Captum-based interpretability.

Run:
    streamlit run streamlit_app.py --server.port 8050
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Page configuration (must be first Streamlit call) ────────────
st.set_page_config(
    page_title="Tabula Rasa Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme via CSS ──────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; }
    .st-emotion-cache-1y4p8pa { background-color: #1e293b; }
    h1, h2, h3, .stMarkdown { color: #f1f5f9 !important; }
    .stMetric { background: #1e293b; border-radius: 8px; padding: 8px; border: 1px solid #334155; }
    .stMetric label { color: #94a3b8 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    .stAlert { background: #1e293b; color: #f1f5f9; border: 1px solid #334155; }
    .stSelectbox label, .stSlider label { color: #f1f5f9 !important; }
    .stTextInput input { background: #1e293b; color: #f1f5f9; border: 1px solid #334155; }
    div[data-testid="stExpander"] { background: #1e293b; border: 1px solid #334155; }
    .stDataFrame { background: #1e293b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ───────────────────────────────────────────────────
MATH_SERVER = "http://localhost:8003"
ROUTER_SERVER = "http://localhost:8004"
REFRESH_INTERVAL = 3  # seconds

# ── Helpers ─────────────────────────────────────────────────────


def _health(url: str) -> dict:
    """GET /health from a server, returning status dict or error."""
    try:
        r = requests.get(f"{url}/health", timeout=5)
        if r.ok:
            return r.json()
        return {"status": "error", "http_code": r.status_code}
    except requests.ConnectionError:
        return {"status": "offline"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _fetch_json(url: str, timeout: int = 10) -> dict:
    """GET a JSON endpoint, returning parsed data or an error dict."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r.json()
        return {"_error": f"HTTP {r.status_code}"}
    except requests.ConnectionError:
        return {"_error": "connection refused"}
    except Exception as e:
        return {"_error": str(e)}


def _health_indicator(status: str) -> str:
    """Return a coloured dot + label for a health status string."""
    if status == "ok":
        return "🟢 Online"
    if status == "offline":
        return "🔴 Offline"
    return "🟡 " + status


# ── Page: Home ──────────────────────────────────────────────────


def page_home() -> None:
    st.title("🧠 Tabula Rasa — System Dashboard")
    st.markdown("Developmental AI System — overview of all running services.")

    col1, col2 = st.columns(2)

    # Math server health
    with col1:
        st.subheader("🧮 Math Model Server (port 8003)")
        mh = _health(MATH_SERVER)
        st.metric("Status", _health_indicator(mh.get("status", "unknown")))
        if mh.get("status") == "ok":
            st.metric("Model", mh.get("model", "—"))
            st.metric("Parameters", f"{mh.get('params', 0):,}")
            st.metric("Vocab Size", mh.get("vocab_size", "—"))
            st.metric("Corrections", mh.get("corrections", 0))
        else:
            st.error(f"Unreachable: {mh.get('detail', '')}")

    # Router health
    with col2:
        st.subheader("🔀 Router Server (port 8004)")
        rh = _health(ROUTER_SERVER)
        st.metric("Status", _health_indicator(rh.get("status", "unknown")))
        if rh.get("status") == "ok":
            st.metric("Skills Loaded", rh.get("skills", None) or "—")
        else:
            st.error(f"Unreachable: {rh.get('detail', '')}")

    # Model Info
    st.subheader("⚙️ Model Configuration")
    config = _fetch_json(f"{MATH_SERVER}/api/config")
    if "_error" not in config:
        cols = st.columns(4)
        with cols[0]:
            st.metric("d_model", config.get("d_model", "—"))
        with cols[1]:
            st.metric("Layers", config.get("n_layers", "—"))
        with cols[2]:
            st.metric("Heads", config.get("n_heads", "—"))
        with cols[3]:
            st.metric("d_ff", config.get("d_ff", "—"))
        cols2 = st.columns(4)
        with cols2[0]:
            st.metric("Params (est.)", f"{config.get('_param_estimate', 0):,}")
        with cols2[1]:
            st.metric("Vocab Size", config.get("_vocab_size", "—"))
        with cols2[2]:
            st.metric("Dropout", config.get("dropout", "—"))
        with cols2[3]:
            st.metric("Activation", config.get("activation", "—"))
    else:
        st.info("Model config unavailable — is the math server running?")


# ── Page: Training Monitor ──────────────────────────────────────


def _sse_stream(url: str) -> Iterator[dict]:
    """Yield SSE events from a streaming endpoint."""
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    yield json.loads(line[6:])
    except Exception:
        pass


def page_training_monitor() -> None:
    st.title("📉 Training Monitor")
    st.markdown("Real-time loss curves streamed from the math model server.")

    # Container for the chart
    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    # Data buffers
    steps: list[int] = []
    losses: list[float] = []

    # Initial data
    progress = _fetch_json(f"{MATH_SERVER}/training-progress")
    if "_error" not in progress:
        history = progress.get("history", [])
        for pt in history:
            steps.append(pt.get("step", 0))
            losses.append(pt.get("loss", 0.0))

    fig_placeholder = st.empty()

    # Auto-refresh via SSE
    st.caption(
        f"Listening to SSE stream at {MATH_SERVER}/events — auto-updates every ~3s"
    )
    if st.button("🔄 Refresh Now", type="primary"):
        progress = _fetch_json(f"{MATH_SERVER}/training-progress")
        if "_error" not in progress:
            history = progress.get("history", [])
            steps.clear()
            losses.clear()
            for pt in history:
                steps.append(pt.get("step", 0))
                losses.append(pt.get("loss", 0.0))

    # Plot the loss curve
    if steps:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.plot(steps, losses, color="#3b82f6", linewidth=2)
        ax.set_xlabel("Step", color="#94a3b8")
        ax.set_ylabel("Loss", color="#94a3b8")
        ax.set_title("Training Loss Curve", color="#f1f5f9")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, alpha=0.3, color="#334155")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        fig_placeholder.pyplot(fig)
        plt.close(fig)

        st.metric("Current Loss", f"{losses[-1]:.6f}")
        st.metric("Steps Completed", steps[-1])
    else:
        st.info("No training data available yet.")

    # SSE streaming section (manual polling mode)
    st.subheader("📡 Live Stream")
    events = st.empty()
    stream_data = _fetch_json(f"{MATH_SERVER}/training-progress")
    if "_error" not in stream_data:
        events.json(stream_data)
    else:
        events.info("Waiting for training events...")


# ── Page: Specialist Dashboard ─────────────────────────────────


def page_specialists() -> None:
    st.title("🔧 Specialist Dashboard")
    st.markdown("Overview all specialist models and their training status.")

    skills = _fetch_json(f"{ROUTER_SERVER}/skills")
    if "_error" in skills:
        st.error(f"Router unavailable: {skills['_error']}")
        return

    if isinstance(skills, list):
        df = pd.DataFrame(skills)
    elif isinstance(skills, dict) and "skills" in skills:
        df = pd.DataFrame(skills["skills"])
    else:
        st.json(skills)
        return

    if df.empty:
        st.info("No specialists found.")
        return

    # Normalize columns
    display_cols = [c for c in ["name", "status", "description", "accuracy", "dir", "ops"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

    # Per-specialist training progress
    st.subheader("📊 Training Progress Per Specialist")
    progress = _fetch_json(f"{ROUTER_SERVER}/training-progress")
    if "_error" not in progress:
        st.json(progress)
    else:
        st.caption("Per-intent progress not available.")


# ── Page: Datasets ──────────────────────────────────────────────


def page_datasets() -> None:
    st.title("🗂️ Datasets")
    st.markdown("Dataset sizes and training status per intent.")

    datasets = _fetch_json(f"{ROUTER_SERVER}/datasets")
    if "_error" in datasets:
        st.error(f"Router unavailable: {datasets['_error']}")
        return

    if isinstance(datasets, list):
        df = pd.DataFrame(datasets)
    elif isinstance(datasets, dict) and "datasets" in datasets:
        df = pd.DataFrame(datasets["datasets"])
    else:
        # Try rendering as-is
        st.json(datasets)
        return

    if df.empty:
        st.info("No dataset information available.")
        return

    st.dataframe(df, use_container_width=True)

    # Task queue stats
    st.subheader("📋 Task Queue")
    tasks = _fetch_json(f"{ROUTER_SERVER}/api/tasks")
    if "_error" not in tasks:
        st.json(tasks)
    else:
        st.caption("Task queue stats not available.")


# ── Page: Interpretability ──────────────────────────────────────


def page_interpretability() -> None:
    st.title("🔍 Interpretability")
    st.markdown(
        "Analyze model predictions using Captum — see which input tokens "
        "most influence the output."
    )

    prompt = st.text_input("Math Expression", "12+34=", help="Enter a math expression ending with '='")
    target = st.text_input("Expected output (optional)", "", help="e.g. 46")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    with col2:
        top_k = st.slider("Top-K", 1, 10, 5)
    with col3:
        max_tokens = st.slider("Max New Tokens", 1, 30, 15)

    if st.button("🚀 Analyze", type="primary"):
        if not prompt.endswith("="):
            st.warning("Prompt should end with '='")
            prompt = prompt + "="

        with st.spinner("Running model and analyzing with Captum..."):
            # 1. Generate prediction via math server
            try:
                gen_resp = requests.post(
                    f"{MATH_SERVER}/generate",
                    json={
                        "prompt": prompt,
                        "temperature": temperature,
                        "top_k": top_k,
                        "max_new_tokens": max_tokens,
                    },
                    timeout=15,
                )
                if not gen_resp.ok:
                    st.error(f"Generation failed: HTTP {gen_resp.status_code}")
                    return
                gen_data = gen_resp.json()
            except requests.ConnectionError:
                st.error("Cannot reach math server — is it running on port 8003?")
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

            st.subheader("📝 Model Output")
            st.code(
                f"Prompt: {gen_data.get('prompt', '')}\n"
                f"Result: {gen_data.get('result', '')}\n"
                f"Display: {gen_data.get('display', '')}\n"
                f"Time: {gen_data.get('time_ms', 0)}ms",
                language="text",
            )

            # 2. Try Captum analysis if we can load the model
            try:
                from egefalos.interpretability import (
                    analyze_prediction,
                    generate_html_report,
                    visualize_attention,
                )
            except ImportError as e:
                st.warning(f"Could not load interpretability module: {e}")
                st.info(
                    "Captum analysis requires a locally-loaded model checkpoint. "
                    "Ensure checkpoints exist and the module can import them."
                )
                _render_fallback_analysis(prompt, gen_data)
                return

            # Attempt to load checkpoint
            try:
                analysis = _run_captum_analysis(
                    prompt, target if target else None, analyze_prediction, visualize_attention, generate_html_report
                )
                if analysis:
                    st.subheader("🎯 Token Attribution")
                    if analysis.get("html"):
                        st.components.v1.html(analysis["html"], height=400, scrolling=True)
                    if analysis.get("attributions"):
                        st.dataframe(pd.DataFrame(analysis["attributions"]))
                    if analysis.get("attention"):
                        st.subheader("👁️ Attention Patterns")
                        st.json(analysis["attention"])
            except Exception as e:
                st.error(f"Captum analysis failed: {e}")
                st.info("Falling back to API-based analysis.")
                _render_fallback_analysis(prompt, gen_data)


def _render_fallback_analysis(prompt: str, gen_data: dict) -> None:
    """Render a minimal token-highlight analysis when Captum isn't available."""
    result = gen_data.get("result", "")
    full_text = prompt + result[len(prompt):] if result.startswith(prompt) else prompt + result
    st.subheader("🔤 Token Sequence")
    for i, ch in enumerate(full_text):
        c = ch if ch not in ("<", ">") else f"`{ch}`"
        st.markdown(
            f"<span style='background:#1e293b; padding:2px 6px; margin:2px; "
            f"border-radius:4px; font-family:monospace;'>{c}</span>",
            unsafe_allow_html=True,
        )


def _run_captum_analysis(
    prompt: str,
    target: str | None,
    analyze_prediction_fn: Any,
    visualize_attention_fn: Any,
    generate_html_report_fn: Any,
) -> dict | None:
    """Load checkpoint and run Captum analysis."""
    import torch
    from pathlib import Path

    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer
    from tabula_rasa.model import MathTransformer

    # Find best checkpoint
    base = Path(__file__).parent
    candidates = list(base.rglob("best.pt")) + list(base.rglob("finetuned.pt"))
    if not candidates:
        candidates = list(base.rglob("*.pt"))
    if not candidates:
        return None

    ckpt_path = candidates[0]
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)

    # Infer architecture from state dict
    d_model = sd["token_embedding.weight"].shape[1]
    n_layers = sum(1 for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight"))
    d_ff = sd["layers.0.feed_forward.w1.weight"].shape[0]
    n_heads = d_model // (sd.get("layers.0.attention.rope.inv_freq", torch.tensor([0.0])).shape[0] * 2 or 4)
    if n_heads == 0:
        n_heads = 4

    # Load tokenizer
    tok_dir = ckpt_path.parent
    tok_path = tok_dir / "tokenizer.json"
    if not tok_path.exists():
        tok_path = base / "checkpoints" / "tokenizer.json"
    if not tok_path.exists():
        return None

    tokenizer = MathTokenizer.load(str(tok_path))

    cfg = Config()
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.d_ff = d_ff
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = 32

    model = MathTransformer(cfg)
    model.load_state_dict(sd)
    model.eval()

    # Run Captum analysis
    analysis = analyze_prediction_fn(model, tokenizer, prompt, target_id=None)
    attn = visualize_attention_fn(model, tokenizer, prompt)
    html = generate_html_report_fn(analysis)

    # Build target token ID if target provided
    if target:
        target_ids = tokenizer.encode(target)
        tid = target_ids[0] if target_ids else None
        analysis = analyze_prediction_fn(model, tokenizer, prompt, target_id=tid)

    return {
        "attributions": analysis.get("attributions", []),
        "tokens": analysis.get("tokens", []),
        "attention": attn,
        "html": html,
    }


# ── Navigation ──────────────────────────────────────────────────


def main() -> None:
    pages = {
        "🏠 Home": page_home,
        "📉 Training Monitor": page_training_monitor,
        "🔧 Specialists": page_specialists,
        "🗂️ Datasets": page_datasets,
        "🔍 Interpretability": page_interpretability,
    }

    with st.sidebar:
        st.title("🧠 Tabula Rasa")
        st.caption("v1.3 Developmental AI")
        st.divider()
        choice = st.radio("Navigation", list(pages.keys()))
        st.divider()
        # Mini status bar
        mh = _health(MATH_SERVER)
        rh = _health(ROUTER_SERVER)
        st.markdown(
            f"**Math server:** {_health_indicator(mh.get('status', 'unknown'))}  \n"
            f"**Router:** {_health_indicator(rh.get('status', 'unknown'))}"
        )

    pages[choice]()


if __name__ == "__main__":
    main()
