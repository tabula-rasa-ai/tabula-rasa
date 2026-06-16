"""API server — dashboard + model inference + training progress."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
LOG = Path("training.log")
DASH = Path("Dashboard")
MODEL_CACHE = [None, None, None]  # model, tok, cfg
EXPERIMENTS = Path("experiments.json")
SERVER_START = time.time()

# ─── Training Process Manager ─────────────────────────────────────
TRAINING_PROCESS = {
    "proc": None,  # subprocess.Popen
    "op": None,  # which operation is being trained
    "started_at": 0,
    "pid": None,
}


def train_start(data: dict) -> dict:
    """Launch a specialist training subprocess."""
    global TRAINING_PROCESS
    if TRAINING_PROCESS["proc"] is not None and TRAINING_PROCESS["proc"].poll() is None:
        return {
            "ok": False,
            "error": f'Training already running: {TRAINING_PROCESS["op"]} (PID {TRAINING_PROCESS["pid"]})',
        }

    op = (data or {}).get("op", "add")
    if op not in ("add", "sub", "mul", "div"):
        return {"ok": False, "error": f"Invalid operation: {op}. Use add/sub/mul/div."}

    quick = (data or {}).get("quick", False)
    resume = (data or {}).get("resume", False)
    steps = int((data or {}).get("steps", 0))
    batch = int((data or {}).get("batch", 0))
    lr = float((data or {}).get("lr", 0))

    cmd = [sys.executable, "train_specialist.py", op]
    if quick:
        cmd.append("--quick")
    if resume:
        cmd.append("--resume")
    if steps > 0:
        cmd.extend(["--steps", str(steps)])
    if batch > 0:
        cmd.extend(["--batch", str(batch)])
    if lr > 0:
        cmd.extend(["--lr", str(lr)])

    # Clear old training log so dashboard shows fresh
    import os as _os

    log = Path(f"specialists/math/{op}/training.log")
    if log.exists():
        log.write_text("")

    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    TRAINING_PROCESS = {
        "proc": proc,
        "op": op,
        "started_at": time.time(),
        "pid": proc.pid,
    }
    return {"ok": True, "pid": proc.pid, "op": op, "cmd": " ".join(cmd)}


def train_stop(data=None) -> dict:
    """Stop the running training process gracefully."""
    global TRAINING_PROCESS
    proc = TRAINING_PROCESS.get("proc")
    if proc is None or proc.poll() is not None:
        return {"ok": True, "msg": "No training running"}
    # Send SIGTERM (on Windows this will trigger the signal handler)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    TRAINING_PROCESS["proc"] = None
    return {"ok": True, "msg": f'Stopped training (PID {TRAINING_PROCESS["pid"]})'}


def train_pause(data=None) -> dict:
    """Pause is not natively supported — we stop and user can resume."""
    return train_stop(data)


def train_status(data=None) -> dict:
    """Get status of the training subprocess."""
    proc = TRAINING_PROCESS.get("proc")
    if proc is None:
        return {"running": False, "op": None, "pid": None}
    alive = proc.poll() is None
    if not alive:
        TRAINING_PROCESS["proc"] = None
    return {
        "running": alive,
        "op": TRAINING_PROCESS.get("op"),
        "pid": TRAINING_PROCESS.get("pid"),
        "exit_code": proc.returncode if not alive else None,
        "uptime": int(time.time() - TRAINING_PROCESS["started_at"]) if alive else 0,
    }


# ─── Auto-Train Process Manager ──────────────────────────────────
AUTO_TRAIN_PROCESS = {
    "proc": None,
    "started_at": 0,
    "pid": None,
    "exit_code": None,  # Persisted so first post-exit poll still reports it
}


def auto_train_start(data: dict) -> dict:
    """Launch auto_train.py as a subprocess."""
    global AUTO_TRAIN_PROCESS
    if AUTO_TRAIN_PROCESS["proc"] is not None and AUTO_TRAIN_PROCESS["proc"].poll() is None:
        return {
            "ok": False,
            "error": f'Auto-train already running (PID {AUTO_TRAIN_PROCESS["pid"]})',
        }

    target = float((data or {}).get("target", 50))
    budget = int((data or {}).get("budget", 50000))
    deep = (data or {}).get("deep", False)

    cmd = [sys.executable, "auto_train.py", "--target", str(target), "--budget", str(budget)]
    if deep:
        cmd.append("--deep")

    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    AUTO_TRAIN_PROCESS = {
        "proc": proc,
        "started_at": time.time(),
        "pid": proc.pid,
    }
    return {"ok": True, "pid": proc.pid, "target": target, "budget": budget, "cmd": " ".join(cmd)}


def auto_train_status() -> dict:
    """Check status of auto_train subprocess."""
    global AUTO_TRAIN_PROCESS
    proc = AUTO_TRAIN_PROCESS.get("proc")
    if proc is None:
        # Return persisted exit code even after proc ref is cleared
        result = {"running": False}
        if AUTO_TRAIN_PROCESS.get("exit_code") is not None:
            result["exit_code"] = AUTO_TRAIN_PROCESS["exit_code"]
        return result
    alive = proc.poll() is None
    result = {
        "running": alive,
        "pid": AUTO_TRAIN_PROCESS.get("pid"),
        "uptime": int(time.time() - AUTO_TRAIN_PROCESS["started_at"]) if alive else 0,
    }
    if not alive:
        AUTO_TRAIN_PROCESS["exit_code"] = proc.returncode  # persist before clearing
        AUTO_TRAIN_PROCESS["proc"] = None
        result["exit_code"] = proc.returncode
    # Read history file for progress
    hist_path = Path("auto_train_history.json")
    if hist_path.exists():
        try:
            result["history"] = json.loads(hist_path.read_text())
        except Exception:
            pass
    return result


# ─── CL Metrics ────────────────────────────────────────────────

def get_cl_metrics() -> dict:
    """Load CL benchmark results and return formatted metrics."""
    import torch
    path = Path("experiments/cl_benchmark_results.json")
    if not path.exists():
        return {"available": False, "message": "Run experiments/run_cl_benchmark.py first"}

    try:
        data = json.loads(path.read_text())
    except Exception:
        return {"available": False, "error": "Failed to parse CL benchmark results"}

    methods = {}
    for method_name, results in data.items():
        if not isinstance(results, dict):
            continue
        summary = results.get("summary", {})
        methods[method_name] = {
            "avg_final_accuracy": summary.get("avg_final_accuracy", 0),
            "bwt": summary.get("min_bwt", 0),
            "fwt": summary.get("avg_fwt", 0),
            "total_steps": summary.get("total_steps_to_mastery", 0),
            "time_seconds": summary.get("time_seconds", 0),
        }

    return {"available": True, "methods": methods}


# ─── Specialist Usage ──────────────────────────────────────────

def get_specialist_usage() -> dict:
    """Scan specialists directory and return utilization stats."""
    specialists_dir = Path("specialists/math")
    if not specialists_dir.exists():
        return {"available": False}

    ops = []
    for op_dir in sorted(specialists_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        best_pt = op_dir / "best.pt"
        log_file = op_dir / "training.log"
        stats = {
            "operation": op_dir.name,
            "checkpoint_exists": best_pt.exists(),
        }
        if best_pt.exists():
            stats["size_kb"] = round(best_pt.stat().st_size / 1024, 1)
        if log_file.exists():
            stats["log_size_kb"] = round(log_file.stat().st_size / 1024, 1)
            stats["last_modified"] = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(log_file.stat().st_mtime)
            )
        # Try to extract best accuracy from log
        if log_file.exists():
            try:
                text = log_file.read_text()
                for line in text.split("\n"):
                    if "Best:" in line:
                        m = re.search(r"Best:\s*([\d.]+)%", line)
                        if m:
                            stats["best_accuracy"] = float(m.group(1))
                        break
            except Exception:
                pass
        ops.append(stats)

    return {"available": True, "specialists": ops}


# ─── Batch Inference ───────────────────────────────────────────

def batch_inference(data: dict) -> dict:
    """Run inference on multiple prompts in batch.

    POST body: {"prompts": ["2+2=", "3+4=", ...], "max_tokens": 10}
    """
    prompts = (data or {}).get("prompts", [])
    max_tokens = int((data or {}).get("max_tokens", 10))

    if not prompts:
        return {"ok": False, "error": "No prompts provided"}

    if len(prompts) > 100:
        return {"ok": False, "error": "Max 100 prompts per batch"}

    model, tok, cfg = load_model()
    if model is None:
        return {"ok": False, "error": "No model loaded. Train one first."}

    model.eval()
    results = []
    for prompt in prompts:
        try:
            output = model.generate(
                tok, prompt, max_new_tokens=max_tokens, temperature=0.0
            )
            answer = output.replace(prompt, "").replace("<EOS>", "").strip()
            results.append({"prompt": prompt, "output": output, "answer": answer})
        except Exception as e:
            results.append({"prompt": prompt, "error": str(e)})

    return {"ok": True, "count": len(results), "results": results}


def get_server_status():
    return {
        "pid": os.getpid(),
        "uptime_seconds": time.time() - SERVER_START,
        "port": PORT,
        "model_loaded": MODEL_CACHE[0] is not None,
        "model_path": Path("checkpoints/best.pt").exists() if MODEL_CACHE[0] is None else True,
        "python": sys.executable,
        "script": __file__,
    }


def get_ewc_fisher() -> dict:
    """Load EWC Fisher matrices from disk and return a serializable summary.

    Scans all specialists/math/*/ directories for ewc_fisher.pt files,
    extracts parameter-group Fisher diagonal values, and groups them
    by layer for the heatmap dashboard view.
    """
    import torch

    result = {"params": {}, "task_count": 0, "gamma": 0.9, "source": ""}

    for op_dir in sorted(Path("specialists/math").glob("*/")):
        ewc_path = op_dir / "ewc_fisher.pt"
        if not ewc_path.exists():
            continue
        try:
            state = torch.load(ewc_path, map_location="cpu", weights_only=True)
            fisher = state.get("fisher", {})
            task_count = state.get("task_count", 0)
            gamma = state.get("gamma", 0.9)

            if not fisher:
                continue

            for name, tensor in fisher.items():
                vals = tensor.flatten().tolist()[:256]  # Cap at 256 values per param
                result["params"][name] = {
                    "sum": round(tensor.sum().item(), 4),
                    "mean": round(tensor.mean().item(), 6),
                    "max": round(tensor.max().item(), 4),
                    "values": [round(v, 6) for v in vals],
                    "shape": list(tensor.shape),
                }

            result["task_count"] = max(result["task_count"], task_count)
            result["gamma"] = gamma
            result["source"] = str(op_dir)
        except Exception as e:
            result["error"] = str(e)

    return result


def run_ewc_stress_test() -> dict:
    """Run an EWC stress test: temporarily disable EWC (lambda=0),
    train a few quick batches, report the forgetting effect,
    then restore the original EWC state.

    Returns before/after Fisher statistics for the dashboard heatmap.
    """
    import torch

    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer, count_parameters
    from tabula_rasa.tokenizer import MathTokenizer

    result = {"status": "ok", "before": {}, "after": {}, "delta": {}}

    ewc_files = list(Path("specialists/math").glob("*/ewc_fisher.pt"))
    if not ewc_files:
        return {"status": "error", "message": "No EWC data found. Train with --ewc first."}

    ewc_path = ewc_files[0]
    op_dir = ewc_path.parent
    op = op_dir.name
    ckpt = op_dir / "best.pt"
    if not ckpt.exists():
        ckpt = op_dir / "final.pt"
    if not ckpt.exists():
        return {"status": "error", "message": f"No checkpoint in {op_dir}"}

    tok_path = op_dir / "tokenizer.json"
    if not tok_path.exists():
        return {"status": "error", "message": f"No tokenizer in {op_dir}"}

    try:
        from egefalos.online_ewc import OnlineEWC

        tok = MathTokenizer.load(str(tok_path))
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        model.train()

        ewc = OnlineEWC(model)
        ewc.load(str(ewc_path))

        before_fisher = {}
        for name, tensor in ewc.fisher_dict.items():
            before_fisher[name] = {
                "mean": round(tensor.mean().item(), 6),
                "sum": round(tensor.sum().item(), 4),
            }

        other_ops = [o for o in ["add", "sub", "mul", "div"] if o != op]
        if not other_ops:
            return {"status": "error", "message": "Need at least one other operation"}
        new_op = other_ops[0]
        from torch.utils.data import DataLoader

        from train_specialist import SpecialistDataset

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_ds = SpecialistDataset(tok, new_op, cfg)
        loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)

        stress_steps = min(10, len(loader))
        losses = []
        for i, batch in enumerate(loader):
            if i >= stress_steps:
                break
            x, y = batch
            x, y = x.to("cpu"), y.to("cpu")
            optimizer.zero_grad()
            _, loss, _ = model(x, y)
            loss.backward()
            optimizer.step()
            losses.append(round(loss.item(), 4))

        after_fisher = {}
        for name, tensor in ewc.fisher_dict.items():
            after_fisher[name] = {
                "mean": round(tensor.mean().item(), 6),
                "sum": round(tensor.sum().item(), 4),
            }

        delta_fisher = {}
        for name in before_fisher:
            bm = before_fisher[name]["mean"]
            am = after_fisher.get(name, {}).get("mean", 0)
            delta_fisher[name] = round(abs(am - bm) / max(bm, 1e-10) * 100, 1)

        result["before"] = before_fisher
        result["after"] = after_fisher
        result["delta"] = delta_fisher
        result["losses"] = losses
        result["op"] = op
        result["new_op"] = new_op
        result["steps"] = stress_steps
        result["avg_loss"] = round(sum(losses) / len(losses), 4) if losses else 0
        result["mean_delta_pct"] = round(sum(delta_fisher.values()) / max(1, len(delta_fisher)), 1)

    except Exception as e:
        return {"status": "error", "message": str(e)}

    return result


# ─── Editable config fields (name → (line_regex, type_fn)) ──────────
# Note: type hints like "d_model: int = 128" require (?::\s*\w+)? in the regex
CONFIG_FIELDS = {
    # Architecture
    "d_model": (r"^    d_model(?::\s*\w+)?\s*=", int),
    "n_layers": (r"^    n_layers(?::\s*\w+)?\s*=", int),
    "n_heads": (r"^    n_heads(?::\s*\w+)?\s*=", int),
    "d_ff": (r"^    d_ff(?::\s*\w+)?\s*=", int),
    "max_seq_len": (r"^    max_seq_len(?::\s*\w+)?\s*=", int),
    "dropout": (r"^    dropout(?::\s*\w+)?\s*=", float),
    "activation": (r"^    activation(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "norm_type": (r"^    norm_type(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "pos_encoding": (r"^    pos_encoding(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "weight_init": (r"^    weight_init(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "init_std": (r"^    init_std(?::\s*\w+)?\s*=", float),
    # Training
    "batch_size": (r"^    batch_size(?::\s*\w+)?\s*=", int),
    "learning_rate": (r"^    learning_rate(?::\s*\w+)?\s*=", float),
    "weight_decay": (r"^    weight_decay(?::\s*\w+)?\s*=", float),
    "warmup_steps": (r"^    warmup_steps(?::\s*\w+)?\s*=", int),
    "max_steps": (r"^    max_steps(?::\s*\w+)?\s*=", int),
    "optimizer": (r"^    optimizer(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "lr_schedule": (r"^    lr_schedule(?::\s*\w+)?\s*=", lambda s: s.strip().strip("'")),
    "adam_beta1": (r"^    adam_beta1(?::\s*\w+)?\s*=", float),
    "adam_beta2": (r"^    adam_beta2(?::\s*\w+)?\s*=", float),
    "grad_clip_norm": (r"^    grad_clip_norm(?::\s*\w+)?\s*=", float),
    "label_smoothing": (r"^    label_smoothing(?::\s*\w+)?\s*=", float),
    "use_reversed": (r"^    use_reversed(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "use_loss_masking": (r"^    use_loss_masking(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "use_hard_negative": (r"^    use_hard_negative(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "tie_embeddings": (r"^    tie_embeddings(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
    "eval_temperature": (r"^    eval_temperature(?::\s*\w+)?\s*=", float),
    "eval_max_tokens": (r"^    eval_max_tokens(?::\s*\w+)?\s*=", int),
    # Data
    "train_samples": (r"^    train_samples(?::\s*\w+)?\s*=", lambda s: int(s.replace("_", ""))),
    "eval_samples": (r"^    eval_samples(?::\s*\w+)?\s*=", lambda s: int(s.replace("_", ""))),
    "min_digits": (r"^    min_digits(?::\s*\w+)?\s*=", int),
    "max_digits": (r"^    max_digits(?::\s*\w+)?\s*=", int),
    "use_curriculum": (r"^    use_curriculum(?::\s*\w+)?\s*=", lambda s: s.strip() == "True"),
}


def _config_path() -> Path:
    """Resolve config.py location (moved to src/tabula_rasa/ in refactor)."""
    p = Path("src/tabula_rasa/config.py")
    if p.exists():
        return p
    return Path("config.py")


def get_config_dict():
    """Read current config values from config.py."""
    text = _config_path().read_text(encoding="utf-8")
    vals = {}
    for name, (pat, caster) in CONFIG_FIELDS.items():
        m = re.search(pat + r"\s*([^#\n]+)", text, re.MULTILINE)
        if m:
            try:
                vals[name] = caster(m.group(1).strip())
            except (ValueError, TypeError):
                vals[name] = None
    # Compute param estimate
    tok_vocab = 24
    embed = tok_vocab * vals.get("d_model", 0)
    lm_head = vals.get("d_model", 0) * tok_vocab
    d_model = vals.get("d_model", 0)
    n_layers = vals.get("n_layers", 0)
    d_ff = vals.get("d_ff", 0)
    n_heads = vals.get("n_heads", 1)
    head_dim = d_model // n_heads if n_heads else 0
    attn_per = 4 * d_model * d_model  # wq, wk, wv, wo
    ffn_per = 3 * d_model * d_ff  # w1, w2, w3
    total = embed + lm_head + n_layers * (attn_per + ffn_per)
    vals["_param_estimate"] = total
    vals["_vocab_size"] = tok_vocab
    return vals


def save_config(values: dict) -> list[str]:
    """Update config.py with new values. Returns list of changes made."""
    path = _config_path()
    text = path.read_text()
    changes = []
    for name, value in values.items():
        if name.startswith("_"):
            continue
        entry = CONFIG_FIELDS.get(name)
        if not entry:
            continue
        pat, caster = entry
        # Find the line and replace it
        m = re.search(pat + r"\s*([^#\n]+)", text, re.MULTILINE)
        if m:
            old_val = m.group(1).strip()
            # Format the new value
            if isinstance(value, bool):
                new_str = "True" if value else "False"
            elif isinstance(value, float):
                new_str = f"{value:g}"
            elif isinstance(value, str) and name in (
                "optimizer",
                "lr_schedule",
                "activation",
                "norm_type",
                "pos_encoding",
                "weight_init",
            ):
                new_str = f"'{value}'"
            else:
                new_str = str(value)
            if name == "train_samples" or name == "eval_samples":
                new_str = f"{value:_}" if isinstance(value, int) and value >= 10000 else str(value)
            # Replace using the actual matched text, not regex pattern
            full_line = m.group(0)
            eq_idx = full_line.index("=")
            line_prefix = full_line[: eq_idx + 1]
            old_line = full_line
            new_line = line_prefix + " " + new_str
            text = text.replace(old_line, new_line, 1)
            changes.append(f"{name}: {old_val} → {new_str}")
    path.write_text(text, encoding="utf-8")
    return changes


def get_progress():
    info = {
        "running": False,
        "step": 0,
        "max_steps": 50000,
        "loss": None,
        "best_acc": None,
        "latest_acc": None,
        "steps_per_sec": None,
        "eta_seconds": None,
        "loss_series": [],
        "acc_series": [],
    }
    if not LOG.exists():
        return info
    text = LOG.read_text()
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return info
    for line in reversed(lines):
        m = re.search(r"(?:Step\s+|\[.*?\])\s*(\d+)/(\d+)\s+\|\s+loss=([\d.]+)", line)
        if m:
            info["step"] = int(m.group(1))
            info["max_steps"] = int(m.group(2))
            info["loss"] = float(m.group(3))
            info["running"] = True
            m2 = re.search(r"\|\s+([\d.]+)\s+(?:st|steps)/s", line)
            if m2:
                info["steps_per_sec"] = float(m2.group(1))
                r = info["max_steps"] - info["step"]
                if info["steps_per_sec"] > 0:
                    info["eta_seconds"] = int(r / info["steps_per_sec"])
            break
    for line in lines:
        m = re.search(r"(?:Eval accuracy|Acc):\s+([\d.]+)%\s+\(best:\s+([\d.]+)%", line)
        if m:
            info["latest_acc"] = float(m.group(1))
            info["best_acc"] = float(m.group(2))
    info["loss_series"] = []
    seen = {}
    for line in lines:
        m = re.search(r"(?:Step\s+|\[.*?\])\s*(\d+)/(\d+)\s+\|\s+loss=([\d.]+)", line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            if total != info["max_steps"]:
                continue  # skip entries from other runs
            loss = float(m.group(3))
            seen[step] = loss
    for s in sorted(seen):
        info["loss_series"].append({"step": s, "loss": seen[s]})
    return info


def load_model():
    if MODEL_CACHE[0] is not None:
        return MODEL_CACHE
    import torch

    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer

    # Prefer specialized checkpoints, then general, then periodic checkpoints
    candidates = [
        "specialists/math/add/best.pt",
        "specialists/math/sub/best.pt",
        "specialists/math/mul/best.pt",
        "specialists/math/div/best.pt",
        "specialists/math/general/best.pt",
        "specialists/math/general/final.pt",
    ]
    # Also look for periodic checkpoint files (*.pt in specialist dirs)
    for d in sorted(Path("specialists").rglob("checkpoint_*.pt")):
        candidates.append(str(d))
    ckpt = None
    for c in candidates:
        if Path(c).exists():
            ckpt = c
            break
    if not ckpt:
        return None

    tok = MathTokenizer()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    # Try loading; if shapes mismatch, fall back to dynamic config
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        d = sd["token_embedding.weight"].shape[1]
        L = len([k for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight")])
        ff = sd["layers.0.feed_forward.w1.weight"].shape[0] if L > 0 else 256
        h = {128: 4, 64: 2, 256: 8}.get(d, max(1, d // 32))
        cfg.d_model = d
        cfg.n_layers = L
        cfg.d_ff = ff
        cfg.n_heads = h
        model = MathTransformer(cfg)
        model.load_state_dict(sd, strict=False)

    model.eval()
    MODEL_CACHE[:] = [model, tok, cfg]
    return MODEL_CACHE


def list_available_checkpoints() -> list[dict]:
    """Scan specialists/math/*/ for available checkpoint files."""
    import torch

    base = Path("specialists") / "math"
    results = []
    if not base.exists():
        return results
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        for ckpt_file in ["best.pt", "final.pt"]:
            p = d / ckpt_file
            if p.exists():
                try:
                    state = torch.load(str(p), map_location="cpu", weights_only=True)
                    sd = state["model_state_dict"]
                    d_model = sd["token_embedding.weight"].shape[1]
                    n_layers = len(
                        [
                            k
                            for k in sd
                            if k.startswith("layers.") and k.endswith(".attention.wq.weight")
                        ]
                    )
                    params = sum(v.numel() for v in sd.values())
                    results.append(
                        {
                            "name": d.name,
                            "file": ckpt_file,
                            "path": str(p),
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "params": params,
                            "acc": state.get("acc", state.get("accuracy", None)),
                            "step": state.get("step", state.get("global_step", None)),
                            "size_mb": p.stat().st_size / 1e6,
                        }
                    )
                except Exception as e:
                    results.append(
                        {"name": d.name, "file": ckpt_file, "path": str(p), "error": str(e)}
                    )
                break  # Only best.pt per dir (prefer over final.pt)
    return results


def get_training_status():
    """Check if a training process is currently running and return its status."""
    import re
    import subprocess

    info = {
        "running": False,
        "operation": None,
        "step": 0,
        "max_steps": 0,
        "loss": None,
        "config": {},
        "pid": None,
        "uptime_seconds": 0,
    }

    # Check for running training — look for training.log that exists (even if empty)
    log_paths = [
        Path("specialists/math/add/training.log"),
        Path("specialists/math/sub/training.log"),
        Path("specialists/math/mul/training.log"),
        Path("specialists/math/div/training.log"),
    ]
    try:
        for lp in log_paths:
            if lp.exists():
                age = time.time() - lp.stat().st_mtime
                if age < 600:  # Updated in last 10 min
                    info["running"] = True
    except:
        pass

    # Read training.log for latest progress
    for lp in log_paths:
        if lp.exists():
            age = time.time() - lp.stat().st_mtime
            if age < 600:  # Updated in last 10 min
                text = lp.read_text()
                lines = [l for l in text.strip().split("\n") if l.strip()]
                if lines:
                    last = lines[-1]
                    m = re.search(r"Step\s+(\d+)/(\d+)\s+\|\s+loss=([\d.]+)", last)
                    if m:
                        info["running"] = True
                        info["step"] = int(m.group(1))
                        info["max_steps"] = int(m.group(2))
                        info["loss"] = float(m.group(3))
                        info["operation"] = (
                            lp.parent.parent.name
                            if lp.parent.parent.name != "math"
                            else lp.parent.name
                        )
                        # Get config from config.py
                        try:
                            cfg_text = _config_path().read_text()
                            for key, (pat, _) in CONFIG_FIELDS.items():
                                m2 = re.search(pat + r"\s*([^#\n]+)", cfg_text, re.MULTILINE)
                                if m2:
                                    info["config"][key] = m2.group(1).strip()
                        except:
                            pass
                        info["last_update_seconds_ago"] = int(age)
                        break
    return info


def get_experiments():
    """Read all logged experiments."""
    if not EXPERIMENTS.exists():
        return []
    try:
        data = json.loads(EXPERIMENTS.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception):
        return []


def save_experiment(entry: dict):
    """Append an experiment entry to the log."""
    exps = get_experiments()
    entry["id"] = len(exps) + 1
    entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    exps.append(entry)
    EXPERIMENTS.write_text(json.dumps(exps, indent=2))
    return entry


REGISTRY = Path("registry.json")
EXPORTS_DIR = Path("exports")


def get_registry():
    """List all shared/exported checkpoints."""
    if not REGISTRY.exists():
        return []
    try:
        return json.loads(REGISTRY.read_text())
    except:
        return []


def _save_registry(entries):
    REGISTRY.write_text(json.dumps(entries, indent=2))


def export_checkpoint(data):
    """Package a checkpoint for sharing."""
    import io
    import shutil
    import zipfile

    op = (data or {}).get("operation", "add")
    experiment_id = (data or {}).get("experiment_id", None)

    # Find the checkpoint
    ckpt_paths = [
        Path(f"specialists/math/{op}/best.pt"),
        Path(f"specialists/math/{op}/final.pt"),
    ]
    ckpt = None
    for p in ckpt_paths:
        if p.exists():
            ckpt = p
            break
    if not ckpt:
        return {"error": f"No checkpoint for {op}", "ok": False}

    # Load checkpoint metadata
    try:
        import torch

        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        sd = state["model_state_dict"]
        d_model = sd["token_embedding.weight"].shape[1]
        n_layers = len(
            [k for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight")]
        )
        params = sum(v.numel() for v in sd.values())
        acc = state.get("acc", state.get("accuracy", None))
    except:
        d_model = 128
        n_layers = 4
        params = 0
        acc = None

    # Find matching experiment
    exps = get_experiments()
    exp_data = None
    if experiment_id:
        for e in exps:
            if e.get("id") == experiment_id:
                exp_data = e
                break
    if not exp_data:
        exp_data = {"operation": op, "best_accuracy": acc, "n_layers": n_layers, "d_model": d_model}

    # Create export package
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    export_name = f"{op}_L{n_layers}_d{d_model}_{ts}"
    export_dir = EXPORTS_DIR / export_name
    export_dir.mkdir(exist_ok=True)

    # Copy checkpoint
    shutil.copy2(str(ckpt), str(export_dir / "model.pt"))

    # Save metadata
    meta = {
        "name": export_name,
        "operation": op,
        "accuracy": acc or exp_data.get("best_accuracy"),
        "n_layers": n_layers,
        "d_model": d_model,
        "total_params": params,
        "vocab_size": exp_data.get("vocab_size", 44),
        "activation": exp_data.get("activation", "swiglu"),
        "use_reversed": exp_data.get("use_reversed", True),
        "use_loss_masking": exp_data.get("use_loss_masking", True),
        "timestamp": ts,
        "config": {
            k: exp_data.get(k)
            for k in [
                "dropout",
                "norm_type",
                "pos_encoding",
                "optimizer",
                "lr_schedule",
                "learning_rate",
                "batch_size",
                "max_seq_len",
                "max_digits",
            ]
            if exp_data.get(k)
        },
        "files": {
            "checkpoint": "model.pt",
            "metadata": "metadata.json",
        },
    }
    (export_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Create zip
    zip_path = EXPORTS_DIR / f"{export_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(export_dir / "model.pt"), "model.pt")
        zf.write(str(export_dir / "metadata.json"), "metadata.json")
        # Also copy tokenizer if exists
        tok_path = Path(f"specialists/math/{op}/tokenizer.json")
        if tok_path.exists():
            zf.write(str(tok_path), "tokenizer.json")
            meta["files"]["tokenizer"] = "tokenizer.json"

    # Update registry
    registry = get_registry()
    entry = {
        "id": len(registry) + 1,
        "name": export_name,
        "operation": op,
        "accuracy": meta["accuracy"],
        "n_layers": n_layers,
        "d_model": d_model,
        "params": params,
        "zip_path": str(zip_path),
        "zip_size_kb": zip_path.stat().st_size // 1024,
        "timestamp": ts,
    }
    registry.append(entry)
    _save_registry(registry)

    return {"ok": True, "entry": entry, "export_path": str(zip_path)}


def import_checkpoint(data):
    """Import a shared checkpoint from a zip file path or URL."""
    import shutil
    import urllib.request
    import zipfile

    source = (data or {}).get("source", "")
    url = (data or {}).get("url", "")

    if url:
        # Download from URL
        import tempfile

        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp_path = tmp.name
            tmp.close()
            urllib.request.urlretrieve(url, tmp_path)
            source = tmp_path
        except Exception as e:
            return {"error": f"Download failed: {e}", "ok": False}

    if not source or not Path(source).exists():
        return {"error": "No source file", "ok": False}

    # Extract
    try:
        with zipfile.ZipFile(source, "r") as zf:
            # Read metadata
            if "metadata.json" not in zf.namelist():
                return {"error": "Missing metadata.json", "ok": False}
            meta = json.loads(zf.read("metadata.json"))

            op = meta.get("operation", "add")
            import_dir = Path(f"specialists/math/{op}")
            import_dir.mkdir(parents=True, exist_ok=True)

            # Extract files
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                target = import_dir / name
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())

            # Rename model.pt → best.pt for compatibility
            if (import_dir / "model.pt").exists():
                shutil.copy2(str(import_dir / "model.pt"), str(import_dir / "best.pt"))

            # Add to experiments
            exp_entry = {
                "operation": op,
                "n_layers": meta.get("n_layers", 4),
                "d_model": meta.get("d_model", 128),
                "best_accuracy": meta.get("accuracy"),
                "total_params": meta.get("total_params", 0),
                "vocab_size": meta.get("vocab_size", 44),
                "activation": meta.get("activation", "swiglu"),
                "use_reversed": meta.get("use_reversed", True),
                "use_loss_masking": meta.get("use_loss_masking", True),
                "notes": f"Imported from: {url or source}",
            }
            saved = save_experiment(exp_entry)

            # Invalidate model cache so new checkpoint can be loaded
            MODEL_CACHE[:] = [None, None, None]

            return {
                "ok": True,
                "experiment": saved,
                "operation": op,
                "accuracy": meta.get("accuracy"),
            }
    except Exception as e:
        return {"error": str(e), "ok": False}


def get_chain_status():
    """Read chain_log.txt and training.log for live running status."""
    chain = Path("chain_log.txt")
    training = Path("specialists/math/add/training.log")
    result = {
        "running": False,
        "current_experiment": None,
        "chain_log": [],
        "training_log": [],
        "eta": None,
    }

    # Read chain log (last 10 lines)
    if chain.exists():
        lines = chain.read_text().strip().split("\n")
        # Get last 15 lines but only include meaningful ones
        recent = []
        for l in lines[-20:]:
            l = l.strip()
            if l and not l.startswith("# ---") and len(l) > 10:
                recent.append(l)
        result["chain_log"] = recent[-10:]

        # Check if chain is running — look for recent timestamps
        if recent:
            result["running"] = True
            # Find current experiment label
            for l in reversed(recent):
                if "STARTING:" in l or "Running" in l:
                    result["current_experiment"] = (
                        l.split("STARTING:")[-1].strip().split("--")[0].strip()
                    )
                    break

    # Read training progress
    if training.exists():
        lines = training.read_text().strip().split("\n")
        relevant = [l for l in lines if "Step" in l or "Eval" in l or "Done" in l]
        result["training_log"] = relevant[-5:]

        # Parse step info
        for l in reversed(lines):
            import re

            m = re.search(r"Step\s+(\d+)/(\d+)", l)
            if m:
                result["step"] = int(m.group(1))
                result["max_steps"] = int(m.group(2))
                m2 = re.search(r"loss=([\d.]+)", l)
                if m2:
                    result["loss"] = float(m2.group(1))
                break

    return result


def get_system_resources():
    """Get CPU, memory, thread stats using psutil."""
    import psutil

    p = psutil.Process()
    cpu_percent = p.cpu_percent(interval=0)  # non-blocking
    memory_info = p.memory_info()
    mem_rss_mb = memory_info.rss / 1e6
    mem_vms_mb = memory_info.vms / 1e6
    threads = p.num_threads()

    # System-wide
    sys_cpu = psutil.cpu_percent(interval=0)
    sys_mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()

    return {
        "process": {
            "pid": p.pid,
            "cpu_percent": cpu_percent,
            "memory_rss_mb": round(mem_rss_mb, 1),
            "memory_vms_mb": round(mem_vms_mb, 1),
            "num_threads": threads,
            "status": p.status(),
            "create_time": p.create_time(),
        },
        "system": {
            "cpu_percent": sys_cpu,
            "cpu_count": cpu_count,
            "memory_percent": sys_mem.percent,
            "memory_total_gb": round(sys_mem.total / 1e9, 1),
            "memory_available_gb": round(sys_mem.available / 1e9, 1),
            "memory_used_gb": round(sys_mem.used / 1e9, 1),
        },
    }


def load_checkpoint(checkpoint_path: str):
    """Load a specific checkpoint, update MODEL_CACHE."""
    import torch

    from tabula_rasa.config import Config
    from tabula_rasa.model import MathTransformer
    from tabula_rasa.tokenizer import MathTokenizer

    path = Path(checkpoint_path)
    if not path.exists():
        return None

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]

    tok = MathTokenizer()
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    # Auto-detect architecture from checkpoint
    cfg.d_model = sd["token_embedding.weight"].shape[1]
    cfg.n_layers = len(
        [k for k in sd if k.startswith("layers.") and k.endswith(".attention.wq.weight")]
    )
    if cfg.n_layers > 0:
        cfg.d_ff = sd["layers.0.feed_forward.w1.weight"].shape[0]
    cfg.n_heads = {128: 4, 64: 2, 96: 4, 256: 8, 192: 6}.get(cfg.d_model, max(1, cfg.d_model // 32))

    model = MathTransformer(cfg)
    model.load_state_dict(sd, strict=False)
    model.eval()

    MODEL_CACHE[:] = [model, tok, cfg]
    return MODEL_CACHE


def run_mcts_search(problem: str, sims: int = 32) -> dict:
    """Run MCTS on a problem and return the tree for visualization."""
    import sys as _sys
    _p = Path(__file__).resolve().parent
    if str(_p) not in _sys.path: _sys.path.insert(0, str(_p))
    if str(_p / ".." / "src") not in _sys.path: _sys.path.insert(0, str(_p / ".." / "src"))
    try:
        from egefalos.mcts import MicroMCTS
        from egefalos.math_gym_env import MathGymEnv
        from tabula_rasa.tokenizer import MathTokenizer
        from tabula_rasa.config import Config
        from tabula_rasa.model import MathTransformer

        tok = MathTokenizer()
        cfg = Config()
        cfg.vocab_size = tok.vocab_size
        model = MathTransformer(cfg)
        model.eval()

        env = MathGymEnv(tokenizer=tok, op="add", max_digits=4, max_seq_len=32)
        mcts = MicroMCTS(model=model, tokenizer=tok, num_simulations=sims,
                         c_puct=1.0, discount=0.99, device="cpu")

        obs, info = env.reset()
        input_ids = obs["input_ids"].unsqueeze(0)
        trajectory, _ = mcts.search(input_ids)

        def serialize_node(node, max_depth=6, depth=0):
            if depth > max_depth: return None
            d = {
                "token_ids": node.token_ids if hasattr(node, "token_ids") else [],
                "visit_count": node.visit_count,
                "mean_value": node.mean_value if hasattr(node, "mean_value") else 0.0,
                "prior_prob": node.prior_prob,
                "is_terminal": node.is_terminal,
                "children": {},
            }
            if hasattr(node, "children") and node.children:
                for tid, child in node.children.items():
                    child_d = serialize_node(child, max_depth, depth + 1)
                    if child_d:
                        d["children"][str(tid)] = child_d
            return d

        root_s = serialize_node(mcts.root if hasattr(mcts, "root") else None)
        top_path = tok.decode(trajectory, skip_special=True) if trajectory else ""

        return {
            "root": root_s, "top_path": top_path,
            "nodes": _count_mcts_nodes(root_s),
            "best_value": root_s.get("mean_value", 0) if root_s else 0,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e), "root": None, "top_path": "", "nodes": 0, "best_value": 0}


def _count_mcts_nodes(node):
    if node is None: return 0
    c = 1
    for ch in (node.get("children") or {}).values():
        c += _count_mcts_nodes(ch)
    return c


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/training-progress":
            self._json(get_progress())
        elif path == "/health":
            m = load_model()
            if m:
                _, _, cfg = m
                p = sum(p.numel() for p in m[0].parameters())
                self._json(
                    {
                        "model": f"MathTransformer-{cfg.d_model}x{cfg.n_layers}",
                        "params": p,
                        "corrections": 0,
                    }
                )
            else:
                self._json({"model": "none", "params": 0, "corrections": 0})
        elif path == "/api/ewc-fisher":
            self._json(get_ewc_fisher())
        elif path == "/api/ewc-stress-test":
            self._json(run_ewc_stress_test())
        elif path == "/api/mcts-tree":
            from urllib.parse import parse_qs
            params = parse_qs(urlparse(self.path).query)
            problem = params.get("problem", ["12+34"])[0]
            sims = int(params.get("sims", [32])[0])
            self._json(run_mcts_search(problem, sims))
        elif path == "/api/cl-metrics":
            self._json(get_cl_metrics())
        elif path == "/api/specialist-usage":
            self._json(get_specialist_usage())
        elif path == "/training-log":
            # Find the most recently updated training log
            log_candidates = [
                Path("specialists/math/add/training.log"),
                Path("specialists/math/sub/training.log"),
                Path("specialists/math/mul/training.log"),
                Path("specialists/math/div/training.log"),
                LOG,
            ]
            best = None
            best_mtime = 0
            for lc in log_candidates:
                if lc.exists():
                    mtime = lc.stat().st_mtime
                    if mtime > best_mtime:
                        best = lc
                        best_mtime = mtime
            if best:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(best.read_bytes())
            else:
                self._json({"error": "No log"}, 404)
        elif path == "/server-status":
            self._json(get_server_status())
        elif path == "/restart":
            self._json({"status": "restarting", "pid": os.getpid()})
            # Spawn a new server process before killing ourselves
            subprocess.Popen(
                [sys.executable, __file__, str(PORT)],
                cwd=os.getcwd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give the new process a moment to bind
            time.sleep(0.5)
            os._exit(0)
        elif path == "/api/config":
            self._json(get_config_dict())
        elif path == "/config-raw":
            fp = _config_path()
            if fp.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(fp.read_bytes())
            else:
                self._json({"error": "Not found"}, 404)
        elif path == "/api/checkpoints":
            self._json(list_available_checkpoints())
        elif path == "/api/experiments":
            self._json(get_experiments())
        elif path == "/api/experiments/status":
            self._json(get_training_status())
        elif path == "/api/registry":
            self._json(get_registry())
        elif path == "/api/chain-status":
            self._json(get_chain_status())
        elif path == "/api/system-resources":
            self._json(get_system_resources())
        elif path == "/api/train/auto/status":
            self._json(auto_train_status())
        elif path == "/api/train/status":
            self._json(train_status())
        else:
            self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        data = self._read_body()
        if path == "/generate":
            # Optional checkpoint override
            ckpt = (data or {}).get("checkpoint", None)
            if ckpt:
                m = load_checkpoint(ckpt)
            else:
                m = load_model()
            if not m:
                return self._json({"error": "No model loaded"}, 503)
            model, tok, cfg = m
            prompt = (data or {}).get("prompt", "").strip()
            if not prompt:
                return self._json({"error": "Missing prompt"}, 400)
            if not prompt.endswith("="):
                prompt += "="
            import torch

            with torch.no_grad():
                out = model.generate(tok, prompt, max_new_tokens=10, temperature=0.0)
            self._json(
                {
                    "prompt": prompt,
                    "result": out,
                    "time_ms": 0,
                    "model": f"MT-{cfg.d_model}x{cfg.n_layers}",
                }
            )
        elif path == "/correct":
            m = load_model()
            if not m:
                return self._json({"error": "No model"}, 503)
            model, tok, cfg = m
            prompt = (data or {}).get("prompt", "").rstrip("=")
            ans = (data or {}).get("answer", "")
            import torch

            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            text = f"{prompt}={ans}"
            ids = tok.encode(text, add_special_tokens=True)
            ids = (ids + [tok.pad_id] * cfg.max_seq_len)[: cfg.max_seq_len]
            x = torch.tensor(ids[:-1]).unsqueeze(0)
            y = torch.tensor(ids[1:]).unsqueeze(0)
            for _ in range(5):
                opt.zero_grad()
                _, l, _ = model(x, y)
                l.backward()
                opt.step()
            model.eval()
            self._json({"status": "ok", "final_loss": l.item(), "total_corrections": 1})
        elif path == "/api/config":
            if not data:
                self._json({"error": "No config data"}, 400)
                return
            changes = save_config(data)
            # Invalidate model cache so next load picks up new architecture
            MODEL_CACHE[:] = [None, None, None]
            self._json({"status": "saved", "changes": changes})
        elif path == "/api/load-checkpoint":
            ckpt = (data or {}).get("checkpoint", "")
            if not ckpt:
                self._json({"error": "Missing checkpoint path"}, 400)
                return
            m = load_checkpoint(ckpt)
            if not m:
                self._json({"error": f"Checkpoint not found: {ckpt}"}, 404)
                return
            _, _, cfg = m
            p = sum(p.numel() for p in m[0].parameters())
            self._json(
                {
                    "status": "loaded",
                    "model": f"MathTransformer-{cfg.d_model}x{cfg.n_layers}",
                    "params": p,
                }
            )
        elif path == "/api/experiments":
            if data:
                if "_replace_all" in data:
                    # Client sending full replacement (delete/edit notes)
                    EXPERIMENTS.write_text(json.dumps(data["_replace_all"], indent=2))
                    self._json({"status": "replaced", "count": len(data["_replace_all"])})
                else:
                    entry = save_experiment(data)
                    self._json(entry)
            else:
                self._json({"error": "No experiment data"}, 400)
        elif path == "/api/export":
            self._json(export_checkpoint(data))
        elif path == "/api/import":
            self._json(import_checkpoint(data))
        elif path == "/api/train/start":
            self._json(train_start(data))
            MODEL_CACHE[:] = [None, None, None]  # Invalidate — new model being trained
        elif path == "/api/train/stop":
            self._json(train_stop(data))
        elif path == "/api/train/pause":
            self._json(train_pause(data))
        elif path == "/api/train/auto":
            self._json(auto_train_start(data))
        elif path == "/api/train/auto/status":
            self._json(auto_train_status())
        elif path == "/api/train/resume":
            # Resume = start with --resume flag
            if data is None:
                data = {}
            data["resume"] = True
            self._json(train_start(data))
        elif path == "/api/train/status":
            self._json(train_status(data))
        elif path == "/api/clear-training-log":
            # Clear the most recently updated training log
            for lp in [
                Path(f"specialists/math/{op}/training.log") for op in ["add", "sub", "mul", "div"]
            ]:
                if lp.exists():
                    lp.write_text("")
            self._json({"ok": True})
        else:
            self._json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json(self, d, s=200):
        b = json.dumps(d, ensure_ascii=False).encode()
        try:
            self.send_response(s)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (ConnectionAbortedError, BrokenPipeError, OSError):
            pass  # Client disconnected before response was fully sent

    def _read_body(self):
        l = int(self.headers.get("Content-Length", 0))
        if l == 0:
            return None
        try:
            return json.loads(self.rfile.read(l))
        except:
            return None

    def _serve_static(self, path):
        if path == "/":
            path = "/dashboard.html"
        fp = (DASH / path.lstrip("/")).resolve()
        try:
            fp.relative_to(DASH.resolve())
        except:
            return self._json({"error": "Forbidden"}, 403)
        if fp.is_file():
            ct = mimetypes.guess_type(fp.name)[0] or "application/octet-stream"
            d = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(d)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(d)
        else:
            self._json({"error": "Not found"}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    os.makedirs(str(DASH), exist_ok=True)
    print(f"  API server on http://0.0.0.0:{PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
