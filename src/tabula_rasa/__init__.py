"""Tabula Rasa — core library package.

Learning from scratch, one specialist at a time.
"""

from __future__ import annotations

import importlib
from typing import Any

from .config import Config
from .tokenizer import MathTokenizer

# ── Lazy imports (torch-dependent) ──────────────────────────
# These modules import torch at module level, which crashes on CI
# runners with broken torch installs. We lazy-import them so
# ``import tabula_rasa`` works without torch.

_LAZY: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy-import torch-dependent names on first access."""
    # Module name → attribute dict
    _MODULE_MAP: dict[str, dict[str, str]] = {
        "dataset": {
            "MathDataset": "MathDataset",
            "format_math_sample": "format_math_sample",
            "generate_problem": "generate_problem",
            "generate_test_set": "generate_test_set",
        },
        "eval": {
            "EvalResult": "EvalResult",
            "PerPositionResult": "PerPositionResult",
            "evaluate_accuracy": "evaluate_accuracy",
            "evaluate_adversarial": "evaluate_adversarial",
            "evaluate_compositional": "evaluate_compositional",
            "evaluate_ood": "evaluate_ood",
            "evaluate_per_position": "evaluate_per_position",
            "evaluate_robustness": "evaluate_robustness",
            "full_benchmark": "full_benchmark",
            "solve_problems": "solve_problems",
            "eval_load_model": "load_model",
            "load_model": "load_model",
        },
        "experiments": {
            "ExperimentRun": "ExperimentRun",
            "compare_runs": "compare_runs",
            "list_runs": "list_runs",
            "log_run": "log_run",
            "print_runs": "print_runs",
        },
        "generate": {
            "gen_load_model": "load_model",
            "generate_main": "main",
            "load_model": "load_model",
        },
        "lora": {
            "LoRALayer": "LoRALayer",
            "apply_lora_to_model": "apply_lora_to_model",
            "load_lora_adapters": "load_lora_adapters",
            "save_lora_adapters": "save_lora_adapters",
            "set_lora_trainable": "set_lora_trainable",
            "switch_lora_adapters": "switch_lora_adapters",
        },
        "mcts_inference": {
            "MCTSInference": "MCTSInference",
            "mcts_generate": "mcts_generate",
        },
        "model": {
            "MathTransformer": "MathTransformer",
            "alphazero_loss": "alphazero_loss",
            "count_parameters": "count_parameters",
        },
        "orchestrator": {
            "AnswerResult": "AnswerResult",
            "PreparedSlateOrchestrator": "PreparedSlateOrchestrator",
        },
        "rag": {
            "AnswerResult": "AnswerResult",
            "Document": "Document",
            "RAGEngine": "RAGEngine",
            "RetrievalResult": "RetrievalResult",
            "PreparedSlateOrchestrator": "PreparedSlateOrchestrator",
        },
    }

    for mod_name, attrs in _MODULE_MAP.items():
        if name in attrs:
            mod = importlib.import_module(f".{mod_name}", __package__)
            return getattr(mod, attrs[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Config",
    "MathTokenizer",
    "MathDataset",
    "format_math_sample",
    "generate_problem",
    "generate_test_set",
    "MathTransformer",
    "count_parameters",
    "alphazero_loss",
    "EvalResult",
    "PerPositionResult",
    "evaluate_accuracy",
    "evaluate_adversarial",
    "evaluate_compositional",
    "evaluate_ood",
    "evaluate_per_position",
    "evaluate_robustness",
    "full_benchmark",
    "solve_problems",
    "load_model",
    "ExperimentRun",
    "compare_runs",
    "list_runs",
    "log_run",
    "print_runs",
    "LoRALayer",
    "apply_lora_to_model",
    "load_lora_adapters",
    "save_lora_adapters",
    "set_lora_trainable",
    "switch_lora_adapters",
    "MCTSInference",
    "mcts_generate",
    "PreparedSlateOrchestrator",
    "AnswerResult",
    "RAGEngine",
    "Document",
    "RetrievalResult",
]
