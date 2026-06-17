"""Tabula Rasa — core library package.

Learning from scratch, one specialist at a time.
"""

from .config import Config
from .dataset import MathDataset, format_math_sample, generate_problem, generate_test_set
from .eval import (
    EvalResult,
    PerPositionResult,
    evaluate_accuracy,
    evaluate_adversarial,
    evaluate_compositional,
    evaluate_ood,
    evaluate_per_position,
    evaluate_robustness,
    full_benchmark,
    solve_problems,
)
from .eval import load_model as eval_load_model
from .experiments import ExperimentRun, compare_runs, list_runs, log_run, print_runs
from .generate import load_model as gen_load_model
from .generate import main as generate_main
from .lora import (
    LoRALayer,
    apply_lora_to_model,
    load_lora_adapters,
    save_lora_adapters,
    set_lora_trainable,
    switch_lora_adapters,
)
from .mcts_inference import MCTSInference, mcts_generate
from .model import MathTransformer, alphazero_loss, count_parameters
from .orchestrator import AnswerResult, PreparedSlateOrchestrator
from .rag import Document, RAGEngine, RetrievalResult
from .tokenizer import MathTokenizer

__all__ = [
    "Config",
    "MathTransformer",
    "count_parameters",
    "alphazero_loss",
    "MathTokenizer",
    "MathDataset",
    "generate_problem",
    "format_math_sample",
    "generate_test_set",
    "evaluate_accuracy",
    "evaluate_adversarial",
    "evaluate_compositional",
    "evaluate_ood",
    "evaluate_per_position",
    "evaluate_robustness",
    "full_benchmark",
    "EvalResult",
    "PerPositionResult",
    "solve_problems",
    "ExperimentRun",
    "compare_runs",
    "list_runs",
    "log_run",
    "print_runs",
    "LoRALayer",
    "apply_lora_to_model",
    "save_lora_adapters",
    "load_lora_adapters",
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
