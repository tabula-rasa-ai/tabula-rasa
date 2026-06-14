"""Tabula Rasa — core library package.

Learning from scratch, one specialist at a time.
"""
from .config import Config
from .model import MathTransformer, count_parameters, alphazero_loss
from .tokenizer import MathTokenizer
from .dataset import MathDataset, generate_problem, format_math_sample, generate_test_set
from .eval import evaluate_accuracy, load_model as eval_load_model, solve_problems
from .generate import load_model as gen_load_model, main as generate_main
from .lora import (
    LoRALayer,
    apply_lora_to_model,
    save_lora_adapters,
    load_lora_adapters,
    set_lora_trainable,
    switch_lora_adapters,
)

__all__ = [
    'Config', 'MathTransformer', 'count_parameters', 'alphazero_loss',
    'MathTokenizer',
    'MathDataset', 'generate_problem', 'format_math_sample', 'generate_test_set',
    'evaluate_accuracy', 'solve_problems',
    'LoRALayer', 'apply_lora_to_model', 'save_lora_adapters',
    'load_lora_adapters', 'set_lora_trainable', 'switch_lora_adapters',
]
