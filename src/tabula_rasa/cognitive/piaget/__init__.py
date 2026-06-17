"""Piaget Benchmark Suite — measuring cognitive development stages.

Maps directly to Piaget's stages of cognitive development:
1. Sensorimotor  — Token Identity (can the model distinguish 12 from 21?)
2. Preoperational — Symbolic Routing (does it emit <UNK> for unknown domains?)
3. Concrete Operational — Reversibility (if a+b=c, can it solve c-b=a?)
4. Formal Operational — Hypothetical Reasoning (can the Skeptic predict counters?)

Usage:
    from tabula_rasa.cognitive.piaget.runner import run_piaget_benchmark
    results = run_piaget_benchmark(model, tokenizer)
"""

from .runner import run_piaget_benchmark

__all__ = ['run_piaget_benchmark']
