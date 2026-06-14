"""Configuration for tiny math transformer.
All algorithm knobs in one place — easy to tweak from Model Config dashboard.
"""
from __future__ import annotations


class Config:
    """Configuration class for the Tabula Rasa math transformer.

    All model architecture, training, data, and algorithm settings are
    defined as class-level attributes. Instantiate without arguments to
    use defaults; override attributes as needed.

    Sections:
        - Tokenizer: special token strings
        - Model Architecture: embedding dimension, layers, heads, etc.
        - Training Hyperparameters: batch size, LR, curriculum, optimizer
        - Data: sample counts, digit ranges, scratchpad
        - Language AlphaZero: self-play language acquisition
        - Code AlphaZero: self-play programming
        - Paths: save and data directories
    """

    # ── Tokenizer ───────────────────────────────────────────────
    pad_token: str = '<PAD>'
    bos_token: str = '<BOS>'
    eos_token: str = '<EOS>'
    unk_token: str = '<UNK>'

    # ── Quantization (inference only, requires bitsandbytes) ──
    load_in_8bit: bool = False    # Enable 8-bit quantization (requires bitsandbytes)
    load_in_4bit: bool = False    # Enable 4-bit quantization (requires bitsandbytes)

    # ═══════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE — 1M params (full capacity)
    # ═══════════════════════════════════════════════════════════
    d_model: int = 128         # Embedding / hidden dimension
    n_layers: int = 4          # Number of transformer layers (depth)
    n_heads: int = 4           # Attention heads (must divide d_model)
    d_ff: int = 512            # Feed-forward hidden dimension
    max_seq_len: int = 32      # Enough for scratchpad (1-digit = ~12 tokens)
    dropout: float = 0.1       # Dropout rate (0=off, 0.0-0.5)

    # Architecture variants
    activation: str = 'relu'
    norm_type: str = 'rmsnorm'       # rmsnorm | layernorm
    pos_encoding: str = 'rope'       # learned | rope | none

    # Weight initialization
    weight_init: str = 'normal'      # normal | xavier | kaiming
    init_std: float = 0.02           # Std for normal init (ignored for xavier/kaiming)

    # ═══════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════
    batch_size: int = 32             # Smaller for curriculum + CPU
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 30000
    log_every: int = 500
    eval_every: int = 1000           # More frequent feedback (was 2000)
    save_every: int = 2000           # More frequent checkpoints (was 5000)

    # Curriculum learning: gradually increase difficulty
    use_curriculum: bool = True       # Enable curriculum learning
    curriculum_phases: list[tuple[int, int]] = [
        # (steps, max_digits) - train on this difficulty for N steps
        (5000, 1),              # Steps 0-5000: 1-digit only
        (10000, 2),             # Steps 5000-15000: 1-2 digit
        (10000, 3),             # Steps 15000-25000: 1-3 digit
        (5000, 4),              # Steps 25000+: full 1-4 digit
    ]

    # Optimizer
    optimizer: str = 'adamw'         # adamw | adam | sgd
    lr_schedule: str = 'cosine'      # cosine | linear | constant
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Training algorithm flags
    use_reversed: bool = True         # Reverse digits for add/sub (aligns carry)
    use_loss_masking: bool = True     # ignore_index=-100 for prompt/PAD tokens
    use_hard_negative: bool = True    # Entropy-guided curriculum in ReST loop
    label_smoothing: float = 0.0     # 0=off, >0 softens targets
    tie_embeddings: bool = False      # False=untied (better for algorithmic tasks)
    grad_clip_norm: float = 1.0      # Max gradient norm (0=no clipping)
    use_value_head: bool = False      # AlphaZero-style value head (-1 to +1 intuition)
    alphazero_loss_weight: float = 0.5  # Weight for value loss when use_value_head=True

    # ── Low-Rank Adaptation (LoRA) ──────────────────────────
    use_lora: bool = False             # Enable LoRA fine-tuning
    lora_rank: int = 8                 # LoRA rank (r)
    lora_alpha: float = 1.0            # LoRA scaling alpha
    lora_target_modules: str = 'wq,wk,wv,wo'  # Comma-separated target module substrings

    # Mixed precision (AMP)
    use_amp: bool = False             # Enable torch.cuda.amp (only effective on CUDA)
    gradient_accumulation_steps: int = 1  # Accumulate N micro-batches before optimizer step

    # Evaluation
    eval_temperature: float = 0.0     # 0=greedy, >0 adds randomness
    eval_max_tokens: int = 20         # Max tokens to generate per answer (longer for scratchpad)

    # ═══════════════════════════════════════════════════════════
    # DATA
    # ═══════════════════════════════════════════════════════════
    train_samples: int = 5_000      # 5K prevents memorization (20K was still overfitting)
    eval_samples: int = 100         # Standard eval set
    min_digits: int = 1
    max_digits: int = 4             # 1-4 digits for meaningful carry propagation
    force_carry_ratio: float = 0.5  # ~50% of problems force a carry (a+b >= 10)
    use_scratchpad: bool = True      # Show carry scratchpad in output
    test_hard: bool = False          # Enable hard eval (max_digits+1, tests generalization)

    # ═══════════════════════════════════════════════════════════════
    # LANGUAGE ALPHAZERO — Self-Play Language Acquisition
    # ═══════════════════════════════════════════════════════════════
    # Enable with: use_language_az = True + use_value_head = True
    use_language_az: bool = False         # Enable AlphaZero language self-play
    language_vocab_size: int = 512        # Tokenizer vocab for language (vs math)
    language_max_seq_len: int = 128       # Max tokens per sentence
    language_entropy_threshold: float = 0.5  # Entropy above which MCTS is triggered
    language_mcts_simulations: int = 16   # Micro-MCTS rollouts per search
    language_games_per_session: int = 200  # Games per self-play session
    language_difficulty: str = 'medium'   # 'easy', 'medium', or 'hard'
    language_learning_rate: float = 1e-4  # LR for language self-play training

    # ═══════════════════════════════════════════════════════════════
    # CODE ALPHAZERO — Self-Play Programming via Python Sandbox
    # ═══════════════════════════════════════════════════════════════
    use_code_az: bool = False              # Enable Code AlphaZero self-play
    code_max_seq_len: int = 256            # Max tokens per code snippet
    code_timeout_sec: float = 2.0          # Sandbox execution timeout
    code_syntax_games: int = 100           # Stage 1: syntax games per session
    code_fuzzing_games: int = 50           # Stage 2: fuzzing games per session
    code_algorithm_games: int = 30         # Stage 3: algorithm games per session
    code_learning_rate: float = 1e-4       # LR for code self-play

    # ── Paths ─────────────────────────────────────────────────
    save_dir: str = 'checkpoints'
    data_dir: str = 'data'

    @property
    def device(self) -> str:
        """Return the compute device string ('cuda' or 'cpu').

        Returns:
            'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'.
        """
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_size: int | None = None  # Set after tokenizer is built
