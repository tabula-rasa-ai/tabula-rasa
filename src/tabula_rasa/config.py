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
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"

    # ── Quantization (inference only, requires bitsandbytes) ──
    load_in_8bit: bool = False  # Enable 8-bit quantization (requires bitsandbytes)
    load_in_4bit: bool = False  # Enable 4-bit quantization (requires bitsandbytes)

    # ═══════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE — 1M params (full capacity)
    # ═══════════════════════════════════════════════════════════
    config_preset: str = "1M"  # 1M | 10M — use apply_preset() to set architecture
    d_model: int = 128  # Embedding / hidden dimension
    n_layers: int = 4  # Number of transformer layers (depth)
    n_heads: int = 4  # Attention heads (must divide d_model)
    d_ff: int = 512  # Feed-forward hidden dimension
    max_seq_len: int = 256  # Default context window; supports CoT/MCTS reasoning; KV-cache makes O(n) generation
    dropout: float = 0.1  # Dropout rate (0=off, 0.0-0.5)

    # Architecture variants
    activation: str = "swiglu"  # relu | gelu | swiglu
    norm_type: str = "rmsnorm"  # rmsnorm | layernorm
    pos_encoding: str = "rope"  # learned | rope | none

    # Mixture of Experts (MoE) — replaces FFN with expert routing
    use_moe: bool = False         # Enable MoE layers
    num_experts: int = 4          # Number of experts in each MoE layer
    top_k: int = 2                # Top-K experts to route each token to
    moe_capacity_factor: float = 1.25  # Expert capacity multiplier

    # Weight initialization
    weight_init: str = "normal"  # normal | xavier | kaiming
    init_std: float = 0.02  # Std for normal init (ignored for xavier/kaiming)

    # ═══════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════
    batch_size: int = 128  # CPU throughput is flat above 64; fewer epochs
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 30000
    log_every: int = 500
    eval_every: int = 1000  # More frequent feedback (was 2000)
    save_every: int = 2000  # More frequent checkpoints (was 5000)

    # ── Curriculum (advanced) ─────────────────────────────
    use_curriculum: bool = True  # Enable curriculum learning
    curriculum_phases: list[tuple[int, int]] = [
        # (steps, max_digits) - train on this difficulty for N steps
        (30000, 1),  # Single phase: 1-digit only for clean convergence
    ]
    use_entropy_curriculum: bool = False  # Advance phases based on output entropy
    entropy_threshold: float = 0.5  # Advance when avg entropy < this (lower=more confident)
    entropy_window: int = 3  # Number of evals to smooth over

    # Optimizer
    optimizer: str = "adamw"  # adamw | adam | sgd
    lr_schedule: str = "cosine"  # cosine | linear | constant
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Training algorithm flags
    use_reversed: bool = True  # Reverse digits for add/sub (aligns carry)
    use_loss_masking: bool = True  # ignore_index=-100 for prompt/PAD tokens
    use_hard_negative: bool = True  # Entropy-guided curriculum in ReST loop
    label_smoothing: float = 0.0  # 0=off, >0 softens targets
    tie_embeddings: bool = False  # False=untied (better for algorithmic tasks)
    grad_clip_norm: float = 1.0  # Max gradient norm (0=no clipping)
    use_value_head: bool = False  # AlphaZero-style value head (-1 to +1 intuition)
    alphazero_loss_weight: float = 0.5  # Weight for value loss when use_value_head=True

    # ── PPO / RL ──────────────────────────
    use_rl: bool = False  # Enable PPO reinforcement learning training
    rl_total_steps: int = 5000  # Total environment steps for PPO
    rl_lr: float = 3e-4  # PPO learning rate
    rl_clip_epsilon: float = 0.2  # PPO clip range
    rl_gamma: float = 0.99  # PPO discount factor
    rl_gae_lambda: float = 0.95  # GAE lambda
    rl_value_coef: float = 0.5  # Value loss coefficient
    rl_entropy_coef: float = 0.01  # Entropy bonus coefficient
    rl_update_epochs: int = 4  # PPO update epochs per batch
    rl_batch_size: int = 64  # PPO minibatch size
    rl_horizon: int = 512  # Steps between PPO updates
    rl_max_grad_norm: float = 0.5  # PPO gradient clip norm

    # ── Socratic Stage 3 ──
    use_socratic_stage3: bool = False  # Enable multi-persona arithmetic debate
    stage3_problems: int = 20  # Number of problems to debate in Stage 3

    # ── Low-Rank Adaptation (LoRA) ──────────────────────────
    use_lora: bool = False  # Enable LoRA fine-tuning
    lora_rank: int = 8  # LoRA rank (r)
    lora_alpha: float = 1.0  # LoRA scaling alpha
    lora_target_modules: str = "wq,wk,wv,wo"  # Comma-separated target module substrings

    # Mixed precision (AMP)
    use_amp: bool = False  # Enable torch.cuda.amp (only effective on CUDA)
    gradient_accumulation_steps: int = 1  # Accumulate N micro-batches before optimizer step

    # Evaluation
    eval_temperature: float = 0.0  # 0=greedy, >0 adds randomness
    eval_max_tokens: int = 48  # Max tokens to generate per answer (longer for scratchpad/CoT)

    # ═══════════════════════════════════════════════════════════
    # DATA
    # ═══════════════════════════════════════════════════════════
    train_samples: int = 5_000  # 5K prevents memorization (20K was still overfitting)
    eval_samples: int = 100  # Standard eval set
    min_digits: int = 1
    max_digits: int = 4  # 1-4 digits for meaningful carry propagation
    force_carry_ratio: float = 0.5  # ~50% of problems force a carry (a+b >= 10)
    use_scratchpad: bool = True  # Show carry scratchpad in output
    cot_scratchpad: bool = False  # Use readable column-by-column CoT steps instead of fused carry-digit
    test_hard: bool = False  # Enable hard eval (max_digits+1, tests generalization)

    # ═══════════════════════════════════════════════════════════════
    # LANGUAGE ALPHAZERO — Self-Play Language Acquisition
    # ═══════════════════════════════════════════════════════════════
    # Enable with: use_language_az = True + use_value_head = True
    use_language_az: bool = False  # Enable AlphaZero language self-play
    language_vocab_size: int = 512  # Tokenizer vocab for language (vs math)
    language_max_seq_len: int = 128  # Max tokens per sentence
    language_entropy_threshold: float = 0.5  # Entropy above which MCTS is triggered
    language_mcts_simulations: int = 16  # Micro-MCTS rollouts per search
    language_games_per_session: int = 200  # Games per self-play session
    language_difficulty: str = "medium"  # 'easy', 'medium', or 'hard'
    language_learning_rate: float = 1e-4  # LR for language self-play training

    # ═══════════════════════════════════════════════════════════════
    # CODE ALPHAZERO — Self-Play Programming via Python Sandbox
    # ═══════════════════════════════════════════════════════════════
    use_code_az: bool = False  # Enable Code AlphaZero self-play
    code_max_seq_len: int = 256  # Max tokens per code snippet
    code_timeout_sec: float = 2.0  # Sandbox execution timeout
    code_syntax_games: int = 100  # Stage 1: syntax games per session
    code_fuzzing_games: int = 50  # Stage 2: fuzzing games per session
    code_algorithm_games: int = 30  # Stage 3: algorithm games per session
    code_learning_rate: float = 1e-4  # LR for code self-play

    # ── Paths ─────────────────────────────────────────────────
    save_dir: str = "checkpoints"
    data_dir: str = "data"

    # ── Text Tokenizer (General Text) ───────────────────────
    use_text_tokenizer: bool = False  # Use BPETokenizer instead of MathTokenizer
    text_vocab_size: int = 256  # Target vocab size after BPE learning
    text_max_seq_len: int = 64  # Max sequence length for text tasks
    text_num_merges: int = 50  # BPE merge operations to learn

    @property
    def device(self) -> str:
        """Return the compute device string.

        Priority: cuda > mps (Apple Silicon) > cpu.

        Returns:
            'cuda' if CUDA GPU available, 'mps' if Apple Silicon,
            otherwise 'cpu'.
        """
        import os
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


    def auto_scale(self, op: str = "add", max_digits: int = 1) -> "Config":
        """Automatically pick model size and training params based on task complexity.

        Complexity score = operation_weight * digit_weight:
          op: add=1, sub=1, mul=2, div=2, pow=2
          digits: max_digits * 1.5
          CoT: +2 if enabled
          curriculum phases: +1 per phase beyond 1

        Score -> Preset:
          < 3: 1M model (default, sufficient for 1-digit add/sub)
          3-6: 5M model (multi-digit, multiplication)
          > 6: 10M model (complex reasoning, many phases)

        Also auto-selects batch size, AMP, and gradient accumulation
        based on available hardware.

        Returns self for chaining.
        """
        import os
        import torch

        # Complexity score
        op_weight = {"add": 1, "sub": 1, "mul": 2, "div": 2, "pow": 2}.get(op, 1)
        digit_weight = max_digits * 1.5
        cot_bonus = 2 if getattr(self, "cot_scratchpad", False) else 0
        phases = getattr(self, "curriculum_phases", [])
        curriculum_bonus = max(0, len(phases) - 1)

        complexity = op_weight * digit_weight + cot_bonus + curriculum_bonus

        # Model size
        if complexity < 3:
            preset = "1M"
        elif complexity < 6:
            preset = "5M"
        else:
            preset = "10M"
        self.apply_preset(preset)
        print(f"  [Auto] Task complexity={complexity:.1f} -> preset={preset}")

        # Hardware-aware batch size and AMP
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if mem_gb >= 16:
                self.batch_size = 512
            elif mem_gb >= 8:
                self.batch_size = 256
            else:
                self.batch_size = 128
            self.use_amp = True
            print(f"  [Auto] GPU ({mem_gb:.0f}GB VRAM) -> batch={self.batch_size}, AMP=ON")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.batch_size = 128
            self.use_amp = False
            print(f"  [Auto] Apple Silicon -> batch={self.batch_size}, AMP=OFF")
        else:
            self.batch_size = 128
            self.use_amp = False
            self.num_threads = os.cpu_count() or 4
            torch.set_num_threads(self.num_threads)
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
            os.environ["MKL_NUM_THREADS"] = str(self.num_threads)
            print(f"  [Auto] CPU -> batch={self.batch_size}, AMP=OFF, threads={self.num_threads}")

        # Gradient accumulation for consistent effective batch
        target_effective = 512
        if self.batch_size < target_effective:
            self.gradient_accumulation_steps = max(1, target_effective // self.batch_size)
        else:
            self.gradient_accumulation_steps = 1
        if self.gradient_accumulation_steps > 1:
            print(f"  [Auto] Grad accum: {self.gradient_accumulation_steps} steps")

        return self

    vocab_size: int | None = None


    PRESETS: dict[str, dict] = {
        "1M": {
            "d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
            "max_seq_len": 32, "dropout": 0.1, "params_approx": 1_061_504,
        },
        "5M": {
            "d_model": 256, "n_layers": 5, "n_heads": 8, "d_ff": 1024,
            "max_seq_len": 48, "dropout": 0.1, "params_approx": 5_269_248,
        },
        "10M": {
            "d_model": 320, "n_layers": 6, "n_heads": 8, "d_ff": 1280,
            "max_seq_len": 64, "dropout": 0.1, "params_approx": 9_860_000,
        },
    }

    def apply_preset(self, name: str | None = None) -> "Config":
        """Apply a named architecture preset.

        Sets d_model, n_layers, n_heads, d_ff, max_seq_len, dropout.
        Returns self for chaining.

        Presets:
            "1M"  — Default 1M-param (d=128, L=4, h=4, ff=512, seq=32)
            "5M"  — Scaling law baseline (d=256, L=5, h=8, ff=1024, seq=48)
            "10M" — Scaled 10M-param (d=320, L=6, h=8, ff=1280, seq=64)
        """
        preset = name or self.config_preset
        cfg = self.PRESETS.get(preset)
        if cfg is None:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")
        for k, v in cfg.items():
            if k != "params_approx":
                setattr(self, k, v)
        self.config_preset = preset
        return self
