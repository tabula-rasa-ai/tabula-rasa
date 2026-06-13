"""Configuration for tiny math transformer.
All algorithm knobs in one place — easy to tweak from Model Config dashboard.
"""
class Config:
    # ── Tokenizer ───────────────────────────────────────────────
    pad_token = '<PAD>'
    bos_token = '<BOS>'
    eos_token = '<EOS>'
    unk_token = '<UNK>'

    # ═══════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE — 1M params (full capacity)
    # ═══════════════════════════════════════════════════════════
    d_model = 128         # Embedding / hidden dimension
    n_layers = 4          # Number of transformer layers (depth)
    n_heads = 4           # Attention heads (must divide d_model)
    d_ff = 512            # Feed-forward hidden dimension
    max_seq_len = 32      # Enough for scratchpad (1-digit = ~12 tokens)
    dropout = 0.1         # Dropout rate (0=off, 0.0-0.5)

    # Architecture variants
    activation = 'relu'
    norm_type = 'rmsnorm'       # rmsnorm | layernorm
    pos_encoding = 'rope'       # learned | rope | none

    # Weight initialization
    weight_init = 'normal'      # normal | xavier | kaiming
    init_std = 0.02             # Std for normal init (ignored for xavier/kaiming)

    # ═══════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════
    batch_size = 32             # Smaller for curriculum + CPU
    learning_rate = 0.001
    weight_decay = 0.01
    warmup_steps = 500
    max_steps = 30000
    log_every = 500
    eval_every = 1000           # More frequent feedback (was 2000)
    save_every = 2000           # More frequent checkpoints (was 5000)
    
    # Curriculum learning: gradually increase difficulty
    use_curriculum = True       # Enable curriculum learning
    curriculum_phases = [
        # (steps, max_digits) - train on this difficulty for N steps
        (5000, 1),              # Steps 0-5000: 1-digit only
        (10000, 2),             # Steps 5000-15000: 1-2 digit
        (10000, 3),             # Steps 15000-25000: 1-3 digit
        (5000, 4),              # Steps 25000+: full 1-4 digit
    ]

    # Optimizer
    optimizer = 'adamw'         # adamw | adam | sgd
    lr_schedule = 'cosine'      # cosine | linear | constant
    adam_beta1 = 0.9
    adam_beta2 = 0.999

    # Training algorithm flags
    use_reversed = True         # Reverse digits for add/sub (aligns carry)
    use_loss_masking = True     # ignore_index=-100 for prompt/PAD tokens
    use_hard_negative = True    # Entropy-guided curriculum in ReST loop
    label_smoothing = 0.0       # 0=off, >0 softens targets
    tie_embeddings = False      # False=untied (better for algorithmic tasks)
    grad_clip_norm = 1.0        # Max gradient norm (0=no clipping)
    use_value_head = False      # AlphaZero-style value head (-1 to +1 intuition)
    alphazero_loss_weight = 0.5 # Weight for value loss when use_value_head=True

    # Mixed precision (AMP)
    use_amp = False             # Enable torch.cuda.amp (only effective on CUDA)
    gradient_accumulation_steps = 1  # Accumulate N micro-batches before optimizer step

    # Evaluation
    eval_temperature = 0.0      # 0=greedy, >0 adds randomness
    eval_max_tokens = 20        # Max tokens to generate per answer (longer for scratchpad)

    # ═══════════════════════════════════════════════════════════
    # DATA
    # ═══════════════════════════════════════════════════════════
    train_samples = 5_000      # 5K prevents memorization (20K was still overfitting)
    eval_samples = 100         # Standard eval set
    min_digits = 1
    max_digits = 4             # 1-4 digits for meaningful carry propagation
    force_carry_ratio = 0.5    # ~50% of problems force a carry (a+b >= 10)
    use_scratchpad = True      # Show carry scratchpad in output
    test_hard = False          # Enable hard eval (max_digits+1, tests generalization)

    # ═══════════════════════════════════════════════════════════════
    # LANGUAGE ALPHAZERO — Self-Play Language Acquisition
    # ═══════════════════════════════════════════════════════════════
    # Enable with: use_language_az = True + use_value_head = True
    use_language_az = False         # Enable AlphaZero language self-play
    language_vocab_size = 512       # Tokenizer vocab for language (vs math)
    language_max_seq_len = 128      # Max tokens per sentence
    language_entropy_threshold = 0.5  # Entropy above which MCTS is triggered
    language_mcts_simulations = 16  # Micro-MCTS rollouts per search
    language_games_per_session = 200  # Games per self-play session
    language_difficulty = 'medium'  # 'easy', 'medium', or 'hard'
    language_learning_rate = 1e-4   # LR for language self-play training
    
    # ═══════════════════════════════════════════════════════════════
    # CODE ALPHAZERO — Self-Play Programming via Python Sandbox
    # ═══════════════════════════════════════════════════════════════
    use_code_az = False              # Enable Code AlphaZero self-play
    code_max_seq_len = 256           # Max tokens per code snippet
    code_timeout_sec = 2.0           # Sandbox execution timeout
    code_syntax_games = 100          # Stage 1: syntax games per session
    code_fuzzing_games = 50          # Stage 2: fuzzing games per session
    code_algorithm_games = 30        # Stage 3: algorithm games per session
    code_learning_rate = 1e-4        # LR for code self-play
    
    # ── Paths ─────────────────────────────────────────────────
    save_dir = 'checkpoints'
    data_dir = 'data'

    @property
    def device(self):
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_size = None  # Set after tokenizer is built
