"""Tabula Rasa AI System — learns skills one at a time.

Each skill = a specialist transformer trained from scratch.
The router knows what it knows, and says "I don't know" for what it doesn't.

Usage:
    python3 tabula_rasa.py              # Start the AI
    python3 tabula_rasa.py --learn biology  # Queue a new skill for training
"""

import argparse, json, time, sys, re, os, random, threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from socketserver import ThreadingMixIn

# Ensure src/ is on path before importing tabula_rasa package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEBUG_LOG = Path("debug_tabula.log")

def debug(msg):
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

import torch
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.lora import (
    LoRALayer,
    apply_lora_to_model,
    set_lora_trainable,
    save_lora_adapters,
    switch_lora_adapters,
)


# Module-level flag: train new conversational specialists as LoRA adapters
# on a shared base model instead of full models from scratch.
USE_LORA_FOR_CHAT = True


# ─── Skill Registry ────────────────────────────────────────────────

SKILL_REGISTRY = {
    'addition': {
        'ops': ['+', 'plus', 'add', 'sum'],
        'description': 'Can add two numbers together',
        'status': 'training',
        'dir': 'specialists/math/add',
    },
    'subtraction': {
        'ops': ['-', 'minus', 'subtract', 'difference'],
        'description': 'Can subtract one number from another',
        'status': 'queued',
        'dir': 'specialists/math/sub',
    },
    'multiplication': {
        'ops': ['*', '×', 'multiply', 'times', 'product'],
        'description': 'Can multiply two numbers',
        'status': 'queued',
        'dir': 'specialists/math/mul',
    },
    'division': {
        'ops': ['/', '÷', 'divide', 'quotient'],
        'description': 'Can divide one number by another',
        'status': 'queued',
        'dir': 'specialists/math/div',
    },
    'general_math': {
        'ops': ['=', 'math', 'calculate', 'solve'],
        'description': 'General arithmetic (any operation)',
        'status': 'ready',
        'dir': 'specialists/math/general',
    },
    'language_general': {
        'ops': ['speak', 'say', 'write', 'sentence', 'translate', 'language', 'word', 'phrase'],
        'description': 'Language generation via AlphaZero self-play',
        'status': 'queued',
        'dir': 'specialists/language/general',
    },
    'code_general': {
        'ops': ['code', 'program', 'function', 'algorithm', 'sort', 'fibonacci', 'palindrome', 'python', 'script'],
        'description': 'Code generation via AlphaZero self-play (3 stages)',
        'status': 'queued',
        'dir': 'specialists/code/general',
    },
    # ═══════════════════════════════════════════════════════════
    # 6-STAGE COGNITIVE SYLLABUS
    # ═══════════════════════════════════════════════════════════
    'pattern_specialist': {
        'ops': ['reverse', 'palindrome', 'count', 'next', 'alternate', 'pattern', 'sequence'],
        'description': 'Stage 2: String patterns, reversal, palindrome, sequence completion',
        'status': 'queued',
        'dir': 'specialists/patterns/general',
    },
    'taxonomy_specialist': {
        'ops': ['is_a', 'chain', 'common', 'taxonomy', 'category', 'hierarchy'],
        'description': 'Stage 3: Is-A logic, set theory, Z3-verified taxonomy',
        'status': 'queued',
        'dir': 'specialists/taxonomy/general',
    },
    'state_specialist': {
        'ops': ['I have', 'How many', 'state', 'track', 'narrative', 'inventory'],
        'description': 'Stage 4: State tracking via English-to-code sandbox',
        'status': 'queued',
        'dir': 'specialists/state/general',
    },
    'grammar_specialist': {
        'ops': ['grammar', 'correct sentence', 'subject verb', 'grammatical'],
        'description': 'Stage 5: Grammar as game rules, CFG evaluation',
        'status': 'queued',
        'dir': 'specialists/grammar/general',
    },
    'dialectics_specialist': {
        'ops': ['philosophy', 'ethics', 'meaning', 'wisdom', 'truth', 'debate', 'socratic'],
        'description': 'Stage 6: Socratic philosophy, ethical reasoning, ebook synthesis',
        'status': 'queued',
        'dir': 'specialists/dialectics/general',
    },
    # Auto-trainable conversational skills
    'greeting': {
        'ops': ['hello', 'hi', 'hey', 'greetings'],
        'description': 'Can greet users and respond to greetings',
        'status': 'queued',
        'dir': 'specialists/greeting',
    },
    'explanation_question': {
        'ops': ['why does', 'how does', 'how do', 'how can', 'how is', 'how was', 'where does', 'where is', 'where do', 'when does', 'when did', 'what causes', 'how come'],
        'description': 'Can answer explanation questions about the AI',
        'status': 'queued',
        'dir': 'specialists/explanation_question',
    },
    'definition_question': {
        'ops': ['what is', 'who is', 'define'],
        'description': 'Can answer definition questions',
        'status': 'queued',
        'dir': 'specialists/definition_question',
    },
    'conversation': {
        'ops': ['tell me', 'joke', 'story', 'sing', 'how are you'],
        'description': 'Can engage in general conversation',
        'status': 'queued',
        'dir': 'specialists/conversation',
    },
    'capability_question': {
        'ops': ['what can you', 'what do you', 'capabilities', 'skills'],
        'description': 'Can describe its own capabilities',
        'status': 'queued',
        'dir': 'specialists/capability_question',
    },
    'question': {
        'ops': ['?', 'what', 'who', 'when', 'where', 'why', 'how'],
        'description': 'Can answer general questions',
        'status': 'queued',
        'dir': 'specialists/question',
    },
    'unknown': {
        'ops': [],
        'description': 'Generic catch-all for any unknown query',
        'status': 'queued',
        'dir': 'specialists/unknown',
    },
}

# Per-specialist generation config
SPECIALIST_CONFIG = {
    'greeting':             {'temp': 0.0, 'max_tokens': 50, 'max_seq': 128, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1000},
    'capability_question':  {'temp': 0.0, 'max_tokens': 60, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
    'explanation_question': {'temp': 0.0, 'max_tokens': 70, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
    'definition_question':  {'temp': 0.0, 'max_tokens': 60, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 600},
    'conversation':         {'temp': 0.0, 'max_tokens': 70, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
    'question':             {'temp': 0.0, 'max_tokens': 60, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
    'unknown':              {'temp': 0.0, 'max_tokens': 40, 'max_seq': 96,  'd_model': 64,  'n_layers': 3, 'n_heads': 4, 'd_ff': 128, 'steps': 500},
    'translation':          {'temp': 0.0, 'max_tokens': 60, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
}

# ─── Validation History ──────────────────────────────────────────
# Module-level store for tracking validation loss across training runs.
# Structure: {intent: [{'loss': float, 'level': int, 'timestamp': float}, ...]}
_VALIDATION_HISTORY: dict[str, list[dict]] = {}

def _get_validation_history(intent: str) -> list[dict]:
    """Get the validation loss history for a given intent.

    Args:
        intent: The specialist intent name.

    Returns:
        List of dicts with keys 'loss', 'level', 'timestamp'.
    """
    return _VALIDATION_HISTORY.get(intent, [])


def _store_validation_loss(intent: str, loss: float, level: int) -> None:
    """Record a validation loss measurement for an intent.

    Maintains a rolling window of the last 5 measurements per intent
    to support plateau detection.

    Args:
        intent: The specialist intent name.
        loss: The validation loss value.
        level: The training level at which this was measured.
    """
    if intent not in _VALIDATION_HISTORY:
        _VALIDATION_HISTORY[intent] = []
    _VALIDATION_HISTORY[intent].append({
        'loss': loss,
        'level': level,
        'timestamp': time.time(),
    })
    # Keep only the last 10 entries
    if len(_VALIDATION_HISTORY[intent]) > 10:
        _VALIDATION_HISTORY[intent] = _VALIDATION_HISTORY[intent][-10:]


def scale_config(intent, level=0):
    """Scale model config by level. Each level increases d_model aggressively."""
    base = SPECIALIST_CONFIG.get(intent, SPECIALIST_CONFIG['greeting'])
    if level == 0:
        return dict(base)
    scaled = dict(base)
    # Aggressive scaling: +32 d_model, +500 steps per level
    scaled['d_model'] = min(base['d_model'] + level * 32, 384)
    scaled['d_ff'] = scaled['d_model'] * 2
    scaled['n_heads'] = max(4, scaled['d_model'] // 16)
    scaled['n_layers'] = min(base['n_layers'] + (level >= 3), 6)
    scaled['steps'] = base['steps'] + level * 500
    scaled['max_seq'] = min(base['max_seq'] + level * 32, 320)
    return scaled


def scale_config_with_validation(
    intent: str,
    level: int = 0,
    validation_loss: float | None = None,
) -> dict:
    """Scale model config with validation-loss-aware plateau detection.

    Extends ``scale_config`` by adding a validation check: if the
    validation loss is still dropping significantly (>10%% improvement
    since last check), the current scale is kept -- scaling is *deferred*
    until the loss plateaus (change < 5%% over the last 3 checks).

    This prevents unnecessary model scaling when the current architecture
    is still effectively learning from existing data, saving compute and
    avoiding premature capacity increases.

    Args:
        intent: The specialist intent name (used to look up history).
        level: Requested training level (may be reduced if plateau not
            reached).
        validation_loss: The most recent validation loss measurement.
            If ``None``, the function falls back to the original
            ``scale_config`` behaviour.

    Returns:
        Scaled config dict (same keys as ``scale_config``), possibly
        with a reduced level if validation loss is still dropping.
    """
    if validation_loss is None:
        # No validation data -- fall back to original behaviour
        return scale_config(intent, level)

    # Record this measurement
    _store_validation_loss(intent, validation_loss, level)
    history = _get_validation_history(intent)

    # Need at least 2 measurements to detect improvement
    if len(history) < 2:
        return scale_config(intent, level)

    # --- Plateau detection ---
    # Check if loss is still dropping (>10% improvement since last check)
    prev_loss = history[-2]['loss']
    current_loss = history[-1]['loss']

    if prev_loss > 0 and current_loss < prev_loss:
        improvement = (prev_loss - current_loss) / prev_loss
        if improvement > 0.10:
            # Loss is dropping faster than 10% -- keep current scale,
            # do NOT increase capacity
            effective_level = max(0, level - 1)
            debug(
                f"validation: {intent} loss dropped {improvement*100:.1f}% "
                f"({prev_loss:.4f}->{current_loss:.4f}), "
                f"holding scale at level {effective_level}"
            )
            return scale_config(intent, effective_level)

    # --- Multi-check plateau detection ---
    # Only increase d_model/steps when change is < 5% over 3 checks
    if len(history) >= 4:
        last_three = history[-4:-1]  # The 3 checks *before* current
        if all(l['loss'] > 0 for l in last_three):
            changes = []
            for i in range(1, len(last_three)):
                prev = last_three[i - 1]['loss']
                curr = last_three[i]['loss']
                if prev > 0:
                    changes.append(abs(curr - prev) / prev)
            if changes and all(c < 0.05 for c in changes):
                # Plateau confirmed -- allow full scaling
                debug(
                    f"validation: {intent} plateau detected "
                    f"(change < 5%% over 3 checks), allowing full scale"
                )
                return scale_config(intent, level)

        # Not plateaued yet -- check the most recent pair
        prev_prev = history[-3]['loss']
        prev = history[-2]['loss']
        curr = history[-1]['loss']
        if prev > 0 and prev_prev > 0:
            recent_change_1 = abs(curr - prev) / prev
            recent_change_2 = abs(prev - prev_prev) / prev_prev
            if recent_change_1 < 0.05 and recent_change_2 < 0.05:
                # Recent changes are small -- allow scaling
                return scale_config(intent, level)

    # Default: conservative -- don't scale more than requested
    return scale_config(intent, max(0, level - 1))

def load_dataset(intent):
    """Load QA pairs from datasets/<intent>.json. Returns list of (question, answer)."""
    path = Path(f"datasets/{intent}.json")
    if path.exists():
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            return [(q, a) for q, a in data] if data else []
        except Exception as e:
            debug(f"load_dataset {intent}: {e}")
            return []
    return []

# Future domains to add:
#   'biology': {'ops': ['cell', 'dna', 'organism', 'photosynthesis'], 'status': 'queued'}
#   'spelling': {'ops': ['spell', 'reverse', 'capitalize'], 'status': 'queued'}
#   'logic': {'ops': ['if', 'then', 'and', 'or', 'not'], 'status': 'queued'}


# ─── Skill Detection ───────────────────────────────────────────────

def detect_skill(prompt: str) -> tuple:
    """Detect which skill a prompt needs. Returns (skill_name, confidence)."""
    import re
    prompt_lower = prompt.lower()
    for name, info in SKILL_REGISTRY.items():
        for op in info['ops']:
            # Word-boundary check: 'hi' should match 'hi' but not 'machine'
            if re.search(r'\b' + re.escape(op) + r'\b', prompt_lower) or re.search(r'\b' + re.escape(op) + r'\b', prompt):
                return name, 0.9
    return None, 0.0


# ─── Model Loader ──────────────────────────────────────────────────

class SkillManager:
    """Loads and manages skill models."""

    def __init__(self, base_dir='.'):
        self.base = Path(base_dir)
        self.models = {}
        self.tokenizers = {}
        self.device = 'cpu'
        self.training_queue = {}  # intent -> True if training in progress
        self.training_progress = {}  # intent -> {step, total, loss, status}
        self.skill_levels = {}  # intent -> level (auto-increments on retrain)
        self._query_count = {}  # intent -> query counter for auto-retrain
        self._validation_losses = {}  # intent -> most recent validation loss (float)
        self.pending_question = None  # {prompt, intent, timestamp} — last unanswered question
        self.last_answer = None  # {answer, prompt, skill} — last answer given
        # Shared base model + LoRA adapters (for USE_LORA_FOR_CHAT mode)
        self._base_model = None
        self._base_tokenizer = None
        self._base_config = None
        self._lora_adapters = {}  # intent -> list of LoRALayer references
        self._base_training_lock = threading.Lock()
        # Task queue for background training (replaces raw threading)
        from egefalos.task_queue import TaskQueue
        self._task_queue = TaskQueue(num_workers=1)
        self._task_queue.register("train_intent")(self._train_intent_worker_task)
        # ── Replay Buffer & PII Scrubbing ──
        from egefalos.replay_buffer import ReplayBuffer
        self.replay_buffer = ReplayBuffer(max_size=200)
        self._load_existing()

    def _load_existing(self):
        """Load all currently trained skills."""
        for name, info in SKILL_REGISTRY.items():
            ckpt_path = self._find_checkpoint(info['dir'])
            if ckpt_path:
                try:
                    tok_path = Path(info['dir']) / 'tokenizer.json'
                    # Try BPE tokenizer first for chat skills, fall back to MathTokenizer
                    tok = None
                    try:
                        from tabula_rasa.bpe_tokenizer import BPETokenizer
                        if tok_path.exists():
                            tok = BPETokenizer.load(str(tok_path))
                    except Exception:
                        pass
                    if tok is None:
                        if not tok_path.exists():
                            tok_path = Path('specialists/math/general/tokenizer.json')
                        tok = MathTokenizer.load(str(tok_path))
                    cfg = Config()
                    # Use saved model config if available (chat specialists have different arch)
                    state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    mc = state.get('model_config', {})
                    if mc:
                        cfg.d_model = mc.get('d_model', cfg.d_model)
                        cfg.n_layers = mc.get('n_layers', cfg.n_layers)
                        cfg.n_heads = mc.get('n_heads', cfg.n_heads)
                        cfg.d_ff = mc.get('d_ff', cfg.d_ff)
                        cfg.max_seq_len = mc.get('max_seq_len', cfg.max_seq_len)
                    cfg.vocab_size = tok.vocab_size
                    tok.max_seq_len = cfg.max_seq_len
                    model = MathTransformer(cfg)
                    model.load_state_dict(state['model_state_dict'])
                    # Track level from saved checkpoint
                    level = 0
                    if mc:
                        level = mc.get('level', 0)
                    self.skill_levels[name] = level
                    model.eval()
                    model.to(self.device)
                    self.models[name] = model
                    self.tokenizers[name] = tok
                    info['status'] = 'ready'
                    params = count_parameters(model)
                    acc = state.get('acc', '?')
                    print(f'  [*] Loaded skill: {name} ({params:,} params, acc={acc}%)')
                except Exception as e:
                    print(f'  [!] Failed to load {name}: {e}')

        # After loading full-model skills, try to discover saved LoRA adapters
        self._load_existing_lora()

        # Replay saved corrections from persistent memory
        self._replay_memory()

    def _load_existing_lora(self):
        """Load previously-trained LoRA adapters into the shared base model."""
        if not USE_LORA_FOR_CHAT:
            return
        lora_dirs = sorted(Path('specialists').glob('*/lora.pt'))
        if not lora_dirs:
            return
        # Ensure shared base model exists before loading adapters
        try:
            model = self._ensure_shared_base()
        except Exception:
            return  # Base not available, skip LoRA loading
        for lora_path in lora_dirs:
            intent = lora_path.parent.name
            if intent.startswith('_'):
                continue  # skip internal dirs
            try:
                if intent not in self.models:
                    # First time: apply LoRA wrappers, then load weights
                    has_lora = any(
                        isinstance(m, LoRALayer) for m in model.modules()
                    )
                    if not has_lora:
                        apply_lora_to_model(model, rank=8, alpha=1.0)
                    layers = switch_lora_adapters(model, str(lora_path))
                    self._lora_adapters[intent] = layers
                    self.models[intent] = model
                    self.tokenizers[intent] = self._base_tokenizer
                    level = 0
                    # Try to read level from saved best.pt metadata
                    meta_path = lora_path.parent / 'best.pt'
                    if meta_path.exists():
                        try:
                            meta = torch.load(
                                str(meta_path), map_location='cpu', weights_only=True
                            )
                            level = meta.get('model_config', {}).get('level', 0)
                        except Exception:
                            pass
                    self.skill_levels[intent] = level
                    from tabula_rasa.bpe_tokenizer import BPETokenizer
                    tok_path = lora_path.parent / 'tokenizer.json'
                    tokenizer = (
                        BPETokenizer.load(str(tok_path))
                        if tok_path.exists()
                        else self._base_tokenizer
                    )
                    self.tokenizers[intent] = tokenizer
                    SKILL_REGISTRY.setdefault(intent, {}).update(
                        {'status': 'ready', 'dir': str(lora_path.parent)}
                    )
                    print(
                        f'  [*] LoRA adapter: {intent} ({sum(p.numel() for p in layers[0].parameters() if p.requires_grad):,} lora params)'
                    )
                else:
                    # Model already registered (was full-model) — skip
                    pass
            except Exception as e:
                print(f'  [!] Failed to load LoRA adapter {intent}: {e}')

    def _ensure_shared_base(self):
        """Lazily create the shared base model and character-level tokenizer."""
        if self._base_model is not None:
            return self._base_model

        from tabula_rasa.bpe_tokenizer import BPETokenizer
        from tabula_rasa.config import Config
        from tabula_rasa.model import MathTransformer, count_parameters

        # Character-level tokenizer (96 tokens) — universal, works for any text
        tok = BPETokenizer()

        cfg = Config()
        cfg.d_model = 128
        cfg.n_layers = 4
        cfg.n_heads = 4
        cfg.d_ff = 512
        cfg.vocab_size = tok.vocab_size
        cfg.max_seq_len = 192  # char-level needs more room than BPE
        cfg.use_loss_masking = True
        cfg.use_reversed = False
        cfg.dropout = 0.1

        tok.max_seq_len = cfg.max_seq_len
        model = MathTransformer(cfg)

        # Save base model and tokenizer
        base_dir = Path('specialists/_base')
        base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'd_model': cfg.d_model,
                    'n_layers': cfg.n_layers,
                    'n_heads': cfg.n_heads,
                    'd_ff': cfg.d_ff,
                    'max_seq_len': cfg.max_seq_len,
                    'vocab_size': cfg.vocab_size,
                },
            },
            base_dir / 'model.pt',
        )
        tok.save(str(base_dir / 'tokenizer.json'))

        self._base_model = model
        self._base_tokenizer = tok
        self._base_config = cfg

        params = count_parameters(model)
        debug(f'shared base: {params:,} params, {tok.vocab_size} tokens')
        print(
            f'  [*] Shared base model ready ({params:,} params, {tok.vocab_size} tokens)'
        )
        return model

    def _switch_lora_adapter(self, intent):
        """Switch the shared base model to a specific intent's LoRA adapter."""
        if not USE_LORA_FOR_CHAT:
            return False
        if self._base_model is None:
            return False

        # Already loaded for this intent — just set eval mode
        if intent in self._lora_adapters:
            self._base_model.eval()
            return True

        # Try loading from disk
        lora_path = Path(f'specialists/{intent}/lora.pt')
        if not lora_path.exists():
            return False

        try:
            has_lora = any(
                isinstance(m, LoRALayer) for m in self._base_model.modules()
            )
            if not has_lora:
                apply_lora_to_model(self._base_model, rank=8, alpha=1.0)
            layers = switch_lora_adapters(self._base_model, str(lora_path))
            self._lora_adapters[intent] = layers
            self._base_model.eval()
            return True
        except Exception as e:
            debug(f'switch_lora_adapter({intent}): {e}')
            return False

    def _replay_memory(self):
        """Replay saved corrections from persistent memory."""
        try:
            from egefalos.memory import get_all_corrections, save_correction
            corrections = get_all_corrections()
            if not corrections:
                return
            # Group by skill
            by_skill = {}
            for c in corrections:
                s = c.get('skill', 'general')
                if s not in by_skill:
                    by_skill[s] = []
                by_skill[s].append((c['expr'], c['answer']))
            for skill, pairs in by_skill.items():
                # Find the model for this skill
                model = self.models.get(skill)
                if model is None:
                    # Try general_math as fallback
                    model = self.models.get('general_math')
                if model:
                    # Train on corrections
                    model.train()
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    import time as _t
                    for expr, ans in pairs:
                        text = f'{expr}={ans}'
                        tok = self.tokenizers.get(skill) or self.tokenizers.get('general_math')
                        if tok is None:
                            continue
                        ids = tok.encode(text, add_special_tokens=True)
                        if len(ids) > 64:
                            ids = ids[:64]
                        padded = ids + [tok.pad_id] * (64 - len(ids))
                        x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0)
                        y = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0)
                        for _ in range(3):
                            opt.zero_grad()
                            _, loss = model(x, y)
                            loss.backward()
                            opt.step()
                    model.eval()
                    print(f'  [*] Replayed {len(pairs)} corrections on {skill}')
        except Exception as e:
            print(f'  [!] Memory replay skipped: {e}')

    def _find_checkpoint(self, dir_name):
        d = Path(dir_name)
        for ckpt in ['best.pt', 'final.pt', 'finetuned.pt']:
            p = d / ckpt
            if p.exists():
                return str(p)
        # Fallback to old math-transformer location
        old_base = Path('/c/Users/Admin/math-transformer')
        old_d = old_base / dir_name
        for ckpt in ['best.pt', 'final.pt', 'finetuned.pt']:
            p = old_d / ckpt
            if p.exists():
                return str(p)
        return None

    @property
    def known_skills(self) -> list:
        return [n for n, i in SKILL_REGISTRY.items() if i['status'] == 'ready']

    @property
    def queued_skills(self) -> list:
        return [n for n, i in SKILL_REGISTRY.items() if i['status'] == 'queued']

    @property
    def status_summary(self) -> list:
        return [{'name': n, 'status': i['status'], 'description': i['description']}
                for n, i in SKILL_REGISTRY.items()]

    @property
    def datasets_status(self) -> dict:
        """List all datasets with training/model status for the dashboard."""
        import json as _json
        from pathlib import Path
        datasets_dir = Path('datasets')
        result = {}
        if datasets_dir.exists():
            for ds_file in sorted(datasets_dir.glob('*.json')):
                name = ds_file.stem
                try:
                    data = _json.loads(ds_file.read_text(encoding='utf-8'))
                    if isinstance(data, dict):
                        # Knowledge base format: {category: [[q, a], ...]}
                        pairs = []
                        for cat, entries in data.items():
                            for q, a in entries:
                                pairs.append((q, a))
                    else:
                        # Dataset format: [[q, a], ...]
                        pairs = [(q, a) for q, a in data] if data else []
                    pair_count = len(pairs)
                    has_real_answers = any(
                        'learning about' not in a.lower() and len(a) > 10
                        for q, a in pairs
                    )
                except Exception:
                    pair_count = 0
                    has_real_answers = False
                is_trained = name in self.models
                is_training = name in self.training_queue
                level = self.skill_levels.get(name, 0)
                params = 0
                if is_trained and name in self.models:
                    params = sum(p.numel() for p in self.models[name].parameters())
                result[name] = {
                    'pairs': pair_count,
                    'has_real_answers': has_real_answers,
                    'trained': is_trained,
                    'training': is_training,
                    'level': level,
                    'params': params,
                }
        return result

    @torch.no_grad()
    def ask(self, prompt: str, temperature=0.3, top_k=5, skill=None) -> dict:
        """Route a question to the right skill.
        If `skill` is provided and loaded, force-use that specialist.
        """
        # ── PII Scrubbing ────────────────────────────────────────
        from egefalos.pii_scrubber import scrub_pii
        clean_prompt = scrub_pii(prompt)
        if clean_prompt != prompt:
            debug(f"ask: PII detected and scrubbed in prompt")
            prompt = clean_prompt  # Use scrubbed version downstream
        # ─────────────────────────────────────────────────────────
        force_skill = skill if (skill and skill in self.models) else None
        if force_skill:
            skill = force_skill
            confidence = 0.9
        else:
            skill, confidence = detect_skill(prompt)
        debug(f"ask: prompt={prompt!r} detect_skill={skill!r} conf={confidence}")

        # Multi-specialist orchestration: decompose complex questions
        try:
            from tabula_rasa.orchestrator import SpecialistOrchestrator
            orch = SpecialistOrchestrator(self)
            orch_result = orch.answer(prompt)
            if orch_result is not None:
                return orch_result
        except Exception:
            pass

        # If detect_skill has a confident match for conversation, keep it (don't drop to intent detection)
        if skill is None and confidence >= 0.5:
            # Re-run detect_skill to check if it was conversation
            ds_skill, ds_conf = detect_skill(prompt)
            if ds_skill == 'conversation' and ds_conf >= 0.5:
                skill = 'conversation'
        
        # Conversational learning: if user follows up with an answer to a pending question
        if self.pending_question:
            result = self.learn_from_followup(prompt)
            if result.get('learned'):
                self.last_answer = result
                return {
                    'prompt': prompt, 'answer': None, 'knows': True,
                    'skill': result['intent'],
                    'skill_description': SKILL_REGISTRY.get(result['intent'], {}).get('description', ''),
                    'time_ms': 0, 'status': 'learning',
                    'confidence': 100.0, 'is_confident': True,
                    'message': f"Thanks! I've learned that {result['question']} = {result['answer']}",
                    'training_info': None,
                }

        # If detected skill is not loaded, don't fall back to math for chat skills — auto-train instead
        if skill is not None and skill not in self.models:
            # Chat skills should auto-train instead of falling back to math
            chat_skills = {'greeting', 'explanation_question', 'definition_question', 'conversation', 'capability_question', 'question', 'unknown', 'translation'}
            if skill in chat_skills:
                skill = None  # Will trigger auto-training below
            elif 'general_math' in self.models:
                skill = 'general_math'
            else:
                for s in self.known_skills:
                    skill = s
                    break

        if skill is None or skill not in self.models:
            # No skill knows this — analyze what's needed
            debug(f"ask: no skill loaded for {skill!r}, entering intent detection")
            prompt_lower = prompt.lower()

            # ─── Character/script analysis ───
            known_chars = set('0123456789+-*/=().% ')
            unknown_chars = sorted(set(c for c in prompt if c not in known_chars))

            has_latin = any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in prompt)
            has_greek = any('\u0380' <= c <= '\u03FF' or '\u0400' <= c <= '\u04FF' for c in prompt)
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in prompt)
            has_cjk = any('\u4E00' <= c <= '\u9FFF' for c in prompt)
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in prompt)
            has_math_ops = any(c in '+-*/=' for c in prompt)
            has_punctuation = any(c in '?!:;,.\'"@#' for c in prompt)

            # Determine scripts needed
            scripts_needed = []
            tokenizer_expansions = []
            if has_greek:
                scripts_needed.append('Greek')
                tokenizer_expansions.append('Greek alphabet (α-ω, Α-Ω)')
            if has_cyrillic:
                scripts_needed.append('Cyrillic')
                tokenizer_expansions.append('Cyrillic alphabet')
            if has_latin:
                scripts_needed.append('Latin')
                tokenizer_expansions.append('A-Z, a-z')
            if has_cjk:
                scripts_needed.append('CJK characters')
                tokenizer_expansions.append('Chinese/Japanese/Korean')
            if has_arabic:
                scripts_needed.append('Arabic')
                tokenizer_expansions.append('Arabic script')
            if has_punctuation:
                tokenizer_expansions.append('punctuation (!?.,:;)')

            # Handle conversation specially — use KB retrieval even if not trained
            ds_skill, ds_conf = detect_skill(prompt)
            if ds_skill == 'conversation' and ds_conf >= 0.5:
                ans_ret, meta_ret, score = self._retrieve_answer('conversation', prompt)
                if ans_ret and len(ans_ret) > 5 and 'learning about' not in ans_ret:
                    ans_ret = f"[HERMES-6-16-2026] {ans_ret}"
                    return {'prompt': prompt, 'answer': ans_ret, 'knows': True, 'skill': 'conversation', 'skill_description': SKILL_REGISTRY.get('conversation', {}).get('description', ''), 'time_ms': 0, 'status': 'answered', 'confidence': 100.0, 'is_confident': True, 'message': None, 'training_info': meta_ret}
            
            # ─── Intent detection ───
            intent = 'unknown'
            if any(phrase in prompt_lower for phrase in ['translate', 'what does', 'mean', 'say in']):
                intent = 'translation'
            elif any(phrase in prompt_lower for phrase in ['hi', 'hello', 'hey', 'γεια', 'hola', 'bonjour', 'ciao']):
                intent = 'greeting'
            elif prompt_lower.startswith('what is') or prompt_lower.startswith('who is') or prompt_lower.startswith('what are'):
                intent = 'definition_question'
            elif 'what can you' in prompt_lower or 'what do you' in prompt_lower or prompt_lower.startswith('do you') or prompt_lower.startswith('can you') or prompt_lower.startswith('are you'):
                intent = 'capability_question'
            elif 'how are you' in prompt_lower:
                intent = 'conversation'
                debug(f"DEBUG_INTENT: matched how are you -> conversation")
            elif prompt_lower.startswith('how do') or prompt_lower.startswith('how does') or prompt_lower.startswith('how can') or prompt_lower.startswith('how was') or prompt_lower.startswith('how come') or prompt_lower.startswith('why do') or prompt_lower.startswith('why does') or prompt_lower.startswith('when do') or prompt_lower.startswith('when does') or prompt_lower.startswith('when did') or prompt_lower.startswith('where do') or prompt_lower.startswith('where does') or prompt_lower.startswith('where is'):
                intent = 'explanation_question'
            elif '?' in prompt:
                intent = 'question'
            elif any(phrase in prompt_lower for phrase in ['translate', 'what does', 'mean', 'say in']):
                intent = 'translation'
            elif not has_math_ops and any(c.isalpha() for c in prompt):
                intent = 'conversation'
            debug(f"ask: intent={intent!r}")

            # ─── Build plan ───
            plan_parts = []

            # Tokenizer step
            if tokenizer_expansions:
                plan_parts.append(f'Add {", ".join(tokenizer_expansions[:3])} to tokenizer')
            if unknown_chars:
                sample = [c for c in unknown_chars if not c.isspace()][:6]
                plan_parts.append(f'Add {len(unknown_chars)} new characters: {repr(sample)}')

            # Skill training step
            skill_map = {
                'greeting': ('greeting_agent', 'Train greetings specialist: respond to hi/hello with a greeting'),
                'definition_question': ('qa_specialist', 'Train Q&A specialist: answer "what is X" questions about this topic'),
                'capability_question': ('self_awareness', 'Train self-awareness: answer questions about your own abilities'),
                'explanation_question': ('explanation_specialist', 'Train explanation specialist: answer how/why questions'),
                'question': ('conversation_agent', 'Train conversation specialist: general question answering'),
                'translation': ('translation_specialist', 'Train translation specialist: translate between languages'),
                'conversation': ('language_agent', 'Train language specialist: understand and respond to text'),
            }

            if intent in skill_map:
                suggested_skill_name, plan_text = skill_map[intent]
                plan_parts.append(plan_text)
            elif has_math_ops:
                suggested_skill_name = 'math_specialist'
                plan_parts.append('Train math specialist on this operation type')

                # Check which math skills are missing
                missing_ops = []
                for name, info in SKILL_REGISTRY.items():
                    if info['status'] == 'queued' and name != 'general_math':
                        missing_ops.append(name)
                if missing_ops:
                    plan_parts.append(f'Queue: {", ".join(missing_ops)}')
            elif scripts_needed:
                suggested_skill_name = f'{scripts_needed[0].lower()}_language_agent'
                plan_parts.append(f'Train {scripts_needed[0]} language specialist')
            else:
                suggested_skill_name = None

            # ─── Build response ───
            char_type_labels = []
            if has_greek: char_type_labels.append('Greek')
            if has_latin: char_type_labels.append('Latin')
            if has_cyrillic: char_type_labels.append('Cyrillic')
            if has_cjk: char_type_labels.append('CJK')
            if has_arabic: char_type_labels.append('Arabic')

            suggested_skills_ready = [
                name for name, info in SKILL_REGISTRY.items()
                if info['status'] == 'ready' and any(op in prompt_lower for op in info['ops'])
            ]

            training_plan = ' → '.join(plan_parts) if plan_parts else 'Define training dataset for this domain'

            # ─── Auto-train for any unknown intent ───
            # Check if we already have the model or are training it
            debug(f"ask: intent={intent!r} in_models={intent in self.models} in_queue={intent in self.training_queue}")
            if intent in self.models:
                # Model already trained — try retrieval first, fall back to neural
                ans_ret, meta_ret, score = self._retrieve_answer(intent, prompt)
                # Check if user's question is novel — append to dataset & retrain
                is_novel = True
                pairs = load_dataset(intent)
                if pairs:
                    existing_questions = {q.lower().strip() for q, a in pairs}
                    is_novel = prompt.lower().strip() not in existing_questions
                if is_novel and intent not in self.training_queue:
                    debug(f"ask: novel question for known {intent}, appending & retraining")
                    self._auto_train_intent(intent, prompt)
                elif not is_novel and intent not in self.training_queue:
                    # Retrain counter for known questions
                    self._query_count[intent] = self._query_count.get(intent, 0) + 1
                    if self._query_count[intent] >= 3:
                        self._query_count[intent] = 0
                        level = self.skill_levels.get(intent, 0)
                        if level < 5:
                            debug(f"ask: retraining {intent} (level {level} -> {level+1})")
                            self._auto_train_intent(intent, prompt, retrain=True)
                if score >= 0.3:
                    debug(f"ask: retrieval {intent} match={score:.2f} -> {ans_ret!r}")
                    return {
                        'prompt': prompt, 'answer': ans_ret, 'knows': True,
                        'skill': intent,
                        'skill_description': SKILL_REGISTRY.get(intent, {}).get('description', ''),
                        'time_ms': 0, 'status': 'answered',
                        'confidence': 100.0, 'is_confident': True, 'message': None,
                        'training_info': meta_ret,
                    }
                # Low retrieval → use neural model
                # Switch to the correct LoRA adapter if using shared base
                if USE_LORA_FOR_CHAT and intent in self._lora_adapters:
                    self._switch_lora_adapter(intent)
                model = self.models[intent]
                tok = self.tokenizers[intent]
                t0 = time.time()
                _sc = scale_config(intent, self.skill_levels.get(intent, 0))
                inp = prompt if prompt.endswith('=') else prompt + '='
                full = model.generate(tok, inp, max_new_tokens=_sc['max_tokens'], temperature=_sc['temp'], top_k=0)
                elapsed = time.time() - t0
                if '=' in full:
                    ans = full.split('=', 1)[1].strip()
                else:
                    ans = full.strip()
                debug(f"ask: nn {intent} -> {ans!r} ({elapsed*1000:.0f}ms)")
                return {
                    'prompt': prompt, 'answer': ans, 'knows': True,
                    'skill': intent,
                    'skill_description': SKILL_REGISTRY.get(intent, {}).get('description', ''),
                    'time_ms': round(elapsed * 1000), 'status': 'answered',
                    'confidence': 50.0, 'is_confident': True, 'message': None,
                    'training_info': {
                        'level': self.skill_levels.get(intent, 0),
                        'd_model': _sc.get('d_model', '?'),
                        'steps': _sc.get('steps', '?'),
                        'params': sum(p.numel() for p in model.parameters()),
                    },
                }

            if intent in self.training_queue:
                # Try retrieval even during training — knowledge base answers are instant
                if intent != prompt:
                    try:
                        ans_ret, meta_ret, score = self._retrieve_answer(intent, prompt)
                        if score >= 0.3:
                            debug(f"ask: retrieval {intent} during training match={score:.2f}")
                            return {
                                'prompt': prompt, 'answer': ans_ret, 'knows': True,
                                'skill': intent,
                                'skill_description': SKILL_REGISTRY.get(intent, {}).get('description', ''),
                                'time_ms': 0, 'status': 'answered',
                                'confidence': 100.0, 'is_confident': True, 'message': None,
                                'training_info': meta_ret,
                            }
                    except Exception:
                        pass
                return {
                    'answer': None,
                    'knows': False,
                    'message': f"Training a {intent} specialist right now... ask me again in a moment!",
                    'detected_skill': None,
                    'suggested_skills': [],
                    'status': 'training',
                    'time_ms': 0,
                    'analysis': None,
                }

            # Trigger auto-training (background thread, ~2-5s on CPU)
            self._auto_train_intent(intent, prompt)
            self.pending_question = {'prompt': prompt, 'intent': intent, 'timestamp': time.time()}
            return {
                'answer': None,
                'knows': False,
                'message': f"Training a {intent} specialist... ask again in a moment!",
                'detected_skill': None,
                'suggested_skills': [suggested_skill_name] if suggested_skill_name else [],
                'status': 'training',
                'time_ms': 0,
                'analysis': None,
            }

        # Chat skill: retrieval first (instant, 100% accurate), neural only when needed
        if skill in {'greeting', 'capability_question', 'explanation_question', 'definition_question', 'conversation', 'question', 'unknown', 'translation'}:
            ans_ret, meta_ret, score = self._retrieve_answer(skill, prompt)
            # Good retrieval match (>30% word overlap) → return immediately, train for novelty
            if score >= 0.3:
                debug(f"ask: retrieval {skill} match={score:.2f} -> {ans_ret!r}")
                # Check if user's question is novel (not in dataset) — append & auto-retrain if so
                is_novel = True
                pairs = load_dataset(skill)
                if pairs:
                    existing_questions = {q.lower().strip() for q, a in pairs}
                    is_novel = prompt.lower().strip() not in existing_questions
                if is_novel and skill not in self.training_queue:
                    debug(f"ask: novel question for known {skill}, appending & retraining")
                    self._auto_train_intent(skill, prompt)
                elif not is_novel and skill not in self.training_queue:
                    # Auto-retrain: every 3 familiar questions, bump level
                    self._query_count[skill] = self._query_count.get(skill, 0) + 1
                    if self._query_count[skill] >= 3:
                        self._query_count[skill] = 0
                        level = self.skill_levels.get(skill, 0)
                        if level < 5:
                            debug(f"ask: retraining {skill} (level {level} -> {level+1})")
                            self._auto_train_intent(skill, prompt, retrain=True)
                            debug(f"ask: after retrain call, queue={skill in self.training_queue}")
                elif skill not in self.models and skill not in self.training_queue:
                    # Load model failed (e.g. LoRA adapter) — requeue with retrain if known
                    self._auto_train_intent(skill, prompt, retrain=not is_novel)
                return {
                    'prompt': prompt, 'answer': ans_ret, 'knows': True,
                    'skill': skill,
                    'skill_description': SKILL_REGISTRY[skill]['description'],
                    'time_ms': 0, 'status': 'answered',
                    'confidence': 100.0, 'is_confident': True, 'message': None,
                    'training_info': meta_ret,
                }
            # Low retrieval score → try neural model if loaded
            if skill in self.models:
                # Switch to the correct LoRA adapter if using shared base
                if USE_LORA_FOR_CHAT and skill in self._lora_adapters:
                    self._switch_lora_adapter(skill)
                model = self.models[skill]
                tok = self.tokenizers[skill]
                t0 = time.time()
                _sc = scale_config(skill, self.skill_levels.get(skill, 0))
                inp = prompt if prompt.endswith('=') else prompt + '='
                full = model.generate(tok, inp, max_new_tokens=_sc['max_tokens'], temperature=_sc['temp'], top_k=0)
                elapsed = time.time() - t0
                if '=' in full:
                    ans = full.split('=', 1)[1].strip()
                else:
                    ans = full.strip()
                is_repetitive = len(set(ans)) <= 3 and len(ans) >= 8
                if len(ans) >= 5 and not is_repetitive:
                    debug(f"ask: nn {skill} -> {ans!r} ({elapsed*1000:.0f}ms)")
                    return {
                        'prompt': prompt, 'answer': ans, 'knows': True,
                        'skill': skill,
                        'skill_description': SKILL_REGISTRY[skill]['description'],
                        'time_ms': round(elapsed * 1000), 'status': 'answered',
                        'confidence': 50.0, 'is_confident': True, 'message': None,
                        'training_info': {
                            'level': self.skill_levels.get(skill, 0),
                            'd_model': _sc.get('d_model', '?'),
                            'steps': _sc.get('steps', '?'),
                            'params': sum(p.numel() for p in model.parameters()),
                        },
                    }
            # No good answer from either → auto-train and return best retrieval attempt
            self._auto_train_intent(skill, prompt)
            debug(f"ask: auto-train {skill} for novel query: {prompt!r}")
            return {
                'prompt': prompt, 'answer': ans_ret, 'knows': True,
                'skill': skill,
                'skill_description': SKILL_REGISTRY[skill]['description'],
                'time_ms': 0, 'status': 'answered',
                'confidence': 100.0, 'is_confident': True, 'message': None,
                'training_info': meta_ret,
            }
    @torch.no_grad()
    def _retrieve_answer(self, skill, prompt):
        """Retrieval fallback — find best training example by word overlap.
        Returns (answer, meta, score). Low score means poor match.
        """
        pairs = load_dataset(skill)
        prompt_words = set(prompt.lower().split())
        # Remove common stop words so 'what is tantra?' doesn't match 'what is a black hole?' via 'what' 'is'
        stop_words = {'what', 'is', 'are', 'a', 'an', 'the', 'do', 'does', 'did', 'can', 'will', 'would', 'could', 'should', 'may', 'might', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'about', 'like', 'this', 'that', 'it', 'its', 'you', 'your', 'i', 'me', 'my', 'we', 'our', 'they', 'them', 'he', 'she', 'his', 'her'}
        content_words = prompt_words - stop_words
        scored = []
        for q, a in pairs:
            q_words = set(q.lower().split())
            q_content = q_words - stop_words
            if not q_content and not q_words:
                continue
            # Prefer matches on content words (non-stop words)
            if q_content:
                overlap = len(content_words & q_content) / max(len(q_content), 1)
            else:
                overlap = len(prompt_words & q_words) / max(len(q_words), 1)
            # Bonus: if answer contains a content word from question, bump score
            topic_in_answer = any(w in a.lower() for w in content_words if len(w) > 3)
            if topic_in_answer:
                overlap = min(overlap + 0.2, 1.0)
            scored.append((overlap, a, q))
        if not scored:
            scored = [(0, "I don't know about that yet.", "")]
        best_score = max(s[0] for s in scored)
        # Higher threshold: require content word match (best_score > 0) for non-trivial queries
        if best_score <= 0:
            # No content word match — return fallback
            return ("I don't know about that yet.", {
                'level': self.skill_levels.get(skill, 0),
                'd_model': scale_config(skill, self.skill_levels.get(skill, 0)).get('d_model', '?'),
                'steps': scale_config(skill, self.skill_levels.get(skill, 0)).get('steps', '?'),
                'mode': 'retrieval', 'creativity': 0,
            }, 0.0)
        threshold = best_score * 0.5
        candidates = [(a, q) for s, a, q in scored if s >= threshold]
        if not candidates:
            candidates = [(a, q) for s, a, q in scored]
        import random
        best_answer, best_question = random.choice(candidates)
        creativity = round(min(len(candidates) * 11, 100))
        level = self.skill_levels.get(skill, 0)
        sc = scale_config(skill, level)
        return best_answer, {
            'level': level, 'd_model': sc.get('d_model', '?'),
            'steps': sc.get('steps', '?'), 'mode': 'retrieval',
            'creativity': creativity,
        }, best_score

    def learn_from_followup(self, prompt: str) -> dict:
        """Detect if user is providing the answer to a previous unanswered question.
        
        When a user asks something the system doesn't know, the system marks
        the question as 'pending'. If the user's next message is a statement
        (not a question), treat it as the answer, save to dataset, and retrain.
        """
        # Only learn if there's a pending question from the last unanswered ask
        if not self.pending_question:
            return {'learned': False, 'reason': 'no_pending'}

        # Check if this looks like a statement (answer), not a question
        has_question_mark = '?' in prompt
        has_question_words = any(
            prompt.lower().startswith(w) for w in
            ['what', 'who', 'when', 'where', 'why', 'how', 'is ', 'are ', 'do ', 'can ']
        )
        is_question = has_question_mark or has_question_words
        if is_question:
            return {'learned': False, 'reason': 'follow_up_is_question'}

        # Looks like an answer — capture it
        if len(prompt.strip().split()) < 2:
            return {'learned': False, 'reason': 'too_short'}

        pending = self.pending_question
        intent = pending['intent']
        question = pending['prompt']

        # Save the correction to persistent memory
        try:
            from egefalos.memory import save_correction
            save_correction(question, prompt.strip(), skill=intent)
        except Exception:
            pass

        # Append to dataset file
        import json
        from pathlib import Path
        dataset_path = Path(f"datasets/{intent}.json")
        pairs = load_dataset(intent)
        # Remove any existing entry for the same question (keep latest)
        pairs = [(q, a) for q, a in pairs if q.lower().strip() != question.lower().strip()]
        pairs = pairs + [(question, prompt.strip())]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

        debug(f"learn: user taught {intent} -> {question!r} = {prompt!r} ({len(pairs)} pairs)")

        # Trigger retrain
        if intent not in self.training_queue:
            self._auto_train_intent(intent, question)

        self.pending_question = None  # Clear pending
        return {
            'learned': True,
            'question': question,
            'answer': prompt.strip(),
            'intent': intent,
            'total_pairs': len(pairs),
        }

    def _auto_train_intent(self, intent: str, prompt: str, retrain=False):
        """Auto-train a specialist for an intent type in background."""
        if intent in self.training_queue:
            return  # Already training this intent

        # Intent-specific training data
        level = self.skill_levels.get(intent, 0)
        if retrain:
            level += 1
        self.skill_levels[intent] = level
        # Use validation-aware scaling if we have stored validation loss for this intent
        val_loss = self._validation_losses.get(intent, None)
        sc = scale_config_with_validation(intent, level, validation_loss=val_loss)
        debug(f"train: {intent} level={level} val_loss={val_loss} config={sc}")

        # Try to get CPU count
        try:
            import os
            cpu_count = os.cpu_count() or 1
        except:
            cpu_count = 1
        pairs = load_dataset(intent)
        import json
        dataset_path = Path(f"datasets/{intent}.json")

        # Pre-populate replay buffer from existing dataset
        if pairs and intent not in self.replay_buffer.buffer:
            self.replay_buffer.load_from_dataset(intent, pairs)
            debug(f"replay: pre-populated {intent} buffer with {len(pairs)} pairs")

        # Check if user's question is already in the dataset
        existing_questions = {q.lower().strip() for q, a in pairs}
        is_novel = prompt.lower().strip() not in existing_questions

        if is_novel:
            # Generate synthetic dataset using blank-slate resources
            try:
                from tabula_rasa.dataset_generator import generate_dataset
                # Best-match answer for the seed pair
                seed_answer = f"I'm learning about '{prompt}'. Ask me again and I'll improve!"
                gen_pairs = generate_dataset(
                    intent=intent,
                    seed_prompt=prompt,
                    seed_answer=seed_answer,
                    skill_manager=self,
                    max_pairs=10,
                    output_path=str(dataset_path),
                )
                if gen_pairs:
                    pairs = gen_pairs
                    debug(f"auto-dataset: synthetic dataset generated for {intent} ({len(pairs)} pairs)")
                else:
                    pairs = pairs + [(prompt, seed_answer)]
                    dataset_path.parent.mkdir(parents=True, exist_ok=True)
                    dataset_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
                    debug(f"auto-dataset: fallback to single pair for {intent}")
            except Exception as e:
                import traceback
                debug(f"auto-dataset: generator failed ({e}), using basic append")
                # Fallback: basic append with retrieval match
                best_match_answer = None
                if pairs:
                    prompt_words = set(prompt.lower().split())
                    scored = []
                    for q, a in pairs:
                        q_words = set(q.lower().split())
                        if not q_words: continue
                        overlap = len(prompt_words & q_words) / len(q_words)
                        scored.append((overlap, a))
                    if scored:
                        best_score = max(s[0] for s in scored)
                        if best_score > 0.15:
                            candidates = [a for s, a in scored if s >= max(best_score * 0.4, 0.15)]
                            best_match_answer = random.choice(candidates) if candidates else None
                auto_answer = best_match_answer or f"I'm learning about '{prompt}'. Ask me again and I'll improve!"
                pairs = pairs + [(prompt, auto_answer)]
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
                debug(f"auto-dataset: appended novel question ({len(pairs)} total pairs)")
        else:
            debug(f"auto-dataset: question already in datasets/{intent}.json ({len(pairs)} pairs)")

        if not pairs:
            # Auto-create dataset file with the prompt as first training pair
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            auto_answer = f"I'm learning about '{prompt}'. Ask me again and I'll improve!"
            pairs = [(prompt, auto_answer)]
            dataset_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
            debug(f"auto-dataset: created datasets/{intent}.json with 1 pair")
            # Also add to SKILL_REGISTRY so it persists across restarts (in-memory only here)
            if intent not in SKILL_REGISTRY:
                SKILL_REGISTRY[intent] = {
                    'ops': [],
                    'description': f'Auto-created specialist for {intent}',
                    'status': 'queued',
                    'dir': f'specialists/{intent}',
                }

        if retrain:
            # Append current query again as extra training pair
            pairs = pairs + [(prompt, f"I'm learning more about '{prompt}' every time you ask.")]

        self.training_queue[intent] = True
        self.training_progress[intent] = {
            'step': 0, 'total': sc['steps'],
            'loss': 0, 'status': 'starting',
            'config': sc, 'cpu': cpu_count, 't0': time.time(),
        }
        # Enqueue training task via task queue instead of raw threading
        self._task_queue.enqueue("train_intent", intent=intent, pairs=pairs)

    def _train_intent_worker_task(self, args: dict):
        """Wrapper for task queue: accepts a dict and forwards to _train_intent_worker."""
        return self._train_intent_worker(args["intent"], args["pairs"])

    def _train_intent_worker(self, intent: str, pairs: list):
        """Train a tiny chat specialist on intent-specific data.

        When USE_LORA_FOR_CHAT is enabled, trains a LoRA adapter on the
        shared base model instead of a full model from scratch. This is
        ~10x faster and uses ~50x less storage per intent.
        """
        import torch
        debug(f"train: worker starting intent={intent!r} pairs={len(pairs)}")
        # ── Replay Buffer: mix new pairs with old data ───────────
        mixed_pairs = self.replay_buffer.mix(intent, pairs, replay_ratio=0.5)
        if len(mixed_pairs) > len(pairs):
            debug(
                f"train: replay mixed {len(mixed_pairs)} pairs "
                f"({len(pairs)} new + {len(mixed_pairs)-len(pairs)} replayed)"
            )
            pairs = mixed_pairs
        # Store new pairs in replay buffer for future training runs
        for q, a in pairs:
            self.replay_buffer.add(intent, q, a)
        # ──────────────────────────────────────────────────────────
        try:
            prog = self.training_progress.get(intent, {})
            _sc = prog.get('config', {})
            max_steps = prog.get('total', 500)
            learning_rate = 0.001

            if USE_LORA_FOR_CHAT:
                # ════════════════════════════════════════════════════
                # LoRA PATH: train adapter on shared base model
                # ════════════════════════════════════════════════════
                with self._base_training_lock:
                    model = self._ensure_shared_base()
                    tok = self._base_tokenizer  # character-level

                    print(f"  [*] LoRA auto-training {intent} ({len(pairs)} pairs)...")

                    # Ensure LoRA wrappers exist on the shared base (only once)
                    has_lora = any(
                        isinstance(m, LoRALayer) for m in model.modules()
                    )
                    if not has_lora:
                        apply_lora_to_model(model, rank=8, alpha=1.0)

                    set_lora_trainable(model, trainable=True)
                    lora_params = [
                        p for n, p in model.named_parameters() if 'lora_' in n
                    ]
                    opt = torch.optim.AdamW(lora_params, lr=learning_rate)

                    # Encode training data using shared char-level tokenizer
                    xs, ys = [], []
                    for q, a in pairs:
                        text = f"{q}={a}"
                        ids = tok.encode(text, add_special_tokens=True)
                        ids = (ids + [tok.pad_id] * tok.max_seq_len)[
                            : tok.max_seq_len
                        ]
                        xs.append(torch.tensor(ids[:-1]))
                        ys.append(torch.tensor(ids[1:]))

                    # Hold out up to 1 pair for validation (if we have enough data)
                    val_xs, val_ys = None, None
                    if len(pairs) >= 3:
                        # Use the last pair as validation
                        val_xs = xs[-1].unsqueeze(0)
                        val_ys = ys[-1].unsqueeze(0)
                        xs = xs[:-1]
                        ys = ys[:-1]
                    elif len(pairs) >= 2:
                        # Use the last pair as validation
                        val_xs = xs[-1].unsqueeze(0)
                        val_ys = ys[-1].unsqueeze(0)
                        xs = xs[:-1]
                        ys = ys[:-1]

                    if not xs:
                        # Fallback: use all data for training, no validation
                        xs, ys = [], []
                        for q, a in pairs:
                            text = f"{q}={a}"
                            ids = tok.encode(text, add_special_tokens=True)
                            ids = (ids + [tok.pad_id] * tok.max_seq_len)[
                                : tok.max_seq_len
                            ]
                            xs.append(torch.tensor(ids[:-1]))
                            ys.append(torch.tensor(ids[1:]))
                        val_xs, val_ys = None, None

                    x = torch.stack(xs)
                    y = torch.stack(ys)
                    dataset = torch.utils.data.TensorDataset(x, y)
                    loader = torch.utils.data.DataLoader(
                        dataset, batch_size=min(8, len(pairs)), shuffle=True
                    )

                    model.train()
                    for step in range(max_steps):
                        for bx, by in loader:
                            opt.zero_grad()
                            _, loss, _ = model(bx, by)
                            loss.backward()
                            opt.step()
                        if (step + 1) % 25 == 0 or step == 0:
                            p = self.training_progress.get(intent, {})
                            self.training_progress[intent] = {
                                'step': step + 1,
                                'total': max_steps,
                                'loss': round(loss.item(), 4),
                                'status': 'training',
                                'cpu': p.get('cpu'),
                                't0': p.get('t0'),
                            }
                            print(
                                f"  [*] {intent} step {step+1}/{max_steps}, loss={loss.item():.4f}"
                            )

                    model.eval()

                    # Compute validation loss on held-out pair (if available)
                    val_loss_value = None
                    if val_xs is not None and val_ys is not None:
                        with torch.no_grad():
                            _, val_loss, _ = model(val_xs, val_ys)
                            val_loss_value = val_loss.item()
                            self._validation_losses[intent] = val_loss_value
                            debug(
                                f"train: {intent} validation loss={val_loss_value:.4f}"
                            )

                    # Save LoRA adapter (small file — just A and B matrices)
                    save_dir = Path(f"specialists/{intent}")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    # Collect the current LoRA layers from the model
                    current_lora_layers = [
                        m for m in model.modules() if isinstance(m, LoRALayer)
                    ]
                    save_lora_adapters(current_lora_layers, str(save_dir / 'lora.pt'))

                    # Save minimal metadata (not the full model)
                    torch.save(
                        {
                            'step': max_steps,
                            'loss': loss.item(),
                            'model_config': {
                                'd_model': self._base_config.d_model,
                                'n_layers': self._base_config.n_layers,
                                'n_heads': self._base_config.n_heads,
                                'd_ff': self._base_config.d_ff,
                                'max_seq_len': self._base_config.max_seq_len,
                                'lora_rank': 8,
                                'lora_alpha': 1.0,
                                'level': self.skill_levels.get(intent, 0),
                            },
                        },
                        save_dir / 'best.pt',
                    )
                    # Save the shared tokenizer path ref (don't duplicate)
                    tok.save(str(save_dir / 'tokenizer.json'))

                    # Register
                    original_ops = SKILL_REGISTRY.get(intent, {}).get('ops', [])
                    SKILL_REGISTRY[intent] = {
                        'ops': original_ops,
                        'description': f'Auto-trained {intent} specialist',
                        'status': 'ready',
                        'dir': f'specialists/{intent}',
                    }
                    self.models[intent] = model  # reference to shared base
                    self.tokenizers[intent] = tok
                    self._lora_adapters[intent] = current_lora_layers
                    self.training_progress[intent] = {
                        'step': max_steps,
                        'total': max_steps,
                        'loss': round(loss.item(), 4),
                        'status': 'done',
                        'cpu': self.training_progress.get(intent, {}).get('cpu'),
                        't0': self.training_progress.get(intent, {}).get('t0'),
                    }
                    lora_param_count = sum(p.numel() for p in lora_params)
                    debug(
                        f"train: {intent} loRA done, loss={loss.item():.4f} lora_params={lora_param_count}"
                    )
                    print(
                        f"  [*] LoRA-trained {intent} ready ({lora_param_count:,} adapter params)"
                    )

            else:
                # ════════════════════════════════════════════════════
                # ORIGINAL PATH: full model from scratch
                # ════════════════════════════════════════════════════
                from tabula_rasa.bpe_tokenizer import BPETokenizer
                from tabula_rasa.config import Config

                print(f"  [*] Auto-training {intent} specialist ({len(pairs)} pairs)...")

                tok = BPETokenizer()
                texts = []
                for q, a in pairs:
                    texts.append(f"{q}=")
                    texts.append(a)
                tok.learn_bpe_from_texts(texts, num_merges=50, verbose=False)
                print(f"  [*] BPE vocab: {tok.vocab_size} tokens")

                cfg = Config()
                cfg.d_model = _sc.get('d_model', 64)
                cfg.n_layers = _sc.get('n_layers', 3)
                cfg.n_heads = _sc.get('n_heads', 4)
                cfg.d_ff = _sc.get('d_ff', 128)
                cfg.vocab_size = tok.vocab_size
                cfg.max_seq_len = _sc.get('max_seq', 128)
                cfg.batch_size = len(pairs)
                cfg.max_steps = max_steps
                cfg.learning_rate = learning_rate
                cfg.use_reversed = False
                cfg.use_loss_masking = True

                tok.max_seq_len = cfg.max_seq_len
                model = MathTransformer(cfg)
                opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

                xs, ys = [], []
                for q, a in pairs:
                    text = f"{q}={a}"
                    ids = tok.encode(text, add_special_tokens=True)
                    ids = (ids + [tok.pad_id] * cfg.max_seq_len)[:cfg.max_seq_len]
                    xs.append(torch.tensor(ids[:-1]))
                    ys.append(torch.tensor(ids[1:]))

                # Hold out 1 pair for validation (if we have enough data)
                val_xs, val_ys_full = None, None
                if len(pairs) >= 3:
                    val_xs = xs[-1].unsqueeze(0)
                    val_ys_full = ys[-1].unsqueeze(0)
                    xs = xs[:-1]
                    ys = ys[:-1]
                elif len(pairs) >= 2:
                    val_xs = xs[-1].unsqueeze(0)
                    val_ys_full = ys[-1].unsqueeze(0)
                    xs = xs[:-1]
                    ys = ys[:-1]

                if not xs:
                    # Fallback: use all data for training
                    xs, ys = [], []
                    for q, a in pairs:
                        text = f"{q}={a}"
                        ids = tok.encode(text, add_special_tokens=True)
                        ids = (ids + [tok.pad_id] * cfg.max_seq_len)[:cfg.max_seq_len]
                        xs.append(torch.tensor(ids[:-1]))
                        ys.append(torch.tensor(ids[1:]))
                    val_xs, val_ys_full = None, None

                x = torch.stack(xs)
                y = torch.stack(ys)
                dataset = torch.utils.data.TensorDataset(x, y)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=min(8, len(pairs)), shuffle=True
                )

                model.train()
                for step in range(max_steps):
                    for bx, by in loader:
                        opt.zero_grad()
                        _, loss, _ = model(bx, by)
                        loss.backward()
                        opt.step()
                    if (step + 1) % 25 == 0 or step == 0:
                        p = self.training_progress.get(intent, {})
                        self.training_progress[intent] = {
                            'step': step + 1,
                            'total': max_steps,
                            'loss': round(loss.item(), 4),
                            'status': 'training',
                            'cpu': p.get('cpu'),
                            't0': p.get('t0'),
                        }
                        print(
                            f"  [*] {intent} step {step+1}/{max_steps}, loss={loss.item():.4f}"
                        )

                model.eval()

                # Compute validation loss on held-out pair (if available)
                if val_xs is not None and val_ys_full is not None:
                    with torch.no_grad():
                        _, val_loss_full, _ = model(val_xs, val_ys_full)
                        val_loss_value = val_loss_full.item()
                        self._validation_losses[intent] = val_loss_value
                        debug(
                            f"train: {intent} validation loss={val_loss_value:.4f}"
                        )

                save_dir = Path(f"specialists/{intent}")
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'step': max_steps,
                        'model_state_dict': model.state_dict(),
                        'loss': loss.item(),
                        'model_config': {
                            'd_model': cfg.d_model,
                            'n_layers': cfg.n_layers,
                            'n_heads': cfg.n_heads,
                            'd_ff': cfg.d_ff,
                            'max_seq_len': cfg.max_seq_len,
                            'level': self.skill_levels.get(intent, 0),
                        },
                    },
                    save_dir / 'best.pt',
                )
                tok.save(str(save_dir / 'tokenizer.json'))

                original_ops = SKILL_REGISTRY.get(intent, {}).get('ops', [])
                SKILL_REGISTRY[intent] = {
                    'ops': original_ops,
                    'description': f'Auto-trained {intent} specialist',
                    'status': 'ready',
                    'dir': f'specialists/{intent}',
                }
                self.models[intent] = model
                self.tokenizers[intent] = tok
                self.training_progress[intent] = {
                    'step': max_steps,
                    'total': max_steps,
                    'loss': round(loss.item(), 4),
                    'status': 'done',
                    'cpu': self.training_progress.get(intent, {}).get('cpu'),
                    't0': self.training_progress.get(intent, {}).get('t0'),
                }
                params = count_parameters(model)
                debug(f"train: {intent} done, loss={loss.item():.4f} params={params}")
                print(f"  [*] Auto-trained {intent} specialist ready ({params:,} params)")

        except Exception as e:
            import traceback
            with open('auto_train_errors.log', 'a') as f:
                f.write(f"[{intent}] {e}\n")
                traceback.print_exc(file=f)
            print(f"  [!] Auto-train {intent} failed: {e}")
        finally:
            self.training_queue.pop(intent, None)


# ─── Server ────────────────────────────────────────────────────────

manager = None


class TabulaRasaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f'  [*] {args[0]} {args[1]} {args[2]}\n')

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/health':
            self._send_json({
                'status': 'ok',
                'known_skills': manager.known_skills,
                'queued_skills': manager.queued_skills,
                'total_skills': len(SKILL_REGISTRY),
                'system': 'Tabula Rasa AI',
            })
        elif path == '/memory':
            try:
                from egefalos.memory import load_memory
                mem = load_memory()
                self._send_json(mem)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        elif path == '/skills':
            self._send_json({'skills': manager.status_summary})
        elif path == '/training-progress':
            self._send_json({'training': manager.training_progress})
        elif path == '/api/tasks':
            self._send_json({'tasks': manager._task_queue.stats})
        elif path == '/':
            self._send_json({
                'system': 'Tabula Rasa AI',
                'message': 'I learn skills one at a time, from scratch.',
                'known_skills': manager.known_skills,
                'usage': 'POST /ask with {"question": "12+34="}',
            })
        elif path == '/session-memory-count':
            try:
                mem_file = Path('session_memory.json')
                if mem_file.exists():
                    data = json.loads(mem_file.read_text())
                    self._send_json({'total': len(data)})
                else:
                    self._send_json({'total': 0})
            except Exception as e:
                self._send_json({'total': 0, 'error': str(e)})
        elif path == '/datasets':
            self._send_json({'datasets': manager.datasets_status})
        elif path == '/compose':
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            test_q = qs.get('q', [''])[0]
            if test_q:
                try:
                    from tabula_rasa.orchestrator import SpecialistOrchestrator, decompose_question
                    parts = decompose_question(test_q)
                    self._send_json({'original': test_q, 'sub_questions': parts, 'count': len(parts)})
                except Exception as e:
                    self._send_json({'error': str(e)}, 500)
            else:
                self._send_json({'usage': 'GET /compose?q=what+is+X+and+how+does+Y+work'})
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        data = None
        cl = int(self.headers.get('Content-Length', 0))
        if cl > 0:
            try:
                # Read full body (Windows rfile.read(n) may return partial)
                chunks = []
                remaining = cl
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 4096))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b''.join(chunks)
                data = json.loads(raw)
            except:
                pass

        if path == '/ask' or path == '/generate':
            if not data or 'question' not in data and 'prompt' not in data:
                self._send_json({'error': 'Missing "question" or "prompt"'}, 400)
                return
            prompt = data.get('question') or data.get('prompt', '')
            temperature = float(data.get('temperature', 0.3))
            top_k = int(data.get('top_k', 5))
            forced_skill = data.get('skill', None)
            try:
                result = manager.ask(prompt.strip(), temperature, top_k, forced_skill)
                self._send_json(result)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        elif path == '/learn':
            if not data or 'question' not in data or 'answer' not in data:
                self._send_json({'error': 'Missing \"question\" and \"answer\"'}, 400)
                return
            try:
                # Explicit teaching: user provides a Q&A pair
                question = data['question'].strip()
                answer = data['answer'].strip()
                intent = data.get('skill', None)
                if not intent:
                    intent, _ = detect_skill(question)
                if not intent:
                    intent = 'definition_question'
                # Save to dataset
                import json as _json
                from pathlib import Path
                ds_path = Path(f"datasets/{intent}.json")
                pairs = load_dataset(intent) if ds_path.exists() else []
                # Remove existing entry for same question (dedup)
                pairs = [(q, a) for q, a in pairs if q.lower().strip() != question.lower().strip()]
                pairs = pairs + [(question, answer)]
                ds_path.parent.mkdir(parents=True, exist_ok=True)
                ds_path.write_text(_json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
                # Trigger retrain
                if intent not in manager.training_queue:
                    manager._auto_train_intent(intent, question)
                self._send_json({'learned': True, 'question': question, 'answer': answer, 'intent': intent, 'total_pairs': len(pairs)})
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        elif path == '/session-memory':
            mem_file = Path('session_memory.json')
            if mem_file.exists():
                try:
                    existing = json.loads(mem_file.read_text())
                except:
                    existing = []
            else:
                existing = []
            entry = {
                'timestamp': time.time(),
                'date': time.strftime('%Y-%m-%d %H:%M'),
                'question': data.get('question', ''),
                'correct': data.get('correct', ''),
                'wrong': data.get('wrong', ''),
                'right': data.get('right', ''),
                'notes': data.get('notes', ''),
            }
            existing.append(entry)
            mem_file.write_text(json.dumps(existing, indent=2))
            # Also append to session log
            with open('session_log.txt', 'a') as f:
                f.write(f"\n=== {entry['date']} ===\n")
                f.write(f"Question: {entry['question']}\n")
                f.write(f"Correct: {entry['correct']}\n")
                f.write(f"Wrong: {entry['wrong']}\n")
                f.write(f"Right: {entry['right']}\n")
                f.write(f"Notes: {entry['notes']}\n")
            self._send_json({'saved': True, 'total': len(existing)})
        else:
            self._send_json({'error': 'Not found'}, 404)


def print_welcome():
    print()
    print('  ╔══════════════════════════════════════════╗')
    print('  ║        TABULA RASA AI SYSTEM             ║')
    print('  ║   Learning from scratch, one skill at    ║')
    print('  ║   a time, just like a human brain.       ║')
    print('  ╚══════════════════════════════════════════╝')
    print()
    print(f'  Known skills: {len(manager.known_skills)}')
    for s in manager.known_skills:
        print(f'    ✓ {s}: {SKILL_REGISTRY[s]["description"]}')
    if manager.queued_skills:
        print(f'  Queued (not yet trained):')
        for s in manager.queued_skills:
            print(f'    ○ {s}: {SKILL_REGISTRY[s]["description"]}')
    print()


def main():
    global manager

    parser = argparse.ArgumentParser(description='Tabula Rasa AI')
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    args = parser.parse_args()

    print('[*] Loading existing skills...')
    manager = SkillManager()
    print_welcome()

    print(f'[*] Starting Tabula Rasa AI on http://{args.host}:{args.port}')
    print(f'[*] Ask me anything:')
    print(f'    curl http://{args.host}:{args.port}/ask -d \'{{"question":"12+34="}}\'')
    print(f'    curl http://{args.host}:{args.port}/skills')
    print()

    server = ThreadedHTTPServer((args.host, args.port), TabulaRasaHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[*] Shutting down...')
        server.server_close()


if __name__ == '__main__':
    main()
