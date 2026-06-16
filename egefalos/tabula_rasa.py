"""Tabula Rasa AI System — learns skills one at a time.

Each skill = a specialist transformer trained from scratch.
The router knows what it knows, and says "I don't know" for what it doesn't.

Usage:
    python3 tabula_rasa.py              # Start the AI
    python3 tabula_rasa.py --learn biology  # Queue a new skill for training
"""

import argparse, json, time, sys, re, os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from socketserver import ThreadingMixIn

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
        'ops': ['where', 'why', 'how', 'when'],
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
        'ops': ['tell me', 'joke', 'story', 'sing'],
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
    'definition_question':  {'temp': 0.0, 'max_tokens': 60, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
    'conversation':         {'temp': 0.0, 'max_tokens': 70, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
    'question':             {'temp': 0.0, 'max_tokens': 60, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
    'unknown':              {'temp': 0.0, 'max_tokens': 40, 'max_seq': 96,  'd_model': 64,  'n_layers': 3, 'n_heads': 4, 'd_ff': 128, 'steps': 500},
    'translation':          {'temp': 0.0, 'max_tokens': 60, 'max_seq': 160, 'd_model': 128, 'n_layers': 4, 'n_heads': 8, 'd_ff': 256, 'steps': 1200},
}

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
    prompt_lower = prompt.lower()
    for name, info in SKILL_REGISTRY.items():
        for op in info['ops']:
            if op in prompt_lower or op in prompt:
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

        # Replay saved corrections from persistent memory
        self._replay_memory()

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

    @torch.no_grad()
    def ask(self, prompt: str, temperature=0.3, top_k=5, skill=None) -> dict:
        """Route a question to the right skill.
        If `skill` is provided and loaded, force-use that specialist.
        """
        force_skill = skill if (skill and skill in self.models) else None
        if force_skill:
            skill = force_skill
            confidence = 0.9
        else:
            skill, confidence = detect_skill(prompt)
        debug(f"ask: prompt={prompt!r} detect_skill={skill!r} conf={confidence}")

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
            elif prompt_lower.startswith('how') or prompt_lower.startswith('why') or prompt_lower.startswith('when') or prompt_lower.startswith('where'):
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
            # Good retrieval match (>30% word overlap) → return immediately, train once for creativity
            if score >= 0.3:
                debug(f"ask: retrieval {skill} match={score:.2f} -> {ans_ret!r}")
                # One-shot creativity training (only if never trained before)
                if skill not in self.models and skill not in self.training_queue:
                    self._auto_train_intent(skill, prompt)
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

    def _retrieve_answer(self, skill, prompt):
        """Retrieval fallback — find best training example by word overlap."""
        pairs = load_dataset(skill)
        prompt_words = set(prompt.lower().split())
        scored = []
        for q, a in pairs:
            q_words = set(q.lower().split())
            if not q_words: continue
            overlap = len(prompt_words & q_words) / len(q_words)
            scored.append((overlap, a))
        if not scored:
            scored = [(0, "I don't know about that yet.")]
        best_score = max(s[0] for s in scored)
        # Pick randomly from all matches >= 40% of best score (wider pool = more creativity)
        threshold = max(best_score * 0.4, 0.15)
        candidates = [a for s, a in scored if s >= threshold]
        import random
        best_answer = random.choice(candidates)
        creativity = round(min(len(candidates) * 11, 100))  # each variant ≈ 11%, capped at 100%
        level = self.skill_levels.get(skill, 0)
        sc = scale_config(skill, level)
        return best_answer, {
            'level': level, 'd_model': sc.get('d_model', '?'),
            'steps': sc.get('steps', '?'), 'mode': 'retrieval',
            'creativity': creativity,
        }, best_score

    def _auto_train_intent(self, intent: str, prompt: str, retrain=False):
        """Auto-train a specialist for an intent type in background."""
        if intent in self.training_queue:
            return  # Already training this intent

        # Intent-specific training data
        level = self.skill_levels.get(intent, 0)
        if retrain:
            level += 1
        self.skill_levels[intent] = level
        sc = scale_config(intent, level)
        debug(f"train: {intent} level={level} config={sc}")

        # Try to get CPU count
        try:
            import os
            cpu_count = os.cpu_count() or 1
        except:
            cpu_count = 1
        pairs = load_dataset(intent)
        if not pairs:
            # Auto-create dataset file with the prompt as first training pair
            import json
            dataset_path = Path(f"datasets/{intent}.json")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            auto_answer = f"I'm learning about '{prompt}'. Ask me again and I'll improve!"
            pairs = [(prompt, auto_answer)]
            dataset_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
            debug(f"auto-dataset: created datasets/{intent}.json with 1 pair")
            # Also add to SKILL_REGISTRY so it persists across restarts (in-memory only here)
            if intent not in SKILL_REGISTRY:
                SKILL_REGISTRY[intent] = {
                    'ops': [],
                    'description': f'Auto-created specialist for {intent}',
                    'status': 'queued',
                    'dir': f'specialists/{intent}',
                }

        extra_steps = 0
        if retrain:
            # Append current query as new training pair and increase training
            pairs = pairs + [(prompt, f"I'm learning more about '{prompt}' every time you ask.")]
            extra_steps = 100

        self.training_queue[intent] = True
        self.training_progress[intent] = {
            'step': 0, 'total': sc['steps'] + (extra_steps if retrain else 0),
            'loss': 0, 'status': 'starting',
            'config': sc, 'cpu': cpu_count, 't0': time.time(),
        }
        import threading
        t = threading.Thread(target=self._train_intent_worker, args=(intent, pairs), daemon=True)
        t.start()

    def _train_intent_worker(self, intent: str, pairs: list):
        """Train a tiny chat specialist on intent-specific data."""
        import torch
        debug(f"train: worker starting intent={intent!r} pairs={len(pairs)}")
        try:
            from tabula_rasa.bpe_tokenizer import BPETokenizer
            from tabula_rasa.model import MathTransformer, count_parameters
            from tabula_rasa.config import Config

            print(f"  [*] Auto-training {intent} specialist ({len(pairs)} pairs)...")

            # Build BPE tokenizer from training texts
            tok = BPETokenizer()
            texts = []
            for q, a in pairs:
                texts.append(f"{q}=")
                texts.append(a)
            # Add the separator pattern
            tok.learn_bpe_from_texts(texts, num_merges=50, verbose=False)
            print(f"  [*] BPE vocab: {tok.vocab_size} tokens")

            # Tiny config for fast training
            prog = self.training_progress.get(intent, {})
            _sc = prog.get('config', {})
            cfg = Config()
            cfg.d_model = _sc.get('d_model', 64)
            cfg.n_layers = _sc.get('n_layers', 3)
            cfg.n_heads = _sc.get('n_heads', 4)
            cfg.d_ff = _sc.get('d_ff', 128)
            cfg.vocab_size = tok.vocab_size
            cfg.max_seq_len = _sc.get('max_seq', 128)
            cfg.batch_size = len(pairs)
            cfg.max_steps = prog.get('total', 500)
            cfg.learning_rate = 0.001
            cfg.use_reversed = False
            cfg.use_loss_masking = True

            tok.max_seq_len = cfg.max_seq_len
            model = MathTransformer(cfg)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

            # Encode training data
            import torch
            xs, ys = [], []
            for q, a in pairs:
                text = f"{q}={a}"
                ids = tok.encode(text, add_special_tokens=True)
                ids = (ids + [tok.pad_id] * cfg.max_seq_len)[:cfg.max_seq_len]
                xs.append(torch.tensor(ids[:-1]))
                ys.append(torch.tensor(ids[1:]))

            x = torch.stack(xs)
            y = torch.stack(ys)
            dataset = torch.utils.data.TensorDataset(x, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=min(8, len(pairs)), shuffle=True)

            model.train()
            for step in range(cfg.max_steps):
                for bx, by in loader:
                    opt.zero_grad()
                    _, loss, _ = model(bx, by)
                    loss.backward()
                    opt.step()
                if (step + 1) % 25 == 0 or step == 0:
                    p = self.training_progress.get(intent, {})
                    self.training_progress[intent] = {
                        'step': step + 1, 'total': cfg.max_steps,
                        'loss': round(loss.item(), 4), 'status': 'training',
                        'cpu': p.get('cpu'), 't0': p.get('t0'),
                    }
                    print(f"  [*] {intent} training step {step+1}/{cfg.max_steps}, loss={loss.item():.4f}")

            self.training_progress[intent] = {
                'step': cfg.max_steps, 'total': cfg.max_steps,
                'loss': round(loss.item(), 4), 'status': 'saving',
                'cpu': self.training_progress.get(intent, {}).get('cpu'),
                't0': self.training_progress.get(intent, {}).get('t0'),
            }

            model.eval()

            # Save
            save_dir = Path(f"specialists/{intent}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': cfg.max_steps,
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
            }, save_dir / 'best.pt')
            tok.save(str(save_dir / 'tokenizer.json'))

            # Register and load
            SKILL_REGISTRY[intent] = {
                'ops': [],
                'description': f'Auto-trained {intent} specialist',
                'status': 'ready',
                'dir': f'specialists/{intent}',
            }
            self.models[intent] = model
            self.tokenizers[intent] = tok
            self.training_progress[intent] = {
                'step': cfg.max_steps, 'total': cfg.max_steps,
                'loss': round(loss.item(), 4), 'status': 'done',
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
