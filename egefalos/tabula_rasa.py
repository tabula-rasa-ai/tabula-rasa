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
}

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
                    cfg.vocab_size = tok.vocab_size
                    tok.max_seq_len = cfg.max_seq_len
                    model = MathTransformer(cfg)
                    state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    model.load_state_dict(state['model_state_dict'])
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

        # If detected skill is not loaded, fall back to a loaded one
        if skill is not None and skill not in self.models:
            # Try general_math as fallback
            if 'general_math' in self.models:
                skill = 'general_math'
            else:
                # Any loaded skill
                for s in self.known_skills:
                    skill = s
                    break

        if skill is None or skill not in self.models:
            # No skill knows this — analyze what's needed
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

            # ─── For capability questions ("what can you do"), list available skills ───
            if intent == 'capability_question':
                ready = [name for name, info in SKILL_REGISTRY.items() if info['status'] == 'ready']
                queued = [name for name, info in SKILL_REGISTRY.items() if info['status'] == 'queued' or info['status'] == 'training']
                lines = ["I have these skills:"]
                if ready:
                    lines.append(f"  Ready: {', '.join(ready)}")
                if queued:
                    lines.append(f"  Learning: {', '.join(queued)}")
                if not ready:
                    lines.append("  (none trained yet)")
                lines.append(f"Ask me a math problem like '2+3=' to see me in action!")
                answer_text = '\n'.join(lines)
                return {
                    'answer': answer_text,
                    'knows': True,
                    'message': None,
                    'detected_skill': None,
                    'suggested_skills': [],
                    'status': 'answered',
                    'time_ms': 0,
                    'analysis': None,
                }

            # ─── Auto-train for any unknown intent ───
            # Check if we already have the model or are training it
            if intent in self.models:
                # Model already trained, use it
                model = self.models[intent]
                tok = self.tokenizers[intent]
                t0 = time.time()
                full = model.generate(tok, prompt, max_new_tokens=30, temperature=0.3, top_k=5)
                elapsed = time.time() - t0
                return {
                    'prompt': prompt,
                    'answer': full,
                    'knows': True,
                    'skill': intent,
                    'time_ms': round(elapsed * 1000),
                    'status': 'answered',
                    'confidence': 50.0,
                    'is_confident': True,
                    'message': None,
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

            # Trigger auto-training
            self._auto_train_intent(intent, prompt)
            return {
                'answer': None,
                'knows': False,
                'message': f"I don't know about that yet. Auto-training a {intent} specialist now — try again shortly!",
                'detected_skill': None,
                'suggested_skills': [suggested_skill_name] if suggested_skill_name else [],
                'status': 'training',
                'time_ms': 0,
                'analysis': None,
            }

        model = self.models[skill]
        tok = self.tokenizers[skill]

        t0 = time.time()
        if not prompt.endswith('=') and '=' not in prompt:
            prompt += '='
        full = model.generate(tok, prompt, max_new_tokens=15,
                              temperature=temperature, top_k=top_k)
        elapsed = time.time() - t0

        # Confidence calibration: compare prediction to expected answer
        confidence = None
        is_confident = True
        try:
            pred = full.split('=')[-1].strip() if '=' in full else ''
            pred_clean = ''.join(c for c in pred if c.isdigit() or c == '-')
            # Compute expected answer if possible
            expr_clean = prompt.rstrip('=')
            expected = str(eval(expr_clean))
            match = pred_clean == expected
            if match:
                confidence = 100.0
            else:
                # Numeric distance-based confidence
                try:
                    pred_num = int(pred_clean) if pred_clean and pred_clean != '-' else 0
                    exp_num = int(expected)
                    diff = abs(pred_num - exp_num)
                    confidence = round(max(5.0, 100.0 - diff * 5), 1)
                except:
                    confidence = 10.0
            if not match:
                is_confident = False
        except:
            pass

        return {
            'prompt': prompt,
            'answer': full,
            'knows': True,
            'skill': skill,
            'skill_description': SKILL_REGISTRY[skill]['description'],
            'time_ms': round(elapsed * 1000),
            'status': 'answered',
            'confidence': confidence,
            'is_confident': is_confident,
            'message': None if is_confident else "I have a specialist for this, but I'm not confident in my answer yet.",
        }

    def _auto_train_intent(self, intent: str, prompt: str):
        """Auto-train a specialist for an intent type in background."""
        if intent in self.training_queue:
            return  # Already training this intent

        # Intent-specific training data
        INTENT_DATA = {
            'greeting': [
                ("Hello!", "Hi there! I'm Tabula Rasa, a helpful AI assistant."),
                ("Hi!", "Hello! How can I help you today?"),
                ("Hey!", "Hey there! Ask me anything or try a math problem."),
                ("Good morning!", "Good morning! Ready to learn and help."),
                ("Hello there!", "Hi! I'm Tabula Rasa. Want to solve some math?"),
                ("Hi how are you?", "I'm doing great! Ready to help with math or answer questions."),
                ("Greetings!", "Greetings! I'm a learning AI. Type a math expression like '2+3='!"),
                ("Hey, what's up?", "Not much, just learning new things! Ask me anything."),
                ("Hello, are you there?", "Yes, I'm here! What can I help you with?"),
                ("Hi, nice to meet you!", "Nice to meet you too! I'm Tabula Rasa, a continual learning AI."),
            ],
            'explanation_question': [
                ("Where are you from?", "I'm from the Tabula Rasa AI research project — a system that learns from scratch, one specialist at a time."),
                ("Where do you live?", "I live in the cloud! Or more precisely, on whatever machine I'm running on."),
                ("How do you work?", "I use transformer neural networks trained from scratch on specific skills called specialists."),
                ("Why are you here?", "To learn and assist! I'm designed to continually acquire new skills through training."),
                ("How old are you?", "I'm a young AI — every time I train a new specialist, I grow a little more!"),
                ("How were you made?", "I was built using PyTorch, with a custom transformer architecture trained from random initialization."),
                ("Why do you exist?", "I exist to demonstrate that AIs can learn from scratch, one skill at a time, without forgetting."),
                ("When were you created?", "I was first trained as a tiny math transformer and have been growing since."),
                ("How can you learn?", "Through continual learning techniques like EWC that prevent catastrophic forgetting."),
            ],
            'definition_question': [
                ("What is AI?", "AI is the field of creating machines that can perform tasks requiring human intelligence."),
                ("Who is your creator?", "I was created by the Tabula Rasa AI research project."),
                ("What is machine learning?", "Machine learning is a branch of AI where systems learn patterns from data."),
                ("What is Python?", "Python is a high-level, interpreted programming language."),
                ("What is a transformer?", "A transformer is a neural network architecture that uses self-attention."),
                ("What is deep learning?", "Deep learning uses multi-layer neural networks to learn hierarchical representations."),
            ],
            'conversation': [
                ("Tell me a joke.", "Why did the math book look sad? Because it had too many problems."),
                ("Tell me something interesting.", "Octopuses have three hearts, and two stop beating when they swim!"),
                ("Sing a song.", "I'm not much of a singer! How about a math problem instead?"),
                ("Thanks!", "You're welcome! Ask me anything."),
                ("Goodbye!", "Goodbye! Come back anytime."),
                ("Are you smart?", "I'm learning! My model is small but I grow through continual learning."),
            ],
        }

        pairs = INTENT_DATA.get(intent, [])
        if not pairs:
            # Generate generic training pair from the prompt
            pairs = [(prompt, f"I'm still learning about that. My suggested skill is '{intent}'.")]

        self.training_queue[intent] = True
        import threading
        t = threading.Thread(target=self._train_intent_worker, args=(intent, pairs), daemon=True)
        t.start()

    def _train_intent_worker(self, intent: str, pairs: list):
        """Train a tiny chat specialist on intent-specific data."""
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
            cfg = Config()
            cfg.d_model = 32
            cfg.n_layers = 2
            cfg.n_heads = 2
            cfg.d_ff = 64
            cfg.vocab_size = tok.vocab_size
            cfg.max_seq_len = 64
            cfg.batch_size = len(pairs)
            cfg.max_steps = 100
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
                if (step + 1) % 25 == 0:
                    print(f"  [*] {intent} training step {step+1}/{cfg.max_steps}, loss={loss.item():.4f}")

            model.eval()

            # Save
            save_dir = Path(f"specialists/{intent}")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': cfg.max_steps,
                'model_state_dict': model.state_dict(),
                'loss': loss.item(),
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
            params = count_parameters(model)
            print(f"  [*] Auto-trained {intent} specialist ready ({params:,} params)")
        except Exception as e:
            import traceback
            print(f"  [!] Auto-train {intent} failed: {e}")
            traceback.print_exc()
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
