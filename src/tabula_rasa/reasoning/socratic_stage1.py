"""Socratic Engine — Stage 1: Curious Toddler.

The model reads text, detects concepts it can't predict (high surprise),
generates "What is X?" questions, extracts definitions from context,
and trains on the self-generated Q&A pairs.

This bootstraps foundational vocabulary and understanding.
"""

import random
import re
from pathlib import Path

import torch
from torch.optim import AdamW

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer

# ─── Expanded Tokenizer for Language ─────────────────────────────────

LATIN_CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
LANG_PUNCT = list('?!:;,.\'"@#-')
LANG_EXTRA = ['\n', '\t']

class LangTokenizer(MathTokenizer):
    """Tokenizer extended with Latin alphabet for language understanding."""

    MATH_CHARS = list('0123456789+-*/=().% ') + LATIN_CHARS + LANG_PUNCT


# ─── Document Loader ─────────────────────────────────────────────────

def load_documents(text_dir: str = 'documents') -> list[str]:
    """Load text documents from a directory."""
    d = Path(text_dir)
    if not d.exists():
        # Create sample documents
        d.mkdir(parents=True, exist_ok=True)
        samples = [
            "The sun is a star. It gives light and heat. Plants use sunlight to make food through photosynthesis.",
            "Water is made of hydrogen and oxygen. It is H2O. Water freezes at 0 degrees Celsius.",
            "Gravity is a force that pulls objects toward each other. It keeps planets in orbit around the sun.",
            "A cell is the basic unit of life. All living things are made of cells. Cells have a nucleus.",
            "DNA is the blueprint of life. It contains genetic instructions. DNA is shaped like a double helix.",
        ]
        for i, text in enumerate(samples):
            (d / f'sample_{i}.txt').write_text(text)
        print(f'  [*] Created {len(samples)} sample documents in {text_dir}/')

    docs = []
    for f in sorted(d.glob('*.txt')):
        docs.append(f.read_text().strip())
    print(f'  [*] Loaded {len(docs)} documents')
    return docs


# ─── Curiosity Engine ────────────────────────────────────────────────

class CuriosityEngine:
    """Detects concepts the model finds surprising/unpredictable."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def find_surprising_concepts(self, text: str, top_k=5) -> list[dict]:
        """Find tokens with highest prediction error (surprise)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 3:
            return []

        x = torch.tensor([ids[:-1]], device=self.device)
        y = torch.tensor([ids[1:]], device=self.device)

        self.model.eval()
        logits, loss = self.model(x, y)

        # Per-token loss
        probs = torch.softmax(logits[0], dim=-1)
        target_probs = probs[range(len(ids) - 1), y[0]]
        token_losses = -torch.log(target_probs + 1e-10)

        # Find high-loss tokens (surprising = hard to predict)
        surprises = []
        for i in range(len(token_losses)):
            if token_losses[i] > 1.0:  # threshold: loss > 1.0 = surprising
                # Get the token and surrounding context
                start = max(0, i - 3)
                end = min(len(ids) - 1, i + 4)
                context_ids = ids[start:end]
                context = self.tokenizer.decode(context_ids)
                token_text = self.tokenizer.decode([ids[i]])

                surprises.append({
                    'position': i,
                    'token': token_text,
                    'loss': round(token_losses[i].item(), 3),
                    'context': context,
                    'is_unknown': ids[i] == self.tokenizer.unk_id,
                })

        # Sort by loss descending, return top_k
        surprises.sort(key=lambda x: x['loss'], reverse=True)
        return surprises[:top_k]


# ─── Definition Extractor ────────────────────────────────────────────

def extract_definition(text: str, concept: str) -> str | None:
    """Extract a definition for a concept from surrounding text.

    Heuristic: find sentences containing the concept using patterns like:
    - "X is/are/was/were Y"
    - "X, also known as Y,"
    - "X refers to Y"
    """
    if not concept or len(concept) > 30:
        return None

    # Normalize
    concept_lower = concept.strip().lower()
    sentences = re.split(r'[.!?]+', text)

    for sent in sentences:
        sent = sent.strip()
        sent_lower = sent.lower()
        if concept_lower not in sent_lower:
            continue

        # Check for definition patterns
        patterns = [
            rf'\b{re.escape(concept_lower)}\b is',
            rf'\b{re.escape(concept_lower)}\b are',
            rf'\b{re.escape(concept_lower)}\b was',
            rf'\b{re.escape(concept_lower)}\b refers to',
            rf'\b{re.escape(concept_lower)}\b means',
            rf'\b{re.escape(concept_lower)}\b, also known as',
            rf'\b{re.escape(concept_lower)}\b, or',
        ]
        for pat in patterns:
            if re.search(pat, sent_lower):
                return sent.strip()

    # Fallback: return sentence containing the concept
    for sent in sentences:
        if concept_lower in sent.lower():
            return sent.strip()

    return None


# ─── Self-Q&A Generator ─────────────────────────────────────────────

def generate_qa_pairs(text: str, surprises: list[dict]) -> list[tuple]:
    """Generate (question, answer) pairs from surprising concepts."""
    pairs = []
    seen = set()

    for s in surprises:
        concept = s['token'].strip()
        if not concept or len(concept) < 2:
            continue
        if concept in seen:
            continue
        seen.add(concept)

        definition = extract_definition(text, concept)
        if definition:
            q = f'what is {concept}?'
            a = definition
            pairs.append((q, a))

    return pairs


# ─── Curious Toddler Training ────────────────────────────────────────

def train_curious_toddler(model, tokenizer, documents: list[str],
                          cycles=5, pairs_per_cycle=10, epochs=3):
    """Main Curious Toddler training loop."""
    engine = CuriosityEngine(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    total_pairs = 0
    for cycle in range(cycles):
        print(f'\n  Cycle {cycle + 1}/{cycles}')
        all_pairs = []

        for doc in documents:
            surprises = engine.find_surprising_concepts(doc, top_k=5)
            if not surprises:
                continue
            pairs = generate_qa_pairs(doc, surprises)
            all_pairs.extend(pairs)

        if not all_pairs:
            print('    No Q&A pairs generated')
            continue

        # Take top pairs
        random.shuffle(all_pairs)
        pairs = all_pairs[:pairs_per_cycle]
        print(f'    Generated {len(pairs)} Q&A pairs')

        # Train on pairs
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            n = 0
            for q, a in pairs:
                text = f'{q} {a}'
                ids = tokenizer.encode(text, add_special_tokens=True)
                if len(ids) > 64:
                    ids = ids[:64]
                padded = ids + [tokenizer.pad_id] * (64 - len(ids))
                x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0)
                y = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0)

                optimizer.zero_grad()
                _, loss = model(x, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n += 1

            if n > 0:
                print(f'    Epoch {epoch + 1}: loss={total_loss / n:.4f}')

        total_pairs += len(pairs)
        print(f'    Total pairs learned: {total_pairs}')

    return total_pairs


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('  Socratic Engine — Stage 1: Curious Toddler')
    print('=' * 60)

    # Load or create documents
    docs = load_documents('documents')

    # Build tokenizer with Latin alphabet
    tok = LangTokenizer()
    print(f'  Tokenizer: {tok.vocab_size} tokens')
    tok.save('checkpoints/lang_tokenizer.json')

    # Load generalist model with expanded vocab
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    print(f'  Model: {count_parameters(model):,} params')

    # Train Curious Toddler
    total = train_curious_toddler(model, tok, docs, cycles=3, pairs_per_cycle=5)

    print(f'\n{"=" * 60}')
    print('  Curious Toddler training complete!')
    print(f'  Total Q&A pairs learned: {total}')
    print('  Tokenizer saved to: checkpoints/lang_tokenizer.json')
    print(f'{"=" * 60}')
