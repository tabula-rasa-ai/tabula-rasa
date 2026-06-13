"""Tokenizer for math expressions with combined carry-digit tokens.

Tokens: 0-9, +, -, *, /, =, (, ), ., %, space, special tokens,
plus 20 combined carry-digit tokens: "00"-"09" (carry=0), "10"-"19" (carry=1).
"""
import json
from pathlib import Path


class MathTokenizer:
    """Tokenizer for arithmetic expressions with carry-digit combined tokens."""

    SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

    MATH_CHARS = list('0123456789+-*/=().% ')

    # 20 combined carry-digit tokens: carry(0-1) + digit(0-9)
    CARRY_TOKENS = [f"{c}{d}" for c in range(2) for d in range(10)]

    def __init__(self):
        self._build_vocab()

    def _build_vocab(self):
        all_tokens = self.SPECIAL_TOKENS + self.CARRY_TOKENS + self.MATH_CHARS
        self.stoi = {t: i for i, t in enumerate(all_tokens)}
        self.itos = {i: t for i, t in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

        self.pad_id = self.stoi['<PAD>']
        self.bos_id = self.stoi['<BOS>']
        self.eos_id = self.stoi['<EOS>']
        self.unk_id = self.stoi['<UNK>']

        self._tokens_sorted = sorted(all_tokens, key=len, reverse=True)

    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs using longest-match-first."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        i = 0
        while i < len(text):
            matched = False
            for tok in self._tokens_sorted:
                if tok in self.SPECIAL_TOKENS:
                    continue
                if text[i:i+len(tok)] == tok:
                    ids.append(self.stoi[tok])
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                ids.append(self.stoi.get(ch, self.unk_id))
                i += 1
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special=True):
        """Decode token IDs back to text."""
        chars = []
        for i in ids:
            if skip_special and i in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
                continue
            chars.append(self.itos[i])
        return ''.join(chars)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'stoi': self.stoi, 'itos': {str(k): v for k, v in self.itos.items()}}
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.stoi = data['stoi']
        tok.itos = {int(k): v for k, v in data['itos'].items()}
        tok.vocab_size = len(tok.stoi)
        tok.pad_id = tok.stoi['<PAD>']
        tok.bos_id = tok.stoi['<BOS>']
        tok.eos_id = tok.stoi['<EOS>']
        tok.unk_id = tok.stoi['<UNK>']
        tok._tokens_sorted = sorted(tok.stoi.keys(), key=len, reverse=True)
        return tok


if __name__ == '__main__':
    tok = MathTokenizer()
    print(f'Vocab size: {tok.vocab_size}')
    for test in ['04', '15', '10', '12+34=0406', '7+8=15']:
        ids = tok.encode(test, add_special_tokens=True)
        decoded = tok.decode(ids, skip_special=True)
        print(f'  {test:>15} -> {ids} -> {decoded}')
