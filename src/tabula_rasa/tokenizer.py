"""Tokenizer for math expressions with combined carry-digit tokens.

Tokens: 0-9, +, -, *, /, =, (, ), ., %, space, special tokens,
plus 20 combined carry-digit tokens: "00"-"09" (carry=0), "10"-"19" (carry=1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar


class MathTokenizer:
    """Tokenizer for arithmetic expressions with carry-digit combined tokens.

    Encodes strings like ``"12+34=46"`` into token ID sequences and decodes
    them back. Uses a longest-match-first strategy so combined carry-digit
    tokens (e.g. ``"04"``) are matched before individual digits.

    Attributes:
        stoi: Mapping from token string to integer ID.
        itos: Mapping from integer ID back to token string.
        vocab_size: Total number of tokens in the vocabulary.
        pad_id: Token ID for ``<PAD>``.
        bos_id: Token ID for ``<BOS>``.
        eos_id: Token ID for ``<EOS>``.
        unk_id: Token ID for ``<UNK>``.
        _tokens_sorted: All tokens sorted by length (longest first) for
            longest-match-first encoding.
    """

    SPECIAL_TOKENS: ClassVar[list[str]] = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

    MATH_CHARS: ClassVar[list[str]] = list('0123456789+-*/=().% ')

    # 20 combined carry-digit tokens: carry(0-1) + digit(0-9)
    CARRY_TOKENS: ClassVar[list[str]] = [f"{c}{d}" for c in range(2) for d in range(10)]

    def __init__(self) -> None:
        """Initialize the tokenizer and build the vocabulary."""
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Construct the token-to-ID and ID-to-token lookup tables."""
        all_tokens: list[str] = self.SPECIAL_TOKENS + self.CARRY_TOKENS + self.MATH_CHARS
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.itos: dict[int, str] = {i: t for i, t in enumerate(all_tokens)}
        self.vocab_size: int = len(all_tokens)

        self.pad_id: int = self.stoi['<PAD>']
        self.bos_id: int = self.stoi['<BOS>']
        self.eos_id: int = self.stoi['<EOS>']
        self.unk_id: int = self.stoi['<UNK>']

        self._tokens_sorted: list[str] = sorted(all_tokens, key=len, reverse=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs using longest-match-first.

        Args:
            text: Input string (e.g. ``'12+34=46'``).
            add_special_tokens: If ``True``, prepend ``<BOS>`` and append
                ``<EOS>`` to the encoded sequence.

        Returns:
            List of integer token IDs.
        """
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        i = 0
        while i < len(text):
            matched = False
            for tok in self._tokens_sorted:
                if tok in self.SPECIAL_TOKENS:
                    continue
                if text[i:i + len(tok)] == tok:
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

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of integer token IDs to decode.
            skip_special: If ``True``, skip special tokens (PAD, BOS, EOS,
                UNK) in the output.

        Returns:
            Decoded string.
        """
        chars: list[str] = []
        for i in ids:
            if skip_special and i in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
                continue
            chars.append(self.itos[i])
        return ''.join(chars)

    def save(self, path: str | Path) -> None:
        """Save the tokenizer vocabulary to a JSON file.

        Args:
            path: File path to save to (creates parent directories if needed).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, dict[str, str] | dict[str, str]] = {
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> MathTokenizer:
        """Load a tokenizer vocabulary from a JSON file saved by :meth:`save`.

        Args:
            path: File path to load from.

        Returns:
            A new ``MathTokenizer`` instance with the restored vocabulary.
        """
        with open(path) as f:
            data: dict[str, dict[str, str]] = json.load(f)
        tok: MathTokenizer = cls.__new__(cls)
        tok.stoi = data['stoi']  # type: ignore[assignment]
        tok.itos = {int(k): v for k, v in data['itos'].items()}  # type: ignore[assignment]
        tok.vocab_size = len(tok.stoi)  # type: ignore[arg-type]
        tok.pad_id = tok.stoi['<PAD>']  # type: ignore[index]
        tok.bos_id = tok.stoi['<BOS>']  # type: ignore[index]
        tok.eos_id = tok.stoi['<EOS>']  # type: ignore[index]
        tok.unk_id = tok.stoi['<UNK>']  # type: ignore[index]
        tok._tokens_sorted = sorted(tok.stoi.keys(), key=len, reverse=True)  # type: ignore[arg-type]
        return tok


if __name__ == '__main__':
    tok = MathTokenizer()
    print(f'Vocab size: {tok.vocab_size}')
    for test in ['04', '15', '10', '12+34=0406', '7+8=15']:
        ids = tok.encode(test, add_special_tokens=True)
        decoded = tok.decode(ids, skip_special=True)
        print(f'  {test:>15} -> {ids} -> {decoded}')
