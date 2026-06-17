"""Tests for MathTokenizer — encode, decode, carry-digit tokens."""

import tempfile
from pathlib import Path

from tabula_rasa.tokenizer import MathTokenizer


class TestTokenizerInit:
    def test_vocab_size(self, tok):
        """Tokenizer has expected vocabulary size (+4 for OP_PREFIXES)."""
        assert tok.vocab_size == 50, f"Expected 50 (+4 OP_PREFIXES), got {tok.vocab_size}"

    def test_special_token_ids(self, tok):
        """Special tokens have distinct IDs (including CoT markers)."""
        ids = {tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id, tok.step_id, tok.end_id}
        assert len(ids) == 6

    def test_carry_tokens_present(self, tok):
        """All 20 carry-digit tokens exist in vocabulary."""
        for c in range(2):
            for d in range(10):
                token = f"{c}{d}"
                assert token in tok.stoi, f"Missing carry token: {token}"


class TestTokenizerEncode:
    def test_encode_simple(self, tok):
        """Encoding a math expression returns expected token IDs."""
        ids = tok.encode("12+34=", add_special_tokens=False)
        assert len(ids) > 0
        # "12" is a 2-char token (longest-match), then "+", then "34", then "="
        assert tok.itos[ids[0]] == "12"
        assert tok.itos[ids[-1]] == "="

    def test_encode_with_special(self, tok):
        """Encoding with special tokens adds BOS and EOS."""
        ids = tok.encode("5+3=", add_special_tokens=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id

    def test_encode_carry_token(self, tok):
        """Carry-digit like '04' is encoded as a single token."""
        ids = tok.encode("04", add_special_tokens=False)
        assert len(ids) == 1
        assert tok.itos[ids[0]] == "04"

    def test_encode_carry_priority(self, tok):
        """Longer carry-digit tokens take priority over individual digits."""
        # "04" should be one token "04", not two tokens "0" + "4"
        ids = tok.encode("04", add_special_tokens=False)
        assert len(ids) == 1, f"Expected 1 token, got {len(ids)}: {[tok.itos[i] for i in ids]}"

    def test_encode_full_expression(self, tok):
        """Full expression encodes correctly."""
        ids = tok.encode("12+34=0406", add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == "12+34=0406"

    def test_encode_unknown_char(self, tok):
        """Unknown characters are mapped to UNK token."""
        ids = tok.encode("@", add_special_tokens=False)
        assert ids[0] == tok.unk_id

    def test_encode_empty_string(self, tok):
        """Empty string encodes to empty list (without special)."""
        ids = tok.encode("", add_special_tokens=False)
        assert ids == []

    def test_encode_repeated_carry(self, tok):
        """Multiple consecutive carry-digits tokenize correctly."""
        ids = tok.encode("0405", add_special_tokens=False)
        assert len(ids) == 2
        assert tok.itos[ids[0]] == "04"
        assert tok.itos[ids[1]] == "05"


class TestTokenizerDecode:
    def test_decode_roundtrip(self, tok):
        """encode then decode returns original text (without special)."""
        original = "12+34=46"
        ids = tok.encode(original, add_special_tokens=True)
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == original

    def test_decode_skips_special(self, tok):
        """decode with skip_special=True removes special tokens."""
        ids = [tok.bos_id, tok.stoi["5"], tok.stoi["+"], tok.stoi["3"], tok.eos_id]
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == "5+3"

    def test_decode_with_special(self, tok):
        """decode with skip_special=False retains special tokens."""
        ids = [tok.bos_id, tok.stoi["5"], tok.stoi["+"], tok.stoi["3"]]
        decoded = tok.decode(ids, skip_special=False)
        assert tok.itos[tok.bos_id] in decoded

    def test_decode_padded(self, tok):
        """Decode handles padding tokens correctly."""
        text = "5+3=8"
        ids = tok.encode(text, add_special_tokens=True)
        padded = ids + [tok.pad_id] * 10
        decoded = tok.decode(padded, skip_special=True)
        assert decoded == "5+3=8"

    def test_decode_empty(self, tok):
        """Decode empty list returns empty string."""
        assert tok.decode([], skip_special=True) == ""
        assert tok.decode([], skip_special=False) == ""


class TestTokenizerSaveLoad:
    def test_save_load_roundtrip(self, tok):
        """Save then load produces an equivalent tokenizer."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            tok.save(path)
            loaded = MathTokenizer.load(path)
            assert loaded.vocab_size == tok.vocab_size
            assert loaded.stoi == tok.stoi
            # Verify encode works on loaded
            assert loaded.encode("12+34=46") == tok.encode("12+34=46")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_creates_dir(self, tok):
        """Save creates parent directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "sub" / "tok.json"
            tok.save(str(path))
            assert path.exists()


class TestTokenizerEdgeCases:
    def test_space_preserved(self, tok):
        """Spaces in expressions are preserved."""
        text = "12 + 34 = 46"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == text

    def test_leading_spaces(self, tok):
        """Leading spaces are handled."""
        ids = tok.encode("  5+3=8", add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == "  5+3=8"

    def test_trailing_spaces(self, tok):
        """Trailing spaces are handled."""
        ids = tok.encode("5+3=8  ", add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == "5+3=8  "

    def test_no_space_around_op(self, tok):
        """No-space formatting works."""
        text = "10-4=6"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == text

    def test_multiple_operators(self, tok):
        """Chained operations tokenize correctly."""
        text = "2+3+4=9"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == text

    def test_negative_result(self, tok):
        """Expressions with negative results work (minus sign handled)."""
        text = "3-5=-2"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == text

    def test_decimal_numbers(self, tok):
        """Decimal point is a valid token."""
        text = "3.5+2.5=6.0"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == text

    def test_percent(self, tok):
        """Percent sign is a valid token."""
        text = "10%5=0"
        assert "%" in tok.stoi
        ids = tok.encode(text, add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == text

    def test_all_operations(self, tok):
        """All math operations tokenize correctly."""
        for op, text in [("+", "5+3=8"), ("-", "10-4=6"), ("*", "3*4=12"), ("/", "8/2=4")]:
            ids = tok.encode(text, add_special_tokens=False)
            decoded = "".join(tok.itos[i] for i in ids)
            assert decoded == text, f"Failed for {op}: got '{decoded}'"

    def test_parentheses(self, tok):
        """Parentheses are valid tokens."""
        text = "(1+2)*3=9"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = "".join(tok.itos[i] for i in ids)
        assert decoded == text

    def test_multiline_equal(self, tok):
        """'=' token is always present."""
        assert "=" in tok.stoi
        assert tok.stoi["="] > 0
