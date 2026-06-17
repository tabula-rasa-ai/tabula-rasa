"""Self-Seeded BPE Tokenizer — grows its vocabulary from verified internal monologue.

Standard BPE requires a massive scraped corpus to learn merge rules, which
violates the "blank slate" philosophy. This tokenizer instead learns merges
from the model's *own* verified Socratic debates and corrected outputs.

Pipeline:
1. The Socratic Engine (Stage 2) generates debate exchanges
2. Verified/corrected outputs are logged to `verified_monologue.txt`
3. During the Pythagorean Review / sleep cycle, learn_bpe_from_texts()
   discovers which character pairs merge most frequently
4. New merged tokens (e.g. <therefore>, <if-then>) are added to the vocabulary
5. encode_bpe() uses both the base character tokenizer and learned merges

Usage:
    from bpe_tokenizer import BPETokenizer
    tok = BPETokenizer()                          # starts as character-level
    tok.learn_bpe_from_text(["12+34=46", "5+5=10"])  # learn 5 merges
    ids = tok.encode_bpe("12+34=46")              # uses learned merges
    tok.save("specialists/math/general/tokenizer.json")
"""

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """Character-level tokenizer with optional BPE merge learning.

    Starts as a pure character-level tokenizer (same as MathTokenizer).
    After learn_bpe_from_texts(), it adds merged subword tokens to the
    vocabulary for more efficient encoding of frequent patterns.
    """

    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    # Base characters always available (deduped — no duplicates)
    BASE_CHARS = sorted(
        set(
            "0123456789+-*/=().% abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ\n\t!?,.:;'\"@#&|<>_"
        )
    )

    def __init__(self, base_chars: str = None):
        """Initialize tokenizer. If base_chars is given, use those instead of defaults."""
        self.merges = {}  # {(token_a, token_b): merged_token_id}
        self.merge_rank = {}  # {(token_a, token_b): rank} — for encoding
        self.bpe_vocab_size = 0  # additional tokens from BPE
        self._build_base_vocab(base_chars)

    def _build_base_vocab(self, extra_chars: str = None):
        """Build the base character-level vocabulary."""
        chars = list(self.BASE_CHARS)
        if extra_chars:
            for c in extra_chars:
                if c not in chars:
                    chars.append(c)

        all_tokens = self.SPECIAL_TOKENS + chars
        self.stoi = {ch: i for i, ch in enumerate(all_tokens)}
        self.itos = {i: ch for i, ch in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        self.base_vocab_size = self.vocab_size

        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]

        # Max sequence length (for padding)
        self.max_seq_len = 64

    # ─── Hybrid encoding — character-level with BPE when available ──

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text. Uses BPE merges if available, otherwise character-level.

        This makes BPETokenizer a drop-in replacement for MathTokenizer:
        the same interface but with BPE compression of frequent patterns.
        """
        if self.merges:
            return self.encode_bpe(text, add_special_tokens)
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text, handling merged BPE tokens."""
        chars = []
        for i in ids:
            if skip_special and i in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
                continue
            token = self.itos.get(i, "<UNK>")
            chars.append(token)
        text = "".join(chars)
        return text

    # ─── BPE merge learning ───────────────────────────────────────

    def _get_pair_frequencies(self, token_ids: list[list[int]]):
        """Count adjacent token pair frequencies across a corpus."""
        pairs = Counter()
        for ids in token_ids:
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_pair(self, token_ids: list[list[int]], pair: tuple, new_id: int):
        """Replace all occurrences of a token pair with the merged token ID."""
        new_ids_list = []
        for ids in token_ids:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids_list.append(new_ids)
        return new_ids_list

    def learn_bpe_from_texts(
        self, texts: list[str], num_merges: int = 50, min_frequency: int = 2, verbose: bool = True
    ):
        """Learn BPE merge rules from a list of texts.

        Args:
            texts: List of text strings (verified monologue, corrected math, etc.)
            num_merges: Maximum number of merge operations to learn.
            min_frequency: Minimum number of times a pair must appear to be merged.
            verbose: Print progress.

        Returns:
            Number of merges actually learned.
        """
        if not texts:
            return 0

        # Tokenize each text to base character IDs (no special tokens for BPE)
        token_ids = []
        for text in texts:
            ids = [self.stoi.get(ch, self.unk_id) for ch in text]
            # Filter out special tokens for BPE stats
            ids = [i for i in ids if i >= len(self.SPECIAL_TOKENS)]
            if ids:
                token_ids.append(ids)

        if not token_ids:
            return 0

        # Ensure core chars are already stoi entries
        # Build a reverse lookup for verbose merge reporting
        id_to_str = {i: s for s, i in self.stoi.items()}

        learned = 0
        for merge_idx in range(num_merges):
            pairs = self._get_pair_frequencies(token_ids)

            if not pairs:
                break

            # Filter by minimum frequency
            pairs = Counter({p: c for p, c in pairs.items() if c >= min_frequency})
            if not pairs:
                break

            # Find the most frequent pair
            (pair_a, pair_b), freq = pairs.most_common(1)[0]

            # Create new merged token ID
            new_id = self.vocab_size
            self.vocab_size += 1

            # Get token strings for the pair
            token_a = id_to_str.get(pair_a, f"<tok{pair_a}>")
            token_b = id_to_str.get(pair_b, f"<tok{pair_b}>")
            merged = token_a + token_b

            # Store merge
            self.merges[(token_a, token_b)] = new_id
            self.merge_rank[(token_a, token_b)] = merge_idx

            # Update vocabulary
            self.stoi[merged] = new_id
            self.itos[new_id] = merged

            # Apply merge to all token sequences
            token_ids = self._merge_pair(token_ids, (pair_a, pair_b), new_id)

            if verbose:
                print(
                    f'  [BPE] Merge #{merge_idx+1}: "{token_a}"+"{token_b}" → '
                    f'"{merged}" (freq={freq}, id={new_id})'
                )

            learned += 1

        self.bpe_vocab_size = self.vocab_size - self.base_vocab_size
        if verbose:
            print(f"  [BPE] Learned {learned} merges, vocab now {self.vocab_size} tokens")
            print(f"  [BPE] New words: {sorted(self.merges.keys())[:10]}...")
        return learned

    # ─── BPE encoding ─────────────────────────────────────────────

    def _bpe_tokenize(self, word_ids: list[int]) -> list[int]:
        """Apply learned BPE merges greedily to a sequence of character IDs."""
        if not self.merges:
            return word_ids

        # Work with string IDs via the merge ranking
        ids = list(word_ids)
        while len(ids) > 1:
            # Find the pair with the best (lowest) merge rank
            best_pair = None
            best_rank = float("inf")
            for i in range(len(ids) - 1):
                # Convert IDs to strings for merge lookup
                str_a = self.itos.get(ids[i], f"<{ids[i]}>")
                str_b = self.itos.get(ids[i + 1], f"<{ids[i+1]}>")
                pair = (str_a, str_b)
                if pair in self.merge_rank:
                    rank = self.merge_rank[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = (i, pair)

            if best_pair is None:
                break  # No more merges can be applied

            i, (str_a, str_b) = best_pair
            merged_id = self.merges[(str_a, str_b)]
            ids = ids[:i] + [merged_id] + ids[i + 2 :]

        return ids

    def encode_bpe(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text with BPE merges.

        Uses base character encoding first, then applies learned BPE merges
        to compress frequent character sequences into single tokens.
        """
        # Step 1: Base character encoding
        char_ids = [self.stoi.get(ch, self.unk_id) for ch in text]

        # Step 2: Apply BPE merges
        bpe_ids = self._bpe_tokenize(char_ids)

        # Step 3: Add special tokens
        if add_special_tokens:
            bpe_ids = [self.bos_id] + bpe_ids + [self.eos_id]

        return bpe_ids

    # ─── Save / Load ──────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save tokenizer to JSON, including BPE merges."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "merges": {f"{a}|{b}": mid for (a, b), mid in self.merges.items()},
            "merge_rank": {f"{a}|{b}": r for (a, b), r in self.merge_rank.items()},
            "base_vocab_size": self.base_vocab_size,
            "bpe_vocab_size": self.bpe_vocab_size,
            "max_seq_len": self.max_seq_len,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)

        tok = cls.__new__(cls)
        tok.stoi = data["stoi"]
        # Only keep itos entries that correspond to valid stoi entries
        tok.itos = {}
        for k_str, token in data["itos"].items():
            try:
                idx = int(k_str)
            except (ValueError, TypeError):
                continue
            if token in data["stoi"] and data["stoi"][token] == idx:
                tok.itos[idx] = token
        tok.vocab_size = len(tok.stoi)
        tok.merges = {}
        tok.merge_rank = {}
        for key, mid in data.get("merges", {}).items():
            a, b = key.split("|", 1)
            tok.merges[(a, b)] = mid
        for key, rank in data.get("merge_rank", {}).items():
            a, b = key.split("|", 1)
            tok.merge_rank[(a, b)] = rank
        tok.vocab_size = len(tok.stoi)
        tok.base_vocab_size = data.get("base_vocab_size", tok.vocab_size)
        tok.bpe_vocab_size = data.get("bpe_vocab_size", 0)
        tok.max_seq_len = data.get("max_seq_len", 64)

        tok.pad_id = tok.stoi["<PAD>"]
        tok.bos_id = tok.stoi["<BOS>"]
        tok.eos_id = tok.stoi["<EOS>"]
        tok.unk_id = tok.stoi["<UNK>"]
        return tok


# ─── Self-Seeded BPE Pipeline ─────────────────────────────────────

VERIFIED_LOG = Path("verified_monologue.txt")


def log_verified_output(text: str):
    """Append verified/corrected output to the monologue log.

    Call this after:
    - A Socratic debate produces a verified conclusion
    - A correction is successfully trained
    - The Pythagorean Review validates a logical statement
    """
    with open(VERIFIED_LOG, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return True


def learn_from_verified_log(
    max_merges: int = 30, min_frequency: int = 2, tokenizer: BPETokenizer = None
) -> int:
    """Run BPE merge learning on the verified monologue log.

    Reads the verified_monologue.txt file, learns BPE merges from it,
    and returns the number of merges learned.

    Args:
        max_merges: Maximum merges to learn this cycle.
        min_frequency: Minimum pair frequency to consider.
        tokenizer: Existing tokenizer to extend. Creates new one if None.

    Returns:
        Tuple of (tokenizer, num_merges_learned).
    """
    if not VERIFIED_LOG.exists():
        print("  [BPE] No verified monologue to learn from")
        return tokenizer, 0

    with open(VERIFIED_LOG, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        print("  [BPE] Verified monologue is empty")
        return tokenizer, 0

    print(f"  [BPE] Learning from {len(texts)} lines of verified monologue...")

    if tokenizer is None:
        tokenizer = BPETokenizer()

    learned = tokenizer.learn_bpe_from_texts(
        texts, num_merges=max_merges, min_frequency=min_frequency
    )
    return tokenizer, learned


# Quick test
if __name__ == "__main__":
    tok = BPETokenizer()
    print(f"Base vocab: {tok.vocab_size} tokens")

    # Simulate verified monologue
    monologue = [
        "therefore 12+34=46 is correct",
        "therefore 5+5=10 is correct",
        "therefore if all humans are mortal then socrates is mortal",
        "therefore the sum of two positive numbers is always positive",
        "therefore if a implies b and b is false then a is false",
    ]
    tok.learn_bpe_from_texts(monologue, num_merges=10)

    encoded = tok.encode_bpe("therefore 12+34=46")
    decoded = tok.decode(encoded)
    print(f"\nEncoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab:   {tok.vocab_size} tokens ({tok.bpe_vocab_size} from BPE)")

    # Save and reload test
    tok.save("/tmp/test_bpe_tok.json")
    tok2 = BPETokenizer.load("/tmp/test_bpe_tok.json")
    print(f"\nReloaded vocab: {tok2.vocab_size} tokens")
    print(f"Merges restored: {len(tok2.merges)}")
