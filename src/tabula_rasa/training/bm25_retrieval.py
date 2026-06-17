"""BM25 retrieval for fallback answer matching — replaces brittle word-overlap.

Uses a lightweight Okapi BM25 scorer over the hippocampus SQLite experience
store. No external dependencies — pure Python with IDF precomputation.

Usage:
    from tabula_rasa.training.bm25_retrieval import BM25Retriever
    
    retriever = BM25Retriever()
    retriever.index_from_hippocampus()
    results = retriever.retrieve("what is addition", top_k=3)
    # Returns [(score, input_text, output_text), ...]
"""

import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

# Default stop words for filtering
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "what", "which", "who",
    "whom", "this", "that", "these", "those", "it", "its", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "about", "up", "an", "do"
})

DB_PATH = Path("memory/hippocampus.db")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens, split on non-alpha."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class BM25Retriever:
    """Okapi BM25 retrieval over hippocampus experiences.

    Thread-safe after index is built. Index is lazily built on first
    retrieval call or explicitly via index_from_hippocampus().

    Attributes:
        k1: BM25 k1 parameter (term saturation, default 1.5).
        b: BM25 b parameter (length normalization, default 0.75).
        avgdl: Average document length over indexed corpus.
        N: Total number of documents indexed.
        idf: Precomputed IDF dict {term: idf_value}.
        doc_freqs: Precomputed term frequencies per doc: [{term: count}, ...].
        doc_texts: List of (input_text, output_text) tuples.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.avgdl = 0.0
        self.N = 0
        self.idf: dict[str, float] = {}
        self.doc_freqs: list[Counter] = []
        self.doc_texts: list[tuple[str, str]] = []
        self._indexed = False

    def _tokenize_doc(self, input_text: str, output_text: str) -> list[str]:
        """Tokenize a (prompt, response) pair into combined tokens."""
        return tokenize(input_text) + tokenize(output_text)

    def index_from_hippocampus(self, limit: int = 5000,
                               tier: str = "active",
                               min_error: float = 0.0) -> None:
        """Build BM25 index from hippocampus experiences.

        Args:
            limit: Max number of experiences to index.
            tier: Memory tier to query ('active', 'longterm', 'core').
            min_error: Minimum prediction error threshold (higher = more surprising).
        """
        db_path = Path(DB_PATH)
        if not db_path.exists():
            print(f"[BM25] Hippocampus DB not found at {db_path}")
            self._indexed = True  # empty index — don't keep trying
            return

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT input_text, output_text FROM experiences
                   WHERE tier=? AND prediction_error >= ?
                   ORDER BY id DESC LIMIT ?""",
                (tier, min_error, limit)
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            print(f"[BM25] No hippocampus experiences found (tier={tier})")
            self._indexed = True
            return

        docs: list[list[str]] = []
        self.doc_texts = []
        for row in rows:
            inp = row["input_text"] or ""
            out = row["output_text"] or ""
            tokens = self._tokenize_doc(inp, out)
            if tokens:
                docs.append(tokens)
                self.doc_texts.append((inp, out))

        self.N = len(docs)
        if self.N == 0:
            self._indexed = True
            return

        # Compute term frequencies per doc
        self.doc_freqs = [Counter(d) for d in docs]

        # Compute document frequencies (how many docs contain each term)
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())

        # Compute IDF
        self.idf = {}
        for term, doc_count in df.items():
            self.idf[term] = math.log(1 + (self.N - doc_count + 0.5) / (doc_count + 0.5))

        # Compute average document length
        total_len = sum(len(d) for d in docs)
        self.avgdl = total_len / self.N if self.N > 0 else 1.0

        self._indexed = True
        print(f"[BM25] Indexed {self.N} docs, {len(self.idf)} terms, avgdl={self.avgdl:.1f}")

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[float, str, str]]:
        """Retrieve top-k documents by BM25 score.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of (score, input_text, output_text) tuples, highest score first.
        """
        if not self._indexed:
            self.index_from_hippocampus()

        if self.N == 0 or not self.doc_freqs:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[float, int]] = []  # (score, idx)
        for idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = sum(freqs.values())
            for term in query_tokens:
                if term not in self.idf:
                    continue
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                score += self.idf[term] * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                )
            scores.append((score, idx))

        scores.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                inp, out = self.doc_texts[idx]
                results.append((score, inp, out))

        return results

    def score(self, query: str, input_text: str, output_text: str) -> float:
        """Score a single (input, output) pair against a query.

        Useful for re-ranking or threshold-based filtering.
        """
        query_tokens = tokenize(query)
        doc_tokens = self._tokenize_doc(input_text, output_text)
        doc_len = len(doc_tokens)
        if doc_len == 0 or not query_tokens:
            return 0.0

        freqs = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            score += self.idf[term] * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            )
        return score
