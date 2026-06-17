"""Lightweight RAG engine — TF-IDF based retrieval for Tabula-Rasa.

Indexes knowledge_base.json, chat_data.jsonl, and specialist datasets.
No GPU required. Runs entirely on CPU with scikit-learn.

Usage as a library:
    from tabula_rasa.rag import RAGEngine
    rag = RAGEngine(data_dir="datasets")
    rag.build_index()
    results = rag.retrieve("What is a transformer?", top_k=3)

Usage as a server:
    python3 -m tabula_rasa.rag --port 8300
    curl http://localhost:8300/retrieve -d '{"query":"What is AI?...","top_k":3}'
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Optional scikit-learn ────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class Document:
    """A single indexed document with metadata."""

    text: str
    source: str  # e.g. 'knowledge_base', 'chat_data', 'specialist/greeting'
    metadata: dict = field(default_factory=dict)
    score: float = 0.0  # populated by retrieve()


@dataclass
class RetrievalResult:
    """Result of a RAG query."""

    query: str
    results: list[Document]
    time_ms: float


class RAGEngine:
    """TF-IDF based RAG engine for retrieval-augmented generation.

    Args:
        data_dir: Path to directory containing datasets/ (relative to CWD
            or absolute). Default ``datasets``.
        chat_path: Path to chat_data.jsonl. Default ``chat_data.jsonl``.
        specialist_dir: Path to specialists/ directory. Default ``specialists``.
    """

    def __init__(
        self,
        data_dir: str | Path = "datasets",
        chat_path: str | Path = "chat_data.jsonl",
        specialist_dir: str | Path = "specialists",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.chat_path = Path(chat_path)
        self.specialist_dir = Path(specialist_dir)
        self.documents: list[Document] = []
        self.vectorizer: Any = None
        self.tfidf_matrix: Any = None
        self._index_built = False

    def build_index(self) -> int:
        """Build the TF-IDF index from all data sources.

        Returns:
            Number of documents indexed.
        """
        if not HAS_SKLEARN:
            print("[RAG] scikit-learn not installed. Install with: pip install scikit-learn")
            return 0

        docs: list[Document] = []

        # 1. Knowledge base (datasets/knowledge_base.json)
        kb_path = self.data_dir / "knowledge_base.json"
        if kb_path.exists():
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                if isinstance(kb, dict):
                    for category, pairs in kb.items():
                        if isinstance(pairs, list):
                            for pair in pairs:
                                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                                    q, a = pair[0], pair[1]
                                    docs.append(Document(
                                        text=f"Q: {q}\nA: {a}",
                                        source=f"knowledge_base/{category}",
                                        metadata={"type": "qa", "category": category, "question": q, "answer": a},
                                    ))
            except Exception as e:
                print(f"[RAG] Warning: failed to load {kb_path}: {e}")

        # 2. Chat data (chat_data.jsonl)
        if self.chat_path.exists():
            try:
                with open(self.chat_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        prompt = entry.get("prompt", "")
                        response = entry.get("response", "")
                        if prompt and response:
                            docs.append(Document(
                                text=f"Q: {prompt}\nA: {response}",
                                source="chat_data",
                                metadata={"type": "chat", "question": prompt, "answer": response},
                            ))
            except Exception as e:
                print(f"[RAG] Warning: failed to load {self.chat_path}: {e}")

        # 3. Specialist data (specialists/*/)
        if self.specialist_dir.exists():
            for spec_dir in self.specialist_dir.iterdir():
                if not spec_dir.is_dir() or spec_dir.name.startswith("_"):
                    continue
                # Look for data files in specialist directories
                for ext in ("*.json", "*.jsonl", "*.txt"):
                    for fpath in spec_dir.glob(ext):
                        try:
                            text = fpath.read_text(encoding="utf-8")
                            if text.strip():
                                docs.append(Document(
                                    text=text[:2000],  # cap at 2K chars
                                    source=f"specialist/{spec_dir.name}/{fpath.name}",
                                    metadata={"type": "specialist", "specialist": spec_dir.name},
                                ))
                        except Exception:
                            pass

        if not docs:
            print("[RAG] Warning: no documents found to index")
            return 0

        self.documents = docs
        texts = [d.text for d in docs]

        # Build TF-IDF index
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._index_built = True

        print(f"[RAG] Indexed {len(docs)} documents ({self.tfidf_matrix.shape[1]} features)")
        return len(docs)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[Document]:
        """Retrieve the most relevant documents for a query.

        Args:
            query: The user's question or prompt.
            top_k: Maximum number of results.
            min_score: Minimum cosine similarity threshold.

        Returns:
            List of Document objects sorted by relevance (descending).
        """
        if not self._index_built or self.vectorizer is None or self.tfidf_matrix is None:
            print("[RAG] Index not built. Call build_index() first.")
            return []

        t0 = time.time()
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices above threshold
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            doc = self.documents[idx]
            doc.score = round(score, 4)
            results.append(doc)

        elapsed = (time.time() - t0) * 1000
        if results:
            return results
        return results

    def retrieve_formatted(self, query: str, top_k: int = 3) -> str:
        """Retrieve context and format as a prompt prefix string.

        Returns:
            Formatted context string like:
                "Relevant context:
                [1] (source) Q: ... A: ...
                [2] (source) Q: ... A: ..."
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        lines = ["Relevant context:"]
        for i, doc in enumerate(results, 1):
            source = doc.source
            text = doc.text[:300]  # truncate for prompt
            lines.append(f"[{i}] ({source}) {text}")
        return "\n".join(lines)

    def retrieve_answer(self, query: str) -> str | None:
        """Try to find an exact Q&A match for the query.

        Returns the answer string if a close match is found, None otherwise.
        Useful for direct-answer specialists.
        """
        results = self.retrieve(query, top_k=1, min_score=0.4)
        if results:
            doc = results[0]
            # Check if it's a Q&A document
            meta = doc.metadata
            if meta.get("type") in ("qa", "chat"):
                return meta.get("answer")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Standalone RAG Server
# ═══════════════════════════════════════════════════════════════════════


def run_server(port: int = 8300) -> None:
    """Run a lightweight HTTP RAG server."""
    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from socketserver import ThreadingMixIn
    except ImportError:
        print("HTTP server not available")
        return

    class ThreadedRAGServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    # Build index at startup
    engine = RAGEngine()
    n_docs = engine.build_index()
    if n_docs == 0:
        print("[RAG] No documents indexed. Server will return empty results.")
    else:
        print(f"[RAG] Server ready with {n_docs} indexed documents")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, data: dict, status: int = 200) -> None:
            try:
                body = json.dumps(data, ensure_ascii=False)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
            except Exception:
                pass

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path
            if path == "/health":
                self._send({
                    "status": "ok",
                    "indexed_docs": len(engine.documents),
                    "index_built": engine._index_built,
                })
            elif path == "/stats":
                sources = {}
                for doc in engine.documents:
                    src = doc.source.split("/")[0]
                    sources[src] = sources.get(src, 0) + 1
                self._send({"status": "ok", "documents": len(engine.documents), "sources": sources})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self) -> None:
            cl = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(cl) if cl > 0 else b"{}"
            data = json.loads(raw)

            path = self.path.rstrip("/")

            if path == "/retrieve":
                query = data.get("query", "")
                top_k = int(data.get("top_k", 3))
                min_score = float(data.get("min_score", 0.05))
                format_output = data.get("format", "json")  # json or prompt

                if not query:
                    self._send({"error": "query is required"}, 400)
                    return

                t0 = time.time()
                results = engine.retrieve(query, top_k=top_k, min_score=min_score)
                elapsed = (time.time() - t0) * 1000

                if format_output == "prompt":
                    formatted = engine.retrieve_formatted(query, top_k=top_k)
                    self._send({
                        "query": query,
                        "context": formatted,
                        "num_results": len(results),
                        "time_ms": round(elapsed, 1),
                    })
                else:
                    self._send({
                        "query": query,
                        "results": [
                            {
                                "text": doc.text[:500],
                                "source": doc.source,
                                "score": doc.score,
                                "metadata": doc.metadata,
                            }
                            for doc in results
                        ],
                        "time_ms": round(elapsed, 1),
                    })

            elif path == "/answer":
                query = data.get("query", "")
                if not query:
                    self._send({"error": "query is required"}, 400)
                    return
                answer = engine.retrieve_answer(query)
                self._send({"query": query, "answer": answer, "found": answer is not None})

            elif path == "/rebuild":
                n = engine.build_index()
                self._send({"status": "ok", "indexed_docs": n})

            else:
                self._send({"error": "not found"}, 404)

    server = ThreadedRAGServer(("0.0.0.0", port), Handler)
    print(f"[RAG] Server running on http://localhost:{port}")
    print(f"[RAG] Endpoints: /health, /stats, /retrieve, /answer, /rebuild")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[RAG] Shutting down")
        server.server_close()


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        # Build and print stats
        engine = RAGEngine()
        n = engine.build_index()
        print(f"\nIndexed {n} documents")
        while True:
            query = input("\nQuery (or q to quit): ").strip()
            if query.lower() in ("q", "quit"):
                break
            results = engine.retrieve(query, top_k=3)
            for r in results:
                print(f"  [{r.score:.3f}] ({r.source}) {r.text[:150]}")

    elif "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 8300
        run_server(port=port)

    else:
        run_server(port=8300)
