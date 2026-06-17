"""Prepared Slate Orchestrator — routes queries to the right engine + inference strategy.

The paper calls for a "prepared slate" that combines:
- Base MathTransformer (with RASA, MoE) for arithmetic
- LoRA adapters for specialist switching
- RAG for factual grounding
- MCTS inference for hard problems
- Confidence-based fallback chaining

Usage:
    from tabula_rasa.orchestrator import PreparedSlateOrchestrator

    orch = PreparedSlateOrchestrator(
        math_model=model,
        tokenizer=tok,
        rag_engine=rag,
        llm_url="http://localhost:8201",
    )
    result = orch.answer("12+34")
    result = orch.answer("What is a transformer?")
"""

from __future__ import annotations

import json
import math as _math
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class AnswerResult:
    """Result from the orchestrator."""

    query: str
    answer: str
    strategy: str  # 'math_greedy', 'math_mcts', 'rag_llm', 'rag_direct', 'fallback'
    confidence: float  # 0.0 - 1.0
    time_ms: float
    details: dict = field(default_factory=dict)


class PreparedSlateOrchestrator:
    """Orchestrator that selects the best strategy for each query.

    Strategies (in priority order):
    1. Direct RAG answer — if the question matches a known Q&A exactly
    2. Math greedy — for arithmetic expressions with high confidence
    3. Math MCTS — for arithmetic expressions (fallback from greedy)
    4. RAG + LLM — for general questions with retrieved context
    5. LLM only — fallback for anything else

    Args:
        math_model: Trained MathTransformer (optional, for arithmetic).
        tokenizer: MathTokenizer (required if math_model provided).
        rag_engine: RAGEngine instance (optional, for factual retrieval).
        llm_url: URL of the LLM inference server (optional).
        lora_paths: Dict of specialist_name -> lora_adapter_path.
        mcts_simulations: MCTS rollouts for hard problems (default 16).
        confidence_threshold: Minimum confidence for greedy answer (default 0.6).
    """

    def __init__(
        self,
        math_model: Any = None,
        tokenizer: Any = None,
        rag_engine: Any = None,
        llm_url: str | None = None,
        lora_paths: dict[str, str] | None = None,
        mcts_simulations: int = 16,
        confidence_threshold: float = 0.6,
    ):
        self.math_model = math_model
        self.tokenizer = tokenizer
        self.rag = rag_engine
        self.llm_url = llm_url
        self.lora_paths = lora_paths or {}
        self.mcts_simulations = mcts_simulations
        self.confidence_threshold = confidence_threshold

        # Lazy-loaded MCTS engine
        self._mcts_engine = None

        # Track strategy usage for analysis
        self.strategy_counts: dict[str, int] = {}
        self.total_queries = 0

    # ── Query type detection ────────────────────────────────────

    MATH_PATTERN = re.compile(r"^\s*(\d+\s*[+\-*/]\s*\d+)")
    MATH_KEYWORDS = re.compile(
        r"\b(?:add|sum|plus|minus|times|multiply|divide|calculate|solve|what\s+is\s+\d)", re.I
    )
    GREETING_PATTERN = re.compile(r"^\s*(?:hi|hello|hey|greetings|howdy|good\s+(?:morning|afternoon|evening))\b", re.I)
    DEFINITION_PATTERN = re.compile(
        r"\b(?:what\s+is|what\s+are|define|explain|what\s+does.*mean)\b", re.I
    )

    def _is_math_query(self, query: str) -> bool:
        """Detect if query is a math problem."""
        if self.MATH_PATTERN.match(query):
            return True
        if self.MATH_KEYWORDS.search(query):
            return True
        return False

    def _is_greeting(self, query: str) -> bool:
        return bool(self.GREETING_PATTERN.match(query))

    def _is_definition(self, query: str) -> bool:
        return bool(self.DEFINITION_PATTERN.match(query))

    def _extract_math_expr(self, query: str) -> str | None:
        """Extract the core arithmetic expression from a query."""
        # Direct match: "12+34"
        m = self.MATH_PATTERN.match(query)
        if m:
            return m.group(1).replace(" ", "")
        # "What is 12+34?"
        m = re.search(r"what\s+is\s+(\d+\s*[+\-*/]\s*\d+)", query, re.I)
        if m:
            return m.group(1).replace(" ", "")
        # "calculate 12+34"
        m = re.search(r"(?:calculate|solve)\s+(\d+\s*[+\-*/]\s*\d+)", query, re.I)
        if m:
            return m.group(1).replace(" ", "")
        return None

    # ── Inference strategies ────────────────────────────────────

    def _answer_math_greedy(self, expr: str) -> AnswerResult | None:
        """Solve math using greedy decoding (fast path)."""
        if self.math_model is None or self.tokenizer is None:
            return None
        t0 = time.time()
        prompt = f"{expr}="
        try:
            full = self.math_model.generate(
                self.tokenizer, prompt, max_new_tokens=12, temperature=0.0, top_k=1
            )
            elapsed = (time.time() - t0) * 1000
            if "=" in full:
                pred = full.split("=")[-1].strip()
                pred = "".join(c for c in pred if c.isdigit() or c == "-")
                if pred:
                    # Confidence: estimate from logits if available
                    confidence = self._estimate_confidence(expr)
                    return AnswerResult(
                        query=expr,
                        answer=pred,
                        strategy="math_greedy",
                        confidence=confidence,
                        time_ms=round(elapsed, 1),
                        details={"raw_output": full},
                    )
        except Exception as e:
            return AnswerResult(
                query=expr, answer="", strategy="math_greedy",
                confidence=0.0, time_ms=0, details={"error": str(e)},
            )
        return None

    def _estimate_confidence(self, expr: str) -> float:
        """Estimate confidence in greedy answer using value head or entropy."""
        if self.math_model is None or self.tokenizer is None:
            return 0.5
        try:
            import torch
            device = next(self.math_model.parameters()).device
            prompt = f"{expr}="
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            inp = torch.tensor([ids], device=device)
            with torch.no_grad():
                logits, _, value = self.math_model(inp, inp)
            # Use value head if available
            if hasattr(self.math_model, "value_head") and self.math_model.value_head is not None:
                v = float(value.item())
                return max(0.0, min(1.0, (v + 1.0) / 2.0))  # normalize [-1,1] → [0,1]
            # Fallback: softmax probability of argmax token
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_p = float(probs.max().item())
            return top_p
        except Exception:
            return 0.5

    def _answer_math_mcts(self, expr: str) -> AnswerResult | None:
        """Solve math using MCTS inference (thorough path)."""
        if self.math_model is None or self.tokenizer is None:
            return None
        t0 = time.time()
        try:
            from tabula_rasa.mcts_inference import MCTSInference

            mcts = MCTSInference(
                model=self.math_model,
                tokenizer=self.tokenizer,
                num_simulations=self.mcts_simulations,
                top_k=10,
                temperature=0.5,
                max_new_tokens=12,
            )
            answer, stats = mcts.generate(expr)
            elapsed = (time.time() - t0) * 1000
            # MCTS confidence: tree visit concentration
            if stats.get("tree_size", 0) > 0:
                confidence = min(1.0, stats["tree_size"] / self.mcts_simulations)
            else:
                confidence = 0.3
            return AnswerResult(
                query=expr,
                answer=answer,
                strategy="math_mcts",
                confidence=min(confidence, 0.95),
                time_ms=round(elapsed, 1),
                details={"mcts_stats": stats},
            )
        except Exception as e:
            return AnswerResult(
                query=expr, answer="", strategy="math_mcts",
                confidence=0.0, time_ms=0, details={"error": str(e)},
            )

    def _answer_rag_direct(self, query: str) -> AnswerResult | None:
        """Try to answer directly from RAG knowledge base (no LLM)."""
        if self.rag is None:
            return None
        t0 = time.time()
        try:
            answer = self.rag.retrieve_answer(query)
            elapsed = (time.time() - t0) * 1000
            if answer:
                return AnswerResult(
                    query=query,
                    answer=answer,
                    strategy="rag_direct",
                    confidence=0.9,
                    time_ms=round(elapsed, 1),
                )
        except Exception:
            pass
        return None

    def _answer_rag_llm(self, query: str) -> AnswerResult | None:
        """Answer using LLM with RAG context injection."""
        if self.llm_url is None:
            return None
        t0 = time.time()
        try:
            payload = json.dumps({
                "prompt": query,
                "use_rag": self.rag is not None,
                "max_tokens": 128,
                "temperature": 0.3,
            }).encode()
            req = urllib.request.Request(
                f"{self.llm_url}/",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
            elapsed = (time.time() - t0) * 1000
            response = result.get("response", "")
            rag_used = result.get("rag_used", False)
            return AnswerResult(
                query=query,
                answer=response,
                strategy="rag_llm" if rag_used else "llm_only",
                confidence=0.7 if response else 0.0,
                time_ms=round(elapsed, 1),
                details={"rag_used": rag_used},
            )
        except Exception as e:
            return AnswerResult(
                query=query, answer="", strategy="rag_llm",
                confidence=0.0, time_ms=0, details={"error": str(e)},
            )

    def _answer_fallback(self, query: str) -> AnswerResult:
        """Last-resort fallback."""
        if self._is_greeting(query):
            return AnswerResult(
                query=query,
                answer="Hello! I'm Tabula Rasa. How can I help you today?",
                strategy="fallback_rule",
                confidence=0.8,
                time_ms=0,
            )
        return AnswerResult(
            query=query,
            answer="I'm not sure. I'm learning to answer this type of question.",
            strategy="fallback_unknown",
            confidence=0.1,
            time_ms=0,
        )

    # ── Main entry point ────────────────────────────────────────

    def answer(self, query: str, force_strategy: str | None = None) -> AnswerResult:
        """Answer a query using the best available strategy.

        Args:
            query: User's question or math expression.
            force_strategy: If set, only use this strategy
                ('math_greedy', 'math_mcts', 'rag_direct', 'rag_llm').

        Returns:
            AnswerResult with answer, strategy used, confidence, and timing.
        """
        self.total_queries += 1
        query = query.strip()

        # ── Strategy selection ────────────────────────────────────
        strategies: list[tuple[str, Callable, list[str]]] = []

        if force_strategy:
            # Single forced strategy
            strategy_map = {
                "math_greedy": ("math_greedy", lambda: self._answer_math_greedy(
                    self._extract_math_expr(query) or query
                ), []),
                "math_mcts": ("math_mcts", lambda: self._answer_math_mcts(
                    self._extract_math_expr(query) or query
                ), []),
                "rag_direct": ("rag_direct", lambda: self._answer_rag_direct(query), []),
                "rag_llm": ("rag_llm", lambda: self._answer_rag_llm(query), []),
            }
            if force_strategy in strategy_map:
                strategies = [strategy_map[force_strategy]]
        else:
            # Build strategy list based on query type
            if self._is_math_query(query):
                expr = self._extract_math_expr(query)
                if expr:
                    strategies = [
                        ("math_greedy", lambda e=expr: self._answer_math_greedy(e), []),
                        ("math_mcts", lambda e=expr: self._answer_math_mcts(e), []),
                    ]

            # Always try RAG direct (cheap, fast)
            strategies.append(("rag_direct", lambda: self._answer_rag_direct(query), []))

            # Try RAG + LLM for general questions
            strategies.append(("rag_llm", lambda: self._answer_rag_llm(query), []))

            # Fallback
            strategies.append(("fallback", lambda: self._answer_fallback(query), []))

        # ── Try each strategy in order ────────────────────────────
        best: AnswerResult | None = None
        for strategy_name, strategy_fn, _ in strategies:
            try:
                result = strategy_fn()
                if result is None:
                    continue
                if result.answer:
                    # Accept if confidence above threshold, or if it's the last resort
                    is_last = strategy_name == strategies[-1][0]
                    if result.confidence >= self.confidence_threshold or is_last:
                        best = result
                        break
                    # Keep as best-so-far if we haven't found anything better
                    if best is None or result.confidence > best.confidence:
                        best = result
            except Exception as e:
                continue

        if best is None:
            best = self._answer_fallback(query)

        # Track strategy usage
        self.strategy_counts[best.strategy] = self.strategy_counts.get(best.strategy, 0) + 1

        return best

    def stats(self) -> dict:
        """Return orchestrator usage statistics."""
        return {
            "total_queries": self.total_queries,
            "strategy_counts": dict(self.strategy_counts),
            "strategies_available": {
                "math_model": self.math_model is not None,
                "rag": self.rag is not None,
                "llm": self.llm_url is not None,
                "lora": len(self.lora_paths) > 0,
            },
        }


def run_server(port: int = 8400, **orchestrator_kwargs: Any) -> None:
    """Run the orchestrator as an HTTP server.

    Args:
        port: HTTP port (default 8400).
        **orchestrator_kwargs: Passed to PreparedSlateOrchestrator.
    """
    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from socketserver import ThreadingMixIn
    except ImportError:
        print("HTTP server not available")
        return

    orch = PreparedSlateOrchestrator(**orchestrator_kwargs)

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

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
            path = self.path.rstrip("/")
            if path == "/health":
                self._send({"status": "ok", "stats": orch.stats()})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self) -> None:
            cl = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(cl) if cl > 0 else b"{}"
            data = json.loads(raw)
            path = self.path.rstrip("/")

            if path == "/answer":
                query = data.get("query", "")
                force = data.get("strategy", None)
                if not query:
                    self._send({"error": "query is required"}, 400)
                    return
                result = orch.answer(query, force_strategy=force)
                self._send({
                    "query": result.query,
                    "answer": result.answer,
                    "strategy": result.strategy,
                    "confidence": round(result.confidence, 3),
                    "time_ms": result.time_ms,
                })
            elif path == "/stats":
                self._send(orch.stats())
            else:
                self._send({"error": "not found"}, 404)

    server = ThreadedServer(("0.0.0.0", port), Handler)
    print(f"[Orchestrator] Server on http://localhost:{port}")
    print(f"[Orchestrator] Endpoints: /health, /answer, /stats")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down")
        server.server_close()


if __name__ == "__main__":
    import sys

    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 8400
        run_server(port=port)
    else:
        # Interactive mode
        orch = PreparedSlateOrchestrator(
            llm_url=sys.argv[1] if len(sys.argv) > 1 else None,
        )
        print("Prepared Slate Orchestrator — interactive mode")
        print("Type a question or math expression. 'q' to quit.")
        print()
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("q", "quit", "exit"):
                break
            if not q:
                continue
            result = orch.answer(q)
            print(f"  [{result.strategy}] {result.answer}")
            print(f"  (confidence={result.confidence:.2f}, {result.time_ms:.0f}ms)")
        print(f"\nStats: {orch.stats()}")
