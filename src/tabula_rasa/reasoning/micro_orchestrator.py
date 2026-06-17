"""Micro-Orchestrator — lightweight router with short-term success memory.

Wraps the existing deterministic Router with:
1. Short-term memory of recent dispatch outcomes (success/fail per specialist)
2. Confidence tracking: if a specialist fails N times in a row, deprioritize it
3. Alternative routing: when the first choice fails, cascade to the next-best
4. Telemetry logging for per-specialist success rates over time

Does NOT replace the Router — wraps it. The Router's pattern-matching still
does the initial dispatch; the Orchestrator tracks what happens after and
learns which specialists to trust.
"""

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from tabula_rasa.router.router import Router

# ═══════════════════════════════════════════════════════════════════
# Outcome Memory
# ═══════════════════════════════════════════════════════════════════


class OutcomeRecord:
    """One recorded dispatch outcome."""

    def __init__(self, specialist: str, prompt: str, success: bool,
                 latency_ms: float, confidence: float = 1.0):
        self.specialist = specialist
        self.prompt = prompt
        self.success = success
        self.latency_ms = latency_ms
        self.confidence = confidence
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            'specialist': self.specialist,
            'prompt': self.prompt[:60],
            'success': self.success,
            'latency_ms': round(self.latency_ms, 1),
            'confidence': round(self.confidence, 3),
            'timestamp': round(self.timestamp, 1),
        }


# ═══════════════════════════════════════════════════════════════════
# Micro-Orchestrator
# ═══════════════════════════════════════════════════════════════════


class MicroOrchestrator:
    """Lightweight orchestrator with short-term success memory.

    Wraps a Router instance and adds:
    - Rolling window of recent outcomes (configurable size)
    - Per-specialist success rate tracking
    - Automatic fallback to next-best specialist on failure
    - Confidence-inertia: consecutive failures reduce confidence

    Args:
        router: Optional Router instance (creates one if not provided).
        memory_size: Number of recent outcomes to remember (default: 100).
        fail_threshold: Consecutive failures before deprioritizing (default: 3).
        confidence_decay: Factor to multiply confidence by on failure (default: 0.5).
        persist_path: Optional path to save/load memory across restarts.
    """

    def __init__(
        self,
        router: Optional[Router] = None,
        memory_size: int = 100,
        fail_threshold: int = 3,
        confidence_decay: float = 0.5,
        persist_path: Optional[Path] = None,
    ):
        self.router = router or Router()
        self.memory_size = memory_size
        self.fail_threshold = fail_threshold
        self.confidence_decay = confidence_decay
        self.persist_path = persist_path

        # Rolling window of recent outcomes
        self._history: deque[OutcomeRecord] = deque(maxlen=memory_size)

        # Per-specialist tracking
        self._specialist_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
            'total': 0,
            'successes': 0,
            'failures': 0,
            'consecutive_failures': 0,
            'success_rate': 1.0,
            'confidence': 1.0,
            'total_latency_ms': 0.0,
            'last_routed': 0.0,
        })

        self._total_routed = 0

        # Load persisted state if available
        if persist_path and persist_path.exists():
            self._load()

    # ── Routing ─────────────────────────────────────────────────

    def route(self, prompt: str) -> dict:
        """Route a prompt, tracking outcomes and applying confidence.

        Returns a dict with routing decision and stats:
            specialist, stage, confidence, alternative, history_summary
        """
        # Get the base routing decision
        base = self.router.route(prompt)
        specialist = base.get('specialist', 'unknown')
        stage = base.get('stage', 0)
        base_confidence = base.get('confidence', 1.0)

        # Apply confidence-weighted adjustment using short-term memory
        stats = self._specialist_stats[specialist]

        # If confidence is critically low AND we have an alternative, cascade
        effective_confidence = stats['confidence'] * base_confidence
        use_alternative = (
            effective_confidence < 0.3
            and stats['consecutive_failures'] >= self.fail_threshold
            and self._has_alternative(specialist)
        )

        if use_alternative:
            alt_specialist = self._find_alternative(specialist)
            if alt_specialist:
                alt_stats = self._specialist_stats[alt_specialist]
                result = {
                    'specialist': alt_specialist,
                    'original_specialist': specialist,
                    'stage': stage,
                    'confidence': max(alt_stats['confidence'], 0.5),
                    'alternative': True,
                    'reason': f'{specialist} confidence too low ({effective_confidence:.2f})',
                }
                # Log the re-route
                self._total_routed += 1
                return result

        # Normal route
        result = {
            'specialist': specialist,
            'stage': stage,
            'confidence': effective_confidence,
            'alternative': False,
            'reason': 'normal route',
        }
        self._total_routed += 1
        return result

    def _has_alternative(self, specialist: str) -> bool:
        """Check if there's another specialist that could handle this type."""
        return len(self.router.rules) > 1

    def _find_alternative(self, specialist: str) -> Optional[str]:
        """Find the next best specialist with acceptable confidence."""
        candidates = []
        for rule in sorted(self.router.rules, key=lambda r: -r.priority):
            s = rule.specialist
            if s == specialist:
                continue
            alt_stats = self._specialist_stats.get(s, {})
            candidates.append((s, alt_stats.get('confidence', 1.0)))

        # Pick the one with highest confidence
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0] if candidates else None

    # ── Outcome Recording ──────────────────────────────────────

    def record_success(self, specialist: str, prompt: str,
                       latency_ms: float = 0.0):
        """Record a successful dispatch."""
        stats = self._specialist_stats[specialist]
        stats['total'] += 1
        stats['successes'] += 1
        stats['consecutive_failures'] = 0
        stats['success_rate'] = stats['successes'] / max(1, stats['total'])
        stats['confidence'] = min(1.0, stats['confidence'] + 0.1)
        stats['total_latency_ms'] += latency_ms
        stats['last_routed'] = time.time()

        record = OutcomeRecord(specialist, prompt, True, latency_ms,
                               stats['confidence'])
        self._history.append(record)
        self._maybe_persist()

    def record_failure(self, specialist: str, prompt: str,
                       latency_ms: float = 0.0):
        """Record a failed dispatch."""
        stats = self._specialist_stats[specialist]
        stats['total'] += 1
        stats['failures'] += 1
        stats['consecutive_failures'] += 1
        stats['success_rate'] = stats['successes'] / max(1, stats['total'])
        stats['confidence'] *= self.confidence_decay
        stats['total_latency_ms'] += latency_ms
        stats['last_routed'] = time.time()

        record = OutcomeRecord(specialist, prompt, False, latency_ms,
                               stats['confidence'])
        self._history.append(record)
        self._maybe_persist()

    # ── Stats / Reporting ───────────────────────────────────────

    def get_specialist_stats(self) -> dict[str, dict]:
        """Return per-specialist success/confidence stats."""
        return dict(self._specialist_stats)

    def get_all_specialist_stats(self) -> dict:
        """Detailed stats for all specialists seen by the orchestrator."""
        return {
            name: {
                'total': s['total'],
                'successes': s['successes'],
                'failures': s['failures'],
                'success_rate': round(s['success_rate'], 3),
                'confidence': round(s['confidence'], 3),
                'consecutive_failures': s['consecutive_failures'],
                'avg_latency_ms': round(
                    s['total_latency_ms'] / max(1, s['total']), 1
                ),
            }
            for name, s in self._specialist_stats.items()
            if s['total'] > 0
        }

    def get_recent_history(self, n: int = 10) -> list[dict]:
        """Return the N most recent dispatch records."""
        return [r.to_dict() for r in list(self._history)[-n:]]

    def get_summary(self) -> dict:
        """Overall orchestrator summary."""
        stats = self.get_all_specialist_stats()
        total = sum(s['total'] for s in stats.values())
        successes = sum(s['successes'] for s in stats.values())
        return {
            'total_dispatches': total,
            'overall_success_rate': round(successes / max(1, total), 3),
            'specialist_count': len(stats),
            'memory_utilization': f"{len(self._history)}/{self.memory_size}",
            'specialists': stats,
        }

    # ── Persistence ─────────────────────────────────────────────

    def _maybe_persist(self):
        if self.persist_path and self._total_routed % 10 == 0:
            self._save()

    def _save(self):
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'stats': {
                name: dict(s) for name, s in self._specialist_stats.items()
            },
            'total_routed': self._total_routed,
            'history': [r.to_dict() for r in list(self._history)[-20:]],
        }
        self.persist_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            self._total_routed = data.get('total_routed', 0)
            for name, sdata in data.get('stats', {}).items():
                self._specialist_stats[name] = sdata
        except (json.JSONDecodeError, OSError):
            pass  # Corrupted file — start fresh


# ═══════════════════════════════════════════════════════════════════
# Convenience wrapper: route_and_track
# ═══════════════════════════════════════════════════════════════════


def route_and_track(orchestrator: MicroOrchestrator, prompt: str,
                    specialist_fn=None) -> dict:
    """Route a prompt, execute the specialist function, and record outcome.

    Args:
        orchestrator: MicroOrchestrator instance.
        prompt: The user's prompt.
        specialist_fn: Callable that takes (specialist, prompt) and returns
            (success: bool, latency_ms: float). If None, only routes.

    Returns:
        Dict with routing decision and outcome.
    """
    decision = orchestrator.route(prompt)
    specialist = decision['specialist']

    if specialist_fn is None:
        return decision

    try:
        import time
        t0 = time.time()
        success = specialist_fn(specialist, prompt)
        elapsed = (time.time() - t0) * 1000

        if success:
            orchestrator.record_success(specialist, prompt, elapsed)
        else:
            orchestrator.record_failure(specialist, prompt, elapsed)

        decision['executed'] = True
        decision['success'] = success
        decision['latency_ms'] = round(elapsed, 1)

        # If the first specialist failed and we have a fallback, try it
        if not success:
            alt_decision = orchestrator.route(prompt)
            alt_specialist = alt_decision['specialist']
            if alt_specialist != specialist:
                t0 = time.time()
                alt_success = specialist_fn(alt_specialist, prompt)
                elapsed = (time.time() - t0) * 1000
                if alt_success:
                    orchestrator.record_success(alt_specialist, prompt, elapsed)
                    decision['fallback_specialist'] = alt_specialist
                    decision['fallback_success'] = True
                else:
                    orchestrator.record_failure(alt_specialist, prompt, elapsed)
                    decision['fallback_success'] = False

    except Exception as e:
        orchestrator.record_failure(specialist, prompt)
        decision['error'] = str(e)
        decision['success'] = False

    return decision


# ═══════════════════════════════════════════════════════════════════
# CLI / Quick Test
# ═══════════════════════════════════════════════════════════════════

def main():
    """Test the micro-orchestrator with example prompts."""
    orchestrator = MicroOrchestrator(memory_size=50)
    print("Micro-Orchestrator Test")
    print("=" * 50)

    test_prompts = [
        "12+34=",
        "reverse('hello')",
        "is_a(dog, mammal)",
        "I have 5 apples and eat 2",
        "grammar('She go to school')",
        "What is the meaning of life?",
    ]

    for prompt in test_prompts:
        decision = orchestrator.route(prompt)
        print(f"\n  Prompt: {prompt}")
        print(f"  Route:  {decision['specialist']} (confidence={decision['confidence']:.2f})")
        if decision.get('alternative'):
            print(f"  (was: {decision.get('original_specialist')} — confidence too low)")

        # Simulate success/failure
        if decision['specialist'] != 'math_general':
            orchestrator.record_success(decision['specialist'], prompt, latency_ms=50)
        else:
            # Math specialist sometimes fails
            import random
            if random.random() < 0.3:
                orchestrator.record_failure(decision['specialist'], prompt)
            else:
                orchestrator.record_success(decision['specialist'], prompt)

    print(f"\n{'=' * 50}")
    print("Summary:")
    summary = orchestrator.get_summary()
    print(f"  Total dispatches: {summary['total_dispatches']}")
    print(f"  Overall success rate: {summary['overall_success_rate']*100:.0f}%")
    for name, s in summary['specialists'].items():
        print(f"  {name:25s}: {s['success_rate']*100:.0f}% success, "
              f"confidence={s['confidence']:.2f}, "
              f"consecutive_fails={s['consecutive_failures']}")

    print("\n  Recent history:")
    for r in orchestrator.get_recent_history(5):
        icon = "✓" if r['success'] else "✗"
        print(f"    {icon} {r['specialist']:25s} conf={r['confidence']:.2f}")


if __name__ == "__main__":
    main()
