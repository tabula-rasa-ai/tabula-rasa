"""
Pythagoras — The Nightly Examination of Conscience.

References the Golden Verses of Pythagoras:
  "Never suffer sleep to close thy eyelids, till thrice thou hast
   mentally reviewed the transactions of the day.
   Where have I transgressed? What have I done? What duty have I left undone?"

In the AI context, this is the metacognitive audit layer between
Hippocampus (daily experience) and Neocortex (long-term consolidation).

Flow:
  1. Hippocampus retrieves today's unconsolidated experiences.
  2. Pythagoras reviews three questions.
  3. Purified memories are passed to Neocortex EWC.
  4. Neglected duties are sent to the curriculum queue for tomorrow.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

try:
    from tabula_rasa.cognitive.logic_verifier import LogicVerifier
    HAS_VERIFIER = True
except ImportError:
    HAS_VERIFIER = False

# ─── Curriculum Queue ───────────────────────────────────────────────

CURRICULUM_FILE = Path('memory/tomorrow_curriculum.json')
SESSION_MEMORY_FILE = Path('session_memory.json')


def _load_curriculum() -> list:
    if CURRICULUM_FILE.exists():
        try:
            return json.loads(CURRICULUM_FILE.read_text())
        except:
            return []
    return []


def _save_curriculum(items: list):
    CURRICULUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRICULUM_FILE.write_text(json.dumps(items, indent=2))


def add_tomorrow_focus(topic: str, reason: str = ""):
    """Add a neglected topic to tomorrow's study plan."""
    items = _load_curriculum()
    # Deduplicate
    for item in items:
        if item['topic'].lower() == topic.lower():
            item['count'] = item.get('count', 0) + 1
            item['last_seen'] = time.time()
            _save_curriculum(items)
            return
    items.append({
        'topic': topic,
        'reason': reason,
        'count': 1,
        'added': time.time(),
        'last_seen': time.time(),
    })
    _save_curriculum(items)


def get_tomorrow_focus() -> list:
    """Get today's neglected duties for synthetic data generation."""
    return _load_curriculum()


def clear_tomorrow_focus():
    """Clear after generating training data."""
    CURRICULUM_FILE.write_text('[]')


# ─── Session Memory Integration ─────────────────────────────────────

def _load_session_memory() -> list:
    if SESSION_MEMORY_FILE.exists():
        try:
            return json.loads(SESSION_MEMORY_FILE.read_text())
        except:
            return []
    return []


# ─── The Three Questions ─────────────────────────────────────────────


def _question_1_transgressions(text: str, verifier) -> Optional[str]:
    """
    Q1: Where have I transgressed?
    AI translation: The Logic Audit.
    If the experience contains logical reasoning, verify it.
    If Z3 detects a fallacy, the model 'transgressed'.
    Returns the fallacy name if found, None if clean.
    """
    if not verifier:
        return None

    # Only check texts that look like logical arguments
    indicators = ['therefore', 'because', 'if', 'then', 'all', 'implies',
                  'since', 'hence', 'thus', 'consequently', 'follows',
                  'syllogism', 'premise', 'conclusion', 'valid', 'proof']

    has_logic = any(ind in text.lower() for ind in indicators)
    if not has_logic:
        return None

    # Run the debate argument scoring
    try:
        score = verifier.score_debate_argument(text)
        if score and score.get('fallacies'):
            return str(score['fallacies'][0])
    except Exception:
        pass

    # Also check plain syllogism verification
    try:
        # Try to extract premise/conclusion patterns
        for pattern in [
            r'if\s+(.+?)\s+then\s+(.+?)[\.,]',
            r'All\s+(.+?)\s+are\s+(.+?)[\.,]',
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result = verifier.verify_syllogism(m.group(0), text)
                if result and result.get('verdict') == 'refuted':
                    return 'logical contradiction'
    except Exception:
        pass

    return None


def _question_2_synthesis(experiences: list, cluster_threshold: float = 0.3) -> list:
    """
    Q2: What have I done?
    AI translation: The Synthesis.
    Cluster similar high-surprise experiences into 'Golden Examples'
    to save CPU cycles during EWC training.
    """
    if not experiences:
        return []

    # Simple text-similarity clustering: group by shared keywords
    clusters = {}  # keyword → [experiences]

    for exp in experiences:
        text = (exp.get('input_text', '') + ' ' + exp.get('output_text', '')).lower()
        words = set(re.findall(r'\b[a-z0-9]+\b', text))
        # Skip very short texts
        if len(words) < 3:
            clusters.setdefault('_misc', []).append(exp)
            continue
        # Find best matching cluster
        best_cluster = None
        best_overlap = 0
        for key, cluster_words in clusters.items():
            if key == '_misc':
                continue
            overlap = len(words & cluster_words) / len(words | cluster_words)
            if overlap > cluster_threshold and overlap > best_overlap:
                best_cluster = key
                best_overlap = overlap

        if best_cluster:
            clusters[best_cluster].append(exp)
            # Update cluster words with union
            clusters['__words__' + best_cluster] = words | clusters.get('__words__' + best_cluster, set())
        else:
            key = str(hash(frozenset(words)))[:8]
            clusters[key] = [exp]
            clusters['__words__' + key] = words

    # For each cluster, keep only the highest-surprise example
    golden = []
    for key, cluster_exps in clusters.items():
        if key.startswith('__words__') or key == '_misc':
            continue
        # Sort by prediction_error descending, take top 1
        cluster_exps.sort(key=lambda e: e.get('prediction_error', 0), reverse=True)
        golden.append(cluster_exps[0])  # The 'Golden Example'

    # Also include everything from misc (unclusterable)
    if '_misc' in clusters:
        golden.extend(clusters['_misc'])

    print(f'  [Pythagoras] Clustered {len(experiences)} experiences into '
          f'{len(golden)} golden examples ({(1 - len(golden)/max(len(experiences),1))*100:.0f}% reduction)')
    return golden


def _question_3_duty(experiences: list) -> list:
    """
    Q3: What duty is neglected?
    AI translation: The Epistemic Gap.
    Find queries that hit unknown tokens or routing failures.
    These become tomorrow's curriculum.
    """
    neglected = []

    for exp in experiences:
        text = exp.get('input_text', '') + ' ' + exp.get('output_text', '')

        # Check for <UNK> tokens
        if '<UNK>' in text:
            neglected.append({
                'topic': text[:80],
                'reason': 'Unknown token encountered',
            })

        # Check for router failure flag
        if exp.get('router_failed'):
            neglected.append({
                'topic': exp.get('input_text', '')[:80],
                'reason': 'Router could not dispatch',
            })

        # Check for correction experiences (user taught the right answer)
        if exp.get('is_correction'):
            neglected.append({
                'topic': exp.get('input_text', '')[:80],
                'reason': 'User had to correct — weak knowledge',
            })

    # Also check session memory for user-reported issues
    try:
        session_entries = _load_session_memory()
        for entry in session_entries:
            if entry.get('correct') in ('no', 'partially'):
                neglected.append({
                    'topic': entry.get('question', '')[:80],
                    'reason': f"User reported: {entry.get('wrong', 'incorrect answer')}",
                })
    except Exception:
        pass

    return neglected


# ─── The Review ─────────────────────────────────────────────────────


def pythagorean_review(experiences: list) -> dict:
    """
    The nightly examination of conscience.
    Runs all three Pythagorean questions on today's experiences.

    Args:
        experiences: List of dicts from hippocampus (unconsolidated).

    Returns:
        dict with:
          - 'purified': List of logically sound, golden experiences for EWC.
          - 'neglected': List of topics needing synthetic training data.
          - 'transgressions': Count of purged fallacious memories.
          - 'stats': Summary statistics.
    """
    verifier = LogicVerifier() if HAS_VERIFIER else None

    purified = []
    transgressions = 0
    neglected = []

    print('  [Pythagoras] Beginning the examination of conscience...')

    # ─── Q1: Transgressions (Logic Audit / Purging) ───
    for exp in experiences:
        text = exp.get('input_text', '')
        transgression = _question_1_transgressions(text, verifier)
        if transgression:
            print(f'  [Pythagoras] Transgression found: {transgression}')
            print(f'      Purging: {text[:60]}...')
            transgressions += 1
            continue  # Purge this memory — do not consolidate
        purified.append(exp)

    # ─── Q3: Duty (Epistemic Gap) — run BEFORE Q2 so we capture raw data ───
    neglected = _question_3_duty(experiences)

    # ─── Q2: Synthesis (Cluster into Golden Examples) ───
    golden = _question_2_synthesis(purified)

    # ─── Send neglected duties to curriculum queue ───
    for item in neglected:
        add_tomorrow_focus(item['topic'], item['reason'])

    # ─── Statistics ───
    stats = {
        'total_input': len(experiences),
        'transgressions_purged': transgressions,
        'golden_examples': len(golden),
        'neglected_duties': len(neglected),
        'compression_ratio': f"{len(golden)}/{len(experiences)}"
        if len(experiences) > 0 else "0/0",
    }

    print('  [Pythagoras] Review complete.')
    print(f'      Purged {transgressions} fallacious memories.')
    print(f'      Distilled {len(golden)} golden examples.')
    print(f'      Flagged {len(neglected)} neglected duties for tomorrow.')

    # ─── BPE: Self-Seeded Merge Learning ───
    # During the nightly audit, learn new token merges from the model's own
    # verified monologue (Socratic debates + corrected outputs).
    try:
        from bpe_tokenizer import BPETokenizer, learn_from_verified_log
        tok_path = Path('specialists/math/general/tokenizer.json')
        tok = None
        if tok_path.exists():
            tok = BPETokenizer.load(str(tok_path))
        new_tok, n_merges = learn_from_verified_log(max_merges=30, min_frequency=2, tokenizer=tok)
        if n_merges > 0:
            new_tok.save(str(tok_path))
            print(f'  [BPE] Learned {n_merges} new merges — tokenizer growing')
    except Exception as e:
        print(f'  [BPE] Skip: {e}')

    return {
        'purified': golden,
        'neglected': neglected,
        'transgressions': transgressions,
        'stats': stats,
    }
