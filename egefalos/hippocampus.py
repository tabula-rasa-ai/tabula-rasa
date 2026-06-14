"""Hippocampus — Tiered 3-layer memory architecture (Active → Working → Long-term).

When the model encounters surprising data (high prediction error),
it's instantly saved in Active tier. Promoted to Working on repeated access,
then to Long-term for permanent storage with decay.

Tiers:
    active   — Short-term / current session. High surprise threshold. Auto-pruned after consolidation.
    working  — Task-relevant / multi-session. Tracked by access count + last access time.
    longterm — Cross-session knowledge. Decayed and demoted if unused.
"""

import sqlite3
import json
import time
import math
from pathlib import Path
from typing import Optional
import torch

DB_PATH = Path('memory/hippocampus.db')


def get_db() -> sqlite3.Connection:
    """Get or create the hippocampus database with tiered schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Core table — add new columns if missing (backward-compatible ALTER)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            input_text TEXT NOT NULL,
            output_text TEXT,
            prediction_error REAL NOT NULL,
            skill TEXT DEFAULT 'general',
            is_correction INTEGER DEFAULT 0,
            tier TEXT DEFAULT 'active',
            access_count INTEGER DEFAULT 0,
            last_access REAL DEFAULT 0,
            consolidated INTEGER DEFAULT 0
        )
    """)

    # Add missing columns for databases created with old schema
    for col, col_type in [('tier', 'TEXT DEFAULT \'active\''),
                          ('access_count', 'INTEGER DEFAULT 0'),
                          ('last_access', 'REAL DEFAULT 0')]:
        try:
            conn.execute(f"ALTER TABLE experiences ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_tier_consolidated
        ON experiences(tier, consolidated)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_tier_timestamp
        ON experiences(tier, timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_tier_access
        ON experiences(tier, last_access)
    """)
    conn.commit()
    return conn


# ─── CRUD ────────────────────────────────────────────────────────────

def store_experience(input_text: str, prediction_error: float,
                     output_text: str = None, skill: str = 'general',
                     is_correction: bool = False, tier: str = 'active'):
    """Store a surprising experience in the specified tier (default: active)."""
    conn = get_db()
    ts = time.time()
    conn.execute("""
        INSERT INTO experiences (timestamp, input_text, output_text,
                                 prediction_error, skill, is_correction, tier)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ts, input_text, output_text,
          prediction_error, skill, 1 if is_correction else 0, tier))
    conn.commit()
    conn.close()


def get_unconsolidated(limit: int = 100, tier: str = 'active') -> list[dict]:
    """Get experiences that haven't been consolidated yet, optionally filtered by tier."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, input_text, output_text, prediction_error,
               skill, is_correction, tier, access_count, last_access
        FROM experiences WHERE consolidated = 0 AND tier = ?
        ORDER BY prediction_error DESC
        LIMIT ?
    """, (tier, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_old_memories(limit: int = 50, tier: str = 'longterm',
                     use_per: bool = True) -> list[dict]:
    """Get consolidated memories from a tier, optionally using Prioritized Experience Replay.

    When use_per=True (default), samples are weighted by prediction_error (surprise),
    so higher-error experiences are more likely to be replayed.
    When use_per=False, uses uniform random sampling (ORDER BY RANDOM()).

    Args:
        limit: Maximum number of memories to return.
        tier: Memory tier to sample from ('active', 'working', 'longterm').
        use_per: If True, use prediction_error-weighted sampling. If False, uniform random.

    Returns:
        List of experience dicts.
    """
    conn = get_db()
    total = conn.execute(
        "SELECT COUNT(*) FROM experiences WHERE consolidated = 1 AND tier = ?",
        (tier,)
    ).fetchone()[0]
    if total == 0:
        conn.close()
        return []

    # ── Uniform random sampling (legacy) ──────────────────────────
    if not use_per:
        rows = conn.execute("""
            SELECT id, input_text, output_text, prediction_error,
                   skill, is_correction, tier, access_count, last_access
            FROM experiences WHERE consolidated = 1 AND tier = ?
            ORDER BY RANDOM() LIMIT ?
        """, (tier, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Prioritized Experience Replay (PER) ───────────────────────
    rows = conn.execute("""
        SELECT id, input_text, output_text, prediction_error,
               skill, is_correction, tier, access_count, last_access
        FROM experiences WHERE consolidated = 1 AND tier = ?
    """, (tier,)).fetchall()
    conn.close()

    if not rows:
        return []

    records = [dict(r) for r in rows]
    actual_limit = min(limit, len(records))

    # Build weights: clamp negative/zero errors to a small epsilon
    # so every experience has a non-zero chance of being sampled.
    errors = torch.tensor(
        [max(r['prediction_error'], 1e-8) for r in records]
    )
    weights = torch.softmax(errors, dim=0)

    indices = torch.multinomial(weights, actual_limit, replacement=False)
    return [records[i] for i in indices.tolist()]


def mark_consolidated(ids: list[int]):
    """Mark experiences as consolidated after training."""
    if not ids:
        return
    conn = get_db()
    placeholders = ','.join('?' * len(ids))
    conn.execute(f"""
        UPDATE experiences SET consolidated = 1
        WHERE id IN ({placeholders})
    """, ids)
    conn.commit()
    conn.close()


# ─── Tier Promotion / Demotion ───────────────────────────────────────

def promote_to_working(ids: list[int]):
    """Move experiences from active → working tier."""
    if not ids:
        return
    conn = get_db()
    ts = time.time()
    placeholders = ','.join('?' * len(ids))
    conn.execute(f"""
        UPDATE experiences
        SET tier = 'working', access_count = access_count + 1, last_access = ?
        WHERE id IN ({placeholders}) AND tier = 'active'
    """, [ts] + ids)
    conn.commit()
    conn.close()


def promote_to_longterm(ids: list[int]):
    """Move experiences from working → longterm tier."""
    if not ids:
        return
    conn = get_db()
    ts = time.time()
    placeholders = ','.join('?' * len(ids))
    conn.execute(f"""
        UPDATE experiences
        SET tier = 'longterm', access_count = access_count + 1, last_access = ?
        WHERE id IN ({placeholders}) AND tier = 'working'
    """, [ts] + ids)
    conn.commit()
    conn.close()


def decay_memories(min_access: int = 3, max_age_days: int = 30,
                   decay_rate: float = 0.5,
                   demote_working: bool = False,
                   cleanup_active: bool = False):
    """Full synaptic decay across all 3 memory tiers.

    Always applies:
      - Longterm → Working demotion (if access_count < min_access and old)
      - Unconsolidated longterm deletion (if very old and never consolidated)
      - Synaptic decay: gradual access_count reduction across all tiers

    When enabled (opt-in):
      - demote_working=True: Working → Active demotion
      - cleanup_active=True:  Active cleanup (consolidated→Working,
                               unconsolidated→deleted)

    Args:
        min_access: Minimum access count for retention.
        max_age_days: Age in days beyond which a memory may be demoted/deleted.
        decay_rate: Synaptic decay factor applied each cycle (default 0.5).
                    Reduces access_count by int(decay_rate * max_age_days)
                    across all tiers each cycle to simulate gradual forgetting.
        demote_working: If True, demote unused Working memories back to Active.
        cleanup_active: If True, promote consolidated old Active→Working,
                        delete unconsolidated old Active entries.
    """
    conn = get_db()
    cutoff = time.time() - (max_age_days * 86400)

    # ── Synaptic decay: gradually reduce access_count across all tiers ──
    decay_amount = int(decay_rate * max_age_days)
    if decay_amount > 0:
        conn.execute("""
            UPDATE experiences
            SET access_count = MAX(0, access_count - ?)
            WHERE access_count > 0
        """, (decay_amount,))
        conn.commit()

    # ── Demote longterm → working (existing behavior) ────────────────
    demoted_longterm = conn.execute("""
        UPDATE experiences
        SET tier = 'working'
        WHERE tier = 'longterm'
          AND access_count < ?
          AND timestamp < ?
    """, (min_access, cutoff)).rowcount

    # ── Delete very old unconsolidated longterm (existing behavior) ──
    deleted_longterm = conn.execute("""
        DELETE FROM experiences
        WHERE tier = 'longterm'
          AND consolidated = 0
          AND timestamp < ?
    """, (cutoff,)).rowcount

    # ── Working → Active demotion (opt-in) ────────────────────────────
    demoted_working = 0
    if demote_working:
        demoted_working = conn.execute("""
            UPDATE experiences
            SET tier = 'active'
            WHERE tier = 'working'
              AND access_count < ?
              AND timestamp < ?
        """, (min_access, cutoff)).rowcount

    # ── Active tier cleanup (opt-in) ──────────────────────────────────
    promoted = 0
    deleted_active = 0
    if cleanup_active:
        # Promote consolidated old Active → Working
        promoted = conn.execute("""
            UPDATE experiences
            SET tier = 'working'
            WHERE tier = 'active'
              AND consolidated = 1
              AND timestamp < ?
        """, (cutoff,)).rowcount

        # Delete unconsolidated old Active (never worth saving)
        deleted_active = conn.execute("""
            DELETE FROM experiences
            WHERE tier = 'active'
              AND consolidated = 0
              AND timestamp < ?
        """, (cutoff,)).rowcount

    conn.commit()
    conn.close()

    return {
        'demoted_longterm': demoted_longterm,
        'demoted_working': demoted_working,
        'promoted': promoted,
        'deleted': deleted_longterm + deleted_active,
    }


# ─── Query / Search ──────────────────────────────────────────────────

def get_tiered_stats() -> dict:
    """Return count and avg_error per tier."""
    conn = get_db()
    rows = conn.execute("""
        SELECT tier,
               COUNT(*) AS count,
               AVG(prediction_error) AS avg_error
        FROM experiences
        GROUP BY tier
        ORDER BY
            CASE tier
                WHEN 'active' THEN 1
                WHEN 'working' THEN 2
                WHEN 'longterm' THEN 3
            END
    """).fetchall()
    conn.close()

    stats = {}
    for r in rows:
        stats[r['tier']] = {
            'count': r['count'],
            'avg_error': round(r['avg_error'], 3) if r['avg_error'] else 0.0,
        }
    return stats


def get_priority_stats() -> dict:
    """Return min/mean/max prediction_error across all tiers.

    Useful for monitoring the distribution of surprise values
    and tuning PER sampling parameters.

    Returns:
        Dict with 'min_error', 'mean_error', 'max_error' keys.
    """
    conn = get_db()
    row = conn.execute("""
        SELECT MIN(prediction_error) AS min_error,
               AVG(prediction_error) AS mean_error,
               MAX(prediction_error) AS max_error
        FROM experiences
    """).fetchone()
    conn.close()

    return {
        'min_error': round(row['min_error'], 4) if row['min_error'] is not None else 0.0,
        'mean_error': round(row['mean_error'], 4) if row['mean_error'] is not None else 0.0,
        'max_error': round(row['max_error'], 4) if row['max_error'] is not None else 0.0,
    }


def search_memories(query: str, tier: str = None, limit: int = 10) -> list[dict]:
    """Text-based search across tiers using LIKE.

    Args:
        query: Search term (case-insensitive substring match).
        tier: Optional tier filter (None = search all tiers).
        limit: Max results.

    Returns:
        List of matching experience dicts.
    """
    conn = get_db()
    pattern = f'%{query}%'
    if tier:
        rows = conn.execute("""
            SELECT id, timestamp, input_text, output_text, prediction_error,
                   skill, is_correction, tier, access_count, last_access, consolidated
            FROM experiences
            WHERE (input_text LIKE ? OR output_text LIKE ?)
              AND tier = ?
            ORDER BY prediction_error DESC
            LIMIT ?
        """, (pattern, pattern, tier, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, timestamp, input_text, output_text, prediction_error,
                   skill, is_correction, tier, access_count, last_access, consolidated
            FROM experiences
            WHERE input_text LIKE ? OR output_text LIKE ?
            ORDER BY prediction_error DESC
            LIMIT ?
        """, (pattern, pattern, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent(limit: int = 20, tier: str = 'active') -> list[dict]:
    """Get most recent memories from a tier."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, timestamp, input_text, output_text, prediction_error,
               skill, is_correction, tier, access_count, last_access, consolidated
        FROM experiences
        WHERE tier = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (tier, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Legacy / Backward-compatible helpers ────────────────────────────

def get_stats() -> dict:
    """Get overall hippocampus statistics (backward-compatible)."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    unconsolidated = conn.execute(
        "SELECT COUNT(*) FROM experiences WHERE consolidated = 0"
    ).fetchone()[0]
    avg_error = conn.execute(
        "SELECT AVG(prediction_error) FROM experiences"
    ).fetchone()[0]
    conn.close()
    return {
        'total_experiences': total,
        'unconsolidated': unconsolidated,
        'avg_error': round(avg_error, 3) if avg_error else 0,
    }


def clear():
    """Clear the hippocampus (for testing)."""
    conn = get_db()
    conn.execute("DELETE FROM experiences")
    conn.commit()
    conn.close()


# ─── Surprise Detector ──────────────────────────────────────────────

@torch.no_grad()
def compute_surprise(model, tokenizer, text: str) -> float:
    """Compute prediction error (surprise) for a text."""
    import torch
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 3:
        return 0.0

    device = next(model.parameters()).device
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)

    model.eval()
    _, loss, _ = model(x, y)
    return loss.item()


def auto_store(model, tokenizer, text: str, threshold: float = 1.0,
               output: str = None, skill: str = 'general', tier: str = 'active'):
    """Automatically compute surprise and store if above threshold."""
    error = compute_surprise(model, tokenizer, text)
    if error >= threshold:
        store_experience(text, error, output, skill, tier=tier)
        return True
    return False


if __name__ == '__main__':
    # Test
    store_experience('What is a mitochondrion?', 2.5, 'The powerhouse of the cell', 'biology')
    store_experience('E=mc^2', 1.8, 'Energy equals mass times speed of light squared', 'physics')
    print(f'Stats: {get_stats()}')
    print(f'Unconsolidated (active): {len(get_unconsolidated())}')
    print(f'Unconsolidated (longterm): {len(get_unconsolidated(tier="longterm"))}')
    print(f'Old memories: {len(get_old_memories())}')
    print(f'Tiered stats: {get_tiered_stats()}')
