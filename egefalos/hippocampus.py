"""Hippocampus — Tiered 3-layer memory architecture with Gradient Replay.

Extends the base hippocampus with gradient storage: when storing experiences,
optionally include the parameter gradients so the sleep cycle can apply them
directly (gradient matching) rather than re-training from scratch.

New features:
- gradient_data column stores serialized gradient tensors
- store_experience now accepts optional gradient_bytes
- get_with_gradients() returns experiences with gradient blobs
- gradient_replay() in sleep cycle applies stored gradients
"""

import sqlite3
import json
import time
import math
import io
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

DB_PATH = Path('memory/hippocampus.db')

# Module-level connection pool — WAL mode for concurrent reads
_connection: Optional[sqlite3.Connection] = None
_schema_initialized: bool = False


def get_db() -> sqlite3.Connection:
    """Get the persistent hippocampus database connection (module-level singleton)."""
    global _connection, _schema_initialized
    if _connection is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(str(DB_PATH))
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA synchronous=NORMAL")
    if not _schema_initialized:
        _init_schema(_connection)
        _schema_initialized = True
    return _connection


def _init_schema(conn: sqlite3.Connection):
    """Create tables, add backward-compatible columns, and build indexes."""
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

    # Add missing columns for old databases
    for col, col_type in [
        ('tier', "TEXT DEFAULT 'active'"),
        ('access_count', 'INTEGER DEFAULT 0'),
        ('last_access', 'REAL DEFAULT 0'),
    ]:
        try:
            conn.execute(f"ALTER TABLE experiences ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    # Add gradient_data BLOB column (new for gradient replay)
    try:
        conn.execute("ALTER TABLE experiences ADD COLUMN gradient_data BLOB")
    except sqlite3.OperationalError:
        pass  # Already exists

    # Add latent_data BLOB column (new for latent replay)
    try:
        conn.execute("ALTER TABLE experiences ADD COLUMN latent_data BLOB")
    except sqlite3.OperationalError:
        pass  # Already exists

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
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_has_gradients
        ON experiences(tier, consolidated)
        WHERE gradient_data IS NOT NULL
    """)
    conn.commit()


# ─── Gradient Serialization ────────────────────────────────────────


def serialize_gradients(grad_dict: dict[str, torch.Tensor]) -> bytes:
    """Serialize a dict of gradient tensors to bytes.

    Each tensor is moved to CPU and saved into a single bytes buffer.
    """
    buffer = io.BytesIO()
    cpu_dict = {k: v.detach().cpu() for k, v in grad_dict.items() if v is not None}
    torch.save(cpu_dict, buffer)
    return buffer.getvalue()


def deserialize_gradients(data: bytes) -> dict[str, torch.Tensor]:
    """Deserialize gradient bytes back to tensor dict. Returns empty dict on fail."""
    if not data:
        return {}
    try:
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)
    except Exception:
        return {}


# ─── Latent Serialization ─────────────────────────────────────────


def serialize_latent(tensor: torch.Tensor) -> bytes:
    """Serialize a single latent tensor to bytes."""
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def deserialize_latent(data: bytes) -> Optional[torch.Tensor]:
    """Deserialize latent bytes back to tensor. Returns None on fail."""
    if not data:
        return None
    try:
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)
    except Exception:
        return None


@torch.no_grad()
def capture_latents(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run forward pass and capture the last hidden state (before LM head).

    Used for latent replay: stores the representation so it can be
    replayed later without re-running the full transformer.

    Args:
        model: MathTransformer.
        input_ids: (1, seq_len) tensor of token IDs.

    Returns:
        (d_model,) tensor — the last token's hidden state.
    """
    model.eval()
    # Forward pass through transformer layers only (not LM head)
    x = model.token_embedding(input_ids)
    for layer in model.layers:
        x, _ = layer(x, mask=None)
    # x is (1, seq_len, d_model); take last token
    latent = x[0, -1, :].detach().cpu()  # (d_model,)
    return latent


# ─── CRUD ──────────────────────────────────────────────────────────


def store_experience(
    input_text: str,
    prediction_error: float,
    output_text: str = None,
    skill: str = 'general',
    is_correction: bool = False,
    tier: str = 'active',
    gradient_data: Optional[dict[str, torch.Tensor]] = None,
    latent_data: Optional[torch.Tensor] = None,
):
    """Store an experience with optional gradient and latent information.

    Args:
        input_text: The input/prompt text.
        prediction_error: Surprise or loss value.
        output_text: Expected output or trace.
        skill: Domain/task identifier.
        is_correction: Whether this is a Socratic correction trace.
        tier: Memory tier ('active', 'working', 'longterm').
        gradient_data: Optional dict of parameter_name -> gradient tensor.
                       Serialized to BLOB for gradient replay.
        latent_data: Optional last hidden state tensor (d_model,).
                     Serialized to BLOB for latent replay.
    """
    conn = get_db()
    ts = time.time()
    grad_bytes = serialize_gradients(gradient_data) if gradient_data else None
    latent_bytes = serialize_latent(latent_data) if latent_data is not None else None

    conn.execute("""
        INSERT INTO experiences (timestamp, input_text, output_text,
                                 prediction_error, skill, is_correction,
                                 tier, gradient_data, latent_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, input_text, output_text,
          prediction_error, skill, 1 if is_correction else 0,
          tier, grad_bytes, latent_bytes))
    conn.commit()


def get_unconsolidated(
    limit: int = 100,
    tier: str = 'active',
    include_gradients: bool = False,
) -> list[dict]:
    """Get unconsolidated experiences, optionally with gradient data."""
    conn = get_db()
    cols = ("id, input_text, output_text, prediction_error,"
            "skill, is_correction, tier, access_count, last_access")
    if include_gradients:
        cols = "id, " + ", ".join(
            c for c in cols.split(", ") if c.strip() != "id"
        ) + ", gradient_data"

    rows = conn.execute(f"""
        SELECT {cols}
        FROM experiences WHERE consolidated = 0 AND tier = ?
        ORDER BY prediction_error DESC
        LIMIT ?
    """, (tier, limit)).fetchall()
    result = [dict(r) for r in rows]
    # Deserialize gradients for entries that have them
    if include_gradients:
        for r in result:
            if r.get('gradient_data'):
                r['gradients'] = deserialize_gradients(r['gradient_data'])
            del r['gradient_data']
    return result


def get_old_memories(
    limit: int = 50,
    tier: str = 'longterm',
    use_per: bool = True,
    only_with_gradients: bool = False,
    include_latents: bool = False,
) -> list[dict]:
    """Get consolidated memories, optionally with gradient and latent data."""
    conn = get_db()
    grad_filter = "AND gradient_data IS NOT NULL" if only_with_gradients else ""

    total = conn.execute(
        "SELECT COUNT(*) FROM experiences WHERE consolidated = 1 AND tier = ? "
        + grad_filter,
        (tier,)
    ).fetchone()[0]
    if total == 0:
        return []

    cols = "id, input_text, output_text, prediction_error, "\
           "skill, is_correction, tier, access_count, last_access, gradient_data"
    if include_latents:
        cols += ", latent_data"

    if not use_per:
        rows = conn.execute(f"""
            SELECT {cols}
            FROM experiences WHERE consolidated = 1 AND tier = ? {grad_filter}
            ORDER BY RANDOM() LIMIT ?
        """, (tier, limit)).fetchall()
        result = [dict(r) for r in rows]
        for r in result:
            r['gradients'] = deserialize_gradients(r.get('gradient_data', b''))
            del r['gradient_data']
            if include_latents:
                r['latent'] = deserialize_latent(r.get('latent_data'))
                del r['latent_data']
        return result

    # PER sampling
    rows = conn.execute(f"""
        SELECT {cols}
        FROM experiences WHERE consolidated = 1 AND tier = ? {grad_filter}
    """, (tier,)).fetchall()
    if not rows:
        return []

    records = [dict(r) for r in rows]
    for r in records:
        r['gradients'] = deserialize_gradients(r.get('gradient_data', b''))
        del r['gradient_data']
        if include_latents:
            r['latent'] = deserialize_latent(r.get('latent_data'))
            del r['latent_data']

    actual_limit = min(limit, len(records))
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


# ─── Tier Promotion / Demotion ───────────────────────────────────────


def promote_to_working(ids: list[int]):
    """Move experiences from active -> working tier."""
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


def promote_to_longterm(ids: list[int]):
    """Move experiences from working -> longterm tier."""
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


def decay_memories(min_access: int = 3, max_age_days: int = 30,
                   decay_rate: float = 0.5,
                   demote_working: bool = False,
                   cleanup_active: bool = False):
    """Full synaptic decay across all 3 memory tiers.

    Gradient-enhanced entries (gradient_data IS NOT NULL) are protected
    from deletion in the longterm tier — their gradients may still be
    useful for replay even if access_count is low.
    """
    conn = get_db()
    cutoff = time.time() - (max_age_days * 86400)

    # Synaptic decay
    decay_amount = int(decay_rate * max_age_days)
    if decay_amount > 0:
        conn.execute("""
            UPDATE experiences
            SET access_count = MAX(0, access_count - ?)
            WHERE access_count > 0
        """, (decay_amount,))
        conn.commit()

    # Demote longterm -> working (skip gradient-enhanced entries)
    demoted_longterm = conn.execute("""
        UPDATE experiences
        SET tier = 'working'
        WHERE tier = 'longterm'
          AND access_count < ?
          AND timestamp < ?
          AND gradient_data IS NULL
    """, (min_access, cutoff)).rowcount

    # Delete very old unconsolidated longterm (skip gradient-enhanced)
    deleted_longterm = conn.execute("""
        DELETE FROM experiences
        WHERE tier = 'longterm'
          AND consolidated = 0
          AND timestamp < ?
          AND gradient_data IS NULL
    """, (cutoff,)).rowcount

    # Working -> Active demotion
    demoted_working = 0
    if demote_working:
        demoted_working = conn.execute("""
            UPDATE experiences
            SET tier = 'active'
            WHERE tier = 'working'
              AND access_count < ?
              AND timestamp < ?
        """, (min_access, cutoff)).rowcount

    # Active tier cleanup
    promoted = 0
    deleted_active = 0
    if cleanup_active:
        promoted = conn.execute("""
            UPDATE experiences
            SET tier = 'working'
            WHERE tier = 'active'
              AND consolidated = 1
              AND timestamp < ?
        """, (cutoff,)).rowcount
        deleted_active = conn.execute("""
            DELETE FROM experiences
            WHERE tier = 'active'
              AND consolidated = 0
              AND timestamp < ?
        """, (cutoff,)).rowcount

    conn.commit()

    return {
        'demoted_longterm': demoted_longterm,
        'demoted_working': demoted_working,
        'promoted': promoted,
        'deleted': deleted_longterm + deleted_active,
    }


# ─── Gradient Replay ────────────────────────────────────────────────


@torch.no_grad()
def gradient_replay(
    model: torch.nn.Module,
    lr: float = 0.001,
    limit: int = 50,
    tier: str = 'longterm',
) -> int:
    """Apply stored gradients directly to model parameters (gradient matching).

    Loads gradient-enhanced experiences from the specified tier and applies
    their stored gradients as a direct parameter update, weighted by the
    prediction_error (higher error = stronger update).

    This is more efficient than full re-training because:
    1. No forward pass needed — stored gradients are applied directly
    2. Multiple experiences can be applied in a single update
    3. No batch processing overhead

    Args:
        model: The model to update.
        lr: Learning rate scalar for gradient application.
        limit: Max experiences to replay.
        tier: Memory tier to sample from.

    Returns:
        Number of experiences replayed.
    """
    memories = get_old_memories(
        limit=limit, tier=tier, use_per=True, only_with_gradients=True
    )
    if not memories:
        return 0

    device = next(model.parameters()).device
    param_dict = dict(model.named_parameters())
    replay_count = 0

    for mem in memories:
        grads = mem.get('gradients', {})
        if not grads:
            continue
        error_weight = max(mem['prediction_error'], 0.1)

        for name, grad_tensor in grads.items():
            if name in param_dict and param_dict[name].requires_grad:
                param_dict[name].add_(
                    grad_tensor.to(device) * lr * error_weight
                )
        replay_count += 1

    return replay_count


def replay_latents(
    model: torch.nn.Module,
    tokenizer,
    lr: float = 0.001,
    limit: int = 50,
    tier: str = 'longterm',
    distill_coef: float = 0.5,
) -> int:
    """Replay stored latent representations as distillation targets.

    For each experience with latent data:
    1. Run the input through the full model to get current hidden state
    2. Compute MSE loss between current hidden state and stored latent
    3. Backprop to pull current representations toward stored ones

    This is more efficient than full re-training because:
    - Only needs the forward pass + one loss.backward()
    - Representation-level replay preserves task structure
    - Complements gradient replay (which operates on parameter grads)

    Args:
        model: The model to update.
        tokenizer: MathTokenizer for encoding.
        lr: Learning rate for latent distillation.
        limit: Max experiences to replay.
        tier: Memory tier to sample from.
        distill_coef: Weight of distillation loss vs standard CE.

    Returns:
        Number of experiences replayed.
    """
    experiences = get_old_memories(
        limit=limit, tier=tier, use_per=True, include_latents=True
    )
    if not experiences:
        return 0

    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    replay_count = 0

    for exp in experiences:
        text = exp.get('input_text', '')
        if not text:
            continue

        # Tokenize
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)

        # Forward pass through full model (with gradients)
        logits, loss, _ = model(x, x)  # CE loss on next-token prediction

        # Get current hidden state (with gradients for backprop)
        h = model.token_embedding(x)
        for layer in model.layers:
            h, _ = layer(h, mask=None)
        current_hidden = h[0, -1, :]  # (d_model,)

        # Latent distillation loss
        target = exp.get('latent')
        if target is not None:
            target = target.to(device)
            distill_loss = nn.MSELoss()(current_hidden, target)
            total_loss = loss + distill_coef * distill_loss
        else:
            total_loss = loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        replay_count += 1

        if replay_count % 10 == 0:
            print(f"    [Latent replay] {replay_count}/{limit}: loss={total_loss.item():.4f}")

    return replay_count


# ─── Query / Stats ──────────────────────────────────────────────────


def get_tiered_stats() -> dict:
    """Return count and avg_error per tier."""
    conn = get_db()
    rows = conn.execute("""
        SELECT tier,
               COUNT(*) AS count,
               AVG(prediction_error) AS avg_error,
               SUM(CASE WHEN gradient_data IS NOT NULL THEN 1 ELSE 0 END) AS with_gradients
        FROM experiences
        GROUP BY tier
        ORDER BY
            CASE tier
                WHEN 'active' THEN 1
                WHEN 'working' THEN 2
                WHEN 'longterm' THEN 3
            END
    """).fetchall()
    stats = {}
    for r in rows:
        stats[r['tier']] = {
            'count': r['count'],
            'avg_error': round(r['avg_error'], 3) if r['avg_error'] else 0.0,
            'with_gradients': r['with_gradients'] or 0,
        }
    return stats


def get_priority_stats() -> dict:
    """Return min/mean/max prediction_error across all tiers."""
    conn = get_db()
    row = conn.execute("""
        SELECT MIN(prediction_error) AS min_error,
               AVG(prediction_error) AS mean_error,
               MAX(prediction_error) AS max_error
        FROM experiences
    """).fetchone()
    return {
        'min_error': round(row['min_error'], 4) if row['min_error'] is not None else 0.0,
        'mean_error': round(row['mean_error'], 4) if row['mean_error'] is not None else 0.0,
        'max_error': round(row['max_error'], 4) if row['max_error'] is not None else 0.0,
    }


def search_memories(query: str, tier: str = None, limit: int = 10) -> list[dict]:
    """Text-based search across tiers using LIKE."""
    conn = get_db()
    pattern = f'%{query}%'
    if tier:
        rows = conn.execute("""
            SELECT id, timestamp, input_text, output_text, prediction_error,
                   skill, is_correction, tier, access_count, last_access,
                   consolidated,
                   CASE WHEN gradient_data IS NOT NULL THEN 1 ELSE 0 END AS has_gradients
            FROM experiences
            WHERE (input_text LIKE ? OR output_text LIKE ?)
              AND tier = ?
            ORDER BY prediction_error DESC
            LIMIT ?
        """, (pattern, pattern, tier, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, timestamp, input_text, output_text, prediction_error,
                   skill, is_correction, tier, access_count, last_access,
                   consolidated,
                   CASE WHEN gradient_data IS NOT NULL THEN 1 ELSE 0 END AS has_gradients
            FROM experiences
            WHERE input_text LIKE ? OR output_text LIKE ?
            ORDER BY prediction_error DESC
            LIMIT ?
        """, (pattern, pattern, limit)).fetchall()
    return [dict(r) for r in rows]


def get_recent(limit: int = 20, tier: str = 'active') -> list[dict]:
    """Get most recent memories from a tier."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, timestamp, input_text, output_text, prediction_error,
               skill, is_correction, tier, access_count, last_access,
               consolidated,
               CASE WHEN gradient_data IS NOT NULL THEN 1 ELSE 0 END AS has_gradients
        FROM experiences
        WHERE tier = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (tier, limit)).fetchall()
    return [dict(r) for r in rows]


# ─── Legacy helpers ────────────────────────────────────────────────


def get_stats() -> dict:
    """Get overall hippocampus statistics (backward-compatible)."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    unconsolidated = conn.execute(
        "SELECT COUNT(*) FROM experiences WHERE consolidated = 0"
    ).fetchone()[0]
    with_grads = conn.execute(
        "SELECT COUNT(*) FROM experiences WHERE gradient_data IS NOT NULL"
    ).fetchone()[0]
    avg_error = conn.execute(
        "SELECT AVG(prediction_error) FROM experiences"
    ).fetchone()[0]
    return {
        'total_experiences': total,
        'unconsolidated': unconsolidated,
        'with_gradients': with_grads,
        'avg_error': round(avg_error, 3) if avg_error else 0,
    }


def clear():
    """Clear the hippocampus (for testing)."""
    conn = get_db()
    conn.execute("DELETE FROM experiences")
    conn.commit()


# ─── Surprise Detector ──────────────────────────────────────────────

@torch.no_grad()
def compute_surprise(model, tokenizer, text: str) -> float:
    """Compute prediction error (surprise) for a text."""
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
    store_experience('What is a mitochondrion?', 2.5,
                     'The powerhouse of the cell', 'biology')
    store_experience('E=mc^2', 1.8,
                     'Energy equals mass times speed of light squared', 'physics')
    print(f'Stats: {get_stats()}')
