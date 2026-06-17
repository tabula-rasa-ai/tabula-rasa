"""Sleep cycle: consolidate experiences into long-term memory via Online EWC.

During sleep:
1. Sample high-surprise experiences from hippocampus
2. Compute Fisher information on replay data
3. Merge into running EWC total
4. Train with EWC penalty to consolidate without forgetting
5. Cluster similar experiences and generate summaries
"""
import sys
from pathlib import Path
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import re
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Optional
import argparse
import time
import numpy as np

from tabula_rasa.model import MathTransformer
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from egefalos.online_ewc import OnlineEWC
from egefalos.hippocampus import (
    get_unconsolidated, get_stats, mark_consolidated, decay_memories,
    get_old_memories, store_experience, get_db,
)


# ─── Memory Clustering / Summarization ─────────────────────────────────


def _tokenize_text(text: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer for TF-IDF bag-of-words."""
    return re.findall(r'[a-zA-Z0-9]+|[^\w\s]', text.lower())


def _build_term_freq(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build a term-frequency matrix (tf-idf-like) from a list of texts.

    Returns:
        matrix: (n_docs, n_terms) float32 array of TF-IDF values.
        vocab:  list of unique term strings.
    """
    tokenized = [_tokenize_text(t) for t in texts]
    # Build vocabulary
    vocab_set: set[str] = set()
    for tokens in tokenized:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    if not vocab:
        return np.zeros((len(texts), 0), dtype=np.float32), vocab

    # Document frequency
    df = {term: 0 for term in vocab}
    for tokens in tokenized:
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            df[token] += 1

    n_docs = len(texts)
    idf = {term: math.log(1.0 + n_docs / (1.0 + dfreq))
           for term, dfreq in df.items()}

    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        tf_raw = len(tokens)
        if tf_raw == 0:
            continue
        term_counts: dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
        for j, term in enumerate(vocab):
            tf = term_counts.get(term, 0) / tf_raw
            matrix[i, j] = tf * idf[term]

    return matrix, vocab


def _tfidf_similarity(texts: list[str]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for a list of texts.

    Uses a lightweight TF-IDF-ish representation (no scikit-learn dependency).

    Returns:
        (n_docs, n_docs) float32 similarity matrix (diagonal = 1.0).
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n == 1:
        return np.ones((1, 1), dtype=np.float32)

    matrix, _ = _build_term_freq(texts)  # (n_docs, n_terms)
    if matrix.shape[1] == 0:
        # No terms — return identity
        return np.eye(n, dtype=np.float32)

    # L2-normalise rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
    matrix_normed = matrix / norms  # (n_docs, n_terms)

    # Cosine similarity = dot product of L2-normalised rows
    return np.dot(matrix_normed, matrix_normed.T)  # (n_docs, n_docs)


def _summarize_cluster(experiences: list[dict]) -> str:
    """Generate a summary string for a cluster of similar experiences.

    Strategy: pick the most common output_text among the cluster as the
    representative summary. If there's a tie, concatenate the two most
    common outputs separated by ' | '.

    Args:
        experiences: List of experience dicts, each with 'input_text' and
                     'output_text' keys.

    Returns:
        A single summary string.
    """
    if not experiences:
        return ""

    # Count output_text frequencies
    output_counts: dict[str, int] = {}
    for exp in experiences:
        ot = (exp.get('output_text') or '').strip()
        if ot:
            output_counts[ot] = output_counts.get(ot, 0) + 1

    if not output_counts:
        # Fall back to most common input prefix
        inputs = [e.get('input_text', '') for e in experiences]
        return max(set(inputs), key=inputs.count) if inputs else ""

    # Sort by frequency
    sorted_outputs = sorted(output_counts.items(), key=lambda x: -x[1])
    if len(sorted_outputs) == 0:
        return ""
    if len(sorted_outputs) == 1 or sorted_outputs[0][1] > sorted_outputs[1][1]:
        return sorted_outputs[0][0]
    # Tie — join the top two
    return f"{sorted_outputs[0][0]} | {sorted_outputs[1][0]}"


def cluster_and_summarize(
    similarity_threshold: float = 0.5,
    max_age_days: int = 30,
    max_cluster_size: int = 20,
) -> dict:
    """Cluster similar experiences in the hippocampus and generate summaries.

    Workflow:
    1. Fetch all active + longterm experiences from hippocampus.
    2. Compute pairwise TF-IDF cosine similarity.
    3. Group experiences with similarity > threshold into clusters.
    4. For each cluster of 5+ members, generate a summary by taking the
       most common output_text.
    5. Mark summarised experiences as 'consolidated'.
    6. Store the summary as a new 'core' tier entry.
    7. Prune: delete experiences older than *max_age_days* in 'active'
       tier that have been consolidated.

    Args:
        similarity_threshold: Cosine similarity threshold for clustering
                              (default: 0.5).
        max_age_days: Delete consolidated active experiences older than
                      this many days (default: 30).
        max_cluster_size: Maximum experiences to consider per cluster
                          (default: 20).

    Returns:
        dict with keys:
            - clusters_found: int
            - experiences_summarized: int
            - summaries_stored: int
            - pruned_deleted: int
    """
    conn = get_db()
    result = {
        'clusters_found': 0,
        'experiences_summarized': 0,
        'summaries_stored': 0,
        'pruned_deleted': 0,
    }

    # Fetch all active + longterm (not yet consolidated > active filter)
    rows = conn.execute("""
        SELECT id, input_text, output_text, tier, timestamp
        FROM experiences
        WHERE tier IN ('active', 'longterm')
          AND consolidated = 0
        ORDER BY timestamp DESC
    """).fetchall()

    if not rows:
        print("  [Cluster] No unconsolidated experiences to cluster.")
        return result

    records = [dict(r) for r in rows]
    texts = [r['input_text'] for r in records]
    n = len(records)

    # Compute similarity matrix
    print(f"  [Cluster] Computing similarity for {n} experiences...")
    sim = _tfidf_similarity(texts)

    # Greedy clustering: iterate, find largest cluster from each seed
    assigned = set()
    clusters: list[list[int]] = []

    for i in range(n):
        if i in assigned:
            continue
        # Find all unassigned indices with similarity > threshold
        members = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if sim[i, j] > similarity_threshold:
                members.append(j)
                assigned.add(j)
        if len(members) >= 5:
            clusters.append(members)

    result['clusters_found'] = len(clusters)
    if not clusters:
        print("  [Cluster] No clusters of 5+ similar experiences found.")
        return result

    print(f"  [Cluster] Found {len(clusters)} clusters (>=5 similar experiences each).")

    cutoff_ts = time.time() - (max_age_days * 86400)
    summarized_ids: list[int] = []
    stored_summaries = 0

    for idx, cluster_indices in enumerate(clusters):
        cluster_exp = [records[i] for i in cluster_indices]
        # Limit cluster size
        if len(cluster_exp) > max_cluster_size:
            cluster_exp = cluster_exp[:max_cluster_size]

        # Generate summary
        summary = _summarize_cluster(cluster_exp)
        if not summary:
            continue

        # Collect IDs to mark consolidated
        cluster_ids = [records[i]['id'] for i in cluster_indices]
        summarized_ids.extend(cluster_ids)

        # Find the most common input_text pattern as representative
        input_texts = [e['input_text'] for e in cluster_exp]
        rep_input = max(set(input_texts), key=input_texts.count) if input_texts else ""

        # Store as 'core' tier entry
        conn.execute("""
            INSERT INTO experiences
                (timestamp, input_text, output_text, prediction_error,
                 skill, is_correction, tier, consolidated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            f"[cluster] {rep_input[:80]}" if rep_input else f"[cluster {idx}]",
            summary,
            0.0,
            'core',
            0,
            'longterm',
            1,
        ))
        stored_summaries += 1
        result['experiences_summarized'] += len(cluster_ids)

    conn.commit()
    result['summaries_stored'] = stored_summaries

    # Mark summarised experiences as consolidated
    if summarized_ids:
        mark_consolidated(list(set(summarized_ids)))

    # Prune: delete consolidated active experiences older than max_age_days
    deleted = conn.execute("""
        DELETE FROM experiences
        WHERE tier = 'active'
          AND consolidated = 1
          AND timestamp < ?
    """, (cutoff_ts,)).rowcount
    conn.commit()
    result['pruned_deleted'] = deleted

    print(f"  [Cluster] Summary: {result['experiences_summarized']} experiences → "
          f"{stored_summaries} core summaries, {deleted} pruned.")
    return result


def consolidate_sleep_cycle(
    model_path: Path,
    tokenizer: MathTokenizer,
    config: Config,
    num_samples: int = 2048,
    epochs: int = 3,
    lambda_ewc: float = 1000.0,
    gamma: float = 0.9,
    output_dir: Optional[Path] = None,
    max_seq_len: int = 32,
):
    """Run one sleep cycle: replay + EWC consolidation.

    Args:
        model_path: Path to the specialist's best.pt checkpoint
        tokenizer: MathTokenizer instance
        config: Model configuration
        num_samples: Max replay experiences to consume
        epochs: Training epochs over replay buffer
        lambda_ewc: EWC penalty strength
        gamma: Fisher merge decay rate
        output_dir: Specialist directory (default: parent of model_path)
        max_seq_len: Max token sequence length
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = get_stats()

    print(f"\n{'='*60}")
    print(f"  SLEEP CYCLE — Consolidation")
    print(f"  Device: {device}")
    print(f"  Hippocampus: {stats['total_experiences']} total, {stats['unconsolidated']} unconsolidated")
    print(f"  Avg prediction error: {stats['avg_error']}")
    print(f"{'='*60}")

    # Load model
    if not model_path.exists():
        print(f"  [Sleep] Model not found: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    sd = checkpoint.get('model_state_dict', checkpoint)

    # Auto-detect architecture
    d_model = sd['token_embedding.weight'].shape[1]
    n_layers = len([k for k in sd if k.startswith('layers.') and k.endswith('.attention.wq.weight')])
    d_ff = sd['layers.0.feed_forward.w1.weight'].shape[0] if n_layers > 0 else 256
    n_heads = {128: 4, 64: 2, 96: 4, 256: 8}.get(d_model, max(1, d_model // 32))

    cfg = Config()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.d_ff = d_ff
    cfg.n_heads = n_heads
    cfg.max_seq_len = max_seq_len

    model = MathTransformer(cfg)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.train()

    if output_dir is None:
        output_dir = model_path.parent

    # Initialize EWC
    ewc = OnlineEWC(model, gamma=gamma)
    ewc_path = output_dir / 'ewc_fisher.pt'
    if ewc_path.exists():
        ewc.load(ewc_path)
        print(f"  [Sleep] Loaded EWC from {ewc_path} ({ewc.task_count} prior tasks)")

    # Get unconsolidated experiences (sorted by surprise DESC)
    experiences = get_unconsolidated(limit=num_samples)
    if not experiences:
        print("  [Sleep] No unconsolidated experiences. Nothing to do.")
        return

    print(f"  [Sleep] Loaded {len(experiences)} high-surprise experiences")

    # Also sample consolidated long-term memories with PER weighting
    try:
        legacy_replay = get_old_memories(limit=num_samples // 2, tier='longterm', use_per=True)
        if legacy_replay:
            print(f"  [Sleep] PER replay: {len(legacy_replay)} long-term memories (error-weighted)")
            experiences.extend(legacy_replay)
    except Exception as e:
        print(f"  [Sleep] PER replay skipped: {e}")

    # Tokenize experiences into training batches
    input_ids_list = []
    targets_list = []

    for exp in experiences:
        text = exp.get('input', '') or ''
        try:
            ids = tokenizer.encode(text, add_special_tokens=True)
        except Exception:
            continue

        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        # Create targets: -100 for prompt tokens (ignored in loss)
        # The answer is everything after the last separator or '='
        # Simple heuristic: last third of tokens is the answer
        split_point = max(1, len(ids) * 2 // 3)
        targets = [-100] * split_point + ids[split_point:]

        # Pad to max_seq_len
        pad_len = max_seq_len - len(ids)
        padded_ids = ids + [tokenizer.pad_id] * pad_len
        padded_targets = targets + [-100] * pad_len

        input_ids_list.append(padded_ids[:max_seq_len])
        targets_list.append(padded_targets[:max_seq_len])

    if not input_ids_list:
        print("  [Sleep] No valid experiences after tokenization.")
        return

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)

    dataset = TensorDataset(input_ids_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=min(32, len(input_ids_list)), shuffle=True)

    # Phase 1: Compute Fisher on replay data
    print("  [Sleep] Phase 1: Computing Fisher Information Matrix...")
    new_fisher = ewc.compute_fisher(dataloader, num_samples=len(dataloader))
    print(f"  [Sleep] Fisher computed across {len(new_fisher)} parameter groups")

    # Phase 2: Merge Fisher
    print("  [Sleep] Phase 2: Merging into running Fisher total...")
    ewc.merge_fisher(new_fisher)
    ewc.save_anchor_weights()

    # Phase 3: Train with EWC penalty
    print(f"  [Sleep] Phase 3: Training with EWC penalty (λ={lambda_ewc})...")
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_penalty = 0.0
        num_batches = 0

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward
            _, task_loss, _ = model(input_ids, targets=targets)

            # EWC penalty
            ewc_penalty = ewc.compute_ewc_penalty(lambda_ewc=lambda_ewc)

            # Total loss
            total_loss = task_loss + ewc_penalty

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += task_loss.item()
            epoch_penalty += ewc_penalty.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_penalty = epoch_penalty / max(num_batches, 1)
        print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  ewc_penalty={avg_penalty:.4f}")

    # Phase 4: Save
    print("  [Sleep] Phase 4: Saving consolidated model and EWC state...")

    # Save consolidated checkpoint as best.pt
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': cfg.d_model, 'n_layers': cfg.n_layers,
            'd_ff': cfg.d_ff, 'n_heads': cfg.n_heads,
            'max_seq_len': cfg.max_seq_len,
        },
        'consolidation_step': ewc.consolidation_steps,
        'ewc_task_count': ewc.task_count,
    }, model_path)

    # Save EWC state
    ewc.consolidation_steps += 1
    ewc.save(ewc_path)

    # Mark experiences as consolidated
    exp_ids = [e['id'] for e in experiences if 'id' in e]
    if exp_ids:
        mark_consolidated(exp_ids)

    # Synaptic decay — demote/forget unused memories, clean up tiers
    decay_result = decay_memories(min_access=3, max_age_days=7,
                                  demote_working=True, cleanup_active=True)
    print(f'  Synaptic decay: {decay_result}')

    # Phase 5: Memory clustering and summarization
    print("  [Sleep] Phase 5: Clustering and summarizing similar experiences...")
    cluster_result = cluster_and_summarize(
        similarity_threshold=0.5,
        max_age_days=30,
        max_cluster_size=20,
    )
    print(f'  [Sleep] Clustering result: {cluster_result}')

    print(f"  [Sleep] Consolidation complete.")
    print(f"  [Sleep] Saved to {output_dir}")
    print(f"  [Sleep] Consolidated {len(exp_ids)} experiences")

    # Gradient Replay — apply stored gradients directly (no forward pass needed)
    try:
        from egefalos.hippocampus import gradient_replay
        n_replayed = gradient_replay(model, lr=0.001 * lambda_ewc_scale,
                                      limit=num_samples, tier='longterm')
        if n_replayed > 0:
            print(f"  [Sleep] Gradient replay: {n_replayed} gradient-enhanced experiences applied")
    except Exception as e:
        print(f"  [Sleep] Gradient replay skipped: {e}")

    # Latent Replay — representation-level distillation
    try:
        from egefalos.hippocampus import replay_latents
        n_latent = replay_latents(model, tokenizer, lr=0.0005,
                                   limit=num_samples, tier='longterm',
                                   distill_coef=0.3)
        if n_latent > 0:
            print(f"  [Sleep] Latent replay: {n_latent} representations distilled")
    except Exception as e:
        print(f"  [Sleep] Latent replay: {n_latent} representations distilled")
    print(f"{'='*60}\\n")


def _find_latest_checkpoint() -> Optional[Path]:
    """Find the most recently trained specialist checkpoint."""
    specialist_dirs = sorted(Path('specialists/math').glob('*/best.pt'))
    if specialist_dirs:
        return max(specialist_dirs, key=lambda p: p.stat().st_mtime)
    return None


def _run_daemon_cycle(args):
    """Run one cycle of sleep consolidation, finding the latest checkpoint."""
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = _find_latest_checkpoint()

    if model_path is None or not model_path.exists():
        print(f"  [Sleep] No checkpoint found — no specialists trained yet. Skipping cycle.")
        return

    config = Config()
    tok = MathTokenizer()
    tok.max_seq_len = args.max_seq_len
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent

    print(f"\n{'='*60}")
    print(f"  SLEEP CYCLE — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Checkpoint: {model_path}")
    print(f"{'='*60}")

    consolidate_sleep_cycle(
        model_path=model_path,
        tokenizer=tok,
        config=config,
        num_samples=args.num_samples,
        epochs=args.epochs,
        lambda_ewc=args.lambda_ewc,
        gamma=args.gamma,
        output_dir=output_dir,
        max_seq_len=args.max_seq_len,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sleep cycle consolidation')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to specialist checkpoint (best.pt). Auto-detects latest if omitted.')
    parser.add_argument('--num-samples', type=int, default=2048,
                        help='Max replay samples to consume')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs over replay buffer')
    parser.add_argument('--lambda-ewc', type=float, default=1000.0,
                        help='EWC penalty strength')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Fisher merge decay rate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Specialist output directory')
    parser.add_argument('--max-seq-len', type=int, default=32,
                        help='Max token sequence length')
    parser.add_argument('--daemon', action='store_true',
                        help='Run continuously in daemon mode, cycling every --interval seconds')
    parser.add_argument('--interval', type=int, default=3600,
                        help='Seconds between sleep cycles in daemon mode (default: 3600 = 1 hour)')
    args = parser.parse_args()

    if args.daemon:
        print(f"Sleep cycle daemon starting — interval: {args.interval}s")
        print("Auto-detecting latest specialist checkpoint each cycle.\n")
        cycle = 0
        while True:
            cycle += 1
            print(f"\n--- Cycle #{cycle} ---")
            _run_daemon_cycle(args)
            print(f"\nNext cycle in {args.interval}s...")
            time.sleep(args.interval)
    else:
        _run_daemon_cycle(args)
