"""Lightweight PII (Personally Identifiable Information) Detector & Scrubber.

Pure-regex-based — no external dependencies.  Designed for fast inline
scrubbing of user prompts before they reach the training pipeline.
"""

import re

# ── Regex Patterns ──────────────────────────────────────────────────

# Email addresses
EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Phone numbers (various formats)
PHONE_RE = re.compile(
    r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
)

# US Social Security Numbers (NNN-NN-NNNN)
SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

# Credit card numbers (simplified — 16 digits in groups of 4)
CREDIT_CARD_RE = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')

# IPv4 addresses
IP_RE = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

# ── Common First Names (heuristic for personal info detection) ─────

COMMON_NAMES: set[str] = {
    'john', 'jane', 'mike', 'mary', 'bob', 'alice', 'david', 'sarah',
    'james', 'linda', 'robert', 'patricia', 'michael', 'jennifer',
    'william', 'elizabeth', 'richard', 'susan', 'joseph', 'jessica',
    'thomas', 'karen', 'charles', 'lisa', 'chris', 'amanda',
    'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven',
    'paul', 'andrew', 'joshua', 'kenneth', 'kevin', 'brian',
    'george', 'timothy', 'ronald', 'edward', 'jason', 'jeffrey',
    'ryan', 'jacob', 'gary', 'nicholas', 'eric', 'jonathan',
    'stephen', 'larry', 'justin', 'scott', 'brandon', 'benjamin',
    'samuel', 'raymond', 'gregory', 'frank', 'alexander', 'patrick',
    'jack', 'dennis', 'jerry', 'tyler', 'aaron', 'jose', 'nathan',
    'henry', 'douglas', 'peter', 'adam', 'zachary', 'nathaniel',
    'barbara', 'sandra', 'nancy', 'betty', 'dorothy', 'helen',
    'sandra', 'donna', 'ruth', 'carol', 'michelle', 'emily',
    'deborah', 'rebecca', 'laura', 'cynthia', 'kathleen', 'amy',
    'angela', 'shirley', 'anna', 'brenda', 'pamela', 'emma',
    'samantha', 'katherine', 'virginia', 'rachel', 'andrea', 'carolyn',
    'debra', 'tiffany', 'megan', 'cheryl', 'heather', 'janet',
}


# ── Public API ─────────────────────────────────────────────────────


def scrub_pii(text: str, replacement: str = '[REDACTED]') -> str:
    """Remove PII from *text*, replacing each match with *replacement*.

    Scrubs the following patterns:
        - Email addresses
        - Phone numbers (various formats)
        - US Social Security Numbers (NNN-NN-NNNN)
        - Credit card numbers (16-digit grouped forms)
        - IPv4 addresses

    Args:
        text: The input string to scrub.
        replacement: The string to insert in place of each match
            (default: ``[REDACTED]``).

    Returns:
        The scrubbed string.
    """
    text = EMAIL_RE.sub(replacement, text)
    text = PHONE_RE.sub(replacement, text)
    text = SSN_RE.sub(replacement, text)
    text = CREDIT_CARD_RE.sub(replacement, text)
    text = IP_RE.sub(replacement, text)
    return text


def contains_pii(text: str) -> bool:
    """Return ``True`` if *text* contains any PII patterns.

    Checks the same patterns as :func:`scrub_pii` (email, phone, SSN,
    credit-card, IP), but does **not** check for common names.

    Args:
        text: The input string to check.

    Returns:
        ``True`` if at least one PII pattern is found.
    """
    if EMAIL_RE.search(text):
        return True
    if PHONE_RE.search(text):
        return True
    if SSN_RE.search(text):
        return True
    if CREDIT_CARD_RE.search(text):
        return True
    if IP_RE.search(text):
        return True
    return False


def has_personal_info(text: str) -> bool:
    """Heuristic check for personal / identifying information.

    Looks for common first names in *text* (case-insensitive).  This is
    an intentionally broad heuristic — it will produce false positives
    on text that happens to contain a common name word (e.g. "Will" in
    "free will").  Use as a soft signal only.

    Args:
        text: The input string to check.

    Returns:
        ``True`` if any common name token is found in the text.
    """
    words: set[str] = set(text.lower().split())
    return bool(words & COMMON_NAMES)
