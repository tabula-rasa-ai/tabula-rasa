"""
Grammar Engine — The "Rules of the Game" for Language AlphaZero.

In Chess, the rules prevent illegal moves (Bishop moving straight).
In Language, what prevents illegal sentences?

This module provides:
  1. Context-Free Grammar (CFG) skeleton — lightweight parse tree validation
  2. Morphological agreement rules — plural nouns → plural adjectives
  3. Z3 integration — logical consistency of generated statements
  4. Long-distance dependency scoring — Value Head intuition target

The Grammar Engine scores partial and complete sentences on a -1 to +1
scale, acting as the "environment" that rewards grammatical correctness.
"""

import re

# ═════════════════════════════════════════════════════════════════════
# CFG SKELETON — Context-Free Grammar for sentence structure
# ═════════════════════════════════════════════════════════════════════
# A minimal CFG that captures English sentence structure.
# This is the "board" — it defines legal token sequences.

# Grammar rules (EBNF-like):
#   S       → NP VP
#   NP      → Det (Adj)* N (PP)?
#   VP      → V (NP)? (PP)?
#   PP      → P NP
#   Adj     → Adj Adj?  (multiple adjectives)
#   Det     → 'the' | 'a' | 'an' | ''
#
# POS tags (simple mapping):
POS_TAGS = {
    # Determiners
    'the': 'DET', 'a': 'DET', 'an': 'DET', 'this': 'DET', 'that': 'DET',
    'these': 'DET', 'those': 'DET', 'some': 'DET', 'any': 'DET', 'every': 'DET',
    'all': 'DET', 'no': 'DET', 'each': 'DET', 'my': 'DET', 'his': 'DET',
    'her': 'DET', 'its': 'DET', 'our': 'DET', 'their': 'DET',

    # Prepositions
    'on': 'PREP', 'in': 'PREP', 'under': 'PREP', 'above': 'PREP', 'below': 'PREP',
    'beside': 'PREP', 'between': 'PREP', 'behind': 'PREP', 'in front of': 'PREP',
    'near': 'PREP', 'through': 'PREP', 'about': 'PREP', 'with': 'PREP',
    'without': 'PREP', 'for': 'PREP', 'against': 'PREP', 'toward': 'PREP',
    'inside': 'PREP', 'outside': 'PREP', 'across': 'PREP', 'along': 'PREP',
    'around': 'PREP', 'at': 'PREP', 'by': 'PREP', 'from': 'PREP', 'into': 'PREP',
    'of': 'PREP', 'to': 'PREP', 'during': 'PREP', 'before': 'PREP', 'after': 'PREP',

    # Conjunctions
    'and': 'CONJ', 'or': 'CONJ', 'but': 'CONJ', 'so': 'CONJ', 'because': 'CONJ',
    'although': 'CONJ', 'while': 'CONJ', 'if': 'CONJ', 'when': 'CONJ', 'unless': 'CONJ',
    'since': 'CONJ', 'therefore': 'CONJ', 'however': 'CONJ', 'moreover': 'CONJ',

    # Auxiliary verbs
    'is': 'AUX', 'are': 'AUX', 'was': 'AUX', 'were': 'AUX', 'be': 'AUX',
    'been': 'AUX', 'being': 'AUX', 'has': 'AUX', 'have': 'AUX', 'had': 'AUX',
    'do': 'AUX', 'does': 'AUX', 'did': 'AUX', 'will': 'AUX', 'would': 'AUX',
    'can': 'AUX', 'could': 'AUX', 'shall': 'AUX', 'should': 'AUX', 'may': 'AUX',
    'might': 'AUX', 'must': 'AUX',

    # Common nouns (basic)
    'apple': 'NOUN', 'apples': 'NOUN', 'ball': 'NOUN', 'balls': 'NOUN',
    'sun': 'NOUN', 'water': 'NOUN', 'moon': 'NOUN', 'tree': 'NOUN', 'trees': 'NOUN',
    'bird': 'NOUN', 'birds': 'NOUN', 'fish': 'NOUN', 'fishes': 'NOUN',
    'flower': 'NOUN', 'flowers': 'NOUN', 'man': 'NOUN', 'men': 'NOUN',
    'child': 'NOUN', 'children': 'NOUN', 'teacher': 'NOUN', 'teachers': 'NOUN',
    'artist': 'NOUN', 'artists': 'NOUN', 'thinker': 'NOUN', 'thinkers': 'NOUN',
    'scientist': 'NOUN', 'scientists': 'NOUN', 'student': 'NOUN', 'students': 'NOUN',
    'stone': 'NOUN', 'stones': 'NOUN', 'star': 'NOUN', 'stars': 'NOUN',
    'river': 'NOUN', 'rivers': 'NOUN', 'wind': 'NOUN', 'fire': 'NOUN',
    'woman': 'NOUN', 'women': 'NOUN', 'person': 'NOUN', 'people': 'NOUN',
    'Socrates': 'NOUN', 'human': 'NOUN', 'humanity': 'NOUN', 'truth': 'NOUN',
    'virtue': 'NOUN', 'knowledge': 'NOUN', 'wisdom': 'NOUN',

    # Common adjectives
    'red': 'ADJ', 'round': 'ADJ', 'hot': 'ADJ', 'liquid': 'ADJ', 'bright': 'ADJ',
    'tall': 'ADJ', 'small': 'ADJ', 'slimy': 'ADJ', 'colorful': 'ADJ', 'strong': 'ADJ',
    'curious': 'ADJ', 'wise': 'ADJ', 'creative': 'ADJ', 'rational': 'ADJ',
    'absolute': 'ADJ', 'good': 'ADJ', 'ripe': 'ADJ', 'eager': 'ADJ',
    'heavy': 'ADJ', 'light': 'ADJ', 'fast': 'ADJ', 'cold': 'ADJ', 'warm': 'ADJ',
    'deep': 'ADJ', 'big': 'ADJ', 'large': 'ADJ', 'tiny': 'ADJ',
    'beautiful': 'ADJ', 'ugly': 'ADJ', 'happy': 'ADJ', 'sad': 'ADJ',

    # Common verbs
    'fall': 'VERB', 'falls': 'VERB', 'fell': 'VERB', 'fallen': 'VERB',
    'roll': 'VERB', 'rolls': 'VERB', 'rolled': 'VERB',
    'shine': 'VERB', 'shines': 'VERB', 'shone': 'VERB',
    'flow': 'VERB', 'flows': 'VERB', 'flowed': 'VERB',
    'grow': 'VERB', 'grows': 'VERB', 'grew': 'VERB', 'grown': 'VERB',
    'fly': 'VERB', 'flies': 'VERB', 'flew': 'VERB', 'flown': 'VERB',
    'swim': 'VERB', 'swims': 'VERB', 'swam': 'VERB', 'swum': 'VERB',
    'bloom': 'VERB', 'blooms': 'VERB', 'bloomed': 'VERB',
    'run': 'VERB', 'runs': 'VERB', 'ran': 'VERB',
    'learn': 'VERB', 'learns': 'VERB', 'learned': 'VERB',
    'teach': 'VERB', 'teaches': 'VERB', 'taught': 'VERB',
    'paint': 'VERB', 'paints': 'VERB', 'painted': 'VERB',
    'think': 'VERB', 'thinks': 'VERB', 'thought': 'VERB',
    'exist': 'VERB', 'exists': 'VERB', 'existed': 'VERB',
    'guide': 'VERB', 'guides': 'VERB', 'guided': 'VERB',
    'move': 'VERB', 'moves': 'VERB', 'moved': 'VERB',
    'rise': 'VERB', 'rises': 'VERB', 'rose': 'VERB', 'risen': 'VERB',
    'sing': 'VERB', 'sings': 'VERB', 'sang': 'VERB', 'sung': 'VERB',
    'dance': 'VERB', 'dances': 'VERB', 'danced': 'VERB',
    'speak': 'VERB', 'speaks': 'VERB', 'spoke': 'VERB', 'spoken': 'VERB',
    'orbit': 'VERB', 'orbits': 'VERB', 'orbited': 'VERB',
    'philosophize': 'VERB', 'philosophizes': 'VERB',
    'observe': 'VERB', 'observes': 'VERB', 'observed': 'VERB',
    'question': 'VERB', 'questions': 'VERB', 'questioned': 'VERB',
    'be': 'VERB', 'am': 'VERB', 'are': 'VERB', 'is': 'VERB',
    'was': 'VERB', 'were': 'VERB', 'been': 'VERB',
}

# Plural nouns (for morphological agreement)
PLURAL_NOUNS = {
    'apples', 'balls', 'trees', 'birds', 'fishes', 'flowers',
    'men', 'children', 'teachers', 'artists', 'thinkers',
    'scientists', 'students', 'stones', 'stars', 'rivers',
    'women', 'people',
}

# Singular verb forms (he/she/it -s)
SINGULAR_VERBS = {
    'falls', 'rolls', 'shines', 'flows', 'grows', 'flies',
    'swims', 'blooms', 'runs', 'learns', 'teaches', 'paints',
    'thinks', 'exists', 'guides', 'moves', 'rises', 'sings',
    'dances', 'speaks', 'orbits', 'philosophizes', 'observes',
    'questions', 'is', 'has', 'does',
}

# Plural verb forms (they -)
PLURAL_VERBS = {
    'fall', 'roll', 'shine', 'flow', 'grow', 'fly',
    'swim', 'bloom', 'run', 'learn', 'teach', 'paint',
    'think', 'exist', 'guide', 'move', 'rise', 'sing',
    'dance', 'speak', 'orbit', 'philosophize', 'observe',
    'question', 'are', 'have', 'do',
}


def tokenize_sentence(text: str) -> list[str]:
    """Simple space-and-punctuation tokenizer."""
    # Remove special markers
    text = re.sub(r'\[(?:CONCEPT|SENTENCE|RECONSTRUCT|ENT|PROP|ACT|REL):[^\]]*\]', '', text)
    # Split on whitespace and punctuation
    tokens = re.findall(r"[a-zA-Z']+(?:[.?!])?|[.?!,;]", text)
    return [t for t in tokens if t]


def pos_tag(token: str) -> str:
    """Get the POS tag for a token."""
    token_lower = token.lower().strip('.,;!?')
    return POS_TAGS.get(token_lower, 'UNKNOWN')


# ═════════════════════════════════════════════════════════════════════
# RULE 1: Subject-Verb Agreement
# ═════════════════════════════════════════════════════════════════════

def check_subject_verb_agreement(sentence: str) -> float:
    """Check if the subject and verb agree in number.

    Returns:
        -1.0 (violation) to +1.0 (perfect agreement)
    """
    tokens = tokenize_sentence(sentence)
    if len(tokens) < 3:
        return 0.0  # Too short to judge

    # Find the first noun + verb pair after DET
    found_subject = None
    found_verb = None

    i = 0
    while i < len(tokens):
        t = tokens[i].lower().strip('.,;!?')
        tag = POS_TAGS.get(t, 'UNKNOWN')

        if tag == 'DET' and found_subject is None:
            i += 1
            continue

        if tag == 'NOUN' and found_subject is None:
            found_subject = t
            i += 1
            continue

        if tag in ('VERB', 'AUX') and found_subject is not None and found_verb is None:
            found_verb = t
            break

        i += 1

    if found_subject is None or found_verb is None:
        return 0.0  # Can't determine

    # Check agreement
    subj_plural = found_subject in PLURAL_NOUNS
    verb_singular = found_verb in SINGULAR_VERBS
    verb_plural = found_verb in PLURAL_VERBS

    # "Socrates is" — treat proper nouns as singular
    if found_subject[0].isupper() and found_subject not in PLURAL_NOUNS:
        subj_plural = False

    if subj_plural and verb_plural:
        return 1.0  # "The apples fall" ✓
    elif not subj_plural and verb_singular:
        return 1.0  # "The apple falls" ✓
    elif subj_plural and verb_singular:
        return -1.0  # "The apples falls" ✗
    elif not subj_plural and verb_plural:
        # "The apple fall" — could be subjunctive, usually ungrammatical
        if found_verb not in ('be',):
            return -0.5
        return 0.0

    return 0.0


# ═════════════════════════════════════════════════════════════════════
# RULE 2: Determiner-Noun Agreement
# ═════════════════════════════════════════════════════════════════════

def check_det_noun_agreement(sentence: str) -> float:
    """Check determiners agree with noun number.

    'a/an' → singular nouns only
    'these/those' → plural nouns only

    Returns:
        -1.0 (violation) to +1.0 (perfect)
    """
    tokens = tokenize_sentence(sentence)
    if len(tokens) < 2:
        return 0.0

    score = 1.0
    for i in range(len(tokens) - 1):
        t = tokens[i].lower()
        next_t = tokens[i + 1].lower().strip('.,;!?')

        # 'a'/'an' must be followed by singular noun
        if t in ('a', 'an') and next_t in PLURAL_NOUNS:
            score = min(score, -1.0)

        # 'these'/'those' must be followed by plural noun
        if t in ('these', 'those') and next_t not in PLURAL_NOUNS and next_t in POS_TAGS and POS_TAGS[next_t] == 'NOUN':
            score = min(score, -1.0)

        # 'every'/'each' must be followed by singular noun
        if t in ('each', 'every') and next_t in PLURAL_NOUNS:
            score = min(score, -1.0)

    return score


# ═════════════════════════════════════════════════════════════════════
# RULE 3: Sentence Structure (must have NP + VP)
# ═════════════════════════════════════════════════════════════════════

def check_sentence_structure(sentence: str) -> float:
    """Check that the sentence has a basic NP VP structure.

    Returns:
        -1.0 (structureless) to +1.0 (well-formed)
    """
    tokens = tokenize_sentence(sentence)
    if len(tokens) < 2:
        return -1.0  # Too short

    has_noun = any(POS_TAGS.get(t.lower().strip('.,;!?'), 'UNKNOWN') == 'NOUN' for t in tokens)
    has_verb = any(POS_TAGS.get(t.lower().strip('.,;!?'), 'UNKNOWN') in ('VERB', 'AUX') for t in tokens)

    if has_noun and has_verb:
        return 1.0
    elif has_noun or has_verb:
        return -0.3
    else:
        return -0.8


# ═════════════════════════════════════════════════════════════════════
# RULE 4: Punctuation (comma before conjunction + end punctuation)
# ═════════════════════════════════════════════════════════════════════

def check_punctuation(sentence: str) -> float:
    """Check basic punctuation rules.

    Returns:
        -1.0 to +1.0
    """
    stripped = sentence.strip()
    if not stripped:
        return -1.0

    score = 0.0

    # Must end with . ! or ?
    if any(stripped.endswith(p) for p in ['.', '!', '?']):
        score += 0.5
    else:
        score -= 0.3

    # Check for comma before coordinating conjunction joining sentences
    tokens = tokenize_sentence(stripped)
    for i, t in enumerate(tokens):
        if t.lower() in ('and', 'but', 'or', 'so') and i > 0 and i < len(tokens) - 1:
            prev = tokens[i - 1]
            if prev.endswith(','):
                score += 0.3
            elif len(tokens) > 3:
                score -= 0.1  # Missing comma in compound sentence

    return max(-1.0, min(1.0, score))


# ═════════════════════════════════════════════════════════════════════
# RULE 5: Capitalization (Sentence starts with capital letter)
# ═════════════════════════════════════════════════════════════════════

def check_capitalization(sentence: str) -> float:
    """Check that the sentence begins with a capital letter."""
    stripped = sentence.strip()
    if not stripped:
        return -1.0
    if stripped[0].isupper():
        return 1.0
    return -1.0


# ═════════════════════════════════════════════════════════════════════
# RULE 6: Long-Distance Dependency Score (Value Head Intuition Target)
# ═════════════════════════════════════════════════════════════════════
# This is what the Value Head should learn to predict.
# A sentence starting with "Although..." commits to a comma and
# contrastive clause within ~10 tokens.

def score_long_distance_structure(sentence: str) -> float:
    """Score the structural commitment of long-distance dependencies.

    High score = the sentence has opened and properly closed
    structural commitments (subordinate clauses, conjunctions, etc.)

    Returns:
        -1.0 to +1.0
    """
    tokens = tokenize_sentence(sentence)
    if len(tokens) < 4:
        return 0.0

    score = 0.0
    open_commitments = 0
    closed_commitments = 0

    # Track subordinating conjunctions that open dependencies
    opens_subordinate = {'although', 'though', 'while', 'whereas',
                         'because', 'since', 'when', 'if', 'unless'}

    # Track coordinating conjunctions that close dependencies
    closes_coordinate = {'and', 'but', 'or', 'so', 'yet', 'for', 'nor'}

    for i, t in enumerate(tokens):
        t_lower = t.lower().strip(',')

        if t_lower in opens_subordinate:
            open_commitments += 1
            # Check if there's a comma within 10 tokens
            future_tokens = tokens[i+1:i+11]
            if any(',' in ft or ft == ',' for ft in future_tokens):
                closed_commitments += 1

        if t_lower in closes_coordinate and i > 0:
            # A coordinating conjunction after a comma closes a commitment
            if tokens[i-1].endswith(',') or tokens[i-1] == ',':
                closed_commitments += 1

    if open_commitments == 0:
        score = 0.5  # Simple sentences are fine
    elif closed_commitments >= open_commitments:
        score = 1.0  # All commitments properly closed
    elif closed_commitments > 0:
        score = 0.3  # Partially closed
    else:
        score = -0.5  # Opened commitment never closed

    return score


# ═════════════════════════════════════════════════════════════════════
# COMPOSITE GRAMMAR SCORE
# ═════════════════════════════════════════════════════════════════════

def grammar_score(sentence: str, weights: dict = None) -> float:
    """Compute the overall grammaticality score for a sentence.

    Combines all grammar rules with configurable weights.

    Args:
        sentence: The sentence to evaluate
        weights: Dict of rule_name->weight (default: equal weights)

    Returns:
        float: -1 (completely ungrammatical) to +1 (perfect)
    """
    if weights is None:
        weights = {
            'subject_verb': 0.25,
            'det_noun': 0.15,
            'structure': 0.30,
            'punctuation': 0.15,
            'capitalization': 0.10,
            'long_distance': 0.05,
        }

    scores = {
        'subject_verb': check_subject_verb_agreement(sentence),
        'det_noun': check_det_noun_agreement(sentence),
        'structure': check_sentence_structure(sentence),
        'punctuation': check_punctuation(sentence),
        'capitalization': check_capitalization(sentence),
        'long_distance': score_long_distance_structure(sentence),
    }

    total = sum(scores[k] * weights.get(k, 0) for k in scores)
    return max(-1.0, min(1.0, total))


# ═════════════════════════════════════════════════════════════════════
# Z3 INTEGRATION — Logic Constraint Checking
# ═════════════════════════════════════════════════════════════════════
# Z3 verifies that generated statements are logically consistent
# with the concept being expressed.

def check_z3_consistency(sentence: str, concept_text: str) -> float:
    """Check if a sentence is logically consistent with its source concept.

    Uses the Z3-based logic verifier to detect contradictions.
    For example, if the concept says [ENT:apple][PROP:red] but the
    sentence says "The blue apple", Z3 detects the contradiction.

    Args:
        sentence: Generated sentence
        concept_text: The source concept in [TAG:value] format

    Returns:
        float: -1.0 (contradiction) to +1.0 (consistent)
    """
    try:
        from tabula_rasa.cognitive.logic_verifier import LogicVerifier
        verifier = LogicVerifier()

        # Build TRLD format: concept as premise, sentence as conclusion
        # Check if sentence follows from the concept
        premises = concept_text
        result = verifier.verify_trld(
            f"[PREMISE:{premises}] [CONCLUSION:{sentence}]"
        )

        if result and result.get('valid') is True:
            return 1.0
        elif result and result.get('valid') is False:
            return -1.0
        else:
            return 0.0
    except Exception:
        return 0.0


# ═════════════════════════════════════════════════════════════════════
# GRAMMAR RULE FUNCTIONS FOR MCTS FILTERING
# ═════════════════════════════════════════════════════════════════════
# These are designed to be passed to the MCTS module.

def make_grammar_rule_checker(mode: str = 'score'):
    """Create a grammar rule function compatible with the MCTS module.

    The MCTS module expects functions with signature:
        rule_fn(text: str, check: str = 'score') -> float

    Args:
        mode: 'score' or 'validity'

    Returns:
        Function that computes grammar score
    """
    def _checker(text: str, check: str = 'score') -> float:
        if check == 'validity':
            # For pruning during MCTS expand
            s = grammar_score(text)
            return s
        # Default: score for rollout evaluation
        return grammar_score(text)
    return _checker


def create_grammar_rules() -> list:
    """Create a list of grammar rule functions for MCTS."""
    return [make_grammar_rule_checker()]


# ═════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Test sentences
    tests = [
        ("The red apple falls.", "Simple correct sentence"),
        ("the apple", "Fragment"),
        ("The apples falls.", "Subject-verb disagreement"),
        ("A apples are red.", "Det-noun disagreement"),
        ("The ball rolls.", "Correct"),
        ("Although the weather was cold, the children played outside.", "Long-distance dependency"),
        ("Because it was raining we stayed inside.", "Missing comma after subordinate clause"),
        ("The child learn quickly.", "Subject-verb disagreement (child + learn)"),
        ("The children learn quickly.", "Correct plural"),
        ("Socrates is a human who thinks.", "Correct complex"),
    ]

    print('Grammar Engine Tests:\n')
    for sentence, desc in tests:
        score = grammar_score(sentence)
        subj_v = check_subject_verb_agreement(sentence)
        struct = check_sentence_structure(sentence)
        punct = check_punctuation(sentence)
        cap = check_capitalization(sentence)
        ld = score_long_distance_structure(sentence)

        print(f'  {desc}:')
        print(f'    "{sentence}"')
        print(f'    Overall: {score:+.2f} | SV:{subj_v:+.1f} Struct:{struct:+.1f} '
              f'Punct:{punct:+.1f} Cap:{cap:+.1f} LD:{ld:+.1f}')
        print()

    # Z3 test
    print('Z3 Consistency Test:')
    concept = "[CONCEPT:[ENT:apple][PROP:red][ACT:fall]]"
    good = "The red apple falls."
    bad = "The blue apple rises."
    print(f'  Concept: {concept}')
    print(f'  "{good}" → {check_z3_consistency(good, concept):+.1f}')
    print(f'  "{bad}" → {check_z3_consistency(bad, concept):+.1f}')
