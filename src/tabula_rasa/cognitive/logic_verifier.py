"""Logic Verifier — objective ground truth for Socratic debates.

Uses Z3 Theorem Prover to verify logical consistency of arguments.
Supports two modes:

1. **TRLD (Tabula Rasa Logic Dialect)** — Controlled Natural Language grammar
   with deterministic AST-to-Z3 mapping. This is the primary mode.
   Example: [PREMISE: All humans are mortal] [PREMISE: Socrates is human]
            [CONCLUSION: Socrates is mortal]

2. **Free-text heuristic** — legacy keyword-based scoring (the "Honest Caveat"
   in Section 8.1). Kept as fallback; scores 0-100 but does NOT guarantee
   formal logical validity.

Usage:
    from logic_verifier import LogicVerifier
    v = LogicVerifier()
    # TRLD mode (deterministic, recommended):
    v.verify_trld("[PREMISE: All humans are mortal] [PREMISE: Socrates is human] [CONCLUSION: Socrates is mortal]")
    # Free-text heuristic (fallback):
    v.score_debate_argument("All humans are mortal and Socrates is human therefore...")
"""

import re

try:
    from z3 import *
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════
# TRLD — Tabula Rasa Logic Dialect Grammar
# ═══════════════════════════════════════════════════════════════════
#
# A strict Controlled Natural Language (CNL) that maps deterministically
# to Z3 AST nodes. No ambiguity — every construct has a fixed parse rule.
#
# Grammar (EBNF):
#
#   statement    = { premise_statement | conclusion_statement |
#                    if_statement | then_statement | quantifier_statement |
#                    connective_statement }
#
#   premise_statement    = "[PREMISE:" free_text "]"
#   conclusion_statement = "[CONCLUSION:" free_text "]"
#   if_statement         = "[IF:"   free_text "]"
#   then_statement       = "[THEN:" free_text "]"
#   quantifier_statement = "[QUANTIFIER:" quantifier_type "]"  (future)
#   connective_statement = "[" connective_type ":" free_text "]"
#
#   quantifier_type = "all" | "some" | "none" | "every" | "most"
#   connective_type = "AND" | "OR" | "NOT" | "IMPLIES"
#   free_text       = any text (parsed by deterministic patterns)
#
# The Socratic Engine (Stages 2 & 3) MUST emit TRLD format when
# generating arguments. The Value Judge rewards strict syntax compliance
# and penalizes unstructured prose.

def _trld_tokenize(text: str) -> list[dict]:
    """Tokenize TRLD-formatted text into structured clauses.

    Returns a list of dicts: {tag: str, content: str}
    where tag is 'PREMISE', 'CONCLUSION', 'IF', 'THEN', etc.
    """
    pattern = r'\[(\w+):\s*(.*?)\s*\]'
    clauses = []
    for m in re.finditer(pattern, text, re.DOTALL):
        tag = m.group(1).upper()
        content = m.group(2).strip()
        clauses.append({'tag': tag, 'content': content})
    return clauses


def _trld_detect(text: str) -> bool:
    """Detect if text uses TRLD format (starts with a bracket-tagged clause)."""
    return bool(re.match(r'\s*\[', text))


# ═══════════════════════════════════════════════════════════════════
# Z3 Helpers — Deterministic AST Construction
# ═══════════════════════════════════════════════════════════════════

class TRLDSymbolTable:
    """Manages Z3 sorts, predicates, and constants during TRLD parsing.

    Uses first-order logic with uninterpreted sorts and predicates:
    - "All humans are mortal" → ForAll(x, human(x) => mortal(x))
    - "Socrates is human" → human(socrates)
    - "Socrates is mortal" → mortal(socrates)

    This gives proper variable unification that propositional logic
    cannot achieve.
    """

    def __init__(self):
        # The single uninterpreted sort for all individuals
        self.Subject = DeclareSort('Subject')
        self._predicates = {}      # name → Function(name, Subject, BoolSort())
        self._constants = {}       # name → Const(name, Subject)
        self._variables = {}       # ForAll variable cache

    def predicate(self, name: str):
        """Get or create a unary predicate over Subject.

        Normalizes nouns and verbs to handle singular/plural and verb-form
        variants: "human" and "humans" map to the same predicate;
        "rains", "raining", "rained" all map to "rain".

        e.g. predicate('human') → Function('human', Subject, BoolSort())
        """
        norm = name.strip().lower().replace(' ', '_')
        # Plural → singular normalization
        # Handles: "humans"→"human", "stones"→"stone", "cities"→"city"
        if norm.endswith('ies') and len(norm) > 4:
            norm = norm[:-3] + 'y'
        elif norm.endswith('ss'):
            pass  # keep 'ss' (e.g., "class" stays "class")
        elif norm.endswith('s') and len(norm) > 3:
            norm = norm[:-1]  # strip trailing 's'

        # Verb ending normalization — only for single-word predicates
        # (multi-word compound names like "mortal_being" should NOT be
        # further stripped, as they're noun phrases, not verbs)
        if '_' not in norm:
            if norm.endswith('ing') and len(norm) > 5:
                norm = norm[:-3]
                if len(norm) > 2 and norm[-1] == norm[-2]:
                    norm = norm[:-1]
            elif norm.endswith('ed') and len(norm) > 4:
                norm = norm[:-2]
                if norm.endswith('i'):
                    norm = norm[:-1] + 'y'
        if norm not in self._predicates:
            self._predicates[norm] = Function(norm, self.Subject, BoolSort())
        return self._predicates[norm]

    def constant(self, name: str):
        """Get or create a named individual constant.

        e.g. constant('Socrates') → Const('socrates', Subject)
        """
        norm = name.strip().lower().replace(' ', '_')
        if norm not in self._constants:
            self._constants[norm] = Const(norm, self.Subject)
        return self._constants[norm]

    def fresh_var(self):
        """Get a fresh Z3 variable for ForAll quantification."""
        return Const('x', self.Subject)


def _parse_simple_proposition(text: str, sym: TRLDSymbolTable):
    """Parse a simple proposition like 'Socrates is mortal' or 'It rains'.

    Returns a Z3 expression using first-order predicates over Subject.
    """
    text = text.strip()

    # "X is not Y" or "X does not Y" — negation (check BEFORE simple X is Y)
    m = re.match(r'(\w+(?:\s+\w+)*?)\s+(?:is|are|was|were|does|do|has|have)\s+not\s+(.+)$', text, re.IGNORECASE)
    if m:
        subj = sym.constant(m.group(1).strip())
        pred = sym.predicate(m.group(2).strip())
        return Not(pred(subj))

    # "X cannot Y" or "X can't Y" — negation with modal
    m = re.match(r'(\w+(?:\s+\w+)*?)\s+(?:cannot|can\'?t)\s+(.+)$', text, re.IGNORECASE)
    if m:
        subj = sym.constant(m.group(1).strip())
        pred = sym.predicate(m.group(2).strip())
        return Not(pred(subj))

    # "X is Y" or "X are Y" — X is a constant, Y is a predicate
    # e.g., "Socrates is mortal" → mortal(Socrates)
    # Also handles: gets, becomes, turns, feels, seems, remains, can, will
    m = re.match(r'(\w+(?:\s+\w+)*?)\s+(?:is|are|was|were|gets|get|becomes|become|turns|turn|feels|feel|seems|seem|remains|remain|can|could|will|would|may|might|must|shall)\s+(.+)$', text, re.IGNORECASE)
    if m:
        subject_str = m.group(1).strip()
        predicate_str = m.group(2).strip()
        # Strip leading articles: "a stone" → "stone", "an apple" → "apple"
        predicate_str = re.sub(r'^(an?|the)\s+', '', predicate_str, flags=re.IGNORECASE)
        subj = sym.constant(subject_str)
        pred = sym.predicate(predicate_str)
        return pred(subj)

    # "X has Y" or "X have Y"
    m = re.match(r'(\w+(?:\s+\w+)*?)\s+(?:has|have)\s+(.+)$', text, re.IGNORECASE)
    if m:
        subject_str = m.group(1).strip()
        obj_str = m.group(2).strip()
        subj = sym.constant(subject_str)
        pred_name = f"has_{obj_str.lower().replace(' ', '_')}"
        pred = sym.predicate(pred_name)
        return pred(subj)

    # "It X" (impersonal verb: "it rains", "it holds")
    # Also catches "It is X" or "It's X" for normalization
    m = re.match(r'(?:it|it\'?s)\s+(?:is\s+)?(.+)$', text, re.IGNORECASE)
    if m:
        pred_name = m.group(1).strip().lower().replace(' ', '_')
        # For impersonal statements, use 'it' as the constant
        # (same constant as when "X is Y" pattern parses "It is Y")
        it_const = sym.constant('it')
        pred = sym.predicate(pred_name)
        return pred(it_const)

    # ── Negation checks are handled at the top of this function ──

    # "X Y" bare verb form — subject + bare predicate verb
    # e.g., "Socrates dies" → die(socrates), "Granite exists" → exist(granite)
    # The second word must end in 's', 'ed', or be a known bare verb
    m = re.match(r'(\w+(?:\s+\w+)*?)\s+(\w+)$', text, re.IGNORECASE)
    if m and len(m.group(1).split()) <= 2:  # Limit subject to 2 words
        # This catches cases missed by other patterns.
        # For "Socrates dies": subject="Socrates", verb="dies"
        # The predicate is formed from the verb.
        verb = m.group(2).lower()
        # Normalize verb form: "dies" → "die", "flies" → "fly"
        verb_norm = verb
        if verb_norm.endswith('ies') and len(verb_norm) > 4:
            verb_norm = verb_norm[:-3] + 'y'
        elif verb_norm.endswith('s') and not verb_norm.endswith('ss') and len(verb_norm) > 2:
            verb_norm = verb_norm[:-1]  # strip trailing 's'

        subj = sym.constant(m.group(1).strip())
        pred = sym.predicate(verb_norm)
        return pred(subj)

    # Fallback: try as a single proposition (for atomic statements like "P")
    # Use a dummy constant
    pred_name = text.lower().replace(' ', '_')
    dummy = sym.constant(f'__prop__{pred_name}')
    pred = sym.predicate(pred_name)
    return pred(dummy)


def _parse_quantified_statement(text: str, sym: TRLDSymbolTable):
    """Parse a quantified statement like 'All humans are mortal'.

    Uses Z3 ForAll quantifier with predicates over the Subject sort.
    """
    text = text.strip()

    # "All X are Y" → ForAll(x, X(x) => Y(x))
    # "All X can Y" → ForAll(x, X(x) => Y(x))
    # The verb (are/is/can/etc.) is REQUIRED to delimit domain from predicate.
    m = re.match(r'(All|Every|No|Some|Most)\s+(\w+(?:\s+\w+)*?)\s+(?:(?:are|is|can|could|will|would|may|might|must|shall)\s+)(\w+(?:\s+\w+)*?)$', text, re.IGNORECASE)
    if m:
        quant = m.group(1).lower()
        domain_str = m.group(2).lower().strip().replace(' ', '_')
        predicate_str = m.group(3).strip().lower()

        domain_pred = sym.predicate(domain_str)
        predicate_pred = sym.predicate(predicate_str)
        x = sym.fresh_var()

        if quant == 'no':
            return ForAll([x], Implies(domain_pred(x), Not(predicate_pred(x))))
        elif quant == 'some':
            return Exists([x], And(domain_pred(x), predicate_pred(x)))
        else:
            return ForAll([x], Implies(domain_pred(x), predicate_pred(x)))

    # Bare verb form: "All mortal beings die" → last word is the predicate
    # This is for cases like "All X Y" where Y is a bare verb (die, exist, etc.)
    # The last word becomes the predicate, everything before it is the domain.
    m = re.match(r'(All|Every|No|Some|Most)\s+(.+?)\s+(\w+)$', text, re.IGNORECASE)
    if m:
        quant = m.group(1).lower()
        domain_str = m.group(2).strip().lower().replace(' ', '_')
        verb_str = m.group(3).strip().lower()
        # Normalize verb: "dies" → "die", "flies" → "fly"
        if verb_str.endswith('ies') and len(verb_str) > 4:
            verb_str = verb_str[:-3] + 'y'
        elif verb_str.endswith('s') and not verb_str.endswith('ss') and len(verb_str) > 2:
            verb_str = verb_str[:-1]

        domain_pred = sym.predicate(domain_str)
        predicate_pred = sym.predicate(verb_str)
        x = sym.fresh_var()

        if quant == 'no':
            return ForAll([x], Implies(domain_pred(x), Not(predicate_pred(x))))
        elif quant == 'some':
            return Exists([x], And(domain_pred(x), predicate_pred(x)))
        else:
            return ForAll([x], Implies(domain_pred(x), predicate_pred(x)))

    return None


def _trld_clause_to_z3(clause: dict, sym: TRLDSymbolTable) -> object:
    """Convert a single TRLD clause to a Z3 expression.

    Returns a Z3 Bool expression (formula) or None on failure.
    """
    tag = clause['tag']
    content = clause['content']

    if tag == 'PREMISE':
        # Try quantified form first, then simple proposition
        q = _parse_quantified_statement(content, sym)
        if q is not None:
            return q
        return _parse_simple_proposition(content, sym)

    elif tag == 'CONCLUSION':
        return _parse_simple_proposition(content, sym)

    elif tag == 'IF':
        return _parse_simple_proposition(content, sym)

    elif tag == 'THEN':
        return _parse_simple_proposition(content, sym)

    elif tag == 'NOT':
        return Not(_parse_simple_proposition(content, sym))

    elif tag in ('AND', 'OR'):
        parts = [p.strip() for p in re.split(r'[,;]', content) if p.strip()]
        if len(parts) < 2:
            return _parse_simple_proposition(content, sym)
        exprs = [_parse_simple_proposition(p, sym) for p in parts]
        if tag == 'AND':
            return And(*exprs) if len(exprs) > 1 else exprs[0]
        else:
            return Or(*exprs) if len(exprs) > 1 else exprs[0]

    elif tag == 'IMPLIES':
        parts = re.split(r'\s*,\s*', content, maxsplit=1)
        if len(parts) == 2:
            antecedent = _parse_simple_proposition(parts[0], sym)
            consequent = _parse_simple_proposition(parts[1], sym)
            return Implies(antecedent, consequent)
        return None

    elif tag == 'QUANTIFIER':
        return _parse_quantified_statement(content, sym)

    return None


# ═══════════════════════════════════════════════════════════════════
# LogicVerifier
# ═══════════════════════════════════════════════════════════════════

class LogicVerifier:
    """Verifies logical arguments using Z3 Theorem Prover.

    Two modes:
    1. verify_trld(trld_text) — deterministic TRLD grammar → Z3 AST
    2. score_debate_argument(text) — legacy free-text heuristic (0-100)

    Default: verify_trld is called first. If text doesn't match TRLD,
    falls back to score_debate_argument.
    """

    def __init__(self):
        if not HAS_Z3:
            print('[!] Z3 not installed. Run: pip install z3-solver')
        self.stats = {'verified': 0, 'refuted': 0, 'unknown': 0}
        # Track how often each mode is used
        self.mode_usage = {'trld': 0, 'heuristic': 0}

    # ─── TRLD: Deterministic Mode ─────────────────────────────────

    def verify_trld(self, text: str) -> dict:
        """Verify a TRLD-formatted argument using deterministic Z3 AST mapping.

        Parses [PREMISE: ...], [CONCLUSION: ...], [IF: ...], [THEN: ...],
        and other TRLD tags, constructs Z3 constraints, and checks validity.

        Returns:
            {'valid': bool|None, 'reason': str, 'error': str|None}
        """
        if not HAS_Z3:
            return {'valid': None, 'error': 'Z3 not installed'}

        clauses = _trld_tokenize(text)
        if not clauses:
            return {'valid': None, 'error': 'No TRLD clauses found'}

        self.mode_usage['trld'] += 1
        sym = TRLDSymbolTable()
        solver = Solver()

        premises = []
        conclusions = []
        if_antecedent = None
        then_consequent = None

        # Phase 1: Parse clauses into Z3 expressions
        for clause in clauses:
            tag = clause['tag']
            expr = _trld_clause_to_z3(clause, sym)

            if expr is None:
                continue

            if tag == 'PREMISE':
                premises.append(expr)
            elif tag == 'CONCLUSION':
                conclusions.append(expr)
            elif tag == 'IF':
                if_antecedent = expr
            elif tag == 'THEN':
                then_consequent = expr
            elif tag in ('NOT', 'AND', 'OR', 'IMPLIES', 'QUANTIFIER'):
                premises.append(expr)
            else:
                premises.append(expr)

        # Phase 2: Build the logical implication: premises => conclusion
        if not premises:
            return {'valid': None, 'error': 'No premises found in TRLD text'}

        # Handle IF/THEN as implicit implication
        if if_antecedent is not None and then_consequent is not None:
            premises.append(Implies(if_antecedent, then_consequent))

        # Build combined premise
        premise_expr = And(*premises) if len(premises) > 1 else premises[0]

        if not conclusions:
            # No explicit conclusion — check that premises are consistent (no contradiction)
            solver.add(premise_expr)
            result = solver.check()
            if result == sat:
                return {
                    'valid': True,
                    'reason': 'Premises are consistent (satisfiable)',
                    'premises': [c['content'] for c in clauses],
                }
            elif result == unsat:
                return {
                    'valid': False,
                    'reason': 'Premises contain a contradiction',
                    'premises': [c['content'] for c in clauses],
                }
            else:
                return {
                    'valid': None,
                    'reason': 'Z3 could not determine satisfiability',
                }

        # Check: premises AND NOT conclusion => UNSAT means valid
        conclusion_expr = And(*conclusions) if len(conclusions) > 1 else conclusions[0]
        solver.add(premise_expr)
        solver.add(Not(conclusion_expr))
        result = solver.check()

        if result == unsat:
            self.stats['verified'] += 1
            return {
                'valid': True,
                'reason': 'Logically follows — premises entail conclusion',
                'premises': [c['content'] for c in clauses if c['tag'] in ('PREMISE', 'IF', 'THEN', 'QUANTIFIER', 'NOT', 'AND', 'OR', 'IMPLIES')],
                'conclusion': [c['content'] for c in clauses if c['tag'] == 'CONCLUSION'],
            }
        elif result == sat:
            self.stats['refuted'] += 1
            # Try to get a counter-example
            model = solver.model()
            counter_example = {}
            for d in model.decls():
                counter_example[d.name()] = str(model[d])
            return {
                'valid': False,
                'reason': 'Does not follow logically — counter-example exists',
                'counter_example': counter_example,
            }
        else:
            self.stats['unknown'] += 1
            return {
                'valid': None,
                'reason': 'Z3 could not determine validity (unknown)',
            }

    # ─── Free-text Heuristic Mode (Legacy / Honest Caveat) ─────────

    def verify_syllogism(self, premise: str, conclusion: str) -> dict:
        """Legacy mode: verify a syllogism using heuristic regex patterns.

        NOTE: This is the "Honest Caveat" from Section 8.1. It uses keyword
        heuristics, not formal theorem proving. For deterministic verification,
        use verify_trld() instead.

        Kept as fallback for backward compatibility.
        """
        if not HAS_Z3:
            return {'valid': None, 'error': 'Z3 not installed'}

        # Check if the combined text is TRLD format
        combined = f"{premise} {conclusion}"
        if _trld_detect(combined):
            return self.verify_trld(combined)

        self.mode_usage['heuristic'] += 1

        try:
            s = Solver()
            # Parse simple syllogisms with pattern matching (heuristic)
            # "All X are Y. Z is X. Therefore Z is Y."
            pattern = r'All\s+(\w+)\s+are\s+(\w+)\s*\.?\s*(.*)'
            m = re.match(pattern, premise, re.IGNORECASE)
            if m:
                A, B = m.group(1).lower(), m.group(2).lower()
                p = Bool(f'is_{A}')
                q = Bool(f'is_{B}')
                s.add(Implies(p, q))
                s.add(p)
                s.add(Not(q))
                result = s.check()
                valid = (result == unsat)
                self.stats['verified' if valid else 'refuted'] += 1
                return {
                    'valid': valid,
                    'premise': premise,
                    'conclusion': conclusion,
                    'reason': 'Logically follows (heuristic)' if valid else 'Does not follow (heuristic)',
                    '_mode': 'heuristic_pattern',
                }

            # "If P then Q. P. Therefore Q." (Modus Ponens)
            pattern2 = r'If\s+(.+?)\s+then\s+(.+?)\.?\s*(.+)'
            m = re.match(pattern2, premise, re.IGNORECASE)
            if m:
                P = Bool('P')
                Q = Bool('Q')
                s.add(Implies(P, Q))
                s.add(P)
                s.add(Not(Q))
                result = s.check()
                valid = (result == unsat)
                self.stats['verified' if valid else 'refuted'] += 1
                return {
                    'valid': valid,
                    'premise': premise,
                    'conclusion': conclusion,
                    'reason': 'Valid (Modus Ponens, heuristic)' if valid else 'Invalid (heuristic)',
                    '_mode': 'heuristic_pattern',
                }

            return {'valid': None, 'error': 'Pattern not recognized', '_mode': 'heuristic_failed'}

        except Exception as e:
            return {'valid': None, 'error': str(e), '_mode': 'heuristic_error'}

    def score_debate_argument(self, statement: str) -> dict:
        """Score a debate argument's logical quality using keyword heuristics.

        NOTE: This is the legacy "Honest Caveat" method — it scores 0-100
        based on keyword frequency, not formal logical validity.
        Returns a quality score with reasons and detected fallacies.

        For formal verification, use verify_trld() instead.
        """
        # Check for TRLD format (upgrade path)
        if _trld_detect(statement):
            result = self.verify_trld(statement)
            if result.get('valid') is not None:
                return {
                    'score': 100 if result['valid'] else 20,
                    'reasons': ['TRLD-formatted — deterministic verification applied'],
                    'fallacies': [] if result['valid'] else ['Logical fallacy detected by Z3'],
                    'statement_length': len(statement),
                    '_mode': 'trld_auto',
                    '_verification': result,
                }

        self.mode_usage['heuristic'] += 1

        score = 50  # default
        reasons = []
        fallacies = []

        # Positive indicators
        has_syllogism = bool(re.search(
            r'all\s+\w+\s+are|if\s+.+\s+then|therefore|thus|hence|consequently|follows that',
            statement, re.IGNORECASE))
        has_evidence = bool(re.search(
            r'because|since|as|given that|evidence|research|studies show|data shows|according to',
            statement, re.IGNORECASE))
        has_critical_thinking = bool(re.search(
            r'however|but|yet|on the other hand|alternatively|conversely|nevertheless',
            statement, re.IGNORECASE))
        has_precise_quantifier = bool(re.search(
            r'all|none|every|most|few|some|several|majority|minority',
            statement, re.IGNORECASE))
        has_concrete_example = bool(re.search(
            r'for example|for instance|specifically|such as|e\.g\.',
            statement, re.IGNORECASE))

        if has_syllogism:
            score += 15
            reasons.append('uses structured reasoning')
        if has_evidence:
            score += 15
            reasons.append('cites evidence')
        if has_critical_thinking:
            score += 15
            reasons.append('acknowledges counterarguments')
        if has_precise_quantifier:
            score += 10
            reasons.append('uses precise language')
        if has_concrete_example:
            score += 10
            reasons.append('provides concrete example')

        # Negative indicators (fallacies)
        if re.search(r'straw.?man|ad\s+hominem', statement, re.IGNORECASE):
            score -= 30
            fallacies.append('logical fallacy detected')
        if re.search(r'everyone knows|common sense|obviously|clearly\s+(?!stated)', statement, re.IGNORECASE):
            score -= 15
            fallacies.append('appeal to common belief')
        if re.search(r'always|never|absolutely|certainly|undoubtedly|without a doubt', statement, re.IGNORECASE):
            score -= 10
            fallacies.append('overgeneralization')
        if not has_evidence and not has_syllogism and len(statement) > 50:
            score -= 10
            fallacies.append('unsupported claim')

        score = max(0, min(100, score))
        return {
            'score': score,
            'reasons': reasons,
            'fallacies': fallacies,
            'statement_length': len(statement),
            '_mode': 'heuristic',
        }

    def get_stats(self):
        """Get verification statistics."""
        return {**self.stats, 'mode_usage': self.mode_usage}


def verify_socratic_transcript(transcript: list[dict]) -> list[dict]:
    """Verify every argument in a Socratic debate transcript.

    Uses TRLD mode by default; falls back to heuristic scoring.
    """
    v = LogicVerifier()
    results = []
    for entry in transcript:
        text = entry['text']

        # Prefer TRLD verification
        if _trld_detect(text):
            result = v.verify_trld(text)
            if result.get('valid') is not None:
                result['persona'] = entry.get('persona', 'unknown')
                result['turn'] = entry.get('turn', 0)
                result['score'] = 100 if result['valid'] else 20
                results.append(result)
                continue

        # Fallback to heuristic scoring
        result = v.score_debate_argument(text)
        result['persona'] = entry.get('persona', 'unknown')
        result['turn'] = entry.get('turn', 0)
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    v = LogicVerifier()
    has_z3 = HAS_Z3

    print('═' * 60)
    print('  TRLD — Deterministic Z3 Verification')
    print('═' * 60)

    trld_tests = [
        # Syllogism (Socrates)
        "[PREMISE: All humans are mortal] [PREMISE: Socrates is human] [CONCLUSION: Socrates is mortal]",
        # Invalid version
        "[PREMISE: All humans are mortal] [PREMISE: Socrates is mortal] [CONCLUSION: Socrates is human]",
        # Modus Ponens
        "[IF: It rains] [THEN: The ground gets wet] [PREMISE: It is raining] [CONCLUSION: The ground is wet]",
        # Modus Tollens (if not Q then not P)
        "[IF: It rains] [THEN: The ground gets wet] [PREMISE: The ground is not wet] [CONCLUSION: It is not raining]",
        # Contradiction test
        "[PREMISE: All birds can fly] [PREMISE: Penguins are birds] [PREMISE: Penguins cannot fly]",
    ]

    for i, test in enumerate(trld_tests):
        result = v.verify_trld(test)
        status = 'VALID' if result.get('valid') is True else 'INVALID' if result.get('valid') is False else '?'
        reason = result.get('reason', result.get('error', 'unknown'))
        print(f'\n  Test #{i+1}: [{status}] {reason[:70]}')

    print()
    print('═' * 60)
    print('  Legacy Heuristic (Free-text)')
    print('═' * 60)

    args = [
        "All humans are mortal, and Socrates is human, therefore Socrates is mortal.",
        "I think AI is bad because some people say it is.",
        "Since all observed swans are white, and this bird is a swan, it follows that this bird is white.",
    ]
    for arg in args:
        result = v.score_debate_argument(arg)
        print(f'  Score: {result["score"]:3d} - {arg[:60]}...')
        if result['reasons']:
            print(f'         Reasons: {", ".join(result["reasons"])}')

    print(f'\n  Stats: {v.get_stats()}')
    print(f'\n{"=" * 60}')
    print('  TRLD is now the primary mode. The free-text heuristic')
    print('  (the "Honest Caveat") now auto-detects TRLD and upgrades.')
    print(f'{"=" * 60}')
