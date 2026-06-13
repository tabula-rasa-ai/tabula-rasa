"""Egefalos (εγκέφαλος) — The Brain Module.

Greek for "brain." Contains all cognitive architecture files:
hippocampus, neocortex, sleep cycle, Socratic engine, memory,
and the Tabula Rasa AI consciousness.

|All files in this package import from tabula_rasa package
for core components (model.py, config.py, tokenizer.py).

Exposed modules:
  language_az:   LanguageAlphaZero (Policy/Value heads, self-play)
  mcts:          Entropy-triggered Micro-MCTS for CPU-survival
  grammar_engine: CFG-based grammar rules + Z3 logic integration
  semantic_game: Semantic Reconstruction Game (Speaker/Listener arena)
  code_sandbox:   Python sandboxed execution environment (ast + subprocess)
  code_specialist: CodeAlphaZero (3-stage self-play programming)
  code_curriculum: Developmental curriculum + Pythagorean Code Review
  hippocampus:   Fast short-term memory (surprise-based storage)
  neocortex:     Slow consolidation via EWC + experience replay
  sleep_cycle:   Nightly self-improvement daemon
  pythagoras:    Nightly examination of conscience
  logic_verifier: Z3-based logical verification
  memory:        Persistent correction memory
  socratic_stage1/2/3: Socratic dialectical engine
  graduation_daemon: 6-stage cognitive syllabus orchestrator
  pattern_specialist: Stage 2: string patterns, palindromes, sequences
  taxonomy_specialist: Stage 3: Is-A logic, set theory, Z3 taxonomy
  state_specialist:   Stage 4: English-to-code state tracking via sandbox
  grammar_specialist: Stage 5: AST-like grammar game, CFG evaluation
  router:             Neural Router: dispatch prompts to correct specialist
  piaget:        Piaget-inspired developmental stages
  tabula_rasa:   Core AI consciousness
"""

import sys, os
# Add parent directory to path so imports work
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Expose new AlphaZero language modules
from .language_az import (
    LanguageAlphaZero,
    ConceptNode,
    concept_to_string,
    string_to_concept,
    concept_graph_equal,
    generate_training_concepts,
    language_self_play_session,
)
from .mcts import (
    micro_mcts_search,
    create_mcts_fn,
)
from .grammar_engine import (
    grammar_score,
    create_grammar_rules,
    check_subject_verb_agreement,
    check_sentence_structure,
    score_long_distance_structure,
)
from .semantic_game import (
    SemanticGameRunner,
    alphazero_gymnasium_session,
)

# Code AlphaZero modules
from .code_sandbox import (
    PythonEnvironment,
    check_syntax,
    has_dangerous_imports,
)
from .code_specialist import (
    CodeAlphaZero,
    DevelopmentStage,
    ALGORITHM_PROBLEMS,
    PYTHON_PROGRAMS,
)
from .code_curriculum import (
    full_code_training_session,
    pythagorean_code_review,
    run_syntax_session,
    run_fuzzing_session,
    run_algorithm_session,
)

# 6-Stage Cognitive Syllabus modules
from .graduation_daemon import (
    CognitiveStage,
    STAGE_NAMES,
    STAGE_DIRS,
    daemon_cycle,
    run_graduation_check,
    spawn_stage,
    freeze_weights,
    evaluate_held_out_test,
    load_graduation_state,
    save_graduation_state,
)
from .pattern_specialist import (
    generate_pattern_problem,
    train_pattern_specialist,
    evaluate_patterns,
)
from .taxonomy_specialist import (
    generate_taxonomy_problem,
    train_taxonomy_specialist,
    evaluate_taxonomy,
)
from .state_specialist import (
    StateGame,
    generate_state_problem,
    train_state_specialist,
    evaluate_state,
)
from .grammar_specialist import (
    generate_grammar_problem,
    train_grammar_specialist,
    evaluate_grammar,
)
from .router import (
    Router,
    route,
    route_and_answer,
)
