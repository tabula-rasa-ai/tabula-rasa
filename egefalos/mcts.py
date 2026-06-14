"""
Monte Carlo Tree Search for Language AlphaZero.

CPU Survival Architecture:
  Full MCTS on every token would take hours on 4-core CPU.
  Instead, we use Entropy-Triggered Micro-MCTS:
    - Only runs when the Policy Head is confused (entropy > threshold)
    - Limited to 16 rollouts (vs 80,000 in Chess AlphaZero)
    - Each rollout simulates 5-10 token completions
    - Value Head + Grammar Rules prune bad branches early

This mirrors how human language works: we don't consciously plan every word,
only the ones that genuinely require deliberation.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

# Memory-aware MCTS uses sys.path-based imports from tabula_rasa
import sys
from pathlib import Path

# Determine project root for tabula_rasa imports
_project_root = str(Path(__file__).resolve().parent.parent / 'src')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ═════════════════════════════════════════════════════════════════════
# MCTS NODE
# ═════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """A node in the MCTS tree representing a partial sentence state.
    
    In the Language Game, each node = a token sequence prefix.
    Children = possible next tokens.
    """
    # Game state
    token_ids: list = field(default_factory=list)      # Full sequence so far
    parent: Optional['MCTSNode'] = None
    
    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior_prob: float = 0.0
    
    # Children dict: token_id -> MCTSNode
    children: dict = field(default_factory=dict)
    
    # Terminal state
    is_terminal: bool = False
    terminal_value: float = 0.0
    
    @property
    def mean_value(self) -> float:
        """Q(s,a): Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def best_child(self, c_puct: float = 1.0) -> tuple:
        """Select the best child using PUCT formula.
        
        UCB-1 style with prior probability:
          Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
        
        Args:
            c_puct: Exploration constant (higher = more exploration)
        
        Returns:
            (token_id, child_node)
        """
        best_score = -float('inf')
        best_pair = None
        
        sqrt_parent = math.sqrt(self.visit_count)
        
        for token_id, child in self.children.items():
            # Q-value (exploitation)
            q = child.mean_value
            
            # U-value (exploration): PUCT heuristic
            u = c_puct * child.prior_prob * sqrt_parent / (1 + child.visit_count)
            
            score = q + u
            
            if score > best_score:
                best_score = score
                best_pair = (token_id, child)
        
        if best_pair is None and self.children:
            # Fallback: choose first child
            token_id = next(iter(self.children))
            best_pair = (token_id, self.children[token_id])
        
        return best_pair
    
    def add_child(self, token_id: int, prior_prob: float) -> 'MCTSNode':
        """Add a child node for a token."""
        child = MCTSNode(
            token_ids=self.token_ids + [token_id],
            parent=self,
            prior_prob=prior_prob,
        )
        self.children[token_id] = child
        return child
    
    def update(self, value: float):
        """Backup: update node statistics after a rollout."""
        self.visit_count += 1
        self.total_value += value


# ═════════════════════════════════════════════════════════════════════
# MICRO-MCTS ROLLOUT
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout(model, tokenizer, input_ids: torch.Tensor,
            concept, max_rollout_tokens: int = 8,
            grammar_rules: list = None) -> float:
    """Simulate a random completion of the sentence and score it.
    
    The rollout is the "playout" phase of MCTS — quickly finish the
    sentence using the policy network and score it with the Value Head
    and grammar rules.
    
    Args:
        model: LanguageAlphaZero model
        tokenizer: Tokenizer for encoding
        input_ids: Current input tensor (1, seq_len)
        concept: Original concept being expressed
        max_rollout_tokens: Max tokens to simulate
        grammar_rules: Optional list of grammar constraint functions
    
    Returns:
        float: Score for this rollout (-1 to +1)
    """
    device = input_ids.device
    current_ids = input_ids.clone()
    
    eos_id = tokenizer.eos_id
    
    for step in range(max_rollout_tokens):
        if current_ids.size(1) > model.config.max_seq_len:
            current_ids = current_ids[:, -model.config.max_seq_len:]
        
        logits, _, value = model.forward(current_ids)
        next_logits = logits[0, -1, :]
        
        # Sample from policy (temperature=1.0 for exploration during rollout)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        
        current_ids = torch.cat([current_ids, next_id.unsqueeze(0)], dim=1)
        
        if next_id.item() == eos_id:
            break
    
    # Score the completed rollout
    score = 0.0
    
    # 1. Value Head score (how "intuitive" does the sentence feel?)
    if value is not None:
        score += value.item() * 0.5  # Weight: 0.5
    
    # 2. Length penalty (prefer concise sentences)
    rollout_len = current_ids.size(1) - input_ids.size(1)
    optimal_len = 8  # Expected tokens for a simple sentence
    length_penalty = -abs(rollout_len - optimal_len) * 0.02
    score += length_penalty
    
    # 3. Grammar rule penalties
    if grammar_rules:
        try:
            sentence = tokenizer.decode(current_ids[0].tolist())
            for rule_fn in grammar_rules:
                penalty = rule_fn(sentence)
                score += penalty * 0.3
        except Exception:
            pass
    
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, score))


# ═════════════════════════════════════════════════════════════════════
# MCTS SELECT
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def select(node: MCTSNode) -> MCTSNode:
    """Tree policy: descend the tree following PUCT until a leaf is reached.
    
    Returns:
        The leaf node to expand.
    """
    current = node
    while not current.is_leaf:
        _, best = current.best_child()
        current = best
    return current


# ═════════════════════════════════════════════════════════════════════
# MCTS EXPAND
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def expand(model, tokenizer, node: MCTSNode,
           input_ids: torch.Tensor, top_k: int = 5,
           grammar_rules: list = None) -> list:
    """Expand a leaf node by adding its top-k token children.
    
    Uses the Policy Head to get candidate tokens and their probabilities.
    Filters out tokens that violate grammar rules.
    
    Args:
        model: LanguageAlphaZero model
        tokenizer: Tokenizer
        node: Leaf node to expand
        input_ids: Input tensor at this node's state
        top_k: Number of candidate tokens to consider
        grammar_rules: Optional list of grammar constraint functions
    
    Returns:
        List of (token_id, child) pairs added
    """
    device = input_ids.device
    
    # Get policy probabilities
    logits, _, _ = model.forward(input_ids)
    next_logits = logits[0, -1, :]
    
    # Top-k filtering
    probs = F.softmax(next_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
    
    added = []
    
    for i in range(len(top_indices)):
        token_id = top_indices[i].item()
        prior = top_probs[i].item()
        
        # Grammar rule filtering
        if grammar_rules:
            candidate_ids = node.token_ids + [token_id]
            try:
                candidate_text = tokenizer.decode(candidate_ids)
                valid = True
                for rule_fn in grammar_rules:
                    if rule_fn(candidate_text, check='validity') < -0.5:
                        valid = False
                        break
                if not valid:
                    prior *= 0.1  # Penalize, don't completely prune
            except Exception:
                pass
        
        child = node.add_child(token_id, prior)
        added.append((token_id, child))
    
    return added


# ═════════════════════════════════════════════════════════════════════
# MCTS BACKUP
# ═════════════════════════════════════════════════════════════════════

def backup(node: MCTSNode, value: float):
    """Backup: propagate rollout value up the tree."""
    current = node
    while current is not None:
        current.update(value)
        current = current.parent


# ═════════════════════════════════════════════════════════════════════
# MICRO-MCTS — THE FULL ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def micro_mcts_search(model, tokenizer, input_ids: torch.Tensor,
                      concept, temperature: float = 0.8, top_k: int = 5,
                      simulations: int = 16, max_rollout_tokens: int = 8,
                      grammar_rules: list = None,
                      c_puct: float = 1.0) -> dict:
    """Run a Micro-MCTS search from the current state.
    
    This is the CPU-survival trick: limited to 16 rollouts, each exploring
    5-10 tokens ahead. The Value Head + Grammar Rules provide signal even
    from shallow searches.
    
    Args:
        model: LanguageAlphaZero model
        tokenizer: Tokenizer
        input_ids: Current input tensor (1, seq_len)
        concept: The concept being expressed (for rollout scoring)
        temperature: Temperature for final token selection
        top_k: Number of candidate tokens per node
        simulations: Number of MCTS simulations (default: 16)
        max_rollout_tokens: Max tokens per rollout
        grammar_rules: Optional grammar constraint functions
        c_puct: Exploration constant
    
    Returns:
        dict with:
          - 'best_token': int, the selected token
          - 'best_prob': float, its probability
          - 'visit_counts': dict of token->visit_count
          - 'tree_depth': int, depth of tree searched
    """
    device = input_ids.device
    seq_len = input_ids.size(1)
    
    # Get base probabilities from policy head
    logits, _, _ = model.forward(input_ids)
    base_logits = logits[0, -1, :]
    base_probs = F.softmax(base_logits, dim=-1)
    
    # Build token sequence for root
    root_tokens = input_ids[0].tolist()
    
    # Create root node
    root = MCTSNode(token_ids=root_tokens, prior_prob=1.0)
    
    # ─── Run simulations ───
    for sim in range(simulations):
        # 1. SELECT: traverse tree with PUCT
        leaf = select(root)
        
        # 2. EXPAND: add children to leaf
        if not leaf.is_terminal:
            leaf_input = input_ids.clone()
            if len(leaf.token_ids) > seq_len:
                leaf_tensor = torch.tensor([leaf.token_ids], device=device)
                leaf_input = leaf_tensor
            
            expand(model, tokenizer, leaf, leaf_input,
                   top_k=top_k, grammar_rules=grammar_rules)
        
        # 3. ROLLOUT: simulate from the leaf
        if leaf.children:
            # Pick a child randomly (weighted by visit count for exploitation,
            # or uniformly for exploration during search)
            child_token = random.choice(list(leaf.children.keys()))
            rollout_input = input_ids.clone()
            child_ids = leaf.token_ids + [child_token]
            rollout_input = torch.tensor([child_ids], device=device)
            
            rollout_value = rollout(
                model, tokenizer, rollout_input, concept,
                max_rollout_tokens=max_rollout_tokens,
                grammar_rules=grammar_rules,
            )
        else:
            # Leaf has no children (terminal or empty) — use its value
            rollout_value = leaf.terminal_value if leaf.is_terminal else 0.0
        
        # 4. BACKUP: propagate value up the tree
        backup(leaf, rollout_value)
    
    # ─── Select best token from root children ───
    # Pick the child with the highest visit count (AlphaZero convention)
    best_token = None
    best_prob = 0.0
    visit_counts = {}
    
    max_visits = 0
    for token_id, child in root.children.items():
        visit_counts[token_id] = child.visit_count
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_token = token_id
            best_prob = base_probs[token_id].item()
    
    # Fallback: if no children explored, use best from policy head
    if best_token is None:
        best_token = base_logits.argmax(dim=-1).item()
        best_prob = 1.0
    
    return {
        'best_token': best_token,
        'best_prob': best_prob,
        'visit_counts': visit_counts,
        'tree_depth': max(len(root.token_ids), 1),
    }


# ═════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# ═════════════════════════════════════════════════════════════════════

def create_mcts_fn(model, tokenizer, grammar_rules: list = None,
                   simulations: int = 16, max_rollout_tokens: int = 8,
                   top_k: int = 5):
    """Create a closure that wraps micro_mcts_search for easy passing.
    
    This is what gets passed to LanguageAlphaZero.speaker_generate().
    
    Usage:
        mcts_fn = create_mcts_fn(model, tokenizer, grammar_rules)
        sentence = model.speaker_generate(tok, concept, mcts_fn=mcts_fn)
    """
    def _mcts_fn(model_inner, tok_inner, input_ids, concept,
                 temperature=0.8, top_k_inner=top_k):
        return micro_mcts_search(
            model_inner, tok_inner, input_ids, concept,
            temperature=temperature, top_k=top_k_inner,
            simulations=simulations,
            max_rollout_tokens=max_rollout_tokens,
            grammar_rules=grammar_rules,
        )
    return _mcts_fn


# ═════════════════════════════════════════════════════════════════════
# MICRO-MCTS CLASS — Object-oriented wrapper around micro_mcts_search
# ═════════════════════════════════════════════════════════════════════

class MicroMCTS:
    """Object-oriented MCTS that wraps micro_mcts_search.
    
    Provides a class-based interface that can be subclassed for
    memory-aware or custom MCTS variants.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_simulations: int = 16,
        top_k: int = 5,
        max_rollout_tokens: int = 8,
        c_puct: float = 1.0,
        grammar_rules: list = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_simulations = n_simulations
        self.top_k = top_k
        self.max_rollout_tokens = max_rollout_tokens
        self.c_puct = c_puct
        self.grammar_rules = grammar_rules or []
    
    def search(self, input_ids: torch.Tensor, concept,
               temperature: float = 0.8) -> dict:
        """Run MCTS search from the current state.
        
        Delegates to micro_mcts_search for the actual search logic.
        
        Args:
            input_ids: Input tensor (1, seq_len)
            concept: The concept being expressed
            temperature: Temperature for token selection
        
        Returns:
            dict with 'best_token', 'best_prob', 'visit_counts', 'tree_depth'
        """
        return micro_mcts_search(
            self.model, self.tokenizer, input_ids, concept,
            temperature=temperature, top_k=self.top_k,
            simulations=self.n_simulations,
            max_rollout_tokens=self.max_rollout_tokens,
            grammar_rules=self.grammar_rules,
            c_puct=self.c_puct,
        )
    
    def node_confidence(self, state: str) -> float:
        """Query memory confidence for a state.
        
        Base implementation returns neutral confidence.
        Subclasses override this for memory-aware confidence.
        
        Args:
            state: String representation of the current state.
        
        Returns:
            float 0.0-1.0 where 1.0 = very confident.
        """
        return 0.5  # Neutral when no memory integration


# ═════════════════════════════════════════════════════════════════════
# MEMORY-AWARE MCTS — Queries tiered memory for confidence-based
# simulation allocation and uses math_parser for fast pruning.
# ═════════════════════════════════════════════════════════════════════

class MemoryAwareMCTS(MicroMCTS):
    """MCTS variant that queries tiered memory for confidence-based branching.
    
    Key differences from MicroMCTS:
      - Queries search_memories() to assess confidence in current state.
      - Low-confidence branches get more simulations; high-confidence get fewer.
      - Uses math_parser.verify_equation() for fast deterministic pruning
        during rollouts (prunes syntactically invalid expressions).
    
    The total simulation budget N is preserved, but distribution is shifted
    toward branches the model is less sure about.
    
    Args:
        model: LanguageAlphaZero model
        tokenizer: Tokenizer
        n_simulations: Total simulation budget (default: 16)
        memory: Hippocampus memory module (optional). When None, behaves
                like regular MicroMCTS.
        top_k: Number of candidates per node
        max_rollout_tokens: Max tokens per rollout
        c_puct: Exploration constant
        grammar_rules: Optional grammar constraint functions
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        n_simulations: int = 16,
        memory=None,
        top_k: int = 5,
        max_rollout_tokens: int = 8,
        c_puct: float = 1.0,
        grammar_rules: list = None,
    ):
        super().__init__(
            model, tokenizer,
            n_simulations=n_simulations,
            top_k=top_k,
            max_rollout_tokens=max_rollout_tokens,
            c_puct=c_puct,
            grammar_rules=grammar_rules,
        )
        self.memory = memory
        
        # Math parser integration — lazy import
        self._verify_equation = None
    
    # ── Confidence via memory query ───────────────────────────────────
    
    def node_confidence(self, state: str) -> float:
        """Query tiered memory for confidence in current state.
        
        Searches across ALL tiers (active, working, longterm) for
        memories matching the current state. Returns higher confidence
        when memory has low prediction_error entries for similar states.
        
        Returns:
            float 0.0-1.0 where 1.0 = very confident.
            Defaults to 0.5 when memory is not available.
        """
        if self.memory is None:
            return 0.5  # Neutral when no memory
        
        from egefalos.hippocampus import search_memories
        
        # Search all tiers for similar states
        results = search_memories(state, tier=None, limit=5)
        
        if not results:
            return 0.3  # Unknown state → low confidence
        
        # Average prediction_error across matched memories
        # (Lower error = higher confidence)
        total_errors = sum(r['prediction_error'] for r in results)
        avg_error = total_errors / len(results)
        
        # Map avg_error (typically 0.0-5.0+) to 0.0-1.0 confidence
        # Lower error → higher confidence
        # Error of 0 → confidence 1.0; error ≥ 5 → confidence 0.0
        confidence = max(0.0, min(1.0, 1.0 - (avg_error / 5.0)))
        
        # Boost if result is found across multiple tiers
        tiers_seen = len(set(r['tier'] for r in results))
        if tiers_seen >= 2:
            confidence = min(1.0, confidence + 0.1 * tiers_seen)
        
        return confidence
    
    # ── Dynamic simulation allocation ─────────────────────────────────
    
    def _allocate_simulations(
        self, children: list, total_budget: int
    ) -> list[int]:
        """Distribute simulations proportionally to uncertainty.
        
        For each child, query memory confidence. High-confidence branches
        receive fewer simulations; low-confidence branches receive more.
        
        Args:
            children: List of MCTSNodes (child nodes at a given level).
            total_budget: Total number of simulations to allocate.
        
        Returns:
            List of ints of length len(children), summing to total_budget,
            with simulation counts per child.
        """
        if not children:
            return []
        
        if len(children) == 1:
            return [total_budget]
        
        # Compute confidence for each child
        confidences = []
        for child in children:
            # Use token sequence as the state string
            state = ' '.join(str(t) for t in child.token_ids[-8:])  # last 8 tokens
            confidences.append(self.node_confidence(state))
        
        # Convert confidence to "uncertainty" (inverse)
        # Branches with low confidence should get MORE simulations
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        uncertainties = [max(epsilon, 1.0 - c) for c in confidences]
        
        # Normalize to get proportions
        total_uncertainty = sum(uncertainties)
        proportions = [u / total_uncertainty for u in uncertainties]
        
        # Distribute budget proportionally
        allocations = [max(1, int(p * total_budget)) for p in proportions]
        
        # Adjust to match total_budget exactly (rounding correction)
        diff = total_budget - sum(allocations)
        if diff > 0:
            # Give extra simulations to the most uncertain child
            most_uncertain_idx = uncertainties.index(max(uncertainties))
            allocations[most_uncertain_idx] += diff
        elif diff < 0:
            # Remove from the most confident child
            most_confident_idx = confidences.index(max(confidences))
            allocations[most_confident_idx] = max(
                1, allocations[most_confident_idx] + diff
            )
        
        return allocations
    
    # ── Math parser fast pruning ──────────────────────────────────────
    
    def _get_verify_equation(self):
        """Lazy-import verify_equation from math_parser."""
        if self._verify_equation is None:
            from tabula_rasa.math_parser import verify_equation
            self._verify_equation = verify_equation
        return self._verify_equation
    
    def fast_prune(self, partial_expression: str) -> bool:
        """Check if a partially-generated expression is already invalid.
        
        Uses math_parser.verify_equation() to detect syntax errors or
        mathematically invalid partial expressions. If the parser cannot
        even parse the expression (syntax error), the branch is pruned.
        
        Args:
            partial_expression: A string containing an equation or partial
                                math expression, e.g. '12+34=46' or '12+'.
        
        Returns:
            True if the branch should be PRUNED (invalid).
            False if the branch is VALID (or cannot be determined).
        """
        verify = self._get_verify_equation()
        
        try:
            result = verify(partial_expression)
            # Prune if the equation contains a '=' but is invalid
            # This catches: syntax errors, division by zero, wrong results
            if '=' in partial_expression:
                if result.get('error'):
                    return True  # Syntax/parse error → prune
                return not result.get('valid', True)  # Wrong result → prune
            return False  # No '=' sign yet → still valid partial expression
        except Exception:
            return True  # Unexpected error during verification → prune
    
    # ── Memory-aware search ───────────────────────────────────────────
    
    @torch.no_grad()
    def search(self, input_ids: torch.Tensor, concept,
               temperature: float = 0.8) -> dict:
        """Run memory-aware MCTS search.
        
        The same interface as MicroMCTS.search(), but with:
          1. Dynamic simulation allocation based on memory confidence.
          2. Math parser fast pruning during rollouts.
        
        Args:
            input_ids: Input tensor (1, seq_len)
            concept: The concept being expressed
            temperature: Temperature for token selection
        
        Returns:
            dict with 'best_token', 'best_prob', 'visit_counts',
            'tree_depth', and 'confidence_map'
        """
        device = input_ids.device
        seq_len = input_ids.size(1)
        
        # Get base probabilities from policy head
        logits, _, _ = self.model.forward(input_ids)
        base_logits = logits[0, -1, :]
        base_probs = F.softmax(base_logits, dim=-1)
        
        # Build token sequence for root
        root_tokens = input_ids[0].tolist()
        
        # Create root node
        root = MCTSNode(token_ids=root_tokens, prior_prob=1.0)
        
        # Track confidence per node for diagnostics
        confidence_map = {}
        
        # ─── Run simulations with dynamic allocation ───
        for sim in range(self.n_simulations):
            # 1. SELECT: traverse tree with PUCT
            leaf = select(root)
            
            # 2. EXPAND: add children to leaf
            if not leaf.is_terminal:
                leaf_input = input_ids.clone()
                if len(leaf.token_ids) > seq_len:
                    leaf_tensor = torch.tensor([leaf.token_ids], device=device)
                    leaf_input = leaf_tensor
                
                expand(self.model, self.tokenizer, leaf, leaf_input,
                       top_k=self.top_k, grammar_rules=self.grammar_rules)
            
            # 3. ROLLOUT: simulate from the leaf
            if leaf.children:
                # Get list of children for allocation
                child_items = list(leaf.children.items())
                
                # Dynamic allocation: if we have memory, distribute
                # simulations based on confidence
                if self.memory is not None and len(child_items) > 1:
                    children_list = [c for _, c in child_items]
                    # Track confidence for diagnostics
                    for tok_id, child in child_items:
                        state = ' '.join(str(t) for t in child.token_ids[-8:])
                        conf = self.node_confidence(state)
                        confidence_map.setdefault(tok_id, []).append(conf)
                    
                    # Pick the child with LOWEST confidence (explore uncertain)
                    uncertainties = []
                    for _, child in child_items:
                        state = ' '.join(str(t) for t in child.token_ids[-8:])
                        uncertainties.append(1.0 - self.node_confidence(state))
                    most_uncertain_idx = uncertainties.index(max(uncertainties))
                    child_token, chosen_child = child_items[most_uncertain_idx]
                else:
                    # Default: pick a child randomly
                    child_token = random.choice(list(leaf.children.keys()))
                
                rollout_input = input_ids.clone()
                child_ids = leaf.token_ids + [child_token]
                rollout_input = torch.tensor([child_ids], device=device)
                
                # Math parser fast pruning: check partial expression
                if self._is_math_state(concept):
                    partial_text = self.tokenizer.decode(child_ids)
                    if self.fast_prune(partial_text):
                        # Prune this branch — assign a bad value
                        rollout_value = -1.0
                    else:
                        rollout_value = rollout(
                            self.model, self.tokenizer, rollout_input, concept,
                            max_rollout_tokens=self.max_rollout_tokens,
                            grammar_rules=self.grammar_rules,
                        )
                else:
                    rollout_value = rollout(
                        self.model, self.tokenizer, rollout_input, concept,
                        max_rollout_tokens=self.max_rollout_tokens,
                        grammar_rules=self.grammar_rules,
                    )
            else:
                rollout_value = leaf.terminal_value if leaf.is_terminal else 0.0
            
            # 4. BACKUP: propagate value up the tree
            backup(leaf, rollout_value)
        
        # ─── Select best token from root children ───
        best_token = None
        best_prob = 0.0
        visit_counts = {}
        
        max_visits = 0
        for token_id, child in root.children.items():
            visit_counts[token_id] = child.visit_count
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_token = token_id
                best_prob = base_probs[token_id].item()
        
        # Fallback: if no children explored, use best from policy head
        if best_token is None:
            best_token = base_logits.argmax(dim=-1).item()
            best_prob = 1.0
        
        return {
            'best_token': best_token,
            'best_prob': best_prob,
            'visit_counts': visit_counts,
            'tree_depth': max(len(root.token_ids), 1),
            'confidence_map': confidence_map,
        }
    
    def _is_math_state(self, concept) -> bool:
        """Heuristic: check if the current problem involves math.
        
        Math-aware pruning only applies when the concept looks like
        a math problem (contains digits, operators, or math keywords).
        """
        if concept is None:
            return False
        concept_str = str(concept).lower()
        math_keywords = ['math', 'equation', 'calculate', 'compute', 'solve',
                         '+', '-', '*', '/', '=', 'digit', 'number', 'add',
                         'subtract', 'multiply', 'divide', 'sum', 'count']
        return any(kw in concept_str for kw in math_keywords)
    
    def __repr__(self) -> str:
        return (
            f"MemoryAwareMCTS(n_simulations={self.n_simulations}, "
            f"memory={'set' if self.memory is not None else 'None'}, "
            f"top_k={self.top_k})"
        )


# ═════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER — Create MemoryAwareMCTS with integrated memory
# ═════════════════════════════════════════════════════════════════════

def create_memory_aware_mcts_fn(
    model, tokenizer,
    memory=None,
    n_simulations: int = 16,
    max_rollout_tokens: int = 8,
    top_k: int = 5,
    grammar_rules: list = None,
):
    """Create a closure wrapping MemoryAwareMCTS.search() for easy passing.
    
    Usage:
        mcts_fn = create_memory_aware_mcts_fn(model, tokenizer, memory)
        sentence = model.speaker_generate(tok, concept, mcts_fn=mcts_fn)
    """
    mcts = MemoryAwareMCTS(
        model=model,
        tokenizer=tokenizer,
        memory=memory,
        n_simulations=n_simulations,
        max_rollout_tokens=max_rollout_tokens,
        top_k=top_k,
        grammar_rules=grammar_rules,
    )
    
    def _mcts_fn(model_inner, tok_inner, input_ids, concept,
                 temperature=0.8, top_k_inner=top_k):
        return mcts.search(
            input_ids=input_ids,
            concept=concept,
            temperature=temperature,
        )
    
    return _mcts_fn


# ═════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer
    from language_az import LanguageAlphaZero, ConceptNode, concept_to_string
    
    cfg = Config()
    cfg.use_value_head = True
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.n_heads = 4
    cfg.max_seq_len = 128
    
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    
    model = LanguageAlphaZero(cfg)
    
    # Test MCTS on a simple concept
    concept = ConceptNode(entity='apple', property='red', action='fall')
    prompt_text = f"[CONCEPT:{concept_to_string(concept)}]"
    input_ids = tok.encode(prompt_text, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])
    
    print('Running Micro-MCTS on concept: apple:red:fall')
    result = micro_mcts_search(model, tok, input_tensor, concept,
                                simulations=16, top_k=5)
    print(f'Best token: {result["best_token"]} (prob: {result["best_prob"]:.3f})')
    print(f'Visit counts: {result["visit_counts"]}')
    print(f'Tree depth: {result["tree_depth"]}')
    
    # Test the closure
    mcts_fn = create_mcts_fn(model, tok, simulations=8)
    result2 = mcts_fn(model, tok, input_tensor, concept)
    print(f'\nClosure result: token={result2["best_token"]}')
    
    # Test MicroMCTS class
    print('\n--- Testing MicroMCTS class ---')
    mmcts = MicroMCTS(model, tok, n_simulations=8)
    result3 = mmcts.search(input_tensor, concept)
    print(f'MicroMCTS result: token={result3["best_token"]}')
    
    # Test MemoryAwareMCTS (without memory = behaves like standard MCTS)
    print('\n--- Testing MemoryAwareMCTS (no memory) ---')
    mamcts = MemoryAwareMCTS(model, tok, n_simulations=8)
    result4 = mamcts.search(input_tensor, concept)
    print(f'MemoryAwareMCTS result: token={result4["best_token"]}')
    print(f'Confidence map entries: {len(result4.get("confidence_map", {}))}')
    
    # Test node_confidence with no memory (should return 0.5)
    conf = mamcts.node_confidence('apple:red:fall')
    print(f'node_confidence(apple:red:fall) = {conf:.2f} (expected 0.5)')
