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
