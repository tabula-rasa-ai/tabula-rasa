"""
Semantic Reconstruction Game — The "Chess Board" for Language AlphaZero.

In this game, the AI plays against itself:
  SPEAKER: Given a concept graph, generates a sentence in the target language.
  LISTENER: Reads the sentence, reconstructs the original concept graph.
  REWARD: +1 if concept matches, -1 if wrong, penalty for grammar violations.

This is the core of language acquisition by self-play — the AI learns
to communicate concepts from its own internal representation without
a human teacher.

The game generates synthetic training data for the AlphaZero self-play loop.
"""

import torch
import random
from pathlib import Path

from egefalos.language_az import (
    LanguageAlphaZero, ConceptNode, concept_to_string, string_to_concept,
    concept_graph_equal, generate_training_concepts, concept_to_sentence_key,
    LANGUAGE_TEMPLATES, BASE_CONCEPTS, COMPOUND_CONCEPTS,
    language_self_play_session,
)


# ═════════════════════════════════════════════════════════════════════
# GAME CONFIG
# ═════════════════════════════════════════════════════════════════════

GAME_CONFIG = {
    'concept_difficulty': {
        'easy': {'source': BASE_CONCEPTS[:8], 'num_games': 50},
        'medium': {'source': BASE_CONCEPTS, 'num_games': 200},
        'hard': {'source': BASE_CONCEPTS + COMPOUND_CONCEPTS, 'num_games': 500},
    },
    'reward_weights': {
        'perfect_match': 1.0,
        'entity_match': 0.5,
        'property_match': 0.3,
        'action_match': 0.3,
        'grammar_penalty': -0.2,
        'hallucination_penalty': -0.5,
    },
}


# ═════════════════════════════════════════════════════════════════════
# GAME RUNNER
# ═════════════════════════════════════════════════════════════════════

class SemanticGameRunner:
    """Orchestrates the Semantic Reconstruction Game.
    
    Manages:
      - Concept selection and curriculum difficulty
      - Speaker/Listener interaction through the model
      - Reward computation
      - Training data generation for the Neocortex
      - Game statistics tracking
    """
    
    def __init__(self, model: LanguageAlphaZero, tokenizer,
                 mcts_fn=None, grammar_rules=None):
        self.model = model
        self.tokenizer = tokenizer
        self.mcts_fn = mcts_fn
        self.grammar_rules = grammar_rules or []
        self.stats = {
            'games_played': 0,
            'perfect_games': 0,
            'total_reward': 0.0,
            'avg_grammar_score': 0.0,
        }
    
    def get_concept_curriculum(self, level: str = 'medium') -> list[ConceptNode]:
        """Get concepts for a training session at a given difficulty."""
        config = GAME_CONFIG['concept_difficulty'].get(level, GAME_CONFIG['concept_difficulty']['medium'])
        return config['source']
    
    def play_game(self, concept: ConceptNode) -> dict:
        """Play one game of the Semantic Reconstruction Game."""
        game_result = self.model.play_semantic_game(
            self.tokenizer, concept, mcts_fn=self.mcts_fn
        )
        
        # Apply grammar penalty to reward
        if self.grammar_rules:
            sentence = game_result.get('sentence', '')
            for rule_fn in self.grammar_rules:
                grammar_penalty = rule_fn(sentence) * GAME_CONFIG['reward_weights']['grammar_penalty']
                game_result['reward'] += grammar_penalty
        
        # Clamp reward to [-1, 1]
        game_result['reward'] = max(-1.0, min(1.0, game_result['reward']))
        
        # Update stats
        self.stats['games_played'] += 1
        self.stats['total_reward'] += game_result['reward']
        if game_result['reward'] >= 1.0:
            self.stats['perfect_games'] += 1
        
        return game_result
    
    def run_session(self, num_games: int = 100, difficulty: str = 'medium') -> dict:
        """Run a full play session with curriculum concepts.
        
        Returns:
            dict with 'results', 'stats', 'training_data'
        """
        curriculum = self.get_concept_curriculum(difficulty)
        if not curriculum:
            curriculum = generate_training_concepts(num_games)
        
        results = []
        for i in range(num_games):
            concept = random.choice(curriculum)
            result = self.play_game(concept)
            results.append(result)
        
        avg_reward = (self.stats['total_reward'] / 
                       max(self.stats['games_played'], 1))
        perfect_rate = (self.stats['perfect_games'] / 
                         max(num_games, 1)) * 100
        
        # Generate training data from high-reward games
        training_data = self._extract_training_data(results, threshold=-0.5)
        
        session_stats = {
            'games_played': num_games,
            'avg_reward': avg_reward,
            'perfect_rate': perfect_rate,
            'training_examples': len(training_data),
            'difficulty': difficulty,
        }
        
        return {
            'results': results,
            'stats': session_stats,
            'training_data': training_data,
        }
    
    def _extract_training_data(self, results: list, threshold: float = -0.5) -> list:
        """Extract high-quality experiences for consolidation.
        
        Games with reward > threshold become training examples
        for the Neocortex sleep consolidation.
        """
        training = []
        for r in results:
            if r['reward'] >= threshold:
                concept = r.get('source')
                sentence = r.get('sentence', '')
                
                # Build training text: concept → sentence
                if concept:
                    input_text = concept_to_string(concept)
                    training.append({
                        'input': input_text,
                        'output': sentence,
                        'reward': r['reward'],
                        'perfect': r['reward'] >= 1.0,
                    })
        return training
    
    def get_stats(self) -> dict:
        """Get current game statistics."""
        avg_reward = (self.stats['total_reward'] / 
                       max(self.stats['games_played'], 1))
        perfect_rate = (self.stats['perfect_games'] / 
                         max(self.stats['games_played'], 1)) * 100
        return {
            'games_played': self.stats['games_played'],
            'perfect_games': self.stats['perfect_games'],
            'avg_reward': avg_reward,
            'perfect_rate': perfect_rate,
        }


# ═════════════════════════════════════════════════════════════════════
# ALPHAZERO GYMNASIUM — Nightly Self-Play Session
# ═════════════════════════════════════════════════════════════════════
# Called by the Sleep Cycle during Pythagorean nightly review.

def alphazero_gymnasium_session(model, tokenizer,
                                 num_games: int = 200,
                                 difficulty: str = 'medium',
                                 learning_rate: float = 1e-4,
                                 mcts_fn=None,
                                 grammar_rules=None,
                                 device: str = 'cpu') -> dict:
    """Run a complete AlphaZero self-play training session.
    
    This is the "AlphaZero Gymnasium" that runs during the Sleep Cycle.
    
    Flow:
      1. Generate concepts from curriculum
      2. Play Semantic Reconstruction Games (self-play)
      3. Train Policy + Value heads via AlphaZero loss
      4. Return training statistics
    
    Args:
        model: LanguageAlphaZero model
        tokenizer: Tokenizer
        num_games: Number of games to play
        difficulty: 'easy', 'medium', or 'hard'
        learning_rate: Learning rate for training
        mcts_fn: MCTS function (optional)
        grammar_rules: List of grammar rule checkers
        device: 'cpu' or 'cuda'
    
    Returns:
        dict with training statistics
    """
    from torch.optim import AdamW
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Get or generate concepts
    runner = SemanticGameRunner(model, tokenizer, mcts_fn, grammar_rules)
    curriculum = runner.get_concept_curriculum(difficulty)
    
    if not curriculum:
        curriculum = generate_training_concepts(num_games)
    
    model.train()
    total_loss = 0.0
    games_played = 0
    rewards = []
    perfect_games = 0
    grammar_scores = []
    
    for i in range(num_games):
        concept = random.choice(curriculum)
        
        # ── Play game ──
        result = model.play_semantic_game(tokenizer, concept, mcts_fn)
        reward = result['reward']
        
        # Apply grammar penalty
        if grammar_rules:
            sentence = result.get('sentence', '')
            gs = sum(rule_fn(sentence) for rule_fn in grammar_rules) / max(len(grammar_rules), 1)
            grammar_scores.append(gs)
            reward += gs * GAME_CONFIG['reward_weights']['grammar_penalty']
        
        reward = max(-1.0, min(1.0, reward))
        rewards.append(reward)
        if reward >= 1.0:
            perfect_games += 1
        
        # ── Prepare training data ──
        source_text = f"[CONCEPT:{concept_to_string(concept)}]"
        target_text = result['sentence']
        full_text = f"{source_text} {target_text}"
        
        ids = tokenizer.encode(full_text, add_special_tokens=True)
        if len(ids) > model.config.max_seq_len:
            ids = ids[:model.config.max_seq_len]
        padded = ids + [tokenizer.pad_id] * (model.config.max_seq_len - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(padded[1:], dtype=torch.long, device=device).unsqueeze(0)
        y[y == tokenizer.pad_id] = -100
        
        # ── Forward + AlphaZero loss ──
        logits, _, value = model(x, y)
        reward_z = torch.tensor([[reward]], device=device)
        
        game_loss, (pl, vl) = model.alphazero_loss(logits, y, value, reward_z)
        
        optimizer.zero_grad()
        game_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += game_loss.item()
        games_played += 1
        
        if (i + 1) % 50 == 0:
            avg_r = sum(rewards[-50:]) / max(len(rewards[-50:]), 1)
            avg_l = total_loss / max(games_played, 1)
            print(f'  [AlphaZero Gym] Game {i+1}/{num_games} | '
                  f'reward={avg_r:.3f} | loss={avg_l:.4f} | '
                  f'perfect={perfect_games}/{games_played}')
    
    model.eval()
    avg_reward = sum(rewards) / max(len(rewards), 1)
    
    return {
        'games_played': games_played,
        'avg_reward': avg_reward,
        'perfect_rate': perfect_games / max(games_played, 1) * 100,
        'training_loss': total_loss / max(games_played, 1),
        'avg_grammar_score': sum(grammar_scores) / max(len(grammar_scores), 1) if grammar_scores else 0,
        'total_reward': sum(rewards),
    }


# ═════════════════════════════════════════════════════════════════════
# DEMO
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Testing Semantic Reconstruction Game...')
    
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer
    
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
    
    # Test the game runner
    runner = SemanticGameRunner(model, tok)
    concept = ConceptNode(entity='apple', property='red', action='fall')
    result = runner.play_game(concept)
    print(f'Game result:')
    print(f'  Concept: {concept}')
    print(f'  Sentence: {result["sentence"][:80]}')
    print(f'  Reconstructed: {result.get("reconstructed", "None")}')
    print(f'  Reward: {result["reward"]:.3f}')
    
    print(f'\nStats: {runner.get_stats()}')
    
    # Test full session
    print(f'\nRunning mini gymnasium session (5 games)...')
    from grammar_engine import create_grammar_rules
    rules = create_grammar_rules()
    results = alphazero_gymnasium_session(
        model, tok, num_games=5, difficulty='easy',
        grammar_rules=rules, device='cpu'
    )
    for k, v in results.items():
        print(f'  {k}: {v}')
