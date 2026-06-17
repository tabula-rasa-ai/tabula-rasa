"""
LanguageAlphaZero — Self-Play Language Acquisition via MCTS + Policy/Value Networks.

Extends the transformer architecture with an AlphaZero-style dual head:
  1. Policy Head (Speaker): predicts next token distribution (lm_head)
  2. Value Head (Intuition): outputs scalar -1..+1 predicting semantic reconstruction success

Architecture:
  - Base transformer (shared with MathTransformer) + value_head
  - Entropy-triggered Micro-MCTS: only runs search when policy head is confused
  - Self-play autoencoding game: Speaker generates → Listener reconstructs

Integration:
  - egefalos/sleep_cycle.py: nightly AlphaZero Gymnasium session
  - egefalos/pythagoras.py: semantic audit of reconstruction failures
"""

import math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.config import Config


# ═════════════════════════════════════════════════════════════════════
# CONCEPT GRAPH — Language-agnostic state representation
# ═════════════════════════════════════════════════════════════════════
# The "board state" in the Language Game. Analogous to a chess position
# in AlphaZero — it's what both Speaker and Listener reason about.

class ConceptNode:
    """A single concept in the semantic graph.

    Attributes:
        entity: The subject (e.g. 'apple', 'Socrates')
        property: Attributes (e.g. 'red', 'mortal', 'round')
        action: Verb/action (e.g. 'fall', 'eat', 'exists')
        relation: Relational connector to other nodes (e.g. 'cause', 'on', 'inside')
        children: Sub-nodes for hierarchical concepts
    """
    def __init__(self, entity="", property="", action="", relation="", children=None):
        self.entity = entity
        self.property = property
        self.action = action
        self.relation = relation
        self.children = children or []

    def to_dict(self):
        return {
            'entity': self.entity,
            'property': self.property,
            'action': self.action,
            'relation': self.relation,
            'children': [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            entity=d.get('entity', ''),
            property=d.get('property', ''),
            action=d.get('action', ''),
            relation=d.get('relation', ''),
            children=[cls.from_dict(c) for c in d.get('children', [])],
        )

    def __repr__(self):
        parts = []
        if self.entity: parts.append(f"Entity:{self.entity}")
        if self.property: parts.append(f"Prop:{self.property}")
        if self.action: parts.append(f"Action:{self.action}")
        if self.relation: parts.append(f"Rel:{self.relation}")
        for c in self.children:
            parts.append(f"->{c}")
        return f"Concept({', '.join(parts)})"

    def __eq__(self, other):
        if not isinstance(other, ConceptNode):
            return False
        return (self.entity == other.entity and
                self.property == other.property and
                self.action == other.action and
                self.relation == other.relation and
                self.children == other.children)

    def __hash__(self):
        return hash((self.entity, self.property, self.action, self.relation,
                     tuple(self.children)))


def concept_to_string(concept: ConceptNode) -> str:
    """Convert a concept graph to a structured token-string for encoding.

    Uses a deterministic serialization format:
      [ENT:entity][PROP:property][ACT:action][REL:relation]
      or compound: [ENT:apple][PROP:red][ACT:fall]
    """
    parts = []
    if concept.entity:
        parts.append(f"[ENT:{concept.entity}]")
    if concept.property:
        parts.append(f"[PROP:{concept.property}]")
    if concept.action:
        parts.append(f"[ACT:{concept.action}]")
    if concept.relation:
        parts.append(f"[REL:{concept.relation}]")
    for child in concept.children:
        parts.append(concept_to_string(child))
    return ''.join(parts)


def string_to_concept(text: str) -> ConceptNode:
    """Parse a structured token-string back into a ConceptNode.

    Inverse of concept_to_string.
    """
    import re
    root = ConceptNode()
    # Find all [TAG:value] patterns
    patterns = re.findall(r'\[(\w+):([^\]]+)\]', text)
    for tag, value in patterns:
        tag = tag.upper()
        if tag == 'ENT':
            root.entity = value.strip()
        elif tag == 'PROP':
            root.property = value.strip()
        elif tag == 'ACT':
            root.action = value.strip()
        elif tag == 'REL':
            root.relation = value.strip()
    return root


def concept_graph_equal(a: ConceptNode, b: ConceptNode) -> bool:
    """Check if two concept graphs are structurally identical.

    This is the reward criterion for the Semantic Reconstruction Game.
    """
    return a == b


# ═════════════════════════════════════════════════════════════════════
# CONCEPT VOCABULARY — Curriculum of concepts for self-play
# ═════════════════════════════════════════════════════════════════════
# These are the "training positions" for the Language Game, analogous
# to the opening book in Chess AlphaZero.

BASE_CONCEPTS = [
    # ── Physics / Spatial ──
    ConceptNode(entity='apple', property='red', action='fall'),
    ConceptNode(entity='ball', property='round', action='roll'),
    ConceptNode(entity='sun', property='hot', action='shine'),
    ConceptNode(entity='water', property='liquid', action='flow'),
    ConceptNode(entity='moon', property='bright', action='orbit'),

    # ── Biology ──
    ConceptNode(entity='tree', property='tall', action='grow'),
    ConceptNode(entity='bird', property='small', action='fly'),
    ConceptNode(entity='fish', property='slimy', action='swim'),
    ConceptNode(entity='flower', property='colorful', action='bloom'),

    # ── People / Action ──
    ConceptNode(entity='man', property='strong', action='run'),
    ConceptNode(entity='child', property='curious', action='learn'),
    ConceptNode(entity='teacher', property='wise', action='teach'),
    ConceptNode(entity='artist', property='creative', action='paint'),

    # ── Philosophy (from the logic verifier domain) ──
    ConceptNode(entity='Socrates', property='human', action='think'),
    ConceptNode(entity='truth', property='absolute', action='exist'),
    ConceptNode(entity='virtue', property='good', action='guide'),

    # ── Relations (compound concepts) ──
    ConceptNode(entity='Socrates', children=[
        ConceptNode(relation='is', entity='human'),
        ConceptNode(relation='is', property='mortal'),
    ]),
    ConceptNode(entity='apple', children=[
        ConceptNode(relation='on', entity='tree'),
        ConceptNode(property='ripe', action='fall'),
    ]),
]

COMPOUND_CONCEPTS = [
    ConceptNode(entity='thinker', property='rational', action='philosophize',
                children=[ConceptNode(relation='about', entity='truth')]),
    ConceptNode(entity='scientist', action='observe',
                children=[ConceptNode(relation='through', entity='telescope')]),
    ConceptNode(entity='student', property='eager', action='question',
                children=[ConceptNode(relation='under', entity='teacher')]),
]


# ═════════════════════════════════════════════════════════════════════
# GENERATE TARGET SENTENCES — Concept → Target Language Sentence
# ═════════════════════════════════════════════════════════════════════
# Templates for mapping concept graphs to target language sentences.
# In a full system these would come from ebooks; for self-play we use
# these templates as the "ground truth" the model must learn to match.

LANGUAGE_TEMPLATES = {
    'en': {
        'simple': {
            'apple:red:fall': 'The red apple falls.',
            'ball:round:roll': 'The round ball rolls.',
            'sun:hot:shine': 'The hot sun shines.',
            'water:liquid:flow': 'The liquid water flows.',
            'moon:bright:orbit': 'The bright moon orbits.',
            'tree:tall:grow': 'The tall tree grows.',
            'bird:small:fly': 'The small bird flies.',
            'fish:slimy:swim': 'The slimy fish swims.',
            'flower:colorful:bloom': 'The colorful flower blooms.',
            'man:strong:run': 'The strong man runs.',
            'child:curious:learn': 'The curious child learns.',
            'teacher:wise:teach': 'The wise teacher teaches.',
            'artist:creative:paint': 'The creative artist paints.',
            'Socrates:human:think': 'Socrates is a human who thinks.',
            'truth:absolute:exist': 'Absolute truth exists.',
            'virtue:good:guide': 'Good virtue guides.',
        },
        'compound': {
            'Socrates:is:human': 'Socrates is human.',
            'Socrates:is:mortal': 'Socrates is mortal.',
            'apple:on:tree': 'The apple is on the tree.',
            'apple:ripe:fall': 'The ripe apple falls.',
            'thinker:rational:philosophize:about:truth': 'The rational thinker philosophizes about truth.',
            'scientist:observe:through:telescope': 'The scientist observes through a telescope.',
            'student:eager:question:under:teacher': 'The eager student questions under the teacher.',
        },
    },
    'fr': {
        'simple': {
            'apple:red:fall': 'La pomme rouge tombe.',
            'ball:round:roll': 'La balle ronde roule.',
            'sun:hot:shine': 'Le soleil chaud brille.',
            'water:liquid:flow': 'Leau liquide coule.',
            'moon:bright:orbit': 'La lune brillante orbite.',
            'tree:tall:grow': 'Le grand arbre pousse.',
            'bird:small:fly': 'Le petit oiseau vole.',
            'man:strong:run': 'Lhomme fort court.',
            'child:curious:learn': 'Lenfant curieux apprend.',
            'teacher:wise:teach': 'Le professeur sage enseigne.',
        },
        'compound': {
            'Socrates:is:human': 'Socrate est humain.',
            'Socrates:is:mortal': 'Socrate est mortel.',
        },
    },
}


def concept_to_sentence_key(concept: ConceptNode) -> str:
    """Generate a lookup key for a concept graph.

    E.g., Concept(entity='apple', property='red', action='fall')
          → 'apple:red:fall'
    Concept(entity='Socrates', children=[Concept(relation='is', entity='human')])
          → 'Socrates:is:human'
    """
    parts = []
    if concept.entity:
        parts.append(concept.entity)
    if concept.property:
        parts.append(concept.property)
    if concept.action:
        parts.append(concept.action)
    if concept.relation:
        parts.append(concept.relation)
    # Flatten children into key
    for child in concept.children:
        if child.relation:
            parts.append(child.relation)
        if child.entity:
            parts.append(child.entity)
        if child.property:
            parts.append(child.property)
        if child.action:
            parts.append(child.action)
    return ':'.join(parts)


def generate_training_concepts(num_concepts: int = 32, seed: int = None) -> list[ConceptNode]:
    """Generate a batch of random concepts for self-play training.

    Creates concepts by combining entities, properties, and actions
    in random permutations to cover the combinatorial space.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    entities = ['apple', 'ball', 'sun', 'water', 'tree', 'bird', 'fish',
                'flower', 'man', 'child', 'teacher', 'artist', 'thinker',
                'scientist', 'student', 'stone', 'star', 'river', 'wind', 'fire']
    properties = ['red', 'round', 'hot', 'liquid', 'tall', 'small', 'slimy',
                  'colorful', 'strong', 'curious', 'wise', 'creative', 'rational',
                  'bright', 'heavy', 'light', 'fast', 'cold', 'warm', 'deep']
    actions = ['fall', 'roll', 'shine', 'flow', 'grow', 'fly', 'swim', 'bloom',
               'run', 'learn', 'teach', 'paint', 'think', 'exist', 'guide',
               'move', 'rise', 'sing', 'dance', 'speak']
    relations = ['on', 'under', 'inside', 'above', 'beside', 'near',
                 'through', 'about', 'with', 'for', 'against', 'toward']

    concepts = []
    for _ in range(num_concepts):
        style = rng.choice(['simple', 'compound', 'relational'])
        if style == 'simple':
            concepts.append(ConceptNode(
                entity=rng.choice(entities),
                property=rng.choice(properties),
                action=rng.choice(actions),
            ))
        elif style == 'compound':
            c = ConceptNode(
                entity=rng.choice(entities),
                children=[
                    ConceptNode(relation=rng.choice(relations), entity=rng.choice(entities)),
                ]
            )
            if rng.random() > 0.5:
                c.action = rng.choice(actions)
            concepts.append(c)
        else:
            concepts.append(ConceptNode(
                entity=rng.choice(entities),
                property=rng.choice(properties),
                action=rng.choice(actions),
                relation=rng.choice(relations),
            ))
    return concepts


# ═════════════════════════════════════════════════════════════════════
# LANGUAGE TRANSFORMER WITH VALUE HEAD
# ═════════════════════════════════════════════════════════════════════

class LanguageAlphaZero(MathTransformer):
    """Extends MathTransformer with AlphaZero-style self-play heads.

    Inherits the base transformer architecture (token_embedding, layers, norm)
    and adds:
      - value_head: outputs -1..+1 intuition score
      - policy_head: shared with lm_head (token prediction)
      - language_game_state: tracks self-play game state

    The forward pass returns (logits, loss, value) — 3-tuple compatible
    with the existing training pipeline.
    """

    def __init__(self, config):
        super().__init__(config)

        # The value_head is inherited from MathTransformer — already defined
        # as self.value_head when config.use_value_head = True.
        # Here we just ensure the Value Head is active for language tasks.

        # Self-play state tracking (reset per game)
        self.game_state = {
            'phase': 'idle',           # idle | speaking | listening | evaluating
            'source_concept': None,    # The concept to communicate
            'generated_sentence': '',  # Speaker's output
            'reconstructed': None,     # Listener's reconstruction
            'reward': 0.0,             # Game outcome
        }

    def reset_game(self):
        """Reset the self-play game state."""
        self.game_state = {
            'phase': 'idle',
            'source_concept': None,
            'generated_sentence': '',
            'reconstructed': None,
            'reward': 0.0,
        }

    @torch.no_grad()
    def speaker_generate(self, tokenizer, concept: ConceptNode,
                         max_new_tokens: int = 20,
                         temperature: float = 0.8, top_k: int = 10,
                         entropy_threshold: float = 0.5,
                         mcts_fn=None) -> str:
        """The Speaker: generate a sentence from a concept.

        Uses entropy-triggered Micro-MCTS: if the policy head is confident,
        samples greedily. If confused (high entropy), runs MCTS to search
        the grammatical space.

        Args:
            tokenizer: MathTokenizer-like encoder
            concept: The concept to express
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature for confident regions
            top_k: Top-k sampling
            entropy_threshold: Entropy threshold for triggering MCTS
            mcts_fn: Callable for Micro-MCTS (from mcts module)

        Returns:
            Generated sentence string
        """
        self.eval()
        device = next(self.parameters()).device

        # Encode concept as prompt
        prompt_text = f"[CONCEPT:{concept_to_string(concept)}]"
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)

        generated_tokens = []
        game_state = {'phase': 'speaking', 'source_concept': concept}

        for step in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            logits, _, value = self.forward(input_ids)
            next_logits = logits[0, -1, :]

            # Compute entropy of policy head
            probs = F.softmax(next_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-10))).item()
            top_prob = probs.max().item()

            selected_id = None

            # CPU Survival Trick: only search if confused
            if entropy > entropy_threshold and mcts_fn is not None:
                # Run Micro-MCTS (16 rollouts) to explore alternatives
                mcts_result = mcts_fn(
                    self, tokenizer, input_ids, concept,
                    temperature=temperature, top_k=top_k,
                    simulations=16,
                )
                selected_id = mcts_result['best_token']
            else:
                # Fast sampling / greedy
                if temperature < 0.01:
                    selected_id = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k > 0:
                        top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        next_logits[next_logits < top_vals[-1]] = float('-inf')
                    probs = F.softmax(next_logits, dim=-1)
                    selected_id = torch.multinomial(probs, 1)

            generated_tokens.append(selected_id.item())
            input_ids = torch.cat([input_ids, selected_id.unsqueeze(0)], dim=1)

            if selected_id.item() == tokenizer.eos_id:
                break

        # Decode
        full_ids = input_ids[0].tolist()
        sentence = tokenizer.decode(full_ids)

        # Store in game state
        self.game_state['generated_sentence'] = sentence
        self.game_state['source_concept'] = concept
        self.game_state['phase'] = 'speaking_done'

        return sentence

    @torch.no_grad()
    def listener_reconstruct(self, tokenizer, sentence: str) -> ConceptNode:
        """The Listener: reconstruct a concept graph from a sentence.

        Encodes the sentence and decodes it back to [ENT:][PROP:][ACT:]
        format, then parses into a ConceptNode.

        Returns:
            Reconstructed ConceptNode (may differ from original)
        """
        self.eval()
        device = next(self.parameters()).device

        # Encode sentence as prompt for reconstruction
        prompt_text = f"[SENTENCE:{sentence}][RECONSTRUCT:"
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)

        reconstruction_tokens = []

        for step in range(40):  # Max reconstruction tokens
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            logits, _, value = self.forward(input_ids)
            next_logits = logits[0, -1, :]
            next_id = next_logits.argmax(dim=-1, keepdim=True)

            reconstruction_tokens.append(next_id.item())
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

            # Stop on EOS or closing bracket
            if next_id.item() == tokenizer.eos_id:
                break
            if step > 5:
                # Check for ] which ends the reconstruction
                last_token = tokenizer.decode([next_id.item()]).strip()
                if last_token == ']':
                    break

        # Decode reconstruction
        full_ids = input_ids[0].tolist()
        recon_text = tokenizer.decode(full_ids)

        # Parse concept from reconstruction
        # Look for [ENT:X][PROP:X][ACT:X][REL:X] patterns
        recon_concept = string_to_concept(recon_text)

        # Fallback: try to extract from raw text
        if not recon_concept.entity and not recon_concept.property and not recon_concept.action:
            # Try heuristic extraction from the decoded text
            recon_concept = self._heuristic_extract(recon_text)

        self.game_state['reconstructed'] = recon_concept
        self.game_state['phase'] = 'listening_done'

        return recon_concept

    def _heuristic_extract(self, text: str) -> ConceptNode:
        """Fallback heuristic extraction of concept from raw text."""
        import re
        c = ConceptNode()

        # Find [TAG:value] patterns anywhere in the string
        patterns = re.findall(r'\[(ENT|PROP|ACT|REL):([^\]]+)\]', text)
        for tag, value in patterns:
            tag = tag.upper()
            if tag == 'ENT' and not c.entity:
                c.entity = value.strip()
            elif tag == 'PROP' and not c.property:
                c.property = value.strip()
            elif tag == 'ACT' and not c.action:
                c.action = value.strip()
            elif tag == 'REL' and not c.relation:
                c.relation = value.strip()

        return c

    def compute_reward(self, source: ConceptNode, reconstructed: ConceptNode) -> float:
        """Compute the game reward.

        +1.0: Perfect reconstruction (Mutual Intelligibility)
        -1.0: Completely wrong concept
        -0.5: Partial match (some fields correct)
        +0.5: Close match (same entity, different property/action)

        The reward is the AlphaZero z-target for the Value Head.
        """
        if source is None or reconstructed is None:
            return -1.0

        # Perfect match
        if concept_graph_equal(source, reconstructed):
            return 1.0

        # Partial scoring
        score = 0.0

        # Entity match (most important)
        if source.entity and reconstructed.entity:
            if source.entity.lower() == reconstructed.entity.lower():
                score += 0.5
            elif source.entity[:3] in reconstructed.entity or reconstructed.entity[:3] in source.entity:
                score += 0.2  # Partial name overlap

        # Property match
        if source.property and reconstructed.property:
            if source.property.lower() == reconstructed.property.lower():
                score += 0.3
            elif source.property[:2] in reconstructed.property:
                score += 0.1

        # Action match
        if source.action and reconstructed.action:
            if source.action.lower() == reconstructed.action.lower():
                score += 0.3
            elif source.action[:2] in reconstructed.action:
                score += 0.1

        # Relation match
        if source.relation and reconstructed.relation:
            if source.relation.lower() == reconstructed.relation.lower():
                score += 0.2

        # Penalty for hallucinated entities
        if not source.entity and reconstructed.entity:
            score -= 0.3
        if not source.action and reconstructed.action:
            score -= 0.2

        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, score))

    def play_semantic_game(self, tokenizer, concept: ConceptNode,
                           mcts_fn=None) -> dict:
        """Play one round of the Semantic Reconstruction Game.

        Returns:
            dict with 'reward', 'sentence', 'reconstructed', 'source'
        """
        self.reset_game()
        self.game_state['source_concept'] = concept
        self.game_state['phase'] = 'speaking'

        # 1. Speaker generates sentence from concept
        sentence = self.speaker_generate(tokenizer, concept, mcts_fn=mcts_fn)
        self.game_state['generated_sentence'] = sentence
        self.game_state['phase'] = 'listening'

        # 2. Listener reconstructs concept from sentence
        reconstructed = self.listener_reconstruct(tokenizer, sentence)

        # 3. Compute reward
        reward = self.compute_reward(concept, reconstructed)
        self.game_state['reward'] = reward
        self.game_state['phase'] = 'evaluating'

        return {
            'reward': reward,
            'sentence': sentence,
            'reconstructed': reconstructed,
            'source': concept,
            'value_prediction': None,  # Will be filled if value head active
        }

    def alphazero_loss(self, policy_logits, targets, value_pred, reward_z,
                       config=None):
        """Compute AlphaZero-style dual loss for language self-play.

        policy_loss: Cross-entropy on tokens (standard LM)
        value_loss: MSE between value prediction and actual game reward
        total_loss: policy_loss + alpha * value_loss

        Args:
            policy_logits: From lm_head (batch, seq, vocab)
            targets: Target token indices (batch, seq)
            value_pred: Value head output (batch, 1)
            reward_z: Actual game outcome (-1..+1)
            config: Config object with alphazero_loss_weight

        Returns:
            total_loss, (policy_loss, value_loss)
        """
        if config is None:
            config = self.config

        # Policy loss: next-token prediction
        policy_loss = F.cross_entropy(
            policy_logits.view(-1, policy_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            label_smoothing=getattr(config, 'label_smoothing', 0.0),
        )

        # Value loss: MSE between intuition and actual reward
        value_loss = torch.tensor(0.0, device=policy_logits.device)
        if value_pred is not None and reward_z is not None:
            vp = value_pred.squeeze(-1)
            if vp.dim() != reward_z.dim():
                reward_z = reward_z.view_as(vp)
            value_loss = F.mse_loss(vp, reward_z)

        alpha = getattr(config, 'alphazero_loss_weight', 0.5)
        total_loss = policy_loss + alpha * value_loss

        return total_loss, (policy_loss, value_loss)


# ═════════════════════════════════════════════════════════════════════
# SELF-PLAY TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════

def language_self_play_session(model, tokenizer, num_games: int = 200,
                               concepts: list = None, mcts_fn=None,
                               learning_rate: float = 1e-4,
                               device: str = 'cpu') -> dict:
    """Run a full self-play training session for the Language Game.

    1. Generate N random concepts
    2. For each, play Speaker→Listener game
    3. Compute reward
    4. Train policy + value heads via AlphaZero loss

    Returns:
        dict with 'games_played', 'avg_reward', 'training_loss', 'accuracy'
    """
    from torch.optim import AdamW

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    if concepts is None:
        concepts = generate_training_concepts(num_games, seed=42)

    model.train()
    total_loss = 0.0
    total_games = 0
    rewards = []
    perfect_games = 0

    for i, concept in enumerate(concepts[:num_games]):
        # ── Play the game ──
        game_result = model.play_semantic_game(tokenizer, concept, mcts_fn)
        reward = game_result['reward']
        rewards.append(reward)

        if reward >= 1.0:
            perfect_games += 1

        # ── Prepare training data ──
        # Train the model to both speak (concept→sentence) and listen (sentence→concept)
        # For simplicity, we train on the concept→sentence mapping with the value head

        # Build input: concept → sentence
        source_text = f"[CONCEPT:{concept_to_string(concept)}]"
        target_text = game_result['sentence']
        train_text = f"{source_text}" if model.game_state['phase'] == 'idle' else target_text

        full_text = f"{source_text} {target_text}"
        ids = tokenizer.encode(full_text, add_special_tokens=True)

        if len(ids) > 64:
            ids = ids[:64]
        padded = ids + [tokenizer.pad_id] * (64 - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(padded[1:], dtype=torch.long, device=device).unsqueeze(0)

        # Mask PAD in targets
        y[y == tokenizer.pad_id] = -100

        # ── Forward pass ──
        logits, _, value = model(x, y)

        # ── Compute AlphaZero loss ──
        reward_z = torch.tensor([[reward]], device=device)
        game_loss, _ = model.alphazero_loss(logits, y, value, reward_z)

        optimizer.zero_grad()
        game_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += game_loss.item()
        total_games += 1

        if (i + 1) % 50 == 0:
            avg_r = sum(rewards[-50:]) / max(len(rewards[-50:]), 1)
            print(f'  [LanguageAZ] Game {i+1}/{num_games} | '
                  f'avg reward={avg_r:.3f} | loss={total_loss/total_games:.4f}')

    model.eval()
    avg_reward = sum(rewards) / max(len(rewards), 1)

    return {
        'games_played': total_games,
        'avg_reward': avg_reward,
        'perfect_rate': perfect_games / max(total_games, 1) * 100,
        'training_loss': total_loss / max(total_games, 1),
        'rewards': rewards,
    }


# ═════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Testing LanguageAlphaZero...')

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
    print(f'Model params: {count_parameters(model):,}')
    print(f'Value head params: {sum(p.numel() for p in model.value_head.parameters())}')

    # Test concept encoding/decoding
    concept = ConceptNode(entity='apple', property='red', action='fall')
    encoded = concept_to_string(concept)
    decoded = string_to_concept(encoded)
    print(f'Concept: {concept}')
    print(f'Encoded: {encoded}')
    print(f'Decoded: {decoded}')
    print(f'Perfect round-trip: {concept_graph_equal(concept, decoded)}')

    # Test game
    result = model.play_semantic_game(tok, concept)
    print(f'Game result: reward={result["reward"]}')
    print(f'Sentence: {result["sentence"][:80]}')
    print(f'Reconstructed: {result.get("reconstructed", "None")}')

    # Test self-play
    print(f'\nRunning mini self-play session (10 games)...')
    stats = language_self_play_session(model, tok, num_games=10, device='cpu')
    print(f'Self-play stats:')
    for k, v in stats.items():
        if k != 'rewards':
            print(f'  {k}: {v}')
