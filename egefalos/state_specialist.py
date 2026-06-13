"""
state_specialist.py — Stage 4: The Realm of State & Causality.

The model learns to track how the world changes over time by translating
English narratives into Python code and executing them in the sandbox.

Curriculum:
  1. Inventory tracking: "I have 3 apples. I eat 1. I buy 2. How many?"
  2. Physics state: position, velocity, temperature changes
  3. Multi-step narratives: up to 10 steps with object permanence
  4. Code generation: translate English statements into Python assignments
  5. Assertion verification: the sandbox checks the final state

Mechanism:
  English sentence → Python code (via <CODE_CALL> token)
  Python sandbox executes and verifies state assertions

Graduation: Sandbox confirms 98%+ accuracy on 10-step narratives
"""

import random

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from egefalos.code_sandbox import PythonEnvironment


# ═════════════════════════════════════════════════════════════════════
# STATE NARRATIVE GENERATORS
# ═════════════════════════════════════════════════════════════════════

OBJECTS = ['apples', 'oranges', 'bananas', 'books', 'pens', 'coins',
           'tickets', 'cookies', 'candies', 'marbles', 'pencils',
           'stickers', 'toys', 'cards', 'shells', 'beads', 'buttons']

ACTIONS_ADD = ['buy', 'find', 'get', 'collect', 'receive', 'gather', 'pick up', 'earn']
ACTIONS_REMOVE = ['eat', 'give away', 'lose', 'sell', 'donate', 'drop', 'spend', 'use']
ACTIONS_MULTIPLY = ['double', 'triple', 'multiply by {n}']
ACTIONS_DIVIDE = ['share equally among {n}', 'split into {n} groups']


def generate_narrative(steps: int = 5, rng=None) -> tuple:
    """Generate a multi-step state narrative.
    
    Returns:
        (narrative_text, final_value, python_code)
    """
    if rng is None:
        rng = random.Random()
    
    obj = rng.choice(OBJECTS)
    obj_name = obj  # e.g., "apples"
    var_name = obj.rstrip('s') if obj.endswith('s') else obj  # e.g., "apple"
    
    # Initial state
    init = rng.randint(1, 20)
    value = init
    
    narrative = f'I have {init} {obj_name}.'
    code_lines = [f'{var_name} = {init}']
    
    for step in range(steps):
        action_type = rng.choice(['add', 'remove', 'multiply', 'divide'])
        
        if action_type == 'add':
            delta = rng.randint(1, 15)
            action = rng.choice(ACTIONS_ADD)
            narrative += f' I {action} {delta} {obj_name}.'
            code_lines.append(f'{var_name} += {delta}')
            value += delta
        
        elif action_type == 'remove':
            delta = rng.randint(1, min(value, 10))
            if delta <= 0:
                delta = 1
            action = rng.choice(ACTIONS_REMOVE)
            narrative += f' I {action} {delta} {obj_name}.'
            code_lines.append(f'{var_name} -= {delta}')
            value -= delta
        
        elif action_type == 'multiply':
            factor = rng.choice([2, 3])
            narrative += f' I double my {obj_name}.'
            code_lines.append(f'{var_name} *= {factor}')
            value *= factor
        
        elif action_type == 'divide':
            divisor = rng.choice([2, 3])
            if value % divisor != 0:
                value = (value // divisor) * divisor
            if value <= 0:
                value = rng.randint(4, 20) * divisor
            narrative += f' I share my {obj_name} equally into {divisor} groups.'
            code_lines.append(f'{var_name} //= {divisor}')
            value //= divisor
        
        value = max(0, value)
    
    narrative += f' How many {obj_name} do I have?'
    code_lines.append(f'result = {var_name}')
    code = '\n'.join(code_lines)
    
    return narrative, str(value), code


def generate_physics_narrative(steps: int = 3, rng=None) -> tuple:
    """Generate a physics state narrative (position, speed, temperature)."""
    if rng is None:
        rng = random.Random()
    
    var = rng.choice(['position', 'speed', 'distance', 'height', 'temperature'])
    init = rng.randint(0, 100)
    value = init
    
    narrative = f'The {var} is {init}.'
    code_lines = [f'{var} = {init}']
    
    for _ in range(steps):
        delta = rng.randint(-20, 30)
        direction = 'increases by' if delta >= 0 else 'decreases by'
        narrative += f' It {direction} {abs(delta)}.'
        code_lines.append(f'{var} += {delta}')
        value += delta
        value = max(0, value)
    
    narrative += f' What is the final {var}?'
    code_lines.append(f'result = {var}')
    code = '\n'.join(code_lines)
    
    return narrative, str(value), code


# ═════════════════════════════════════════════════════════════════════
# STATE SELF-PLAY — Translate English → Code → Verify
# ═════════════════════════════════════════════════════════════════════

class StateGame:
    """The State Tracking game.
    
    The model reads an English narrative, generates Python code to track
    the state variables, and the sandbox verifies the result.
    """
    
    def __init__(self, timeout=2):
        self.env = PythonEnvironment(timeout_sec=timeout)
    
    def play(self, model, tokenizer, narrative: str, expected: str,
             code: str = None) -> dict:
        """Play one state-tracking game.
        
        If code is provided, we evaluate that code directly.
        If not, the model must generate the code from the narrative.
        
        Returns:
            {'reward': -1..+1, 'feedback': str, 'code': str}
        """
        if code is None:
            # Model must generate the code from the narrative
            prompt = f'[STATE:{narrative}]'
            generated = model.generate_code(tokenizer, prompt,
                                             max_new_tokens=100,
                                             temperature=0.5, top_k=10)
            
            # Extract code
            if hasattr(tokenizer, 'decode'):
                generated_text = generated
            else:
                generated_text = generated
            
            # Heuristic: extract the code block
            code = self._extract_code(generated_text)
        
        if not code:
            return {'reward': -1.0, 'feedback': 'No code generated', 'code': ''}
        
        # Build test: run the code and check the result
        test = f'assert result == {expected}, f"Got {{result}}, expected {expected}"'
        
        result = self.env.evaluate_move(code, [test])
        
        return {
            'reward': result['reward'],
            'feedback': result['feedback'],
            'code': code,
            'syntax_valid': result['syntax_valid'],
        }
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from model output."""
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            if '=' in stripped and not stripped.startswith('['):
                code_lines.append(stripped)
                in_code = True
            elif in_code and stripped and not stripped.startswith('['):
                code_lines.append(stripped)
        
        return '\n'.join(code_lines) if code_lines else ''


# ═════════════════════════════════════════════════════════════════════
# NARRATIVE GENERATORS MAP
# ═════════════════════════════════════════════════════════════════════

NARRATIVE_GENERATORS = [
    (generate_narrative, {'steps': 3}),
    (generate_narrative, {'steps': 5}),
    (generate_narrative, {'steps': 8}),
    (generate_narrative, {'steps': 10}),
    (generate_physics_narrative, {'steps': 3}),
    (generate_physics_narrative, {'steps': 5}),
]


def generate_state_problem(rng=None):
    """Generate a random state-tracking problem."""
    if rng is None:
        rng = random.Random()
    generator, kwargs = rng.choice(NARRATIVE_GENERATORS)
    narrative, expected, code = generator(**kwargs, rng=rng)
    return narrative, expected, code


def generate_dataset(count: int, seed: int = 42) -> list:
    """Generate a dataset of state-tracking problems."""
    rng = random.Random(seed)
    return [generate_state_problem(rng) for _ in range(count)]


# ═════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════

def train_state_specialist(model, tokenizer,
                            train_steps: int = 2000,
                            batch_size: int = 32,
                            learning_rate: float = 5e-4,
                            device: str = 'cpu') -> dict:
    """Train the State Tracking Specialist."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    game = StateGame(timeout=2)
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    step = 0
    
    while step < train_steps:
        problems = generate_dataset(batch_size, seed=step)
        
        for narrative, expected, code in problems:
            # Play the state game (model generates code from narrative)
            result = game.play(model, tokenizer, narrative, expected)
            reward = result['reward']
            
            if reward >= 0.5:
                correct += 1
            total += 1
            
            # Train on the narrative → code mapping
            prompt = f'[STATE:{narrative}]'
            model_code = result.get('code', '')
            if model_code:
                full_text = f'{prompt} {model_code}'
                ids = tokenizer.encode(full_text, add_special_tokens=True)
                if len(ids) > model.config.max_seq_len:
                    ids = ids[:model.config.max_seq_len]
                padded = ids + [tokenizer.pad_id] * (model.config.max_seq_len - len(ids))
                x = torch.tensor(padded[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(padded[1:], dtype=torch.long, device=device).unsqueeze(0)
                y[y == tokenizer.pad_id] = -100
                
                logits, loss, _ = model(x, y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
        
        step += 1
        
        if step % 200 == 0:
            acc = correct / max(total, 1)
            print(f'  [State] Step {step:>5d}/{train_steps} | '
                  f'loss={total_loss/max(step,1):.4f} | '
                  f'sandbox_acc={acc*100:.1f}%')
    
    model.eval()
    return {
        'stage': 'state',
        'steps': train_steps,
        'final_loss': total_loss / max(train_steps, 1),
        'sandbox_accuracy': correct / max(total, 1) * 100,
    }


# ═════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════

def evaluate_state(model, tokenizer, num_problems: int = 50,
                    device: str = 'cpu') -> float:
    """Evaluate state-tracking accuracy via sandbox verification."""
    game = StateGame(timeout=2)
    problems = generate_dataset(num_problems, seed=999)
    correct = 0
    
    for narrative, expected, code in problems:
        result = game.play(model, tokenizer, narrative, expected)
        if result['reward'] >= 0.5:
            correct += 1
    
    return correct / max(num_problems, 1)


# ═════════════════════════════════════════════════════════════════════
# DEMO
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== Stage 4: State & Causality ===')
    rng = random.Random(42)
    
    for i in range(5):
        narrative, expected, code = generate_state_problem(rng)
        print(f'\n  Narrative: {narrative[:80]}')
        print(f'  Python:    {code}')
        print(f'  Expected:  {expected}')
        
        # Verify with sandbox
        from egefalos.code_sandbox import PythonEnvironment
        env = PythonEnvironment(timeout_sec=2)
        test = f'assert result == {expected}, f"Got {{result}}, expected {expected}"'
        r = env.evaluate_move(code, [test])
        print(f'  Sandbox:   reward={r["reward"]:+.1f}')
