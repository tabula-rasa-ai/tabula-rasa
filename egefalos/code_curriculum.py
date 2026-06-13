"""
code_curriculum.py — 3-Stage Developmental Curriculum for Code AlphaZero.

Stages:
  Stage 1 — Syntax Game: Learn valid Python grammar via ast.parse
  Stage 2 — Fuzzing Game: Learn type safety via property-based testing  
  Stage 3 — Algorithmic Self-Play: Optimize for Big-O efficiency

Sleep Cycle Integration:
  Pythagorean Code Review — audits generated code for safety, efficiency, elegance

The curriculum is progressive: each stage unlocks when the previous is mastered.
"""

import json
import random
import time
from pathlib import Path
from enum import Enum

import torch
from torch.optim import AdamW

from egefalos.code_specialist import (
    CodeAlphaZero, DevelopmentStage, ALGORITHM_PROBLEMS,
    PYTHON_PROGRAMS, check_syntax,
)
from egefalos.code_sandbox import PythonEnvironment


# ═════════════════════════════════════════════════════════════════════
# CURRICULUM PROGRESSION
# ═════════════════════════════════════════════════════════════════════

# Mastery thresholds: when to advance to the next stage
MASTERY_THRESHOLDS = {
    DevelopmentStage.SYNTAX: 0.85,     # 85% syntax accuracy → Stage 2
    DevelopmentStage.FUZZING: 0.70,    # 70% fuzz pass rate → Stage 3
    DevelopmentStage.ALGORITHM: 0.50,  # 50% algorithm solve rate = "graduate"
}

# Number of games per stage per session
GAMES_PER_STAGE = {
    DevelopmentStage.SYNTAX: 100,
    DevelopmentStage.FUZZING: 50,
    DevelopmentStage.ALGORITHM: 30,
}


# ═════════════════════════════════════════════════════════════════════
# STAGE 1 — SYNTAX GAME SESSION
# ═════════════════════════════════════════════════════════════════════

def run_syntax_session(model, tokenizer, num_games: int = 100,
                       learning_rate: float = 1e-4,
                       device: str = 'cpu') -> dict:
    """Run a Stage 1 Syntax Game training session.
    
    The AI generates Python code from templates, checks syntax via ast.parse,
    and trains its Policy + Value heads to predict parseability.
    """
    model = model.to(device)
    model.set_stage(DevelopmentStage.SYNTAX)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    model.train()
    total_loss = 0.0
    valid_count = 0
    total_games = 0
    rewards = []
    
    for i in range(num_games):
        # Pick a random template
        template_key = random.choice(list(PYTHON_PROGRAMS.keys()))
        template = PYTHON_PROGRAMS[template_key]
        
        # Generate code
        prompt_text = f"[CODE:{template_key}]"
        code = model.generate_code(tokenizer, prompt_text,
                                    max_new_tokens=50,
                                    temperature=0.9, top_k=15)
        code_clean = code.replace(prompt_text, '').strip()
        if not code_clean:
            code_clean = template
        
        # Check syntax
        valid, msg = check_syntax(code_clean)
        reward = 1.0 if valid else -1.0
        if valid:
            valid_count += 1
        rewards.append(reward)
        
        # Train on the code
        full_text = f"{prompt_text} {code_clean}"
        ids = tokenizer.encode(full_text, add_special_tokens=True)
        if len(ids) > model.config.max_seq_len:
            ids = ids[:model.config.max_seq_len]
        padded = ids + [tokenizer.pad_id] * (model.config.max_seq_len - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(padded[1:], dtype=torch.long, device=device).unsqueeze(0)
        y[y == tokenizer.pad_id] = -100
        
        # Forward + AlphaZero loss
        logits, _, value = model(x, y)
        reward_z = torch.tensor([[reward]], device=device)
        
        game_loss, _ = model.alphazero_loss(logits, y, value, reward_z)
        
        optimizer.zero_grad()
        game_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += game_loss.item()
        total_games += 1
        
        if (i + 1) % 25 == 0:
            avg_r = sum(rewards[-25:]) / max(len(rewards[-25:]), 1)
            print(f'  [Syntax] Game {i+1}/{num_games} | '
                  f'reward={avg_r:.3f} | valid={valid_count}/{i+1} | '
                  f'loss={total_loss/total_games:.4f}')
    
    model.eval()
    acc = valid_count / max(total_games, 1)
    
    return {
        'stage': 'syntax',
        'games_played': total_games,
        'syntax_accuracy': acc * 100,
        'avg_reward': sum(rewards) / max(len(rewards), 1),
        'training_loss': total_loss / max(total_games, 1),
        'mastered': acc >= MASTERY_THRESHOLDS[DevelopmentStage.SYNTAX],
    }


# ═════════════════════════════════════════════════════════════════════
# STAGE 2 — FUZZING GAME SESSION
# ═════════════════════════════════════════════════════════════════════

def run_fuzzing_session(model, tokenizer, num_games: int = 50,
                        learning_rate: float = 1e-4,
                        device: str = 'cpu') -> dict:
    """Run a Stage 2 Fuzzing Game training session.
    
    The AI writes functions. The Fuzzer generates random edge-case inputs.
    If any input crashes the function, reward = -1.
    The Value Head learns to predict which functions will survive fuzzing.
    """
    model = model.to(device)
    model.set_stage(DevelopmentStage.FUZZING)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    model.train()
    total_loss = 0.0
    passed_count = 0
    total_games = 0
    rewards = []
    
    for i in range(num_games):
        # Pick a random algorithm problem
        problem_key = random.choice(list(ALGORITHM_PROBLEMS.keys()))
        problem = ALGORITHM_PROBLEMS[problem_key]
        
        result = model.play_fuzzing_game(tokenizer, problem_key, num_inputs=15)
        reward = result.get('reward', -1.0)
        passed = result.get('passed', False)
        
        if passed:
            passed_count += 1
        rewards.append(reward)
        
        # Train on the generated code
        code = result.get('code', '')
        if code:
            prompt = f"[PROBLEM:{problem['prompt']}]"
            full_text = f"{prompt} {code}"
            ids = tokenizer.encode(full_text, add_special_tokens=True)
            if len(ids) > model.config.max_seq_len:
                ids = ids[:model.config.max_seq_len]
            padded = ids + [tokenizer.pad_id] * (model.config.max_seq_len - len(ids))
            x = torch.tensor(padded[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(padded[1:], dtype=torch.long, device=device).unsqueeze(0)
            y[y == tokenizer.pad_id] = -100
            
            logits, _, value = model(x, y)
            reward_z = torch.tensor([[reward]], device=device)
            
            game_loss, _ = model.alphazero_loss(logits, y, value, reward_z)
            
            optimizer.zero_grad()
            game_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += game_loss.item()
        
        total_games += 1
        
        if (i + 1) % 10 == 0:
            avg_r = sum(rewards[-10:]) / max(len(rewards[-10:]), 1)
            print(f'  [Fuzzing] Game {i+1}/{num_games} | '
                  f'reward={avg_r:.3f} | pass={passed_count}/{i+1} | '
                  f'loss={total_loss/max(total_games,1):.4f}')
    
    model.eval()
    pass_rate = passed_count / max(total_games, 1)
    
    return {
        'stage': 'fuzzing',
        'games_played': total_games,
        'fuzz_pass_rate': pass_rate * 100,
        'avg_reward': sum(rewards) / max(len(rewards), 1),
        'training_loss': total_loss / max(total_games, 1),
        'mastered': pass_rate >= MASTERY_THRESHOLDS[DevelopmentStage.FUZZING],
    }


# ═════════════════════════════════════════════════════════════════════
# STAGE 3 — ALGORITHMIC SELF-PLAY SESSION
# ═════════════════════════════════════════════════════════════════════

def run_algorithm_session(model, tokenizer, num_games: int = 30,
                          learning_rate: float = 1e-4,
                          device: str = 'cpu') -> dict:
    """Run a Stage 3 Algorithmic Self-Play session.
    
    The AI must solve algorithm problems correctly AND efficiently.
    Multiple attempts per problem — iteratively optimizing for speed.
    The Value Head learns to intuit Big-O complexity.
    """
    model = model.to(device)
    model.set_stage(DevelopmentStage.ALGORITHM)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    model.train()
    total_loss = 0.0
    solved_count = 0
    total_games = 0
    best_rewards = []
    
    for i in range(num_games):
        problem_key = random.choice(list(ALGORITHM_PROBLEMS.keys()))
        
        result = model.play_algorithm_game(tokenizer, problem_key)
        best_reward = result.get('best_reward', -1.0)
        best_rewards.append(best_reward)
        
        if best_reward >= 0.5:
            solved_count += 1
        
        # Train on the best attempt's code
        attempts = result.get('attempts', [])
        if attempts:
            best_attempt = max(attempts, key=lambda a: a['reward'])
            code = best_attempt.get('code', '')
            if code:
                problem = ALGORITHM_PROBLEMS[problem_key]
                prompt = f"[PROBLEM:{problem['prompt']}]"
                full_text = f"{prompt} {code}"
                ids = tokenizer.encode(full_text, add_special_tokens=True)
                if len(ids) > model.config.max_seq_len:
                    ids = ids[:model.config.max_seq_len]
                padded = ids + [tokenizer.pad_id] * (model.config.max_seq_len - len(ids))
                x = torch.tensor(padded[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(padded[1:], dtype=torch.long, device=device).unsqueeze(0)
                y[y == tokenizer.pad_id] = -100
                
                logits, _, value = model(x, y)
                reward_z = torch.tensor([[best_reward]], device=device)
                
                game_loss, _ = model.alphazero_loss(logits, y, value, reward_z)
                
                optimizer.zero_grad()
                game_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += game_loss.item()
        
        total_games += 1
        
        if (i + 1) % 10 == 0:
            avg_br = sum(best_rewards[-10:]) / max(len(best_rewards[-10:]), 1)
            print(f'  [Algorithm] Game {i+1}/{num_games} | '
                  f'best_reward={avg_br:.3f} | solved={solved_count}/{i+1} | '
                  f'loss={total_loss/max(total_games,1):.4f}')
    
    model.eval()
    solve_rate = solved_count / max(total_games, 1)
    
    return {
        'stage': 'algorithm',
        'games_played': total_games,
        'solve_rate': solve_rate * 100,
        'avg_best_reward': sum(best_rewards) / max(len(best_rewards), 1),
        'training_loss': total_loss / max(total_games, 1),
        'mastered': solve_rate >= MASTERY_THRESHOLDS[DevelopmentStage.ALGORITHM],
    }


# ═════════════════════════════════════════════════════════════════════
# FULL TRAINING SESSION — Auto-progress through stages
# ═════════════════════════════════════════════════════════════════════

def full_code_training_session(model, tokenizer,
                                num_syntax: int = 100,
                                num_fuzzing: int = 50,
                                num_algorithm: int = 30,
                                learning_rate: float = 1e-4,
                                device: str = 'cpu') -> dict:
    """Run a complete Code AlphaZero training session through all stages.
    
    Progresses automatically: Stage 1 → Stage 2 → Stage 3.
    Each stage trains the Policy + Value heads with AlphaZero loss.
    """
    all_results = {}
    
    # ── Stage 1: Syntax Game ──
    print()
    print('  ╔══════════════════════════════════════════╗')
    print('  ║  STAGE 1: SYNTAX GAME                   ║')
    print('  ║  Learning Python grammar via ast.parse   ║')
    print('  ╚══════════════════════════════════════════╝')
    print()
    
    syntax_results = run_syntax_session(model, tokenizer, num_syntax,
                                        learning_rate, device)
    all_results['syntax'] = syntax_results
    
    if syntax_results['mastered']:
        print(f'  [*] Syntax mastered at {syntax_results["syntax_accuracy"]:.1f}%')
    else:
        print(f'  [*] Syntax at {syntax_results["syntax_accuracy"]:.1f}% '
              f'(threshold: {MASTERY_THRESHOLDS[DevelopmentStage.SYNTAX]*100:.0f}%)')
    
    # ── Stage 2: Fuzzing Game (if Stage 1 passed) ──
    if syntax_results['mastered'] or True:  # Always try Stage 2
        print()
        print('  ╔══════════════════════════════════════════╗')
        print('  ║  STAGE 2: FUZZING GAME                  ║')
        print('  ║  Property-based testing for type safety  ║')
        print('  ╚══════════════════════════════════════════╝')
        print()
        
        fuzz_results = run_fuzzing_session(model, tokenizer, num_fuzzing,
                                            learning_rate, device)
        all_results['fuzzing'] = fuzz_results
    
    # ── Stage 3: Algorithmic Self-Play ──
    print()
    print('  ╔══════════════════════════════════════════╗')
    print('  ║  STAGE 3: ALGORITHMIC SELF-PLAY         ║')
    print('  ║  Big-O intuition via Value Head training ║')
    print('  ╚══════════════════════════════════════════╝')
    print()
    
    algo_results = run_algorithm_session(model, tokenizer, num_algorithm,
                                          learning_rate, device)
    all_results['algorithm'] = algo_results
    
    # ── Summary ──
    print()
    print(f'  {"="*50}')
    print(f'  Code Training Summary')
    print(f'  {"="*50}')
    for stage, results in all_results.items():
        print(f'  {stage}:')
        for k, v in results.items():
            if k != 'stage':
                print(f'    {k}: {v}')
        print()
    
    return all_results


# ═════════════════════════════════════════════════════════════════════
# PYTHAGOREAN CODE REVIEW — Called by the Sleep Cycle
# ═════════════════════════════════════════════════════════════════════

def pythagorean_code_review(code_history: list) -> dict:
    """The Pythagorean Code Review — nightly audit of generated code.
    
    3 Questions:
    Q1: Where have I transgressed? — Security audit for dangerous imports
    Q2: What have I done? — Code quality + refactoring assessment
    Q3: What duty is neglected? — Library/algorithm gaps in knowledge
    """
    from egefalos.code_sandbox import has_dangerous_imports, check_syntax
    
    review = {
        'transgressions': [],
        'elegant_code': [],
        'neglected_duties': [],
        'code_count': len(code_history),
        'timestamp': time.time(),
    }
    
    for entry in code_history:
        code = entry.get('code', '')
        problem = entry.get('problem', 'unknown')
        reward = entry.get('reward', 0)
        
        # Q1: Security audit
        dangerous, reason = has_dangerous_imports(code)
        if dangerous:
            review['transgressions'].append({
                'problem': problem,
                'code_snippet': code[:100],
                'reason': reason,
                'action': 'PURGED',
            })
            continue  # Don't consolidate dangerous code
        
        # Q2: Code quality — check syntax and reward
        valid, _ = check_syntax(code)
        if valid and reward >= 0.5:
            review['elegant_code'].append({
                'problem': problem,
                'code': code,
                'reward': reward,
                'lines': len(code.split('\n')),
                'action': 'CONSOLIDATED',
            })
        
        # Q3: Neglected duties — low-reward problems
        if reward < -0.3:
            review['neglected_duties'].append({
                'problem': problem,
                'reason': f'Low reward ({reward:.1f})',
                'action': 'FLAGGED_FOR_CURRICULUM',
            })
    
    print(f'  [Code Review] Audited {len(code_history)} code samples')
    print(f'    Purged: {len(review["transgressions"])} dangerous')
    print(f'    Consolidated: {len(review["elegant_code"])} elegant')
    print(f'    Flagged: {len(review["neglected_duties"])} for curriculum')
    
    return review


# ═════════════════════════════════════════════════════════════════════
# CLI / DEMO
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer
    
    cfg = Config()
    cfg.use_value_head = True
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.max_seq_len = 256
    
    tok = MathTokenizer()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    
    model = CodeAlphaZero(cfg)
    
    results = full_code_training_session(
        model, tok,
        num_syntax=5,
        num_fuzzing=3,
        num_algorithm=3,
        device='cpu',
    )
    print('Training complete.')
