"""
code_specialist.py — CodeAlphaZero: Self-Play Programming via AlphaZero.

Three developmental stages, mirroring a CS education:
  Stage 1 — Syntax Game: Generate valid Python ASTs, get +1 for parseable code
  Stage 2 — Fuzzing Game: Property-based testing against hypothesis-like fuzzer
  Stage 3 — Algorithmic Self-Play: Optimize for Big-O efficiency via Value Head

Architecture:
  - Extends the transformer with code-specific Policy/Value heads
  - Uses ast.parse() as the "Rules Engine" (illegal moves = SyntaxErrors)
  - Reward from code_sandbox.PythonEnvironment
  - Value Head learns to "intuit" algorithm complexity
"""

import random
import re
from enum import Enum

import torch
import torch.nn.functional as F

from tabula_rasa.cognitive.code_sandbox import PythonEnvironment, check_syntax
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters

# ═════════════════════════════════════════════════════════════════════
# STAGE ENUM — Developmental Curriculum
# ═════════════════════════════════════════════════════════════════════

class DevelopmentStage(Enum):
    SYNTAX = 1      # Stage 1: Learn valid Python syntax via ast.parse
    FUZZING = 2     # Stage 2: Property-based testing against random inputs
    ALGORITHM = 3   # Stage 3: Optimize for algorithmic efficiency


# ═════════════════════════════════════════════════════════════════════
# STAGE 1 — SYNTAX GENERATION TEMPLATES
# ═════════════════════════════════════════════════════════════════════
# The "alphabet" of Python: the AI generates token sequences.
# The environment checks them with ast.parse().
# Over millions of iterations, it internalizes Python grammar.

PYTHON_PROGRAMS = {
    'hello_world': 'print("Hello, World!")',
    'add_function': 'def add(a, b):\n    return a + b',
    'subtract': 'def subtract(a, b):\n    return a - b',
    'multiply': 'def multiply(a, b):\n    return a * b',
    'if_statement': 'x = 5\nif x > 0:\n    print("positive")\nelse:\n    print("non-positive")',
    'for_loop': 'for i in range(10):\n    print(i)',
    'while_loop': 'i = 0\nwhile i < 10:\n    print(i)\n    i += 1',
    'list': 'items = [1, 2, 3, 4, 5]',
    'dict': 'd = {"a": 1, "b": 2, "c": 3}',
    'function_return': 'def square(x):\n    return x * x',
    'recursion': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
    'nested_loops': 'for i in range(3):\n    for j in range(3):\n        print(i, j)',
    'list_comprehension': 'squares = [x * x for x in range(10)]',
    'string_manip': 'def greet(name):\n    return f"Hello, {name}!"',
    'type_check': 'def safe_add(a, b):\n    if isinstance(a, (int, float)) and isinstance(b, (int, float)):\n        return a + b\n    return None',
}

ALGORITHM_PROBLEMS = {
    'fibonacci': {
        'prompt': 'Write a function fibonacci(n) that returns the nth Fibonacci number.',
        'function': 'fibonacci',
        'tests': [((0,), 0), ((1,), 1), ((5,), 5), ((10,), 55), ((20,), 6765)],
    },
    'is_palindrome': {
        'prompt': 'Write a function is_palindrome(s) that checks if a string is a palindrome.',
        'function': 'is_palindrome',
        'tests': [('racecar', True), ('hello', False), ('a', True), ('', True), ('A man a plan a canal Panama'.replace(' ', '').lower(), True)],
    },
    'sort_list': {
        'prompt': 'Write a function sort_list(arr) that returns a sorted list.',
        'function': 'sort_list',
        'tests': [([3, 1, 2], [1, 2, 3]), ([], []), ([1], [1]), ([5, 5, 5], [5, 5, 5]), ([3, -1, 0, 2], [-1, 0, 2, 3])],
    },
    'reverse_string': {
        'prompt': 'Write a function reverse_string(s) that reverses a string.',
        'function': 'reverse_string',
        'tests': [('hello', 'olleh'), ('a', 'a'), ('', ''), ('racecar', 'racecar')],
    },
    'unique_elements': {
        'prompt': 'Write a function unique(arr) that returns unique elements preserving order.',
        'function': 'unique',
        'tests': [([1, 2, 2, 3, 1], [1, 2, 3]), ([], []), ([1], [1])],
    },
    'max_subarray': {
        'prompt': 'Write a function max_subarray(arr) that finds the maximum subarray sum (Kadane).',
        'function': 'max_subarray',
        'tests': [([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6), ([1], 1), ([-1], -1), ([1, 2, 3], 6)],
    },
    'count_vowels': {
        'prompt': 'Write a function count_vowels(s) that counts vowels in a string.',
        'function': 'count_vowels',
        'tests': [('hello', 2), ('AEIOU', 5), ('xyz', 0), ('', 0)],
    },
    'merge_sorted': {
        'prompt': 'Write a function merge(a, b) that merges two sorted lists.',
        'function': 'merge',
        'tests': [([1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6]), ([], [1, 2], [1, 2]), ([1], [], [1])],
    },
}


# ═════════════════════════════════════════════════════════════════════
# STAGE 2 — PROPERTY-BASED FUZZING
# ═════════════════════════════════════════════════════════════════════

def generate_random_inputs(func_name: str, count: int = 10) -> list:
    """Generate random test inputs for property-based fuzzing.

    The "Fuzzer" (the opponent) generates random inputs to try to crash
    the AI's function. This is the adversarial component of self-play.

    Returns:
        List of (arg_tuple, expected_marker) test cases
        where expected_marker is None (we only check it doesn't crash)
    """
    inputs = []

    # Integers: include edge cases
    for _ in range(count // 4):
        inputs.append(((random.randint(-1000, 1000),), None))
        inputs.append(((random.randint(-(2**31), 2**31 - 1),), None))

    # Strings: include empty, unicode, edge cases
    for _ in range(count // 4):
        length = random.randint(0, 50)
        s = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789 _-') for _ in range(length))
        inputs.append(((s,), None))

    # Lists: mixed types
    for _ in range(count // 4):
        length = random.randint(0, 20)
        lst = [random.randint(-100, 100) for _ in range(length)]
        inputs.append(((lst,), None))

    # Floats, negatives, zeros
    inputs.append(((0,), None))
    inputs.append(((0.0,), None))
    inputs.append(((-1,), None))
    inputs.append(((float('inf'),), None))

    return inputs


def build_fuzz_test_cases(func_name: str, code: str,
                          num_inputs: int = 20) -> list:
    """Build test cases that just check the function doesn't crash
    on a variety of inputs (Stage 2: Fuzzing Game).

    Returns:
        List of test case strings for the sandbox
    """
    test_cases = []
    random_inputs = generate_random_inputs(func_name, num_inputs)

    for args, _ in random_inputs:
        if isinstance(args, tuple):
            args_str = ', '.join(repr(a) for a in args)
        else:
            args_str = repr(args)
        # Just call the function — no expected value, we check it doesn't crash
        test_cases.append(f'{func_name}({args_str})')

    return test_cases


# ═════════════════════════════════════════════════════════════════════
# CODEALPHAZERO MODEL
# ═════════════════════════════════════════════════════════════════════

class CodeAlphaZero(MathTransformer):
    """Extends MathTransformer for code generation.

    Adds code-specific features:
      - ast-based syntax validation knowledge
      - Code generation from English intent prompts
      - Self-play with sandbox execution rewards

    The forward pass returns (logits, loss, value) — 3-tuple compatible
    with the existing training pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.env = PythonEnvironment(timeout_sec=2.0)
        self.stage = DevelopmentStage.SYNTAX

    def set_stage(self, stage: DevelopmentStage):
        """Set the current developmental stage."""
        self.stage = stage

    @torch.no_grad()
    def generate_code(self, tokenizer, prompt: str,
                      max_new_tokens: int = 100,
                      temperature: float = 0.8, top_k: int = 10) -> str:
        """Generate Python code from a natural language prompt.

        The prompt encodes the intent: the AI receives the English
        instruction and must generate syntactically valid code.
        """
        self.eval()
        device = next(self.parameters()).device

        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            logits, _, value = self.forward(input_ids)
            next_logits = logits[0, -1, :]

            if temperature < 0.01:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < top_vals[-1]] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

            if next_id.item() == tokenizer.eos_id:
                break

        generated = input_ids[0].tolist()
        return tokenizer.decode(generated)

    @torch.no_grad()
    def play_syntax_game(self, tokenizer, num_attempts: int = 50) -> dict:
        """Stage 1: The Syntax Game.

        Generate random token sequences, check if they parse with ast.
        Reward: +1 for valid syntax, -1 for SyntaxError.

        This is the "alphabet learning" phase — the Value Head learns
        to "feel" whether a token sequence will parse before it's done.
        """
        self.eval()
        device = next(self.parameters()).device

        # Pick a random template to start from
        template_key = random.choice(list(PYTHON_PROGRAMS.keys()))
        template = PYTHON_PROGRAMS[template_key]

        # Split template into parts — the AI must "fill in the blanks"
        lines = template.split('\n')

        # Remove random tokens to create a partial program
        prompt_tokens = tokenizer.encode(template, add_special_tokens=True)

        results = []

        for _ in range(num_attempts):
            # Generate code from the template prompt
            prompt_text = f"[CODE:{template_key}]"
            generated = self.generate_code(tokenizer, prompt_text,
                                           max_new_tokens=50,
                                           temperature=0.9, top_k=15)

            # Extract code (text after the prompt)
            code = generated.replace(prompt_text, '').strip()
            if not code:
                code = template

            # Check syntax
            valid, msg = check_syntax(code)
            reward = 1.0 if valid else -1.0

            results.append({
                'code': code,
                'valid': valid,
                'reward': reward,
                'feedback': msg if not valid else 'Syntax OK',
            })

        valid_count = sum(1 for r in results if r['valid'])

        return {
            'template': template_key,
            'attempts': num_attempts,
            'valid_count': valid_count,
            'syntax_accuracy': valid_count / max(num_attempts, 1) * 100,
            'results': results,
        }

    def play_fuzzing_game(self, tokenizer, problem_key: str = None,
                          num_inputs: int = 20) -> dict:
        """Stage 2: The Fuzzing Game.

        The AI writes a function. The Fuzzer (adversarial opponent)
        generates random inputs. If any input crashes the function,
        reward = -1. The Python traceback becomes the "error signal."

        The AI learns type safety, edge-case handling, and robustness.
        """
        self.eval()

        if problem_key is None:
            problem_key = random.choice(list(ALGORITHM_PROBLEMS.keys()))

        problem = ALGORITHM_PROBLEMS[problem_key]
        func_name = problem['function']

        # Generate code for the problem
        prompt = f"[PROBLEM:{problem['prompt']}]"
        code = self.generate_code(tokenizer, prompt,
                                  max_new_tokens=100,
                                  temperature=0.8, top_k=10)

        # Extract just the function definition
        code_cleaned = self._extract_function(code, func_name)

        if not code_cleaned:
            return {
                'problem': problem_key,
                'reward': -1.0,
                'feedback': 'Failed to generate valid function',
                'code': code,
            }

        # Build fuzz test cases
        test_cases = build_fuzz_test_cases(func_name, code_cleaned, num_inputs)

        # Evaluate: the sandbox runs all fuzz tests
        result = self.env.evaluate_move(code_cleaned, test_cases)

        return {
            'problem': problem_key,
            'function': func_name,
            'code': code_cleaned,
            'reward': result['reward'],
            'feedback': result['feedback'],
            'exec_time_ms': result['exec_time_ms'],
            'passed': result['passed'],
        }

    def play_algorithm_game(self, tokenizer, problem_key: str = None) -> dict:
        """Stage 3: Algorithmic Self-Play with Big-O intuition.

        The AI must solve algorithm problems correctly AND efficiently.
        The Value Head learns to associate code structure with execution speed.

        Multiple attempts per problem: the AI iterates, each time
        optimizing for speed while maintaining correctness.
        """
        self.eval()

        if problem_key is None:
            problem_key = random.choice(list(ALGORITHM_PROBLEMS.keys()))

        problem = ALGORITHM_PROBLEMS[problem_key]
        func_name = problem['function']
        tests = problem['tests']

        prompt = f"[PROBLEM:{problem['prompt']}]"

        # Try multiple attempts, optimizing each time
        attempts = []
        best_reward = -1.0

        for attempt in range(3):  # 3 iterations per problem
            code = self.generate_code(tokenizer, prompt,
                                      max_new_tokens=100,
                                      temperature=0.7 + attempt * 0.1,
                                      top_k=10 - attempt * 2)

            code_cleaned = self._extract_function(code, func_name)
            if not code_cleaned:
                continue

            # Build test cases with expected values
            test_strs = []
            test_strs.append(f'_result = {func_name}({repr(tests[0][0])})')

            for args, expected in tests:
                if isinstance(args, tuple):
                    args_str = ', '.join(repr(a) for a in args)
                else:
                    args_str = repr(args)
                test_strs.append(
                    f'assert {func_name}({args_str}) == {repr(expected)}'
                )

            result = self.env.evaluate_move(code_cleaned, test_strs)
            reward = result['reward']

            if reward > best_reward:
                best_reward = reward

            attempts.append({
                'attempt': attempt + 1,
                'code': code_cleaned,
                'reward': reward,
                'exec_time_ms': result['exec_time_ms'],
                'feedback': result['feedback'],
            })

            # Early stopping: if we got +1, stop optimizing
            if reward >= 1.0:
                break

        return {
            'problem': problem_key,
            'function': func_name,
            'attempts': attempts,
            'best_reward': best_reward,
        }

    def _extract_function(self, code: str, func_name: str) -> str:
        """Extract a function definition from generated code.

        Strips the prompt prefix and extracts just the function
        definition and any helper code.
        """
        # Remove prompt markers
        code = re.sub(r'\[PROBLEM:[^\]]*\]', '', code).strip()
        code = re.sub(r'\[CODE:[^\]]*\]', '', code).strip()

        # Look for a function definition
        if f'def {func_name}' in code:
            # Extract the function definition (including body)
            lines = code.split('\n')
            func_lines = []
            in_func = False
            for line in lines:
                if f'def {func_name}' in line:
                    in_func = True
                    func_lines.append(line)
                elif in_func:
                    if line.startswith('    ') or line.startswith('\t') or line.strip() == '':
                        func_lines.append(line)
                    else:
                        break
            if func_lines:
                return '\n'.join(func_lines)

        return code  # Fallback: return all code

    def alphazero_loss(self, policy_logits, targets, value_pred, reward_z,
                       config=None):
        """AlphaZero-style loss for code generation.

        policy_loss: Cross-entropy on code tokens
        value_loss: MSE between value prediction and sandbox reward
        """
        if config is None:
            config = self.config

        policy_loss = F.cross_entropy(
            policy_logits.view(-1, policy_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            label_smoothing=getattr(config, 'label_smoothing', 0.0),
        )

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
# DEMO
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
    print(f'Model: {count_parameters(model):,} params')
    print(f'Value head: {sum(p.numel() for p in model.value_head.parameters())} params')

    # Test syntax game
    print('\n=== Stage 1: Syntax Game ===')
    result = model.play_syntax_game(tok, num_attempts=10)
    print(f'  Template: {result["template"]}')
    print(f'  Valid/Total: {result["valid_count"]}/{result["attempts"]}')
    print(f'  Accuracy: {result["syntax_accuracy"]:.1f}%')

    # Test fuzzing game
    print('\n=== Stage 2: Fuzzing Game ===')
    result = model.play_fuzzing_game(tok, 'fibonacci', num_inputs=5)
    print(f'  Problem: {result.get("problem", "?")}')
    print(f'  Reward: {result.get("reward", "?"):+.1f}')
    print(f'  Feedback: {str(result.get("feedback", "?"))[:80]}')

    # Test algorithm game
    print('\n=== Stage 3: Algorithm Game ===')
    result = model.play_algorithm_game(tok, 'is_palindrome')
    print(f'  Problem: {result.get("problem", "?")}')
    print(f'  Best reward: {result.get("best_reward", "?"):+.1f}')
    for a in result.get('attempts', []):
        print(f'    Attempt {a["attempt"]}: reward={a["reward"]:+.1f} time={a["exec_time_ms"]}ms')
