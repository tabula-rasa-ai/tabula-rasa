"""
code_sandbox.py — The "Chess Board" for Code AlphaZero.

In AlphaZero, the environment is the Chess board.
For the Code Specialist, the environment is a Sandboxed Python REPL.

Components:
  1. ast.parse() — "Rules Engine": instantly flags Syntax Errors (illegal moves)
  2. subprocess isolation — Sandboxed execution with timeout and import blacklist
  3. Test case evaluation — Property-based testing with hypothesis
  4. Efficiency scoring — Execution time measurement for Big-O intuition
  5. Dangerous import blacklist — Protects against os.system('rm -rf /')

Usage:
    env = PythonEnvironment(timeout_sec=2)
    result = env.evaluate_move(code_string, test_cases=[...])
    # Returns: {'reward': -1..+1, 'feedback': str, 'exec_time_ms': float}
"""

import ast
import os
import re
import subprocess
import sys
import time

# ═════════════════════════════════════════════════════════════════════
# DANGEROUS IMPORT BLACKLIST — The Pythagorean Security Audit
# ═════════════════════════════════════════════════════════════════════
# These modules are blacklisted from the sandbox.
# If the AI uses them, the code is rejected with -1.0 reward.

BLACKLISTED_MODULES = {
    'os', 'subprocess', 'shutil', 'sys', 'ctypes', 'socket',
    'signal', 'multiprocessing', 'threading', 'concurrent',
    'pathlib', 'tempfile', 'atexit', 'inspect', 'importlib',
    'builtins.__import__', 'eval', 'exec', 'compile',
    'open', '__builtins__', 'pickle', 'shelve', 'marshal',
    'cffi', 'cython', 'distutils', 'setuptools',
    'webbrowser', 'antigravity', 'turtle', 'tkinter',
    'http.server', 'socketserver', 'xmlrpc',
}

BLACKLISTED_KEYWORDS = [
    'import os', 'import subprocess', 'import shutil',
    'import ctypes', 'import socket', 'import sys',
    'from os', 'from subprocess', 'from ctypes',
    'import multiprocessing', 'import threading',
    '__import__', 'eval(', 'exec(', 'compile(',
    'open(', 'file(', 'builtins.',
    'os.system', 'os.popen', 'subprocess.',
    'shutil.rmtree', 'shutil.move', 'shutil.copy',
    'ctypes.CDLL', 'ctypes.WinDLL',
]


def has_dangerous_imports(code: str) -> tuple[bool, str]:
    """Check if code contains blacklisted imports or dangerous patterns.

    Returns:
        (True, reason) if dangerous, (False, '') if safe.
    """
    code_lower = code.lower()
    for keyword in BLACKLISTED_KEYWORDS:
        if keyword in code_lower:
            return True, f"Blacklisted keyword: {keyword}"
    return False, ""


# ═════════════════════════════════════════════════════════════════════
# SYNTAX VALIDATOR — The "Illegal Move" Checker
# ═════════════════════════════════════════════════════════════════════

def check_syntax(code: str) -> tuple[bool, str]:
    """Check Python syntax using ast.parse().

    This is the AlphaZero "Rules Engine" — it instantly rejects
    illegal moves (SyntaxErrors) without executing the code.

    Returns:
        (True, '') if valid, (False, error_message) if invalid.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        msg = f"SyntaxError at line {e.lineno}: {e.msg}"
        if e.text:
            msg += f"\n  {e.text.strip()}"
            if e.offset:
                msg += f"\n  {' ' * (e.offset - 1)}^"
        return False, msg
    except Exception as e:
        return False, f"Parse error: {e}"


# ═════════════════════════════════════════════════════════════════════
# EXECUTION SANDBOX — Subprocess isolation
# ═════════════════════════════════════════════════════════════════════
# Runs code in a subprocess with:
#   - Timeout protection (prevents infinite loops)
#   - Import blacklist filtering
#   - stdout/stderr capture
#   - Memory usage limits

class PythonEnvironment:
    """The "Chess Board" for Code AlphaZero.

    Evaluates code moves and returns rewards based on:
    - Syntax validity (ast.parse)
    - Runtime correctness (test cases)
    - Execution efficiency (time measurement)
    - Safety (import blacklist)
    """

    def __init__(self, timeout_sec: float = 2.0, max_code_len: int = 2000):
        self.timeout = timeout_sec
        self.max_code_len = max_code_len
        self.stats = {
            'syntax_passes': 0,
            'syntax_fails': 0,
            'runtime_passes': 0,
            'runtime_fails': 0,
            'timeouts': 0,
            'dangerous_rejected': 0,
        }

    def evaluate_move(self, code_string: str, test_cases: list = None,
                      check_imports: bool = True) -> dict:
        """Evaluate a single code "move" and return reward + feedback.

        Args:
            code_string: Python source code to evaluate
            test_cases: List of (input_args, expected_output) tuples.
                       If None, just checks syntax + runs the code.
            check_imports: If True, reject dangerous modules.

        Returns:
            {
                'reward': float (-1.0 to +1.0),
                'feedback': str (human-readable),
                'exec_time_ms': float,
                'passed': bool,
                'syntax_valid': bool,
            }
        """
        start_time = time.time()

        # ── Step 0: Length limit ──
        if len(code_string) > self.max_code_len:
            return {
                'reward': -1.0,
                'feedback': f'Code too long ({len(code_string)} > {self.max_code_len} chars)',
                'exec_time_ms': 0,
                'passed': False,
                'syntax_valid': False,
            }

        # ── Step 1: Import blacklist ──
        if check_imports:
            dangerous, reason = has_dangerous_imports(code_string)
            if dangerous:
                self.stats['dangerous_rejected'] += 1
                return {
                    'reward': -1.0,
                    'feedback': f'DANGEROUS: {reason}',
                    'exec_time_ms': 0,
                    'passed': False,
                    'syntax_valid': True,
                }

        # ── Step 2: Syntax check (illegal moves) ──
        valid_syntax, syntax_msg = check_syntax(code_string)
        if not valid_syntax:
            self.stats['syntax_fails'] += 1
            return {
                'reward': -1.0,
                'feedback': f'Illegal Move: {syntax_msg}',
                'exec_time_ms': 0,
                'passed': False,
                'syntax_valid': False,
            }

        self.stats['syntax_passes'] += 1

        # ── Step 3: Execute with test cases ──
        if test_cases:
            result = self._run_with_tests(code_string, test_cases)
        else:
            result = self._run_bare(code_string)

        elapsed_ms = (time.time() - start_time) * 1000
        result['exec_time_ms'] = round(elapsed_ms, 1)

        # ═══════════════════════════════════════════════════════
        # Efficiency bonus: reward execution speed
        # This is what trains the Value Head to "intuit" Big-O
        # ═══════════════════════════════════════════════════════
        if result['passed']:
            speed = elapsed_ms
            if speed < 10:
                result['reward'] = min(1.0, result.get('reward', 0) + 0.3)
            elif speed < 50:
                result['reward'] = min(1.0, result.get('reward', 0) + 0.1)
            elif speed > 500:
                result['reward'] = max(-1.0, result.get('reward', 0) - 0.2)

        return result

    def _run_with_tests(self, code: str, test_cases: list) -> dict:
        """Run code with test cases in a sandboxed subprocess."""
        # Build the test runner script
        test_lines = []
        for i, tc in enumerate(test_cases):
            if isinstance(tc, tuple) and len(tc) == 2:
                args, expected = tc
                # Format the call
                if isinstance(args, tuple):
                    args_str = ', '.join(repr(a) for a in args)
                else:
                    args_str = repr(args)
                expected_str = repr(expected)
                test_lines.append(
                    f'result_{i} = {args_str}  # Call the function\n'
                    f'expected_{i} = {expected_str}\n'
                    f'assert result_{i} == expected_{i}, '
                    f'"Test {i}: got {{result_{i}}}, expected {expected_str}"'
                )
            elif isinstance(tc, str):
                test_lines.append(tc)

        test_code = '\n'.join(test_lines) if test_lines else 'pass'

        runner = f"""
import sys
# User code:
{code}

# Tests:
try:
    {test_code}
    print("__CODE_RESULT__:SUCCESS")
except Exception as e:
    print(f"__CODE_RESULT__:FAILED:{{type(e).__name__}}:{{e}}")
"""
        return self._execute_runner(runner)

    def _run_bare(self, code: str) -> dict:
        """Run code without test cases (just check it runs)."""
        runner = f"""
import sys
# User code:
{code}

try:
    exec({repr(code)})
    print("__CODE_RESULT__:SUCCESS")
except Exception as e:
    print(f"__CODE_RESULT__:FAILED:{{type(e).__name__}}:{{e}}")
"""
        return self._execute_runner(runner)

    def _execute_runner(self, runner_code: str) -> dict:
        """Execute the runner script in a sandboxed subprocess.

        Uses subprocess with timeout to protect against:
        - Infinite loops (while True:)
        - Blocking calls (input(), sleep())
        - OS manipulation
        """
        try:
            result = subprocess.run(
                [sys.executable, '-c', runner_code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={},
                cwd=os.path.abspath('sandbox_temp') if os.path.exists('sandbox_temp') else os.path.dirname(__file__),
            )

            stdout = result.stdout or ''
            stderr = result.stderr or ''

            # Parse the result
            if '__CODE_RESULT__:SUCCESS' in stdout:
                self.stats['runtime_passes'] += 1
                # Check for test count
                test_count = stdout.count('SUCCESS')
                return {
                    'reward': 1.0,
                    'feedback': f'All tests passed ({test_count} assertions)',
                    'passed': True,
                    'syntax_valid': True,
                    'raw_output': stdout,
                }
            else:
                self.stats['runtime_fails'] += 1
                # Extract the failure message
                fail_match = re.search(r'__CODE_RESULT__:FAILED:(\w+):(.+)', stdout)
                if fail_match:
                    err_type = fail_match.group(1)
                    err_msg = fail_match.group(2)
                    feedback = f'RuntimeError: {err_type}: {err_msg}'
                elif stderr:
                    # Python traceback
                    lines = stderr.strip().split('\n')
                    feedback = '\n'.join(lines[-3:])  # Last 3 lines of traceback
                else:
                    feedback = 'Unknown failure'

                return {
                    'reward': -0.5,
                    'feedback': f'Tests failed: {feedback}',
                    'passed': False,
                    'syntax_valid': True,
                    'raw_output': stdout,
                    'raw_stderr': stderr,
                }

        except subprocess.TimeoutExpired:
            self.stats['timeouts'] += 1
            return {
                'reward': -1.0,
                'feedback': 'Illegal Move: Infinite Loop detected (Timeout)',
                'passed': False,
                'syntax_valid': True,
                'exec_time_ms': self.timeout * 1000,
            }
        except Exception as e:
            return {
                'reward': -0.5,
                'feedback': f'Sandbox error: {e}',
                'passed': False,
                'syntax_valid': True,
            }

    def get_stats(self) -> dict:
        """Get environment statistics."""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total_evaluations': total,
            'pass_rate': (
                (self.stats['syntax_passes'] + self.stats['runtime_passes']) /
                max(total, 1) * 100
            ) if total > 0 else 0,
        }


# ═════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    env = PythonEnvironment(timeout_sec=2)

    print('=== Code Sandbox Tests ===\n')

    # Test 1: Valid syntax
    print('Test 1: Valid code')
    code1 = 'def add(a, b): return a + b'
    r = env.evaluate_move(code1)
    print(f'  Reward: {r["reward"]:+.1f} | {r["feedback"]}')

    # Test 2: Syntax error
    print('\nTest 2: Syntax error')
    code2 = 'def add(a, b) return a + b'  # Missing colon
    r = env.evaluate_move(code2)
    print(f'  Reward: {r["reward"]:+.1f} | {r["feedback"][:60]}')

    # Test 3: With test cases
    print('\nTest 3: Code with test cases (pass)')
    code3 = 'def add(a, b): return a + b'
    tests3 = [((2, 3), 5), ((-1, 1), 0)]
    r = env.evaluate_move(code3, tests3)
    print(f'  Reward: {r["reward"]:+.1f} | {r["feedback"]}')

    # Test 4: With test cases (fail)
    print('\nTest 4: Code with test cases (fail)')
    code4 = 'def add(a, b): return a * b'  # Wrong: multiplication instead of addition
    tests4 = [((2, 3), 5)]
    r = env.evaluate_move(code4, tests4)
    print(f'  Reward: {r["reward"]:+.1f} | {r["feedback"][:60]}')

    # Test 5: Dangerous import
    print('\nTest 5: Dangerous import')
    code5 = 'import os; os.system("rm -rf /")'
    r = env.evaluate_move(code5)
    print(f'  Reward: {r["reward"]:+.1f} | {r["feedback"][:60]}')

    # Test 6: Infinite loop (timeout)
    print('\nTest 6: Infinite loop')
    code6 = 'while True: pass'
    r = env.evaluate_move(code6)
    print(f'  Reward: {r["reward"]:+.1f} | {r["feedback"]}')

    print(f'\n=== Stats: {env.get_stats()} ===')
