"""Tool Use System — lets specialists call external tools (calculator, web, code).

Specialists can request tool calls during generation by emitting a special
syntax: [[tool_name(param1, param2)]]. The tool system intercepts these,
executes the tool, and injects the result back into the context.

Supported tools:
- calculator(expr)    — Evaluate a math expression
- python(code)        — Execute Python code in a sandbox
- web_search(query)   — Search the web (requires internet)
- today()             — Get current date
- random(min, max)    — Generate a random integer

Usage:
    from tabula_rasa.reasoning.tool_use import ToolUse, TOOL_REGISTRY
    tools = ToolUse()
    result = tools.execute("What is [[calculator(24*7)]]?")
    # -> "What is 168?"
"""

import ast
import math
import random
import re
import time
from typing import Any, Callable, Optional

from tabula_rasa.math_parser import evaluate as math_eval, parse_expression


# ═══════════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════════


def _tool_calculator(expr: str) -> str:
    """Evaluate a mathematical expression using the math_parser.

    Supports: +, -, *, /, parentheses, and basic arithmetic.
    Returns the result as a string.
    """
    expr = expr.strip()
    # Try math_parser first
    parsed = parse_expression(expr)
    if parsed is not None:
        op, a, b, _ = parsed
        try:
            result = math_eval(a, b, op)
            return str(result)
        except (ValueError, ZeroDivisionError) as e:
            return f"Error: {e}"

    # Fall back to safe eval for compound expressions
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "int": int, "float": float,
        "math": math,
    }
    try:
        result = eval(expr, safe_globals, safe_locals)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _tool_python(code: str) -> str:
    """Execute Python code in a restricted sandbox and return the output.

    WARNING: This is NOT a fully secure sandbox. Only use with trusted input.
    For production, replace with RestrictedPython or a subprocess jail.
    """
    import io
    import sys

    code = code.strip()
    # Remove markdown code fences if present
    code = re.sub(r'^```(?:python)?\n', '', code)
    code = re.sub(r'\n```$', '', code)

    # Capture stdout
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    try:
        # Compile and exec
        compiled = compile(code, '<tool_python>', 'exec')
        exec(compiled, {"__builtins__": __builtins__, "print": print})
        output = captured.getvalue()
        return output if output else "(no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout


def _tool_today(dummy: str = "") -> str:
    """Get today's date as YYYY-MM-DD."""
    return time.strftime("%Y-%m-%d")


def _tool_random(min_max: str) -> str:
    """Generate a random integer between min and max (inclusive).

    Args: "min, max" or "min max"
    """
    parts = min_max.replace(",", " ").split()
    if len(parts) >= 2:
        try:
            lo, hi = int(parts[0]), int(parts[1])
            return str(random.randint(lo, hi))
        except ValueError:
            return f"Error: invalid range '{min_max}'"
    return f"Error: need min and max, got '{min_max}'"


def _tool_help(tool_name: str = "") -> str:
    """Show available tools and their usage."""
    if tool_name.strip():
        tool_name = tool_name.strip().lower()
        for name, meta in TOOL_REGISTRY.items():
            if name == tool_name:
                return f"**{name}**: {meta['doc']}\nSyntax: [[{name}({meta['syntax']})]]"
        return f"Unknown tool: {tool_name}. Available: {', '.join(TOOL_REGISTRY.keys())}"
    lines = ["Available tools:", ""]
    for name, meta in sorted(TOOL_REGISTRY.items()):
        lines.append(f"  [[{name}({meta['syntax']})]] — {meta['doc']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════════════════


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    'calculator': {
        'fn': _tool_calculator,
        'doc': 'Evaluate a math expression',
        'syntax': 'expression',
    },
    'python': {
        'fn': _tool_python,
        'doc': 'Execute Python code and return output',
        'syntax': 'code',
    },
    'today': {
        'fn': _tool_today,
        'doc': 'Get current date',
        'syntax': '',
    },
    'random': {
        'fn': _tool_random,
        'doc': 'Generate a random integer',
        'syntax': 'min, max',
    },
    'help': {
        'fn': _tool_help,
        'doc': 'Show available tools',
        'syntax': 'tool_name (optional)',
    },
}


# ═══════════════════════════════════════════════════════════════════
# Tool Use Engine
# ═══════════════════════════════════════════════════════════════════


_TOOL_CALL_RE = re.compile(r'\[\[(\w+)\(([^)]*)\)\]\]')


def _find_tool_calls(text: str) -> list[tuple[str, str, int, int]]:
    """Find all [[tool(args)]] calls in text, handling nested parentheses.

    Uses a manual balanced-paren parser so it works with arbitrary nesting
    depth (e.g. [[python(print(sum(range(10))))]]).
    """
    calls = []
    i = 0
    while i < len(text):
        # Look for [[
        start = text.find('[[', i)
        if start == -1:
            break

        # Find tool name
        paren = text.find('(', start + 2)
        if paren == -1 or paren > start + 30:
            i = start + 2
            continue

        tool_name = text[start + 2:paren]
        if not tool_name.isidentifier():
            i = paren + 1
            continue

        # Parse balanced parentheses for args
        depth = 1
        pos = paren + 1
        while pos < len(text) and depth > 0:
            if text[pos] == '(':
                depth += 1
            elif text[pos] == ')':
                depth -= 1
            pos += 1

        if depth != 0:
            i = paren + 1
            continue

        args = text[paren + 1:pos - 1]
        end = pos

        # Check for closing ]]
        if end + 1 < len(text) and text[end:end + 2] == ']]':
            calls.append((tool_name.lower(), args, start, end + 2))
            i = end + 2
        else:
            i = end

    return calls


class ToolUse:
    """Tool-use engine for specialists.

    Parses [[tool_name(args)]] syntax in text, executes the corresponding
    tool, and replaces the call with the result.

    Args:
        registry: Optional custom tool registry. Uses TOOL_REGISTRY by default.
        max_recursion: Max nested tool calls (default: 3).
    """

    def __init__(
        self,
        registry: Optional[dict[str, dict[str, Any]]] = None,
        max_recursion: int = 3,
    ):
        self.registry = registry or TOOL_REGISTRY
        self.max_recursion = max_recursion
        self.call_history: list[dict] = []

    def execute(
        self,
        text: str,
        context: Optional[dict] = None,
        _depth: int = 0,
    ) -> str:
        """Execute all tool calls in the given text.

        Args:
            text: Text containing [[tool(args)]] calls.
            context: Optional context dict passed to tools.
            _depth: Internal recursion counter.

        Returns:
            Text with tool calls replaced by their results.
        """
        if _depth > self.max_recursion:
            return text

        def _find_calls(text: str) -> list[tuple[str, str, int, int]]:
            """Find all tool calls in text."""
            return _find_tool_calls(text)

        calls = _find_calls(text)
        if not calls:
            return text

        # Process right-to-left so positions stay valid
        result = text
        for tool_name, raw_args, start, end in reversed(calls):

            if tool_name not in self.registry:
                self.call_history.append({
                    'tool': tool_name,
                    'args': raw_args,
                    'result': f"Unknown tool: {tool_name}",
                    'time_ms': 0,
                    'success': False,
                })
                replacement = f"[Unknown tool: {tool_name}]"
            else:
                meta = self.registry[tool_name]
                t0 = time.time()
                try:
                    tool_result = meta['fn'](raw_args)
                    elapsed = (time.time() - t0) * 1000
                    self.call_history.append({
                        'tool': tool_name,
                        'args': raw_args,
                        'result': str(tool_result)[:200],
                        'time_ms': round(elapsed, 1),
                        'success': True,
                    })
                    replacement = str(tool_result)
                except Exception as e:
                    elapsed = (time.time() - t0) * 1000
                    self.call_history.append({
                        'tool': tool_name,
                        'args': raw_args,
                        'result': str(e),
                        'time_ms': round(elapsed, 1),
                        'success': False,
                    })
                    replacement = f"[Error: {e}]"

            result = result[:start] + replacement + result[end:]

        return result

    def clear_history(self):
        """Clear the tool call history."""
        self.call_history = []

    def get_recent_calls(self, n: int = 10) -> list[dict]:
        """Return the N most recent tool calls."""
        return self.call_history[-n:]


# ═══════════════════════════════════════════════════════════════════
# Specialist-Integrated Tool Use
# ═══════════════════════════════════════════════════════════════════


def augment_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    tools: Optional[ToolUse] = None,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    top_k: int = 5,
) -> str:
    """Generate text with tool-use augmentation.

    The model generates text normally. If the output contains [[tool()]]
    calls, the tools are executed and the results are appended, allowing
    the model to continue generating with the tool output in context.

    Args:
        model: MathTransformer model.
        tokenizer: MathTokenizer.
        prompt: Input prompt.
        tools: ToolUse instance (created if None).
        max_new_tokens: Max generation tokens.
        temperature: Sampling temperature.

    Returns:
        Generated text with tool results integrated.
    """
    if tools is None:
        tools = ToolUse()

    # Initial generation
    output = model.generate(
        tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    # Check for tool calls
    if '[[' in output:
        # Extract just the new content (after prompt)
        new_content = output[len(prompt):] if output.startswith(prompt) else output
        # Execute tool calls
        tool_result = tools.execute(new_content)
        # Append tool result
        return output + '\n' + tool_result

    return output


# ═══════════════════════════════════════════════════════════════════
# CLI / Quick Test
# ═══════════════════════════════════════════════════════════════════


def main():
    """Test the tool-use system with example calls."""
    tools = ToolUse()

    print("Tool Use System — Test")
    print("=" * 50)

    test_cases = [
        "What is [[calculator(24*7)]]?",
        "Today is [[today()]].",
        "A random number: [[random(1, 100)]]",
        "[[help()]]",
        "Calculator: [[calculator(99+1)]] and [[calculator(100/4)]]",
        "[[python(print(sum(range(10))))]]",
    ]

    for case in test_cases:
        result = tools.execute(case)
        print(f"\n  Input:  {case}")
        print(f"  Output: {result}")

    print(f"\n{'=' * 50}")
    print(f"Tool calls executed: {len(tools.call_history)}")
    for call in tools.call_history:
        icon = "✓" if call['success'] else "✗"
        print(f"  {icon} {call['tool']}({call['args']}) -> {call['result'][:60]} ({call['time_ms']}ms)")


if __name__ == "__main__":
    main()
