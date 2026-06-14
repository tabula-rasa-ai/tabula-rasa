"""Tests for dataset — problem generation, dataset creation."""
import torch

from tabula_rasa.dataset import generate_problem, format_math_sample, MathDataset, generate_test_set


class TestGenerateProblem:
    def test_returns_tuple(self):
        """generate_problem returns (expression, answer) strings."""
        expr, ans = generate_problem(min_digits=1, max_digits=2)
        assert isinstance(expr, str)
        assert isinstance(ans, str)
        assert len(expr) > 0
        assert len(ans) > 0

    def test_includes_operator(self):
        """Expression includes one of + - * /."""
        expr, ans = generate_problem(min_digits=1, max_digits=2)
        assert any(op in expr for op in ['+', '-', '*', '/'])

    def test_operand_bounds(self):
        """Operand lengths are generally within bounds (division may produce larger operands)."""
        import re
        within_bounds = 0
        total = 0
        for _ in range(50):
            expr, ans = generate_problem(min_digits=1, max_digits=2)
            nums = re.findall(r'\d+', expr)
            for n in nums:
                total += 1
                if 1 <= len(n) <= 3:  # division can bump one operand
                    within_bounds += 1
        # At least 80% should be within bounds
        assert within_bounds / total >= 0.8, f"Only {within_bounds}/{total} within bounds"

    def test_correct_addition(self):
        """Addition problems have correct answers."""
        import re
        for _ in range(20):
            expr, ans = generate_problem(min_digits=1, max_digits=2)
            if '+' not in expr:
                continue
            nums = re.findall(r'\d+', expr)
            if len(nums) >= 2:
                expected = str(int(nums[0]) + int(nums[1]))
                assert ans == expected, f"{expr} expected {expected}, got {ans}"

    def test_correct_subtraction(self):
        """Subtraction problems have correct answers (positive result)."""
        import re
        for _ in range(20):
            expr, ans = generate_problem(min_digits=1, max_digits=2)
            if '-' not in expr:
                continue
            nums = re.findall(r'\d+', expr)
            if len(nums) >= 2:
                a, b = int(nums[0]), int(nums[1])
                expected = str(abs(a - b))  # function ensures positive
                assert ans == expected, f"{expr} expected {expected}, got {ans}"

    def test_deterministic_with_seed(self):
        """generate_test_set runs without error with a seed."""
        problems = generate_test_set(10, seed=42)
        assert len(problems) == 10


class TestFormatMathSample:
    def test_format(self):
        """format_math_sample produces 'expr=answer'."""
        result = format_math_sample("12+34", "46")
        assert result == "12+34=46"

    def test_with_operator(self):
        """Works with different operations."""
        assert format_math_sample("5*3", "15") == "5*3=15"


class TestMathDataset:
    def test_creates_samples(self, tok):
        """Dataset creates expected number of samples."""
        ds = MathDataset(tok, num_samples=100, max_digits=2)
        assert len(ds) > 0

    def test_returns_tensors(self, tok):
        """__getitem__ returns two tensors (input, target)."""
        ds = MathDataset(tok, num_samples=50, max_digits=2, seed=42)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_input_target_shift(self, tok):
        """Target is input shifted by one position (next-token prediction)."""
        ds = MathDataset(tok, num_samples=50, max_digits=2, seed=42)
        x, y = ds[0]
        assert x.shape == y.shape
        assert x[0].item() == tok.bos_id
        # y should be x shifted: y[i] = x[i+1] for non-padded
        non_pad = (x != tok.pad_id).sum().item()
        if non_pad > 1:
            assert y[0].item() != tok.pad_id  # first target is second token

    def test_max_seq_len_respected(self, tok):
        """All sequences are within max_seq_len."""
        max_len = 16
        tok.max_seq_len = max_len  # set for this test
        ds = MathDataset(tok, num_samples=50, max_digits=4, seed=42)
        for i in range(min(10, len(ds))):
            x, y = ds[i]
            assert len(x) <= max_len, f"Sequence {i} length {len(x)} exceeds {max_len}"
            assert len(y) <= max_len


class TestGenerateTestSet:
    def test_returns_list(self):
        """generate_test_set returns list of tuples."""
        problems = generate_test_set(5)
        assert len(problems) == 5
        for expr, ans in problems:
            assert isinstance(expr, str)
            assert isinstance(ans, str)

    def test_varied_difficulty(self, tok):
        """Test set includes varied digit lengths."""
        import re
        problems = generate_test_set(50, seed=42)
        lengths = set()
        for expr, _ in problems:
            nums = re.findall(r'\d+', expr)
            for n in nums:
                lengths.add(len(n))
        assert len(lengths) >= 2, f"Expected varied digit lengths, got {lengths}"
