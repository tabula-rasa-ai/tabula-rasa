"""Parse Socratic dialogue logs into DPO preference pairs (chosen/rejected JSONL).

Usage:
    python3 scripts/extract_dpo_pairs.py socratic_dialogue_log.json -o dpo_pairs.jsonl
    python3 scripts/extract_dpo_pairs.py memory/socratic_telemetry.json -o dpo_pairs.jsonl
    python3 scripts/extract_dpo_pairs.py memory/socratic_telemetry.json -o dpo_pairs.jsonl --min-cols 2

Extracts (chosen, rejected) pairs where:
  - chosen: the final correct trace (after Socratic critique)
  - rejected: the initial wrong trace (before critique)

Each line in the output JSONL has:
  {"prompt": "...", "chosen": "...", "rejected": "...", "expression": "...", "expected": "..."}
"""

import json
import sys
from pathlib import Path


def extract_from_dialogue_log(data: dict) -> list[dict]:
    """Extract pairs from socratic_dialogue_log.json format."""
    pairs = []
    results = data.get("results", [])
    for r in results:
        if not r.get("improved", False):
            continue
        expr = r.get("expr", "")
        expected = str(r.get("true_answer", ""))
        # These fields depend on the specific log format
        baseline = r.get("model_a_answer", "")
        socratic = r.get("socratic_answer", "")
        if baseline and socratic and socratic != baseline:
            pairs.append({
                "prompt": f"{expr}=",
                "chosen": socratic,
                "rejected": baseline,
                "expression": expr,
                "expected": expected,
            })
    return pairs


def extract_from_telemetry(data: list[dict]) -> list[dict]:
    """Extract pairs from memory/socratic_telemetry.json format.

    Each entry has 'results' with rounds showing traces.
    A correction is: round 1 trace (wrong) vs last round trace (correct).
    """
    pairs = []
    for entry in data:
        summary = entry.get("summary", {})
        results = entry.get("results", [])
        for r in results:
            rounds = r.get("rounds", [])
            if len(rounds) < 2:
                continue
            passed = r.get("passed", False)
            if not passed:
                continue  # Only use successful corrections

            first_trace = rounds[0].get("trace", "")
            last_trace = rounds[-1].get("trace", "")
            if first_trace and last_trace and first_trace != last_trace:
                pairs.append({
                    "prompt": f"{r['expression']}=",
                    "chosen": last_trace,
                    "rejected": first_trace,
                    "expression": r["expression"],
                    "expected": r["expected"],
                    "n_rounds": r.get("n_rounds", len(rounds)),
                    "avg_time_ms": summary.get("avg_time_ms", 0),
                })
    return pairs


def extract_from_correction_log(data: list[dict]) -> list[dict]:
    """Extract pairs from correction log format.

    If entries have 'first_round_trace' and 'correct_trace' fields,
    those are direct chosen/rejected pairs.
    """
    pairs = []
    for entry in data if isinstance(data, list) else []:
        chosen = entry.get("correct_trace", "") or entry.get("chosen", "")
        rejected = entry.get("first_round_trace", "") or entry.get("rejected", "")
        prompt = entry.get("hint", "") or entry.get("prompt", "")
        expr = entry.get("expression", "") or entry.get("problem", "")
        expected = entry.get("expected", "") or entry.get("expected_answer", "")
        if chosen and rejected and chosen != rejected:
            pairs.append({
                "prompt": prompt or f"{expr}=",
                "chosen": chosen,
                "rejected": rejected,
                "expression": expr,
                "expected": expected,
            })
    return pairs


def extract_pairs(data) -> list[dict]:
    """Auto-detect format and extract pairs."""
    if isinstance(data, dict):
        if "results" in data:
            pairs = extract_from_dialogue_log(data)
            if pairs:
                return pairs
        # Try as correction log
        for v in data.values():
            if isinstance(v, list) and v:
                return extract_from_correction_log(v)
    elif isinstance(data, list):
        # Check for telemetry format (has 'results' in first entry)
        if data and isinstance(data[0], dict):
            if "results" in data[0] and "config" in data[0]:
                return extract_from_telemetry(data)
            return extract_from_correction_log(data)
    return []


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract DPO preference pairs from Socratic logs"
    )
    parser.add_argument("input", type=str, help="Path to Socratic log JSON")
    parser.add_argument("-o", "--output", type=str, default="dpo_pairs.jsonl",
                        help="Output JSONL path (default: dpo_pairs.jsonl)")
    parser.add_argument("--min-cols", type=int, default=0,
                        help="Minimum number of columns in trace (filters noisy short traces)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    pairs = extract_pairs(data)

    # Filter by minimum columns
    if args.min_cols > 0:
        pairs = [p for p in pairs if len(p["chosen"]) >= args.min_cols]

    if not pairs:
        print(f"No correction pairs found in {args.input}")
        print("The log may not contain improved examples. Try a different log or run")
        print("a Socratic experiment with --socratic on a trained model first.")
        sys.exit(0)

    with open(args.output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Extracted {len(pairs)} DPO preference pairs")
    print(f"Output: {args.output}")
    print("\nSample pair:")
    print(json.dumps(pairs[0], indent=2))


if __name__ == "__main__":
    main()
