"""RouterDataset — synthetic prompt generation for neural router training.

Generates 5,000+ diverse human-language prompts across 4 intent categories.
No LLM required — uses template-based generation with rich randomization to
produce natural variety in phrasing, vocabulary, and structure.

Categories:
  0 = MATH    — arithmetic, calculations, equations, numbers, money, measurement
  1 = CODE    — programming, algorithms, code generation, loops, data structures
  2 = MEMORY  — remember/remind, store/recall facts, to-do items, passwords
  3 = CHAT    — greetings, opinions, casual conversation, trivia, questions
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ─── Intent definitions (matches router_model.py) ─────────────────

INTENT_NAMES = ["MATH", "CODE", "MEMORY", "CHAT"]


# ═══════════════════════════════════════════════════════════════════
# MATH TEMPLATES (0)
# ═══════════════════════════════════════════════════════════════════

_MATH_OPERATIONS = [
    "add", "subtract", "multiply", "divide", "calculate", "compute"
]

_MATH_PHRASES_1 = [
    "What is {a} plus {b}?",
    "Calculate {a} + {b}",
    "What is {a} minus {b}?",
    "{a} - {b} = ?",
    "What is {a} times {b}?",
    "Multiply {a} by {b}",
    "{a} * {b}",
    "Divide {a} by {b}",
    "What is {a} / {b}?",
    "If I have {a} apples and buy {b} more, how many do I have?",
    "There are {a} students and {b} teachers. How many people total?",
    "I ran {a} miles on Monday and {b} on Tuesday. Total?",
    "A pizza has {a} slices. I eat {b}. How many left?",
    "Can you solve {a}x + {b} = 0 for x?",
    "What is {a} percent of {b}?",
]

_MATH_PHRASES_2 = [
    "If a train travels at {a} mph for {b} hours, how far does it go?",
    "Convert {a} {unit1} to {unit2}",
    "A rectangle is {a} x {b}. What is the area?",
    "The bill is ${a}. A {b}% tip would be how much?",
    "Split ${a} evenly among {b} people",
    "What is the square root of {a}?",
    "Calculate {a} to the power of {b}",
    "How many seconds in {a} minutes?",
    "If I save ${a} per week for {b} weeks, how much total?",
    "A recipe needs {a} cups of flour for {b} servings. Scale to {c} servings.",
    "{a} km in miles?",
    "What is {a} degrees Celsius in Fahrenheit?",
    "The temperature dropped from {a} to {b}. What's the change?",
    "I weigh {a} kg. Convert to pounds.",
    "A {a}% discount on ${b}. What's the final price?",
]

_MATH_PREFIXES = [
    "Hey, ", "Can you ", "Please ", "Quick ", "I need to ", "Help me ",
    "", "", "", "",
]

_MATH_TEMPLATES = _MATH_PHRASES_1 + _MATH_PHRASES_2

_UNITS = [
    ("meters", "feet"), ("km", "miles"), ("kg", "pounds"),
    ("gallons", "liters"), ("inches", "cm"), ("F", "C"),
]

_AMOUNTS_A = list(range(1, 1001))
_AMOUNTS_B = list(range(1, 100))
_AMOUNTS_C = list(range(1, 20))


def _gen_math(num: int) -> list[tuple[str, str]]:
    """Generate MATH prompts."""
    samples = []
    for _ in range(num):
        a = random.choice(_AMOUNTS_A)
        b = random.choice(_AMOUNTS_B)
        c = random.choice(_AMOUNTS_C)
        unit1, unit2 = random.choice(_UNITS)
        prefix = random.choice(_MATH_PREFIXES)
        template = random.choice(_MATH_TEMPLATES)
        prompt = prefix + template.format(a=a, b=b, c=c, unit1=unit1, unit2=unit2)
        # Ensure it's at least somewhat natural — strip edge artifacts
        prompt = prompt.replace("  ", " ").strip()
        samples.append((prompt, "MATH"))
    return samples


# ═══════════════════════════════════════════════════════════════════
# CODE TEMPLATES (1)
# ═══════════════════════════════════════════════════════════════════

_CODE_ACTIONS = [
    "write", "create", "implement", "build", "code", "program", "make"
]

_CODE_LANGUAGES = ["Python", "JavaScript", "Java", "C++", "Rust", "Go", "TypeScript", "Ruby"]

_CODE_TEMPLATES = [
    "{action} a function that {task}",
    "{action} a {lang} program that {task}",
    "How do I {task} in {lang}?",
    "Can you {action} code to {task}?",
    "Write a loop that {task}",
    "Implement {ds} in {lang}",
    "Code a {algo} algorithm",
    "{action} a {lang} script to {task}",
    "I need a {lang} function for {task}",
    "What's the {lang} syntax for {task}?",
    "Show me how to {task} in {lang}",
    "Write a {ds} class with {features}",
    "How would you implement {algo} from scratch?",
    "I'm trying to {task} in my {lang} project",
    "Can you refactor this {lang} code? {task}",
    "Generate {lang} code that {task}",
    "Write unit tests for {task}",
    "What's the best way to {task} in {lang}?",
    "Create a CLI tool that {task}",
    "Build a REST API endpoint for {task}",
]

_CODE_TASKS = [
    "reverses a string",
    "finds the largest number in an array",
    "checks if a word is a palindrome",
    "calculates the fibonacci sequence",
    "sorts a list of numbers",
    "merges two sorted arrays",
    "finds duplicates in a list",
    "counts word frequency in a text",
    "validates an email address",
    "encodes a URL",
    "parses a JSON string",
    "makes an HTTP request",
    "reads a file line by line",
    "calculates the factorial of a number",
    "finds the longest common prefix",
    "implements a binary search",
    "converts between data formats",
    "generates all permutations of a list",
    "finds the shortest path in a graph",
    "implements a LRU cache",
    "checks balanced parentheses",
    "finds the median of two arrays",
    "implements a simple neural network layer",
    "parses command-line arguments",
    "creates a web server",
]

_CODE_DATASTRUCTURES = [
    "a linked list", "a binary tree", "a hash map", "a stack", "a queue",
    "a priority queue", "a graph", "a trie", "a bloom filter", "a skip list",
]

_CODE_ALGOS = [
    "binary search", "quicksort", "bubble sort", "merge sort", "Dijkstra's",
    "BFS", "DFS", "A* search", "dynamic programming", "greedy",
    "recursive backtracking", "two-pointer", "sliding window",
]

_CODE_FEATURES = [
    "add, delete, and search methods",
    "traversal and pretty-print",
    "generic type support",
    "iterator protocol",
    "serialization and deserialization",
    "thread-safe operations",
    "min/max operations",
    "range queries",
]


def _gen_code(num: int) -> list[tuple[str, str]]:
    """Generate CODE prompts."""
    samples = []
    for _ in range(num):
        action = random.choice(_CODE_ACTIONS)
        lang = random.choice(_CODE_LANGUAGES)
        task = random.choice(_CODE_TASKS)
        ds = random.choice(_CODE_DATASTRUCTURES)
        algo = random.choice(_CODE_ALGOS)
        features = random.choice(_CODE_FEATURES)
        template = random.choice(_CODE_TEMPLATES)
        prompt = template.format(
            action=action, lang=lang, task=task,
            ds=ds, algo=algo, features=features,
        )
        prompt = prompt.replace("  ", " ").strip()
        samples.append((prompt, "CODE"))
    return samples


# ═══════════════════════════════════════════════════════════════════
# MEMORY TEMPLATES (2)
# ═══════════════════════════════════════════════════════════════════

_MEMORY_TEMPLATES = [
    "Remember that {fact}",
    "Save this: {fact}",
    "Can you remember {fact}?",
    "I need to remember {fact}",
    "Remind me {fact}",
    "Don't forget {fact}",
    "Store this for later: {fact}",
    "Keep this in memory: {fact}",
    "Note to self: {fact}",
    "Make a note that {fact}",
    "What is my {info_item}?",
    "Tell me my {info_item}",
    "I forgot my {info_item}. What is it?",
    "What did I store about {topic}?",
    "Remind me to {action}",
    "Set a reminder to {action}",
    "Remind me at {time} to {action}",
    "I need to {action} later — remind me",
    "Save my {info_item} somewhere safe",
    "Can you hold this info: {fact}",
    "Memorize this for me: {fact}",
    "Put this in my notes: {fact}",
    "Remember my preference: {pref}",
    "Log this: {fact}",
    "Keep track of {tracking}",
    "Remember that I {past_event}",
    "What did I ask you to remember about {topic}?",
    "Check my saved info for {topic}",
    "I stored something earlier about {topic} — what was it?",
    "Recall my {info_item}",
]

_MEMORY_FACTS = [
    "my username is admin",
    "my email is user@example.com",
    "the wifi password is 'guest1234'",
    "my phone number is 555-1234",
    "my address is 123 Main St",
    "I like my coffee black",
    "I have a meeting at 3pm",
    "my favorite color is blue",
    "I prefer dark mode",
    "I live in New York",
    "my dog's name is Max",
    "my birthday is Jan 15",
    "I'm allergic to peanuts",
]

_MEMORY_INFO_ITEMS = [
    "password", "username", "email address", "phone number",
    "home address", "favorite food", "shirt size",
    "account number", "security question answer",
]

_MEMORY_TOPICS = [
    "work", "my account", "my preferences", "dinner plans",
    "the meeting", "my schedule", "my contacts",
]

_MEMORY_ACTIONS = [
    "call mom tomorrow",
    "buy milk on the way home",
    "pay the electricity bill by Friday",
    "pick up dry cleaning",
    "send the email to Sarah",
    "schedule a dentist appointment",
    "water the plants",
    "take out the trash",
    "finish the report by Monday",
    "RSVP to the wedding",
]

_MEMORY_TIMES = ["3pm", "5:30", "tonight", "tomorrow morning", "this weekend"]

_MEMORY_PREFS = [
    "I don't like spam emails",
    "notifications off after 10pm",
    "always use metric units",
    "show temperatures in Celsius",
]

_MEMORY_TRACKING = [
    "how many books I read this month",
    "my spending this week",
    "how many times I went to the gym",
    "my water intake",
]

_MEMORY_PAST = [
    "finished reading 'Dune' last week",
    "bought new running shoes",
    "set up the new router",
    "updated my phone settings",
    "changed my password yesterday",
]


def _gen_memory(num: int) -> list[tuple[str, str]]:
    """Generate MEMORY prompts."""
    samples = []
    for _ in range(num):
        template = random.choice(_MEMORY_TEMPLATES)

        placeholder_values = {
            "fact": random.choice(_MEMORY_FACTS),
            "info_item": random.choice(_MEMORY_INFO_ITEMS),
            "topic": random.choice(_MEMORY_TOPICS),
            "action": random.choice(_MEMORY_ACTIONS),
            "time": random.choice(_MEMORY_TIMES),
            "pref": random.choice(_MEMORY_PREFS),
            "tracking": random.choice(_MEMORY_TRACKING),
            "past_event": random.choice(_MEMORY_PAST),
        }

        # Try to fill all placeholders; skip if missing ones remain
        try:
            prompt = template.format(**placeholder_values)
        except KeyError:
            # Template uses a subset of available keys — try with just
            # the keys present in the template
            import re
            used_keys = set(re.findall(r"\{(\w+)\}", template))
            kwargs = {k: placeholder_values[k] for k in used_keys if k in placeholder_values}
            prompt = template.format(**kwargs)

        prompt = prompt.replace("  ", " ").strip()
        samples.append((prompt, "MEMORY"))
    return samples


# ═══════════════════════════════════════════════════════════════════
# CHAT TEMPLATES (3)
# ═══════════════════════════════════════════════════════════════════

_CHAT_TEMPLATES = [
    "Hi, how are you?",
    "Hello!",
    "What's up?",
    "Good morning!",
    "How's it going?",
    "What do you think about {topic}?",
    "Tell me something interesting",
    "I'm bored, entertain me",
    "What's the meaning of life?",
    "Do you like {thing}?",
    "Tell me a joke",
    "What's your favorite {thing}?",
    "How does {topic} work?",
    "Can you explain {topic} to me?",
    "Why is the sky blue?",
    "What's the weather like today?",
    "Who invented {topic}?",
    "Tell me about {topic}",
    "What's the capital of {country}?",
    "How old is {thing}?",
    "I'm feeling {emotion} today",
    "That's really cool!",
    "Thanks for your help",
    "I appreciate that",
    "What time is it?",
    "What day is it today?",
    "Where can I find {thing}?",
    "How do I get to {place}?",
    "What's a good {thing} restaurant?",
    "Do you know any good {thing}?",
    "What's the difference between {a} and {b}?",
    "Can you recommend a {thing}?",
    "I need advice about {topic}",
    "How does this {thing} work?",
    "What is {topic}?",
    "Nice to meet you!",
    "How was your day?",
    "What are you working on?",
    "I'm learning about {topic}",
    "That makes sense, thanks",
]

_CHAT_TOPICS = [
    "AI", "machine learning", "the future of technology",
    "space exploration", "climate change", "music",
    "movies", "books", "cooking", "photography",
    "philosophy", "history", "sports", "science",
    "education", "travel", "fitness", "gaming",
]

_CHAT_THINGS = [
    "movie genre", "type of music", "book", "hobby",
    "sport", "food", "animal", "color", "season",
    "holiday destination", "programming language",
]

_CHAT_COUNTRIES = [
    "France", "Japan", "Brazil", "India", "Australia",
    "Egypt", "Canada", "Germany", "Kenya", "Thailand",
]

_CHAT_EMOTIONS = ["happy", "sad", "curious", "excited", "tired", "confused", "grateful"]

_CHAT_PLACES = ["airport", "museum", "restaurant", "library", "park", "gym"]

_CHAT_AS = ["Python", "Java", "C", "HTML", "CSS", "SQL"]
_CHAT_BS = ["JavaScript", "C++", "C#", "XML", "SASS", "NoSQL"]


def _gen_chat(num: int) -> list[tuple[str, str]]:
    """Generate CHAT prompts."""
    samples = []
    for _ in range(num):
        template = random.choice(_CHAT_TEMPLATES)
        kwargs = {
            "topic": random.choice(_CHAT_TOPICS),
            "thing": random.choice(_CHAT_THINGS),
            "country": random.choice(_CHAT_COUNTRIES),
            "emotion": random.choice(_CHAT_EMOTIONS),
            "place": random.choice(_CHAT_PLACES),
            "a": random.choice(_CHAT_AS),
            "b": random.choice(_CHAT_BS),
        }
        import re
        used_keys = set(re.findall(r"\{(\w+)\}", template))
        used_kwargs = {k: kwargs[k] for k in used_keys if k in kwargs}
        if used_kwargs:
            prompt = template.format(**used_kwargs)
        else:
            prompt = template
        prompt = prompt.replace("  ", " ").strip()
        samples.append((prompt, "CHAT"))
    return samples


# ═══════════════════════════════════════════════════════════════════
# AMBIGUOUS / EDGE-CASE PROMPTS (Phase 2 — nuance training)
# ═══════════════════════════════════════════════════════════════════

_AMBIGUOUS_TEMPLATES = [
    # MATH or MEMORY — "I have 5 apples and eat 2"
    ("I have {a} apples and I eat {b}.", ["MATH", "MEMORY"]),
    ("I have {a} dollars, spend {b}. How much left?", ["MATH", "MEMORY"]),
    ("I bought {a} books and gave away {b}.", ["MATH", "MEMORY"]),
    # MEMORY or CHAT — personal question
    ("What's my name?", ["MEMORY", "CHAT"]),
    ("Do you remember my favorite color?", ["MEMORY", "CHAT"]),
    ("What did I say yesterday?", ["MEMORY", "CHAT"]),
    # MATH or CODE — "calculate" could be either
    ("Calculate the sum of all numbers from 1 to 100", ["MATH", "CODE"]),
    ("Write a program that calculates the average", ["CODE", "MATH"]),
    ("Find all prime numbers up to 100", ["MATH", "CODE"]),
    # CHAT or MEMORY
    ("Tell me something I told you before", ["MEMORY", "CHAT"]),
    ("Recall our previous conversation", ["MEMORY", "CHAT"]),
    # CODE or CHAT
    ("Explain recursion", ["CODE", "CHAT"]),
    ("What's a good way to learn programming?", ["CODE", "CHAT"]),
    ("Can you teach me Python?", ["CODE", "CHAT"]),
    # Multi-intent (Phase 3): MATH + MEMORY
    ("Calculate the tip and remind me to pay tomorrow", ["MATH", "MEMORY"]),
    ("Split the bill with friends and remember who paid", ["MATH", "MEMORY"]),
    ("Figure out the discount and save the price for later", ["MATH", "MEMORY"]),
    # Multi-intent: CODE + MEMORY
    ("Write a function and save it in my notes", ["CODE", "MEMORY"]),
    ("Store this code snippet and remind me to test it", ["CODE", "MEMORY"]),
    # Multi-intent: MATH + CODE
    ("Write a program to calculate fibonacci numbers", ["CODE", "MATH"]),
    ("Implement a sorting algorithm and count the comparisons", ["CODE", "MATH"]),
    # Triple intent
    ("Calculate my expenses, save the budget, and write a summary script", ["MATH", "MEMORY", "CODE"]),
    ("Remind me to test the code that calculates interest rates", ["MEMORY", "CODE", "MATH"]),
]


def _gen_ambiguous(num: int) -> list[tuple[str, list[str]]]:
    """Generate ambiguous prompts that span multiple intents."""
    samples = []
    for _ in range(num):
        template, intents = random.choice(_AMBIGUOUS_TEMPLATES)
        a = random.randint(2, 20)
        b = random.randint(1, a - 1) if a > 1 else 1
        try:
            prompt = template.format(a=a, b=b)
            samples.append((prompt, intents))
        except KeyError:
            samples.append((template, intents))
    return samples


# ═══════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════


def generate_router_dataset(
    total: int = 5000,
    ambiguous_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """Generate the full router training dataset.

    Args:
        total: Total number of samples to generate
        ambiguous_ratio: Fraction of samples that are ambiguous (multi-intent)
        seed: Random seed for reproducibility

    Returns:
        List of dicts: {"text": str, "intents": list[str], "label": int}
    """
    random.seed(seed)

    # Simple (single-intent) samples
    num_simple = int(total * (1 - ambiguous_ratio))
    num_math = num_simple // 4
    num_code = num_simple // 4
    num_memory = num_simple // 4
    num_chat = num_simple - num_math - num_code - num_memory  # use remainder

    simple: list[tuple[str, str]] = []
    simple += _gen_math(num_math)
    simple += _gen_code(num_code)
    simple += _gen_memory(num_memory)
    simple += _gen_chat(num_chat)

    # Ambiguous (multi-intent) samples
    num_ambiguous = total - len(simple)
    ambiguous = _gen_ambiguous(max(num_ambiguous, 0))

    # Build dataset
    dataset = []
    for text, intent_name in simple:
        dataset.append({
            "text": text,
            "intents": [intent_name],
            "label": INTENT_NAMES.index(intent_name),
        })

    for text, intent_names in ambiguous:
        dataset.append({
            "text": text,
            "intents": intent_names,
            "label": INTENT_NAMES.index(intent_names[0]),  # primary intent for Phase 1
        })

    # Shuffle
    random.shuffle(dataset)
    return dataset


def generate_multi_label_dataset(
    total: int = 5000,
    ambiguous_ratio: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    """Generate a dataset suitable for multi-label (sigmoid) training (Phase 3).

    Same as generate_router_dataset but with more ambiguous samples and
    multi-hot labels.

    Returns:
        List of dicts: {"text": str, "intents": list[str], "multi_hot": list[float]}
    """
    ds = generate_router_dataset(total, ambiguous_ratio=ambiguous_ratio, seed=seed)
    for item in ds:
        multi_hot = [0.0] * len(INTENT_NAMES)
        for intent_name in item["intents"]:
            multi_hot[INTENT_NAMES.index(intent_name)] = 1.0
        item["multi_hot"] = multi_hot
    return ds


def save_dataset(
    dataset: list[dict],
    path: str | Path = "data/router_dataset.json",
) -> Path:
    """Save dataset to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  [Dataset] Saved {len(dataset)} samples to {path}")
    return path


def load_dataset(path: str | Path = "data/router_dataset.json") -> list[dict]:
    """Load dataset from JSON file."""
    with open(path) as f:
        return json.load(f)


def print_dataset_stats(dataset: list[dict]) -> None:
    """Print dataset statistics."""
    from collections import Counter

    intent_counts: Counter = Counter()
    ambiguity_counts: Counter = Counter()
    avg_len = 0

    for item in dataset:
        for intent in item["intents"]:
            intent_counts[intent] += 1
        ambiguity_counts[len(item["intents"])] += 1
        avg_len += len(item["text"])

    avg_len /= max(len(dataset), 1)

    print(f"  Dataset: {len(dataset)} samples")
    print(f"  Avg prompt length: {avg_len:.1f} chars")
    print(f"  Intent distribution: {dict(intent_counts)}")
    print(f"  Ambiguity: {dict(ambiguity_counts)} (1=single, 2+=multi)")


# ═══════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Generating Router Dataset ===\n")

    ds = generate_router_dataset(total=5000, ambiguous_ratio=0.1)
    print_dataset_stats(ds)

    # Show some samples
    print("\n  --- Sample prompts ---")
    for item in ds[:8]:
        intents = ", ".join(item["intents"])
        print(f'  [{intents:20s}] {item["text"][:80]}')

    # Save
    save_dataset(ds)
