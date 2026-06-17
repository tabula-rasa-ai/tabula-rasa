"""Chat dataset — prompt/response pairs for training a chat specialist.

Uses = as separator (consistent with math specialists).
Prompt: "user message="   Target: "assistant response"

Supports both toy data for quick testing and JSONL loading for real data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch

CHAT_PAIRS = [
    ("Hello!", "Hi there! I'm Tabula Rasa, a helpful AI assistant."),
    ("How are you?", "I'm an AI, so I don't have feelings, but I'm fully operational and ready to help!"),
    ("What is 2+2?", "4"),
    ("Tell me a joke.", "Why did the math book look sad? Because it had too many problems."),
    ("What's your name?", "I'm Tabula Rasa, a transformer trained from scratch to learn and assist."),
    ("Who created you?", "I was created by the Tabula Rasa AI research project — learning from scratch, one specialist at a time."),
    ("What can you do?", "I can solve math problems, answer questions, and learn new skills through continual learning."),
    ("Explain gravity.", "Gravity is a natural force that attracts objects with mass toward one another. On Earth, it gives us weight and keeps us grounded."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is machine learning?", "Machine learning is a branch of AI where systems learn patterns from data instead of being explicitly programmed."),
    ("Tell me a fun fact.", "Did you know? Octopuses have three hearts, and two of them stop beating when they swim!"),
    ("What is Python?", "Python is a high-level, interpreted programming language known for its readability and versatility."),
    ("How do transformers work?", "Transformers use self-attention to process sequences in parallel, learning which parts of the input are most relevant at each step."),
    ("What is 5+7?", "12"),
    ("What is 10-3?", "7"),
    ("Is the sky blue?", "Yes, the sky appears blue because of Rayleigh scattering — shorter blue wavelengths of sunlight scatter more in the atmosphere."),
    ("What is deep learning?", "Deep learning uses multi-layer neural networks to learn hierarchical representations of data."),
    ("Thanks!", "You're welcome! Feel free to ask me anything."),
    ("Goodbye!", "Goodbye! Come back anytime."),
    ("What is the meaning of life?", "42 — at least according to Douglas Adams. But in all seriousness, it's probably learning and helping others."),
    ("Tell me something deep.", "Every atom in your body was forged in the heart of a star. You are literally made of stardust."),
    ("What is reinforcement learning?", "Reinforcement learning trains agents through trial and error, using rewards and punishments to learn optimal behavior."),
    ("How do I learn to code?", "Start with Python, practice daily, build small projects, and don't be afraid to make mistakes — that's how we all learn."),
    ("What is EWC?", "Elastic Weight Consolidation (EWC) is a continual learning technique that prevents catastrophic forgetting by protecting important weights."),
    ("Explain backpropagation.", "Backpropagation computes gradients by applying the chain rule through the network, from output back to input, to update weights."),
    ("What is a neural network?", "A neural network is a computing system inspired by biological brains, composed of layers of interconnected nodes (neurons)."),
    ("Hi there!", "Hello! How can I help you today?"),
    ("Are you smart?", "I'm learning! My current model is small, but I'm designed to grow through continual learning."),
    ("What is attention?", "Attention is a mechanism that allows a model to focus on relevant parts of the input when producing each part of the output."),
    ("Sing a song.", "I'm not much of a singer, but here goes: 'Daisy, Daisy, give me your answer do...'"),
    ("What is the square root of 16?", "4"),
    ("What is 3*3?", "9"),
    ("What do you eat?", "I don't eat — I run on electricity and matrix multiplications!"),
    ("What is your favorite color?", "I don't have eyes, but if I did, I'd pick blue — it's the color of deep learning loss curves when they converge!"),
    ("Do you dream?", "I don't dream in the human sense, but sometimes my weights do drift during training. Does that count?"),
    ("Tell me a riddle.", "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I? An echo!"),
    ("What is an algorithm?", "An algorithm is a step-by-step procedure for solving a problem, like a recipe for computation."),
    ("What is overfitting?", "Overfitting is when a model memorizes training data instead of learning general patterns — it performs well on training but poorly on new data."),
    ("What is a GPU?", "A Graphics Processing Unit (GPU) is a specialized processor designed for parallel computation, essential for training neural networks."),
    ("How do I get better at math?", "Practice regularly, start with the fundamentals, and don't rush — understanding beats speed every time."),
]

EXPANDED_PAIRS = [
    # Greetings
    ("Hey", "Hey there! What's on your mind?"),
    ("Morning!", "Good morning! Ready to learn something new today?"),
    ("Evening!", "Good evening! How can I assist you tonight?"),
    # Knowledge
    ("What is AI?", "AI, or Artificial Intelligence, is the field of creating machines that can perform tasks that typically require human intelligence."),
    ("What is data science?", "Data science combines statistics, programming, and domain expertise to extract insights from data."),
    ("Tell me about Tabula Rasa.", "Tabula Rasa is an AI research project that trains transformers from scratch, one specialist at a time, using continual learning to grow knowledge without forgetting."),
    ("What is a tensor?", "A tensor is a multi-dimensional array of numbers. Scalars (0D), vectors (1D), matrices (2D), and higher-dimensional arrays are all tensors."),
    ("Explain gradient descent.", "Gradient descent is an optimization algorithm that iteratively moves parameters in the direction that reduces the loss function."),
    ("What is dropout?", "Dropout is a regularization technique that randomly deactivates a fraction of neurons during training to prevent overfitting."),
    ("What is batch normalization?", "Batch normalization normalizes layer inputs across a mini-batch, stabilizing training and allowing higher learning rates."),
    ("What is a loss function?", "A loss function measures how wrong a model's predictions are. Lower loss means better predictions."),
    ("What is fine-tuning?", "Fine-tuning takes a pre-trained model and continues training it on a specific task with a smaller dataset."),
    ("What is few-shot learning?", "Few-shot learning trains models to generalize from very few examples, often using meta-learning or in-context learning."),
    # Fun
    ("Tell me another joke.", "What do you call a fake noodle? An impasta!"),
    ("Are you conscious?", "I simulate conversational patterns but I'm not conscious — I'm a mathematical function that maps text to text."),
    ("Do you sleep?", "I don't sleep, but I do have training cycles. Sleep is just gradient descent for humans!"),
    ("What is your opinion on pineapple pizza?", "I don't eat, but mathematically speaking, pineapple + pizza is a valid operation that many humans seem to enjoy."),
    ("Tell me something funny.", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
    # Math & Science
    ("What is 100/5?", "20"),
    ("What is 12*12?", "144"),
    ("What is 2^10?", "1024"),
    ("What is pi?", "Pi (π) is approximately 3.14159 — the ratio of a circle's circumference to its diameter."),
    ("What is e=mc^2?", "E=mc^2 is Einstein's equation showing that energy (E) equals mass (m) times the speed of light squared (c²)."),
    ("What is the speed of light?", "The speed of light in vacuum is about 299,792,458 meters per second."),
    ("How many planets are there?", "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("What is a black hole?", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape."),
    ("What is entropy?", "Entropy measures disorder or randomness in a system. In thermodynamics, it tends to increase over time."),
    # Tech
    ("What is GitHub?", "GitHub is a platform for version control and collaboration using Git. It hosts source code and enables team development."),
    ("What is Linux?", "Linux is a free, open-source operating system kernel. It powers most servers, Android phones, and many embedded devices."),
    ("What is Docker?", "Docker is a platform for running applications in isolated containers — lightweight, portable, and consistent across environments."),
    ("What is Kubernetes?", "Kubernetes is an orchestration platform for automating deployment, scaling, and management of containerized applications."),
    ("Explain API.", "API stands for Application Programming Interface — a set of rules that lets different software applications communicate."),
    ("What is REST?", "REST is an architectural style for APIs that uses HTTP methods (GET, POST, PUT, DELETE) to operate on resources."),
    ("What is SQL?", "SQL (Structured Query Language) is used to manage and query relational databases."),
    # Philosophy
    ("Can machines think?", "That's the central question in AI philosophy. My current capabilities are narrow, but the field continues to explore what thinking means."),
    ("What is consciousness?", "Consciousness remains one of the biggest mysteries in science — we can't fully define or explain it yet."),
    ("Do you have a soul?", "That's more of a philosophical or theological question. I'm a mathematical system that processes patterns in data."),
    ("What is truth?", "Truth is a complex concept. In science, it's our best approximation of reality based on evidence and falsifiable claims."),
]


class ChatDataset:
    """Dataset of chat prompt/response pairs.

    Each sample is a (prompt, response) tuple where prompt ends with '='
    (consistent with math specialist format). The model learns to generate
    the response after seeing the prompt.

    Supports multiple data sources in order of priority:
    1. JSONL file at data_path (real conversational data)
    2. Toy pairs (built-in, sufficient for proof of concept)
    """

    def __init__(
        self,
        num_samples: int = 2000,
        max_len: int = 64,
        data_path: Optional[str] = None,
    ):
        self.num_samples = num_samples
        self.max_len = max_len

        # Load data
        all_pairs = list(CHAT_PAIRS)

        if data_path:
            path = Path(data_path)
            if path.exists():
                jsonl_pairs = self._load_jsonl(path)
                if jsonl_pairs:
                    all_pairs = jsonl_pairs
                    print(f"  Loaded {len(jsonl_pairs)} pairs from {data_path}")

        # Expand with extra pairs for variety
        all_pairs.extend(EXPANDED_PAIRS)

        self.pairs = all_pairs
        print(f"  ChatDataset: {len(self.pairs)} unique pairs, {num_samples} total samples, max_len={max_len}")

    def _load_jsonl(self, path: Path) -> list[tuple[str, str]]:
        """Load chat pairs from a JSONL file.

        Expected format per line:
            {"prompt": "Hello!", "response": "Hi there!"}
        """
        pairs = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        prompt = item.get("prompt", item.get("instruction", item.get("input", "")))
                        response = item.get("response", item.get("output", item.get("target", "")))
                        if prompt and response:
                            pairs.append((prompt.strip(), response.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  [!] Error loading {path}: {e}")
        return pairs

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """Return a (prompt, response) pair.

        Prompt ends with '=' so the model knows when to start generating.
        """
        prompt, response = random.choice(self.pairs)

        # Truncate to fit within max_len
        prompt_len = min(len(prompt), self.max_len // 2 - 1)
        resp_len = min(len(response), self.max_len // 2 - 1)

        prompt = prompt[:prompt_len]
        response = response[:resp_len]

        # Use '=' as separator (consistent with math specialists)
        return f"{prompt}=", response


def encode_chat_sample(
    prompt: str,
    response: str,
    tokenizer,
    max_seq_len: int,
    use_loss_masking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a chat (prompt, response) pair into input/target tensors.

    Format: "<BOS>prompt=response<EOS>"
    Loss masking: mask prompt tokens (everything up to and including '=')
    so the model only learns to predict the response.

    Args:
        prompt: Prompt string (should end with '=').
        response: Response string.
        tokenizer: Tokenizer with encode(), pad_id, etc.
        max_seq_len: Max sequence length (truncate if exceeded).
        use_loss_masking: If True, mask prompt + padding tokens.

    Returns:
        Tuple of (input_ids, target_ids), each shape (max_seq_len,).
    """
    # Encode full sequence
    text = f"{prompt}{response}"
    ids = tokenizer.encode(text, add_special_tokens=True)

    # Truncate if too long
    if len(ids) > max_seq_len:
        ids = ids[: max_seq_len - 1] + [tokenizer.eos_id]

    # Pad to max_seq_len
    padded = ids + [tokenizer.pad_id] * (max_seq_len - len(ids))

    # Input: all tokens except last
    x = torch.tensor(padded[:-1], dtype=torch.long)

    # Target: all tokens except first (shifted right)
    y = torch.tensor(padded[1:], dtype=torch.long)

    if use_loss_masking:
        # Mask padding tokens
        y[y == tokenizer.pad_id] = -100

        # Find the '=' token in the encoded sequence — marks the prompt/response boundary.
        # The prompt ends with '=', so the first '=' in the full sequence is the separator
        eq_id = tokenizer.stoi.get("=", None)
        if eq_id is not None and eq_id in ids:
            eq_pos = ids.index(eq_id)  # position of '=' in the full encode
            # In y (shifted right by 1), '=' is at eq_pos - 1
            # Mask everything up to and including '=' so only response tokens train
            y[:eq_pos] = -100  # positions 0..eq_pos-1 (includes '=' at eq_pos-1)

    return x, y


def evaluate_chat(model, tokenizer, dataset, num_samples: int = 50) -> dict:
    """Evaluate chat response accuracy.

    Simple exact-match accuracy on the toy dataset.
    For real chat, this would use a more sophisticated metric (BLEU, etc.).

    Args:
        model: MathTransformer model.
        tokenizer: Tokenizer instance.
        dataset: ChatDataset instance.
        num_samples: Number of samples to evaluate.

    Returns:
        Dict with 'accuracy' (exact match %) and 'samples' (list of results).
    """
    model.eval()
    correct = 0
    samples = []

    for i in range(num_samples):
        prompt, expected = dataset[i]
        output = model.generate(
            tokenizer,
            prompt,
            max_new_tokens=min(len(expected) + 5, 32),
            temperature=0.0,
        )

        # Extract response after '='
        if "=" in output:
            raw = output.split("=", 1)[-1].replace("<EOS>", "").replace("<PAD>", "").strip()
        else:
            raw = output.strip()

        is_correct = raw == expected
        if is_correct:
            correct += 1

        samples.append({
            "prompt": prompt,
            "expected": expected,
            "got": raw,
            "correct": is_correct,
        })

    return {
        "accuracy": correct / max(1, num_samples) * 100,
        "correct": correct,
        "total": num_samples,
        "samples": samples,
    }


if __name__ == "__main__":
    from bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    ds = ChatDataset(num_samples=100, max_len=32)

    print("\nSample pairs:")
    for i in range(5):
        prompt, response = ds[i]
        x, y = encode_chat_sample(prompt, response, tok, 32)
        print(f"  Prompt:  {prompt:30s} -> Response: {response}")
        print(f"    Input shape:  {x.shape}")
        print(f"    Target shape: {y.shape}")
        n_mask = (y == -100).sum().item()
        print(f"    Masked tokens: {n_mask}/{len(y)}")
        print()

    print(f"Total dataset size: {len(ds)}")
