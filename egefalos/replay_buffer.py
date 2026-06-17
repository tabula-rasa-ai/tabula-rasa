"""Experience Replay Buffer for Tabula Rasa.

When a specialist trains on a new user query, it mixes in old data
from the replay buffer to prevent catastrophic forgetting.
"""

import random


class ReplayBuffer:
    """Stores past (prompt, answer) pairs per intent for replay during training.

    When a specialist trains on new data, the replay buffer allows mixing in
    a sample of previously-seen examples to maintain stability and prevent
    forgetting of earlier-learned patterns.
    """

    def __init__(self, max_size: int = 200):
        """Initialise the replay buffer.

        Args:
            max_size: Maximum number of (prompt, answer) pairs to keep per
                intent.  Older entries beyond this limit are dropped.
        """
        self.buffer: dict[str, list[tuple[str, str]]] = {}
        self.max_size = max_size

    def add(self, intent: str, prompt: str, answer: str) -> None:
        """Add a new experience to the buffer.

        Args:
            intent: The specialist/skill intent name.
            prompt: The user's question/input.
            answer: The specialist's output/target.
        """
        if intent not in self.buffer:
            self.buffer[intent] = []
        self.buffer[intent].append((prompt, answer))
        # Trim to max size (keep newest)
        if len(self.buffer[intent]) > self.max_size:
            self.buffer[intent] = self.buffer[intent][-self.max_size:]

    def sample(self, intent: str, k: int = 16) -> list[tuple[str, str]]:
        """Sample *k* random experiences from the buffer for an intent.

        Args:
            intent: The intent to sample from.
            k: Maximum number of samples to return.

        Returns:
            A list of (prompt, answer) tuples drawn uniformly at random.
            Returns an empty list if the intent has no data.
        """
        if intent not in self.buffer or not self.buffer[intent]:
            return []
        k = min(k, len(self.buffer[intent]))
        return random.sample(self.buffer[intent], k)

    def mix(
        self,
        intent: str,
        new_pairs: list[tuple[str, str]],
        replay_ratio: float = 0.5,
    ) -> list[tuple[str, str]]:
        """Mix new training pairs with replayed old pairs.

        If *replay_ratio* is 0.5 and *new_pairs* has 10 items, 5 replay
        items are sampled and interleaved with the new items::

            new[0], replay[0], new[1], replay[1], ...

        Args:
            intent: The intent to replay from.
            new_pairs: The newly-generated training pairs.
            replay_ratio: Fraction of ``len(new_pairs)`` to sample for replay.

        Returns:
            Mixed list of (prompt, answer) tuples.  Total length is
            approximately ``len(new_pairs) * (1 + replay_ratio)``, or
            ``len(new_pairs)`` if the buffer is empty for this intent.
        """
        replay = self.sample(intent, k=int(len(new_pairs) * replay_ratio))
        if not replay:
            return new_pairs

        # Interleave: new, replay, new, replay, ...
        mixed: list[tuple[str, str]] = []
        for i in range(max(len(new_pairs), len(replay))):
            if i < len(new_pairs):
                mixed.append(new_pairs[i])
            if i < len(replay):
                mixed.append(replay[i])
        return mixed

    def load_from_dataset(
        self, intent: str, dataset: list[tuple[str, str]]
    ) -> None:
        """Pre-populate the buffer from an existing dataset.

        Useful during startup so that the first training run already has a
        pool of replay candidates.

        Args:
            intent: The intent key.
            dataset: List of (prompt, answer) tuples to seed the buffer.
        """
        self.buffer[intent] = list(dataset)
        if len(self.buffer[intent]) > self.max_size:
            self.buffer[intent] = self.buffer[intent][-self.max_size:]
