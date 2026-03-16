"""History tracking for RAG-based novelty avoidance.

Tracks joke history and provides different strategies for selecting
which jokes to include in the prompt to help the model avoid repeats.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from appinfra.log import Logger


class HistoryMode(Enum):
    """Mode for selecting jokes to include in prompt."""

    RECENT = "recent"  # Most recent jokes
    MOST_FREQUENT = "most_frequent"  # Most frequently hit


@dataclass
class JokeRecord:
    """A joke to avoid."""

    joke: str
    fact_id: int | None = None


class JokeHistory:
    """Tracks joke history and selects which to include in prompts.

    Supports different modes:
    - recent: Return the N most recent jokes
    - most_frequent: Return the N most frequently hit jokes

    The frequency mode helps when the model keeps hitting the same classics
    repeatedly - showing "you've tried this 47 times" is more effective than
    just showing the last 5 attempts.
    """

    def __init__(
        self,
        lg: Logger,
        mode: HistoryMode = HistoryMode.RECENT,
        history_size: int = 100,
        attach_count: int = 5,
    ) -> None:
        """Initialize joke history.

        Args:
            lg: Logger instance.
            mode: Strategy for selecting jokes to attach.
            history_size: Rolling window size for tracking jokes.
            attach_count: Number of jokes to attach to prompt.
        """
        self._lg = lg
        self._mode = mode
        self._attach_count = attach_count
        self._history: deque[JokeRecord] = deque(maxlen=history_size)
        self._frequency: Counter[str] = Counter()

    def record(self, joke: str, fact_id: int | None = None) -> None:
        """Record a joke to avoid.

        Args:
            joke: The joke text.
            fact_id: Fact ID of the joke (if known).
        """
        # Decrement frequency of evicted entry when history is full
        if len(self._history) == self._history.maxlen:
            evicted = self._history[0]
            self._frequency[evicted.joke] -= 1
            if self._frequency[evicted.joke] <= 0:
                del self._frequency[evicted.joke]

        self._history.append(JokeRecord(joke=joke, fact_id=fact_id))
        self._frequency[joke] += 1

    def contains(self, joke: str) -> bool:
        """Check if exact joke text exists in history.

        Args:
            joke: The joke text to check.

        Returns:
            True if joke exists in history.
        """
        return joke in self._frequency

    def get_for_prompt(self) -> list[tuple[JokeRecord, int]]:
        """Get jokes to include in prompt.

        Returns:
            List of (JokeRecord, count) tuples.
            For recent mode, count is always 1.
            For frequency mode, count is how many times that joke was hit.
        """
        if self._mode == HistoryMode.RECENT:
            result = self._get_recent()
        elif self._mode == HistoryMode.MOST_FREQUENT:
            result = self._get_most_frequent()
        else:
            return []

        self._log_jokes(result)
        return result

    def _log_jokes(self, jokes: list[tuple[JokeRecord, int]]) -> None:
        """Log each joke being attached to prompt."""
        for record, count in jokes:
            self._lg.trace(
                "attaching to prompt",
                extra={
                    "freq": count,
                    "fact": record.fact_id,
                    "joke": record.joke[:100] if record.joke else None,
                },
            )

    def _get_recent(self) -> list[tuple[JokeRecord, int]]:
        """Get most recent jokes."""
        recent = list(self._history)[-self._attach_count :]
        return [(r, 1) for r in recent]

    def _get_most_frequent(self) -> list[tuple[JokeRecord, int]]:
        """Get most frequently hit jokes."""
        if not self._frequency:
            return self._get_recent()

        top_targets = self._frequency.most_common(self._attach_count)
        result: list[tuple[JokeRecord, int]] = []
        for target, count in top_targets:
            for record in reversed(self._history):
                if record.joke == target:
                    result.append((record, count))
                    break
        return result

    @classmethod
    def from_config(cls, lg: Logger, config: dict[str, Any]) -> JokeHistory:
        """Create history from config dict.

        Args:
            lg: Logger instance.
            config: Dict with mode, history, attach keys.

        Returns:
            Configured JokeHistory.
        """
        mode_str = config.get("mode", "recent")
        # Map old config values to new enum
        if mode_str == "avoid_most_frequent":
            mode_str = "most_frequent"
        try:
            mode = HistoryMode(mode_str)
        except ValueError:
            mode = HistoryMode.RECENT

        return cls(
            lg=lg,
            mode=mode,
            history_size=config.get("history", 100),
            attach_count=config.get("attach", 5),
        )
