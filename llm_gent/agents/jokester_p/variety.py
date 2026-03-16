"""Variety checking for joke generation.

Ensures structural diversity by tracking first n-grams and
using Levenshtein distance to reject similar jokes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from appinfra import DotDict
    from appinfra.log import Logger


@dataclass
class VarietyCheck:
    """Result of variety checking."""

    is_varied: bool
    reason: str | None = None


class VarietyChecker:
    """Checks joke variety using n-gram and Levenshtein distance.

    Tracks recent joke openings and full texts to ensure structural
    and content diversity.
    """

    def __init__(self, lg: Logger, config: DotDict) -> None:
        """Initialize variety checker.

        Args:
            lg: Logger instance.
            config: Variety config with ngram and levenshtein settings.
        """
        self._lg = lg

        ngram_config = config.get("ngram", {})
        self._ngram_enabled: bool = ngram_config.get("enabled", False)
        self._ngram_n: int = ngram_config.get("n", 3)
        self._ngram_history: deque[str] = deque(maxlen=ngram_config.get("history", 20))

        lev_config = config.get("levenshtein", {})
        self._lev_enabled: bool = lev_config.get("enabled", False)
        self._lev_threshold: float = lev_config.get("threshold", 0.7)
        self._lev_history: deque[str] = deque(maxlen=lev_config.get("history", 10))

        self._enabled: bool = self._ngram_enabled or self._lev_enabled

    @property
    def enabled(self) -> bool:
        """Whether variety checking is enabled."""
        return self._enabled

    def check(self, joke_text: str) -> VarietyCheck:
        """Check if joke has sufficient variety from recent jokes.

        Args:
            joke_text: The joke text to check.

        Returns:
            VarietyCheck with is_varied and reason if rejected.
        """
        if not self._enabled:
            return VarietyCheck(is_varied=True)

        # Check n-gram variety
        ngram_result = self._check_ngram(joke_text)
        if not ngram_result.is_varied:
            return ngram_result

        # Check Levenshtein distance
        lev_result = self._check_levenshtein(joke_text)
        if not lev_result.is_varied:
            return lev_result

        return VarietyCheck(is_varied=True)

    def record(self, joke_text: str) -> None:
        """Record a successful joke for future variety checks.

        Args:
            joke_text: The joke text to record.
        """
        if self._ngram_enabled:
            opening = self._extract_ngram(joke_text)
            if opening:
                self._ngram_history.append(opening)

        if self._lev_enabled:
            self._lev_history.append(joke_text)

    def get_blocked_openings(self) -> list[str]:
        """Get list of n-gram openings to avoid in prompt.

        Returns:
            List of opening phrases that would be rejected.
        """
        if not self._ngram_enabled:
            return []
        return list(self._ngram_history)

    def _check_ngram(self, joke_text: str) -> VarietyCheck:
        """Check if joke opening is sufficiently different from recent jokes."""
        if not self._ngram_enabled or not self._ngram_history:
            return VarietyCheck(is_varied=True)

        opening = self._extract_ngram(joke_text)
        if not opening:
            return VarietyCheck(is_varied=True)

        if opening in self._ngram_history:
            self._lg.debug(
                "joke rejected by ngram variety",
                extra={"opening": opening, "history_size": len(self._ngram_history)},
            )
            return VarietyCheck(is_varied=False, reason=f"opening '{opening}' used recently")

        return VarietyCheck(is_varied=True)

    def _check_levenshtein(self, joke_text: str) -> VarietyCheck:
        """Check if joke is sufficiently different from recent jokes."""
        if not self._lev_enabled or not self._lev_history:
            return VarietyCheck(is_varied=True)

        from rapidfuzz import fuzz

        if self._lev_threshold <= 0:
            return VarietyCheck(is_varied=True)

        for recent in self._lev_history:
            similarity = fuzz.ratio(joke_text.lower(), recent.lower()) / 100.0
            if similarity >= self._lev_threshold:
                self._lg.debug(
                    "joke rejected by levenshtein variety",
                    extra={"similarity": round(similarity, 2), "threshold": self._lev_threshold},
                )
                return VarietyCheck(
                    is_varied=False,
                    reason=f"too similar to recent joke ({similarity:.0%})",
                )

        return VarietyCheck(is_varied=True)

    def _extract_ngram(self, text: str) -> str:
        """Extract first n words as opening pattern."""
        words = text.split()[: self._ngram_n]
        return " ".join(words).lower() if words else ""
