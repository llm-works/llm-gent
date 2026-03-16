"""Joke generation with novelty checking and retry logic.

Handles the complete joke generation cycle including LLM calls,
novelty validation, denylist filtering, and retry logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_infer.client.types import AdapterInfo
from pydantic import AliasChoices, BaseModel, Field

from .history import JokeHistory, JokeRecord
from .novelty import NoveltyCheck, NoveltyChecker
from .variety import VarietyChecker


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ...core.traits.builtin.directive import DirectiveTrait
    from ...core.traits.builtin.llm import LLMTrait


class Joke(BaseModel):
    """Structured joke output."""

    text: str = Field(min_length=1, validation_alias=AliasChoices("text", "joke"))
    style: str = Field(default="unknown", min_length=1)  # pun, one-liner, etc.


@dataclass
class GenerationAttempt:
    """Result of a joke generation cycle with all metadata."""

    joke: Joke | None
    run_attempts: int
    cumulative_attempts: int
    model_name: str
    adapter: AdapterInfo | None  # Full adapter info from LLM response
    novelty: NoveltyCheck | None = None

    @property
    def success(self) -> bool:
        """Whether a novel joke was generated."""
        return self.joke is not None

    @property
    def adapter_fallback(self) -> bool:
        """Whether adapter fell back to base model."""
        return self.adapter.fallback if self.adapter else False

    @property
    def max_similarity(self) -> float:
        """Similarity to closest existing joke (0.0-1.0)."""
        return self.novelty.max_similarity if self.novelty else 0.0

    @property
    def similar_joke(self) -> str | None:
        """The closest existing joke text."""
        return self.novelty.similar_joke if self.novelty else None


class JokeGenerator:
    """Generates novel jokes with retry logic and validation.

    Handles the complete generation cycle:
    1. Build prompt with context and constraints
    2. Call LLM for structured joke output
    3. Validate against denylist
    4. Check novelty via embedding similarity
    5. Retry on failure up to max_retries
    """

    def __init__(
        self,
        lg: Logger,
        llm_trait: LLMTrait,
        novelty_checker: NoveltyChecker,
        variety_checker: VarietyChecker | None = None,
        directive_trait: DirectiveTrait | None = None,
        max_retries: int = 3,
        denylist: list[str] | None = None,
        joke_history: JokeHistory | None = None,
    ) -> None:
        """Initialize joke generator.

        Args:
            lg: Logger instance.
            llm_trait: LLM trait for generation.
            novelty_checker: Checker for joke novelty.
            variety_checker: Checker for structural variety.
            directive_trait: Optional directive trait for system prompt.
            max_retries: Maximum retry attempts (default: 3).
            denylist: Words/phrases to filter (case-insensitive).
            joke_history: Tracker for joke history (default: recent mode).
        """
        self._lg = lg
        self._llm = llm_trait
        self._novelty = novelty_checker
        self._variety = variety_checker
        self._directive = directive_trait
        self._max_retries = max_retries
        self._denylist = [term.lower() for term in (denylist or [])]
        self._cumulative_attempts = 0
        self._history = joke_history or JokeHistory(lg)

    def generate(self, context: list[str]) -> GenerationAttempt:
        """Generate a novel joke with retry loop.

        Args:
            context: Recent successful jokes for style inspiration.

        Returns:
            GenerationAttempt with result and metadata.
        """
        max_attempts = self._max_retries + 1

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._lg.debug("retrying joke generation...", extra={"attempt": attempt})

            result = self._try_single_attempt(context, attempt)
            if result is not None:
                return self._success_attempt(attempt, *result)

        self._lg.debug("failed to generate novel joke", extra={"attempts": max_attempts})
        return GenerationAttempt(
            joke=None,
            run_attempts=max_attempts,
            cumulative_attempts=self._cumulative_attempts,
            model_name="unknown",
            adapter=None,
        )

    def record_success(self, joke_text: str, fact_id: int) -> None:
        """Record successful joke so future generations avoid it."""
        self._history.record(joke_text, fact_id)

    def get_recent_jokes(self, limit: int) -> list[str]:
        """Get recent jokes from in-memory history for style inspiration.

        Args:
            limit: Maximum number of jokes to return.

        Returns:
            List of recent joke texts.
        """
        recent = list(self._history._history)[-limit:]
        return [r.joke for r in recent if r.joke]

    def check_novelty(
        self, joke: Joke, model_name: str, adapter: AdapterInfo | None
    ) -> GenerationAttempt:
        """Check novelty for a pre-generated joke.

        Args:
            joke: The joke to check.
            model_name: Model that generated the joke.
            adapter: Adapter info if applicable.

        Returns:
            GenerationAttempt with novelty result.
        """
        # Fast exact-match check against in-memory history first
        if self._history.contains(joke.text):
            self._history.record(joke.text)  # Bump frequency for prompt avoidance
            novelty = NoveltyCheck(is_novel=False, max_similarity=1.0, similar_joke=joke.text)
            return self._make_attempt(None, model_name, adapter, novelty)

        novelty = self._novelty.check(joke.text, current_model=model_name, current_adapter=adapter)
        result_joke = joke if novelty.is_novel else None
        return self._make_attempt(result_joke, model_name, adapter, novelty)

    def _make_attempt(
        self,
        joke: Joke | None,
        model_name: str,
        adapter: AdapterInfo | None,
        novelty: NoveltyCheck,
    ) -> GenerationAttempt:
        """Build a GenerationAttempt for novelty check results."""
        return GenerationAttempt(
            joke=joke,
            run_attempts=1,
            cumulative_attempts=1,
            model_name=model_name,
            adapter=adapter,
            novelty=novelty,
        )

    def _success_attempt(
        self,
        attempt: int,
        joke: Joke,
        model: str,
        adapter: AdapterInfo | None,
        novelty: NoveltyCheck | None,
    ) -> GenerationAttempt:
        """Build successful generation attempt."""
        cumulative = self._cumulative_attempts
        self._cumulative_attempts = 0
        return GenerationAttempt(
            joke=joke,
            run_attempts=attempt,
            cumulative_attempts=cumulative,
            model_name=model,
            adapter=adapter,
            novelty=novelty,
        )

    def _try_single_attempt(
        self, context: list[str], attempt: int
    ) -> tuple[Joke, str, AdapterInfo | None, NoveltyCheck | None] | None:
        """Try to generate a single novel joke."""
        self._cumulative_attempts += 1
        retry_feedback = "" if attempt == 1 else "Try a completely different style."

        joke, model_name, adapter = self._call_llm(context, retry_feedback)

        if joke is None:
            self._lg.warning("LLM failed to generate joke", extra={"attempt": attempt})
            return None

        return self._validate(joke, model_name, adapter, attempt)

    def _call_llm(
        self, context: list[str], retry_feedback: str
    ) -> tuple[Joke | None, str, AdapterInfo | None]:
        """Generate joke via LLM with structured output."""
        import appinfra.time
        from llm_infer.client import BackendRequestError, BackendTimeoutError

        from ...core.llm.types import Message

        prompt = self._build_prompt(context, retry_feedback)
        self._lg.debug("sending LLM request...")
        start_t = appinfra.time.start()
        try:
            result = self._llm.complete([Message(role="user", content=prompt)], output_schema=Joke)
        except (BackendTimeoutError, BackendRequestError) as e:
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self._lg.error(
                "LLM request failed",
                extra={
                    "after": appinfra.time.since(start_t),
                    "prompt": prompt_preview,
                    "error": str(e),
                },
            )
            return None, "unknown", None
        self._lg.debug("LLM request completed", extra={"after": appinfra.time.since(start_t)})

        if result.parsed is None:
            self._lg.warning("LLM failed to generate structured joke")
            return None, result.model, result.adapter

        return result.parsed, result.model, result.adapter

    async def generate_async(self, context: list[str]) -> GenerationAttempt:
        """Generate a novel joke with retry loop (async version).

        Same as generate() but uses async I/O for concurrent requests.
        Use with asyncio.gather() for parallel generation.

        Args:
            context: Recent successful jokes for style inspiration.

        Returns:
            GenerationAttempt with result and metadata.
        """
        max_attempts = self._max_retries + 1

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._lg.debug("retrying joke generation...", extra={"attempt": attempt})

            result = await self._try_single_attempt_async(context, attempt)
            if result is not None:
                return self._success_attempt(attempt, *result)

        self._lg.debug("failed to generate novel joke", extra={"attempts": max_attempts})
        return GenerationAttempt(
            joke=None,
            run_attempts=max_attempts,
            cumulative_attempts=self._cumulative_attempts,
            model_name="unknown",
            adapter=None,
        )

    async def _try_single_attempt_async(
        self, context: list[str], attempt: int
    ) -> tuple[Joke, str, AdapterInfo | None, NoveltyCheck | None] | None:
        """Try to generate a single novel joke (async version).

        Note: _cumulative_attempts is incremented without synchronization during
        parallel execution. This is intentional - the counter is approximate and
        used only for metrics/logging, not for correctness.
        """
        self._cumulative_attempts += 1
        retry_feedback = "" if attempt == 1 else "Try a completely different style."

        joke, model_name, adapter = await self._call_llm_async(context, retry_feedback)

        if joke is None:
            self._lg.warning("LLM failed to generate joke", extra={"attempt": attempt})
            return None

        # Run sync validation in thread to avoid blocking event loop
        return await asyncio.to_thread(self._validate, joke, model_name, adapter, attempt)

    async def _call_llm_async(
        self, context: list[str], retry_feedback: str
    ) -> tuple[Joke | None, str, AdapterInfo | None]:
        """Generate joke via LLM with structured output (async version)."""
        import appinfra.time
        from llm_infer.client import BackendRequestError, BackendTimeoutError

        from ...core.llm.types import Message

        prompt = self._build_prompt(context, retry_feedback)
        self._lg.debug("sending LLM request...")
        start_t = appinfra.time.start()
        try:
            result = await self._llm.complete_async(
                [Message(role="user", content=prompt)], output_schema=Joke
            )
        except (BackendTimeoutError, BackendRequestError) as e:
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self._lg.error(
                "LLM request failed",
                extra={
                    "after": appinfra.time.since(start_t),
                    "prompt": prompt_preview,
                    "error": str(e),
                },
            )
            return None, "unknown", None
        self._lg.debug("LLM request completed", extra={"after": appinfra.time.since(start_t)})

        if result.parsed is None:
            self._lg.warning("LLM failed to generate structured joke")
            return None, result.model, result.adapter

        return result.parsed, result.model, result.adapter

    def _validate(
        self, joke: Joke, model_name: str, adapter: AdapterInfo | None, attempt: int
    ) -> tuple[Joke, str, AdapterInfo | None, NoveltyCheck | None] | None:
        """Validate joke against denylist, completeness, variety, and novelty checks.

        Note: During parallel generation, multiple coroutines may call this method
        concurrently via asyncio.to_thread(). The _history and _variety checks are
        best-effort during parallel execution - race conditions may allow briefly
        duplicate openings. This is mitigated by the re-check in _try_save_candidate.
        """
        if not self._passes_basic_checks(joke.text, attempt):
            return None

        novelty = self._novelty.check(joke.text, current_model=model_name, current_adapter=adapter)
        if not novelty.is_novel:
            if novelty.similar_joke:
                self._history.record(novelty.similar_joke, novelty.similar_fact)
            return None

        if self._variety:
            self._variety.record(joke.text)
        return joke, model_name, adapter, novelty

    def _passes_basic_checks(self, text: str, attempt: int) -> bool:
        """Run basic validation checks (completeness, denylist, history, variety)."""
        if self._is_corrupted(text):
            self._lg.debug("joke rejected - corrupted output", extra={"attempt": attempt})
            return False
        if self._is_incomplete(text):
            self._lg.debug(
                "joke incomplete (question without punchline)", extra={"attempt": attempt}
            )
            return False
        if self._contains_denied(text):
            self._lg.debug("joke denied by denylist", extra={"attempt": attempt})
            return False
        if self._history.contains(text):
            self._lg.debug("joke rejected by history cache", extra={"attempt": attempt})
            self._history.record(text)
            return False
        if self._variety:
            variety = self._variety.check(text)
            if not variety.is_varied:
                self._lg.debug(
                    "joke rejected by variety", extra={"attempt": attempt, "reason": variety.reason}
                )
                return False
        return True

    def _is_incomplete(self, text: str) -> bool:
        """Check if joke is incomplete (question without punchline)."""
        text = text.strip()
        # Incomplete if ends with ? and has no content after the question.
        # Note: This heuristic has false positives for jokes where the question mark
        # IS the punchline (e.g., "She looked surprised?"). Accept this tradeoff
        # since incomplete jokes (setup only) are more common failure modes.
        if not text.endswith("?"):
            return False
        # Check if there's any text after the last question mark (punchline)
        last_q = text.rfind("?")
        after_q = text[last_q + 1 :].strip()
        return len(after_q) == 0

    def _is_corrupted(self, text: str) -> bool:
        """Check if joke contains non-ASCII garbage (e.g., Chinese text from model confusion)."""
        # Allow basic ASCII printable + common punctuation only
        # Reject if >10% of chars are non-ASCII (allows occasional smart quotes, etc.)
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return non_ascii > len(text) * 0.1

    def _contains_denied(self, text: str) -> bool:
        """Check if text contains denied words/phrases."""
        if not self._denylist:
            return False
        text_lower = text.lower()
        return any(term in text_lower for term in self._denylist)

    def _build_prompt(self, context: list[str], retry_feedback: str) -> str:
        """Build prompt for joke generation."""
        directive = self._directive.directive.prompt if self._directive else ""

        context_text = ""
        if context:
            context_text = "\n\nRecent jokes you've told:\n" + "\n".join(f"- {j}" for j in context)

        avoid_text = ""
        jokes_to_avoid = self._history.get_for_prompt()
        if jokes_to_avoid:
            avoid_text = self._format_avoid_section(jokes_to_avoid)

        openings_text = ""
        if self._variety:
            blocked = self._variety.get_blocked_openings()
            if blocked:
                openings_text = self._format_blocked_openings(blocked)

        retry_text = f"\n\n{retry_feedback}" if retry_feedback else ""

        return f"""{directive}{context_text}
{avoid_text}{openings_text}{retry_text}

Return your joke in JSON format with 'text' and 'style' fields."""

    def _format_blocked_openings(self, openings: list[str]) -> str:
        """Format blocked openings for the prompt.

        Args:
            openings: List of opening phrases to avoid.
        """
        # Dedupe and format as capitalized
        unique = list(dict.fromkeys(openings))  # preserve order, remove dupes
        formatted = [f'"{o.title()}"' for o in unique]
        return f"\n\nDO NOT start your joke with: {', '.join(formatted)}"

    def _format_avoid_section(self, jokes: list[tuple[JokeRecord, int]]) -> str:
        """Format the avoid section with jokes to avoid.

        Args:
            jokes: List of (JokeRecord, count) tuples.
        """
        lines = [
            "\n\nDO NOT generate jokes similar to these - they are too close to existing jokes:"
        ]
        for record, count in jokes:
            if count > 1:
                lines.append(f'- BLOCKED {count}x: "{record.joke}"')
            else:
                lines.append(f'- "{record.joke}"')
        lines.append(
            "\nGenerate something with a COMPLETELY different topic, setup, and punchline."
        )
        return "\n".join(lines)
