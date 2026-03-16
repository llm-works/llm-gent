"""Rating helpers for jokester-p agent.

Provides thin wrappers around RatingTrait for inline and batch rating.
For CLI batch rating without agent setup, use core.training.BatchRatingService.
"""

from __future__ import annotations

from appinfra.log import Logger

from llm_gent.core.memory.rating import Result
from llm_gent.core.traits.builtin.rating import RatingTrait


class InlineRater:
    """Rates jokes during generation using RatingTrait."""

    def __init__(self, lg: Logger, rating_trait: RatingTrait) -> None:
        self._lg = lg
        self._rating = rating_trait

    def rate(self, fact_id: int, joke_text: str) -> None:
        """Rate a joke using all configured providers."""
        try:
            results = self._rating.rate_fact_with_all_providers(
                fact_id=fact_id, content=joke_text, fact_type="solution"
            )
            for result in results:
                self._log_rating(fact_id, result)
        except Exception as e:
            self._lg.warning("inline rating failed", extra={"exception": e, "fact_id": fact_id})

    def _log_rating(self, fact_id: int, result: Result) -> None:
        """Log a single rating result."""
        stars_visual = "★" * result.stars + "☆" * (5 - result.stars)
        reasoning = (
            result.reasoning[:100] + "..." if len(result.reasoning) > 100 else result.reasoning
        )
        self._lg.info(
            f"inline rating: {stars_visual}",
            extra={
                "fact_id": fact_id,
                "stars": result.stars,
                "model": result.model,
                "reasoning": reasoning,
            },
        )


class BatchRater:
    """Batches jokes for efficient batch rating during generation.

    Instead of rating each joke immediately (one LLM call per joke),
    this rater queues jokes and flushes when the batch is full.
    Reduces LLM calls by ~5x with default batch_size=5.

    Usage:
        rater = BatchRater(lg, rating_trait, batch_size=5)
        rater.queue(fact_id, joke_text, schema="hn_exp")  # Auto-flushes when batch is full
        rater.flush()  # Flush remaining before exit
    """

    def __init__(
        self,
        lg: Logger,
        rating_trait: RatingTrait,
        batch_size: int,
        fact_type: str = "solution",
        max_chars: int | None = None,
    ) -> None:
        self._lg = lg
        self._rating = rating_trait
        self._batch_size = batch_size
        self._fact_type = fact_type
        self._max_chars = max_chars
        self._queue: list[tuple[int, str, str | None]] = []

    def queue(self, fact_id: int, content: str, schema: str | None = None) -> None:
        """Queue a fact for rating. Flushes when batch is full.

        Skips items that exceed max_chars limit (if configured).
        """
        if self._max_chars is not None and len(content) >= self._max_chars:
            self._lg.debug(
                "skipped rating: exceeds max_chars",
                extra={"fact_id": fact_id, "length": len(content), "max_chars": self._max_chars},
            )
            return
        self._queue.append((fact_id, content, schema))
        self._lg.debug(
            "queued for batch rating",
            extra={"fact_id": fact_id, "queue_size": len(self._queue), "schema": schema},
        )
        if len(self._queue) >= self._batch_size:
            self.flush()

    def flush(self) -> int:
        """Rate all queued items. Returns count rated."""
        if not self._queue:
            return 0

        items_by_schema = self._group_by_schema()
        total_rated, processed_schemas = self._process_schemas(items_by_schema)

        # Remove successfully processed items from queue
        if processed_schemas:
            self._queue = [
                (fid, content, s) for fid, content, s in self._queue if s not in processed_schemas
            ]

        return total_rated

    def _group_by_schema(self) -> dict[str | None, list[tuple[int, str]]]:
        """Group queued items by schema for batch processing."""
        items_by_schema: dict[str | None, list[tuple[int, str]]] = {}
        for fact_id, content, schema in self._queue:
            if schema not in items_by_schema:
                items_by_schema[schema] = []
            items_by_schema[schema].append((fact_id, content))
        return items_by_schema

    def _process_schemas(
        self, items_by_schema: dict[str | None, list[tuple[int, str]]]
    ) -> tuple[int, list[str | None]]:
        """Process items grouped by schema. Returns (total_rated, processed_schemas)."""
        total_rated = 0
        processed_schemas: list[str | None] = []

        for schema, items in items_by_schema.items():
            try:
                rated = self._rating.rate_items(items, self._fact_type, schema)
                total_rated += rated
                self._log_batch_results(items, rated, schema)
                processed_schemas.append(schema)
            except Exception as e:
                self._lg.warning(
                    "batch rating failed for schema",
                    extra={"schema": schema, "exception": e},
                )

        return total_rated, processed_schemas

    def _log_batch_results(
        self, items: list[tuple[int, str]], rated: int, schema: str | None
    ) -> None:
        """Log batch completion and individual ratings."""
        self._lg.debug(
            "flushed batch for rating",
            extra={"count": len(items), "rated": rated, "schema": schema},
        )

        for fact_id, content in items:
            stars = self._rating.get_fact_rating(fact_id, schema)
            if stars is not None:
                clamped = max(0, min(5, stars))
                stars_visual = "★" * clamped + "☆" * (5 - clamped)
                preview = content[:80] + "..." if len(content) > 80 else content
                self._lg.info(
                    f"rated: {stars_visual}",
                    extra={"fact_id": fact_id, "stars": stars, "preview": preview},
                )

    @property
    def pending_count(self) -> int:
        """Number of items currently queued."""
        return len(self._queue)
