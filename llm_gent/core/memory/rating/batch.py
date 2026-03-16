"""Batch rating service for CLI usage.

Provides batch rating without requiring full agent/trait setup.
Uses direct SQL on atomic_facts tables.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass

from appinfra.db.pg import PG
from appinfra.log import Logger
from sqlalchemy import text

from llm_gent.core.llm import LLMCaller
from llm_gent.core.training import StarRatedItem

from ..schema import validate_schema_name
from .models import BatchItem, BatchRequest, Result
from .service import Service as CoreRatingService


@dataclass
class UnratedItem:
    """An unrated item from the database."""

    id: int
    content: str


class BatchRatingService:
    """Service for batch rating facts via CLI.

    Uses the core rating service but provides a simpler interface
    for CLI usage without requiring full agent setup.

    Works with any context_key - not agent-specific.
    """

    def __init__(
        self,
        lg: Logger,
        pg: PG,
        llm_caller: LLMCaller,
        prompt_template: str,
        context_key: str,
        batch_size: int = 5,
        provider: str = "local",
        model: str = "auto",
        fact_type: str = "solution",
        schema: str | None = None,
        max_chars: int | None = None,
    ) -> None:
        self._lg = lg
        self._pg = pg
        self._context_key = context_key
        self._core_service = CoreRatingService(lg, llm_caller)
        self._prompt_template = prompt_template
        self._batch_size = batch_size
        self._provider = provider
        self._model = model
        self._fact_type = fact_type
        self._schema = schema
        self._max_chars = max_chars

    def _schema_prefix(self) -> str:
        """Get schema prefix for table names."""
        if self._schema and self._schema != "public":
            return f"{validate_schema_name(self._schema)}."
        return ""

    def get_unrated(self, limit: int | None = None) -> list[UnratedItem]:
        """Get unrated facts from the database.

        Respects max_chars filter if configured.
        """
        prefix = self._schema_prefix()
        limit_clause = "LIMIT :limit" if limit is not None else ""
        max_chars_clause = "AND length(af.content) < :max_chars" if self._max_chars else ""
        sql = text(f"""
            SELECT af.id, af.content
            FROM {prefix}atomic_facts af
            LEFT JOIN {prefix}atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key
              AND af.type = :fact_type
              AND afd.id IS NULL
              {max_chars_clause}
            ORDER BY af.created_at ASC
            {limit_clause}
        """)
        params: dict[str, object] = {
            "context_key": self._context_key,
            "fact_type": self._fact_type,
        }
        if limit is not None:
            params["limit"] = limit
        if self._max_chars is not None:
            params["max_chars"] = self._max_chars
        with self._pg.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [UnratedItem(id=r[0], content=r[1]) for r in rows]

    def rate_batches(
        self, limit: int | None = None, batch_size: int | None = None
    ) -> Generator[list[StarRatedItem], None, None]:
        """Rate unrated facts, yielding results batch by batch.

        Args:
            limit: Maximum facts to rate (None = all).
            batch_size: Items per batch (None = use configured default).

        Yields:
            List of StarRatedItem for each batch.
        """
        items = self.get_unrated(limit)
        if not items:
            return

        size = batch_size if batch_size is not None else self._batch_size
        if size <= 0:
            raise ValueError(f"batch_size must be > 0, got {size}")

        for i in range(0, len(items), size):
            batch = items[i : i + size]
            batch_ratings, failed = self._rate_batch(batch)

            self._lg.debug(
                "batch completed",
                extra={"rated": len(batch_ratings), "failed": failed},
            )

            yield batch_ratings

    def _rate_batch(self, items: list[UnratedItem]) -> tuple[list[StarRatedItem], int]:
        """Rate a single batch. Returns (rated_items, failed_count)."""
        batch_items = [BatchItem(fact=item.id, content=item.content) for item in items]
        content_map = {item.id: item.content for item in items}
        request = BatchRequest(
            items=batch_items,
            prompt_template=self._prompt_template,
            model=self._model,
            provider=self._provider,
            backend=self._provider,
        )

        try:
            results = self._core_service.rate_batch(request)
            rated: list[StarRatedItem] = []
            for result in results:
                if self._save_rating(result):
                    rated.append(
                        StarRatedItem(
                            id=result.fact,
                            content=content_map.get(result.fact, ""),
                            score=result.stars,
                        )
                    )
            return (rated, len(items) - len(rated))
        except Exception as e:
            self._lg.warning("batch rating failed", extra={"exception": e})
            return ([], len(items))

    def _save_rating(self, result: Result) -> bool:
        """Save a rating to the database."""
        prefix = self._schema_prefix()
        sql = text(f"""
            INSERT INTO {prefix}atomic_feedback_details
                (fact_id, signal, strength, provider_type, context, provider)
            VALUES (:fact_id, :signal, :strength, :provider_type, CAST(:context AS jsonb), :provider)
            ON CONFLICT (fact_id) DO NOTHING
        """)
        params = {
            "fact_id": result.fact,
            "signal": result.signal,
            "strength": result.strength,
            "provider_type": result.provider_type.value,
            "context": json.dumps({"stars": result.stars, "reasoning": result.reasoning}),
            "provider": result.provider,
        }
        try:
            with self._pg.connect() as conn:
                conn.execute(sql, params)
                conn.commit()
            return True
        except Exception as e:
            self._lg.warning(
                "failed to save rating", extra={"exception": e, "fact_id": result.fact}
            )
            return False
