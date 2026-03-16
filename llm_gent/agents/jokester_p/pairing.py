"""Preference pairing service for jokester-p agent.

Creates preference pairs from rated jokes for DPO training.
Uses core/training infrastructure for pairing algorithms.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sqlalchemy import text

from llm_gent.core.training import (
    StarFilter,
    StarPairingResult,
    StarRatedItem,
    pair_by_margin,
    pair_by_threshold,
)

from .storage import _validate_schema_name


if TYPE_CHECKING:
    from appinfra.db.pg import PG
    from appinfra.log import Logger


# Re-export types for backward compatibility
__all__ = ["PairingService", "StarFilter"]


class PairingService:
    """Service for creating preference pairs from rated jokes.

    Fetches rated jokes from the database and uses core pairing algorithms
    to create preference pairs for DPO training.
    """

    def __init__(self, lg: Logger, pg: PG, context_key: str, schema: str | None = None) -> None:
        self._lg = lg
        self._pg = pg
        self._context_key = context_key
        self._schema = schema

    def _schema_prefix(self) -> str:
        """Get schema prefix for table names."""
        if self._schema and self._schema != "public":
            return f"{_validate_schema_name(self._schema)}."
        return ""

    def get_rated_jokes(
        self, max_chars: int | None = None, model: str | None = None
    ) -> list[StarRatedItem]:
        """Get all rated jokes sorted by stars (desc)."""
        prefix = self._schema_prefix()
        joins_sql, filters_sql, params = self._build_rated_jokes_filters(max_chars, model)
        sql = text(f"""
            SELECT DISTINCT ON (af.id)
                af.id, af.content, (afd.context->>'stars')::int as stars
            FROM {prefix}atomic_facts af
            JOIN {prefix}atomic_feedback_details afd ON af.id = afd.fact_id
            {joins_sql}
            WHERE af.context_key = :context_key
              AND afd.context->>'stars' IS NOT NULL {filters_sql}
            ORDER BY af.id, afd.id DESC
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        rated = [StarRatedItem(id=r[0], content=r[1], score=r[2]) for r in rows]
        rated.sort(key=lambda item: (-item.score, item.id))
        return rated

    def _resolve_model_pattern(self, model: str, prefix: str) -> tuple[str, bool]:
        """Resolve model filter pattern, expanding abbreviated md5 if needed.

        Supports formats:
            - "qwen2.5-7b" -> ("%qwen2.5-7b%", False) - model name substring
            - "08e43bfcb746" -> ("%08e43bfcb746%", True) - full md5
            - "08..b746" -> expands to full md5, errors if not unique

        Returns:
            (pattern, is_adapter) tuple
        """
        # Check for abbreviated md5 format: XX..XXXX
        abbrev_match = re.match(r"^([0-9a-f]{2})\.\.([0-9a-f]{4})$", model, re.IGNORECASE)
        if abbrev_match:
            prefix_part, suffix_part = abbrev_match.groups()
            full_md5 = self._expand_abbreviated_md5(prefix_part, suffix_part, prefix)
            return f"%{full_md5}%", True

        # Check for full md5 format: 12 hex chars
        if re.match(r"^[0-9a-f]{12}$", model, re.IGNORECASE):
            return f"%{model}%", True

        # Regular model name
        return f"%{model}%", False

    def _expand_abbreviated_md5(
        self, prefix_part: str, suffix_part: str, schema_prefix: str
    ) -> str:
        """Expand abbreviated md5 to full, error if not unique."""
        sql = text(f"""
            SELECT DISTINCT adapter_info->>'md5' as md5
            FROM {schema_prefix}agent_jokester_training
            WHERE adapter_info->>'md5' LIKE :pattern
        """)
        pattern = f"{prefix_part}%{suffix_part}"

        with self._pg.connect() as conn:
            rows = conn.execute(sql, {"pattern": pattern}).fetchall()

        if len(rows) == 0:
            raise ValueError(f"No adapter found matching '{prefix_part}..{suffix_part}'")
        if len(rows) > 1:
            matches = [r[0] for r in rows]
            raise ValueError(
                f"Ambiguous adapter '{prefix_part}..{suffix_part}' matches multiple: {matches}"
            )

        return str(rows[0][0])

    def _build_rated_jokes_filters(
        self, max_chars: int | None, model: str | None
    ) -> tuple[str, str, dict[str, object]]:
        """Build SQL fragments and params for rated jokes query."""
        prefix = self._schema_prefix()
        joins: list[str] = []
        filters: list[str] = []
        params: dict[str, object] = {"context_key": self._context_key}

        if max_chars:
            joins.append(f"JOIN {prefix}atomic_solution_details asd ON asd.fact_id = af.id")
            filters.append("AND length(asd.answer_text) < :max_chars")
            params["max_chars"] = max_chars

        if model:
            model_pattern, is_adapter = self._resolve_model_pattern(model, prefix)
            joins.append(f"JOIN {prefix}agent_jokester_model_usage u ON u.fact_id = af.id")
            if is_adapter:
                joins.append(f"JOIN {prefix}agent_jokester_training t ON t.fact_id = af.id")
                filters.append("AND t.adapter_info->>'md5' LIKE :model_pattern")
            else:
                filters.append("AND u.model_name LIKE :model_pattern")
            params["model_pattern"] = model_pattern

        return "\n            ".join(joins), " ".join(filters), params

    def _validate_strategy(self, strategy: str) -> None:
        """Validate pairing strategy."""
        if strategy not in ("relative", "threshold"):
            raise ValueError(
                f"Unknown pairing strategy: {strategy!r}. Expected 'relative' or 'threshold'."
            )

    def create_pairs(
        self,
        strategy: str = "relative",
        margin: int = 1,
        high_threshold: int = 4,
        low_threshold: int = 2,
        min_pairs: int | None = None,
        max_pairs: int | None = None,
        max_chars: int | None = None,
        no_reuse: bool = False,
        model: str | None = None,
        chosen_stars: StarFilter | None = None,
        rejected_stars: StarFilter | None = None,
        length_epsilons: list[int | None] | None = None,
    ) -> StarPairingResult:
        """Create preference pairs from rated jokes."""
        self._validate_strategy(strategy)
        rated = self._fetch_rated_jokes(max_chars, model)

        self._lg.debug(
            "creating pairs...",
            extra={"strategy": strategy, "margin": margin, "length_epsilons": length_epsilons},
        )

        result = self._apply_pairing_strategy(
            rated,
            strategy,
            margin,
            high_threshold,
            low_threshold,
            min_pairs,
            max_pairs,
            no_reuse,
            chosen_stars,
            rejected_stars,
            length_epsilons,
        )

        self._lg.debug("created pairs", extra={"count": len(result.pairs)})
        return result

    def _fetch_rated_jokes(self, max_chars: int | None, model: str | None) -> list[StarRatedItem]:
        """Fetch rated jokes from database."""
        self._lg.debug("fetching rated jokes...", extra={"max_chars": max_chars, "model": model})
        rated = self.get_rated_jokes(max_chars=max_chars, model=model)
        self._lg.debug("fetched rated jokes", extra={"count": len(rated)})
        return rated

    def _apply_pairing_strategy(
        self,
        rated: list[StarRatedItem],
        strategy: str,
        margin: int,
        high_threshold: int,
        low_threshold: int,
        min_pairs: int | None,
        max_pairs: int | None,
        no_reuse: bool,
        chosen_stars: StarFilter | None,
        rejected_stars: StarFilter | None,
        length_epsilons: list[int | None] | None,
    ) -> StarPairingResult:
        """Apply the selected pairing strategy."""
        if strategy == "relative":
            return pair_by_margin(
                rated,
                margin=margin,
                min_pairs=min_pairs,
                max_pairs=max_pairs,
                no_reuse=no_reuse,
                chosen_filter=chosen_stars,
                rejected_filter=rejected_stars,
                length_epsilons=length_epsilons,
            )
        return pair_by_threshold(
            rated,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            max_pairs=max_pairs,
        )
