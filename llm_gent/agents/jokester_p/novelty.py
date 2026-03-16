"""Novelty checking for joke generation.

Provides embedding-based similarity checking to ensure jokes are never repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from llm_infer.client.types import AdapterInfo
from llm_kelt.memory.atomic import EmbeddingFilter, Fact
from sqlalchemy import or_


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ...core.traits.builtin.learn import LearnTrait


class IsolationMode(Enum):
    """Novelty isolation modes."""

    OFF = "off"
    REFERENCE = "reference"
    MODEL = "model"
    ADAPTER = "adapter"

    @classmethod
    def from_config(cls, value: str | None) -> IsolationMode:
        """Parse mode from config value."""
        if value is None or value.lower() in ("off", "none", ""):
            return cls.OFF
        try:
            return cls(value.lower())
        except ValueError:
            return cls.OFF


@dataclass
class NoveltyCheck:
    """Result of novelty checking against existing jokes."""

    is_novel: bool
    max_similarity: float
    similar_joke: str | None
    similar_fact: int | None = None


class IsolationError(Exception):
    """Raised when reference model isolation cannot be enforced."""

    pass


class NoveltyChecker:
    """Checks joke novelty using embedding similarity.

    Uses RAG (recall) to find similar existing jokes and compares
    similarity scores against a threshold to determine novelty.
    Fails closed: if the check fails, the joke is rejected to
    maintain the "never repeat" guarantee.

    Supports multiple isolation modes:
    - OFF: Compare against all jokes (no isolation)
    - REFERENCE: Binary split - reference model vs local model
    - MODEL: Per-model isolation - each model sees only its own jokes
    - ADAPTER: Per-adapter isolation - each adapter sees only its own jokes
    """

    def __init__(
        self,
        lg: Logger,
        learn_trait: LearnTrait,
        config: DotDict,
    ) -> None:
        """Initialize novelty checker.

        Args:
            lg: Logger instance.
            learn_trait: Learn trait for RAG recall.
            config: Novelty config with similarity.threshold, mode, reference.model.
        """
        self._lg = lg
        self._learn = learn_trait
        self._config = config

        similarity_config = config.get("similarity", {})
        self._threshold = similarity_config.get("threshold", 0.85)
        self._mode = IsolationMode.from_config(config.get("mode"))

        self._current_schema = config.get("current_schema", "public")
        reference_config = config.get("reference", {})
        self._reference_model = reference_config.get("model") if reference_config else None
        self._reference_schema = reference_config.get("schema") if reference_config else None

        self._warn_if_schema_mismatch()

    @property
    def has_embedder(self) -> bool:
        """Whether embedder is available for novelty checking."""
        return self._learn.has_embedder

    def _warn_if_schema_mismatch(self) -> None:
        """Warn if mode=reference but current schema differs from reference schema."""
        if self._mode != IsolationMode.REFERENCE:
            return
        if not self._reference_schema:
            return
        if self._current_schema == self._reference_schema:
            return
        self._lg.warning(
            "mode=reference but operating in different schema than reference; "
            "novelty checking won't compare against reference model jokes",
            extra={
                "current_schema": self._current_schema,
                "reference_schema": self._reference_schema,
            },
        )

    def check(
        self,
        joke_text: str,
        current_model: str | None = None,
        current_adapter: AdapterInfo | None = None,
    ) -> NoveltyCheck:
        """Check if joke is novel using embedding similarity.

        Args:
            joke_text: The joke text to check.
            current_model: The model generating this joke (for model/reference filtering).
            current_adapter: The adapter info (for adapter filtering).

        Returns:
            NoveltyCheck with is_novel, similarity score, and closest match.
        """
        # Reject empty text early - empty strings cause embedding failures
        if not joke_text or not joke_text.strip():
            self._lg.warning("novelty check rejected empty joke text")
            return NoveltyCheck(is_novel=False, max_similarity=1.0, similar_joke=None)

        self._lg.debug(
            "checking joke novelty...",
            extra={"joke": joke_text, "model": current_model, "mode": self._mode.value},
        )
        return self._search_and_evaluate(joke_text, current_model, current_adapter)

    def _search_and_evaluate(
        self,
        joke_text: str,
        current_model: str | None,
        current_adapter: AdapterInfo | None,
    ) -> NoveltyCheck:
        """Search for similar jokes and evaluate novelty."""
        try:
            schema = self._resolve_search_schema(current_adapter)
            embedding_filter = self._build_filter(current_model, current_adapter, schema)
            self._lg.debug(
                "novelty search",
                extra={
                    "schema": schema,
                    "model": current_model,
                    "has_filter": embedding_filter is not None,
                },
            )
            similar_facts = self._learn.recall(
                query=joke_text,
                top_k=1,
                categories=["joke"],
                embedding_filter=embedding_filter,
                schema=schema,
            )
            self._lg.debug("novelty search result", extra={"found": len(similar_facts)})

            if not similar_facts:
                self._lg.debug("joke is novel (no existing jokes)", extra={"joke": joke_text})
                return NoveltyCheck(is_novel=True, max_similarity=0.0, similar_joke=None)

            return self._evaluate_similarity(similar_facts[0], schema)

        except Exception as e:
            return self._handle_search_error(e, joke_text)

    def _resolve_search_schema(self, adapter: AdapterInfo | None) -> str:
        """Resolve schema to search in based on adapter."""
        if adapter is None or not adapter.actual:
            return str(self._current_schema)
        try:
            return self._learn.resolve_schema_for_adapter(adapter)
        except Exception as e:
            # Manifest lookup failed - fall back to current schema
            self._lg.debug(
                "schema resolution failed, using current schema",
                extra={"adapter": adapter.actual if adapter else None, "error": str(e)},
            )
            return str(self._current_schema)

    def _handle_search_error(self, e: Exception, joke_text: str) -> NoveltyCheck:
        """Handle search errors - fail closed unless table doesn't exist."""
        # Table doesn't exist means no jokes saved yet - joke is automatically novel
        if self._is_table_not_exists_error(e):
            self._lg.debug("metadata table does not exist yet, treating as novel")
            return NoveltyCheck(is_novel=True, max_similarity=0.0, similar_joke=None)

        self._lg.warning(
            "novelty check failed, rejecting joke to maintain never-repeat guarantee",
            extra={"exception": e, "joke": joke_text},
        )
        return NoveltyCheck(is_novel=False, max_similarity=1.0, similar_joke=None)

    def _is_table_not_exists_error(self, error: Exception) -> bool:
        """Check if error is due to table not existing (lazy table creation)."""
        error_str = str(error).lower()
        # PostgreSQL: "relation X does not exist"
        # SQLite: "no such table"
        return "does not exist" in error_str or "no such table" in error_str

    def _get_training_table(self, schema: str) -> Any:
        """Get a schema-qualified training table reference."""
        from sqlalchemy import Column, MetaData, String, Table
        from sqlalchemy.dialects.postgresql import BIGINT, BOOLEAN

        metadata = MetaData(schema=schema)
        return Table(
            "agent_jokester_training",
            metadata,
            Column("fact_id", BIGINT),
            Column("base_model", String),
            Column("adapter_version", String),
            Column("is_base_model", BOOLEAN),
        )

    def _schema_fact_ids_query(self, schema: str, where_clause: str, **params: str | None) -> Any:
        """Build schema-qualified subquery for fact_ids from training metadata."""
        from sqlalchemy import select

        t = self._get_training_table(schema)
        conditions = self._parse_where_conditions(t, where_clause, params)
        query = select(t.c.fact_id)
        return query.where(*conditions).scalar_subquery() if conditions else query.scalar_subquery()

    def _parse_where_conditions(
        self, t: Any, where_clause: str, params: dict[str, Any]
    ) -> list[Any]:
        """Parse where clause string into SQLAlchemy conditions."""
        conditions = []
        if "is_base_model = true" in where_clause.lower():
            conditions.append(t.c.is_base_model == True)  # noqa: E712
        if "base_model = :model" in where_clause:
            conditions.append(t.c.base_model == params.get("model"))
        if "adapter_version = :adapter" in where_clause:
            conditions.append(t.c.adapter_version == params.get("adapter"))
        if "base_model ILIKE :ref_pattern" in where_clause:
            # Use parameterized ILIKE pattern for SQL injection safety
            conditions.append(t.c.base_model.ilike(params.get("ref_pattern")))
        return conditions

    def _build_filter(
        self, current_model: str | None, current_adapter: AdapterInfo | None, schema: str
    ) -> EmbeddingFilter | None:
        """Build embedding filter based on isolation mode."""
        if self._mode == IsolationMode.OFF:
            return None
        if self._mode == IsolationMode.REFERENCE:
            return self._build_reference_filter(current_model, schema)
        if self._mode == IsolationMode.MODEL:
            return self._build_model_filter(current_model, schema)
        if self._mode == IsolationMode.ADAPTER:
            return self._build_adapter_filter(current_model, current_adapter, schema)
        return None

    # =========================================================================
    # Reference mode: binary split (reference vs local)
    # =========================================================================

    def _build_reference_filter(self, current_model: str | None, schema: str) -> EmbeddingFilter:
        """Build filter for reference mode: reference vs local split."""
        if not self._reference_model:
            raise IsolationError("reference_model is required when mode is 'reference'")
        if not current_model:
            raise IsolationError("current_model is required when mode is 'reference'")

        is_reference = self._reference_model.lower() in current_model.lower()
        if is_reference:
            return self._filter_reference_model_jokes(schema)
        return self._filter_local_model_jokes(schema)

    def _filter_reference_model_jokes(self, schema: str) -> EmbeddingFilter:
        """Filter to only reference model jokes."""
        subquery = self._schema_fact_ids_query(
            schema, "base_model ILIKE :ref_pattern", ref_pattern=f"%{self._reference_model}%"
        )
        self._lg.debug("filtering to reference model jokes only", extra={"schema": schema})
        return EmbeddingFilter().where(Fact.id.in_(subquery))

    def _filter_local_model_jokes(self, schema: str) -> EmbeddingFilter:
        """Filter to local model jokes (exclude reference, include orphaned facts)."""
        ref_subquery = self._schema_fact_ids_query(
            schema, "base_model ILIKE :ref_pattern", ref_pattern=f"%{self._reference_model}%"
        )
        all_subquery = self._schema_fact_ids_query(schema, "1=1")
        self._lg.debug("filtering to exclude reference model jokes", extra={"schema": schema})
        # Include facts that: (1) have non-reference metadata, OR (2) have no metadata
        return EmbeddingFilter().where(
            or_(Fact.id.not_in(ref_subquery), ~Fact.id.in_(all_subquery))
        )

    # =========================================================================
    # Model mode: per-model isolation
    # =========================================================================

    def _build_model_filter(self, current_model: str | None, schema: str) -> EmbeddingFilter:
        """Build filter for model mode: each model sees only its own jokes."""
        if not current_model:
            raise IsolationError("current_model is required when mode is 'model'")

        subquery = self._schema_fact_ids_query(schema, "base_model = :model", model=current_model)
        self._lg.debug(
            "filtering to same model jokes", extra={"model": current_model, "schema": schema}
        )
        return EmbeddingFilter().where(Fact.id.in_(subquery))

    # =========================================================================
    # Adapter mode: per-adapter isolation
    # =========================================================================

    def _build_adapter_filter(
        self, current_model: str | None, current_adapter: AdapterInfo | None, schema: str
    ) -> EmbeddingFilter:
        """Build filter for adapter mode: each adapter sees only its own jokes."""
        adapter_name = current_adapter.actual if current_adapter else None

        if adapter_name:
            # Fine-tuned model: only see jokes from same adapter
            subquery = self._schema_fact_ids_query(
                schema, "adapter_version = :adapter", adapter=adapter_name
            )
            self._lg.debug(
                "filtering to same adapter jokes", extra={"adapter": adapter_name, "schema": schema}
            )
        else:
            # Base model (no adapter): only see jokes from same base model
            subquery = self._schema_fact_ids_query(
                schema, "is_base_model = true AND base_model = :model", model=current_model
            )
            self._lg.debug(
                "filtering to same base model jokes",
                extra={"model": current_model, "schema": schema},
            )

        return EmbeddingFilter().where(Fact.id.in_(subquery))

    # =========================================================================
    # Similarity evaluation
    # =========================================================================

    def _evaluate_similarity(self, closest: Any, schema: str) -> NoveltyCheck:
        """Evaluate similarity of joke against closest match."""
        fact_id = closest.entity.id
        if closest.score >= self._threshold:
            self._log_result(closest, schema, is_novel=False)
            return NoveltyCheck(
                is_novel=False,
                max_similarity=closest.score,
                similar_joke=closest.entity.content,
                similar_fact=fact_id,
            )

        self._log_result(closest, schema, is_novel=True)
        return NoveltyCheck(
            is_novel=True,
            max_similarity=closest.score,
            similar_joke=closest.entity.content,
            similar_fact=fact_id,
        )

    def _log_result(self, similar: Any, schema: str, is_novel: bool) -> None:
        """Log similarity check result."""
        msg = "joke is novel" if is_novel else "joke too similar"
        self._lg.debug(
            msg,
            extra={
                "similarity": similar.score,
                "existing": similar.entity.content,
                "fact": similar.entity.id,
                "schema": schema,
            },
        )
