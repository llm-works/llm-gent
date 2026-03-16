"""Storage helper for jokester-p agent.

Handles joke persistence, model usage tracking, and training metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from llm_infer.client.types import AdapterInfo
from llm_kelt.scoped_client import ScopedClient

from llm_gent.core.memory.schema import validate_schema_name

from .schema import ModelUsage, TrainingMetadata


# Re-export for backward compatibility (cli.py and pairing.py import from here)
_validate_schema_name = validate_schema_name


if TYPE_CHECKING:
    from ...core.traits.builtin.learn import LearnTrait
    from ...core.traits.builtin.storage import StorageTrait
    from .generate import Joke


class Storage:
    """Manages joke persistence and metadata.

    Handles:
    - Saving jokes to atomic_facts via LearnTrait
    - Model usage tracking (which model generated each joke)
    - Training metadata (adapter versions, fallback tracking)

    Agent tables are created lazily per-schema when first needed.
    """

    def __init__(
        self,
        lg: Logger,
        storage_trait: StorageTrait,
        learn_trait: LearnTrait,
        agent_name: str,
        default_schema: str = "public",
    ) -> None:
        """Initialize storage helper.

        Args:
            lg: Logger instance.
            storage_trait: StorageTrait for database access.
            learn_trait: LearnTrait for saving jokes.
            agent_name: Name of the agent.
            default_schema: Default PostgreSQL schema when no adapter info.
        """
        self._lg = lg
        self._storage = storage_trait.storage
        self._learn = learn_trait
        self._agent_name = agent_name
        self._default_schema = default_schema
        self._initialized_schemas: set[str] = set()  # Track which schemas have tables

    def save_joke(
        self,
        joke: Joke,
        model_name: str,
        attempts: int,
        adapter: AdapterInfo | None = None,
    ) -> tuple[int, str]:
        """Save joke and record metadata.

        Uses the adapter's manifest to determine which schema to write to.
        This ensures jokes are stored in the same schema the adapter was trained on.

        Args:
            joke: Generated joke.
            model_name: Actual model used.
            attempts: Number of generation attempts.
            adapter: Full adapter info from LLM response.

        Returns:
            Tuple of (fact_id, schema) for the saved joke.
        """
        # Resolve schema once and reuse for both save and metadata
        schema = self._resolve_schema(adapter)
        client = self._learn.get_client_for_schema(schema)

        fact_id = client.atomic.solutions.record(
            agent_name=self._agent_name,
            problem="Tell one short, original joke",
            problem_context={"style_preference": "varied"},
            answer={"text": joke.text, "style": joke.style},
            answer_text=joke.text,
            tokens_used=0,
            latency_ms=0,
            category="joke",
            source="agent",
        )
        self._lg.debug(
            "joke saved", extra={"fact_id": fact_id, "style": joke.style, "schema": schema}
        )

        self._record_joke_metadata_in_schema(fact_id, model_name, attempts, adapter, schema)
        return fact_id, schema

    def _get_client_for_adapter(self, adapter: AdapterInfo | None) -> ScopedClient:
        """Get kelt client for the adapter's schema.

        Args:
            adapter: Adapter info from LLM response, or None for base model.

        Returns:
            ScopedClient configured for the appropriate schema.
        """
        schema = self._resolve_schema(adapter)
        return self._learn.get_client_for_schema(schema)

    def _resolve_schema(self, adapter: AdapterInfo | None) -> str:
        """Resolve the schema for an adapter, or return default.

        Validates schema name to prevent SQL injection.
        """
        if adapter is None:
            return _validate_schema_name(self._default_schema)
        schema = self._learn.resolve_schema_for_adapter(adapter)
        return _validate_schema_name(schema)

    def _ensure_tables_in_schema(self, schema: str) -> None:
        """Ensure agent tables exist in the given schema (lazy creation)."""
        if schema in self._initialized_schemas:
            return

        engine = self._learn.kelt.database.engine
        with engine.connect() as conn:
            for table_class in [ModelUsage, TrainingMetadata]:
                self._create_table_if_not_exists(conn, schema, table_class)
            conn.commit()

        self._initialized_schemas.add(schema)
        self._lg.debug("agent tables ensured in schema", extra={"schema": schema})

    def _create_table_if_not_exists(self, conn: object, schema: str, table_class: type) -> None:
        """Create a table in the specified schema if it doesn't exist.

        Uses to_metadata() to create a schema-specific copy of the table,
        avoiding mutation of the shared table definition (race condition safe).
        """
        from sqlalchemy import MetaData

        original_table = table_class.__table__  # type: ignore[attr-defined]
        # Create a new metadata with the target schema and copy the table into it
        temp_metadata = MetaData(schema=schema)
        schema_table = original_table.to_metadata(temp_metadata)
        schema_table.create(conn, checkfirst=True)

    def record_joke_metadata(
        self,
        fact_id: int,
        model_name: str,
        attempts: int,
        adapter: AdapterInfo | None = None,
    ) -> None:
        """Record model usage and training metadata for a joke.

        Args:
            fact_id: ID of the fact in atomic_facts.
            model_name: Actual model used (from LLM response).
            attempts: Number of generation attempts needed.
            adapter: Full adapter info from LLM response.

        Raises:
            Exception: If metadata recording fails critically (re-raises after logging).
        """
        schema = self._resolve_schema(adapter)
        self._record_joke_metadata_in_schema(fact_id, model_name, attempts, adapter, schema)

    def _record_joke_metadata_in_schema(
        self,
        fact_id: int,
        model_name: str,
        attempts: int,
        adapter: AdapterInfo | None,
        schema: str,
    ) -> None:
        """Record metadata in a specific schema (avoids duplicate resolution)."""
        self._ensure_tables_in_schema(schema)

        usage_failed = self._try_record_model_usage(fact_id, model_name, attempts, schema)
        training_failed = self._try_record_training_metadata(fact_id, model_name, adapter, schema)
        self._log_metadata_failures(fact_id, usage_failed, training_failed)

    def _try_record_model_usage(
        self, fact_id: int, model_name: str, attempts: int, schema: str
    ) -> bool:
        """Try to record model usage, return True if failed."""
        try:
            self._record_model_usage(fact_id, model_name, attempts, schema)
            return False
        except Exception as e:
            self._lg.warning(
                "model usage recording failed",
                extra={"exception": e, "fact_id": fact_id, "model": model_name},
            )
            return True

    def _try_record_training_metadata(
        self, fact_id: int, model_name: str, adapter: AdapterInfo | None, schema: str
    ) -> bool:
        """Try to record training metadata, return True if failed."""
        try:
            self._record_training_metadata(fact_id, model_name, adapter, schema)
            return False
        except Exception as e:
            self._lg.warning(
                "training metadata recording failed",
                extra={"exception": e, "fact_id": fact_id, "model": model_name},
            )
            return True

    def _log_metadata_failures(
        self, fact_id: int, usage_failed: bool, training_failed: bool
    ) -> None:
        """Log summary if any metadata recording failed."""
        if usage_failed or training_failed:
            self._lg.warning(
                "metadata recording incomplete",
                extra={
                    "fact_id": fact_id,
                    "usage_failed": usage_failed,
                    "training_failed": training_failed,
                },
            )

    def _record_model_usage(
        self, fact_id: int, model_name: str, attempts: int, schema: str
    ) -> None:
        """Record model usage metadata in agent-specific table."""
        from sqlalchemy import text

        sql = text(f"""
            INSERT INTO {schema}.agent_jokester_model_usage
            (fact_id, model_name, model_role, attempts, context_key, created_at)
            VALUES (:fact_id, :model_name, :model_role, :attempts, :context_key, NOW())
        """)
        params = {
            "fact_id": fact_id,
            "model_name": model_name,
            "model_role": "sole",
            "attempts": attempts,
            "context_key": self._agent_name,
        }
        engine = self._learn.kelt.database.engine
        with engine.connect() as conn:
            conn.execute(sql, params)
            conn.commit()

        self._lg.debug("model usage recorded", extra={"fact_id": fact_id, "schema": schema})

    def _record_training_metadata(
        self, fact_id: int, model_name: str, adapter: AdapterInfo | None, schema: str
    ) -> None:
        """Record training metadata."""
        import json

        from sqlalchemy import text

        is_finetuned = adapter is not None and adapter.actual and not adapter.fallback
        adapter_version = adapter.actual if adapter and is_finetuned else None
        adapter_info_json = json.dumps(asdict(adapter)) if adapter else None

        sql = text(f"""
            INSERT INTO {schema}.agent_jokester_training
            (fact_id, base_model, adapter_version, is_base_model, adapter_fallback,
             adapter_info, context_key, created_at)
            VALUES (:fact_id, :base_model, :adapter_version, :is_base_model, :adapter_fallback,
                    :adapter_info, :context_key, NOW())
        """)
        params = {
            "fact_id": fact_id,
            "base_model": model_name,
            "adapter_version": adapter_version,
            "is_base_model": not is_finetuned,
            "adapter_fallback": False,
            "adapter_info": adapter_info_json,
            "context_key": self._agent_name,
        }
        engine = self._learn.kelt.database.engine
        with engine.connect() as conn:
            conn.execute(sql, params)
            conn.commit()

        self._lg.debug("training metadata recorded", extra={"fact_id": fact_id, "schema": schema})

    def get_adapter_count(
        self,
        adapter_name: str,
        md5: str | None = None,
        schema: str | None = None,
        max_chars: int | None = None,
    ) -> int:
        """Get count of jokes generated by an adapter.

        Args:
            adapter_name: Adapter name to count.
            md5: Optional md5 to filter by specific version.
            schema: Schema to query (defaults to adapter's manifest schema or default).
            max_chars: Optional max character limit filter.

        Returns:
            Number of jokes generated by this adapter.
        """
        from sqlalchemy import text

        # Resolve schema from adapter manifest if not provided
        if schema is None:
            schema = self._resolve_schema_for_adapter_name(adapter_name)

        sql = self._build_adapter_count_sql(md5, schema, max_chars)
        params = self._build_adapter_count_params(adapter_name, md5, max_chars)

        try:
            engine = self._learn.kelt.database.engine
            with engine.connect() as conn:
                result = conn.execute(text(sql), params)
                row = result.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            # Table not existing is expected with lazy creation - just return 0
            if "does not exist" in str(e).lower():
                self._lg.debug(
                    "adapter count table not yet created", extra={"adapter": adapter_name}
                )
            else:
                self._lg.warning("adapter count query failed", extra={"exception": e})
            return 0

    def _build_adapter_count_sql(
        self, md5: str | None, schema: str, max_chars: int | None = None
    ) -> str:
        """Build SQL for adapter count query."""
        schema = _validate_schema_name(schema)
        md5_clause = "AND adapter_info->>'md5' = :md5" if md5 else ""
        chars_join = (
            f"JOIN {schema}.atomic_solution_details asd ON asd.fact_id = t.fact_id"
            if max_chars
            else ""
        )
        chars_clause = "AND LENGTH(asd.answer_text) < :max_chars" if max_chars else ""
        return f"""
            SELECT COUNT(*) as count
            FROM {schema}.agent_jokester_training t
            {chars_join}
            WHERE adapter_info->>'actual' = :adapter_name
            {md5_clause} {chars_clause}
        """

    def get_base_model_count(
        self, model_name: str, schema: str | None = None, max_chars: int | None = None
    ) -> int:
        """Get count of jokes generated by a specific base model (no adapter).

        Args:
            model_name: The base model name to count (e.g., 'qwen2.5-7b-instruct').
            schema: Schema to query (defaults to default_schema).
            max_chars: Optional max character limit filter.

        Returns:
            Number of jokes generated by this base model.
        """
        from sqlalchemy import text

        schema = schema or self._default_schema
        sql = text(self._build_base_model_count_sql(schema, max_chars))
        params: dict[str, Any] = {"model_name": model_name}
        if max_chars:
            params["max_chars"] = max_chars

        try:
            engine = self._learn.kelt.database.engine
            with engine.connect() as conn:
                result = conn.execute(sql, params)
                row = result.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            if "does not exist" in str(e).lower():
                self._lg.debug("base model count table not yet created")
            else:
                self._lg.warning("base model count query failed", extra={"exception": e})
            return 0

    def _build_base_model_count_sql(self, schema: str, max_chars: int | None = None) -> str:
        """Build SQL for base model count query."""
        schema = _validate_schema_name(schema)
        chars_join = (
            f"JOIN {schema}.atomic_solution_details asd ON asd.fact_id = t.fact_id"
            if max_chars
            else ""
        )
        chars_clause = "AND LENGTH(asd.answer_text) < :max_chars" if max_chars else ""
        return f"""
            SELECT COUNT(*) as count
            FROM {schema}.agent_jokester_training t
            {chars_join}
            WHERE t.is_base_model = true AND t.base_model = :model_name
            {chars_clause}
        """

    def _resolve_schema_for_adapter_name(self, adapter_name: str) -> str:
        """Resolve schema for an adapter by looking up its manifest.

        Extracts md5 from adapter name (e.g., 'jokester-p-sft-e6a8a798834d' -> 'e6a8a798834d')
        and looks up the manifest to find source.schema_name.
        """
        factory = self._learn._get_train_factory()
        if factory is None:
            return self._default_schema

        # Extract md5 from adapter name (last 12 chars or segment after last hyphen)
        md5 = adapter_name.rsplit("-", 1)[-1] if "-" in adapter_name else adapter_name

        manifest = factory.manifest.get_manifest(md5)
        if manifest and manifest.source and manifest.source.schema_name:
            return str(manifest.source.schema_name)

        return self._default_schema

    def _build_adapter_count_params(
        self, adapter_name: str, md5: str | None, max_chars: int | None = None
    ) -> dict[str, Any]:
        """Build params for adapter count query."""
        params: dict[str, Any] = {"adapter_name": adapter_name}
        if md5:
            params["md5"] = md5
        if max_chars:
            params["max_chars"] = max_chars
        return params
