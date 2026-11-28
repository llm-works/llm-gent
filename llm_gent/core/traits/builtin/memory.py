# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Memory trait for agent kelt-backed memory, feedback, and learned completions."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal

from appinfra import DotDict
from appinfra.log import Logger
from llm_infer.client import ChatClient, ChatResponse, EmbeddingClient
from llm_kelt import Client as KeltClient
from llm_kelt import ClientView, SchemaMode
from llm_kelt.core import Database
from llm_kelt.core.types import ScoredEntity
from llm_kelt.memory.atomic import EmbeddingFilter, Fact
from llm_kelt.memory.isolation import ClientContext

from ...llm.config import resolve_llm_defaults
from ...llm.types import CompletionResult
from ...runnable import ExecutionResult


if TYPE_CHECKING:
    from ...agent import Agent


# Type alias for Memory configuration
MemoryConfig = DotDict
"""The platform's memory settings, as DotDict.

Expected fields:
    schema: Schema config dict with 'name' (default: {"name": "public"}).
    llm: LLM configuration for learned completions.
    db: Database configuration dict (with url, extensions, etc.).
    embedder_url: URL for embedding service (None = no RAG).
    embedder_model: Model name for embeddings (default: "default").
    embedder_timeout: Embedder timeout in seconds (default: 30.0).
    training: Kelt training configuration dict (default_profiles, etc.).
"""


class MemoryTrait:
    """Kelt-backed memory on the platform: the shared clients, one per platform.

    The platform builds one MemoryTrait over its memory clients and one kelt
    client, on the platform's schema, so the schema is prepared once however
    many agents there are. It holds nothing per agent. An agent holds its
    own :class:`MemoryBinding` from :meth:`bind`: its context key and the
    kelt view on it, built without database I/O. ``agent.require(MemoryTrait)``
    returns that binding, which carries the memory operations.

    The platform owns the database, chat client and embedder; the trait
    closes none of them. Adapter-manifest-driven schema resolution lives in
    TrainingTrait (training.py).

    Example:
        from llm_gent import AgentFactory, PlatformFactory

        platform = PlatformFactory(lg).from_config({
            "memory": {
                "db": {"url": "postgresql://localhost/kelt"},
                "embedder_url": "http://localhost:8001/v1",
            },
        })
        agent = AgentFactory(platform).from_config({
            "identity": {"name": "my-agent"},
            "traits": {"required": ["memory"]},
        })
        result = agent.require(MemoryTrait).complete("What do I prefer?")
    """

    trait_name = "memory"

    def __init__(
        self,
        lg: Logger,
        config: MemoryConfig,
        *,
        database: Database,
        chat_client: ChatClient,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        """Build the shared trait and its kelt client over the platform's clients.

        Args:
            lg: Logger instance.
            config: The platform's memory settings (schema, llm, training, ...).
            database: The platform's ``Database``; may be a stub for tests.
            chat_client: The platform's ``ChatClient`` for learned completions.
            embedder: The platform's ``EmbeddingClient`` for RAG, or None.
        """
        self._lg = lg
        self.config = config
        self._database = database
        self._client = chat_client
        self._embedder = embedder
        self._schema = _schema_name(config.get("schema"), database.schema)
        self._llm_defaults: dict[str, Any] = resolve_llm_defaults(config.get("llm") or DotDict())
        self._root = KeltClient(
            lg=lg,
            database=database,
            context=ClientContext(schema_name=self._schema),
            embedder=embedder,
            llm_client=chat_client,
            schema_mode=SchemaMode.ENSURE,
            training_config=config.get("training"),
        )

    def bind(self, agent: Agent) -> MemoryBinding:
        """``agent``'s memory: its identity's context key, on the platform's schema; no I/O."""
        return MemoryBinding(self, agent.identity.context_key)

    def on_start(self) -> None:
        """Nothing to start: the clients are the platform's and already built."""

    def on_stop(self) -> None:
        """Nothing to stop: the platform closes the clients."""

    def client_view(self, context_key: str, schema: str | None = None) -> ClientView:
        """The kelt client scoped to ``context_key`` on ``schema`` (default: the platform's).

        A schema other than the platform's is prepared on its first use.
        """
        return self._root.view(context_key, schema_name=schema or self._schema)

    @property
    def default_schema(self) -> str:
        """The schema memory reads and writes: the platform's."""
        return self._schema

    @property
    def llm_defaults(self) -> dict[str, Any]:
        """Model settings for learned completions, from the platform's memory config."""
        return self._llm_defaults

    @property
    def database(self) -> Database:
        """The platform's memory database."""
        return self._database

    @property
    def embedder(self) -> EmbeddingClient | None:
        """Access the embedder (None if not configured)."""
        return self._embedder

    @property
    def client(self) -> ChatClient:
        """Access the LLM router."""
        return self._client


class MemoryBinding:
    """An agent's memory: its context key and the kelt view on it; the agent holds it.

    Built by :meth:`MemoryTrait.bind`. The platform's clients are reached
    through the trait; the binding holds only the agent's partition.

    Capabilities:
        - complete(): Generate completions with automatic fact injection
        - remember(): Store facts about the user
        - recall(): Search facts by semantic similarity
        - record_feedback(): Record feedback on responses
        - record_preference(): Record preference pairs for training
        - record_solution(): Record execution solutions
    """

    def __init__(self, trait: MemoryTrait, context_key: str) -> None:
        """Bind ``trait`` to ``context_key``.

        Args:
            trait: The platform's memory trait.
            context_key: The partition the agent's memory lives in.
        """
        self._trait = trait
        self._context_key = context_key
        self._kelt = trait.client_view(context_key)
        self._embedder = trait.embedder
        self._llm_defaults = trait.llm_defaults

    def on_start(self) -> None:
        """Nothing to start: building the binding did no I/O."""

    def on_stop(self) -> None:
        """Nothing to release: the binding holds a key and a view, no connection."""

    # =========================================================================
    # Access
    # =========================================================================

    @property
    def trait(self) -> MemoryTrait:
        """The platform's memory trait this binding is of."""
        return self._trait

    @property
    def context_key(self) -> str:
        """The partition the agent's memory lives in."""
        return self._context_key

    @property
    def kelt(self) -> ClientView:
        """The kelt client scoped to the agent's context key."""
        return self._kelt

    def get_client_for_schema(self, schema: str) -> ClientView:
        """The agent's context key on ``schema``; the schema is prepared on first use."""
        return self._trait.client_view(self._context_key, schema)

    @property
    def default_schema(self) -> str:
        """The schema memory reads and writes: the platform's."""
        return self._trait.default_schema

    @property
    def database(self) -> Database:
        """The platform's memory database."""
        return self._trait.database

    @property
    def embedder(self) -> EmbeddingClient | None:
        """Access the embedder (None if not configured)."""
        return self._embedder

    @property
    def has_embedder(self) -> bool:
        """Check if embedder is available for RAG."""
        return self._embedder is not None

    @property
    def client(self) -> ChatClient:
        """Access the LLM router."""
        return self._trait.client

    # =========================================================================
    # Completions
    # =========================================================================

    def complete(
        self,
        query: str,
        system_prompt: str = "",
        include_facts: bool = True,
        rag: bool = False,
        rag_top_k: int = 10,
        rag_min_similarity: float = 0.5,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Generate a completion with optional fact injection.

        Args:
            query: User query.
            system_prompt: Base system prompt.
            include_facts: Whether to inject facts into prompt.
            rag: Use RAG-based fact retrieval (requires embedder).
            rag_top_k: Max facts for RAG.
            rag_min_similarity: Min similarity for RAG.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            CompletionResult with response and metadata.
        """
        prompt = self._build_completion_prompt(
            system_prompt, query, include_facts, rag, rag_top_k, rag_min_similarity
        )

        response = self.client.chat(
            messages=[{"role": "user", "content": query}],
            system=prompt or None,
            model=self._llm_defaults.get("model"),
            temperature=temperature
            if temperature is not None
            else self._llm_defaults.get("temperature", 0.7),
            max_tokens=max_tokens
            if max_tokens is not None
            else self._llm_defaults.get("max_tokens"),
        )

        return self._response_to_result(response)

    def _build_completion_prompt(
        self,
        base_prompt: str,
        query: str,
        include_facts: bool,
        rag: bool,
        rag_top_k: int,
        rag_min_similarity: float,
    ) -> str:
        """Build system prompt with optional fact injection."""
        if not include_facts:
            return base_prompt

        if rag:
            if self._embedder is None:
                raise ValueError("RAG requires embedder - configure embedder_url in MemoryConfig")
            return self.build_prompt_rag(
                base_prompt=base_prompt,
                query=query,
                top_k=rag_top_k,
                min_similarity=rag_min_similarity,
            )

        return self.build_prompt(base_prompt=base_prompt)

    def _response_to_result(self, response: ChatResponse) -> CompletionResult:
        """Convert LLM response to CompletionResult."""
        tokens_used = 0
        if response.usage:
            tokens_used = response.usage.total_tokens or (
                (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
            )

        return CompletionResult(
            id=str(uuid.uuid4()),
            content=response.content,
            model=response.model or self._llm_defaults.get("model", "unknown"),
            tokens_used=tokens_used,
            latency_ms=0,  # Latency tracking not implemented for completion operations
            tool_calls=None,
        )

    # =========================================================================
    # Memory operations
    # =========================================================================

    def remember(
        self,
        fact: str,
        category: str = "general",
        source: Literal["user", "inferred", "conversation", "system"] = "user",
        confidence: float = 1.0,
    ) -> int:
        """Store a fact about the user.

        If embedder is configured, the fact is also embedded for semantic search.
        The embedding is computed before the fact is stored, so an embedder
        that fails stores nothing.

        Args:
            fact: The fact to store.
            category: Category for organization.
            source: How the fact was obtained.
            confidence: Confidence level 0.0-1.0.

        Returns:
            Fact ID.
        """
        embedder = self._embedder
        embedding = embedder.embed(fact) if embedder is not None else None
        fact_id = self.kelt.atomic.assertions.add(
            fact, category=category, source=source, confidence=confidence
        )
        if embedder is not None and embedding is not None:
            self.kelt.atomic.embeddings.set_embedding(
                fact_id=fact_id,
                embedding=embedding.embedding,
                model=embedder.model,
            )

        return fact_id

    def forget(self, fact_id: int) -> None:
        """Remove a stored fact.

        Args:
            fact_id: ID of the fact to remove.
        """
        self.kelt.atomic.assertions.delete(fact_id)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
        categories: list[str] | None = None,
        embedding_filter: EmbeddingFilter | None = None,
        schema: str | None = None,
    ) -> list[ScoredEntity[Fact]]:
        """Search facts by semantic similarity.

        Args:
            query: Text to search for similar facts.
            top_k: Maximum results.
            min_similarity: Minimum similarity threshold (0-1).
            categories: Filter to these categories (None = all). Deprecated: use embedding_filter.
            embedding_filter: EmbeddingFilter for flexible filtering (recommended).
            schema: PostgreSQL schema to search in (default: the platform's).

        Returns:
            List of ScoredEntity[Fact] sorted by similarity.

        Raises:
            ValueError: If embedder not configured.
        """
        if self._embedder is None:
            raise ValueError("recall() requires embedder - configure embedder_url in MemoryConfig")

        client = self.kelt if schema is None else self.get_client_for_schema(schema)
        embedding = self._embedder.embed(query)
        return client.atomic.embeddings.search_similar(
            query=embedding.embedding,
            model=self._embedder.model,
            top_k=top_k,
            min_similarity=min_similarity,
            categories=categories,
            filter=embedding_filter,
        )

    # =========================================================================
    # Context building
    # =========================================================================

    def build_prompt(
        self,
        base_prompt: str,
        max_facts: int = 100,
        categories: list[str] | None = None,
    ) -> str:
        """Build system prompt with injected facts (all mode).

        Args:
            base_prompt: Base system prompt.
            max_facts: Maximum facts to include.
            categories: Filter to these categories.

        Returns:
            System prompt with facts appended.
        """
        return self.kelt.context_builder.build_system_prompt(
            base_prompt=base_prompt,
            categories=categories,
            max_facts=max_facts,
        )

    def build_prompt_rag(
        self,
        base_prompt: str,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> str:
        """Build system prompt with RAG-injected facts.

        Args:
            base_prompt: Base system prompt.
            query: Query for semantic fact retrieval.
            top_k: Maximum facts to retrieve.
            min_similarity: Minimum similarity threshold.

        Returns:
            System prompt with relevant facts appended.

        Raises:
            ValueError: If embedder not configured.
        """
        if self._embedder is None:
            raise ValueError("build_prompt_rag() requires embedder")

        scored_facts = self.recall(query, top_k=top_k, min_similarity=min_similarity)
        facts = [sf.entity for sf in scored_facts]

        return self.kelt.context_builder.build_system_prompt_from_facts(
            base_prompt=base_prompt,
            facts=facts,
        )

    # =========================================================================
    # Feedback
    # =========================================================================

    def record_feedback(
        self,
        content: str,
        signal: Literal["positive", "negative"],
        context: dict[str, Any] | None = None,
    ) -> int:
        """Record feedback on content.

        Args:
            content: The content that received feedback.
            signal: Whether it was good or bad.
            context: Additional context (query, model, etc.).

        Returns:
            Feedback ID.
        """
        return self.kelt.atomic.feedback.record(
            signal=signal,
            comment=content,
            context=context,
        )

    def record_preference(
        self,
        context: str,
        chosen: str,
        rejected: str,
    ) -> int:
        """Record a preference pair (chosen over rejected).

        Args:
            context: The context/prompt for the pair.
            chosen: The preferred response.
            rejected: The rejected response.

        Returns:
            Preference ID.
        """
        return self.kelt.atomic.preferences.record(
            context=context,
            chosen=chosen,
            rejected=rejected,
        )

    # =========================================================================
    # Solutions
    # =========================================================================

    def record_solution(
        self,
        agent_name: str,
        problem: str,
        result: ExecutionResult,
        summary: str,
    ) -> None:
        """Record execution solution from ExecutionResult.

        Args:
            agent_name: Name of the agent that solved the problem.
            problem: The problem/task that was executed.
            result: The execution result with outcome details.
            summary: Human-readable summary of the solution.
        """
        self.kelt.atomic.solutions.record(
            agent_name=agent_name,
            problem=problem,
            problem_context={
                "iterations": result.iterations,
                "trace_id": result.trace_id,
            },
            answer={
                "success": result.success,
                "output": result.content,
                "iterations": result.iterations,
            },
            answer_text=summary,
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
            category="execution",
            source="agent",
        )


def _schema_name(schema_config: dict[str, Any] | None, database_schema: str | None) -> str:
    """The schema a memory config names (``schema.name``), else the database's, else ``public``.

    An explicit schema that differs from the database's still fails in kelt's
    schema-conflict check.
    """
    return str((schema_config or {}).get("name") or database_schema or "public")
