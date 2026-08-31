# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Memory trait for agent kelt-backed memory, feedback, and learned completions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from appinfra import DotDict
from llm_infer.client import ChatClient, ChatResponse, EmbeddingClient
from llm_kelt import Client as KeltClient
from llm_kelt import SchemaMode
from llm_kelt.core import Database
from llm_kelt.core.types import ScoredEntity
from llm_kelt.inference import ContextBuilder
from llm_kelt.memory.atomic import EmbeddingFilter, Fact
from llm_kelt.memory.isolation import ClientContext
from llm_kelt.scoped_client import ScopedClient

from ...llm.types import CompletionResult
from ...runnable import ExecutionResult
from ..base import BaseTrait
from .llm import _resolve_llm_defaults


if TYPE_CHECKING:
    from ...agent import Agent, Identity


# Type alias for Memory configuration
MemoryConfig = DotDict
"""Memory configuration as DotDict.

Expected fields:
    identity: Resolved Identity (required).
    schema: Schema config dict with 'name' (default: {"name": "public"}).
    llm: LLM configuration for learned completions.
    db: Database configuration dict (with url, extensions, etc.).
    embedder_url: URL for embedding service (None = no RAG).
    embedder_model: Model name for embeddings (default: "default").
    embedder_timeout: Embedder timeout in seconds (default: 30.0).
    training: Kelt training configuration dict (default_profiles, etc.).
"""


class MemoryTrait(BaseTrait):
    """Memory capability trait for kelt-backed facts, feedback, and learned completions.

    Wraps llm_kelt.Client, ChatClient, and EmbeddingClient to provide
    memory-enabled completions with fact injection and feedback capture.

    Capabilities:
        - complete(): Generate completions with automatic fact injection
        - remember(): Store facts about the user
        - recall(): Search facts by semantic similarity
        - record_feedback(): Record feedback on responses
        - record_preference(): Record preference pairs for training
        - record_solution(): Record execution solutions

    Adapter-manifest-driven schema resolution lives in TrainingTrait (training.py).

    Dependency ownership: the trait imports no external factory. Database,
    ChatClient, and EmbeddingClient are built by ``TraitFactory._create_memory``
    and injected at construction. Two seams:

    - Config-time (standard path): ``TraitFactory.create_memory_trait`` builds
      the three clients from ``memory_config`` and passes them with
      ``owns_chat_client=True`` and ``owns_embedder=True`` so the trait closes
      them on ``on_stop``. Note: Database/PG lifecycle is not managed by this
      trait — the underlying connection pool outlives the trait. This is
      intentional for shared-pool scenarios; explicit pool shutdown belongs
      at the application layer.
    - Direct injection (test / advanced): pass ``chat_client``, ``embedder``,
      or ``database`` yourself with the corresponding ``owns_*=False``; caller
      retains close responsibility. See ``.with_chat_client()``,
      ``.with_embedder()``, and ``.with_database()`` for immutable-view
      fluent overrides.

    Example:
        from llm_gent.agent import AgentFactory

        # Standard construction — TraitFactory builds the external clients.
        agent = AgentFactory(lg).from_config({
            "identity": "my-agent",
            "llm": {"backends": {"local": {...}}},
            "memory": {
                "db": {"url": "postgresql://localhost/kelt"},
                "embedder_url": "http://localhost:8001/v1",
            },
        })
        agent.start()

        # Learned completion with fact injection
        result = agent.get_trait(MemoryTrait).complete("What do I prefer?")

    Lifecycle:
        - ``__init__``: composes ``KeltClient`` and ``ContextBuilder`` from the
          injected database, chat client, and embedder.
        - ``on_start()``: resolves LLM defaults from config (no client build).
        - ``on_stop()``: closes the chat client / embedder iff the matching
          ``owns_*`` flag is True.
    """

    def __init__(
        self,
        agent: Agent,
        config: MemoryConfig,
        *,
        database: Database,
        chat_client: ChatClient,
        embedder: EmbeddingClient | None = None,
        owns_chat_client: bool = False,
        owns_embedder: bool = False,
    ) -> None:
        """Initialize memory trait with injected clients.

        Args:
            agent: The agent this trait belongs to.
            config: Memory configuration (identity, schema, llm defaults, etc.).
            database: Live ``Database``. Built by ``TraitFactory`` in the
                standard path; may be a stub for tests.
            chat_client: Live ``ChatClient`` for learned completions.
            embedder: Live ``EmbeddingClient`` for RAG, or None if no embedder
                is configured.
            owns_chat_client: If True, ``on_stop`` closes ``chat_client``. Set
                by ``TraitFactory`` for factory-built clients. Callers that
                inject their own client keep this False and clean up themselves.
            owns_embedder: Same semantic for ``embedder``.
        """
        super().__init__(agent)
        self.config = config
        self._database = database
        self._client = chat_client
        self._embedder = embedder
        self._owns_chat_client = owns_chat_client
        self._owns_embedder = owns_embedder
        self._llm_defaults: dict[str, Any] = {}
        self._kelt: KeltClient = self._create_kelt_client(database, embedder, chat_client)
        self._context: ContextBuilder = ContextBuilder(self._kelt.atomic.assertions)

    def _create_kelt_client(
        self, database: Database, embedder: EmbeddingClient | None, llm_client: ChatClient | None
    ) -> KeltClient:
        """Create schema-agnostic kelt client.

        Args:
            database: Database instance.
            embedder: EmbeddingClient instance (None if not configured).
            llm_client: LLM client instance (None if not configured).

        Returns:
            KeltClient scoped to default_schema. Use client.with_schema("X")
            to override for specific operations.
        """
        identity = self._resolve_identity()
        if identity is None:
            raise ValueError("MemoryConfig must have identity set")

        # Use default_schema for default operations; callers can override with with_schema()
        context = ClientContext(context_key=identity.context_key, schema_name=self.default_schema)

        return KeltClient(
            lg=self.agent.lg,
            database=database,
            context=context,
            embedder=embedder,
            llm_client=llm_client,
            schema_mode=SchemaMode.ENSURE,
            training_config=self.config.get("training"),
        )

    def _resolve_identity(self) -> Identity | None:
        """Resolve Identity from config."""
        identity: Identity | None = self.config.get("identity")
        return identity

    def on_start(self) -> None:
        """Resolve config-derived LLM defaults. Clients are already injected."""
        self.agent.lg.trace(
            "starting memory trait...",
            extra={
                "agent": self.agent.name,
                "owns_chat_client": self._owns_chat_client,
                "owns_embedder": self._owns_embedder,
                "has_embedder": self._embedder is not None,
            },
        )
        self._llm_defaults = _resolve_llm_defaults(self.config.get("llm") or DotDict())
        self.agent.lg.trace(
            "memory trait started",
            extra={
                "agent": self.agent.name,
                "owns_chat_client": self._owns_chat_client,
                "owns_embedder": self._owns_embedder,
            },
        )

    def on_stop(self) -> None:
        """Close the chat client and embedder iff this trait owns their lifecycles."""
        self.agent.lg.trace(
            "stopping memory trait...",
            extra={
                "agent": self.agent.name,
                "owns_chat_client": self._owns_chat_client,
                "owns_embedder": self._owns_embedder,
            },
        )
        try:
            if self._owns_chat_client:
                self._client.close()
        finally:
            if self._owns_embedder and self._embedder is not None:
                self._embedder.close()
        self.agent.lg.trace(
            "memory trait stopped",
            extra={
                "agent": self.agent.name,
                "closed_chat_client": self._owns_chat_client,
                "closed_embedder": self._owns_embedder and self._embedder is not None,
            },
        )

    def with_chat_client(self, chat_client: ChatClient) -> Self:
        """Return a new trait bound to ``chat_client``, detached from the registry.

        Immutable-view fluent (mirrors ``LLMTrait.with_router``): ``self`` stays
        canonical for ``agent.get_trait(MemoryTrait)`` and its chat client is
        unchanged. The returned instance shares ``agent``, ``config``,
        ``database``, and ``embedder`` but is not registered.

        Ownership: ``owns_chat_client`` on the returned trait is False. The
        caller owns the injected client's lifecycle. For a persistent swap,
        call ``agent.replace_trait(new)``.
        """
        new = type(self)(
            self.agent,
            self.config,
            database=self._database,
            chat_client=chat_client,
            embedder=self._embedder,
            owns_chat_client=False,
            owns_embedder=False,
        )
        self.agent.lg.debug(
            "memory trait detached with new chat client",
            extra={"agent": self.agent.name, "detached_from_registry": True},
        )
        return new

    def with_embedder(self, embedder: EmbeddingClient | None) -> Self:
        """Return a new trait bound to ``embedder``, detached from the registry.

        See ``.with_chat_client()`` for the ownership and registry semantics.
        """
        new = type(self)(
            self.agent,
            self.config,
            database=self._database,
            chat_client=self._client,
            embedder=embedder,
            owns_chat_client=False,
            owns_embedder=False,
        )
        self.agent.lg.debug(
            "memory trait detached with new embedder",
            extra={
                "agent": self.agent.name,
                "detached_from_registry": True,
                "has_embedder": embedder is not None,
            },
        )
        return new

    def with_database(self, database: Database) -> Self:
        """Return a new trait bound to ``database``, detached from the registry.

        See ``.with_chat_client()`` for the ownership and registry semantics.
        Database has no owned lifecycle on this trait; ownership stays with
        whoever built it (``TraitFactory`` in the standard path).
        """
        new = type(self)(
            self.agent,
            self.config,
            database=database,
            chat_client=self._client,
            embedder=self._embedder,
            owns_chat_client=False,
            owns_embedder=False,
        )
        self.agent.lg.debug(
            "memory trait detached with new database",
            extra={"agent": self.agent.name, "detached_from_registry": True},
        )
        return new

    # =========================================================================
    # Schema-aware client access
    # =========================================================================

    def get_client_for_schema(self, schema: str) -> ScopedClient:
        """Get a scoped client for a specific schema.

        Uses the new with_schema() API for per-operation schema selection.
        Schema and tables are created lazily on first use.

        Args:
            schema: PostgreSQL schema name.

        Returns:
            ScopedClient configured for the specified schema.

        Raises:
            RuntimeError: If trait not started.
        """
        return self.kelt.with_schema(schema)

    @property
    def default_schema(self) -> str:
        """Default schema name from config (schema.name), falling back to 'public'."""
        schema_config = self.config.get("schema") or {}
        return str(schema_config.get("name") or "public")

    @property
    def kelt(self) -> KeltClient:
        """Access the kelt client."""
        return self._kelt

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
        return self._client

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
            system=prompt if prompt else None,
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
        import uuid

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

        Args:
            fact: The fact to store.
            category: Category for organization.
            source: How the fact was obtained.
            confidence: Confidence level 0.0-1.0.

        Returns:
            Fact ID.
        """
        fact_id = self.kelt.atomic.assertions.add(
            fact, category=category, source=source, confidence=confidence
        )

        # Embed for semantic search if embedder available
        if self._embedder is not None:
            embedding = self._embedder.embed(fact)
            self.kelt.atomic.embeddings.set_embedding(
                fact_id=fact_id,
                embedding=embedding.embedding,
                model=self._embedder.model,
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
            schema: PostgreSQL schema to search in (uses with_schema for scoping).

        Returns:
            List of ScoredEntity[Fact] sorted by similarity.

        Raises:
            ValueError: If embedder not configured.
        """
        if self._embedder is None:
            raise ValueError("recall() requires embedder - configure embedder_url in MemoryConfig")

        embedding = self._embedder.embed(query)
        client = self.kelt if schema is None else self.kelt.with_schema(schema)
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
        return self._context.build_system_prompt(
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

        return self._context.build_system_prompt_from_facts(
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
