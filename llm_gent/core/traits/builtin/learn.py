"""Learn trait for agent memory and feedback capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from appinfra.db.pg import PG
from llm_infer.client import ChatClient, ChatResponse
from llm_infer.client import Factory as LLMClientFactory
from llm_infer.client.types import AdapterInfo
from llm_kelt import Client as KeltClient
from llm_kelt.core import Database
from llm_kelt.core.types import ScoredEntity
from llm_kelt.inference import ContextBuilder, Embedder
from llm_kelt.memory.atomic import EmbeddingFilter, Fact
from llm_kelt.memory.isolation import ClientContext
from llm_kelt.scoped_client import ScopedClient
from llm_kelt.training import Factory as TrainFactory

from ...llm.types import CompletionResult
from ...runnable import ExecutionResult
from ..base import BaseTrait
from .llm import _resolve_llm_defaults


if TYPE_CHECKING:
    from llm_gent.core.agent import Agent, Identity


from appinfra import DotDict


class ManifestNotFoundError(Exception):
    """Raised when adapter manifest cannot be found for schema resolution."""

    pass


# Type alias for Learn configuration
LearnConfig = DotDict
"""Learn configuration as DotDict.

Expected fields:
    identity: Resolved Identity (required).
    schema: Schema config dict with 'name' and 'enforce' (default: {"name": "public"}).
    llm: LLM configuration for learned completions.
    db: Database configuration dict (with url, extensions, etc.).
    embedder_url: URL for embedding service (None = no RAG).
    embedder_model: Model name for embeddings (default: "default").
    embedder_timeout: Embedder timeout in seconds (default: 30.0).
    training: Training configuration dict (default_profiles, etc.).
    adapters: Adapter registry config dict (lora.base_path for manifest lookups).
"""


class LearnTrait(BaseTrait):
    """Learn capability trait for memory, feedback, and learned completions.

    Wraps llm_kelt.Client, ChatClient, and Embedder to provide
    learning-enabled completions with memory and feedback.

    Capabilities:
        - complete(): Generate completions with automatic fact injection
        - remember(): Store facts about the user
        - recall(): Search facts by semantic similarity
        - record_feedback(): Record feedback on responses
        - record_preference(): Record preference pairs for training

    Example:
        from llm_gent.core.traits import LearnTrait, LearnConfig, LLMConfig
        from llm_gent.core.agent import Identity

        agent = Agent(lg, config)
        learn_trait = LearnTrait(agent, LearnConfig(
            identity=Identity.from_name("my-agent"),
            llm=LLMConfig(base_url="http://localhost:8000/v1"),
            embedder_url="http://localhost:8001/v1",
        ))
        agent.add_trait(learn_trait)
        agent.start()

        # Learned completion with fact injection
        result = agent.get_trait(LearnTrait).complete("What do I prefer?")

        # Store a fact
        agent.get_trait(LearnTrait).remember("User prefers concise answers")

    Lifecycle:
        - on_start(): Creates Client, ChatClient, Embedder, ContextBuilder
        - on_stop(): Closes ChatClient and Embedder
    """

    def __init__(self, agent: Agent, config: LearnConfig) -> None:
        """Initialize learn trait.

        Args:
            agent: The agent this trait belongs to.
            config: Learn configuration.
        """
        super().__init__(agent)
        self.config = config
        self._database: Database | None = None
        self._kelt: KeltClient | None = None
        self._train_factory: TrainFactory | None = None  # initialized in on_start
        self._embedder: Embedder | None = None
        self._context: ContextBuilder | None = None
        self._client: ChatClient | None = None
        self._llm_defaults: dict[str, Any] = {}

    def _create_kelt_client(
        self, database: Database, embedder: Embedder | None, llm_client: ChatClient | None
    ) -> KeltClient:
        """Create schema-agnostic kelt client.

        Args:
            database: Database instance.
            embedder: Embedder instance (None if not configured).
            llm_client: LLM client instance (None if not configured).

        Returns:
            Schema-agnostic KeltClient. Use client.with_schema("X") for
            per-operation schema selection.
        """
        identity = self._resolve_identity()
        if identity is None:
            raise ValueError("LearnConfig must have identity set")

        # Schema-agnostic context - no schema_name
        context = ClientContext(context_key=identity.context_key)

        return KeltClient(
            lg=self.agent.lg,
            database=database,
            context=context,
            embedder=embedder,
            llm_client=llm_client,
            ensure_schema=True,
            training_config=self.config.get("training"),
        )

    def _resolve_identity(self) -> Identity | None:
        """Resolve Identity from config."""
        identity: Identity | None = self.config.get("identity")
        return identity

    def on_start(self) -> None:
        """Create learn client, LLM client, and embedder on agent start."""
        from llm_gent.core.errors import ConfigError

        # Create database from config
        if self.config.db is None:
            raise ConfigError("LearnConfig.db is required")
        pg = PG(self.agent.lg, self.config.db)
        self._database = Database(self.agent.lg, pg)

        # Create LLM client for completions (before Client)
        llm_config: DotDict = self.config.get("llm") or DotDict()
        self._client = LLMClientFactory(self.agent.lg).from_config(llm_config)
        self._llm_defaults = _resolve_llm_defaults(llm_config)

        # Create embedder if URL provided (before Client)
        if self.config.embedder_url:
            self._embedder = Embedder(
                base_url=self.config.embedder_url,
                model=self.config.embedder_model,
                timeout=self.config.embedder_timeout,
            )
        else:
            self._embedder = None

        # Create kelt client with embedder and llm_client already available
        self._kelt = self._create_kelt_client(self._database, self._embedder, self._client)
        self._context = ContextBuilder(self._kelt.atomic.assertions)

        # Train factory created lazily via _get_train_factory() when needed

    def on_stop(self) -> None:
        """Close LLM client and embedder on agent stop."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None
        self._kelt = None
        self._train_factory = None
        self._context = None
        self._database = None

    def _get_train_factory(self) -> TrainFactory | None:
        """Get or create training factory for manifest lookups.

        Returns:
            TrainFactory if adapters.lora.base_path is configured, None otherwise.
        """
        if self._train_factory is not None:
            return self._train_factory

        from pathlib import Path

        adapters_config = self.config.get("adapters") or {}
        lora_config = adapters_config.get("lora") or {}
        base_path = lora_config.get("base_path")

        if not base_path:
            self.agent.lg.warning(
                "no adapters.lora.base_path configured, manifest lookup disabled",
                extra={"adapters_config": adapters_config},
            )
            return None

        registry_path = Path(base_path).expanduser()
        if not registry_path.exists():
            self.agent.lg.warning(
                "adapter registry path does not exist, manifest lookup disabled",
                extra={"path": str(registry_path)},
            )
            return None

        self._train_factory = TrainFactory(self.agent.lg, registry_path)
        return self._train_factory

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

    def resolve_schema_for_adapter(self, adapter_info: AdapterInfo) -> str:
        """Resolve the schema for an adapter by looking up its manifest.

        If `enforce=true` is set in schema config, always uses the config schema
        regardless of the adapter's manifest.

        Args:
            adapter_info: Adapter info from LLM response.

        Returns:
            Schema name from manifest, or default schema if not found/not configured.
        """
        # Config override: if enforce=true, use config schema instead of manifest lookup
        schema_config = self.config.get("schema", {})
        if schema_config.get("enforce", False):
            return self._default_schema

        md5 = adapter_info.md5
        if not md5:
            self.agent.lg.debug("no md5 in adapter info, using default schema")
            return self._default_schema

        # Try manifest lookup (returns None if adapters not configured)
        train_factory = self._get_train_factory()
        if train_factory is None:
            self.agent.lg.debug("no train factory, using default schema")
            return self._default_schema

        manifest = train_factory.manifest.get_manifest(md5)
        if manifest and manifest.source and manifest.source.schema_name:
            schema = str(manifest.source.schema_name)
            self.agent.lg.trace("resolved schema from manifest", extra={"schema": schema})
            return schema

        raise ManifestNotFoundError(
            f"Manifest lookup failed for md5={md5}. "
            f"Cannot determine schema - refusing to use default to prevent data corruption."
        )

    @property
    def _default_schema(self) -> str:
        """Get default schema from config (schema.name)."""
        schema_config = self.config.get("schema", {})
        return str(schema_config.get("name") or "public")

    @property
    def kelt(self) -> KeltClient:
        """Access the kelt client.

        Raises:
            RuntimeError: If trait not started.
        """
        if self._kelt is None:
            raise RuntimeError("LearnTrait not started - ensure agent.start() was called")
        return self._kelt

    @property
    def embedder(self) -> Embedder | None:
        """Access the embedder (None if not configured)."""
        return self._embedder

    @property
    def has_embedder(self) -> bool:
        """Check if embedder is available for RAG."""
        return self._embedder is not None

    @property
    def client(self) -> ChatClient:
        """Access the LLM router.

        Raises:
            RuntimeError: If trait not started.
        """
        if self._client is None:
            raise RuntimeError("LearnTrait not started - ensure agent.start() was called")
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
                raise ValueError("RAG requires embedder - configure embedder_url in LearnConfig")
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
                model_name=self._embedder.model,
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
            raise ValueError("recall() requires embedder - configure embedder_url in LearnConfig")

        embedding = self._embedder.embed(query)
        client = self.kelt if schema is None else self.kelt.with_schema(schema)
        return client.atomic.embeddings.search_similar(
            query=embedding.embedding,
            model_name=self._embedder.model,
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
        if self._context is None:
            raise RuntimeError("LearnTrait not started")

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
        if self._context is None:
            raise RuntimeError("LearnTrait not started")
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
