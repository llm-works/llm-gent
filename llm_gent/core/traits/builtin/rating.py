"""Rating trait for automated LLM-based content rating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from appinfra import DotDict

from llm_gent.core.memory.rating import (
    AtomicFactsBackend,
    BatchConfig,
    BatchItem,
    BatchRequest,
    ConfigParser,
    Criteria,
    CriteriaConfig,
    ProviderConfig,
    ProviderType,
)
from llm_gent.core.memory.rating import Request as RatingRequest
from llm_gent.core.memory.rating import Result as RatingResult
from llm_gent.core.memory.rating import Service as RatingService

from ..base import BaseTrait


if TYPE_CHECKING:
    from llm_gent.core.agent import Agent


# Type alias for Rating configuration
RatingConfig = DotDict
"""Rating configuration as DotDict.

Expected fields:
    providers: List of rating provider configurations
        - type: Provider type ("llm" or "manual")
        - backend: LLM backend configuration
        - enabled: Whether this provider is enabled (default: True)
    models: Type-specific criteria and prompts
        atomic: Criteria for atomic facts
            solution: Prompt and criteria for solution facts
    batch_size: Number of items to rate per batch (default: 10)
    auto: Skip confirmation prompts (default: False)
"""


@dataclass
class _BatchContext:
    """Context for batch rating operations."""

    context_key: str
    prompt: str
    criteria: list[Criteria]
    model: str
    provider_id: str
    backend: AtomicFactsBackend


class RatingTrait(BaseTrait):
    """Automated LLM-based content rating trait.

    Provides automated rating capabilities using LLM backends to evaluate
    agent-generated content. Stores ratings in atomic_feedback_details with
    source tracking for multi-rater scenarios.

    **IMPORTANT:** RatingTrait depends on LearnTrait. LearnTrait must be attached
    to the agent BEFORE RatingTrait.

    Capabilities:
        - rate_unrated(): Find and rate unrated content automatically
        - rate_fact(): Rate a specific fact by ID
        - rate_fact_with_all_providers(): Rate with all providers (for inline rating)
        - get_unrated_count(): Count unrated facts
        - Multi-provider support: Use multiple LLMs for rating comparison

    Example:
        from llm_gent.core.traits import RatingTrait, LearnTrait

        rating_config = RatingConfig(
            providers=[{
                "type": "llm",
                "backend": {"model": "claude-sonnet-4.5", ...},
                "enabled": True
            }],
            models={
                "atomic": {
                    "solution": {
                        "prompt": "Rate this joke...",
                        "criteria": [
                            {"name": "humor", "description": "Is it funny?"},
                            {"name": "originality", "description": "Is it original?"}
                        ]
                    }
                }
            },
            batch_size=10,
            auto=True
        )

        # Attach traits
        agent.add_trait(LearnTrait(agent, learn_config))
        agent.add_trait(RatingTrait(agent, rating_config))
        agent.start()

        # Rate content
        results = agent.get_trait(RatingTrait).rate_fact_with_all_providers(
            fact_id=123, content="Why did the..."
        )

    Lifecycle:
        - on_start(): Initialize rating providers and validate configuration
        - on_stop(): Cleanup resources
    """

    def __init__(
        self,
        agent: Agent,
        config: RatingConfig | None = None,
    ) -> None:
        """Initialize rating trait.

        Args:
            agent: The agent this trait belongs to.
            config: Rating configuration (None = use defaults).
        """
        super().__init__(agent)
        self.config = config or RatingConfig()
        self._providers: list[ProviderConfig] = []
        self._criteria: dict[str, CriteriaConfig] = {}  # fact_type -> CriteriaConfig
        self._service: RatingService | None = None
        self._backend: AtomicFactsBackend | None = None
        self._batch: BatchConfig | None = None

    def on_start(self) -> None:
        """Initialize rating service and backend.

        Raises:
            TraitNotFoundError: If LearnTrait or LLMTrait are not attached.
        """
        from llm_gent.core.llm import LLMCaller

        from .learn import LearnTrait
        from .llm import LLMTrait

        # Require dependencies
        learn_trait = self.agent.require_trait(LearnTrait)
        llm_trait = self.agent.require_trait(LLMTrait)

        # Initialize rating service and backend with LLMCaller wrapper
        caller = LLMCaller(self.agent.lg, llm_trait.router)
        self._service = RatingService(self.agent.lg, caller)

        # Get schema from LearnTrait config for backend operations
        schema_config = learn_trait.config.get("schema", {})
        schema = str(schema_config.get("name")) if schema_config.get("name") else None
        self._backend = AtomicFactsBackend(self.agent.lg, learn_trait.kelt.database, schema)

        self._parse_config()
        self._log_started()

    def _parse_config(self) -> None:
        """Parse provider, criteria, and batch configurations."""
        parser = ConfigParser(self.agent.lg)
        self._providers = parser.parse_providers(self.config.get("providers", []))
        self._criteria = parser.parse_criteria(self.config.get("models", {}))
        self._batch = parser.parse_batch(self.config.get("batch_size"))

    def _log_started(self) -> None:
        """Log trait started with config summary."""
        self.agent.lg.debug(
            "rating trait started",
            extra={
                "providers": len(self._providers),
                "enabled": sum(1 for p in self._providers if p.enabled),
                "types": list(self._criteria.keys()),
                "auto": self.config.get("auto", False),
            },
        )

    def on_stop(self) -> None:
        """Clean up rating resources."""
        self._providers = []
        self._criteria = {}
        self._service = None
        self._backend = None
        self._batch = None
        self.agent.lg.debug("rating trait stopped")

    # =========================================================================
    # Rating operations
    # =========================================================================

    def get_unrated_count(
        self,
        fact_type: str | None = None,
        category: str | None = None,
    ) -> int:
        """Count unrated facts.

        Args:
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").

        Returns:
            Number of unrated facts matching filters.
        """
        from .learn import LearnTrait

        if not self._backend:
            raise RuntimeError("Rating backend not initialized - call on_start() first")

        learn_trait = self.agent.require_trait(LearnTrait)
        context_key = str(learn_trait.kelt.context.context_key)

        return self._backend.get_unrated_count(context_key, fact_type, category)

    def rate_unrated(
        self,
        limit: int = 10,
        fact_type: str | None = None,
        category: str | None = None,
        provider_index: int = 0,
    ) -> list[RatingResult]:
        """Rate unrated facts automatically using configured providers.

        Args:
            limit: Maximum number of facts to rate.
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").
            provider_index: Index of provider to use (default: 0 = first enabled).

        Returns:
            List of rating results.
        """
        from .learn import LearnTrait

        if not self._backend:
            raise RuntimeError("Rating backend not initialized - call on_start() first")

        provider = self._select_provider(provider_index)
        learn_trait = self.agent.require_trait(LearnTrait)
        context_key = str(learn_trait.kelt.context.context_key)

        results = []
        for fact in self._backend.unrated_facts(context_key, fact_type, category, limit):
            result = self._try_rate_fact(fact, provider)
            if result:
                results.append(result)

        return results

    def rate_batch(
        self,
        limit: int | None = None,
        batch_size: int | None = None,
        fact_type: str | None = None,
        category: str | None = None,
        provider_index: int = 0,
    ) -> tuple[int, int]:
        """Rate unrated facts in batches for efficiency.

        Args:
            limit: Maximum facts to rate (None = all unrated).
            batch_size: Items per batch (None = use config default).
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").
            provider_index: Index of provider to use (default: 0).

        Returns:
            Tuple of (rated_count, failed_count).
        """
        ctx = self._prepare_batch_context(fact_type, provider_index)
        size = batch_size or (self._batch.size if self._batch else 5)
        max_facts = 10000 if limit is None else limit

        facts = list(ctx.backend.unrated_facts(ctx.context_key, fact_type, category, max_facts))
        if not facts:
            return (0, 0)

        return self._rate_facts_in_batches(
            facts, size, ctx.prompt, ctx.criteria, ctx.model, ctx.provider_id
        )

    def _prepare_batch_context(self, fact_type: str | None, provider_index: int) -> _BatchContext:
        """Prepare context for batch rating."""
        from .learn import LearnTrait

        if not self._backend or not self._service:
            raise RuntimeError("Rating not initialized - call on_start() first")

        provider = self._select_provider(provider_index)
        learn_trait = self.agent.require_trait(LearnTrait)
        resolved_type = fact_type or "solution"
        type_criteria = self._criteria.get(resolved_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {resolved_type}")

        backend_type = provider.backend.get("type", "unknown")
        return _BatchContext(
            context_key=str(learn_trait.kelt.context.context_key),
            prompt=type_criteria.prompt,
            criteria=type_criteria.criteria,
            model=provider.model,
            provider_id=f"llm_{provider.model}_{backend_type}",
            backend=self._backend,
        )

    def _rate_facts_in_batches(
        self,
        facts: list[dict[str, Any]],
        batch_size: int,
        prompt_template: str,
        criteria: list[Criteria],
        model: str,
        provider_id: str,
        schema: str | None = None,
    ) -> tuple[int, int]:
        """Rate facts in batches and save results."""
        assert self._service is not None
        total_rated, total_failed = 0, 0

        for i in range(0, len(facts), batch_size):
            batch_facts = facts[i : i + batch_size]
            rated, failed = self._process_batch(
                batch_facts, prompt_template, criteria, model, provider_id, i, schema
            )
            total_rated += rated
            total_failed += failed

        return (total_rated, total_failed)

    def _process_batch(
        self,
        batch_facts: list[dict[str, Any]],
        prompt_template: str,
        criteria: list[Criteria],
        model: str,
        provider_id: str,
        batch_start: int,
        schema: str | None = None,
    ) -> tuple[int, int]:
        """Process a single batch of facts. Returns (rated, failed) counts."""
        assert self._service is not None
        items = [BatchItem(fact=f["id"], content=f["content"]) for f in batch_facts]
        request = BatchRequest(
            items=items,
            prompt_template=prompt_template,
            model=model,
            provider=provider_id,
            criteria=criteria,
        )

        try:
            results = self._service.rate_batch(request)
            for result in results:
                self._save_rating(result, schema)
            return (len(results), len(items) - len(results))
        except Exception as e:
            self.agent.lg.warning(
                "batch rating failed",
                extra={"exception": e, "batch_start": batch_start, "batch_size": len(items)},
            )
            return (0, len(items))

    def _try_rate_fact(
        self, fact: dict[str, Any], provider: ProviderConfig, schema: str | None = None
    ) -> RatingResult | None:
        """Attempt to rate a single fact and save."""
        try:
            result = self.rate_fact(fact["id"], fact["content"], fact["type"], provider)
            self._save_rating(result, schema)
            return result
        except Exception as e:
            self.agent.lg.warning(
                "rating failed",
                extra={
                    "exception": e,
                    "fact_id": fact["id"],
                    "fact_type": fact["type"],
                    "provider_type": provider.provider_type,
                },
            )
            return None

    def rate_fact(
        self,
        fact_id: int,
        content: str,
        fact_type: str,
        provider: ProviderConfig,
    ) -> RatingResult:
        """Rate a specific fact using an LLM provider.

        Args:
            fact_id: ID of the fact to rate.
            content: Content to rate.
            fact_type: Type of fact (e.g., "solution", "prediction").
            provider: Rating provider to use.

        Returns:
            Rating result with scores and reasoning.
        """
        if not self._service:
            raise RuntimeError("Rating service not initialized - call on_start() first")

        # Get type-specific criteria
        type_criteria = self._criteria.get(fact_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {fact_type}")

        # Use rating service to rate content
        # Build unique provider identifier (handles model="auto" case)
        backend_type = provider.backend.get("type", "unknown")
        provider_id = f"llm_{provider.model}_{backend_type}"
        request = RatingRequest(
            fact=fact_id,
            content=content,
            prompt_template=type_criteria.prompt,
            criteria=type_criteria.criteria,
            model=provider.model,
            provider=provider_id,
            temperature=0.3,
        )
        return self._service.rate_content(request)

    def rate_fact_with_all_providers(
        self,
        fact_id: int,
        content: str,
        fact_type: str = "solution",
    ) -> list[RatingResult]:
        """Rate a fact with all enabled providers and save ratings.

        This is the primary method for agents to use for inline rating.
        Rates the content with each enabled LLM provider and saves all ratings.

        Args:
            fact_id: ID of the fact to rate.
            content: Content to rate (e.g., joke text).
            fact_type: Type of fact (default: "solution").

        Returns:
            List of rating results from each provider.
        """
        from .llm import LLMTrait

        # Verify LLMTrait is available
        self.agent.require_trait(LLMTrait)

        results = []
        enabled_providers = [
            p for p in self._providers if p.enabled and p.provider_type == ProviderType.LLM
        ]

        for provider in enabled_providers:
            result = self._rate_and_save_with_provider(fact_id, content, fact_type, provider)
            if result:
                results.append(result)

        return results

    def rate_items(
        self,
        items: list[tuple[int, str]],
        fact_type: str = "solution",
        schema: str | None = None,
    ) -> int:
        """Rate specific items in a batch.

        Args:
            items: List of (fact_id, content) tuples to rate.
            fact_type: Type of facts (default: "solution").
            schema: Optional schema for saving ratings.

        Returns:
            Number of items successfully rated.
        """
        if not items:
            return 0

        ctx = self._prepare_batch_context(fact_type, provider_index=0)
        batch_facts = [{"id": fid, "content": content} for fid, content in items]
        rated, _ = self._process_batch(
            batch_facts, ctx.prompt, ctx.criteria, ctx.model, ctx.provider_id, 0, schema
        )
        return rated

    def _rate_and_save_with_provider(
        self,
        fact_id: int,
        content: str,
        fact_type: str,
        provider: ProviderConfig,
        schema: str | None = None,
    ) -> RatingResult | None:
        """Rate a fact with a single provider and save."""
        try:
            result = self.rate_fact(fact_id, content, fact_type, provider)
            self._save_rating(result, schema)
            self.agent.lg.debug(
                "rated fact",
                extra={"fact_id": fact_id, "model": provider.model, "stars": result.stars},
            )
            return result
        except Exception as e:
            self.agent.lg.warning(
                "rating failed",
                extra={"exception": e, "fact_id": fact_id, "provider": provider.provider_type},
            )
            return None

    def _select_provider(self, index: int = 0) -> ProviderConfig:
        """Select rating provider by index.

        Args:
            index: Index of enabled provider to use (default: 0).

        Returns:
            Selected provider.

        Raises:
            RuntimeError: If no enabled providers configured.
            IndexError: If index out of range.
        """
        enabled_providers = [p for p in self._providers if p.enabled]

        if not enabled_providers:
            raise RuntimeError("No enabled rating providers configured")

        if index >= len(enabled_providers):
            raise IndexError(
                f"Provider index {index} out of range (have {len(enabled_providers)} enabled)"
            )

        return enabled_providers[index]

    def _save_rating(self, result: RatingResult, schema: str | None = None) -> None:
        """Save rating using backend.

        Args:
            result: Rating result to save.
            schema: Optional schema for the rating (uses backend default if not provided).
        """
        if not self._backend:
            raise RuntimeError("Rating backend not initialized - call on_start() first")

        self._backend.save_rating(result=result, source="llm_rater", schema=schema)

    def get_fact_rating(self, fact_id: int, schema: str | None = None) -> int | None:
        """Get the star rating for a specific fact.

        Args:
            fact_id: ID of the fact to look up.
            schema: Optional schema override.

        Returns:
            Star rating (1-5) if rated, None if not rated.
        """
        if not self._backend:
            return None
        return self._backend.get_fact_rating(fact_id, schema)
