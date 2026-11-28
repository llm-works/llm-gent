# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Rating trait for automated LLM-based content rating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.log import Logger
from llm_infer.client import ChatClient

from ...llm import Caller as LLMCaller
from ...memory.rating import (
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
from ...memory.rating import Request as RatingRequest
from ...memory.rating import Result as RatingResult
from ...memory.rating import Service as RatingService
from .memory import MemoryTrait


if TYPE_CHECKING:
    from ...agent import Agent


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

    prompt: str
    criteria: list[Criteria]
    model: str
    provider_id: str


class RatingTrait:
    """Automated LLM-based content rating on the platform: the shared service and store.

    A :class:`ScopedTrait`: each agent rates its own facts by its own criteria.
    The platform builds one RatingTrait when memory is configured: a rating
    service over the platform's model router, and the backend that reads
    facts and stores ratings in memory's database, on the platform's memory
    schema (prepared when the platform builds memory). It holds nothing per
    agent. An agent holds its own :class:`RatingBinding` from :meth:`bind`:
    its context key and the rating config from its ``rating`` section.
    ``agent.require(RatingTrait)`` returns that binding, which carries the
    rating operations.

    Example:
        platform = PlatformFactory(lg).from_config({"memory": {"db": {...}}})
        agent = AgentFactory(platform).from_config({
            "identity": {"name": "jokester"},
            "traits": {"required": ["rating"]},
            "rating": {
                "providers": [{"type": "llm", "backend": {"model": "claude-sonnet-4.5"}}],
                "models": {"atomic": {"solution": {
                    "prompt": "Rate this joke...",
                    "criteria": [{"name": "humor", "description": "Is it funny?"}],
                }}},
            },
        })
        results = agent.require(RatingTrait).rate_fact_with_all_providers(
            fact_id=123, content="Why did the..."
        )
    """

    trait_name = "rating"

    def __init__(self, lg: Logger, router: ChatClient, memory: MemoryTrait) -> None:
        """Build the shared rating service and backend.

        Args:
            lg: Logger instance.
            router: The platform's model router rating calls go through.
            memory: The platform's memory trait, whose database and schema
                hold the facts and their ratings.
        """
        self._lg = lg
        self._service = RatingService(lg, LLMCaller(lg, router))
        self._backend = AtomicFactsBackend(lg, memory.database, memory.default_schema)

    def bind(self, agent: Agent) -> RatingBinding:
        """``agent``'s rating: its identity's context key and its ``rating`` config; no I/O."""
        return RatingBinding(agent.lg, self, agent.identity.context_key, agent.config.get("rating"))

    def on_start(self) -> None:
        """Nothing to start: the service and backend are built and hold no connection."""

    def on_stop(self) -> None:
        """Nothing to stop: the platform closes the router and the database."""

    @property
    def service(self) -> RatingService:
        """The rating service over the platform's router."""
        return self._service

    @property
    def backend(self) -> AtomicFactsBackend:
        """The store that reads facts and saves ratings, on the platform's memory schema."""
        return self._backend


class RatingBinding:
    """An agent's rating: its context key and its rating config; the agent holds it.

    Built by :meth:`RatingTrait.bind`. The service and backend are the
    trait's; the binding holds the agent's partition and the providers,
    criteria and batch size its config names.

    Capabilities:
        - rate_unrated(): Find and rate unrated content automatically
        - rate_fact(): Rate a specific fact by ID
        - rate_fact_with_all_providers(): Rate with all providers (for inline rating)
        - get_unrated_count(): Count unrated facts
        - Multi-provider support: Use multiple LLMs for rating comparison
    """

    def __init__(
        self,
        lg: Logger,
        trait: RatingTrait,
        context_key: str,
        config: RatingConfig | None = None,
    ) -> None:
        """Bind ``trait`` to an agent's context key and parse its rating config.

        Args:
            lg: The agent's logger.
            trait: The platform's rating trait.
            context_key: The agent's context key; its facts are the ones rated.
            config: The agent's rating config (None = defaults).
        """
        self._lg = lg
        self._trait = trait
        self._context_key = str(context_key)
        self.config = config or RatingConfig()
        parser = ConfigParser(lg)
        self._providers: list[ProviderConfig] = parser.parse_providers(
            self.config.get("providers", [])
        )
        self._criteria: dict[str, CriteriaConfig] = parser.parse_criteria(
            self.config.get("models", {})
        )
        self._batch: BatchConfig | None = parser.parse_batch(self.config.get("batch_size"))

    def on_start(self) -> None:
        """Nothing to start: the binding holds no resource."""

    def on_stop(self) -> None:
        """Nothing to release: the binding holds no resource."""

    @property
    def trait(self) -> RatingTrait:
        """The platform's rating trait this binding is of."""
        return self._trait

    @property
    def context_key(self) -> str:
        """The agent's context key: the facts this binding rates."""
        return self._context_key

    # =========================================================================
    # Rating operations
    # =========================================================================

    def get_unrated_count(
        self,
        fact_type: str | None = None,
        category: str | None = None,
    ) -> int:
        """Count the agent's unrated facts.

        Args:
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").

        Returns:
            Number of unrated facts matching filters.
        """
        return self._trait.backend.get_unrated_count(self._context_key, fact_type, category)

    def rate_unrated(
        self,
        limit: int = 10,
        fact_type: str | None = None,
        category: str | None = None,
        provider_index: int = 0,
    ) -> list[RatingResult]:
        """Rate the agent's unrated facts automatically using configured providers.

        Args:
            limit: Maximum number of facts to rate.
            fact_type: Filter by fact type (e.g., "solution").
            category: Filter by category (e.g., "joke").
            provider_index: Index of provider to use (default: 0 = first enabled).

        Returns:
            List of rating results.
        """
        provider = self._select_provider(provider_index)
        facts = self._trait.backend.unrated_facts(self._context_key, fact_type, category, limit)
        results = []
        for fact in facts:
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
        """Rate the agent's unrated facts in batches for efficiency.

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

        facts = list(
            self._trait.backend.unrated_facts(self._context_key, fact_type, category, max_facts)
        )
        if not facts:
            return (0, 0)

        return self._rate_facts_in_batches(
            facts, size, ctx.prompt, ctx.criteria, ctx.model, ctx.provider_id
        )

    def _prepare_batch_context(self, fact_type: str | None, provider_index: int) -> _BatchContext:
        """Prepare context for batch rating."""
        provider = self._select_provider(provider_index)
        resolved_type = fact_type or "solution"
        type_criteria = self._criteria.get(resolved_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {resolved_type}")

        backend_type = provider.backend.get("type", "unknown")
        return _BatchContext(
            prompt=type_criteria.prompt,
            criteria=type_criteria.criteria,
            model=provider.model,
            provider_id=f"llm_{provider.model}_{backend_type}",
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
        items = [BatchItem(fact=f["id"], content=f["content"]) for f in batch_facts]
        request = BatchRequest(
            items=items,
            prompt_template=prompt_template,
            model=model,
            provider=provider_id,
            criteria=criteria,
        )

        try:
            results = self._trait.service.rate_batch(request)
            for result in results:
                self._save_rating(result, schema)
            return (len(results), len(items) - len(results))
        except Exception as e:
            self._lg.warning(
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
            self._lg.warning(
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
        type_criteria = self._criteria.get(fact_type)
        if not type_criteria:
            raise ValueError(f"No criteria configured for fact type: {fact_type}")

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
        return self._trait.service.rate_content(request)

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
            self._lg.debug(
                "rated fact",
                extra={"fact_id": fact_id, "model": provider.model, "stars": result.stars},
            )
            return result
        except Exception as e:
            self._lg.warning(
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
        """Save a rating through the trait's backend (its schema unless ``schema`` is given)."""
        self._trait.backend.save_rating(result=result, source="llm_rater", schema=schema)

    def get_fact_rating(self, fact_id: int, schema: str | None = None) -> int | None:
        """Get the star rating for a specific fact.

        Args:
            fact_id: ID of the fact to look up.
            schema: Optional schema override.

        Returns:
            Star rating (1-5) if rated, None if not rated.
        """
        return self._trait.backend.get_fact_rating(fact_id, schema)
