"""Factory for creating jokester agent instances."""

from __future__ import annotations

from dataclasses import dataclass

from appinfra import DotDict
from appinfra.log import Logger

from ...core.agent import Factory as BaseFactory
from ...core.traits import TraitName as TN
from ...core.traits.builtin.directive import DirectiveTrait
from ...core.traits.builtin.learn import LearnTrait
from ...core.traits.builtin.llm import LLMTrait
from ...core.traits.builtin.rating import RatingTrait
from ...core.traits.builtin.storage import StorageTrait
from .agent import JokesterAgent
from .cli import JokesterCLI
from .generate import JokeGenerator
from .history import JokeHistory
from .novelty import NoveltyChecker
from .rating import BatchRater
from .storage import Storage
from .variety import VarietyChecker


@dataclass
class Components:
    """Wired components for jokester agent."""

    generator: JokeGenerator
    storage: Storage
    rater: BatchRater | None


class Factory(BaseFactory):
    """Factory for jokester agent.

    Handles all component wiring: NoveltyChecker, JokeGenerator, Storage.
    Agent just uses the pre-configured components.
    """

    agent_class = JokesterAgent
    required_traits = [TN.DIRECTIVE, TN.LLM, TN.LEARN, TN.STORAGE, TN.RATING]
    default_tools = {}
    cli_tool = JokesterCLI

    @classmethod
    def create_components(cls, agent: JokesterAgent) -> Components:
        """Create and wire all components for the agent.

        Called by agent.start() after traits are started.

        Args:
            agent: The agent instance with started traits.

        Returns:
            Components dataclass with generator and storage.

        Raises:
            RuntimeError: If embedder not configured.
        """
        lg = agent._lg
        config = agent.config

        storage_trait = agent.require_trait(StorageTrait)
        # Agent tables are created lazily in Storage when first needed

        learn_trait = agent.require_trait(LearnTrait)
        cls._validate_embedder(lg, learn_trait)

        kelt_config = config.get("kelt", {})
        schema_config = kelt_config.get("schema", {})
        schema_name = schema_config.get("name") or "public"

        generator = cls._create_generator(agent, lg, config, learn_trait)
        storage = Storage(lg, storage_trait, learn_trait, agent.name, schema_name)
        rater = cls._create_rater(agent, lg, config)

        return Components(generator=generator, storage=storage, rater=rater)

    @classmethod
    def _validate_embedder(cls, lg: Logger, learn_trait: LearnTrait) -> None:
        """Validate embedder is configured for novelty checking."""
        if not learn_trait.has_embedder:
            lg.error("embedder not configured - required for novelty checking")
            raise RuntimeError(
                "Jokester agent requires embedder for guaranteed novelty checking. "
                "Configure embedder_url in learn section."
            )

    @classmethod
    def _create_generator(
        cls, agent: JokesterAgent, lg: Logger, config: DotDict, learn_trait: LearnTrait
    ) -> JokeGenerator:
        """Create JokeGenerator with novelty and variety checkers."""
        kelt_config = config.get("kelt", {})
        novelty_config = config.get("novelty", {})

        return JokeGenerator(
            lg=lg,
            llm_trait=agent.require_trait(LLMTrait),
            novelty_checker=cls._create_novelty_checker(
                lg, learn_trait, novelty_config, kelt_config
            ),
            variety_checker=VarietyChecker(lg, novelty_config.get("variety", {})),
            directive_trait=agent.get_trait(DirectiveTrait),
            max_retries=config.get("max_retries", 3),
            denylist=config.get("denylist", []),
            joke_history=JokeHistory.from_config(lg, novelty_config.get("rag", {})),
        )

    @classmethod
    def _create_novelty_checker(
        cls, lg: Logger, learn_trait: LearnTrait, novelty_config: DotDict, kelt_config: DotDict
    ) -> NoveltyChecker:
        """Create NoveltyChecker from config."""
        schema_config = kelt_config.get("schema", {})
        resolved_config = cls._resolve_novelty_config(
            lg, novelty_config, kelt_config.get("reference", {}), schema_config.get("name")
        )
        return NoveltyChecker(lg, learn_trait, resolved_config)

    @classmethod
    def _resolve_novelty_config(
        cls,
        lg: Logger,
        novelty_config: DotDict,
        reference_config: DotDict,
        current_schema: str | None,
    ) -> DotDict:
        """Resolve and validate novelty config, merging reference from kelt config."""
        resolved = DotDict(novelty_config)
        resolved["current_schema"] = current_schema or "public"

        if reference_config:
            model = cls._normalize_reference_model(lg, reference_config.get("model"))
            resolved["reference"] = DotDict(
                {
                    "model": model,
                    "schema": reference_config.get("schema"),
                }
            )

        return resolved

    @classmethod
    def _normalize_reference_model(cls, lg: Logger, value: object) -> str | None:
        """Normalize reference_model config value to string or None."""
        if value is None:
            return None
        if isinstance(value, str):
            return value if value.strip() else None
        lg.warning(
            "reference.model should be a string, coercing",
            extra={"value": value, "type": type(value).__name__},
        )
        return str(value) if value else None

    @classmethod
    def _create_rater(cls, agent: JokesterAgent, lg: Logger, config: DotDict) -> BatchRater | None:
        """Create BatchRater if auto-rating is enabled."""
        rating_config = config.get("rating", {})
        if not rating_config.get("auto", False):
            return None
        rating_trait = agent.get_trait(RatingTrait)
        if not rating_trait:
            return None
        batch_size = rating_config.get("batch_size", 5)
        max_chars = config.get("target", {}).get("max_chars")
        return BatchRater(lg, rating_trait, batch_size, max_chars=max_chars)
