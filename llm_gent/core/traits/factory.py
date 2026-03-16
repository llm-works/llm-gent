"""Trait factory for creating Trait instances from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra import DotDict


if TYPE_CHECKING:
    from llm_gent.core.agent import Agent, Identity
    from llm_gent.core.platform import PlatformContext
    from llm_gent.core.traits import TraitName
    from llm_gent.core.traits.builtin.directive import Directive, DirectiveTrait, MethodTrait
    from llm_gent.core.traits.builtin.learn import LearnConfig, LearnTrait
    from llm_gent.core.traits.builtin.llm import LLMConfig, LLMTrait

from .base import Trait


class Factory:
    """Factory for creating Trait instances from configuration.

    Has reference to platform for accessing configs.

    Example:
        trait_factory = Factory(lg, platform)

        # Generic creation
        trait = trait_factory.create("llm", agent, agent_config={}, identity=agent.identity)

        # Or direct
        llm_trait = trait_factory.create_llm_trait(agent, platform.llm_config())
    """

    def __init__(self, platform: PlatformContext) -> None:
        """Initialize factory.

        Args:
            platform: Platform context for accessing configs and logger.
        """
        from . import TraitName

        self._platform = platform
        self._lg = platform.logger

        # Map trait types to creator functions
        self._creators: dict[TraitName, Callable[..., Trait]] = {
            TraitName.DIRECTIVE: self._create_directive,
            TraitName.LLM: self._create_llm,
            TraitName.LEARN: self._create_learn,
            TraitName.RATING: self._create_rating,
            TraitName.STORAGE: self._create_storage,
            TraitName.METHOD: self._create_method,
        }

    def create(
        self,
        trait_name: TraitName,
        agent: Agent,
    ) -> Trait:
        """Create a trait by name - uses mapping to route to specific creators.

        Args:
            trait_name: Trait type (TraitName enum).
            agent: Agent instance that will own this trait (accesses agent.config and agent.identity).

        Returns:
            Created trait instance.

        Raises:
            ConfigError: If required configuration missing or trait type unknown.
        """
        from ..errors import ConfigError

        creator = self._creators.get(trait_name)
        if not creator:
            raise ConfigError(f"Unknown trait type: {trait_name}")

        return creator(agent)

    def _create_directive(self, agent: Agent) -> DirectiveTrait:
        """Route to create_directive_trait."""
        return self.create_directive_trait(agent, agent.config.get("directive"))

    def _create_llm(self, agent: Agent) -> LLMTrait:
        """Route to create_llm_trait.

        Merges agent-level llm config (from agent YAML) with global llm config.

        Agent config formats:
            llm: anthropic                    # Select backend (simplest)
            llm: { default: anthropic }       # Explicit default
            llm: { default: local, backends: { local: { adapter: x } } }  # Select + override
        """
        llm_config: DotDict = DotDict(self._platform.llm_config())

        agent_llm_config = agent.config.get("llm")
        if agent_llm_config:
            # String shorthand: "llm: anthropic" -> select that backend
            if isinstance(agent_llm_config, str):
                llm_config = DotDict({**dict(llm_config), "default": agent_llm_config})
            else:
                llm_config = self._merge_llm_config(llm_config, agent_llm_config)

        return self.create_llm_trait(agent, llm_config)

    def _merge_llm_config(self, base: DotDict, override: dict[str, Any]) -> DotDict:
        """Merge agent-level llm config into global config.

        Performs deep merge at the backends level, so agent can override
        specific backend settings (like adapter) without replacing
        the entire backend config.
        """
        import copy

        result = copy.deepcopy(dict(base))

        # Merge backends
        if "backends" in override and "backends" in result:
            for backend_name, backend_override in override["backends"].items():
                if backend_name in result["backends"]:
                    # Skip None backends (from YAML with all content commented out)
                    if backend_override:
                        result["backends"][backend_name].update(backend_override)
                else:
                    result["backends"][backend_name] = backend_override
        elif "backends" in override:
            result["backends"] = override["backends"]

        # Merge top-level keys (except backends, already handled)
        for key, value in override.items():
            if key != "backends":
                result[key] = value

        return DotDict(result)

    def _create_learn(self, agent: Agent) -> LearnTrait:
        """Route to create_learn_trait.

        Merges platform learn config with agent-level kelt config (schema, identity overrides).
        """
        learn_config_raw = self._platform.learn_config()
        learn_config: DotDict | None = DotDict(learn_config_raw) if learn_config_raw else None

        # Merge agent's kelt.schema config (dict with name/enforce)
        agent_kelt = agent.config.get("kelt", {})
        if agent_kelt and learn_config and agent_kelt.get("schema"):
            learn_config["schema"] = agent_kelt["schema"]

        return self.create_learn_trait(agent, agent.identity, learn_config)

    def _create_rating(self, agent: Agent) -> Trait:
        """Route to create_rating_trait."""
        from .builtin.rating import RatingTrait

        rating_config = agent.config.get("rating")
        # RatingTrait uses LLMTrait for LLM access (no separate client needed)
        return RatingTrait(agent, rating_config)

    def _create_storage(self, agent: Agent) -> Trait:
        """Route to create_storage_trait."""
        from .builtin.storage import StorageTrait

        return StorageTrait(agent)

    def _create_method(self, agent: Agent) -> MethodTrait:
        """Route to create_method_trait."""
        return self.create_method_trait(agent, agent.config.get("method"))

    def create_llm_trait(self, agent: Agent, llm_config: LLMConfig | None) -> LLMTrait:
        """Create LLMTrait with LLM backend configuration.

        Args:
            agent: Agent instance that will own this trait.
            llm_config: LLM backend configuration dict.

        Returns:
            Configured LLMTrait instance.

        Raises:
            ConfigError: If llm_config is None or invalid.
        """
        from ..errors import ConfigError
        from .builtin.llm import LLMTrait

        if not llm_config:
            raise ConfigError("LLM configuration required but not provided")

        return LLMTrait(agent, llm_config)

    def create_directive_trait(
        self, agent: Agent, config: str | dict[str, Any] | Directive | None
    ) -> DirectiveTrait:
        """Create DirectiveTrait from string or dict config.

        Args:
            agent: Agent instance that will own this trait.
            config: Directive prompt string or dict with 'prompt' key.

        Returns:
            Configured DirectiveTrait instance.

        Raises:
            ConfigError: If config is None or invalid.
        """
        from ..errors import ConfigError
        from .builtin.directive import Directive, DirectiveTrait

        if not config:
            raise ConfigError("Directive configuration required but not provided")

        if isinstance(config, str):
            directive = Directive(prompt=config)
        elif isinstance(config, dict):
            directive = Directive(**config)
        elif isinstance(config, Directive):
            directive = config
        else:
            raise ConfigError(
                f"Directive config must be str, dict, or Directive, got {type(config).__name__}"
            )

        return DirectiveTrait(agent, directive)

    def create_learn_trait(
        self, agent: Agent, identity: Identity | None, learn_config: LearnConfig | None
    ) -> LearnTrait:
        """Create LearnTrait with agent-specific identity.

        Args:
            agent: Agent instance that will own this trait.
            identity: Agent's identity for memory addressing.
            learn_config: Learning configuration dict (db, embedder_url, etc.).

        Returns:
            Configured LearnTrait instance.

        Raises:
            ConfigError: If learn_config is None or missing required fields.
        """
        from ..errors import ConfigError
        from .builtin.learn import LearnConfig, LearnTrait

        if not learn_config:
            raise ConfigError("Learning configuration required but not provided")

        if "db" not in learn_config:
            raise ConfigError("Learning configuration missing required 'db' field")

        config = LearnConfig(
            identity=identity,
            schema=learn_config.get("schema"),
            llm=learn_config.get("llm", {}),
            db=learn_config["db"],
            embedder_url=learn_config.get("embedder_url"),
            embedder_model=learn_config.get("embedder_model", "default"),
            embedder_timeout=learn_config.get("embedder_timeout", 30.0),
            training=learn_config.get("training"),
            adapters=learn_config.get("adapters"),
        )
        return LearnTrait(agent, config)

    def create_method_trait(self, agent: Agent, method: str | None) -> MethodTrait:
        """Create MethodTrait from method string.

        Args:
            agent: Agent instance that will own this trait.
            method: Method/approach description.

        Returns:
            Configured MethodTrait.

        Raises:
            ConfigError: If method is None or empty.
        """
        from ..errors import ConfigError
        from .builtin.directive import MethodTrait

        if not method:
            raise ConfigError("Method configuration required but not provided")

        return MethodTrait(agent, method)
