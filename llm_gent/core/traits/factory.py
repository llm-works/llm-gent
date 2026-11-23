# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Trait factory: builds traits on the platform's shared clients.

Dependency-inversion pattern
----------------------------

Traits receive their external dependencies through the constructor; they
import no resource factory of their own. :class:`PlatformFactory` builds the
shared clients once per platform (the model router, and the memory trait
over the memory database, chat client and embedder) and gives them to this
factory; the platform closes the clients. Model access is not a trait: every
agent reaches the platform's router (``platform.llm``, ``Platform.saia``).

An agent's memory is a view of the platform's one :class:`MemoryTrait`,
bound to the agent's identity (``MemoryTrait.view``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from llm_infer.client import ChatClient

from .base import Trait


if TYPE_CHECKING:
    from ..agent import Agent
    from . import TraitName
    from .builtin.directive import Directive, DirectiveTrait, MethodTrait
    from .builtin.memory import MemoryTrait
    from .builtin.training import TrainingTrait


class Factory:
    """Builds traits on the platform's shared clients.

    Built by :class:`PlatformFactory`; the platform owns the injected clients.

    Example:
        platform = PlatformFactory(lg).from_config({})
        # reads agent.config["directive"]; needs no platform clients
        directive_trait = platform.trait_factory.create(TraitName.DIRECTIVE, agent)
    """

    def __init__(
        self,
        lg: Logger,
        router: ChatClient,
        *,
        memory_config: Mapping[str, Any] | None = None,
        memory: MemoryTrait | None = None,
    ) -> None:
        """Initialize the factory with the platform's shared clients.

        Args:
            lg: Logger instance.
            router: The platform's shared router; rating calls go through it.
            memory_config: The platform's memory settings, or None.
            memory: The platform's shared memory trait, or None without a
                memory database.
        """
        from . import TraitName

        self._lg = lg
        self._router = router
        self._memory_config = memory_config
        self._memory = memory

        self._creators: dict[TraitName, Callable[..., Trait]] = {
            TraitName.DIRECTIVE: self._create_directive,
            TraitName.MEMORY: self._create_memory,
            TraitName.RATING: self._create_rating,
            TraitName.STORAGE: self._create_storage,
            TraitName.METHOD: self._create_method,
            TraitName.TRAINING: self._create_training,
        }

    @property
    def has_memory(self) -> bool:
        """The platform's memory clients are built, so a MemoryTrait can be created."""
        return self._memory is not None

    def create(
        self,
        trait_name: TraitName,
        agent: Agent,
    ) -> Trait:
        """Create a trait by name.

        Args:
            trait_name: Trait type (TraitName enum).
            agent: Agent the trait is for (its config and identity are read).

        Returns:
            Created trait instance.

        Raises:
            ConfigError: If required configuration is missing or the trait type is unknown.
        """
        from ..errors import ConfigError

        creator = self._creators.get(trait_name)
        if not creator:
            raise ConfigError(f"Unknown trait type: {trait_name}")

        return creator(agent)

    def _create_directive(self, agent: Agent) -> DirectiveTrait:
        """Route to create_directive_trait."""
        return self.create_directive_trait(agent, agent.config.get("directive"))

    def _create_memory(self, agent: Agent) -> MemoryTrait:
        """The platform's memory, viewed as the agent's identity.

        A non-empty ``kelt.schema`` on the agent replaces the platform's
        schema: its ``name``, or ``public`` without one. Without it the
        agent shares the platform's schema.

        Raises:
            ConfigError: The platform was built without memory.
        """
        from ..errors import ConfigError

        if self._memory is None:
            raise ConfigError("Memory trait requires a platform built with memory configuration")
        schema = (agent.config.get("kelt") or {}).get("schema")
        name = (schema.get("name") or "public") if schema else None
        return self._memory.view(agent.identity, schema=name)

    def _create_training(self, agent: Agent) -> TrainingTrait:
        """Route to create_training_trait.

        Reads schema and adapters from the platform's memory settings; the
        agent's ``kelt.schema`` overrides the schema.
        """
        source: dict[str, Any] = dict(self._memory_config) if self._memory_config else {}

        agent_kelt = agent.config.get("kelt", {})
        if agent_kelt and agent_kelt.get("schema"):
            source["schema"] = agent_kelt["schema"]

        return self.create_training_trait(agent, source)

    def _create_rating(self, agent: Agent) -> Trait:
        """Route to create_rating_trait."""
        from .builtin.rating import RatingTrait

        return RatingTrait(agent, self._router, agent.config.get("rating"))

    def _create_storage(self, agent: Agent) -> Trait:
        """Route to create_storage_trait."""
        from .builtin.storage import StorageTrait

        return StorageTrait(agent)

    def _create_method(self, agent: Agent) -> MethodTrait:
        """Route to create_method_trait."""
        return self.create_method_trait(agent, agent.config.get("method"))

    def create_directive_trait(
        self, agent: Agent, config: str | dict[str, Any] | Directive | None
    ) -> DirectiveTrait:
        """Create DirectiveTrait from string or dict config.

        Args:
            agent: Agent the trait is for.
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

    def create_training_trait(
        self, agent: Agent, source_config: dict[str, Any] | None
    ) -> TrainingTrait:
        """Create TrainingTrait from a config with schema + adapters keys.

        Args:
            agent: Agent the trait is for.
            source_config: Config dict; ``schema`` (name/enforce) and ``adapters``
                (lora.base_path) are read. Missing keys yield an empty TrainingTrait
                that falls back to the default schema for every lookup.

        Returns:
            Configured TrainingTrait instance.
        """
        from .builtin.training import TrainingConfig, TrainingTrait

        source: dict[str, Any] = dict(source_config) if source_config else {}
        config = TrainingConfig(
            schema=source.get("schema"),
            adapters=source.get("adapters"),
        )
        return TrainingTrait(agent, config)

    def create_method_trait(self, agent: Agent, method: str | None) -> MethodTrait:
        """Create MethodTrait from method string.

        Args:
            agent: Agent the trait is for.
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
