# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Trait factory: builds traits on the platform's shared clients.

Dependency-inversion pattern
----------------------------

Traits receive their external dependencies through the constructor; they
import no resource factory of their own. :class:`PlatformFactory` builds the
shared clients once per platform (the model router, and the memory database,
chat client and embedder) and gives them to this factory, which injects them
into each trait it builds with ``owns_*=False``: the platform closes them,
not the trait. Model access is not a trait: every agent reaches the
platform's router (``platform.llm``, ``Platform.saia``).

Runtime override
----------------

Each dependency-owning trait exposes per-resource fluent overrides
(``.with_chat_client``, ``.with_embedder``, ``.with_database``,
``.with_train_factory``) that return a new instance
bound to the injected resource, detached from the agent's trait registry
and with ``owns_*=False``. Use ``agent.replace_trait(new)`` for a
persistent swap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.log import Logger
from llm_infer.client import ChatClient, EmbeddingClient
from llm_kelt.core import Database

from .base import Trait


if TYPE_CHECKING:
    from ..agent import Agent, Identity
    from . import TraitName
    from .builtin.directive import Directive, DirectiveTrait, MethodTrait
    from .builtin.memory import MemoryConfig, MemoryTrait
    from .builtin.training import TrainingTrait


@dataclass(frozen=True)
class MemoryResources:
    """The memory clients a platform shares: its database, chat client and embedder."""

    database: Database
    chat_client: ChatClient
    embedder: EmbeddingClient | None = None


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
        memory: MemoryResources | None = None,
    ) -> None:
        """Initialize the factory with the platform's shared clients.

        Args:
            lg: Logger instance.
            router: The platform's shared router; rating calls go through it.
            memory_config: The platform's memory settings, or None.
            memory: The platform's shared memory clients, or None without a
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
        """Route to create_memory_trait.

        Merges the platform's memory settings with the agent's ``kelt`` config
        (its schema override).
        """
        memory_config: DotDict | None = (
            DotDict(dict(self._memory_config)) if self._memory_config else None
        )

        agent_kelt = agent.config.get("kelt", {})
        if agent_kelt and memory_config and agent_kelt.get("schema"):
            memory_config["schema"] = agent_kelt["schema"]

        return self.create_memory_trait(agent, agent.identity, memory_config)

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

    def create_memory_trait(
        self, agent: Agent, identity: Identity | None, memory_config: MemoryConfig | None
    ) -> MemoryTrait:
        """Create a MemoryTrait on the platform's shared memory clients.

        The database, chat client and embedder are the platform's; the trait
        does not close them (``owns_chat_client=False``, ``owns_embedder=False``).

        Args:
            agent: Agent the trait is for.
            identity: Agent's identity for memory addressing.
            memory_config: Memory settings (``db``, ``schema``, ``llm``, ...).

        Returns:
            Configured MemoryTrait.

        Raises:
            ConfigError: If memory_config is None or missing required fields, or
                the platform has no memory clients.
        """
        from ..errors import ConfigError
        from .builtin.memory import MemoryTrait

        config = self._build_memory_config(identity, memory_config)
        if self._memory is None:
            raise ConfigError("Memory trait requires a platform built with memory configuration")

        return MemoryTrait(
            agent,
            config,
            database=self._memory.database,
            chat_client=self._memory.chat_client,
            embedder=self._memory.embedder,
            owns_chat_client=False,
            owns_embedder=False,
        )

    def _build_memory_config(
        self, identity: Identity | None, memory_config: MemoryConfig | None
    ) -> MemoryConfig:
        """Validate and normalize the caller's memory config."""
        from ..errors import ConfigError
        from .builtin.memory import MemoryConfig

        if not memory_config:
            raise ConfigError("Memory configuration required but not provided")
        if "db" not in memory_config:
            raise ConfigError("Memory configuration missing required 'db' field")
        if identity is None:
            raise ConfigError("Memory configuration requires identity")

        return MemoryConfig(
            identity=identity,
            schema=memory_config.get("schema"),
            llm=memory_config.get("llm", {}),
            db=memory_config["db"],
            embedder_url=memory_config.get("embedder_url"),
            embedder_model=memory_config.get("embedder_model", "default"),
            embedder_timeout=memory_config.get("embedder_timeout", 30.0),
            training=memory_config.get("training"),
        )

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
