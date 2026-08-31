# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Trait factory for creating Trait instances from configuration.

Dependency-inversion pattern
----------------------------

Traits receive their external dependencies through the constructor; this
factory owns the construction of those dependencies. Traits import no
resource factory of their own.

Concretely:

- ``_create_llm`` builds the ``ChatClient`` router (via
  ``llm_infer.client.Factory``) and passes it to ``LLMTrait`` with
  ``owns_router=True``.
- ``_create_memory`` builds ``PG`` + ``Database`` + ``ChatClient`` +
  ``EmbeddingClient`` and passes them to ``MemoryTrait`` with
  ``owns_chat_client=True`` and ``owns_embedder=True`` (when configured).
- ``_create_training`` leaves ``TrainFactory`` unbuilt — the trait builds
  it lazily on first lookup to preserve the "no I/O in on_start" contract.

The trait's ``on_stop`` closes each factory-built resource iff its
``owns_*`` flag is True; injected resources keep their lifecycle with
whoever passed them in.

Runtime override
----------------

Each dependency-owning trait exposes per-resource fluent overrides
(``.with_router``, ``.with_chat_client``, ``.with_embedder``,
``.with_database``, ``.with_train_factory``) that return a new instance
bound to the injected resource, detached from the agent's trait registry
and with ``owns_*=False``. Use ``agent.replace_trait(new)`` for a
persistent swap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from appinfra import DotDict
from llm_infer.client import ChatClient, EmbeddingClient
from llm_kelt.core import Database


if TYPE_CHECKING:
    from ..agent import Agent, Identity
    from ..platform import PlatformContext
    from . import TraitName
    from .builtin.directive import Directive, DirectiveTrait, MethodTrait
    from .builtin.llm import LLMConfig, LLMTrait
    from .builtin.memory import MemoryConfig, MemoryTrait
    from .builtin.training import TrainingTrait

from .base import Trait


class TraitFactory:
    """Factory for creating Trait instances from configuration.

    Has reference to platform for accessing configs.

    Example:
        trait_factory = TraitFactory(platform)

        # Generic creation
        trait = trait_factory.create(TraitName.LLM, agent)

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
            TraitName.MEMORY: self._create_memory,
            TraitName.RATING: self._create_rating,
            TraitName.STORAGE: self._create_storage,
            TraitName.METHOD: self._create_method,
            TraitName.TRAINING: self._create_training,
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
            result_backends = cast(dict[str, Any], result["backends"])
            for backend_name, backend_override in override["backends"].items():
                if backend_name in result_backends:
                    # Skip None backends (from YAML with all content commented out)
                    if backend_override and result_backends[backend_name] is not None:
                        result_backends[backend_name].update(backend_override)
                else:
                    result_backends[backend_name] = backend_override
        elif "backends" in override:
            result["backends"] = override["backends"]

        # Merge top-level keys (except backends, already handled)
        for key, value in override.items():
            if key != "backends":
                result[key] = value

        return DotDict(result)

    def _create_memory(self, agent: Agent) -> MemoryTrait:
        """Route to create_memory_trait.

        Merges platform learn config with agent-level kelt config (schema, identity overrides).
        Platform config key is still ``learn`` (see PlatformContext.learn_config).
        """
        learn_config_raw = self._platform.learn_config()
        learn_config: DotDict | None = DotDict(learn_config_raw) if learn_config_raw else None

        # Merge agent's kelt.schema config (dict with name/enforce)
        agent_kelt = agent.config.get("kelt", {})
        if agent_kelt and learn_config and agent_kelt.get("schema"):
            learn_config["schema"] = agent_kelt["schema"]

        return self.create_memory_trait(agent, agent.identity, learn_config)

    def _create_training(self, agent: Agent) -> TrainingTrait:
        """Route to create_training_trait.

        Reads schema + adapters from platform learn config (kelt-side settings share
        that block on disk). Agent-level kelt.schema overrides platform schema.
        """
        learn_config_raw = self._platform.learn_config()
        source: dict[str, Any] = dict(learn_config_raw) if learn_config_raw else {}

        agent_kelt = agent.config.get("kelt", {})
        if agent_kelt and agent_kelt.get("schema"):
            source["schema"] = agent_kelt["schema"]

        return self.create_training_trait(agent, source)

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

        Builds the ``ChatClient`` here (owning its lifecycle) and injects it
        into ``LLMTrait``. The trait itself imports no factory. See
        ``LLMTrait`` docstring for the two seams (config-time build vs
        direct injection).

        Args:
            agent: Agent instance that will own this trait.
            llm_config: LLM backend configuration dict.

        Returns:
            Configured LLMTrait instance with a factory-owned router.

        Raises:
            ConfigError: If llm_config is None or invalid.
        """
        from llm_infer.client import Factory as LLMClientFactory

        from ..errors import ConfigError
        from .builtin.llm import LLMTrait

        if not llm_config:
            raise ConfigError("LLM configuration required but not provided")

        router = LLMClientFactory(agent.lg).from_config(llm_config)
        return LLMTrait(agent, router, llm_config, owns_router=True)

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

    def create_memory_trait(
        self, agent: Agent, identity: Identity | None, memory_config: MemoryConfig | None
    ) -> MemoryTrait:
        """Create MemoryTrait with factory-owned database, chat client, and embedder.

        Builds the three external clients here (owning their lifecycle) and
        injects them into ``MemoryTrait``. The trait itself imports no factory.
        See ``MemoryTrait`` docstring for the two seams (config-time build vs
        direct injection via ``.with_*()``).

        Args:
            agent: Agent instance that will own this trait.
            identity: Agent's identity for memory addressing.
            memory_config: Memory configuration dict (db, embedder_url, etc.).

        Returns:
            Configured MemoryTrait with factory-owned chat client and embedder
            (``owns_chat_client=True``, ``owns_embedder=True`` when configured).

        Raises:
            ConfigError: If memory_config is None or missing required fields.
        """
        from .builtin.memory import MemoryTrait

        config = self._build_memory_config(identity, memory_config)
        database = self._build_memory_database(agent, config)
        chat_client, embedder = self._build_memory_clients(agent, config)

        return MemoryTrait(
            agent,
            config,
            database=database,
            chat_client=chat_client,
            embedder=embedder,
            owns_chat_client=True,
            owns_embedder=embedder is not None,
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

    def _build_memory_database(self, agent: Agent, config: MemoryConfig) -> Database:
        """Build the kelt Database wrapper from the memory config."""
        from appinfra.db.pg import PG

        return Database(agent.lg, PG(agent.lg, config.db))

    def _build_memory_clients(
        self, agent: Agent, config: MemoryConfig
    ) -> tuple[ChatClient, EmbeddingClient | None]:
        """Build the chat client and (optional) embedder from the memory config."""
        from llm_infer.client import Factory as LLMClientFactory

        client_factory = LLMClientFactory(agent.lg)
        chat_client = client_factory.from_config(config.get("llm") or DotDict())

        embedder = None
        if config.embedder_url:
            try:
                embedder = client_factory.embeddings(
                    base_url=config.embedder_url,
                    model=config.embedder_model,
                    timeout=config.embedder_timeout,
                )
            except Exception:
                chat_client.close()
                raise
        return chat_client, embedder

    def create_training_trait(
        self, agent: Agent, source_config: dict[str, Any] | None
    ) -> TrainingTrait:
        """Create TrainingTrait from a config with schema + adapters keys.

        Args:
            agent: Agent instance that will own this trait.
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
