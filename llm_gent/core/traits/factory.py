# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Trait factory: builds traits on the platform's shared clients.

Dependency-inversion pattern
----------------------------

Traits receive their external dependencies through the constructor; they
import no resource factory of their own. :class:`PlatformFactory` builds the
shared clients once per platform and gives them to this factory; the
platform closes the clients. Model access is not a trait: every agent
reaches the platform's router (``platform.llm``, ``Platform.saia``).

A trait the platform holds (``platform.traits``) is not built here: an agent
holds a binding of a scoped one (``ScopedTrait.bind``: ``MemoryTrait``,
``RatingTrait``) and reaches a shared one directly (``TrainingTrait``). This
factory builds the traits an agent still builds and holds itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from .base import TraitBinding


if TYPE_CHECKING:
    from ..agent import Agent
    from . import TraitName
    from .builtin.directive import Directive, DirectiveTrait, MethodTrait


class Factory:
    """Builds an agent's own traits on the platform's shared clients.

    Built by :class:`PlatformFactory`; the platform owns the injected clients.

    Example:
        platform = PlatformFactory(lg).from_config({})
        # reads agent.config["directive"]; needs no platform clients
        directive_trait = platform.trait_factory.create(TraitName.DIRECTIVE, agent)
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize the factory.

        Args:
            lg: Logger instance.
        """
        from . import TraitName

        self._lg = lg

        self._creators: dict[TraitName, Callable[..., TraitBinding]] = {
            TraitName.DIRECTIVE: self._create_directive,
            TraitName.STORAGE: self._create_storage,
            TraitName.METHOD: self._create_method,
        }

    def creates(self, trait_name: TraitName) -> bool:
        """Whether :meth:`create` builds a ``trait_name`` trait."""
        return trait_name in self._creators

    def create(
        self,
        trait_name: TraitName,
        agent: Agent,
    ) -> TraitBinding:
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

    def _create_storage(self, agent: Agent) -> TraitBinding:
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
