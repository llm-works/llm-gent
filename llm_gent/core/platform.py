# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Platform context - central resource management for llm-gent platform.

The PlatformContext provides:
- Platform configuration (LLM backends, learning database, etc.)
- Shared logger

Agents create their own trait/tool instances using factories and platform config.
Catalogs of available traits/tools are defined in their packages (ALL_TRAITS, ALL_TOOLS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from .tools.factory import ToolFactory
from .traits.factory import TraitFactory


if TYPE_CHECKING:
    from .traits.builtin.llm import LLMConfig
    from .traits.builtin.memory import MemoryConfig


class PlatformContext:
    """Central container for all platform resources.

    This encapsulates all shared resources and provides them to agent factories.
    Think of it as the "platform runtime environment" that apps (agents) run within.

    Lifecycle:
        1. Platform creates PlatformContext during startup
        2. Initializes all shared resources (DB, LLM clients, traits)
        3. Passes context to factories when creating agents
        4. Factories access resources from context
        5. Platform calls cleanup() on shutdown

    Example:
        # Platform startup
        platform = PlatformContext.from_config(server_config, logger)

        # Creating agents
        factory = Factory(platform)  # Factory gets full platform access
        agent = factory.create(agent_config)

        # Platform shutdown
        platform.cleanup()
    """

    def __init__(
        self,
        lg: Logger,
        config: dict[str, Any],
    ) -> None:
        """Initialize platform context.

        Args:
            lg: Platform logger.
            config: Platform configuration dict with llm/learn configs.
        """
        self._lg = lg
        self._config = config

        # Create factories for trait/tool creation
        self._tool_factory = ToolFactory(lg)
        self._trait_factory = TraitFactory(platform=self)

        self._lg.info("platform context initialized")

    @classmethod
    def from_config(
        cls,
        lg: Logger,
        llm_config: LLMConfig,
        learn_config: MemoryConfig | None = None,
    ) -> PlatformContext:
        """Create platform context from configuration.

        Args:
            lg: Platform logger.
            llm_config: LLM backend configuration.
            learn_config: Optional learning backend configuration.

        Returns:
            Initialized PlatformContext with all resources ready.
        """
        # Store configs
        config = {
            "llm": llm_config,
            "learn": learn_config,
        }

        return cls(lg=lg, config=config)

    @property
    def logger(self) -> Logger:
        """Platform logger."""
        return self._lg

    @property
    def config(self) -> dict[str, Any]:
        """Platform configuration."""
        return self._config

    @property
    def trait_factory(self) -> TraitFactory:
        """Trait factory for creating trait instances."""
        return self._trait_factory

    @property
    def tool_factory(self) -> ToolFactory:
        """Tool factory for creating tool instances."""
        return self._tool_factory

    def llm_config(self) -> dict[str, Any]:
        """Get LLM configuration.

        Returns:
            LLM backend configuration dict.
        """
        return self._config.get("llm", {})  # type: ignore[no-any-return]

    def learn_config(self) -> dict[str, Any] | None:
        """Get learning configuration.

        Returns:
            Learning backend configuration dict, or None if not configured.
        """
        return self._config.get("learn")

    def cleanup(self) -> None:
        """Cleanup platform resources.

        Called during platform shutdown to release resources.
        Currently minimal - agents manage their own trait lifecycles.
        """
        self._lg.info("cleaning up platform context")

        # Future: Close shared resources (message bus, etc.)

        self._lg.info("platform context cleaned up")

    def __repr__(self) -> str:
        """String representation of platform context."""
        has_llm = "llm" in self._config
        has_learn = self._config.get("learn") is not None
        return f"PlatformContext(llm={has_llm}, learn={has_learn})"
