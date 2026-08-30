# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent factory base class and configuration utilities.

Config schema (agent-config dict passed to ``create`` / ``from_config``):

    {
        "identity":  {"name": str, "context_key": str | None},   # required
        "llm":       str | dict,                                  # LLMTrait reads
        "directive": str | dict,                                  # DirectiveTrait reads
        "method":    str,                                         # MethodTrait reads
        "kelt":      dict,                                        # Memory/Training read
        "tools":     {name: config, ...},                         # ToolsTrait reads
        "traits":    {"required": [name, ...]},                   # selects traits to attach
    }

Trait selection: ``config["traits"]["required"]`` (list of trait names, wins)
or the factory-class ``required_traits`` class variable. Each builtin trait
reads its own top-level key from ``agent.config``; nesting under
``traits.<name>`` is NOT recognized.

Two construction paths:

- ``AgentFactory(lg)`` — bare-Logger tutorial path. Synthesizes an empty
  ``PlatformContext(lg, {})``; platform-level ``llm`` / ``learn`` blocks are
  empty, so the full trait configuration must live in the agent-config.
- ``AgentFactory(platform)`` — advanced/multi-agent path. Uses the supplied
  ``PlatformContext``; agent-config keys override platform-level defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from appinfra import DotDict
from appinfra.log import Logger

from ..errors import ConfigError, TraitNotFoundError
from ..traits import TraitName
from .agent import Agent
from .helpers import _substitute_in_dict


if TYPE_CHECKING:
    from ..platform import PlatformContext


class AgentFactory:
    """Factory for creating :class:`Agent` instances from configuration.

    Default construction returns a plain :class:`Agent` — no subclass
    required. Advanced setups (custom agent class, factory-level trait
    requirements, default tools) subclass and override the class variables.

    Trait requirements (priority order):
        1. Config: ``config["traits"]["required"] = ["llm", ...]``
        2. Class variable: ``required_traits = [TraitName.LLM, ...]``
        3. None — agent starts without traits.

    Tool configuration (priority order):
        1. Config: ``config["tools"] = {name: cfg, ...}``
        2. Class variable: ``default_tools = {name: cfg, ...}``

    Tutorial-shape usage (bare Logger, no subclass, no PlatformContext)::

        from llm_gent import AgentFactory, LLMTrait
        from appinfra.log import create_lg

        lg = create_lg("hello", "info")
        agent = AgentFactory(lg).from_config({
            "identity": {"name": "hello"},
            "llm": {"default": "local", "backends": {...}},
            "directive": "You are a helpful assistant.",
            "traits": {"required": ["llm", "directive"]},
        })
        agent.start()
        result = agent.require_trait(LLMTrait).complete(
            [{"role": "user", "content": "Say hello."}]
        )
        agent.stop()

    Advanced multi-agent usage::

        class CustomFactory(AgentFactory):
            agent_class = MyAgent
            required_traits = [TraitName.LLM, TraitName.MEMORY]

        platform = PlatformContext.from_config(lg, llm_cfg, learn_cfg)
        agent = CustomFactory(platform).create(config)
    """

    # Default agent class; subclasses override to instantiate custom Agent subclasses.
    # Typed as Optional so subclasses can set None to trigger the ConfigError guard.
    agent_class: ClassVar[type[Agent] | None] = Agent

    # Optional: declare required traits at factory level
    required_traits: ClassVar[list[TraitName]] = []

    # Optional: declare default tools at factory level
    default_tools: ClassVar[dict[str, dict[str, Any]]] = {}

    # Optional: CLI tool class for agent-specific commands
    # When set, enables: ./llm-gent.py agent <agent-name> <command>
    cli_tool: ClassVar[type | None] = None

    def __init__(self, platform: Logger | PlatformContext) -> None:
        """Initialize factory with a Logger or PlatformContext.

        Args:
            platform: Either a bare :class:`Logger` (tutorial path — an
                empty :class:`PlatformContext` is synthesized, so all trait
                config must live in agent-config) or a fully-configured
                :class:`PlatformContext` (advanced path — platform-level
                configs are exposed to trait creators). Parameter name
                retained as ``platform`` for compatibility with existing
                keyword callers.
        """
        from ..platform import PlatformContext

        if isinstance(platform, Logger):
            self._platform = PlatformContext(lg=platform, config={})
        else:
            self._platform = platform
        self._lg = self._platform.logger

    def from_config(self, config: dict[str, Any]) -> Agent:
        """Build and wire an agent from a plain dict config.

        Convenience wrapper over :meth:`create`: accepts a raw dict (the
        tutorial-shape entry point) and returns the un-started agent. The
        caller is expected to call ``agent.start()`` / ``agent.stop()``
        (or use ``async with agent``).

        Args:
            config: Agent config (see module docstring for the schema).

        Returns:
            Configured agent, traits attached, not yet started.
        """
        dotdict = config if isinstance(config, DotDict) else DotDict(config)
        return self.create(dotdict)

    def create(
        self,
        config: DotDict,
        variables: dict[str, str] | None = None,
    ) -> Agent:
        """Create agent instance with standard initialization.

        Args:
            config: Full config dict from manifest.
            variables: Optional environment variable substitutions.

        Returns:
            Configured agent instance.

        Raises:
            ConfigError: If agent_class not set or required config fields missing.
        """
        if self.agent_class is None:
            raise ConfigError(f"{self.__class__.__name__} must set agent_class class variable")

        # Apply variable substitutions to config if provided
        if variables:
            config = _substitute_in_dict(config, variables)

        # Instantiate agent (agent resolves identity from config)
        agent = self.agent_class(
            lg=self._lg,
            config=config,
        )

        # Create and attach traits (all handled uniformly)
        self._attach_traits(agent, config)

        return agent

    def _attach_traits(self, agent: Agent, config: DotDict) -> None:
        """Create and attach traits for the agent.

        Only creates traits that are explicitly requested via:
        - config['traits']['required']
        - self.required_traits class variable

        Args:
            agent: Agent instance.
            config: Full config dict from manifest.

        Raises:
            TraitNotFoundError: If required traits cannot be created.
        """
        # Determine which traits to create
        traits_to_create = self._determine_required_traits(config)

        # Create each requested trait using platform.trait_factory
        for trait_name in traits_to_create:
            trait = self._create_trait(trait_name, agent, config)
            agent.add_trait(trait)
            self._lg.debug("created trait", extra={"agent": agent.name, "trait": trait_name.value})

        # Validate all required traits were created
        self._validate_trait_requirements(agent, config, traits_to_create)

        # Configure tools from YAML or factory defaults
        self._configure_tools(agent, config)

    def _determine_required_traits(self, config: DotDict) -> list[TraitName]:
        """Determine which traits to create for this agent.

        Args:
            config: Full config dict from manifest.

        Returns:
            List of trait names to create.
        """
        traits_config = config.get("traits", {})
        if "required" in traits_config:
            # Config takes priority - convert strings to TraitName
            try:
                return [TraitName(name) for name in traits_config["required"]]
            except ValueError as e:
                raise ConfigError(
                    f"Unknown trait in traits.required: {e}. "
                    f"Valid traits: {[t.value for t in TraitName]}"
                ) from e
        elif self.required_traits:
            # Use factory class variable
            return self.required_traits
        else:
            # No traits specified - create none
            return []

    def _create_trait(self, trait_name: TraitName, agent: Agent, config: DotDict) -> Any:
        """Create a trait instance using platform.trait_factory.

        Delegates to trait_factory.create() which handles validation and routing.

        Args:
            trait_name: Type of trait to create.
            agent: Agent instance (needed for agent-specific config like identity).
            config: Full config dict from manifest.

        Returns:
            Created trait instance.

        Raises:
            ConfigError: If required configuration is missing.
            ValueError: If trait type is unknown.
        """
        return self._platform.trait_factory.create(
            trait_name=trait_name,
            agent=agent,
        )

    def _build_trait_class_map(self) -> dict[TraitName, type]:
        """Build mapping from trait names to trait classes for validation."""
        from ..traits.builtin.directive import DirectiveTrait, MethodTrait
        from ..traits.builtin.http import HTTPTrait
        from ..traits.builtin.llm import LLMTrait
        from ..traits.builtin.memory import MemoryTrait
        from ..traits.builtin.rating import RatingTrait
        from ..traits.builtin.saia import SAIATrait
        from ..traits.builtin.storage import StorageTrait
        from ..traits.builtin.tools import ToolsTrait
        from ..traits.builtin.training import TrainingTrait

        return {
            TraitName.DIRECTIVE: DirectiveTrait,
            TraitName.LLM: LLMTrait,
            TraitName.MEMORY: MemoryTrait,
            TraitName.RATING: RatingTrait,
            TraitName.STORAGE: StorageTrait,
            TraitName.METHOD: MethodTrait,
            TraitName.HTTP: HTTPTrait,
            TraitName.SAIA: SAIATrait,
            TraitName.TOOLS: ToolsTrait,
            TraitName.TRAINING: TrainingTrait,
        }

    def _validate_trait_requirements(
        self, agent: Agent, config: DotDict, required: list[TraitName]
    ) -> None:
        """Validate that required traits were successfully created.

        Args:
            agent: Agent instance with traits attached.
            config: Full config dict from manifest.
            required: List of trait names that should have been created.

        Raises:
            TraitNotFoundError: If required traits are missing.
        """
        if not required:
            return

        trait_class_map = self._build_trait_class_map()

        # Validate each required trait is attached
        missing: list[str] = []
        for trait_name in required:
            trait_class = trait_class_map.get(trait_name)
            if trait_class is None or agent.get_trait(trait_class) is None:
                missing.append(trait_name.value)

        if missing:
            raise TraitNotFoundError(
                f"{agent.name} requires traits {missing} but they were not created. "
                f"Check platform configuration (e.g., 'learn' section for MemoryTrait)."
            )

    def _configure_tools(self, agent: Agent, config: DotDict) -> None:
        """Configure tools for the agent from YAML or factory defaults.

        Tool configuration priority:
        1. config['tools'] (YAML config)
        2. self.default_tools class variable
        3. No tools (empty ToolsTrait)

        Args:
            agent: Agent instance.
            config: Full config dict from manifest.
        """
        from ..traits.builtin.memory import MemoryTrait
        from ..traits.builtin.tools import ToolsTrait

        # Determine which tools to configure
        tools_config = config.get("tools", self.default_tools)
        if not tools_config:
            return

        # Bind MemoryTrait to platform.tool_factory if available
        memory_trait = agent.get_trait(MemoryTrait)
        if memory_trait:
            self._platform.tool_factory.set_memory_trait(memory_trait)

        try:
            # Get or create ToolsTrait and populate with configured tools
            tools_trait = agent.get_trait(ToolsTrait)
            is_new = tools_trait is None
            if is_new:
                tools_trait = ToolsTrait(agent)

            assert tools_trait is not None  # Either retrieved or just created
            self._create_and_register_tools(agent, tools_config, tools_trait)

            # Attach ToolsTrait if any tools were created (only if newly created)
            if is_new and tools_trait.has_tools():
                agent.add_trait(tools_trait)
        finally:
            # Clear memory_trait to avoid leaking state between agents
            if memory_trait:
                self._platform.tool_factory.set_memory_trait(None)

    def _create_and_register_tools(
        self, agent: Agent, tools_config: dict[str, dict[str, Any]], tools_trait: Any
    ) -> None:
        """Create and register tools from configuration."""
        for tool_name, tool_config in tools_config.items():
            tool = self._platform.tool_factory.create(tool_name, tool_config)
            if tool is None:
                self._lg.warning(
                    "tool could not be created",
                    extra={"agent": agent.name, "tool": tool_name},
                )
                continue

            tools_trait.register(tool)
            self._lg.debug(
                "created tool for agent",
                extra={"agent": agent.name, "tool": tool_name},
            )
