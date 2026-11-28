# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent factory base class and configuration utilities.

Config schema (agent-config dict passed to ``create`` / ``from_config``):

    {
        "identity":  {"name": str, "context_key": str | None},   # required
        "directive": str | dict,                                  # DirectiveTrait reads
        "method":    str,                                         # MethodTrait reads
        "tools":     {name: config, ...},                         # ToolsTrait reads
        "traits":    {"required": [name, ...]},                   # selects traits to attach
    }

Trait selection: ``config["traits"]["required"]`` (list of trait names, wins)
or the factory-class ``required_traits`` class variable. Each builtin trait
reads its own top-level key from ``agent.config``; nesting under
``traits.<name>`` is NOT recognized. A required name the platform holds a
scoped trait for (``memory``) is bound to the agent: the agent holds its
binding. One whose platform trait is not scoped needs nothing: the agent
finds the platform's instance.

The factory builds agents on a :class:`~llm_gent.core.Platform`:
traits use the platform's shared clients, and agent-config keys override the
platform's defaults. Every agent reaches models through the platform's
router; an agent refuses an ``llm`` key (models are configured on the
platform) and ``kelt.schema`` (the memory schema is the platform's).
"""

from __future__ import annotations

from typing import Any, ClassVar

from appinfra import DotDict

from ..errors import ConfigError
from ..platform import Platform
from ..traits import TraitName
from ..traits.base import ScopedTrait
from ..traits.builtin.memory import MemoryBinding, MemoryTrait
from .agent import Agent
from .helpers import _substitute_in_dict


class Factory:
    """Factory for creating :class:`Agent` instances from configuration.

    Default construction returns a plain :class:`Agent` — no subclass
    required. Advanced setups (custom agent class, factory-level trait
    requirements, default tools) subclass and override the class variables.

    Trait requirements (priority order):
        1. Config: ``config["traits"]["required"] = ["directive", ...]``
        2. Class variable: ``required_traits = [TraitName.MEMORY, ...]``
        3. None — agent starts without traits.

    Tool configuration (priority order):
        1. Config: ``config["tools"] = {name: cfg, ...}``
        2. Class variable: ``default_tools = {name: cfg, ...}``

    Usage::

        from llm_gent import AgentFactory, PlatformFactory

        with PlatformFactory(lg).from_config({}) as platform:
            agent = AgentFactory(platform).from_config({
                "identity": {"name": "hello"},
                "directive": "You are a helpful assistant.",
                "traits": {"required": ["directive"]},
            })

    A prompt agent that answers (``agent.ask(...)``) comes from
    :class:`llm_gent.agents.default.Factory`, which adds SAIA over the
    platform's router and wires traits from the config's sections.

    Custom agent types::

        class CustomFactory(AgentFactory):
            agent_class = MyAgent
            required_traits = [TraitName.MEMORY]

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

    def __init__(self, platform: Platform) -> None:
        """Initialize the factory on a platform.

        Args:
            platform: The platform whose shared clients the agents' traits use.
        """
        self._platform = platform
        self._lg = platform.logger

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
            ConfigError: If agent_class not set, or the agent refuses the
                config (missing ``identity.name``, an ``llm`` key or
                ``kelt.schema``).
        """
        if self.agent_class is None:
            raise ConfigError(f"{self.__class__.__name__} must set agent_class class variable")

        # Apply variable substitutions to config if provided
        if variables:
            config = _substitute_in_dict(config, variables)

        agent = self._new_agent(self.agent_class, config)

        # Create and attach traits (all handled uniformly)
        self._attach_traits(agent, config)

        return agent

    def _new_agent(self, agent_class: type[Agent], config: DotDict) -> Agent:
        """An ``agent_class`` instance on ``config``; it resolves its identity from the config.

        A factory whose agent class takes more than the logger and config
        overrides this.
        """
        return agent_class(lg=self._lg, platform=self._platform, config=config)

    def _attach_traits(self, agent: Agent, config: DotDict) -> None:
        """Create and attach traits for the agent.

        Only creates traits that are explicitly requested via:
        - config['traits']['required']
        - self.required_traits class variable

        A name the platform holds a scoped trait for is bound to the agent
        (``agent.use``); one the platform holds a trait for that is not
        scoped needs nothing; any other is built by the platform's trait
        factory. The factory knows nothing of what any of them does.

        Args:
            agent: Agent instance.
            config: Full config dict from manifest.

        Raises:
            ConfigError: A required trait is neither on the platform nor built
                by its trait factory.
        """
        # Determine which traits to create
        traits_to_create = self._determine_required_traits(config)

        for trait_name in traits_to_create:
            self._add_required_trait(agent, trait_name)

        # Configure tools from YAML or factory defaults
        self._configure_tools(agent, config)

    def _add_required_trait(self, agent: Agent, trait_name: TraitName) -> None:
        """Give ``agent`` the ``trait_name`` trait: the platform's, else one built for it.

        A scoped platform trait is bound to the agent; one that is not scoped
        needs nothing, as the agent finds the platform's instance. A built
        trait reads its config from ``agent.config``.
        """
        shared = self._platform.traits.get_by_name(trait_name)
        if shared is not None:
            if isinstance(shared, ScopedTrait):
                agent.use(type(shared))
                self._lg.debug(
                    "bound trait", extra={"agent": agent.name, "trait": trait_name.value}
                )
            return
        if not self._platform.trait_factory.creates(trait_name):
            raise ConfigError(
                f"trait '{trait_name.value}' is not on this platform: the platform builds it "
                "from its config (memory needs a memory section)"
            )
        agent.add_trait(self._create_trait(trait_name, agent))
        self._lg.debug("created trait", extra={"agent": agent.name, "trait": trait_name.value})

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

    def _create_trait(self, trait_name: TraitName, agent: Agent) -> Any:
        """Create a trait instance using platform.trait_factory.

        Delegates to trait_factory.create() which handles validation and routing.
        The trait reads its config from ``agent.config``, the config the agent
        was built on.

        Args:
            trait_name: Type of trait to create.
            agent: The agent the trait is for; its config and identity are read.

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
        from ..traits.builtin.tools import ToolsTrait

        # Determine which tools to configure
        tools_config = config.get("tools", self.default_tools)
        if not tools_config:
            return

        # Get or create ToolsTrait and populate with configured tools
        tools_trait = agent.get(ToolsTrait)
        is_new = tools_trait is None
        if is_new:
            tools_trait = ToolsTrait(agent)

        assert tools_trait is not None  # Either retrieved or just created
        self._create_and_register_tools(agent, tools_config, tools_trait, agent.get(MemoryTrait))

        # Attach ToolsTrait if any tools were created (only if newly created)
        if is_new and tools_trait.has_tools():
            agent.add_trait(tools_trait)

    def _create_and_register_tools(
        self,
        agent: Agent,
        tools_config: dict[str, dict[str, Any]],
        tools_trait: Any,
        memory: MemoryBinding | None,
    ) -> None:
        """Create and register tools from configuration, memory tools on ``memory``."""
        for tool_name, tool_config in tools_config.items():
            tool = self._platform.tool_factory.create(tool_name, tool_config, memory=memory)
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
