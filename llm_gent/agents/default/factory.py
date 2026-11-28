# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Factory for creating Agent from configuration.

Creates agents from YAML/dict configuration, wiring up traits automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from llm_kelt.conversation import DbSessionStorage, SessionStorage

from ...core.agent import Agent as CoreAgent
from ...core.agent import Factory as BaseFactory
from ...core.errors import ConfigError
from ...core.llm.caller import Caller
from ...core.platform import Platform
from ...core.tools.names import ToolName
from ...core.traits import TraitName
from ...core.traits.builtin.memory import MemoryTrait
from ...core.traits.builtin.saia import SAIAConfig, SAIATrait
from .agent import Agent, ChatAgent


if TYPE_CHECKING:
    from llm_saia import Backend


_SECTION_TRAITS = (("directive", TraitName.DIRECTIVE), ("method", TraitName.METHOD))
"""Config sections a prompt agent declares, and the trait each one attaches."""

_MEMORY_TOOLS = (ToolName.REMEMBER, ToolName.RECALL)
"""Tools that need a MemoryTrait."""


class Factory(BaseFactory):
    """Factory for creating prompt-based Agent instances.

    Extends base Factory with SAIA integration for LLM-driven agents.
    Uses base Factory infrastructure for identity, traits, and tools.

    Example:
        platform = PlatformFactory(lg).from_config({"llm": llm_config, "memory": memory_config})
        factory = Factory(platform)

        config = {
            "identity": {"name": "explorer"},
            "directive": "You are a codebase exploration agent.",
            "method": "Think step by step...",
            "tools": {"shell": {}, "read_file": {}},
        }

        agent = factory.create(config)
        agent.start()
    """

    agent_class = Agent  # Use default prompt-based agent

    # A prompt agent gets these traits plus the ones its config's sections declare
    # (see _determine_required_traits); traits.required in the config overrides both.
    required_traits = []
    default_tools = {}  # Tools specified in YAML config

    def __init__(self, platform: Platform) -> None:
        """Initialize factory.

        Args:
            platform: The platform whose shared router SAIA runs on.
        """
        super().__init__(platform)
        self._backend: Backend | None = None
        self._session_storage = _session_storage(platform)

    def _new_agent(self, agent_class: type[CoreAgent], config: DotDict) -> CoreAgent:
        """A prompt agent; with a ``chat`` section, a :class:`ChatAgent` whose sessions
        persist where the platform's memory lives.

        The default ``Agent`` class becomes ``ChatAgent``; a factory's own class
        must be one. A factory's own ``ChatAgent`` class has chat without the section.

        Raises:
            ConfigError: A ``chat`` section for a factory's own class without chat.
        """
        if "chat" not in config and not issubclass(agent_class, ChatAgent):
            return super()._new_agent(agent_class, config)
        chat_class = ChatAgent if agent_class is Agent else agent_class
        if not issubclass(chat_class, ChatAgent):
            raise ConfigError(f"{agent_class.__name__} has no chat; the chat section needs one")
        return chat_class(self._lg, self._platform, config, session_storage=self._session_storage)

    def _get_backend(self) -> Backend:
        """The SAIA backend over the platform's shared router, made on first use."""
        if self._backend is None:
            from llm_infer.client import SAIAAdapter

            self._backend = SAIAAdapter(client=self._platform.llm)
        return self._backend

    def from_config(self, config: dict[str, Any]) -> Agent:
        """Build a prompt agent from a plain dict config; see :meth:`create`."""
        return self.create(config)

    def create(
        self,
        config: dict[str, Any],
        variables: dict[str, str] | None = None,
    ) -> Agent:
        """Create prompt-based Agent from configuration.

        Extends base create() to set the agent's ``default_prompt`` (what
        ``run_once()`` runs) from ``task.description``.

        Args:
            config: Configuration dictionary.
            variables: Variable substitutions for {{VAR}} patterns.

        Returns:
            Configured Agent ready to start.
        """
        from appinfra import DotDict

        # Agent reads default_prompt from the top level of its config
        task_config = config.get("task", {})
        config["default_prompt"] = task_config.get("description", config.get("default_prompt", ""))

        # Convert to DotDict once for consistent usage
        dotdict_config = DotDict(config)

        # Use base Factory.create() - handles identity, traits, tools, and variable substitution
        agent = super().create(dotdict_config, variables=variables)

        # Add SAIA trait and event handlers (prompt-agent specific)
        self._add_saia_trait(agent, dotdict_config)  # type: ignore[arg-type]
        self._configure_event_handlers(agent, dotdict_config)  # type: ignore[arg-type]

        # Add HTTP trait if configured (handled separately from base factory)
        self._add_http_trait(agent, dotdict_config)  # type: ignore[arg-type]

        return agent  # type: ignore[return-value]

    def _determine_required_traits(self, config: DotDict) -> list[TraitName]:
        """The factory's ``required_traits`` plus the traits the config's sections declare.

        ``directive`` and ``method`` sections attach their traits. Memory tools
        (``remember`` / ``recall``) attach a MemoryTrait when the platform has
        memory; without it those tools are skipped, as the tool factory reports.
        A config's ``traits.required`` replaces both.
        """
        if "required" in config.get("traits", {}):
            return super()._determine_required_traits(config)
        traits = [name for section, name in _SECTION_TRAITS if config.get(section)]
        tools = config.get("tools") or {}
        if self._platform.traits.has(MemoryTrait) and any(t in tools for t in _MEMORY_TOOLS):
            traits.append(TraitName.MEMORY)
        return list(dict.fromkeys([*self.required_traits, *traits]))

    def _add_saia_trait(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add SAIATrait for LLM operations."""
        # Build system prompt from identity and method traits
        system_prompt = self._build_system_prompt(agent)

        saia_config = SAIAConfig(
            terminal_tool=config.get("terminal_tool", "complete_task"),
            max_iterations=_task_setting(config, "max_iterations"),
            timeout_secs=_task_setting(config, "timeout_secs"),
            system_prompt=system_prompt,
        )
        agent.add_trait(SAIATrait(agent, self._get_backend(), saia_config))

    def _add_http_trait(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add HTTPTrait with HTTPHandler if HTTP is configured.

        Args:
            agent: Agent instance to add HTTP trait to.
            config: Configuration dictionary (may contain 'http' section).
        """
        http_config_dict = config.get("http")
        if not http_config_dict:
            return

        from ...core.traits.builtin.http import HTTPConfig, HTTPTrait
        from .http import HTTPHandler

        # Create HTTPHandler to handle protocol methods; its model calls use the platform's router
        handler = HTTPHandler(agent, Caller(self._lg, self._platform.llm))

        # Parse HTTP config (dict with host, port, title, description)
        http_config = HTTPConfig(**http_config_dict)

        # Create and attach HTTPTrait with handler
        agent.add_trait(HTTPTrait(agent, config=http_config, handler=handler))
        self._lg.debug("http trait added", extra={"agent": agent.name})

    def _build_system_prompt(self, agent: Agent) -> str | None:
        """Build system prompt from identity and method traits."""
        # Deferred import to avoid circular dependency
        from ...core.traits.builtin.directive import DirectiveTrait, MethodTrait

        parts: list[str] = []

        directive_trait = agent.get(DirectiveTrait)
        if directive_trait is not None:
            parts.append(directive_trait.directive.prompt)

        method_trait = agent.get(MethodTrait)
        if method_trait is not None:
            parts.append(f"## Method\n{method_trait.method}")

        if not parts:
            return None

        return "\n\n".join(parts)

    def _configure_event_handlers(self, agent: Agent, config: dict[str, Any]) -> None:
        """Configure event handlers from YAML config.

        Registers default handlers (chronological for schedule, semantic for question).
        If events are specified in config, use those settings instead.

        Args:
            agent: The agent to configure handlers for.
            config: Agent configuration dict (may contain 'events' section).
        """
        events_config = config.get("events", {})

        # Default event configurations
        defaults = {
            "schedule": {"recall_strategy": "chronological", "recall_limit": 5},
            "question": {"recall_strategy": "semantic", "recall_limit": 5},
        }

        # Merge defaults with user config (user overrides defaults)
        for event_name, default_config in defaults.items():
            user_config = events_config.get(event_name, {})
            event_config = {**default_config, **user_config}
            handler = self._create_event_handler(agent, event_name, event_config)
            agent._dispatcher.on(event_name, handler)

    def _create_event_handler(
        self, agent: Agent, event_name: str, event_config: dict[str, Any]
    ) -> Any:
        """Create an event handler from configuration.

        Args:
            agent: The agent this handler is for.
            event_name: Name of the event (schedule, question).
            event_config: Event configuration dict with recall_strategy, etc.

        Returns:
            Async handler function.
        """
        recall_strategy = event_config.get("recall_strategy", "chronological")
        recall_limit = event_config.get("recall_limit", 5)

        async def handler(**kwargs: Any) -> Any:
            """Custom event handler from YAML config."""
            task = kwargs.get("task") or kwargs.get("question", "")
            return await agent.handle_task(
                task=task,
                recall_strategy=recall_strategy,
                recall_limit=recall_limit,
            )

        return handler


def _session_storage(platform: Platform) -> SessionStorage | None:
    """Session storage in the memory database and schema; ``None`` on a platform without memory."""
    memory = platform.traits.get(MemoryTrait)
    if memory is None:
        return None
    database = memory.database.scoped(memory.default_schema)
    return DbSessionStorage(platform.logger, database.session)


def _task_setting(config: dict[str, Any], key: str) -> Any:
    """``task.<key>`` (where the agent YAML sets it), else top-level ``<key>``, else 0 (no limit)."""
    task_config = config.get("task") or {}
    return task_config.get(key, config.get(key, 0))
