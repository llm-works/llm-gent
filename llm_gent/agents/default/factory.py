"""Factory for creating Agent from configuration.

Creates agents from YAML/dict configuration, wiring up traits automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.agent import Factory as BaseFactory
from ...core.traits.builtin.saia import SAIAConfig, SAIATrait
from .agent import Agent


if TYPE_CHECKING:
    from llm_saia import Backend


class Factory(BaseFactory):
    """Factory for creating prompt-based Agent instances.

    Extends base Factory with SAIA integration for LLM-driven agents.
    Uses base Factory infrastructure for identity, traits, and tools.

    Example:
        platform = PlatformContext.from_config(lg, llm_config, learn_config)
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

    # Prompt agents typically need these traits (can override in YAML)
    required_traits = []  # Let YAML config specify (directive, method, tools optional)
    default_tools = {}  # Tools specified in YAML config

    def __init__(self, platform: Any) -> None:
        """Initialize factory.

        Args:
            platform: Platform context with all resources.
        """
        super().__init__(platform)
        self._backend: Backend | None = None

    def _get_backend(self) -> Backend:
        """Get or create SAIA backend from llm_config."""
        if self._backend is None:
            from llm_infer.client import Factory as LLMClientFactory
            from llm_infer.client import SAIAAdapter

            llm_router = LLMClientFactory(self._lg).from_config(self._platform.llm_config())
            self._backend = SAIAAdapter(client=llm_router)
        return self._backend

    def create(
        self,
        config: dict[str, Any],
        variables: dict[str, str] | None = None,
    ) -> Agent:
        """Create prompt-based Agent from configuration.

        Extends base create() to extract default_prompt from task config.

        Args:
            config: Configuration dictionary.
            variables: Variable substitutions for {{VAR}} patterns.

        Returns:
            Configured Agent ready to start.
        """
        from appinfra import DotDict

        # Extract default_prompt and inject into config.config for base Factory
        task_config = config.get("task", {})
        default_prompt = task_config.get("description", "")

        if "config" not in config:
            config["config"] = {}
        config["config"]["default_prompt"] = default_prompt

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

    def _add_saia_trait(self, agent: Agent, config: dict[str, Any]) -> None:
        """Add SAIATrait for LLM operations."""
        # Build system prompt from identity and method traits
        system_prompt = self._build_system_prompt(agent)

        saia_config = SAIAConfig(
            terminal_tool=config.get("terminal_tool", "complete_task"),
            max_iterations=config.get("max_iterations", 0),
            timeout_secs=config.get("timeout_secs", 0),
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

        # Create HTTPHandler to handle protocol methods
        handler = HTTPHandler(agent)

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

        directive_trait = agent.get_trait(DirectiveTrait)
        if directive_trait is not None:
            parts.append(directive_trait.directive.prompt)

        method_trait = agent.get_trait(MethodTrait)
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
