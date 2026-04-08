"""Tool factory for creating Tool instances from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger


if TYPE_CHECKING:
    from llm_gent.core.tools.base import BaseTool, Tool
    from llm_gent.core.tools.builtin.web_fetch import WebFetchTool
    from llm_gent.core.traits.builtin.learn import LearnTrait


class ToolFactory:
    """Factory for creating Tool instances from configuration.

    Supports built-in tool types and custom tool registration.

    Example:
        factory = ToolFactory(lg)

        # Create built-in tools
        shell = factory.create(ToolFactory.SHELL, {"allowed_commands": ["ls", "grep"]})
        reader = factory.create(ToolFactory.READ_FILE, {"allowed_paths": ["/home"]})

        # Register custom tool type
        factory.register("my_tool", lambda config: MyTool(**config))
        custom = factory.create("my_tool", {"option": "value"})
    """

    # Canonical tool type constants
    SHELL = "shell"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    HTTP_FETCH = "http_fetch"
    WEB_FETCH = "web_fetch"
    WEB_SEARCH = "web_search"
    COMPLETE_TASK = "complete_task"
    REMEMBER = "remember"
    RECALL = "recall"

    # Built-in tool type aliases
    _ALIASES: dict[str, str] = {
        "file_read": READ_FILE,
        "file_write": WRITE_FILE,
        "fetch": HTTP_FETCH,
    }

    def __init__(self, lg: Logger) -> None:
        """Initialize factory with built-in tool creators.

        Args:
            lg: Logger instance, passed through to tools that need it.
        """
        self._lg = lg
        self._custom_creators: dict[str, Callable[[dict[str, Any]], Tool]] = {}
        self._learn_trait: LearnTrait | None = None
        self._web_fetch: WebFetchTool | None = None
        self._web_fetch_shared = False  # True once handed to another tool

    # ------------------------------------------------------------------
    # Built-in tool creators
    # ------------------------------------------------------------------

    def _create_simple(self, tool_type: str, config: dict[str, Any]) -> Tool:
        """Create a tool whose __init__ takes only config kwargs.

        Note: these tools predate the Logger-injection pattern used by web
        tools.  Adding ``lg`` here is a follow-up task, not a regression.
        """
        from llm_gent.core.tools.builtin import (
            FileReadTool,
            FileWriteTool,
            HTTPFetchTool,
            ShellTool,
        )

        classes: dict[str, type[BaseTool]] = {
            self.SHELL: ShellTool,
            self.READ_FILE: FileReadTool,
            self.WRITE_FILE: FileWriteTool,
            self.HTTP_FETCH: HTTPFetchTool,
        }
        return classes[tool_type](**config)

    def _get_or_create_web_fetch(self) -> WebFetchTool:
        """Return the cached WebFetchTool, creating a default lazily."""
        if self._web_fetch is None:
            from llm_gent.core.tools.builtin import WebFetchTool

            self._web_fetch = WebFetchTool(lg=self._lg)
        return self._web_fetch

    def _create_web_fetch(self, config: dict[str, Any]) -> Tool:
        """Create WebFetchTool — new instance if config provided, cached otherwise.

        When custom config is supplied the new instance replaces the cached
        one so that a subsequent ``web_search`` creation inherits the same
        security constraints (allowed_domains, block_private_ips, etc.).

        Raises:
            ValueError: If reconfiguring after the instance was already shared
                with another tool (e.g., WebSearchTool).
        """
        if config:
            if self._web_fetch_shared:
                raise ValueError(
                    "Cannot reconfigure WebFetchTool after it was already shared "
                    "with WebSearchTool. Create web_fetch before web_search."
                )
            from llm_gent.core.tools.builtin import WebFetchTool

            self._web_fetch = WebFetchTool(lg=self._lg, **config)
            return self._web_fetch
        return self._get_or_create_web_fetch()

    def _create_web_search(self, config: dict[str, Any]) -> Tool:
        """Create WebSearchTool using the agent's WebFetchTool.

        If ``web_fetch`` was already created (possibly with custom security
        config), that instance is reused.  Otherwise a default is created.
        """
        from llm_gent.core.tools.builtin import WebSearchTool

        web_fetch = self._get_or_create_web_fetch()
        tool = WebSearchTool(lg=self._lg, web_fetch=web_fetch, **config)
        self._web_fetch_shared = True
        return tool

    def _create_complete_task(self) -> Tool:
        """Create CompleteTaskTool (takes no config)."""
        from llm_gent.core.tools.builtin import CompleteTaskTool

        return CompleteTaskTool()

    def _create_memory_tool(self, tool_type: str) -> Tool | None:
        """Create remember or recall tool.

        Returns None if LearnTrait not available (tool will be skipped).
        """
        from llm_gent.core.tools.builtin import RecallTool, RememberTool

        if self._learn_trait is None:
            return None

        if tool_type == self.REMEMBER:
            return RememberTool(self._learn_trait)
        return RecallTool(self._learn_trait)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_learn_trait(self, learn_trait: LearnTrait | None) -> None:
        """Set LearnTrait for memory tools (remember/recall).

        Args:
            learn_trait: LearnTrait instance or None.
        """
        self._learn_trait = learn_trait

    def register(self, tool_type: str, creator: Callable[[dict[str, Any]], Tool]) -> None:
        """Register a custom tool type.

        Args:
            tool_type: Tool type identifier.
            creator: Callable that takes config dict and returns Tool.

        Example:
            factory.register("my_tool", lambda c: MyTool(**c))
        """
        self._custom_creators[tool_type] = creator

    def create(self, tool_type: str, config: dict[str, Any] | None = None) -> Tool | None:
        """Create a tool from type and configuration.

        Args:
            tool_type: Tool type (e.g., "shell", "read_file").
            config: Tool-specific configuration.

        Returns:
            Configured Tool instance, or None if tool cannot be created
            (e.g., memory tools without LearnTrait).

        Raises:
            ValueError: If tool type is unknown.
        """
        config = config or {}
        canonical = self._ALIASES.get(tool_type, tool_type)

        if canonical in (self.REMEMBER, self.RECALL):
            return self._create_memory_tool(canonical)
        if canonical in (self.SHELL, self.READ_FILE, self.WRITE_FILE, self.HTTP_FETCH):
            return self._create_simple(canonical, config)
        if canonical == self.WEB_FETCH:
            return self._create_web_fetch(config)
        if canonical == self.WEB_SEARCH:
            return self._create_web_search(config)
        if canonical == self.COMPLETE_TASK:
            return self._create_complete_task()

        # Custom-registered tools
        custom = self._custom_creators.get(canonical)
        if custom is not None:
            return custom(config)

        raise ValueError(f"Unknown tool type: {tool_type}")
