"""Tool factory for creating Tool instances from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger


if TYPE_CHECKING:
    from llm_gent.core.tools.base import BaseTool, Tool, WebSearchBackend
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
        self._web_search_backend: WebSearchBackend | None = None

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
        """Create WebFetchTool — new instance if config provided, cached otherwise."""
        if config:
            from llm_gent.core.tools.builtin import WebFetchTool

            self._web_fetch = WebFetchTool(lg=self._lg, **config)
            return self._web_fetch
        return self._get_or_create_web_fetch()

    def set_web_search_backend(self, backend: WebSearchBackend) -> None:
        """Set the search backend for WebSearchTool creation.

        Must be called before ``create("web_search", ...)``.

        Args:
            backend: Search backend implementation.
        """
        self._web_search_backend = backend

    def _create_web_search(self, config: dict[str, Any]) -> Tool:
        """Create WebSearchTool using an injected search backend.

        Raises:
            ValueError: If no backend has been set via ``set_web_search_backend()``.
        """
        from llm_gent.core.tools.builtin import WebSearchTool

        backend = config.get("backend") or self._web_search_backend
        if backend is None:
            raise ValueError(
                "WebSearchTool requires a search backend. Call "
                "factory.set_web_search_backend(backend) or pass "
                "'backend' in config before creating 'web_search'."
            )
        rest = {k: v for k, v in config.items() if k != "backend"}
        return WebSearchTool(lg=self._lg, backend=backend, **rest)

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
