"""Tool factory for creating Tool instances from configuration."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger


if TYPE_CHECKING:
    from ..traits.builtin.learn import LearnTrait
    from .base import BaseTool, Tool
    from .builtin.web.backend import WebSearchBackend, WebSearchBackendFactory
    from .builtin.web.fetch import WebFetchTool


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
        from .builtin import (
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
            from .builtin import WebFetchTool

            self._web_fetch = WebFetchTool(lg=self._lg)
        return self._web_fetch

    def _create_web_fetch(self, config: dict[str, Any]) -> Tool:
        """Create WebFetchTool — new instance if config provided, cached otherwise."""
        if config:
            from .builtin import WebFetchTool

            self._web_fetch = WebFetchTool(lg=self._lg, **config)
            return self._web_fetch
        return self._get_or_create_web_fetch()

    # Lazy-loaded backend factories: type name -> (module_path, class_name)
    _lazy_backends: dict[str, tuple[str, str]] = {
        "brave": (
            "llm_gent.core.tools.builtin.web.brave",
            "Factory",
        ),
        "serper": (
            "llm_gent.core.tools.builtin.web.serper",
            "Factory",
        ),
    }

    def set_web_search_backend(self, backend: WebSearchBackend) -> None:
        """Set the search backend for WebSearchTool creation.

        Must be called before ``create("web_search", ...)``.

        Args:
            backend: Search backend implementation.
        """
        self._web_search_backend = backend

    def _resolve_backend(self, backend_config: dict[str, Any]) -> WebSearchBackend:
        """Resolve a search backend from a config dict.

        Supports two resolution modes:

        - ``type``: look up a built-in backend factory by name (e.g. ``brave``).
        - ``factory``: dynamic-import a dotted path to a
          :class:`WebSearchBackendFactory` subclass.

        Remaining keys under ``config`` are passed to the factory's
        ``create()`` method.

        Args:
            backend_config: Dict with ``type`` or ``factory`` key and
                optional ``config`` sub-dict.

        Returns:
            Constructed backend instance.

        Raises:
            ValueError: If neither ``type`` nor ``factory`` is specified,
                both are specified, or the type name is unknown.
            ImportError: If the factory path cannot be imported.
        """
        from appinfra import DotDict

        has_factory = "factory" in backend_config
        has_type = "type" in backend_config

        if has_factory and has_type:
            raise ValueError("web_search backend config must specify 'type' or 'factory', not both")

        config = DotDict(backend_config.get("config") or {})

        if has_factory:
            factory_cls = self._import_factory(backend_config["factory"])
            return factory_cls.create(self._lg, config)

        if has_type:
            factory_cls = self._ensure_backend_registered(backend_config["type"])
            return factory_cls.create(self._lg, config)

        raise ValueError("web_search backend config must include 'type' or 'factory' key")

    def _ensure_backend_registered(self, backend_type: str) -> type[WebSearchBackendFactory]:
        """Look up a backend factory by type name, lazy-loading if needed.

        Args:
            backend_type: Registered backend type name (e.g. ``"brave"``).

        Returns:
            The factory class.

        Raises:
            ValueError: If the type name is not registered.
        """
        if backend_type in self._lazy_backends:
            module_path, class_name = self._lazy_backends[backend_type]
            from .builtin.web.backend import validated_factory

            module = importlib.import_module(module_path)
            return validated_factory(getattr(module, class_name), f"built-in type {backend_type!r}")

        available = ", ".join(sorted(self._lazy_backends))
        raise ValueError(
            f"Unknown web_search backend type: {backend_type!r}. Available: {available}"
        )

    @staticmethod
    def _import_factory(dotted_path: str) -> type[WebSearchBackendFactory]:
        """Dynamic-import a :class:`WebSearchBackendFactory` from a dotted path.

        Args:
            dotted_path: Fully-qualified path like
                ``"llm_private.websearch.ddg.Factory"``.

        Returns:
            The imported factory class.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the class is not found in the module.
        """
        module_path, _, class_name = dotted_path.rpartition(".")
        if not module_path:
            raise ImportError(f"Invalid factory path {dotted_path!r}: expected 'module.ClassName'")
        from .builtin.web.backend import validated_factory

        module = importlib.import_module(module_path)
        return validated_factory(getattr(module, class_name), dotted_path)

    def _create_web_search(self, config: dict[str, Any]) -> Tool | None:
        """Create WebSearchTool using a search backend.

        The backend can be provided in three ways (checked in order):

        1. ``config["backend"]`` is already a :class:`WebSearchBackend`
           instance (programmatic injection, backward-compatible).
        2. ``config["backend"]`` is a dict with ``type`` or ``factory``
           key — resolved via :meth:`_resolve_backend`.
        3. A backend was previously set via :meth:`set_web_search_backend`.

        Returns ``None`` when no backend has been configured, allowing the
        agent factory to skip web_search gracefully.
        """
        from .builtin.web.backend import WebSearchBackend as _WSB
        from .builtin.web.search import WebSearchTool

        backend_value = config.get("backend")
        if backend_value is None:
            backend_value = self._web_search_backend

        if backend_value is None:
            self._lg.warning(
                "web_search skipped: no search backend configured — call "
                "factory.set_web_search_backend() or pass 'backend' in config"
            )
            return None

        if isinstance(backend_value, _WSB):
            backend = backend_value
        elif isinstance(backend_value, dict):
            backend = self._resolve_backend(backend_value)
        else:
            raise ValueError(
                f"web_search 'backend' must be a WebSearchBackend instance "
                f"or a config dict, got {type(backend_value).__name__}"
            )

        rest = {k: v for k, v in config.items() if k != "backend"}
        return WebSearchTool(lg=self._lg, backend=backend, **rest)

    def _create_complete_task(self) -> Tool:
        """Create CompleteTaskTool (takes no config)."""
        from .builtin import CompleteTaskTool

        return CompleteTaskTool()

    def _create_memory_tool(self, tool_type: str) -> Tool | None:
        """Create remember or recall tool.

        Returns None if LearnTrait not available (tool will be skipped).
        """
        from .builtin import RecallTool, RememberTool

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
