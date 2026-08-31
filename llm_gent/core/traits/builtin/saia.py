# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""SAIA trait for verb-based LLM operations.

Provides optional SAIA integration for agents that want structured
verb vocabulary (complete, verify, confirm, etc.) rather than raw LLM calls.

This trait is optional - agents can use LLMTrait directly for raw access,
or add SAIATrait for structured operations, or use both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from appinfra.log import Logger
from llm_saia import SAIA, Backend, TaskResult, ToolDef

from ...runnable import ExecutionResult
from ..base import BaseTrait


if TYPE_CHECKING:
    from ...agent import Agent
    from ...tools.registry import Registry


@dataclass
class SAIAConfig:
    """Configuration for SAIATrait."""

    terminal_tool: str = "complete_task"
    """Name of the tool that signals task completion."""

    max_iterations: int = 0
    """Max tool-calling iterations (0 = unlimited)."""

    timeout_secs: float = 0
    """Timeout in seconds (0 = no timeout)."""

    system_prompt: str | None = None
    """Default system prompt for SAIA operations."""


class SAIATrait(BaseTrait):
    """Provides SAIA verb vocabulary to agents.

    SAIA offers structured LLM operations through semantic verbs:
    - complete(): Task execution with tool loop
    - verify(): Check if artifact satisfies predicate
    - confirm(): Yes/no confirmation
    - classify(): Categorize into options
    - critique(): Find counter-arguments
    - extract(): Pull structured data from content
    - And more...

    Dependency ownership: ``backend`` is always injected at construction
    (caller owns its lifecycle). The ``SAIA`` instance is built by the trait
    in ``on_start`` via ``SAIA.builder`` — the trait owns it and drops the
    reference on ``on_stop``. Advanced callers can inject a pre-built
    ``SAIA`` via ``saia=`` at construction; ``owns_saia=False`` keeps the
    reference across stop/start. See ``.with_saia()`` for the immutable-view
    fluent form.

    Example:
        from llm_saia.backends.anthropic import AnthropicBackend

        # Create agent with SAIA
        agent = MyAgent(lg, config)  # Your Agent subclass
        agent.add_trait(SAIATrait(
            agent,
            backend=AnthropicBackend(),
            config=SAIAConfig(terminal_tool="complete_task"),
        ))
        agent.start()

        # Use SAIA verbs
        saia_trait = agent.require_trait(SAIATrait)
        result = await saia_trait.saia.complete("Analyze this code...")
        verified = await saia_trait.saia.verify(output, "is valid JSON")
    """

    def __init__(
        self,
        agent: Agent,
        backend: Backend,
        config: SAIAConfig | None = None,
        *,
        saia: SAIA | None = None,
        owns_saia: bool = False,
    ) -> None:
        """Initialize SAIA trait.

        Args:
            agent: The agent this trait belongs to.
            backend: SAIA backend instance (caller owns lifecycle).
            config: SAIA configuration.
            saia: Pre-built SAIA instance for injection. When None, on_start
                builds one from ``backend`` + ``config`` via ``SAIA.builder``.
            owns_saia: If True, ``on_stop`` drops the instance reference. Set
                automatically to True when the trait builds SAIA in on_start.
                Injected instances default to False so the caller retains
                ownership across stop/start cycles.
        """
        super().__init__(agent)
        self.backend = backend
        self.config = config or SAIAConfig()
        self._saia: SAIA | None = saia
        self._owns_saia = owns_saia

    def on_start(self) -> None:
        """Build SAIA instance from backend + config, unless already injected."""
        if self._saia is not None:
            self.agent.lg.debug("SAIA trait started with injected instance")
            return

        tools, executor = self._get_tools_and_executor()
        self.agent.lg.debug(
            "SAIA tools configured",
            extra={
                "tool_count": len(tools),
                "tool_names": [t.name for t in tools],
                "has_executor": executor is not None,
            },
        )

        self._saia = self._build_saia(tools, executor)
        self._owns_saia = True
        self.agent.lg.debug("SAIA trait started")

    def _build_saia(self, tools: list[ToolDef], executor: Any) -> SAIA:
        """Assemble a SAIA instance from ``self.backend`` + ``self.config``."""
        builder = (
            SAIA.builder()
            .backend(self.backend)
            .max_iterations(self.config.max_iterations)
            .timeout(self.config.timeout_secs)
            .logger(self.agent.lg)
        )
        if tools and executor:
            builder = builder.tools(tools, executor)
        if self.config.system_prompt:
            builder = builder.system(self.config.system_prompt)
        if self.config.terminal_tool:
            builder = builder.terminal_tool(self.config.terminal_tool)
        return builder.build()

    def on_stop(self) -> None:
        """Drop the SAIA reference iff this trait owns it."""
        # Backend cleanup handled by caller (they own the backend)
        if self._owns_saia:
            self._saia = None
            self._owns_saia = False
        self.agent.lg.debug("SAIA trait stopped")

    def with_saia(self, saia: SAIA) -> Self:
        """Return a new trait bound to ``saia``, detached from the registry.

        Immutable-view fluent (mirrors ``LLMTrait.with_router``): ``self``
        stays canonical for ``agent.get_trait(SAIATrait)`` and its instance is
        unchanged. ``owns_saia`` on the returned trait is False; the caller
        retains ownership. For a persistent swap, call
        ``agent.replace_trait(new)``.
        """
        return type(self)(self.agent, self.backend, self.config, saia=saia, owns_saia=False)

    @property
    def saia(self) -> SAIA:
        """Access the SAIA instance.

        Raises:
            RuntimeError: If trait not started.
        """
        if self._saia is None:
            raise RuntimeError("SAIATrait not started - ensure agent.start() was called")
        return self._saia

    def to_execution_result(self, saia_result: TaskResult) -> ExecutionResult:
        """Convert SAIA TaskResult to ExecutionResult.

        Args:
            saia_result: Result from SAIA execution.

        Returns:
            ExecutionResult with converted fields.
        """
        return ExecutionResult(
            success=saia_result.completed,
            content=saia_result.output,
            iterations=saia_result.iterations,
            tokens_used=saia_result.score.total_tokens if saia_result.score else 0,
            trace_id=saia_result.trace.trace_id,
        )

    def _get_tools_and_executor(self) -> tuple[list[ToolDef], Any]:
        """Get tools and executor from ToolsTrait if available."""
        from .tools import ToolsTrait

        tools_trait = self.agent.get_trait(ToolsTrait)
        if tools_trait is None or not tools_trait.has_tools():
            return [], None

        tools = [_tool_to_tooldef(t) for t in tools_trait.registry.list_tools()]
        executor = _create_executor(tools_trait.registry, self.agent.lg)

        return tools, executor


def _tool_to_tooldef(tool: Any) -> ToolDef:
    """Convert llm-gent Tool to SAIA ToolDef."""
    return ToolDef(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def _create_executor(registry: Registry, lg: Logger) -> Any:
    """Create async tool executor for SAIA.

    Runs sync tool.execute() in a thread pool to avoid blocking the event loop.
    """
    import asyncio
    import inspect

    async def executor(name: str, arguments: dict[str, Any]) -> str:
        tool = registry.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"

        try:
            # Run sync tool.execute in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(tool.execute, **arguments)

            # Handle case where tool.execute might be async (future-proofing)
            if inspect.isawaitable(result):
                result = await result

            if result.success:
                return result.output
            return f"Error: {result.error or 'Tool execution failed'}"
        except Exception as e:
            lg.warning("tool execution failed", extra={"tool": name, "exception": e})
            return f"Error executing {name}: {e}"

    return executor
