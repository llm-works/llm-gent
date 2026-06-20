"""Tool execution for LLM ↔ tool interaction.

This module provides:
- SimpleToolExecutor: Pure tool execution without loop/decisions (used by SAIA)
- ToolExecutionResult: Result type for backward compatibility
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..llm import Message
from .base import ToolCall, ToolCallResult, ToolResult


if TYPE_CHECKING:
    from appinfra.log import Logger

    from .registry import Registry


class ToolExecutionResult(BaseModel):
    """Result from tool execution loop.

    Note: This class is kept for backward compatibility. New code should
    use TaskResult from llm_gent.core.task instead.
    """

    content: str
    """Final response content from the LLM."""

    tool_calls: list[ToolCallResult]
    """All tool calls made during execution."""

    messages: list[Message]
    """Complete message history including all tool interactions."""

    iterations: int
    """Number of LLM round-trips."""

    total_tokens: int
    """Total tokens used across all iterations."""

    terminal_data: dict[str, Any] | None = None
    """Data from terminal tool call (e.g., task completion info)."""


class SimpleToolExecutor:
    """Pure tool execution - no loop, no decisions.

    Executes tool calls and returns results. All decision logic about
    what to execute and when to stop is handled by SAIA.

    Example:
        executor = SimpleToolExecutor(lg, registry)
        results = executor.execute(tool_calls, parse_errors={})
        for result in results:
            if result.result.terminal:
                # Terminal tool was called
                break
    """

    def __init__(self, lg: Logger, registry: Registry) -> None:
        """Initialize executor.

        Args:
            lg: Logger instance.
            registry: Registry of available tools.
        """
        self._lg = lg
        self._registry = registry

    def execute(
        self,
        tool_calls: list[ToolCall],
        parse_errors: dict[str, str],
    ) -> list[ToolCallResult]:
        """Execute tool calls and return results.

        Args:
            tool_calls: Tool calls to execute.
            parse_errors: Dict mapping call_id to parse error message. If a call
                has a parse error, return the error instead of executing.

        Returns:
            List of ToolCallResult in the same order as input.
        """
        tool_names = [c.name for c in tool_calls]
        self._lg.trace("executing tools", extra={"tools": tool_names})

        results = []
        for call in tool_calls:
            if call.id in parse_errors:
                result = ToolResult(success=False, output="", error=parse_errors[call.id])
            else:
                result = self._execute_single_tool(call)
            results.append(ToolCallResult(call_id=call.id, name=call.name, result=result))

        return results

    def _execute_single_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self._registry.get(call.name)
        if tool is None:
            self._lg.warning("unknown tool", extra={"tool": call.name})
            return ToolResult(success=False, output="", error=f"Unknown tool: {call.name}")

        args_preview = self._truncate_args(call.arguments)
        self._lg.debug("executing tool", extra={"tool": call.name, "input": args_preview})

        try:
            result = tool.execute(**call.arguments)
            self._log_tool_result(call.name, result)
            return result
        except Exception as e:
            self._lg.warning("tool error", extra={"tool": call.name, "exception": e})
            return ToolResult(success=False, output="", error=f"Tool execution error: {e}")

    def _log_tool_result(self, tool_name: str, result: ToolResult) -> None:
        """Log tool execution result."""
        if result.success:
            output_preview = self._truncate_str(result.output, 300)
            self._lg.trace(
                "tool result",
                extra={"tool": tool_name, "success": True, "output": output_preview},
            )
        else:
            self._lg.trace(
                "tool result",
                extra={"tool": tool_name, "success": False, "error": result.error},
            )

    def _truncate_str(self, s: str | None, max_len: int = 200) -> str:
        """Truncate string for logging."""
        if not s:
            return ""
        return s[:max_len] + "..." if len(s) > max_len else s

    def _truncate_args(self, args: dict[str, Any], max_value_len: int = 100) -> dict[str, Any]:
        """Truncate argument values for logging."""
        result = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > max_value_len:
                result[k] = v[:max_value_len] + "..."
            else:
                result[k] = v
        return result
