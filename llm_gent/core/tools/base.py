"""Base types for tool use."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool
    """Whether the tool executed successfully."""

    output: str
    """Tool output (stdout for shell, content for file read, etc.)."""

    error: str | None = None
    """Error message if success is False."""

    terminal: bool = False
    """If True, this tool call ends the execution loop immediately."""

    terminal_data: dict[str, Any] | None = None
    """Data to capture when terminal=True (e.g., task completion info)."""


class ToolCall(BaseModel):
    """A tool call from the LLM."""

    id: str
    """Unique identifier for this tool call."""

    name: str
    """Name of the tool to call."""

    arguments: dict[str, Any] = Field(default_factory=dict)
    """Arguments to pass to the tool."""


class ToolCallResult(BaseModel):
    """Result of a tool call, ready to send back to LLM."""

    call_id: str
    """ID of the tool call this result is for."""

    name: str
    """Name of the tool that was called."""

    result: ToolResult
    """The tool's execution result."""


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools that agents can use.

    Tools extend agent capabilities by allowing them to take actions
    and gather information from external systems.

    Example:
        class MyTool:
            name = "my_tool"
            description = "Does something useful"
            parameters = {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First argument"}
                },
                "required": ["arg1"]
            }

            def execute(self, arg1: str) -> ToolResult:
                return ToolResult(success=True, output=f"Got {arg1}")
    """

    @property
    def name(self) -> str:
        """Unique name for this tool."""
        ...

    @property
    def description(self) -> str:
        """Description of what this tool does (shown to LLM)."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments matching the parameters schema.

        Returns:
            ToolResult with success status and output/error.
        """
        ...


class BaseTool:
    """Convenience base class for tools.

    Provides common structure. Subclasses must define name, description,
    parameters, and implement execute().

    Example:
        class GreetTool(BaseTool):
            name = "greet"
            description = "Greet someone by name"
            parameters = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"}
                },
                "required": ["name"]
            }

            def execute(self, name: str) -> ToolResult:
                return ToolResult(success=True, output=f"Hello, {name}!")
    """

    name: str
    description: str
    parameters: dict[str, Any]
    terminal: bool = False
    """If True, calling this tool ends the execution loop."""

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must be implemented by subclasses."""
        raise NotImplementedError

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            Dict in OpenAI's function format for the tools parameter.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
