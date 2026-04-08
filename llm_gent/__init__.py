"""Agent framework with learning capabilities."""

from importlib.metadata import PackageNotFoundError, version

from llm_kelt.core.types import ScoredEntity
from llm_kelt.memory.atomic import Fact

from .core.agent import Agent, Config, Identity
from .core.errors import AgentError, ConfigError
from .core.llm import (
    CompletionResult,
    HTTPBackend,
    LLMBackend,
    LLMError,
    Message,
    StructuredOutputError,
)
from .core.task import Task, TaskCompletion, TaskResult, TaskStatus
from .core.tools import (
    BaseTool,
    CompleteTaskTool,
    FileReadTool,
    FileWriteTool,
    HTTPFetchTool,
    RecallTool,
    Registry,
    RememberTool,
    ShellTool,
    Tool,
    ToolCall,
    ToolCallResult,
    ToolExecutionResult,
    ToolResult,
    WebFetchTool,
    WebSearchBackend,
    WebSearchTool,
)
from .core.tools.factory import ToolFactory
from .core.traits import (
    BaseTrait,
    Directive,
    DirectiveTrait,
    HTTPConfig,
    HTTPTrait,
    LearnConfig,
    LearnTrait,
    LLMConfig,
    LLMTrait,
    MethodTrait,
    SAIAConfig,
    SAIATrait,
    ToolsTrait,
    Trait,
)
from .core.traits.factory import Factory as TraitFactory


try:
    __version__ = version("llm-gent")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    # Agents
    "Agent",
    "Config",
    "Identity",
    # Errors
    "AgentError",
    "ConfigError",
    # Factories
    "ToolFactory",
    "TraitFactory",
    # Tools
    "BaseTool",
    "CompleteTaskTool",
    "FileReadTool",
    "FileWriteTool",
    "HTTPFetchTool",
    "RecallTool",
    "RememberTool",
    "ShellTool",
    "WebFetchTool",
    "WebSearchBackend",
    "WebSearchTool",
    "Tool",
    "ToolCall",
    "ToolCallResult",
    "ToolExecutionResult",
    "Registry",
    "ToolResult",
    # Traits
    "BaseTrait",
    "Directive",
    "DirectiveTrait",
    "LearnConfig",
    "LearnTrait",
    "LLMConfig",
    "LLMTrait",
    "MethodTrait",
    "SAIAConfig",
    "SAIATrait",
    "Trait",
    # HTTP
    "HTTPConfig",
    "HTTPTrait",
    # LLM
    "CompletionResult",
    "HTTPBackend",
    "LLMBackend",
    "LLMError",
    "Message",
    "StructuredOutputError",
    # Memory
    "Fact",
    "ScoredEntity",
    # Tasks
    "Task",
    "TaskCompletion",
    "TaskResult",
    "TaskStatus",
    # Tools trait
    "ToolsTrait",
]
