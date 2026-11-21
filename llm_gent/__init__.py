# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors
#
# Maintained by LLM Works LLC (https://llm-works.ai) and contributors.

"""Agent framework with learning capabilities."""

from importlib.metadata import PackageNotFoundError, version

from llm_kelt.core.types import ScoredEntity
from llm_kelt.memory.atomic import Fact

from .core.agent import Agent, Identity
from .core.agent import Factory as AgentFactory
from .core.cost import Callback as CostCallback
from .core.cost import Tracker as CostTracker
from .core.errors import AgentError, ConfigError
from .core.llm import Backend as LLMBackend
from .core.llm import Caller as LLMCaller
from .core.llm import (
    CompletionResult,
    HTTPBackend,
    Message,
    StructuredOutputError,
)
from .core.llm import Error as LLMError
from .core.platform import Factory as PlatformFactory
from .core.task import Completion as TaskCompletion
from .core.task import Result as TaskResult
from .core.task import Status as TaskStatus
from .core.task import Task
from .core.tools import (
    BaseTool,
    CompleteTaskTool,
    FetchResult,
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
    WebSearchBackendFactory,
    WebSearchTool,
)
from .core.tools.factory import Factory as ToolFactory
from .core.traits import (
    BaseTrait,
    Directive,
    DirectiveTrait,
    HTTPConfig,
    HTTPTrait,
    LLMConfig,
    LLMTrait,
    ManifestNotFoundError,
    MemoryConfig,
    MemoryTrait,
    MethodTrait,
    SAIAConfig,
    SAIATrait,
    ToolsTrait,
    TrainingConfig,
    TrainingTrait,
    Trait,
)
from .core.traits.factory import Factory as TraitFactory
from .schema import (
    SchemaManager,
    SchemaMode,
    SchemaState,
    SchemaStatus,
    SchemaVersionError,
    ensure_schema,
)


try:
    __version__ = version("llm-gent")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    # Agents
    "Agent",
    "Identity",
    # Cost
    "CostCallback",
    "CostTracker",
    # Errors
    "AgentError",
    "ConfigError",
    # Factories
    "AgentFactory",
    "PlatformFactory",
    "ToolFactory",
    "TraitFactory",
    # Tools
    "BaseTool",
    "CompleteTaskTool",  # cq: keep re-exports llm_gent.core.tools' list
    "FetchResult",
    "FileReadTool",
    "FileWriteTool",
    "HTTPFetchTool",
    "RecallTool",
    "RememberTool",
    "ShellTool",
    "WebFetchTool",
    "WebSearchBackend",
    "WebSearchBackendFactory",
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
    "LLMConfig",
    "LLMTrait",
    "ManifestNotFoundError",
    "MemoryConfig",
    "MemoryTrait",
    "MethodTrait",
    "SAIAConfig",
    "SAIATrait",
    "Trait",
    "TrainingConfig",
    "TrainingTrait",
    # HTTP
    "HTTPConfig",
    "HTTPTrait",
    # LLM
    "CompletionResult",
    "HTTPBackend",
    "LLMBackend",
    "LLMCaller",
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
    # Schema management
    "SchemaManager",
    "SchemaMode",
    "SchemaState",
    "SchemaStatus",
    "SchemaVersionError",
    "ensure_schema",
]
