"""Core agent framework components.

This package contains the core abstractions for building agents:
- runnable: Runnable interface for executable agents
- agent: Base Agent class with trait composition
- config: Agent configuration
- conversation: Conversation management utilities
- errors: Core exceptions
- factory: Factory classes for creating traits and tools
- task: Task definitions and results
- traits: Composable agent behaviors (SAIA, LLM, HTTP, Learn, etc.)
- tools: Tool definitions and registry
- llm: LLM backend abstractions
"""

from .agent import Agent, Config, ExecutionResult
from .conv import (
    Compactor,
    Conversation,
    SlidingWindowCompactor,
    SummarizingCompactor,
)
from .conv import (
    Config as ConversationConfig,
)
from .errors import (
    AgentError,
    ConfigError,
    DuplicateTraitError,
    TraitError,
    TraitNotFoundError,
)
from .platform import PlatformContext
from .runnable import Runnable
from .task import Task, TaskResult
from .tools.factory import ToolFactory
from .traits.factory import Factory as TraitFactory


__all__ = [
    # Runnable
    "ExecutionResult",
    "Runnable",
    # Agents
    "Agent",
    "Config",
    # Errors
    "AgentError",
    "ConfigError",
    "DuplicateTraitError",
    "TraitError",
    "TraitNotFoundError",
    # Platform
    "PlatformContext",
    # Factories
    "ToolFactory",
    "TraitFactory",
    # Conversation (from llm-kelt)
    "Compactor",
    "ConversationConfig",
    "Conversation",
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    # Tasks
    "Task",
    "TaskResult",
]
