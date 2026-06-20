"""Runtime infrastructure for operating agents.

This package provides the operational infrastructure for running agents:
- Core: Orchestrates agent subprocess/thread lifecycle
- AgentRegistry: Manages agent configurations
- AgentRunner: External agent gateway to the swarm
- ManagedAgentRunner: Internal runner for hub-managed agents
- Handler: Protocol for request dispatch callbacks
- AgentHandle/AgentInfo: Agent state tracking
- State: Lifecycle states (from appinfra.service)
"""

from appinfra.service import State

from .core import Core
from .handle import AgentHandle, AgentInfo
from .handler import Handler
from .messages import AgentMessage, AgentResponse, MessageType
from .registry import AgentRegistry
from .runner import AgentRunner, ManagedAgentRunner


# Backward compatibility alias
AgentState = State


__all__ = [
    # Core types
    "AgentHandle",
    "AgentInfo",
    "AgentRegistry",
    "AgentRunner",
    "Core",
    "Handler",
    "ManagedAgentRunner",
    # State (appinfra.service.State with alias)
    "State",
    "AgentState",  # Deprecated alias for State
    # Messages
    "AgentMessage",
    "AgentResponse",
    "MessageType",
]
