"""Agent-specific message types for runtime communication.

Defines the message protocol between Core (main process) and AgentService
(subprocess/thread). Uses appinfra.service.Message as the base but adds
agent-specific semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class MessageType(StrEnum):
    """Message type constants for agent communication.

    Using StrEnum ensures:
    - Typos are caught at development time
    - IDE autocomplete works
    - Values are still strings for serialization
    """

    # Lifecycle messages
    SHUTDOWN = "shutdown"
    STARTED = "started"
    ERROR = "error"

    # Request/response types
    ASK = "ask"
    ASK_RESPONSE = "ask_response"
    FEEDBACK = "feedback"
    FEEDBACK_RESPONSE = "feedback_response"
    GET_INSIGHTS = "get_insights"
    INSIGHTS_RESPONSE = "insights_response"

    # Scheduled execution
    RUN_CYCLE = "run_cycle"
    CYCLE_COMPLETE = "cycle_complete"
    CYCLE_ERROR = "cycle_error"


@dataclass
class AgentMessage:
    """Message for agent communication.

    Compatible with appinfra.service.Channel which expects messages
    with an `id` attribute for request/response correlation.

    Attributes:
        id: Unique message identifier.
        type: Message type (e.g., 'ask', 'shutdown', 'cycle').
        payload: Message-specific data.
    """

    type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class AgentResponse:
    """Response to an agent request.

    Attributes:
        id: Unique response identifier.
        type: Response type (e.g., 'ask_response').
        request_id: ID of the request this responds to (for correlation).
        success: Whether the request was handled successfully.
        error: Error message if success is False.
        payload: Response data.
    """

    type: str = ""
    request_id: str = ""
    success: bool = True
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)


# Backward compatibility aliases
Message = AgentMessage
Request = AgentMessage
Response = AgentResponse
