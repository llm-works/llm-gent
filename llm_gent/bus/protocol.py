"""Bus protocol: envelope, base message types, and v1 messages.

All bus communication uses envelope-wrapped messages for versioning and routing.
Messages are Pydantic models serialized to JSON for transport.

Serialization flow:
    Message -> to_envelope() -> Envelope -> model_dump_json() -> bytes
    bytes -> Envelope.model_validate_json() -> unwrap(REGISTRY) -> Message
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Base message types
# =============================================================================


class Message(BaseModel):
    """Base message type for bus communication."""

    message_type: ClassVar[str] = "base"

    def to_envelope(self, version: int = 1) -> Envelope:
        """Wrap this message in an envelope for transport.

        Args:
            version: Protocol version.

        Returns:
            Envelope ready for serialization.
        """
        return Envelope(
            version=version,
            msg_type=self.message_type,
            payload=self.model_dump(mode="json"),
        )


class Request(Message):
    """Request message with unique ID for correlation."""

    message_type: ClassVar[str] = "request"

    id: str = Field(default_factory=lambda: uuid4().hex[:12])


class Response(Message):
    """Response message correlated to a request."""

    message_type: ClassVar[str] = "response"

    id: str
    success: bool = True
    error: str | None = None


# =============================================================================
# Envelope
# =============================================================================


class Envelope(BaseModel):
    """Transport envelope wrapping any message.

    Provides protocol versioning, message type routing, and optional
    source/target for agent-to-agent communication.
    """

    version: int = Field(default=1, ge=1)
    msg_type: str
    source: str | None = None
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    def unwrap(self, registry: dict[str, type[Message]]) -> Message:
        """Deserialize payload into a typed message.

        Args:
            registry: Mapping of message_type -> Message class.

        Returns:
            Deserialized message instance.

        Raises:
            ValueError: If message type is unknown.
        """
        msg_class = registry.get(self.msg_type)
        if msg_class is None:
            raise ValueError(f"unknown message type: {self.msg_type}")
        return msg_class.model_validate(self.payload)

    def to_bytes(self) -> bytes:
        """Serialize envelope to bytes for transport."""
        return self.model_dump_json().encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> Envelope:
        """Deserialize envelope from bytes."""
        return cls.model_validate_json(data)


# =============================================================================
# V1 protocol messages
# =============================================================================


class AgentStats(BaseModel):
    """Runtime statistics reported by agents in heartbeats."""

    ticks: int = 0
    errors: int = 0
    llm_tokens_used: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


# -- Registration --


class RegisterRequest(Request):
    """Agent requests to join the swarm."""

    message_type: ClassVar[str] = "register_request"

    agent_id: str
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    health_url: str | None = None


class RegisterResponse(Response):
    """Hub acknowledges registration."""

    message_type: ClassVar[str] = "register_response"

    agent_id: str
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# -- Heartbeat --


class HeartbeatRequest(Request):
    """Agent liveness signal with optional stats."""

    message_type: ClassVar[str] = "heartbeat_request"

    agent_id: str
    stats: AgentStats = Field(default_factory=AgentStats)


class HeartbeatResponse(Response):
    """Hub acknowledges heartbeat."""

    message_type: ClassVar[str] = "heartbeat_response"

    agent_id: str
    ack_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


# -- Unregister --


class UnregisterRequest(Request):
    """Agent requests to leave the swarm."""

    message_type: ClassVar[str] = "unregister_request"

    agent_id: str


class UnregisterResponse(Response):
    """Hub acknowledges unregistration."""

    message_type: ClassVar[str] = "unregister_response"

    agent_id: str


# -- Error escalation --


class ErrorReport(BaseModel):
    """Structured error report from an agent."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    severity: str = "error"
    source: str = ""
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorRequest(Request):
    """Agent escalates an error to the hub."""

    message_type: ClassVar[str] = "error_request"

    agent_id: str
    error: ErrorReport
    escalation_reason: str = "severity"


class ErrorResponse(Response):
    """Hub acknowledges error receipt."""

    message_type: ClassVar[str] = "error_response"

    acknowledged: bool = True


# -- Agent operations (hub -> agent) --


class AskRequest(Request):
    """Hub asks an agent a question."""

    message_type: ClassVar[str] = "ask_request"

    question: str


class AskResponse(Response):
    """Agent responds to a question."""

    message_type: ClassVar[str] = "ask_response"

    response: str = ""


class FeedbackRequest(Request):
    """Hub sends feedback to an agent."""

    message_type: ClassVar[str] = "feedback_request"

    message: str


class FeedbackResponse(Response):
    """Agent acknowledges feedback."""

    message_type: ClassVar[str] = "feedback_response"


class ShutdownRequest(Request):
    """Hub tells an agent to shut down."""

    message_type: ClassVar[str] = "shutdown_request"


class ShutdownResponse(Response):
    """Agent acknowledges shutdown."""

    message_type: ClassVar[str] = "shutdown_response"


# =============================================================================
# Message registry
# =============================================================================


MESSAGE_REGISTRY: dict[str, type[Message]] = {
    "register_request": RegisterRequest,
    "register_response": RegisterResponse,
    "heartbeat_request": HeartbeatRequest,
    "heartbeat_response": HeartbeatResponse,
    "unregister_request": UnregisterRequest,
    "unregister_response": UnregisterResponse,
    "error_request": ErrorRequest,
    "error_response": ErrorResponse,
    "ask_request": AskRequest,
    "ask_response": AskResponse,
    "feedback_request": FeedbackRequest,
    "feedback_response": FeedbackResponse,
    "shutdown_request": ShutdownRequest,
    "shutdown_response": ShutdownResponse,
}
