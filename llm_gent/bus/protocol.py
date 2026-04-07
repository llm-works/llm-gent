"""Bus protocol: envelope, base message types, and v1 messages.

All bus communication uses envelope-wrapped messages for versioning and routing.
Messages are Pydantic models serialized to JSON for transport.

Serialization flow:
    Message -> to_envelope() -> Envelope -> model_dump_json() -> bytes
    bytes -> Envelope.model_validate_json() -> unwrap(REGISTRY) -> Message
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Message tiers (FIX-style session/application separation)
# =============================================================================


class MessageTier(StrEnum):
    """Classification tier for bus messages.

    Inspired by FIX protocol session/application separation:
    - SYSTEM: infrastructure (heartbeat, registration, shutdown)
    - APPLICATION: business logic (ask, feedback, errors)
    - CUSTOM: agent-defined protocols (relay, collaboration)
    """

    SYSTEM = "system"
    APPLICATION = "application"
    CUSTOM = "custom"


# =============================================================================
# Base message types
# =============================================================================


class Message(BaseModel):
    """Base message type for bus communication."""

    message_type: ClassVar[str] = "base"
    tier: ClassVar[MessageTier] = MessageTier.SYSTEM

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
            tier=self.tier.value,
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
    tier: str = "system"
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
    """Heartbeat challenge or agent-initiated liveness signal.

    Two usage modes:
    - Hub broadcast: hub publishes with round_id set, agent_id empty.
      Agents respond with HeartbeatResponse.
    - Agent p2p: agent sends via DEALER with agent_id and stats set.
      Hub responds with HeartbeatResponse.
    """

    message_type: ClassVar[str] = "heartbeat_request"

    agent_id: str = ""
    round_id: str = ""
    stats: AgentStats = Field(default_factory=AgentStats)


class HeartbeatResponse(Response):
    """Heartbeat response from agent (broadcast) or hub (p2p).

    When responding to a broadcast, agent sets agent_id, round_id, and stats.
    When hub acknowledges a p2p heartbeat, hub sets agent_id and ack_time.
    """

    message_type: ClassVar[str] = "heartbeat_response"

    agent_id: str = ""
    round_id: str = ""
    stats: AgentStats = Field(default_factory=AgentStats)
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
    tier: ClassVar[MessageTier] = MessageTier.APPLICATION

    agent_id: str
    error: ErrorReport
    escalation_reason: str = "severity"


class ErrorResponse(Response):
    """Hub acknowledges error receipt."""

    message_type: ClassVar[str] = "error_response"
    tier: ClassVar[MessageTier] = MessageTier.APPLICATION

    acknowledged: bool = True


# -- Agent operations (hub -> agent) --


class AskRequest(Request):
    """Hub asks an agent a question."""

    message_type: ClassVar[str] = "ask_request"
    tier: ClassVar[MessageTier] = MessageTier.APPLICATION

    question: str


class AskResponse(Response):
    """Agent responds to a question."""

    message_type: ClassVar[str] = "ask_response"
    tier: ClassVar[MessageTier] = MessageTier.APPLICATION

    response: str = ""


class FeedbackRequest(Request):
    """Hub sends feedback to an agent."""

    message_type: ClassVar[str] = "feedback_request"
    tier: ClassVar[MessageTier] = MessageTier.APPLICATION

    message: str


class FeedbackResponse(Response):
    """Agent acknowledges feedback."""

    message_type: ClassVar[str] = "feedback_response"
    tier: ClassVar[MessageTier] = MessageTier.APPLICATION


class ShutdownRequest(Request):
    """Hub tells a specific agent to shut down (p2p via DEALER)."""

    message_type: ClassVar[str] = "shutdown_request"
    tier: ClassVar[MessageTier] = MessageTier.SYSTEM


class ShutdownResponse(Response):
    """Agent acknowledges shutdown."""

    message_type: ClassVar[str] = "shutdown_response"
    tier: ClassVar[MessageTier] = MessageTier.SYSTEM


class ShutdownNotice(Message):
    """Hub broadcasts impending shutdown to all agents.

    System-tier broadcast — not a request, no response expected.
    Agents should clean up within the grace period before the hub
    tears down the bus.
    """

    message_type: ClassVar[str] = "shutdown_notice"

    reason: str = ""
    grace_period_secs: float = 5.0


# -- Swarm membership notices --


class AgentJoined(Message):
    """Hub broadcasts when an agent joins the swarm.

    System-tier broadcast — no response expected.
    """

    message_type: ClassVar[str] = "agent_joined"

    agent_id: str
    agent_type: str = "external"
    capabilities: list[str] = Field(default_factory=list)


class AgentLeft(Message):
    """Hub broadcasts when an agent leaves the swarm.

    System-tier broadcast — no response expected.
    """

    message_type: ClassVar[str] = "agent_left"

    agent_id: str
    reason: str = "voluntary"  # voluntary | dead | shutdown


# -- Relay (agent-to-agent via hub) --


class RelayRequest(Request):
    """Agent asks hub to forward a request to another agent.

    The hub forwards the inner payload to the target agent,
    waits for the response, and returns it to the sender.
    """

    message_type: ClassVar[str] = "relay_request"
    tier: ClassVar[MessageTier] = MessageTier.CUSTOM

    from_agent: str
    to_agent: str
    inner_type: str
    inner_payload: dict[str, Any] = Field(default_factory=dict)


class RelayResponse(Response):
    """Hub returns the target agent's response to the sender."""

    message_type: ClassVar[str] = "relay_response"
    tier: ClassVar[MessageTier] = MessageTier.CUSTOM

    from_agent: str
    inner_type: str
    inner_payload: dict[str, Any] = Field(default_factory=dict)


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
    "shutdown_notice": ShutdownNotice,
    "agent_joined": AgentJoined,
    "agent_left": AgentLeft,
    "relay_request": RelayRequest,
    "relay_response": RelayResponse,
}
