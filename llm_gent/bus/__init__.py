"""Agent communication bus for swarm orchestration.

Provides ZMQ-based message bus with envelope protocol for agent-to-agent
and agent-to-hub communication within a swarm.
"""

from .protocol import (
    AgentStats,
    Envelope,
    ErrorReport,
    ErrorRequest,
    ErrorResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    Message,
    RegisterRequest,
    RegisterResponse,
    Request,
    Response,
    ShutdownNotice,
    UnregisterRequest,
    UnregisterResponse,
)
from .transport import (
    BusError,
    BusTimeoutError,
    CoordinatorBusConfig,
    WorkerBusConfig,
    ZMQCoordinatorBus,
    ZMQWorkerBus,
)


__all__ = [
    # Protocol
    "AgentStats",
    "Envelope",
    "ErrorReport",
    "ErrorRequest",
    "ErrorResponse",
    "HeartbeatRequest",
    "HeartbeatResponse",
    "Message",
    "RegisterRequest",
    "RegisterResponse",
    "Request",
    "Response",
    "ShutdownNotice",
    "UnregisterRequest",
    "UnregisterResponse",
    # Transport
    "BusError",
    "BusTimeoutError",
    "CoordinatorBusConfig",
    "WorkerBusConfig",
    "ZMQCoordinatorBus",
    "ZMQWorkerBus",
]
