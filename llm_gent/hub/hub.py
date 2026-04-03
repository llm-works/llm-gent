"""Swarm hub coordinator.

The Hub is the central orchestrator for a swarm. It:
- Owns the ZMQ coordinator bus (binds sockets)
- Maintains the agent registry (membership, health, stats)
- Handles bus protocol messages (register, heartbeat, unregister, error)
- Runs periodic health monitoring
- Delegates injected agent lifecycle to the runtime Core
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..bus.protocol import (
    ErrorRequest,
    ErrorResponse,
    HeartbeatRequest,
    Message,
    RegisterRequest,
    RegisterResponse,
    Request,
    Response,
    UnregisterRequest,
    UnregisterResponse,
)
from ..bus.registry import AgentRegistry, AgentType
from ..bus.registry import AgentStats as RegistryAgentStats
from ..bus.transport import CoordinatorBusConfig, ZMQCoordinatorBus


if TYPE_CHECKING:
    from appinfra.log import Logger


@dataclass
class HubConfig:
    """Configuration for the swarm hub."""

    bus: CoordinatorBusConfig = field(default_factory=CoordinatorBusConfig)
    dead_timeout: float = 90.0
    health_check_interval: float = 30.0
    max_restarts: int = 3


class Hub:
    """Swarm hub coordinator.

    Manages agent registration, health monitoring, and bus communication.
    The hub is the single coordinator for a swarm -- one hub per
    ``llm-gent serve`` instance.

    Usage::

        hub = Hub(lg, config)
        hub.start()
        # ... hub runs, agents connect via bus ...
        hub.stop()
    """

    def __init__(self, lg: Logger, config: HubConfig | None = None) -> None:
        self._lg = lg
        self._config = config or HubConfig()
        self._bus = ZMQCoordinatorBus(lg, self._config.bus)
        self._registry = AgentRegistry(dead_timeout=self._config.dead_timeout)
        self._health_thread: threading.Thread | None = None
        self._running = False

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry."""
        return self._registry

    @property
    def bus(self) -> ZMQCoordinatorBus:
        """Access the coordinator bus."""
        return self._bus

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the hub: bind bus sockets, begin health monitoring."""
        self._bus.start()
        self._bus.on_request(self._handle_request)
        self._bus.subscribe("heartbeat", self._handle_heartbeat_topic)

        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="hub-health"
        )
        self._health_thread.start()

        self._lg.info("hub started")

    def stop(self) -> None:
        """Stop the hub: shut down health monitoring and bus."""
        self._running = False
        if self._health_thread is not None:
            self._health_thread.join(timeout=5.0)
            self._health_thread = None

        self._bus.stop()
        self._lg.info("hub stopped")

    # =========================================================================
    # Bus message handlers
    # =========================================================================

    def _handle_request(self, request: Request, sender_id: str | None) -> Response:
        """Dispatch incoming bus requests to the appropriate handler."""
        if isinstance(request, RegisterRequest):
            return self._handle_register(request, sender_id)
        if isinstance(request, UnregisterRequest):
            return self._handle_unregister(request)
        if isinstance(request, ErrorRequest):
            return self._handle_error(request)
        return Response(id=request.id, success=False, error="unknown request type")

    def _handle_register(self, request: RegisterRequest, sender_id: str | None) -> RegisterResponse:
        """Handle agent registration request."""
        info = self._registry.register(
            agent_id=request.agent_id,
            agent_type=AgentType.EXTERNAL,
            capabilities=request.capabilities,
            metadata=request.metadata,
            health_url=request.health_url,
        )
        self._lg.info(
            "agent registered",
            extra={
                "agent_id": request.agent_id,
                "capabilities": request.capabilities,
                "zmq_identity": sender_id,
            },
        )
        return RegisterResponse(
            id=request.id,
            agent_id=request.agent_id,
            registered_at=info.registered_at,
        )

    def _handle_unregister(self, request: UnregisterRequest) -> UnregisterResponse:
        """Handle agent unregistration request."""
        removed = self._registry.unregister(request.agent_id)
        if removed:
            self._lg.info("agent unregistered", extra={"agent_id": request.agent_id})
        else:
            self._lg.warning("unregister for unknown agent", extra={"agent_id": request.agent_id})
        return UnregisterResponse(id=request.id, agent_id=request.agent_id)

    def _handle_error(self, request: ErrorRequest) -> ErrorResponse:
        """Handle error escalation from an agent."""
        self._lg.warning(
            "agent error escalation",
            extra={
                "agent_id": request.agent_id,
                "severity": request.error.severity,
                "source": request.error.source,
                "message": request.error.message,
                "reason": request.escalation_reason,
            },
        )
        return ErrorResponse(id=request.id, acknowledged=True)

    def _handle_heartbeat_topic(self, message: Message) -> None:
        """Handle heartbeat published on the heartbeat topic."""
        if not isinstance(message, HeartbeatRequest):
            return

        stats = RegistryAgentStats(
            ticks=message.stats.ticks,
            errors=message.stats.errors,
            llm_tokens_used=message.stats.llm_tokens_used,
            extra=message.stats.extra,
        )
        known = self._registry.heartbeat(message.agent_id, stats)
        if not known:
            self._lg.debug("heartbeat from unknown agent", extra={"agent_id": message.agent_id})

    # =========================================================================
    # Injected agent management
    # =========================================================================

    def register_injected(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an injected (hub-managed) agent in the registry.

        Called by the hub when spawning an agent from config, before the
        agent connects to the bus.

        Args:
            agent_id: Agent identifier.
            capabilities: Agent capabilities.
            metadata: Agent metadata.
        """
        self._registry.register(
            agent_id=agent_id,
            agent_type=AgentType.INJECTED,
            capabilities=capabilities or [],
            metadata=metadata or {},
        )
        self._lg.info("injected agent registered", extra={"agent_id": agent_id})

    # =========================================================================
    # Health monitoring
    # =========================================================================

    def _health_loop(self) -> None:
        """Periodic health check loop."""
        import time

        while self._running:
            time.sleep(self._config.health_check_interval)
            if not self._running:
                break
            self._check_health()

    def _check_health(self) -> None:
        """Check agent health and handle dead agents."""
        dead = self._registry.get_dead()
        for agent_info in dead:
            self._lg.warning(
                "agent is dead",
                extra={
                    "agent_id": agent_info.id,
                    "agent_type": agent_info.agent_type,
                    "last_heartbeat": agent_info.last_heartbeat.isoformat(),
                },
            )
            if agent_info.agent_type == AgentType.INJECTED:
                self._handle_dead_injected(agent_info.id)
            else:
                self._registry.unregister(agent_info.id)

        unhealthy = self._registry.get_unhealthy()
        for agent_info in unhealthy:
            self._lg.debug(
                "agent unhealthy",
                extra={
                    "agent_id": agent_info.id,
                    "last_heartbeat": agent_info.last_heartbeat.isoformat(),
                },
            )

    def _handle_dead_injected(self, agent_id: str) -> None:
        """Handle a dead injected agent (restart logic)."""
        restart_count = self._registry.increment_restart(agent_id)
        if restart_count <= self._config.max_restarts:
            self._lg.info(
                "restarting dead injected agent",
                extra={
                    "agent_id": agent_id,
                    "restart_count": restart_count,
                    "max_restarts": self._config.max_restarts,
                },
            )
            # TODO: trigger restart via Core
            # For now, just log the intent. Actual restart integration
            # comes when Hub is wired into ServeTool.
        else:
            self._lg.warning(
                "injected agent exceeded max restarts, removing",
                extra={
                    "agent_id": agent_id,
                    "restart_count": restart_count,
                },
            )
            self._registry.unregister(agent_id)
