"""Swarm hub coordinator.

The Hub is the single coordinator for a swarm. It owns:
- ZMQ bus (coordinator sockets)
- Unified agent registry (membership, health, config, state)
- appinfra Manager (injected agent lifecycle with restart policies)
- BufferedChannel per agent (controller-to-agent request/response)

All agent operations go through the Hub: start, stop, ask, feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from appinfra.service import BufferedChannel, RestartPolicy, ThreadRunner

from ..bus.protocol import (
    AskRequest,
    AskResponse,
    ErrorRequest,
    ErrorResponse,
    FeedbackRequest,
    HeartbeatRequest,
    Message,
    RegisterRequest,
    RegisterResponse,
    Request,
    Response,
    UnregisterRequest,
    UnregisterResponse,
)
from ..bus.transport import CoordinatorBusConfig, ZMQCoordinatorBus
from ..runtime.service import AgentService
from .registry import AgentEntry, AgentStats, AgentType, Registry


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ..bus.transport import WorkerBusConfig


@dataclass
class HubConfig:
    """Configuration for the swarm hub."""

    bus: CoordinatorBusConfig = field(default_factory=CoordinatorBusConfig)
    dead_timeout: float = 90.0
    health_check_interval: float = 30.0
    max_restarts: int = 3


class Hub:
    """Swarm hub coordinator.

    Single entry point for all agent operations: registration, lifecycle,
    communication, and monitoring.

    Usage::

        hub = Hub(lg, config, bus_config)
        hub.start()
        hub.start_agent("my-agent", agent_config, llm_config)
        response = hub.ask("my-agent", "hello")
        hub.stop()
    """

    def __init__(
        self,
        lg: Logger,
        config: HubConfig,
        bus_config: WorkerBusConfig,
        llm_config: Any = None,
        learn_config: Any = None,
        variables: dict[str, str] | None = None,
        factory_module: str = "llm_gent.agents.default",
    ) -> None:
        self._lg = lg
        self._config = config
        self._bus_config = bus_config
        self._llm_config = llm_config
        self._learn_config = learn_config
        self._variables = variables or {}
        self._factory_module = factory_module

        self._bus = ZMQCoordinatorBus(lg, config.bus)
        self._registry = Registry(dead_timeout=config.dead_timeout)
        self._runners: dict[str, ThreadRunner] = {}
        self._channels: dict[str, BufferedChannel[Any, Any]] = {}

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def bus(self) -> ZMQCoordinatorBus:
        return self._bus

    # =========================================================================
    # Hub lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the hub: bind bus, begin accepting connections."""
        self._bus.start()
        self._bus.on_request(self._handle_bus_request)
        self._bus.subscribe("heartbeat", self._handle_heartbeat)
        self._lg.info("hub started")

    def stop(self) -> None:
        """Stop all agents and shut down the bus."""
        for name in list(self._runners.keys()):
            try:
                self.stop_agent(name)
            except Exception as e:
                self._lg.warning("error stopping agent", extra={"agent": name, "exception": e})

        self._bus.stop()
        self._lg.info("hub stopped")

    # =========================================================================
    # Agent lifecycle (injected agents)
    # =========================================================================

    def start_agent(
        self,
        name: str,
        config: DotDict,
        llm_config: Any | None = None,
        learn_config: Any | None = None,
    ) -> AgentEntry:
        """Start an injected agent as a threaded service.

        Registers the agent, creates a BufferedChannel over ZMQ transport,
        and starts an AgentService in a ThreadRunner with restart policy.
        """
        config["name"] = name
        entry = self._registry.register(agent_id=name, agent_type=AgentType.INJECTED, config=config)

        self._create_channel(name)
        runner = self._create_runner(name, config, llm_config, learn_config)
        runner.start()
        self._runners[name] = runner

        self._lg.info("agent started", extra={"agent": name})
        return entry

    def _create_channel(self, name: str) -> None:
        """Create ZMQ transport + BufferedChannel for an agent."""
        transport = self._bus.create_agent_transport(name)
        channel: BufferedChannel[Any, Any] = BufferedChannel(transport)
        self._channels[name] = channel

    def _create_runner(
        self, name: str, config: DotDict, llm_config: Any | None, learn_config: Any | None
    ) -> ThreadRunner:
        """Create AgentService wrapped in a ThreadRunner with restart policy."""
        service = AgentService(
            lg=self._lg,
            agent_name=name,
            config=config,
            llm_config=llm_config or self._llm_config,
            bus_config=self._bus_config,
            learn_config=learn_config or self._learn_config,
            variables=self._variables,
            factory_module=config.get("module", self._factory_module),
        )
        policy = RestartPolicy(
            max_retries=self._config.max_restarts,
            restart_on_failure=True,
        )
        return ThreadRunner(service, policy=policy)

    def stop_agent(self, name: str) -> None:
        """Stop an injected agent.

        Args:
            name: Agent name.
        """
        runner = self._runners.pop(name, None)
        if runner is not None:
            runner.stop()

        channel = self._channels.pop(name, None)
        if channel is not None:
            channel.close()

        self._bus.remove_agent_transport(name)
        self._registry.unregister(name)
        self._lg.info("agent stopped", extra={"agent": name})

    # =========================================================================
    # Agent communication
    # =========================================================================

    def ask(self, name: str, question: str, timeout: float = 60.0) -> str:
        """Ask an agent a question.

        Args:
            name: Agent name.
            question: Question text.
            timeout: Response timeout.

        Returns:
            Agent's response string.
        """
        channel = self._require_channel(name)
        req = AskRequest(question=question)
        resp = channel.submit(req, timeout=timeout)
        if isinstance(resp, AskResponse):
            return resp.response
        return ""

    def feedback(self, name: str, message: str, timeout: float = 30.0) -> None:
        """Send feedback to an agent.

        Args:
            name: Agent name.
            message: Feedback text.
            timeout: Response timeout.
        """
        channel = self._require_channel(name)
        channel.submit(FeedbackRequest(message=message), timeout=timeout)

    def broadcast(self, message: Message) -> None:
        """Broadcast a message to all agents."""
        self._bus.broadcast(message)

    def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a topic."""
        self._bus.publish(topic, message)

    # =========================================================================
    # Bus request handling (agent → hub)
    # =========================================================================

    def _handle_bus_request(self, request: Request, sender_id: str | None) -> Response:
        """Dispatch incoming bus requests from agents."""
        if isinstance(request, RegisterRequest):
            return self._handle_register(request, sender_id)
        if isinstance(request, UnregisterRequest):
            return self._handle_unregister(request)
        if isinstance(request, ErrorRequest):
            return self._handle_error(request)
        return Response(id=request.id, success=False, error="unknown request type")

    def _handle_register(self, req: RegisterRequest, sender_id: str | None) -> RegisterResponse:
        """Handle external agent registration."""
        entry = self._registry.register(
            agent_id=req.agent_id,
            agent_type=AgentType.EXTERNAL,
            capabilities=req.capabilities,
            metadata=req.metadata,
        )
        self._lg.info(
            "agent registered",
            extra={"agent_id": req.agent_id, "capabilities": req.capabilities},
        )
        return RegisterResponse(id=req.id, agent_id=req.agent_id, registered_at=entry.registered_at)

    def _handle_unregister(self, req: UnregisterRequest) -> UnregisterResponse:
        """Handle agent unregistration."""
        self._registry.unregister(req.agent_id)
        self._lg.info("agent unregistered", extra={"agent_id": req.agent_id})
        return UnregisterResponse(id=req.id, agent_id=req.agent_id)

    def _handle_error(self, req: ErrorRequest) -> ErrorResponse:
        """Handle error escalation from agent."""
        self._lg.warning(
            "agent error",
            extra={
                "agent_id": req.agent_id,
                "severity": req.error.severity,
                "message": req.error.message,
            },
        )
        return ErrorResponse(id=req.id, acknowledged=True)

    def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat on pub/sub topic."""
        if not isinstance(message, HeartbeatRequest):
            return
        stats = AgentStats(
            ticks=message.stats.ticks,
            errors=message.stats.errors,
            llm_tokens_used=message.stats.llm_tokens_used,
            extra=message.stats.extra,
        )
        self._registry.heartbeat(message.agent_id, stats)

    # =========================================================================
    # Internal
    # =========================================================================

    def _require_channel(self, name: str) -> BufferedChannel[Any, Any]:
        """Get channel for an agent, raising if not found."""
        channel = self._channels.get(name)
        if channel is None:
            raise KeyError(f"No channel for agent: {name}")
        return channel
