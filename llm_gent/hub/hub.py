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
from appinfra.service import BufferedChannel, ProcessRunner, RestartPolicy, ThreadRunner

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
    RelayRequest,
    RelayResponse,
    Request,
    Response,
    UnregisterRequest,
    UnregisterResponse,
)
from ..bus.transport import CoordinatorBusConfig, ZMQCoordinatorBus
from ..runtime.service import AgentService
from .registry import AgentEntry, AgentType, Registry


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
        self._runners: dict[str, ThreadRunner | ProcessRunner] = {}
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
        self._bus.set_route_validator(lambda agent_id: self._registry.get(agent_id) is not None)
        self._bus.register_async_request("relay_request")
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

        try:
            self._create_channel(name)
            runner = self._create_runner(name, config, llm_config, learn_config)
            runner.start()
        except Exception:
            self._cleanup_agent_resources(name)
            self._registry.unregister(name)
            raise
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
    ) -> ThreadRunner | ProcessRunner:
        """Create AgentService wrapped in a runner with restart policy.

        Respects the ``execution`` config field: ``"thread"`` uses ThreadRunner,
        ``"process"`` (default) uses ProcessRunner for subprocess isolation.
        """
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
        execution = config.get("execution", "process")
        if execution == "thread":
            return ThreadRunner(service, policy=policy)
        return ProcessRunner(service, policy=policy)

    def stop_agent(self, name: str) -> None:
        """Stop an injected agent.

        Args:
            name: Agent name.
        """
        runner = self._runners.pop(name, None)
        if runner is not None:
            runner.stop()

        self._cleanup_agent_resources(name)
        self._registry.unregister(name)
        self._lg.info("agent stopped", extra={"agent": name})

    def _cleanup_agent_resources(self, name: str) -> None:
        """Clean up channel and transport for an agent."""
        channel = self._channels.pop(name, None)
        if channel is not None:
            channel.close()
        self._bus.remove_agent_transport(name)

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

        Raises:
            RuntimeError: If the agent returned a failure response.
        """
        channel = self._require_channel(name)
        req = AskRequest(question=question)
        resp = channel.submit(req, timeout=timeout)
        if isinstance(resp, AskResponse):
            if not resp.success:
                raise RuntimeError(f"Agent {name} ask failed: {resp.error}")
            return resp.response
        return ""

    def feedback(self, name: str, message: str, timeout: float = 30.0) -> None:
        """Send feedback to an agent.

        Args:
            name: Agent name.
            message: Feedback text.
            timeout: Response timeout.

        Raises:
            RuntimeError: If the agent returned a failure response.
        """
        channel = self._require_channel(name)
        resp = channel.submit(FeedbackRequest(message=message), timeout=timeout)
        if hasattr(resp, "success") and not resp.success:
            raise RuntimeError(f"Agent {name} feedback failed: {getattr(resp, 'error', 'unknown')}")

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
        if isinstance(request, RelayRequest):
            return self._handle_relay(request)
        return Response(id=request.id, success=False, error="unknown request type")

    def _handle_register(self, req: RegisterRequest, sender_id: str | None) -> RegisterResponse:
        """Handle agent registration.

        Uses the registry's atomic register_or_merge to avoid TOCTOU races.
        Injected agents get capabilities/metadata merged; external agents
        are registered normally.
        """
        entry = self._registry.register_or_merge(
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

    def _handle_relay(self, req: RelayRequest) -> RelayResponse:
        """Relay a request from one agent to another.

        Deserializes the inner request, forwards it to the target agent
        via the target's channel, and wraps the response back.
        """
        from ..bus.protocol import MESSAGE_REGISTRY

        target_channel = self._channels.get(req.to_agent)
        if target_channel is None:
            return self._relay_error(req, f"Target agent not found: {req.to_agent}")

        inner_class = MESSAGE_REGISTRY.get(req.inner_type)
        if inner_class is None:
            return self._relay_error(req, f"Unknown inner message type: {req.inner_type}")

        try:
            inner_msg = inner_class.model_validate(req.inner_payload)
            inner_resp = target_channel.submit(inner_msg, timeout=30.0)
            return RelayResponse(
                id=req.id,
                from_agent=req.to_agent,
                inner_type=getattr(inner_resp, "message_type", "response"),
                inner_payload=inner_resp.model_dump(mode="json")
                if hasattr(inner_resp, "model_dump")
                else {},
            )
        except Exception as e:
            return self._relay_error(req, str(e))

    def _relay_error(self, req: RelayRequest, error: str) -> RelayResponse:
        """Create an error RelayResponse."""
        return RelayResponse(
            id=req.id, success=False, error=error, from_agent=req.to_agent, inner_type="error"
        )

    def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat on pub/sub topic."""
        if not isinstance(message, HeartbeatRequest):
            return
        self._registry.heartbeat(message.agent_id, message.stats)

    # =========================================================================
    # Internal
    # =========================================================================

    def get_insights(self, name: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get available insights for an agent from registry data.

        Returns stats and health information since full agent-side
        insights (recent results) are not yet available via the bus.
        """
        entry = self._registry.get(name)
        if entry is None:
            return []
        return [
            {
                "type": "stats",
                "ticks": entry.stats.ticks,
                "errors": entry.stats.errors,
                "health": entry.health.value,
                "last_heartbeat": entry.last_heartbeat.isoformat(),
                "last_run": entry.last_run.isoformat() if entry.last_run else None,
                "restart_count": entry.restart_count,
            }
        ]

    def _require_channel(self, name: str) -> BufferedChannel[Any, Any]:
        """Get channel for an agent, raising if not found."""
        channel = self._channels.get(name)
        if channel is None:
            raise KeyError(f"No channel for agent: {name}")
        return channel
