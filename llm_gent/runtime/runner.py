"""Agent runners for swarm participation.

Three classes:
    BaseAgentRunner (ABC): Shared bus plumbing — connect, register, heartbeat,
        broadcast handling, channel polling, relay. Not exported.
    ManagedAgentRunner: Internal runner for hub-managed agents. Wraps an Agent
        instance, adds scheduling (Ticker) and agent.run() delegation.
    AgentRunner: External runner for standalone agents joining the swarm.
        Takes a Handler, supports blocking run() and background start()/stop().
        Use connect() classmethod to discover bus config from the hub's HTTP API.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from appinfra.service import BufferedChannel, ChannelTimeoutError
from appinfra.time import Ticker, TickerMode

from llm_gent.bus.protocol import (
    AgentJoined,
    AgentLeft,
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HeartbeatRequest,
    Message,
    RegisterRequest,
    RelayRequest,
    RelayResponse,
    Request,
    Response,
    ShutdownNotice,
    ShutdownRequest,
    ShutdownResponse,
    UnregisterRequest,
)

from .handler import Handler


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_gent.bus.transport import WorkerBusConfig, ZMQWorkerBus
    from llm_gent.core.agent import Agent


# =============================================================================
# Base (ABC, not exported)
# =============================================================================


class BaseAgentRunner(ABC):
    """ABC for swarm participants.

    Handles all bus plumbing: ZMQ connect/disconnect, registration,
    heartbeat responses, broadcast handling, and channel polling.
    Subclasses define the run loop and request dispatch behavior.
    """

    def __init__(
        self,
        lg: Logger,
        agent_id: str,
        handler: Handler,
        bus_config: WorkerBusConfig,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._lg = lg
        self._agent_id = agent_id
        self._handler = handler
        self._bus_config = bus_config
        self._capabilities = capabilities or []
        self._metadata = metadata or {}
        self._stop_event = threading.Event()
        self._bus: ZMQWorkerBus | None = None
        self._channel: BufferedChannel[Any, Any] | None = None

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def _running(self) -> bool:
        """Thread-safe running check."""
        return not self._stop_event.is_set()

    @_running.setter
    def _running(self, value: bool) -> None:
        """Thread-safe running setter."""
        if value:
            self._stop_event.clear()
        else:
            self._stop_event.set()

    def request_shutdown(self) -> None:
        """Request the runner to stop (thread-safe)."""
        self._stop_event.set()

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Main loop (blocking). Connect to bus, run loop, disconnect.

        Exceptions propagate to the caller so restart policies can trigger.
        """
        self._lg.debug("starting runner...", extra={"agent": self._agent_id})

        try:
            self._stop_event.clear()
            self._connect_bus()
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            self._disconnect_bus()
            self._on_stopped()

    @abstractmethod
    def _run_loop(self) -> None:
        """Main execution loop. Subclasses define their loop behavior."""

    def _on_stopped(self) -> None:
        """Hook called after the runner stops. Override for cleanup."""
        self._lg.info("runner stopped", extra={"agent": self._agent_id})

    # -------------------------------------------------------------------------
    # Bus lifecycle
    # -------------------------------------------------------------------------

    def _connect_bus(self) -> None:
        """Connect bus, create channel from DEALER transport, register with hub."""
        from llm_gent.bus.transport import ZMQWorkerBus

        self._bus = ZMQWorkerBus(self._lg, self._agent_id, self._bus_config)
        self._bus.start()
        # ZMQ async connect needs time to complete TCP handshake before
        # messages can be sent reliably. 200ms is typically sufficient.
        # TODO: Replace with ZMQ socket monitor events for deterministic connect.
        time.sleep(0.2)

        # Create BufferedChannel from the DEALER transport
        if self._bus.transport is None:
            raise RuntimeError("bus transport not available after start")
        self._channel = BufferedChannel(self._bus.transport)

        # Subscribe to broadcast topic for hub-initiated heartbeats
        self._bus.subscribe("broadcast", self._handle_broadcast)
        self._register_with_hub()

    def _register_with_hub(self) -> None:
        """Send registration request to the hub."""
        if self._channel is None:
            return
        req = RegisterRequest(
            agent_id=self._agent_id,
            capabilities=self._capabilities,
            metadata=self._metadata,
        )
        try:
            self._channel.submit(req, timeout=5.0)
            self._lg.info("registered on bus", extra={"agent": self._agent_id})
        except Exception as e:
            self._lg.warning(
                "bus registration failed",
                extra={"agent": self._agent_id, "exception": e},
            )

    def _disconnect_bus(self) -> None:
        """Unregister and disconnect."""
        import contextlib

        if self._channel is not None:
            with contextlib.suppress(Exception):
                self._channel.submit(UnregisterRequest(agent_id=self._agent_id), timeout=2.0)
            self._channel.close()
            self._channel = None

        if self._bus is not None:
            self._bus.stop()
            self._bus = None

    # -------------------------------------------------------------------------
    # Broadcast handling
    # -------------------------------------------------------------------------

    def _handle_broadcast(self, message: Message) -> None:
        """Handle broadcast messages from hub (system-tier).

        Responds to:
        - HeartbeatRequest: reply with stats on heartbeat topic.
        - ShutdownNotice: initiate graceful shutdown.
        - AgentJoined/AgentLeft: log membership changes.
        """
        if isinstance(message, HeartbeatRequest):
            self._respond_heartbeat(message)
        elif isinstance(message, ShutdownNotice):
            self._handle_shutdown_notice(message)
        elif isinstance(message, AgentJoined):
            self._lg.info("agent joined swarm", extra={"agent_id": message.agent_id})
        elif isinstance(message, AgentLeft):
            self._lg.info("agent left swarm", extra={"agent_id": message.agent_id})

    def _respond_heartbeat(self, request: HeartbeatRequest) -> None:
        """Respond to hub heartbeat broadcast with agent stats."""
        if self._bus is None:
            return
        try:
            self._bus.publish_heartbeat(
                stats=self._get_stats(),
                round_id=request.round_id,
                request_id=request.id,
            )
        except Exception as e:
            self._lg.debug(
                "heartbeat response failed",
                extra={"agent": self._agent_id, "exception": e},
            )

    def _get_stats(self) -> dict[str, Any]:
        """Get current agent stats for heartbeat. Override to customize."""
        return {"ticks": 0, "errors": 0}

    def _handle_shutdown_notice(self, notice: ShutdownNotice) -> None:
        """Handle hub shutdown broadcast — begin graceful shutdown."""
        self._lg.info(
            "hub shutdown notice received",
            extra={
                "agent": self._agent_id,
                "reason": notice.reason,
                "grace_secs": notice.grace_period_secs,
            },
        )
        self._stop_event.set()

    # -------------------------------------------------------------------------
    # Request polling and dispatch
    # -------------------------------------------------------------------------

    def _poll_requests(self) -> None:
        """Check for and handle one incoming request on the channel."""
        if self._channel is None:
            return

        try:
            msg = self._channel.recv(timeout=0.05)
            if isinstance(msg, Request):
                response = self._handle_request(msg)
                self._channel.send(response)
        except ChannelTimeoutError:
            pass  # no message waiting
        except Exception as e:
            self._lg.debug("poll error", extra={"exception": e})

    def _handle_request(self, request: Request) -> Response:
        """Dispatch incoming request to handler."""
        if isinstance(request, AskRequest):
            return self._handle_ask(request)
        if isinstance(request, FeedbackRequest):
            return self._handle_feedback(request)
        if isinstance(request, ShutdownRequest):
            return self._handle_shutdown(request)
        return Response(id=request.id, success=False, error="unknown request type")

    def _handle_ask(self, request: AskRequest) -> AskResponse:
        """Handle ask request via handler."""
        try:
            response_text = self._handler.on_ask(request.question)
            return AskResponse(id=request.id, response=response_text)
        except Exception as e:
            return AskResponse(id=request.id, success=False, error=str(e))

    def _handle_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Handle feedback request via handler."""
        try:
            self._handler.on_feedback(request.message)
            return FeedbackResponse(id=request.id)
        except Exception as e:
            return FeedbackResponse(id=request.id, success=False, error=str(e))

    def _handle_shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Handle shutdown request."""
        self._lg.info("shutdown requested", extra={"agent": self._agent_id})
        try:
            self._handler.on_shutdown()
        except Exception as e:
            self._lg.warning("handler on_shutdown failed", extra={"exception": e})
        self._stop_event.set()
        return ShutdownResponse(id=request.id)

    # -------------------------------------------------------------------------
    # Agent-to-agent relay
    # -------------------------------------------------------------------------

    def relay(self, to_agent: str, message: Message, timeout: float = 30.0) -> RelayResponse:
        """Send a request to another agent via the hub relay.

        Args:
            to_agent: Target agent name.
            message: Request to relay.
            timeout: Seconds to wait for response.

        Returns:
            RelayResponse containing the target agent's response.

        Raises:
            RuntimeError: If not connected to bus.
        """
        if self._channel is None:
            raise RuntimeError("not connected to bus")

        relay_req = RelayRequest(
            from_agent=self._agent_id,
            to_agent=to_agent,
            inner_type=message.message_type,
            inner_payload=message.model_dump(mode="json"),
        )
        resp = self._channel.submit(relay_req, timeout=timeout)
        if isinstance(resp, RelayResponse):
            return resp
        # Shouldn't happen -- hub always returns RelayResponse
        return RelayResponse(
            id=relay_req.id,
            from_agent=to_agent,
            inner_type="response",
        )


# =============================================================================
# ManagedAgentRunner (internal, hub-managed agents)
# =============================================================================


class _AgentHandler:
    """Adapts an Agent instance to the Handler protocol."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def on_ask(self, question: str) -> str:
        return self._agent.ask(question)

    def on_feedback(self, message: str) -> None:
        self._agent.record_feedback(message)

    def on_shutdown(self) -> None:
        pass  # shutdown is handled by the runner's stop event


class ManagedAgentRunner(BaseAgentRunner):
    """Runner for hub-managed (injected) agents.

    Wraps an Agent instance, adapting it to the Handler protocol.
    Adds scheduled execution via appinfra's Ticker and supports
    agent-controlled loops via agent.run().
    """

    def __init__(
        self,
        lg: Logger,
        agent: Agent,
        bus_config: WorkerBusConfig,
        schedule_interval: float | None = None,
    ) -> None:
        super().__init__(
            lg=lg,
            agent_id=agent.name,
            handler=_AgentHandler(agent),
            bus_config=bus_config,
        )
        self._agent = agent
        self._schedule_interval = schedule_interval

        if schedule_interval is not None and schedule_interval > 0:
            self._ticker: Ticker | None = Ticker(
                lg,
                secs=schedule_interval,
                mode=TickerMode.FLEX,
                initial=True,
            )
        else:
            self._ticker = None

    def _get_stats(self) -> dict[str, Any]:
        """Report agent's cycle count in heartbeat stats."""
        return {"ticks": self._agent.cycle_count, "errors": 0}

    def _on_stopped(self) -> None:
        """Stop the agent on runner shutdown."""
        try:
            self._agent.stop()
        except Exception as e:
            self._lg.warning("error stopping agent", extra={"exception": e})
        self._lg.info("runner stopped", extra={"agent": self._agent_id})

    # -------------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main execution loop: handle incoming requests + scheduled cycles."""
        if hasattr(self._agent, "run") and callable(self._agent.run):
            self._lg.debug("delegating to agent.run()", extra={"agent": self._agent_id})
            self._run_agent_loop()
            return

        self._run_framework_loop()

    def _run_framework_loop(self) -> None:
        """Framework-controlled loop with request polling + scheduling."""
        self._lg.trace(
            "entering framework run loop",
            extra={"agent": self._agent_id, "mode": self._get_execution_mode()},
        )
        while self._running:
            # Poll for incoming requests (non-blocking)
            self._poll_requests()

            if self._should_run_cycle():
                self._run_cycle()
            else:
                time.sleep(min(self._calculate_sleep(), 0.5))

    def _run_agent_loop(self) -> None:
        """Delegate to agent's run() method, poll requests in background.

        Note: ShutdownNotice sets _stop_event but cannot interrupt a blocking
        agent.run() call.  Agents using run() should check runner._stop_event
        periodically for cooperative cancellation.
        """
        # Start request poller in background thread
        poller = threading.Thread(
            target=self._request_poll_loop, daemon=True, name=f"req-{self._agent_id}"
        )
        poller.start()

        try:
            self._agent.run()  # type: ignore[attr-defined]
        finally:
            self._stop_event.set()

    def _request_poll_loop(self) -> None:
        """Background thread polling for requests (for agent-controlled loops)."""
        while self._running:
            self._poll_requests()
            time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------

    def _get_execution_mode(self) -> str:
        if self._schedule_interval == 0:
            return "continuous"
        if self._ticker:
            return "scheduled"
        return "message-only"

    def _calculate_sleep(self) -> float:
        if self._ticker is not None:
            return max(0.0, self._ticker.time_until_next_tick())
        if self._schedule_interval == 0:
            return 0.0
        return 0.5

    def _should_run_cycle(self) -> bool:
        if self._ticker is not None:
            return self._ticker.try_tick()
        return self._schedule_interval == 0

    def _run_cycle(self) -> None:
        """Run one scheduled execution cycle."""
        self._lg.debug("running scheduled cycle", extra={"agent": self._agent_id})
        try:
            self._agent.run_once()
        except Exception as e:
            self._lg.warning("cycle failed", extra={"agent": self._agent_id, "exception": e})


# =============================================================================
# AgentRunner (external, standalone agents)
# =============================================================================


class AgentRunner(BaseAgentRunner):
    """Runner for external agents joining the swarm.

    Takes a Handler implementation and manages bus connectivity.
    External agents use this as their gateway to the swarm without
    inheriting from Agent or knowing bus protocol details.

    Supports two execution modes:
    - ``run()``: Blocking event loop (simple agents).
    - ``start()``/``stop()``: Background thread (agents with own main loop).

    Use ``connect()`` to discover bus config from the hub's HTTP API::

        runner = AgentRunner.connect(
            lg=lg,
            handler=MyHandler(),
            agent_id="my-agent",
            hub_url="http://hub:8080",
        )
        runner.run()
    """

    def __init__(
        self,
        lg: Logger,
        handler: Handler,
        agent_id: str,
        bus_config: WorkerBusConfig,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            lg=lg,
            agent_id=agent_id,
            handler=handler,
            bus_config=bus_config,
            capabilities=capabilities,
            metadata=metadata,
        )
        self._bg_thread: threading.Thread | None = None

    @classmethod
    def connect(
        cls,
        lg: Logger,
        handler: Handler,
        agent_id: str,
        hub_url: str,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRunner:
        """Create a runner by discovering bus config from the hub's HTTP API.

        Args:
            lg: Logger instance.
            handler: Handler implementation for request dispatch.
            agent_id: Unique agent identifier.
            hub_url: Hub's HTTP base URL (e.g., "http://localhost:8080").
            capabilities: Agent capabilities to advertise.
            metadata: Additional metadata for registration.

        Returns:
            Configured AgentRunner ready to run().

        Raises:
            ConnectionError: If the hub is unreachable or returns an error.
        """
        import urllib.request

        url = f"{hub_url.rstrip('/')}/bus/config"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                import json

                data = json.loads(resp.read())
        except Exception as e:
            raise ConnectionError(f"failed to fetch bus config from {url}: {e}") from e

        import dataclasses

        from llm_gent.bus.transport import WorkerBusConfig

        fields = {f.name for f in dataclasses.fields(WorkerBusConfig)}
        bus_config = WorkerBusConfig(**{k: v for k, v in data.items() if k in fields})
        return cls(
            lg=lg,
            handler=handler,
            agent_id=agent_id,
            bus_config=bus_config,
            capabilities=capabilities,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Background execution
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the runner in a background thread.

        Use this when the external agent has its own main loop and
        wants bus connectivity on the side. Call stop() to shut down.

        Raises:
            RuntimeError: If already running.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            raise RuntimeError("runner is already started")

        self._bg_thread = threading.Thread(
            target=self.run, daemon=True, name=f"agent-runner-{self._agent_id}"
        )
        self._bg_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background runner.

        Args:
            timeout: Seconds to wait for the background thread to finish.
        """
        self._stop_event.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=timeout)
            self._bg_thread = None

    # -------------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Simple poll-and-dispatch loop for external agents."""
        self._lg.trace(
            "entering external agent run loop",
            extra={"agent": self._agent_id},
        )
        while self._running:
            self._poll_requests()
            if self._running:
                time.sleep(0.05)
