"""Agent runner - runs a single agent in a subprocess or thread.

The AgentRunner is the entry point for agent subprocesses/threads. It:
- Connects to the hub via ZMQ bus
- Responds to hub-initiated heartbeat broadcasts
- Uses appinfra BufferedChannel for request/response (ask, feedback, shutdown)
- Handles scheduled execution using appinfra.time.Ticker
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from appinfra.service import BufferedChannel, ChannelTimeoutError
from appinfra.time import Ticker, TickerMode

from llm_gent.bus.protocol import (
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


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_gent.bus.transport import WorkerBusConfig, ZMQWorkerBus
    from llm_gent.core.agent import Agent


class AgentRunner:
    """Runs a single agent in a subprocess or thread.

    Connects to the hub via ZMQ bus for pub/sub (heartbeats, events)
    and uses appinfra BufferedChannel over the DEALER transport for
    request/response (ask, feedback, shutdown from controller).
    """

    def __init__(
        self,
        lg: Logger,
        agent: Agent,
        bus_config: WorkerBusConfig,
        schedule_interval: float | None = None,
    ) -> None:
        self._lg = lg
        self._agent = agent
        self._bus_config = bus_config
        self._stop_event = threading.Event()
        self._schedule_interval = schedule_interval
        self._bus: ZMQWorkerBus | None = None
        self._channel: BufferedChannel[Any, Any] | None = None

        if schedule_interval is not None and schedule_interval > 0:
            self._ticker: Ticker | None = Ticker(
                lg,
                secs=schedule_interval,
                mode=TickerMode.FLEX,
                initial=True,
            )
        else:
            self._ticker = None

    @property
    def _running(self) -> bool:
        """Thread-safe running check."""
        return not self._stop_event.is_set()

    @_running.setter
    def _running(self, value: bool) -> None:
        """Thread-safe running setter (for service.py compatibility)."""
        if value:
            self._stop_event.clear()
        else:
            self._stop_event.set()

    def request_shutdown(self) -> None:
        """Request the runner to stop (thread-safe)."""
        self._stop_event.set()

    def run(self) -> None:
        """Main loop (blocking). Connect to bus, then process requests + cycles.

        Exceptions propagate to the caller (ThreadRunner/ProcessRunner) so
        restart_on_failure policies can trigger.
        """
        self._lg.debug("starting runner...", extra={"agent": self._agent.name})

        try:
            self._stop_event.clear()
            self._connect_bus()
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            self._disconnect_bus()
            self._stop_agent()

    # -------------------------------------------------------------------------
    # Bus lifecycle
    # -------------------------------------------------------------------------

    def _connect_bus(self) -> None:
        """Connect bus, create channel from DEALER transport, register with hub."""
        from llm_gent.bus.transport import ZMQWorkerBus

        self._bus = ZMQWorkerBus(self._lg, self._agent.name, self._bus_config)
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

        # Register with hub via channel (request/response)
        req = RegisterRequest(agent_id=self._agent.name)
        try:
            self._channel.submit(req, timeout=5.0)
            self._lg.info("registered on bus", extra={"agent": self._agent.name})
        except Exception as e:
            self._lg.warning(
                "bus registration failed",
                extra={"agent": self._agent.name, "exception": e},
            )

    def _disconnect_bus(self) -> None:
        """Unregister and disconnect."""
        import contextlib

        if self._channel is not None:
            with contextlib.suppress(Exception):
                self._channel.submit(UnregisterRequest(agent_id=self._agent.name), timeout=2.0)
            self._channel.close()
            self._channel = None

        if self._bus is not None:
            self._bus.stop()
            self._bus = None

    def _stop_agent(self) -> None:
        """Stop the agent."""
        try:
            self._agent.stop()
        except Exception as e:
            self._lg.warning("error stopping agent", extra={"exception": e})
        self._lg.info("runner stopped", extra={"agent": self._agent.name})

    def _handle_broadcast(self, message: Message) -> None:
        """Handle broadcast messages from hub (system-tier).

        Responds to:
        - HeartbeatRequest: reply with stats on heartbeat topic.
        - ShutdownNotice: initiate graceful shutdown.
        """
        if isinstance(message, HeartbeatRequest):
            self._respond_heartbeat(message)
        elif isinstance(message, ShutdownNotice):
            self._handle_shutdown_notice(message)

    def _respond_heartbeat(self, request: HeartbeatRequest) -> None:
        """Respond to hub heartbeat broadcast with agent stats."""
        if self._bus is None:
            return
        try:
            self._bus.publish_heartbeat(
                stats={"ticks": self._agent.cycle_count, "errors": 0},
                round_id=request.round_id,
                request_id=request.id,
            )
        except Exception as e:
            self._lg.debug(
                "heartbeat response failed",
                extra={"agent": self._agent.name, "exception": e},
            )

    def _handle_shutdown_notice(self, notice: ShutdownNotice) -> None:
        """Handle hub shutdown broadcast — begin graceful shutdown."""
        self._lg.info(
            "hub shutdown notice received",
            extra={
                "agent": self._agent.name,
                "reason": notice.reason,
                "grace_secs": notice.grace_period_secs,
            },
        )
        self._stop_event.set()

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main execution loop: handle incoming requests + scheduled cycles."""
        if hasattr(self._agent, "run") and callable(self._agent.run):
            self._lg.debug("delegating to agent.run()", extra={"agent": self._agent.name})
            self._run_agent_loop()
            return

        self._run_framework_loop()

    def _run_framework_loop(self) -> None:
        """Framework-controlled loop with request polling + scheduling."""
        self._lg.trace(
            "entering framework run loop",
            extra={"agent": self._agent.name, "mode": self._get_execution_mode()},
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
            target=self._request_poll_loop, daemon=True, name=f"req-{self._agent.name}"
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

    # -------------------------------------------------------------------------
    # Request handling
    # -------------------------------------------------------------------------

    def _handle_request(self, request: Request) -> Response:
        """Handle incoming request from controller."""
        if isinstance(request, AskRequest):
            return self._handle_ask(request)
        if isinstance(request, FeedbackRequest):
            return self._handle_feedback(request)
        if isinstance(request, ShutdownRequest):
            return self._handle_shutdown(request)
        return Response(id=request.id, success=False, error="unknown request type")

    def _handle_ask(self, request: AskRequest) -> AskResponse:
        """Handle ask request."""
        try:
            response_text = self._agent.ask(request.question)
            return AskResponse(id=request.id, response=response_text)
        except Exception as e:
            return AskResponse(id=request.id, success=False, error=str(e))

    def _handle_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Handle feedback request."""
        try:
            self._agent.record_feedback(request.message)
            return FeedbackResponse(id=request.id)
        except Exception as e:
            return FeedbackResponse(id=request.id, success=False, error=str(e))

    def _handle_shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Handle shutdown request."""
        self._lg.info("shutdown requested", extra={"agent": self._agent.name})
        self._stop_event.set()
        return ShutdownResponse(id=request.id)

    # -------------------------------------------------------------------------
    # Agent-to-agent relay
    # -------------------------------------------------------------------------

    def relay(self, to_agent: str, message: Message, timeout: float = 30.0) -> RelayResponse:
        """Send a request to another agent via the hub relay.

        The hub forwards the message to the target agent's channel,
        waits for the response, and returns it.

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
            from_agent=self._agent.name,
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
        self._lg.debug("running scheduled cycle", extra={"agent": self._agent.name})
        try:
            self._agent.run_once()
        except Exception as e:
            self._lg.warning("cycle failed", extra={"agent": self._agent.name, "exception": e})
