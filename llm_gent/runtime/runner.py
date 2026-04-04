"""Agent runner - runs a single agent in a subprocess or thread.

The AgentRunner is the entry point for agent subprocesses/threads. It:
- Connects to the hub via ZMQ bus
- Registers with the hub and sends periodic heartbeats
- Handles incoming requests (ask, feedback, shutdown) via the bus
- Handles scheduled execution using appinfra.time.Ticker
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from appinfra.time import Ticker, TickerMode

from llm_gent.bus.protocol import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    RegisterRequest,
    Request,
    Response,
    ShutdownRequest,
    ShutdownResponse,
    UnregisterRequest,
)
from llm_gent.bus.transport import ZMQWorkerBus


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_gent.bus.transport import WorkerBusConfig
    from llm_gent.core.agent import Agent


class AgentRunner:
    """Runs a single agent in a subprocess or thread.

    Connects to the hub via ZMQ bus for all communication:
    - Registration and heartbeats (agent -> hub)
    - Ask, feedback, shutdown requests (hub -> agent)
    - Scheduled execution via Ticker

    The main loop is sync and simple:
    1. Connect to bus, register with hub
    2. Wait for incoming requests or scheduled tick
    3. Repeat until shutdown request received
    """

    def __init__(
        self,
        lg: Logger,
        agent: Agent,
        bus_config: WorkerBusConfig,
        schedule_interval: float | None = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """Initialize the runner.

        Args:
            lg: Logger instance.
            agent: The agent to run (must be started by caller).
            bus_config: Bus config for connecting to the hub.
            schedule_interval: Optional interval in seconds for scheduled execution.
                              None = no scheduling (message-only mode)
                              0 = continuous execution (tight loop)
                              >0 = scheduled with interval
            heartbeat_interval: Seconds between heartbeats to hub.
        """
        self._lg = lg
        self._agent = agent
        self._bus_config = bus_config
        self._running = False
        self._schedule_interval = schedule_interval
        self._heartbeat_interval = heartbeat_interval
        self._bus: ZMQWorkerBus | None = None

        if schedule_interval is not None and schedule_interval > 0:
            self._ticker: Ticker | None = Ticker(
                lg,
                secs=schedule_interval,
                mode=TickerMode.FLEX,
                initial=True,
            )
        else:
            self._ticker = None

    def run(self) -> None:
        """Main loop (blocking, sync). Called in subprocess/thread.

        Connects to bus, registers with hub, then enters the main loop.
        Exits when shutdown request received or bus disconnects.
        """
        self._lg.debug("starting runner...", extra={"agent": self._agent.name})

        try:
            self._running = True
            self._connect_bus()
            self._run_loop()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self._lg.warning("runner error", extra={"agent": self._agent.name, "exception": e})
        finally:
            self._disconnect_bus()
            self._stop_agent()

    # -------------------------------------------------------------------------
    # Bus lifecycle
    # -------------------------------------------------------------------------

    def _connect_bus(self) -> None:
        """Connect to hub bus, register, and start heartbeat thread."""
        self._bus = ZMQWorkerBus(self._lg, self._agent.name, self._bus_config)
        self._bus.start()
        self._bus.on_request(self._handle_request)
        time.sleep(0.2)

        req = RegisterRequest(agent_id=self._agent.name)
        try:
            self._bus.send(req, timeout=5.0)
            self._lg.info("registered on bus", extra={"agent": self._agent.name})
        except Exception as e:
            self._lg.warning(
                "bus registration failed", extra={"agent": self._agent.name, "exception": e}
            )

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name=f"hb-{self._agent.name}"
        )
        self._heartbeat_thread.start()

    def _disconnect_bus(self) -> None:
        """Unregister from hub and disconnect."""
        if self._bus is None:
            return

        import contextlib

        with contextlib.suppress(Exception):
            self._bus.send(UnregisterRequest(agent_id=self._agent.name), timeout=2.0)

        self._bus.stop()
        self._bus = None

    def _stop_agent(self) -> None:
        """Stop the agent."""
        try:
            self._agent.stop()
        except Exception as e:
            self._lg.warning("error stopping agent", extra={"exception": e})
        self._lg.info("runner stopped", extra={"agent": self._agent.name})

    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the hub."""
        while self._running and self._bus is not None:
            time.sleep(self._heartbeat_interval)
            if not self._running or self._bus is None:
                break
            try:
                self._bus.publish_heartbeat(
                    {
                        "ticks": self._agent.cycle_count,
                        "errors": 0,
                    }
                )
            except Exception as e:
                self._lg.debug(
                    "heartbeat send failed", extra={"agent": self._agent.name, "exception": e}
                )

    # -------------------------------------------------------------------------
    # Request handling (hub -> agent via bus)
    # -------------------------------------------------------------------------

    def _handle_request(self, request: Request, sender_id: str | None) -> Response:
        """Handle incoming request from hub."""
        if isinstance(request, AskRequest):
            return self._handle_ask(request)
        if isinstance(request, FeedbackRequest):
            return self._handle_feedback(request)
        if isinstance(request, ShutdownRequest):
            return self._handle_shutdown(request)
        return Response(id=request.id, success=False, error="unknown request type")

    def _handle_ask(self, request: AskRequest) -> AskResponse:
        """Handle ask request from hub."""
        self._lg.debug("handling ask", extra={"agent": self._agent.name})
        try:
            response_text = self._agent.ask(request.question)
            return AskResponse(id=request.id, response=response_text)
        except Exception as e:
            return AskResponse(id=request.id, success=False, error=str(e))

    def _handle_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Handle feedback request from hub."""
        self._lg.debug("handling feedback", extra={"agent": self._agent.name})
        try:
            self._agent.record_feedback(request.message)
            return FeedbackResponse(id=request.id)
        except Exception as e:
            return FeedbackResponse(id=request.id, success=False, error=str(e))

    def _handle_shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Handle shutdown request -- signal the main loop to exit."""
        self._lg.info("shutdown requested", extra={"agent": self._agent.name})
        self._running = False
        return ShutdownResponse(id=request.id)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main execution loop."""
        if hasattr(self._agent, "run") and callable(self._agent.run):
            self._lg.debug("delegating to agent.run()", extra={"agent": self._agent.name})
            self._run_agent_loop()
            return

        self._run_framework_loop()

    def _run_framework_loop(self) -> None:
        """Framework-controlled loop with optional scheduling."""
        self._lg.trace(
            "entering framework run loop",
            extra={"agent": self._agent.name, "mode": self._get_execution_mode()},
        )
        while self._running:
            if self._should_run_cycle():
                self._run_cycle()
            else:
                time.sleep(min(self._calculate_sleep(), 1.0))

    def _run_agent_loop(self) -> None:
        """Delegate loop control to agent's run() method."""
        try:
            self._agent.run()  # type: ignore[attr-defined]
        except Exception as e:
            self._lg.warning("agent run failed", extra={"agent": self._agent.name, "exception": e})

    def _get_execution_mode(self) -> str:
        """Get execution mode string for logging."""
        if self._schedule_interval == 0:
            return "continuous"
        if self._ticker:
            return "scheduled"
        return "message-only"

    def _calculate_sleep(self) -> float:
        """Calculate sleep time based on execution mode."""
        if self._ticker is not None:
            return max(0.0, self._ticker.time_until_next_tick())
        if self._schedule_interval == 0:
            return 0.0
        return 1.0

    def _should_run_cycle(self) -> bool:
        """Check if a scheduled cycle should run now."""
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
