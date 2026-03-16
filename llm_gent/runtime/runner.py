"""Agent runner - runs a single agent in a subprocess.

The AgentRunner is the entry point for agent subprocesses. It:
- Runs a sync main loop processing messages from the channel
- Handles scheduled execution using appinfra.time.Ticker
- Responds to shutdown messages cleanly
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from appinfra.time import Ticker, TickerMode

from .transport import Message, MessageType, Response


if TYPE_CHECKING:
    from appinfra.log import Logger

    from llm_gent.core.agent import Agent
    from llm_gent.runtime.transport import Channel


class AgentRunner:
    """Runs a single agent in a subprocess.

    The runner is responsible for:
    - Running the main message loop
    - Handling scheduled execution
    - Processing commands from Core

    This class runs in the subprocess - it receives a channel for
    communication with the main process (Core).

    The main loop is intentionally sync and simple:
    1. Wait for message (with timeout if scheduled)
    2. Handle message or run scheduled cycle
    3. Repeat until shutdown
    """

    def __init__(
        self,
        lg: Logger,
        agent: Agent,
        channel: Channel,
        schedule_interval: float | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            lg: Logger instance.
            agent: The agent to run (must be started by caller).
            channel: Communication channel to Core.
            schedule_interval: Optional interval in seconds for scheduled execution.
                              None = no scheduling (message-only mode)
                              0 = continuous execution (tight loop)
                              >0 = scheduled with interval
        """
        self._lg = lg
        self._agent = agent
        self._channel = channel
        self._running = False
        self._schedule_interval = schedule_interval

        # Create Ticker for scheduled execution (not for continuous/none)
        if schedule_interval is not None and schedule_interval > 0:
            self._ticker: Ticker | None = Ticker(
                lg,
                secs=schedule_interval,
                mode=TickerMode.FLEX,  # Fixed-rate timing with no catch-up
                initial=True,  # Run immediately on first cycle
            )
        else:
            self._ticker = None

    def run(self) -> None:
        """Main loop (blocking, sync). Called in subprocess.

        Enters the message processing loop. The agent must already be started.
        Exits when shutdown message received or channel closes.
        """
        self._lg.debug("starting runner...", extra={"agent": self._agent.name})

        try:
            self._running = True

            # Notify Core we're running
            self._channel.send(
                Message(type=MessageType.STARTED, payload={"name": self._agent.name})
            )

            self._run_loop()

        except KeyboardInterrupt:
            pass  # Clean shutdown, no traceback
        except Exception as e:
            self._lg.warning("runner error", extra={"agent": self._agent.name, "exception": e})
            self._channel.send(
                Message(type=MessageType.ERROR, payload={"name": self._agent.name, "error": str(e)})
            )
        finally:
            self._cleanup()

    def _run_loop(self) -> None:
        """Main message processing loop."""
        # If agent has run(), delegate loop control to agent
        if hasattr(self._agent, "run") and callable(self._agent.run):
            self._lg.debug("delegating to agent.run()", extra={"agent": self._agent.name})
            self._run_agent_loop()
            return

        self._run_framework_loop()

    def _run_framework_loop(self) -> None:
        """Legacy framework-controlled loop for agents without run()."""
        self._lg.trace(
            "entering framework run loop",
            extra={"agent": self._agent.name, "mode": self._get_execution_mode()},
        )
        while self._running:
            timeout = self._calculate_timeout()
            msg = self._channel.recv(timeout=timeout)
            if msg is None:
                if self._should_run_cycle():
                    self._run_cycle()
            else:
                self._handle_message(msg)

    def _get_execution_mode(self) -> str:
        """Get execution mode string for logging."""
        if self._schedule_interval == 0:
            return "continuous"
        if self._ticker:
            return "scheduled"
        return "message-only"

    def _calculate_timeout(self) -> float:
        """Calculate recv timeout based on execution mode.

        Returns:
            float: Timeout in seconds for channel.recv()
        """
        if self._ticker is not None:
            # Scheduled mode - use Ticker's timing (clamped to non-negative)
            return max(0.0, self._ticker.time_until_next_tick())
        elif self._schedule_interval == 0:
            # Continuous mode - non-blocking
            return 0.0
        else:
            # Message-only mode (no scheduling) - poll periodically
            return 60.0

    def _should_run_cycle(self) -> bool:
        """Check if cycle should run now.

        Returns:
            bool: True if it's time to run a cycle
        """
        if self._ticker is not None:
            # Scheduled mode - let Ticker decide
            return self._ticker.try_tick()
        else:
            # Continuous mode (interval=0) or message-only (interval=None)
            return self._schedule_interval == 0

    def _run_cycle(self) -> None:
        """Run one scheduled execution cycle."""
        self._lg.debug("running scheduled cycle", extra={"agent": self._agent.name})
        try:
            result = self._agent.run_once()
            self._channel.send(
                Message(
                    type=MessageType.CYCLE_COMPLETE,
                    payload={
                        "name": self._agent.name,
                        "success": result.success,
                        "iterations": result.iterations,
                    },
                )
            )
        except Exception as e:
            self._lg.warning("cycle failed", extra={"agent": self._agent.name, "exception": e})
            self._channel.send(
                Message(
                    type=MessageType.CYCLE_ERROR,
                    payload={"name": self._agent.name, "error": str(e)},
                )
            )

    def _run_agent_loop(self) -> None:
        """Delegate loop control to agent's run() method.

        Agent manages its own loop, schedule, and exit conditions.
        """
        try:
            result = self._agent.run()  # type: ignore[attr-defined]
            self._channel.send(
                Message(
                    type=MessageType.CYCLE_COMPLETE,
                    payload={
                        "name": self._agent.name,
                        "success": result.success,
                        "iterations": result.iterations,
                    },
                )
            )
        except Exception as e:
            self._lg.warning("agent run failed", extra={"agent": self._agent.name, "exception": e})
            self._channel.send(
                Message(
                    type=MessageType.CYCLE_ERROR,
                    payload={"name": self._agent.name, "error": str(e)},
                )
            )

    def _handle_message(self, msg: Message) -> None:
        """Handle message from Core."""
        self._lg.debug("received message", extra={"agent": self._agent.name, "type": msg.type})

        match msg.type:
            case MessageType.SHUTDOWN:
                self._running = False

            case MessageType.ASK:
                self._handle_ask(msg)

            case MessageType.FEEDBACK:
                self._handle_feedback(msg)

            case MessageType.GET_INSIGHTS:
                self._handle_get_insights(msg)

            case MessageType.RUN_CYCLE:
                self._run_cycle()

            case _:
                self._lg.warning(
                    "unknown message type", extra={"agent": self._agent.name, "type": msg.type}
                )

    def _handle_ask(self, msg: Message) -> None:
        """Handle ask request."""
        question = msg.payload.get("question", "")
        self._lg.debug(
            "handling ask", extra={"agent": self._agent.name, "question_len": len(question)}
        )

        try:
            response_text = self._agent.ask(question)
            self._lg.debug(
                "ask completed",
                extra={"agent": self._agent.name, "response_len": len(response_text)},
            )
            self._channel.send(
                Response(
                    id=uuid4().hex,
                    type=MessageType.ASK_RESPONSE,
                    request_id=msg.id,
                    success=True,
                    payload={"response": response_text},
                )
            )
        except Exception as e:
            self._lg.warning("ask failed", extra={"agent": self._agent.name, "exception": e})
            self._send_error_response(msg, str(e))

    def _handle_feedback(self, msg: Message) -> None:
        """Handle feedback request."""
        message = msg.payload.get("message", "")
        self._lg.debug("handling feedback", extra={"agent": self._agent.name})

        try:
            self._agent.record_feedback(message)
            self._lg.debug("feedback recorded", extra={"agent": self._agent.name})
            self._channel.send(
                Response(
                    id=uuid4().hex,
                    type=MessageType.FEEDBACK_RESPONSE,
                    request_id=msg.id,
                    success=True,
                )
            )
        except Exception as e:
            self._lg.warning("feedback failed", extra={"agent": self._agent.name, "exception": e})
            self._send_error_response(msg, str(e))

    def _handle_get_insights(self, msg: Message) -> None:
        """Handle get_insights request."""
        limit = msg.payload.get("limit", 10)
        self._lg.debug("handling get_insights", extra={"agent": self._agent.name, "limit": limit})

        try:
            insights = self._build_insights(limit)
            self._lg.debug(
                "get_insights completed", extra={"agent": self._agent.name, "count": len(insights)}
            )
            self._channel.send(
                Response(
                    id=uuid4().hex,
                    type=MessageType.INSIGHTS_RESPONSE,
                    request_id=msg.id,
                    success=True,
                    payload={
                        "insights": insights,
                        "cycle_count": self._agent.cycle_count,
                    },
                )
            )
        except Exception as e:
            self._lg.warning(
                "get_insights failed", extra={"agent": self._agent.name, "exception": e}
            )
            self._send_error_response(msg, str(e))

    def _build_insights(self, limit: int) -> list[dict[str, Any]]:
        """Build insights list from recent results."""
        results = self._agent.get_recent_results(limit)
        return [
            {
                "success": r.success,
                "content": r.content[:500],
                "iterations": r.iterations,
            }
            for r in results
        ]

    def _send_error_response(self, msg: Message, error: str) -> None:
        """Send error response for a request."""
        # Map request types to response types
        response_type_map: dict[str, str] = {
            MessageType.ASK: MessageType.ASK_RESPONSE,
            MessageType.FEEDBACK: MessageType.FEEDBACK_RESPONSE,
            MessageType.GET_INSIGHTS: MessageType.INSIGHTS_RESPONSE,
        }
        response_type = response_type_map.get(msg.type, MessageType.ERROR)

        self._channel.send(
            Response(
                id=uuid4().hex,
                type=response_type,
                request_id=msg.id,
                success=False,
                error=error,
            )
        )

    def _cleanup(self) -> None:
        """Clean up on exit."""
        try:
            self._agent.stop()
        except Exception as e:
            self._lg.warning("error stopping agent", extra={"exception": e})

        self._channel.close()
        self._lg.info("runner stopped", extra={"agent": self._agent.name})
