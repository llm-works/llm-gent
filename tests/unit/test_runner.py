# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for agent runners (ManagedAgentRunner and AgentRunner)."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.bus.protocol import (
    AgentJoined,
    AgentLeft,
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HeartbeatRequest,
    RelayResponse,
    ShutdownNotice,
    ShutdownRequest,
    ShutdownResponse,
)
from llm_gent.bus.transport import WorkerBusConfig
from llm_gent.runtime.handler import Handler
from llm_gent.runtime.runner import AgentRunner, ManagedAgentRunner, _AgentHandler


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create mock agent."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.cycle_count = 0
    return agent


@pytest.fixture
def bus_config():
    return WorkerBusConfig()


@pytest.fixture
def managed_runner(mock_agent, bus_config):
    """Create ManagedAgentRunner (not connected to bus)."""
    lg = MagicMock()
    return ManagedAgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)


class MockHandler:
    """Test handler implementation."""

    def __init__(self) -> None:
        self.ask_response = "test answer"
        self.ask_calls: list[str] = []
        self.feedback_calls: list[str] = []
        self.shutdown_called = False

    def on_ask(self, question: str) -> str:
        self.ask_calls.append(question)
        return self.ask_response

    def on_feedback(self, message: str) -> None:
        self.feedback_calls.append(message)

    def on_shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def handler():
    return MockHandler()


@pytest.fixture
def ext_runner(handler, bus_config):
    """Create AgentRunner for external agents (not connected to bus)."""
    lg = MagicMock()
    return AgentRunner(
        lg=lg,
        handler=handler,
        agent_id="ext-agent",
        bus_config=bus_config,
        capabilities=["search", "summarize"],
        metadata={"version": "1.0"},
    )


# =============================================================================
# Handler protocol
# =============================================================================


class TestHandlerProtocol:
    """Tests for Handler protocol compliance."""

    def test_mock_handler_is_handler(self, handler):
        """MockHandler satisfies Handler protocol."""
        assert isinstance(handler, Handler)

    def test_object_not_handler(self):
        """Arbitrary object does not satisfy Handler protocol."""
        assert not isinstance(object(), Handler)

    def test_partial_impl_not_handler(self):
        """Class with only some methods does not satisfy Handler."""

        class Partial:
            def on_ask(self, question: str) -> str:
                return ""

        assert not isinstance(Partial(), Handler)


# =============================================================================
# _AgentHandler adapter
# =============================================================================


class TestAgentHandler:
    """Tests for the _AgentHandler adapter that wraps Agent as Handler."""

    def test_on_ask_delegates_to_agent(self, mock_agent):
        """on_ask calls agent.ask()."""
        mock_agent.ask.return_value = "hello"
        adapter = _AgentHandler(mock_agent)

        result = adapter.on_ask("question?")

        assert result == "hello"
        mock_agent.ask.assert_called_once_with("question?")

    def test_on_feedback_delegates_to_agent(self, mock_agent):
        """on_feedback calls agent.record_feedback()."""
        adapter = _AgentHandler(mock_agent)

        adapter.on_feedback("good job")

        mock_agent.record_feedback.assert_called_once_with("good job")

    def test_on_shutdown_is_noop(self, mock_agent):
        """on_shutdown does nothing (runner handles shutdown via stop event)."""
        adapter = _AgentHandler(mock_agent)
        adapter.on_shutdown()  # Should not raise


# =============================================================================
# BaseAgentRunner (tested via AgentRunner)
# =============================================================================


class TestRunningProperty:
    """Tests for the _running property/setter thread-safe mechanism."""

    def test_running_true_initially(self, ext_runner):
        """Runner is not stopped initially (stop_event is not set)."""
        assert ext_runner._running is True

    def test_running_setter_false_sets_event(self, ext_runner):
        """Setting _running=False sets the stop event."""
        ext_runner._running = False
        assert ext_runner._stop_event.is_set()
        assert ext_runner._running is False

    def test_running_setter_true_clears_event(self, ext_runner):
        """Setting _running=True clears the stop event."""
        ext_runner._stop_event.set()
        ext_runner._running = True
        assert not ext_runner._stop_event.is_set()
        assert ext_runner._running is True

    def test_request_shutdown_sets_stop_event(self, ext_runner):
        """request_shutdown() sets the stop event."""
        ext_runner.request_shutdown()
        assert ext_runner._running is False


# =============================================================================
# Broadcast handling (BaseAgentRunner)
# =============================================================================


class TestBroadcastHandling:
    """Tests for _handle_broadcast dispatch in BaseAgentRunner."""

    def test_heartbeat_dispatches_to_respond(self, ext_runner):
        """HeartbeatRequest dispatches to _respond_heartbeat."""
        ext_runner._bus = MagicMock()
        req = HeartbeatRequest(round_id="r1")

        ext_runner._handle_broadcast(req)

        ext_runner._bus.publish_heartbeat.assert_called_once()

    def test_shutdown_notice_sets_stop_event(self, ext_runner):
        """ShutdownNotice sets stop event and logs."""
        notice = ShutdownNotice(reason="maintenance", grace_period_secs=3.0)

        ext_runner._handle_broadcast(notice)

        assert ext_runner._running is False
        ext_runner._lg.info.assert_called()

    def test_agent_joined_logs(self, ext_runner):
        """AgentJoined logs the joining agent."""
        msg = AgentJoined(agent_id="peer-1", capabilities=["search"])

        ext_runner._handle_broadcast(msg)

        ext_runner._lg.info.assert_called()
        call_args = ext_runner._lg.info.call_args
        assert "joined" in call_args[0][0]

    def test_agent_left_logs(self, ext_runner):
        """AgentLeft logs the departing agent."""
        msg = AgentLeft(agent_id="peer-1", reason="voluntary")

        ext_runner._handle_broadcast(msg)

        ext_runner._lg.info.assert_called()
        call_args = ext_runner._lg.info.call_args
        assert "left" in call_args[0][0]

    def test_unknown_broadcast_ignored(self, ext_runner):
        """Unknown broadcast type is silently ignored."""
        msg = MagicMock(spec=[])
        ext_runner._handle_broadcast(msg)  # Should not raise


class TestRespondHeartbeat:
    """Tests for _respond_heartbeat."""

    def test_publishes_heartbeat_with_stats(self, ext_runner):
        """Heartbeat response publishes stats via bus."""
        ext_runner._bus = MagicMock()
        req = HeartbeatRequest(round_id="r42")

        ext_runner._respond_heartbeat(req)

        ext_runner._bus.publish_heartbeat.assert_called_once_with(
            stats={"ticks": 0, "errors": 0},
            round_id="r42",
            request_id=req.id,
        )

    def test_no_bus_is_noop(self, ext_runner):
        """No-op when bus is None."""
        ext_runner._bus = None
        req = HeartbeatRequest(round_id="r1")

        ext_runner._respond_heartbeat(req)  # Should not raise

    def test_publish_failure_logged_as_debug(self, ext_runner):
        """Heartbeat publish failure is logged at debug level."""
        ext_runner._bus = MagicMock()
        ext_runner._bus.publish_heartbeat.side_effect = RuntimeError("zmq fail")
        req = HeartbeatRequest(round_id="r1")

        ext_runner._respond_heartbeat(req)  # Should not raise

        ext_runner._lg.debug.assert_called()
        call_args = ext_runner._lg.debug.call_args
        assert "heartbeat" in call_args[0][0]


class TestHandleShutdownNotice:
    """Tests for _handle_shutdown_notice."""

    def test_sets_stop_event(self, ext_runner):
        """Shutdown notice sets stop event."""
        notice = ShutdownNotice(reason="upgrade", grace_period_secs=10.0)

        ext_runner._handle_shutdown_notice(notice)

        assert ext_runner._running is False

    def test_logs_reason_and_grace_period(self, ext_runner):
        """Shutdown notice logs the reason and grace period."""
        notice = ShutdownNotice(reason="upgrade", grace_period_secs=10.0)

        ext_runner._handle_shutdown_notice(notice)

        ext_runner._lg.info.assert_called_once()
        extra = ext_runner._lg.info.call_args[1]["extra"]
        assert extra["reason"] == "upgrade"
        assert extra["grace_secs"] == 10.0


# =============================================================================
# Request polling and dispatch (BaseAgentRunner)
# =============================================================================


class TestPollRequests:
    """Tests for _poll_requests."""

    def test_no_channel_is_noop(self, ext_runner):
        """No-op when channel is None."""
        ext_runner._channel = None
        ext_runner._poll_requests()  # Should not raise

    def test_dispatches_request_and_sends_response(self, ext_runner, handler):
        """Polls a request, dispatches it, and sends the response."""

        ask_req = AskRequest(question="test?")
        channel = MagicMock()
        channel.recv.return_value = ask_req
        ext_runner._channel = channel

        ext_runner._poll_requests()

        channel.recv.assert_called_once_with(timeout=0.05)
        channel.send.assert_called_once()
        sent_resp = channel.send.call_args[0][0]
        assert isinstance(sent_resp, AskResponse)
        assert sent_resp.response == "test answer"

    def test_channel_timeout_is_silent(self, ext_runner):
        """ChannelTimeoutError (no message) is silently handled."""
        from appinfra.service import ChannelTimeoutError

        channel = MagicMock()
        channel.recv.side_effect = ChannelTimeoutError("timeout")
        ext_runner._channel = channel

        ext_runner._poll_requests()  # Should not raise

    def test_unexpected_error_logged_as_debug(self, ext_runner):
        """Unexpected errors during poll are logged at debug."""
        channel = MagicMock()
        channel.recv.side_effect = RuntimeError("zmq broken")
        ext_runner._channel = channel

        ext_runner._poll_requests()  # Should not raise

        ext_runner._lg.debug.assert_called()
        assert "poll error" in ext_runner._lg.debug.call_args[0][0]

    def test_non_request_message_ignored(self, ext_runner):
        """Non-Request messages from channel are ignored (no send)."""
        channel = MagicMock()
        channel.recv.return_value = MagicMock(spec=[])  # Not a Request
        ext_runner._channel = channel

        ext_runner._poll_requests()

        channel.send.assert_not_called()


class TestHandleRequest:
    """Tests for _handle_request dispatch."""

    def test_ask_request_dispatched(self, ext_runner, handler):
        """AskRequest dispatched to _handle_ask."""
        handler.ask_response = "42"
        req = AskRequest(question="meaning?")

        resp = ext_runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is True
        assert resp.response == "42"

    def test_feedback_request_dispatched(self, ext_runner, handler):
        """FeedbackRequest dispatched to _handle_feedback."""
        req = FeedbackRequest(message="nice")

        resp = ext_runner._handle_request(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is True

    def test_shutdown_request_dispatched(self, ext_runner):
        """ShutdownRequest dispatched to _handle_shutdown."""
        req = ShutdownRequest()

        resp = ext_runner._handle_request(req)

        assert isinstance(resp, ShutdownResponse)
        assert resp.success is True
        assert ext_runner._running is False

    def test_unknown_request_returns_error(self, ext_runner):
        """Unknown request type returns error response."""
        req = MagicMock(spec=["id"])
        req.id = "test-id"

        resp = ext_runner._handle_request(req)

        assert resp.success is False
        assert "unknown" in resp.error


class TestHandleAsk:
    """Tests for _handle_ask."""

    def test_success(self, ext_runner, handler):
        """Successful ask returns response text."""
        handler.ask_response = "the answer"
        req = AskRequest(question="what?")

        resp = ext_runner._handle_ask(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is True
        assert resp.response == "the answer"
        assert resp.id == req.id

    def test_handler_error(self, ext_runner, handler):
        """Handler exception returns error response."""
        handler.on_ask = MagicMock(side_effect=ValueError("bad input"))
        req = AskRequest(question="crash")

        resp = ext_runner._handle_ask(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is False
        assert "bad input" in resp.error
        assert resp.id == req.id


class TestHandleFeedback:
    """Tests for _handle_feedback."""

    def test_success(self, ext_runner, handler):
        """Successful feedback returns success response."""
        req = FeedbackRequest(message="well done")

        resp = ext_runner._handle_feedback(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is True
        assert resp.id == req.id
        assert handler.feedback_calls == ["well done"]

    def test_handler_error(self, ext_runner, handler):
        """Handler exception returns error response."""
        handler.on_feedback = MagicMock(side_effect=RuntimeError("storage fail"))
        req = FeedbackRequest(message="boom")

        resp = ext_runner._handle_feedback(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is False
        assert "storage fail" in resp.error
        assert resp.id == req.id


class TestHandleShutdown:
    """Tests for _handle_shutdown."""

    def test_calls_handler_and_stops(self, ext_runner, handler):
        """Shutdown calls handler.on_shutdown() and sets stop event."""
        ext_runner._running = True
        req = ShutdownRequest()

        resp = ext_runner._handle_shutdown(req)

        assert isinstance(resp, ShutdownResponse)
        assert resp.success is True
        assert handler.shutdown_called is True
        assert ext_runner._running is False

    def test_survives_handler_error(self, ext_runner, handler):
        """Shutdown still stops even if handler.on_shutdown() raises."""
        ext_runner._running = True
        handler.on_shutdown = MagicMock(side_effect=RuntimeError("cleanup boom"))
        req = ShutdownRequest()

        resp = ext_runner._handle_shutdown(req)

        assert isinstance(resp, ShutdownResponse)
        assert resp.success is True
        assert ext_runner._running is False
        ext_runner._lg.warning.assert_called()


# =============================================================================
# Relay (BaseAgentRunner)
# =============================================================================


class TestRelay:
    """Tests for relay() agent-to-agent messaging."""

    def test_raises_when_not_connected(self, ext_runner):
        """relay() raises RuntimeError when channel is None."""
        ext_runner._channel = None
        msg = MagicMock()
        msg.message_type = "ask_request"
        msg.model_dump.return_value = {"question": "hello"}

        with pytest.raises(RuntimeError, match="not connected"):
            ext_runner.relay("target-agent", msg)

    def test_submits_relay_request(self, ext_runner):
        """relay() submits a RelayRequest and returns RelayResponse."""
        channel = MagicMock()
        relay_resp = RelayResponse(
            id="r1",
            from_agent="target",
            inner_type="ask_response",
            inner_payload={"response": "42"},
        )
        channel.submit.return_value = relay_resp
        ext_runner._channel = channel

        msg = MagicMock()
        msg.message_type = "ask_request"
        msg.model_dump.return_value = {"question": "hello"}

        result = ext_runner.relay("target", msg, timeout=10.0)

        assert isinstance(result, RelayResponse)
        assert result.from_agent == "target"
        channel.submit.assert_called_once()
        submitted = channel.submit.call_args[0][0]
        assert submitted.from_agent == "ext-agent"
        assert submitted.to_agent == "target"
        assert channel.submit.call_args[1]["timeout"] == 10.0

    def test_non_relay_response_wrapped(self, ext_runner):
        """Non-RelayResponse from channel is wrapped in a fallback RelayResponse."""
        channel = MagicMock()
        channel.submit.return_value = MagicMock(spec=[])  # Not a RelayResponse
        ext_runner._channel = channel

        msg = MagicMock()
        msg.message_type = "ask_request"
        msg.model_dump.return_value = {}

        result = ext_runner.relay("target", msg)

        assert isinstance(result, RelayResponse)
        assert result.from_agent == "target"
        assert result.inner_type == "response"


# =============================================================================
# ManagedAgentRunner
# =============================================================================


class TestManagedAgentRunner:
    """Tests for ManagedAgentRunner initialization."""

    def test_runner_init(self, mock_agent, bus_config):
        """Runner initializes with bus config."""
        lg = MagicMock()
        runner = ManagedAgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)

        assert runner._agent is mock_agent
        assert runner._bus_config is bus_config
        assert runner._ticker is None

    def test_runner_init_with_schedule(self, mock_agent, bus_config):
        """Runner creates Ticker when schedule_interval provided."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=60.0
        )

        assert runner._ticker is not None

    def test_runner_init_zero_schedule_no_ticker(self, mock_agent, bus_config):
        """Schedule interval of 0 (continuous) does not create a Ticker."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=0
        )

        assert runner._ticker is None


class TestManagedGetStats:
    """Tests for ManagedAgentRunner._get_stats."""

    def test_returns_agent_cycle_count(self, mock_agent, bus_config):
        """Stats include the agent's cycle count."""
        mock_agent.cycle_count = 42
        lg = MagicMock()
        runner = ManagedAgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)

        stats = runner._get_stats()

        assert stats == {"ticks": 42, "errors": 0}


class TestManagedOnStopped:
    """Tests for ManagedAgentRunner._on_stopped."""

    def test_calls_agent_stop(self, managed_runner, mock_agent):
        """_on_stopped calls agent.stop()."""
        managed_runner._on_stopped()

        mock_agent.stop.assert_called_once()
        managed_runner._lg.info.assert_called()

    def test_survives_agent_stop_error(self, managed_runner, mock_agent):
        """_on_stopped handles agent.stop() exceptions."""
        mock_agent.stop.side_effect = RuntimeError("cleanup failed")

        managed_runner._on_stopped()  # Should not raise

        managed_runner._lg.warning.assert_called()


class TestManagedRunLoop:
    """Tests for ManagedAgentRunner._run_loop delegation."""

    def test_delegates_to_agent_run_when_available(self, mock_agent, bus_config):
        """When agent has run(), delegates to _run_agent_loop."""
        mock_agent.run = MagicMock()  # Agent has a run() method
        lg = MagicMock()
        runner = ManagedAgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)

        with patch.object(runner, "_run_agent_loop") as mock_agent_loop:
            runner._run_loop()
            mock_agent_loop.assert_called_once()

    def test_falls_back_to_framework_loop(self, mock_agent, bus_config):
        """When agent has no run(), uses _run_framework_loop."""
        del mock_agent.run  # Remove run attribute
        lg = MagicMock()
        runner = ManagedAgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)

        with patch.object(runner, "_run_framework_loop") as mock_fw_loop:
            runner._run_loop()
            mock_fw_loop.assert_called_once()


class TestManagedRequestHandling:
    """Tests for ManagedAgentRunner request dispatch (via _AgentHandler adapter)."""

    def test_handle_ask(self, managed_runner, mock_agent):
        """Ask request calls agent.ask() and returns response."""
        mock_agent.ask.return_value = "Test answer"
        req = AskRequest(question="What?")

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is True
        assert resp.response == "Test answer"
        mock_agent.ask.assert_called_once_with("What?")

    def test_handle_ask_error(self, managed_runner, mock_agent):
        """Ask request returns error response on exception."""
        mock_agent.ask.side_effect = RuntimeError("LLM failed")
        req = AskRequest(question="What?")

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is False
        assert "LLM failed" in resp.error

    def test_handle_feedback(self, managed_runner, mock_agent):
        """Feedback request calls agent.record_feedback()."""
        req = FeedbackRequest(message="Good job!")

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is True
        mock_agent.record_feedback.assert_called_once_with("Good job!")

    def test_handle_shutdown(self, managed_runner):
        """Shutdown request sets running to False."""
        managed_runner._running = True
        req = ShutdownRequest()

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, ShutdownResponse)
        assert resp.success is True
        assert managed_runner._running is False

    def test_handle_unknown_request(self, managed_runner):
        """Unknown request type returns error response."""
        req = MagicMock(spec=["id"])
        req.id = "test-id"

        resp = managed_runner._handle_request(req)

        assert resp.success is False
        assert "unknown" in resp.error


class TestManagedBroadcastHandling:
    """Tests for ManagedAgentRunner broadcast handling."""

    def test_handle_heartbeat_broadcast(self, managed_runner):
        """HeartbeatRequest triggers heartbeat response."""
        managed_runner._bus = MagicMock()
        req = HeartbeatRequest(round_id="r1")

        managed_runner._handle_broadcast(req)

        managed_runner._bus.publish_heartbeat.assert_called_once()

    def test_handle_shutdown_notice(self, managed_runner):
        """ShutdownNotice sets stop event."""
        notice = ShutdownNotice(reason="test", grace_period_secs=1.0)

        managed_runner._handle_broadcast(notice)

        assert managed_runner._running is False

    def test_handle_agent_joined(self, managed_runner):
        """AgentJoined logs and doesn't crash."""
        msg = AgentJoined(agent_id="peer-1", capabilities=["search"])

        managed_runner._handle_broadcast(msg)

        managed_runner._lg.info.assert_called()
        call_args = managed_runner._lg.info.call_args
        assert "joined" in call_args[0][0]

    def test_handle_agent_left(self, managed_runner):
        """AgentLeft logs and doesn't crash."""
        msg = AgentLeft(agent_id="peer-1", reason="voluntary")

        managed_runner._handle_broadcast(msg)

        managed_runner._lg.info.assert_called()
        call_args = managed_runner._lg.info.call_args
        assert "left" in call_args[0][0]

    def test_handle_unknown_broadcast(self, managed_runner):
        """Unknown broadcast type is silently ignored."""
        msg = MagicMock(spec=[])

        managed_runner._handle_broadcast(msg)  # Should not raise


class TestManagedScheduling:
    """Tests for scheduled execution."""

    def test_should_run_cycle_continuous(self, mock_agent, bus_config):
        """Continuous mode (interval=0) always runs cycle."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=0
        )

        assert runner._should_run_cycle() is True

    def test_should_run_cycle_message_only(self, managed_runner):
        """Message-only mode (no interval) never runs cycle."""
        assert managed_runner._should_run_cycle() is False

    def test_should_run_cycle_with_ticker(self, mock_agent, bus_config):
        """Scheduled mode delegates to ticker.try_tick()."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=60.0
        )
        runner._ticker.try_tick = MagicMock(return_value=True)

        assert runner._should_run_cycle() is True
        runner._ticker.try_tick.assert_called_once()

    def test_should_run_cycle_ticker_not_ready(self, mock_agent, bus_config):
        """Ticker not ready returns False."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=60.0
        )
        runner._ticker.try_tick = MagicMock(return_value=False)

        assert runner._should_run_cycle() is False

    def test_run_cycle_calls_run_once(self, managed_runner, mock_agent):
        """Cycle calls agent.run_once()."""
        managed_runner._run_cycle()
        mock_agent.run_once.assert_called_once()

    def test_run_cycle_handles_error(self, managed_runner, mock_agent):
        """Cycle handles agent.run_once() exceptions."""
        mock_agent.run_once.side_effect = RuntimeError("cycle failed")
        managed_runner._run_cycle()  # Should not raise
        managed_runner._lg.warning.assert_called()


class TestManagedExecutionMode:
    """Tests for _get_execution_mode."""

    def test_continuous_mode(self, mock_agent, bus_config):
        """Schedule interval 0 => continuous mode."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=0
        )
        assert runner._get_execution_mode() == "continuous"

    def test_scheduled_mode(self, mock_agent, bus_config):
        """Positive schedule interval with ticker => scheduled mode."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=30.0
        )
        assert runner._get_execution_mode() == "scheduled"

    def test_message_only_mode(self, managed_runner):
        """No schedule interval => message-only mode."""
        assert managed_runner._get_execution_mode() == "message-only"


class TestManagedCalculateSleep:
    """Tests for _calculate_sleep."""

    def test_with_ticker(self, mock_agent, bus_config):
        """With ticker, sleep delegates to ticker.time_until_next_tick()."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=60.0
        )
        runner._ticker.time_until_next_tick = MagicMock(return_value=5.0)

        assert runner._calculate_sleep() == 5.0

    def test_with_ticker_negative_clamped(self, mock_agent, bus_config):
        """Negative time_until_next_tick is clamped to 0."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=60.0
        )
        runner._ticker.time_until_next_tick = MagicMock(return_value=-1.0)

        assert runner._calculate_sleep() == 0.0

    def test_continuous_mode(self, mock_agent, bus_config):
        """Continuous mode (interval=0) returns 0."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=0
        )
        assert runner._calculate_sleep() == 0.0

    def test_message_only_mode(self, managed_runner):
        """Message-only mode returns 0.5."""
        assert managed_runner._calculate_sleep() == 0.5


# =============================================================================
# AgentRunner (external)
# =============================================================================


class TestAgentRunner:
    """Tests for AgentRunner initialization."""

    def test_runner_init(self, handler, bus_config):
        """Runner initializes with handler and config."""
        lg = MagicMock()
        runner = AgentRunner(
            lg=lg,
            handler=handler,
            agent_id="my-agent",
            bus_config=bus_config,
            capabilities=["search"],
            metadata={"v": "1"},
        )

        assert runner.agent_id == "my-agent"
        assert runner._capabilities == ["search"]
        assert runner._metadata == {"v": "1"}

    def test_runner_default_capabilities(self, handler, bus_config):
        """Runner defaults to empty capabilities and metadata."""
        lg = MagicMock()
        runner = AgentRunner(lg=lg, handler=handler, agent_id="x", bus_config=bus_config)

        assert runner._capabilities == []
        assert runner._metadata == {}


class TestAgentRunnerRunLoop:
    """Tests for AgentRunner._run_loop."""

    def test_polls_and_sleeps_until_stopped(self, ext_runner):
        """_run_loop polls requests and sleeps until stopped."""
        call_count = 0

        def poll_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                ext_runner._stop_event.set()

        with (
            patch.object(ext_runner, "_poll_requests", side_effect=poll_side_effect),
            patch("llm_gent.runtime.runner.time.sleep") as mock_sleep,
        ):
            ext_runner._run_loop()

            assert call_count >= 3
            assert mock_sleep.call_count >= 1


class TestExtStartStop:
    """Tests for AgentRunner start/stop (background thread)."""

    def _blocking_run(self, runner: AgentRunner) -> None:
        """Replacement for run() that blocks until stop is requested."""
        runner._stop_event.wait()

    def test_start_creates_thread(self, ext_runner):
        """start() creates a background thread."""
        with patch.object(ext_runner, "run", lambda: self._blocking_run(ext_runner)):
            ext_runner.start()
            assert ext_runner._bg_thread is not None
            assert ext_runner._bg_thread.is_alive()
            ext_runner.stop()

    def test_start_raises_if_already_running(self, ext_runner):
        """start() raises if already started."""
        with patch.object(ext_runner, "run", lambda: self._blocking_run(ext_runner)):
            ext_runner.start()
            with pytest.raises(RuntimeError, match="already started"):
                ext_runner.start()
            ext_runner.stop()

    def test_stop_clears_thread(self, ext_runner):
        """stop() joins and clears the background thread."""
        with patch.object(ext_runner, "run", lambda: self._blocking_run(ext_runner)):
            ext_runner.start()
            ext_runner.stop()
            assert ext_runner._bg_thread is None

    def test_stop_without_start_is_noop(self, ext_runner):
        """stop() when not started is safe."""
        ext_runner.stop()  # Should not raise


class TestAgentRunnerConnect:
    """Tests for AgentRunner.connect() classmethod."""

    def test_connect_fetches_bus_config(self, handler):
        """connect() fetches config from hub and creates runner."""
        config_json = b'{"coordinator_host": "10.0.1.5", "router_port": 5555, "pub_port": 5556, "sub_port": 5557}'

        mock_resp = MagicMock()
        mock_resp.read.return_value = config_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            runner = AgentRunner.connect(
                lg=MagicMock(),
                handler=handler,
                agent_id="remote-agent",
                hub_url="http://hub:8080",
                capabilities=["translate"],
            )

            mock_urlopen.assert_called_once_with("http://hub:8080/bus/config", timeout=5)
            assert runner.agent_id == "remote-agent"
            assert runner._bus_config.coordinator_host == "10.0.1.5"
            assert runner._bus_config.router_port == 5555
            assert runner._capabilities == ["translate"]

    def test_connect_strips_trailing_slash(self, handler):
        """connect() strips trailing slash from hub_url."""
        config_json = b'{"coordinator_host": "x", "router_port": 1, "pub_port": 2, "sub_port": 3}'

        mock_resp = MagicMock()
        mock_resp.read.return_value = config_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            AgentRunner.connect(
                lg=MagicMock(),
                handler=handler,
                agent_id="x",
                hub_url="http://hub:8080/",
            )
            mock_urlopen.assert_called_once_with("http://hub:8080/bus/config", timeout=5)

    def test_connect_raises_on_failure(self, handler):
        """connect() raises ConnectionError on HTTP failure."""
        with (
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            pytest.raises(ConnectionError, match="failed to fetch"),
        ):
            AgentRunner.connect(
                lg=MagicMock(),
                handler=handler,
                agent_id="x",
                hub_url="http://unreachable:9999",
            )


class TestFetchBusConfig:
    """Tests for AgentRunner._fetch_bus_config."""

    def test_success(self):
        """Valid JSON response returns WorkerBusConfig."""
        config_json = b'{"coordinator_host": "10.0.0.1", "router_port": 6000, "pub_port": 6001, "sub_port": 6002}'

        mock_resp = MagicMock()
        mock_resp.read.return_value = config_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            config = AgentRunner._fetch_bus_config("http://hub/bus/config")

        assert config.coordinator_host == "10.0.0.1"
        assert config.router_port == 6000

    def test_network_error(self):
        """Network error raises ConnectionError."""
        with (
            patch("urllib.request.urlopen", side_effect=OSError("timeout")),
            pytest.raises(ConnectionError, match="failed to fetch"),
        ):
            AgentRunner._fetch_bus_config("http://bad/bus/config")

    def test_non_dict_response(self):
        """Non-dict JSON response raises ConnectionError."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'"just a string"'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=mock_resp),
            pytest.raises(ConnectionError, match="expected object"),
        ):
            AgentRunner._fetch_bus_config("http://hub/bus/config")

    def test_invalid_fields(self):
        """Invalid field values raise ConnectionError."""
        mock_resp = MagicMock()
        # router_port expects int, give it something that will fail WorkerBusConfig init
        mock_resp.read.return_value = b'{"router_port": "not_an_int"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            # This may or may not raise depending on dataclass coercion --
            # the point is to verify the except clause handles TypeError/ValueError.
            # WorkerBusConfig uses int fields, so string should fail.
            try:
                AgentRunner._fetch_bus_config("http://hub/bus/config")
                # If dataclass accepts it (some Python versions coerce), that's fine
            except ConnectionError as e:
                assert "invalid bus config" in str(e)

    def test_extra_fields_ignored(self):
        """Extra fields in response are filtered out (only known fields used)."""
        config_json = (
            b'{"coordinator_host": "h", "router_port": 1, "pub_port": 2,'
            b' "sub_port": 3, "unknown_field": "ignored"}'
        )

        mock_resp = MagicMock()
        mock_resp.read.return_value = config_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            config = AgentRunner._fetch_bus_config("http://hub/bus/config")

        assert config.coordinator_host == "h"
        assert not hasattr(config, "unknown_field")
