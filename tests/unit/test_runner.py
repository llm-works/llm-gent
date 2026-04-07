"""Tests for AgentRunner."""

from unittest.mock import MagicMock

import pytest

from llm_gent.bus.protocol import (
    AgentJoined,
    AgentLeft,
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HeartbeatRequest,
    ShutdownNotice,
    ShutdownRequest,
    ShutdownResponse,
)
from llm_gent.bus.transport import WorkerBusConfig
from llm_gent.runtime.runner import AgentRunner


pytestmark = pytest.mark.unit


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
def runner(mock_agent, bus_config):
    """Create runner (not connected to bus)."""
    lg = MagicMock()
    return AgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)


class TestAgentRunner:
    """Tests for AgentRunner initialization."""

    def test_runner_init(self, mock_agent, bus_config):
        """Runner initializes with bus config."""
        lg = MagicMock()
        runner = AgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)

        assert runner._agent is mock_agent
        assert runner._bus_config is bus_config
        assert runner._ticker is None

    def test_runner_init_with_schedule(self, mock_agent, bus_config):
        """Runner creates Ticker when schedule_interval provided."""
        lg = MagicMock()
        runner = AgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=60.0)

        assert runner._ticker is not None


class TestRequestHandling:
    """Tests for _handle_request (processes requests from channel)."""

    def test_handle_ask(self, runner, mock_agent):
        """Ask request calls agent.ask() and returns response."""
        mock_agent.ask.return_value = "Test answer"
        req = AskRequest(question="What?")

        resp = runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is True
        assert resp.response == "Test answer"
        mock_agent.ask.assert_called_once_with("What?")

    def test_handle_ask_error(self, runner, mock_agent):
        """Ask request returns error response on exception."""
        mock_agent.ask.side_effect = RuntimeError("LLM failed")
        req = AskRequest(question="What?")

        resp = runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is False
        assert "LLM failed" in resp.error

    def test_handle_feedback(self, runner, mock_agent):
        """Feedback request calls agent.record_feedback()."""
        req = FeedbackRequest(message="Good job!")

        resp = runner._handle_request(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is True
        mock_agent.record_feedback.assert_called_once_with("Good job!")

    def test_handle_shutdown(self, runner):
        """Shutdown request sets running to False."""
        runner._running = True
        req = ShutdownRequest()

        resp = runner._handle_request(req)

        assert isinstance(resp, ShutdownResponse)
        assert resp.success is True
        assert runner._running is False

    def test_handle_unknown_request(self, runner):
        """Unknown request type returns error response."""
        req = MagicMock(spec=["id"])
        req.id = "test-id"

        resp = runner._handle_request(req)

        assert resp.success is False
        assert "unknown" in resp.error


class TestBroadcastHandling:
    """Tests for _handle_broadcast (processes hub broadcasts)."""

    def test_handle_heartbeat_broadcast(self, runner):
        """HeartbeatRequest triggers heartbeat response."""
        runner._bus = MagicMock()
        req = HeartbeatRequest(round_id="r1")

        runner._handle_broadcast(req)

        runner._bus.publish_heartbeat.assert_called_once()

    def test_handle_shutdown_notice(self, runner):
        """ShutdownNotice sets stop event."""
        notice = ShutdownNotice(reason="test", grace_period_secs=1.0)

        runner._handle_broadcast(notice)

        assert runner._running is False

    def test_handle_agent_joined(self, runner):
        """AgentJoined logs and doesn't crash."""
        msg = AgentJoined(agent_id="peer-1", capabilities=["search"])

        runner._handle_broadcast(msg)

        runner._lg.info.assert_called()
        call_args = runner._lg.info.call_args
        assert "joined" in call_args[0][0]

    def test_handle_agent_left(self, runner):
        """AgentLeft logs and doesn't crash."""
        msg = AgentLeft(agent_id="peer-1", reason="voluntary")

        runner._handle_broadcast(msg)

        runner._lg.info.assert_called()
        call_args = runner._lg.info.call_args
        assert "left" in call_args[0][0]

    def test_handle_unknown_broadcast(self, runner):
        """Unknown broadcast type is silently ignored."""
        msg = MagicMock(spec=[])

        runner._handle_broadcast(msg)  # Should not raise


class TestScheduling:
    """Tests for scheduled execution."""

    def test_should_run_cycle_continuous(self, mock_agent, bus_config):
        """Continuous mode always runs cycle."""
        lg = MagicMock()
        runner = AgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=0)

        assert runner._should_run_cycle() is True

    def test_should_run_cycle_message_only(self, runner):
        """Message-only mode never runs cycle."""
        assert runner._should_run_cycle() is False

    def test_run_cycle_calls_run_once(self, runner, mock_agent):
        """Cycle calls agent.run_once()."""
        runner._run_cycle()
        mock_agent.run_once.assert_called_once()

    def test_run_cycle_handles_error(self, runner, mock_agent):
        """Cycle handles agent.run_once() exceptions."""
        mock_agent.run_once.side_effect = RuntimeError("cycle failed")
        runner._run_cycle()  # Should not raise
