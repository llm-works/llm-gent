"""Tests for Agent Runner."""

from unittest.mock import MagicMock

import pytest

from llm_gent.core.agent import ExecutionResult
from llm_gent.runtime import AgentRunner
from llm_gent.runtime.messages import AgentMessage, AgentResponse, MessageType


pytestmark = pytest.mark.unit


class TestAgentRunner:
    """Tests for AgentRunner."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_channel(self):
        channel = MagicMock()
        channel.recv.return_value = None
        return channel

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with required methods."""
        agent = MagicMock()
        agent.name = "test-agent"
        agent.cycle_count = 0
        agent.run_once.return_value = ExecutionResult(success=True, content="Done", iterations=1)
        agent.ask.return_value = "Agent response"
        agent.get_recent_results.return_value = []
        return agent

    @pytest.fixture
    def runner(self, mock_logger, mock_agent, mock_channel):
        return AgentRunner(
            lg=mock_logger,
            agent=mock_agent,
            channel=mock_channel,
        )

    def test_runner_init(self, runner, mock_agent):
        """Runner initializes with correct attributes."""
        assert runner._agent.name == "test-agent"
        assert runner._running is False
        assert runner._schedule_interval is None

    def test_runner_init_with_schedule(self, mock_logger, mock_agent, mock_channel):
        """Runner accepts schedule interval."""
        runner = AgentRunner(
            lg=mock_logger,
            agent=mock_agent,
            channel=mock_channel,
            schedule_interval=300,
        )

        assert runner._schedule_interval == 300.0

    def test_calculate_timeout_no_schedule(self, runner):
        """Calculate timeout returns 60s when no schedule."""
        timeout = runner._calculate_timeout()
        assert timeout == 60.0

    def test_calculate_timeout_continuous(self, mock_logger, mock_agent, mock_channel):
        """Calculate timeout returns 0s for continuous mode."""
        runner = AgentRunner(
            lg=mock_logger,
            agent=mock_agent,
            channel=mock_channel,
            schedule_interval=0,
        )

        timeout = runner._calculate_timeout()
        assert timeout == 0.0

    def test_calculate_timeout_with_schedule(self, mock_logger, mock_agent, mock_channel):
        """Calculate timeout uses Ticker for scheduled mode."""
        runner = AgentRunner(
            lg=mock_logger,
            agent=mock_agent,
            channel=mock_channel,
            schedule_interval=30,
        )

        # Ticker should provide time until next tick
        timeout = runner._calculate_timeout()
        # Should be close to 30s on first call (initial=True)
        assert 0.0 <= timeout <= 30.0

    def test_should_run_cycle_no_schedule(self, runner):
        """Should not run cycle when no schedule."""
        assert runner._should_run_cycle() is False

    def test_should_run_cycle_continuous(self, mock_logger, mock_agent, mock_channel):
        """Should always run cycle in continuous mode."""
        runner = AgentRunner(
            lg=mock_logger,
            agent=mock_agent,
            channel=mock_channel,
            schedule_interval=0,
        )

        assert runner._should_run_cycle() is True

    def test_should_run_cycle_with_schedule(self, mock_logger, mock_agent, mock_channel):
        """Should run cycle based on Ticker in scheduled mode."""
        runner = AgentRunner(
            lg=mock_logger,
            agent=mock_agent,
            channel=mock_channel,
            schedule_interval=1,
        )

        # Ticker with initial=True should allow first tick immediately
        assert runner._should_run_cycle() is True
        # Second call should return False (not ready yet)
        assert runner._should_run_cycle() is False

    def test_handle_shutdown(self, runner, mock_channel):
        """Handle shutdown message stops the runner."""
        runner._running = True
        msg = AgentMessage(type=MessageType.SHUTDOWN)

        runner._handle_message(msg)

        assert runner._running is False

    def test_handle_ask(self, runner, mock_channel, mock_agent):
        """Handle ask calls agent.ask and returns response."""
        msg = AgentMessage(
            type=MessageType.ASK, payload={"question": "test question"}, id="test-id"
        )

        runner._handle_ask(msg)

        mock_agent.ask.assert_called_once_with("test question")
        mock_channel.send.assert_called_once()
        call = mock_channel.send.call_args[0][0]
        assert isinstance(call, AgentResponse)
        assert call.success is True
        assert call.payload["response"] == "Agent response"

    def test_handle_feedback(self, runner, mock_channel, mock_agent):
        """Handle feedback calls agent.record_feedback."""
        msg = AgentMessage(
            type=MessageType.FEEDBACK, payload={"message": "Good job!"}, id="test-id"
        )

        runner._handle_feedback(msg)

        mock_agent.record_feedback.assert_called_once_with("Good job!")
        call = mock_channel.send.call_args[0][0]
        assert call.success is True

    def test_handle_get_insights(self, runner, mock_channel, mock_agent):
        """Handle get_insights returns agent results."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = "Test content"
        mock_result.iterations = 1
        mock_agent.get_recent_results.return_value = [mock_result]

        msg = AgentMessage(type=MessageType.GET_INSIGHTS, payload={"limit": 5}, id="test-id")

        runner._handle_get_insights(msg)

        mock_agent.get_recent_results.assert_called_once_with(5)
        call = mock_channel.send.call_args[0][0]
        assert call.success is True
        assert len(call.payload["insights"]) == 1

    def test_handle_run_cycle(self, runner, mock_channel, mock_agent):
        """Handle run_cycle calls _run_cycle."""
        msg = AgentMessage(type=MessageType.RUN_CYCLE)

        runner._handle_message(msg)

        mock_agent.run_once.assert_called_once()

    def test_run_cycle_sends_complete(self, runner, mock_channel, mock_agent):
        """Run cycle sends CYCLE_COMPLETE on success."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.iterations = 3
        mock_agent.run_once.return_value = mock_result

        runner._run_cycle()

        call = mock_channel.send.call_args[0][0]
        assert call.type == MessageType.CYCLE_COMPLETE
        assert call.payload["success"] is True
        assert call.payload["iterations"] == 3

    def test_run_cycle_sends_error_on_failure(self, runner, mock_channel, mock_agent):
        """Run cycle sends CYCLE_ERROR on exception."""
        mock_agent.run_once.side_effect = RuntimeError("Cycle failed")

        runner._run_cycle()

        call = mock_channel.send.call_args[0][0]
        assert call.type == MessageType.CYCLE_ERROR
        assert "Cycle failed" in call.payload["error"]

    def test_send_error_response_maps_types(self, runner, mock_channel):
        """Send error response maps request types to response types."""
        msg = AgentMessage(type=MessageType.ASK, id="test-id")

        runner._send_error_response(msg, "Error message")

        call = mock_channel.send.call_args[0][0]
        assert call.type == MessageType.ASK_RESPONSE
        assert call.request_id == "test-id"
        assert call.success is False
        assert call.error == "Error message"

    def test_cleanup_stops_agent(self, runner, mock_channel, mock_agent):
        """Cleanup stops agent and closes channel."""
        runner._cleanup()

        mock_agent.stop.assert_called_once()
        mock_channel.close.assert_called_once()

    def test_cleanup_handles_agent_stop_error(self, runner, mock_channel, mock_logger, mock_agent):
        """Cleanup handles errors from agent.stop()."""
        mock_agent.stop.side_effect = RuntimeError("Stop failed")

        runner._cleanup()  # Should not raise

        mock_channel.close.assert_called_once()

    def test_handle_unknown_message_type(self, runner, mock_channel, mock_logger):
        """Handle unknown message type logs warning."""
        msg = AgentMessage(type="unknown_type")

        runner._handle_message(msg)

        mock_logger.warning.assert_called()


class TestAgentRunnerIntegration:
    """Integration tests for AgentRunner.run()."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with required methods."""
        agent = MagicMock()
        agent.name = "test"
        agent.cycle_count = 0
        return agent

    def test_run_sends_started_and_stops(self, mock_logger, mock_agent):
        """Run sends STARTED message and handles shutdown via framework loop."""
        mock_channel = MagicMock()

        # Force AgentRunner to use framework loop (not agent.run())
        mock_agent.run = None

        # Simulate shutdown after startup
        def recv_side_effect(timeout=None):
            return AgentMessage(type=MessageType.SHUTDOWN)

        mock_channel.recv.side_effect = recv_side_effect

        runner = AgentRunner(
            lg=mock_logger,
            agent=mock_agent,
            channel=mock_channel,
        )

        runner.run()

        # Verify shutdown was processed through framework loop
        mock_channel.recv.assert_called()
        mock_agent.stop.assert_called_once()

        # Check that STARTED message was sent
        calls = mock_channel.send.call_args_list
        assert any(call[0][0].type == MessageType.STARTED for call in calls)
