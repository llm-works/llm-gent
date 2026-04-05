"""Tests for runtime Core orchestrator."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict
from appinfra.service import State

from llm_gent.bus.protocol import AskResponse, FeedbackResponse
from llm_gent.runtime.core import Core
from llm_gent.runtime.handle import AgentHandle


pytestmark = pytest.mark.unit


@pytest.fixture
def lg():
    """Mock logger."""
    return MagicMock()


@pytest.fixture
def registry(lg):
    """Create a real AgentRegistry."""
    from llm_gent.runtime.registry import AgentRegistry

    return AgentRegistry(lg)


@pytest.fixture
def bus():
    """Mock coordinator bus."""
    mock_bus = MagicMock()
    mock_bus.create_agent_transport.return_value = MagicMock()
    return mock_bus


@pytest.fixture
def core(lg, registry, bus):
    """Create a Core instance with mocked bus."""
    from llm_gent.bus.transport import WorkerBusConfig

    return Core(
        lg=lg,
        registry=registry,
        llm_config=DotDict({"model": "test"}),
        bus=bus,
        bus_config=WorkerBusConfig(),
    )


def _register_agent(registry, name: str = "test-agent") -> AgentHandle:
    """Register a test agent in the registry."""
    config = DotDict({"name": name, "execution": "thread"})
    return registry.register(name, config)


# =============================================================================
# Start tests
# =============================================================================


class TestCoreStart:
    """Tests for Core.start()."""

    def test_start_not_found(self, core):
        """Start raises KeyError for unknown agent."""
        with pytest.raises(KeyError, match="Agent not found"):
            core.start("ghost")

    def test_start_already_active(self, core, registry):
        """Start raises RuntimeError if agent is already running."""
        handle = _register_agent(registry)
        handle.state = State.RUNNING

        with pytest.raises(RuntimeError, match="already active"):
            core.start("test-agent")

    @patch("llm_gent.runtime.core.AgentService")
    @patch("llm_gent.runtime.core.ThreadRunner")
    def test_start_success(self, mock_runner_cls, mock_svc_cls, core, registry, bus):
        """Successful start sets state to RUNNING."""
        _register_agent(registry)

        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        info = core.start("test-agent")

        assert info.status == State.RUNNING.value
        bus.create_agent_transport.assert_called_once_with("test-agent")
        mock_runner.start.assert_called_once()

    @patch("llm_gent.runtime.core.AgentService")
    @patch("llm_gent.runtime.core.ThreadRunner")
    def test_start_failure_sets_failed_state(self, mock_runner_cls, mock_svc_cls, core, registry):
        """Failed start sets state to FAILED with error message."""
        _register_agent(registry)

        mock_runner = MagicMock()
        mock_runner.start.side_effect = RuntimeError("boom")
        mock_runner_cls.return_value = mock_runner

        info = core.start("test-agent")

        assert info.status == State.FAILED.value
        assert info.error == "boom"

    @patch("llm_gent.runtime.core.AgentService")
    @patch("llm_gent.runtime.core.ThreadRunner")
    def test_start_failure_cleans_up_resources(
        self, mock_runner_cls, mock_svc_cls, core, registry, bus
    ):
        """Failed start cleans up channel and transport."""
        _register_agent(registry)

        mock_runner = MagicMock()
        mock_runner.start.side_effect = RuntimeError("boom")
        mock_runner_cls.return_value = mock_runner

        core.start("test-agent")

        # Channel should be removed
        assert "test-agent" not in core._channels
        # Transport should be cleaned up
        bus.remove_agent_transport.assert_called_once_with("test-agent")


# =============================================================================
# Stop tests
# =============================================================================


class TestCoreStop:
    """Tests for Core.stop()."""

    def test_stop_not_found(self, core):
        """Stop raises KeyError for unknown agent."""
        with pytest.raises(KeyError, match="Agent not found"):
            core.stop("ghost")

    def test_stop_not_running_is_noop(self, core, registry):
        """Stop on non-running agent returns current state."""
        _register_agent(registry)

        info = core.stop("test-agent")
        assert info.status == State.CREATED.value

    @patch("llm_gent.runtime.core.AgentService")
    @patch("llm_gent.runtime.core.ThreadRunner")
    def test_stop_running_agent(self, mock_runner_cls, mock_svc_cls, core, registry, bus):
        """Stop sets state to STOPPED and cleans up runner."""
        _register_agent(registry)
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        core.start("test-agent")

        info = core.stop("test-agent")
        assert info.status == State.STOPPED.value
        mock_runner.stop.assert_called_once()
        bus.remove_agent_transport.assert_called_with("test-agent")

    @patch("llm_gent.runtime.core.AgentService")
    @patch("llm_gent.runtime.core.ThreadRunner")
    def test_stop_failure_sets_failed(self, mock_runner_cls, mock_svc_cls, core, registry, bus):
        """Stop failure sets FAILED state with error."""
        _register_agent(registry)
        mock_runner = MagicMock()
        mock_runner.stop.side_effect = RuntimeError("stop failed")
        mock_runner_cls.return_value = mock_runner

        core.start("test-agent")

        info = core.stop("test-agent")
        assert info.status == State.FAILED.value
        assert info.error == "stop failed"


# =============================================================================
# Ask / Feedback tests
# =============================================================================


class TestCoreAsk:
    """Tests for Core.ask()."""

    def test_ask_not_found(self, core):
        """Ask raises KeyError for unknown agent."""
        with pytest.raises(KeyError, match="Agent not found"):
            core.ask("ghost", "question")

    def test_ask_not_running(self, core, registry):
        """Ask raises RuntimeError if agent not running."""
        _register_agent(registry)

        with pytest.raises(RuntimeError, match="not running"):
            core.ask("test-agent", "question")

    def test_ask_no_channel(self, core, registry):
        """Ask raises RuntimeError if no channel for agent."""
        handle = _register_agent(registry)
        handle.state = State.RUNNING

        with pytest.raises(RuntimeError, match="No channel"):
            core.ask("test-agent", "question")

    def test_ask_success(self, core, registry):
        """Ask returns response string from channel."""
        handle = _register_agent(registry)
        handle.state = State.RUNNING

        mock_channel = MagicMock()
        mock_channel.submit.return_value = AskResponse(id="x", response="hello")
        core._channels["test-agent"] = mock_channel

        result = core.ask("test-agent", "question")
        assert result == "hello"


class TestCoreFeedback:
    """Tests for Core.feedback()."""

    def test_feedback_not_found(self, core):
        """Feedback raises KeyError for unknown agent."""
        with pytest.raises(KeyError, match="Agent not found"):
            core.feedback("ghost", "message")

    def test_feedback_not_running(self, core, registry):
        """Feedback raises RuntimeError if agent not running."""
        _register_agent(registry)

        with pytest.raises(RuntimeError, match="not running"):
            core.feedback("test-agent", "message")

    def test_feedback_success(self, core, registry):
        """Feedback submits through channel."""
        handle = _register_agent(registry)
        handle.state = State.RUNNING

        mock_channel = MagicMock()
        mock_channel.submit.return_value = FeedbackResponse(id="x")
        core._channels["test-agent"] = mock_channel

        core.feedback("test-agent", "good job")
        mock_channel.submit.assert_called_once()


# =============================================================================
# Shutdown tests
# =============================================================================


class TestCoreShutdown:
    """Tests for Core.shutdown()."""

    @patch("llm_gent.runtime.core.AgentService")
    @patch("llm_gent.runtime.core.ThreadRunner")
    def test_shutdown_stops_running_agents(self, mock_runner_cls, mock_svc_cls, core, registry):
        """Shutdown stops all running agents."""
        _register_agent(registry, "agent-1")
        _register_agent(registry, "agent-2")

        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        core.start("agent-1")
        core.start("agent-2")

        core.shutdown()

        # Both runners should be stopped
        assert mock_runner.stop.call_count == 2

    @patch("llm_gent.runtime.core.AgentService")
    @patch("llm_gent.runtime.core.ThreadRunner")
    def test_shutdown_handles_stop_errors(self, mock_runner_cls, mock_svc_cls, core, registry, lg):
        """Shutdown continues stopping agents even when one fails."""
        _register_agent(registry, "agent-1")
        _register_agent(registry, "agent-2")

        mock_runner_1 = MagicMock()
        mock_runner_1.stop.side_effect = RuntimeError("error-1")
        mock_runner_2 = MagicMock()
        mock_runner_cls.side_effect = [mock_runner_1, mock_runner_2]

        core.start("agent-1")
        core.start("agent-2")

        core.shutdown()

        # Both should be attempted despite agent-1 failing
        mock_runner_1.stop.assert_called_once()
        mock_runner_2.stop.assert_called_once()
        # Error is logged by stop() internally
        lg.warning.assert_any_call(
            "error stopping agent",
            extra={"agent": "agent-1", "exception": mock_runner_1.stop.side_effect},
        )

    def test_shutdown_skips_non_running(self, core, registry):
        """Shutdown ignores agents that aren't running."""
        _register_agent(registry, "created")

        core.shutdown()
        # No errors -- non-running agents are skipped
