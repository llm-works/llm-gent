# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for runtime Core orchestrator."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict
from appinfra.service import State

from llm_gent.bus.protocol import AskResponse, FeedbackResponse, Response
from llm_gent.runtime.core import Core
from llm_gent.runtime.handle import AgentHandle
from llm_gent.runtime.handler import Answer


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
def runtime():
    """Mock service runtime; spawn() gives a new runner per call."""
    mock_runtime = MagicMock()
    mock_runtime.spawn.side_effect = lambda *_args, **_kwargs: MagicMock()
    return mock_runtime


@pytest.fixture
def core(lg, runtime, registry, bus):
    """Create a Core instance with mocked runtime and bus."""
    from llm_gent.bus.transport import WorkerBusConfig

    return Core(
        lg=lg,
        runtime=runtime,
        registry=registry,
        llm_config=DotDict({"model": "test"}),
        bus=bus,
        bus_config=WorkerBusConfig(),
    )


def _register_agent(registry, name: str = "test-agent", execution: str = "thread") -> AgentHandle:
    """Register a test agent in the registry."""
    config = DotDict({"name": name, "execution": execution})
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
    def test_start_success(self, mock_svc_cls, core, registry, bus, runtime):
        """Successful start spawns the service on a thread, waits until it runs, sets RUNNING."""
        _register_agent(registry)

        info = core.start("test-agent")

        assert info.status == State.RUNNING.value
        bus.create_agent_transport.assert_called_once_with("test-agent")
        runtime.spawn.assert_called_once_with(mock_svc_cls.return_value, None, isolate=False)
        runtime.ready.assert_called_once()

    @patch("llm_gent.runtime.core.AgentService")
    def test_start_process_execution_isolates(self, mock_svc_cls, core, registry, runtime):
        """``execution: process`` spawns the service in a process of its own."""
        _register_agent(registry, execution="process")

        core.start("test-agent")

        assert runtime.spawn.call_args.kwargs == {"isolate": True}

    @patch("llm_gent.runtime.core.AgentService")
    def test_start_failure_sets_failed_state(self, mock_svc_cls, core, registry, runtime):
        """Failed start sets state to FAILED with error message."""
        _register_agent(registry)
        runtime.ready.side_effect = RuntimeError("boom")

        info = core.start("test-agent")

        assert info.status == State.FAILED.value
        assert info.error == "boom"

    @patch("llm_gent.runtime.core.AgentService")
    def test_start_failure_cleans_up_resources(self, mock_svc_cls, core, registry, bus, runtime):
        """Failed start takes the service out of the runtime and cleans up channel and transport."""
        _register_agent(registry)
        runtime.ready.side_effect = RuntimeError("boom")

        core.start("test-agent")

        runtime.stop_service.assert_called_once()
        assert "test-agent" not in core._channels
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
    def test_stop_running_agent(self, mock_svc_cls, core, registry, bus, runtime):
        """Stop stops the agent's service on the runtime, sets STOPPED and cleans up."""
        _register_agent(registry)
        core.start("test-agent")
        runner = core._runners["test-agent"]

        info = core.stop("test-agent")

        assert info.status == State.STOPPED.value
        runtime.stop_service.assert_called_once_with(runner)
        bus.remove_agent_transport.assert_called_with("test-agent")

    @patch("llm_gent.runtime.core.AgentService")
    def test_stop_failure_sets_failed(self, mock_svc_cls, core, registry, bus, runtime):
        """Stop failure sets FAILED state with error."""
        _register_agent(registry)
        core.start("test-agent")
        runtime.stop_service.side_effect = RuntimeError("stop failed")

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
        """Ask returns the response and its session from the channel."""
        handle = _register_agent(registry)
        handle.state = State.RUNNING

        mock_channel = MagicMock()
        mock_channel.submit.return_value = AskResponse(id="x", response="hello", session="s-1")
        core._channels["test-agent"] = mock_channel

        result = core.ask("test-agent", "question")
        assert result == Answer(response="hello", session="s-1")

    def test_ask_passes_the_session(self, core, registry):
        """A session given to ask goes out on the request."""
        handle = _register_agent(registry)
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        mock_channel.submit.return_value = AskResponse(id="x", response="hi", session="s-1")
        core._channels["test-agent"] = mock_channel

        core.ask("test-agent", "question", session="s-1", kwargs={"job": "j-1"})

        sent = mock_channel.submit.call_args[0][0]
        assert (sent.session, sent.kwargs) == ("s-1", {"job": "j-1"})

    def test_ask_answered_with_another_response_raises(self, core, registry):
        """A runner that does not know the ask answers with a plain Response: its error is raised."""
        handle = _register_agent(registry)
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        mock_channel.submit.return_value = Response(
            id="x", success=False, error="unknown request type"
        )
        core._channels["test-agent"] = mock_channel

        with pytest.raises(RuntimeError, match="unknown request type"):
            core.ask("test-agent", "question")


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
    def test_shutdown_stops_running_agents(self, mock_svc_cls, core, registry, runtime):
        """Shutdown stops all running agents."""
        _register_agent(registry, "agent-1")
        _register_agent(registry, "agent-2")
        core.start("agent-1")
        core.start("agent-2")

        core.shutdown()

        assert runtime.stop_service.call_count == 2

    @patch("llm_gent.runtime.core.AgentService")
    def test_shutdown_handles_stop_errors(self, mock_svc_cls, core, registry, lg, runtime):
        """Shutdown continues stopping agents even when one fails."""
        _register_agent(registry, "agent-1")
        _register_agent(registry, "agent-2")
        core.start("agent-1")
        core.start("agent-2")
        error = RuntimeError("error-1")
        runtime.stop_service.side_effect = [error, None]

        core.shutdown()

        # Both should be attempted despite the first failing
        assert runtime.stop_service.call_count == 2
        # Error is logged by stop() internally
        lg.warning.assert_any_call(
            "error stopping agent",
            extra={"agent": "agent-1", "exception": error},
        )

    def test_shutdown_skips_non_running(self, core, registry):
        """Shutdown ignores agents that aren't running."""
        _register_agent(registry, "created")

        core.shutdown()
        # No errors -- non-running agents are skipped
