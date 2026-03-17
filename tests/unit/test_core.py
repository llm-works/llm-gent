"""Tests for Runtime Core."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra.service import State

from llm_gent.core.traits.builtin.llm import LLMConfig
from llm_gent.runtime import AgentRegistry, Core
from llm_gent.runtime.messages import AgentResponse, MessageType


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def llm_config():
    return LLMConfig(base_url="http://localhost:8000/v1")


@pytest.fixture
def registry(mock_logger):
    return AgentRegistry(lg=mock_logger)


@pytest.fixture
def core(mock_logger, registry, llm_config):
    """Create Core with mocked log listener."""
    with patch("llm_gent.runtime.core.LogQueueListener"):
        core = Core(
            lg=mock_logger,
            registry=registry,
            llm_config=llm_config,
        )
    return core


class TestCoreInit:
    """Tests for Core initialization."""

    def test_init_creates_log_queue(self, mock_logger, registry, llm_config):
        """Core creates log queue and listener on init."""
        with patch("llm_gent.runtime.core.LogQueueListener") as mock_listener_class:
            Core(lg=mock_logger, registry=registry, llm_config=llm_config)
            mock_listener_class.assert_called_once()
            mock_listener_class.return_value.start.assert_called_once()

    def test_init_with_learn_config(self, mock_logger, registry, llm_config):
        """Core accepts optional LearnConfig."""
        mock_learn_config = MagicMock()
        with patch("llm_gent.runtime.core.LogQueueListener"):
            core = Core(
                lg=mock_logger,
                registry=registry,
                llm_config=llm_config,
                learn_config=mock_learn_config,
            )
        assert core._learn_config is mock_learn_config

    def test_init_with_variables(self, mock_logger, registry, llm_config):
        """Core accepts optional variables dict."""
        variables = {"API_KEY": "test-key"}
        with patch("llm_gent.runtime.core.LogQueueListener"):
            core = Core(
                lg=mock_logger,
                registry=registry,
                llm_config=llm_config,
                variables=variables,
            )
        assert core._variables == variables

    def test_registry_property(self, core, registry):
        """Core exposes registry property."""
        assert core.registry is registry


class TestCoreStart:
    """Tests for Core.start()."""

    def test_start_not_found(self, core):
        """Start raises KeyError for nonexistent agent."""
        with pytest.raises(KeyError, match="not found"):
            core.start("nonexistent")

    def test_start_transitions_to_running(self, core, registry):
        """Start transitions state through STARTING to RUNNING."""
        registry.register("test", {"name": "test"})

        with patch.object(core, "_spawn_process"):
            info = core.start("test")

        assert info.status == "running"

    def test_start_clears_previous_error(self, core, registry):
        """Start clears any previous error state."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.error = "Previous error"

        with patch.object(core, "_spawn_process"):
            core.start("test")

        assert handle.error is None

    def test_start_error_transitions_to_error(self, core, registry, mock_logger):
        """Start error transitions to ERROR state."""
        registry.register("test", {"name": "test"})

        with patch.object(core, "_spawn_process", side_effect=RuntimeError("spawn failed")):
            info = core.start("test")

        assert info.status == "failed"
        assert "spawn failed" in info.error

    def test_start_error_cleans_up_resources(self, core, registry):
        """Start cleans up IPC resources on spawn failure."""
        import multiprocessing as mp

        registry.register("test", {"name": "test"})
        handle = registry.get("test")

        # Simulate partial spawn: channel created, then error
        mock_channel = MagicMock()
        mock_process = MagicMock(spec=mp.Process)
        mock_process.is_alive.return_value = True

        def spawn_with_partial_setup(h):
            h.channel = mock_channel
            h.process = mock_process
            raise RuntimeError("spawn failed after channel created")

        with patch.object(core, "_spawn_process", side_effect=spawn_with_partial_setup):
            core.start("test")

        # Verify cleanup occurred
        mock_channel.close.assert_called_once()
        mock_process.terminate.assert_called()
        assert handle.channel is None
        assert handle.process is None

    def test_start_rejects_duplicate_start(self, core, registry):
        """Start raises RuntimeError if agent is already active."""
        registry.register("test", {"name": "test"})
        handle = registry.get("test")

        # Set agent to RUNNING state
        handle.state = State.RUNNING

        with pytest.raises(RuntimeError, match="already active"):
            core.start("test")

    def test_start_rejects_starting_agent(self, core, registry):
        """Start raises RuntimeError if agent is in STARTING state."""
        registry.register("test", {"name": "test"})
        handle = registry.get("test")

        # Set agent to STARTING state
        handle.state = State.STARTING

        with pytest.raises(RuntimeError, match="already active"):
            core.start("test")


class TestCoreStop:
    """Tests for Core.stop()."""

    def test_stop_not_found(self, core):
        """Stop raises KeyError for nonexistent agent."""
        with pytest.raises(KeyError, match="not found"):
            core.stop("nonexistent")

    def test_stop_not_running_is_noop(self, core, registry):
        """Stop on non-running agent is no-op."""
        registry.register("test", {})

        info = core.stop("test")

        assert info.status == "created"

    def test_stop_running_agent(self, core, registry):
        """Stop terminates running agent."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        handle.channel = mock_channel
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        handle.process = mock_process

        info = core.stop("test")

        assert info.status == "stopped"
        mock_channel.send.assert_called()  # Shutdown message sent


class TestCoreAsk:
    """Tests for Core.ask()."""

    def test_ask_not_found(self, core):
        """Ask raises KeyError for nonexistent agent."""
        with pytest.raises(KeyError, match="not found"):
            core.ask("nonexistent", "question")

    def test_ask_not_running(self, core, registry):
        """Ask raises RuntimeError if agent not running."""
        registry.register("test", {})

        with pytest.raises(RuntimeError, match="not running"):
            core.ask("test", "question")

    def test_ask_success(self, core, registry):
        """Ask returns response from agent."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        mock_channel.submit.return_value = AgentResponse(
            type=MessageType.ASK_RESPONSE,
            success=True,
            payload={"response": "Test answer"},
        )
        handle.channel = mock_channel

        response = core.ask("test", "What is the answer?")

        assert response == "Test answer"

    def test_ask_failure(self, core, registry):
        """Ask raises RuntimeError on failure response."""
        from appinfra.service import ChannelError

        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        mock_channel.submit.side_effect = ChannelError("Request failed: Agent error")
        handle.channel = mock_channel

        with pytest.raises(RuntimeError, match="Agent error"):
            core.ask("test", "What is the answer?")


class TestCoreFeedback:
    """Tests for Core.feedback()."""

    def test_feedback_not_found(self, core):
        """Feedback raises KeyError for nonexistent agent."""
        with pytest.raises(KeyError, match="not found"):
            core.feedback("nonexistent", "message")

    def test_feedback_not_running(self, core, registry):
        """Feedback raises RuntimeError if agent not running."""
        registry.register("test", {})

        with pytest.raises(RuntimeError, match="not running"):
            core.feedback("test", "message")

    def test_feedback_success(self, core, registry):
        """Feedback sends message to agent."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        mock_channel.submit.return_value = AgentResponse(
            type=MessageType.FEEDBACK_RESPONSE,
            success=True,
        )
        handle.channel = mock_channel

        core.feedback("test", "Good job!")  # Should not raise

    def test_feedback_failure(self, core, registry):
        """Feedback raises RuntimeError on failure."""
        from appinfra.service import ChannelError

        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        mock_channel.submit.side_effect = ChannelError("Request failed: Feedback error")
        handle.channel = mock_channel

        with pytest.raises(RuntimeError, match="Feedback error"):
            core.feedback("test", "message")


class TestCoreGetInsights:
    """Tests for Core.get_insights()."""

    def test_get_insights_not_found(self, core):
        """get_insights raises KeyError for nonexistent agent."""
        with pytest.raises(KeyError, match="not found"):
            core.get_insights("nonexistent")

    def test_get_insights_not_running(self, core, registry):
        """get_insights raises RuntimeError if agent not running."""
        registry.register("test", {})

        with pytest.raises(RuntimeError, match="not running"):
            core.get_insights("test")

    def test_get_insights_success(self, core, registry):
        """get_insights returns insights from agent."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        insights_data = [{"success": True, "content": "Insight 1", "parsed": None, "iterations": 1}]
        mock_channel.submit.return_value = AgentResponse(
            type=MessageType.INSIGHTS_RESPONSE,
            success=True,
            payload={"insights": insights_data},
        )
        handle.channel = mock_channel

        insights = core.get_insights("test", limit=5)

        assert insights == insights_data

    def test_get_insights_failure(self, core, registry):
        """get_insights raises RuntimeError on failure."""
        from appinfra.service import ChannelError

        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_channel = MagicMock()
        mock_channel.submit.side_effect = ChannelError("Request failed: Insights error")
        handle.channel = mock_channel

        with pytest.raises(RuntimeError, match="Insights error"):
            core.get_insights("test")


class TestCoreShutdown:
    """Tests for Core.shutdown()."""

    def test_shutdown_stops_all_running(self, core, registry, mock_logger):
        """Shutdown stops all running agents."""
        registry.register("agent1", {})
        registry.register("agent2", {})

        # Set agent1 to running
        handle1 = registry.get("agent1")
        handle1.state = State.RUNNING
        handle1.channel = MagicMock()
        handle1.process = MagicMock()
        handle1.process.is_alive.return_value = False

        core.shutdown()

        # agent1 should have been stopped
        assert handle1.state == State.STOPPED

    def test_shutdown_handles_stop_errors(self, core, registry, mock_logger):
        """Shutdown handles errors when stopping agents."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        # Mock stop to raise an exception
        with patch.object(core, "stop", side_effect=RuntimeError("Stop failed")):
            core.shutdown()  # Should not raise

        # Verify warning was logged
        mock_logger.warning.assert_called()


class TestCoreTerminateProcess:
    """Tests for Core._terminate_process()."""

    def test_terminate_sends_shutdown_message(self, core, registry):
        """Terminate sends shutdown message to agent."""
        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        mock_channel = MagicMock()
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        handle.channel = mock_channel
        handle.process = mock_process

        core._terminate_process(handle)

        mock_channel.send.assert_called_once()

    def test_terminate_handles_channel_error(self, core, registry):
        """Terminate handles channel send errors gracefully."""
        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        mock_channel = MagicMock()
        mock_channel.send.side_effect = RuntimeError("Channel error")
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        handle.channel = mock_channel
        handle.process = mock_process

        core._terminate_process(handle)  # Should not raise

    def test_terminate_kills_stubborn_process(self, core, registry):
        """Terminate kills process that won't stop."""
        import multiprocessing as mp

        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        mock_channel = MagicMock()
        mock_process = MagicMock(spec=mp.Process)
        # Process stays alive through terminate
        mock_process.is_alive.side_effect = [True, True, True]
        handle.channel = mock_channel
        handle.process = mock_process

        core._terminate_process(handle)

        mock_process.terminate.assert_called()
        mock_process.kill.assert_called()

    def test_terminate_clears_handle(self, core, registry):
        """Terminate clears process and channel from handle."""
        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        handle.channel = MagicMock()
        handle.process = MagicMock()
        handle.process.is_alive.return_value = False

        core._terminate_process(handle)

        assert handle.process is None
        assert handle.channel is None


class TestCoreCleanupFailedStart:
    """Tests for Core._cleanup_failed_start()."""

    def test_cleanup_with_channel_only(self, core, registry):
        """Cleanup handles case where only channel was created."""
        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        mock_channel = MagicMock()
        handle.channel = mock_channel
        handle.process = None

        core._cleanup_failed_start(handle)

        mock_channel.close.assert_called_once()
        assert handle.channel is None

    def test_cleanup_with_process_only(self, core, registry):
        """Cleanup handles case where only process was created."""
        import multiprocessing as mp

        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        handle.channel = None
        mock_process = MagicMock(spec=mp.Process)
        mock_process.is_alive.return_value = False
        handle.process = mock_process

        core._cleanup_failed_start(handle)

        mock_process.terminate.assert_called_once()
        assert handle.process is None

    def test_cleanup_handles_channel_close_error(self, core, registry):
        """Cleanup suppresses channel close errors."""
        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        mock_channel = MagicMock()
        mock_channel.close.side_effect = RuntimeError("Close failed")
        handle.channel = mock_channel

        core._cleanup_failed_start(handle)  # Should not raise

        assert handle.channel is None

    def test_cleanup_kills_stubborn_process(self, core, registry):
        """Cleanup kills process that won't terminate."""
        import multiprocessing as mp

        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(name="test", config={})
        mock_process = MagicMock(spec=mp.Process)
        mock_process.is_alive.return_value = True
        handle.process = mock_process

        core._cleanup_failed_start(handle)

        mock_process.terminate.assert_called()
        mock_process.kill.assert_called()
        assert handle.process is None


class TestCoreBuildRunnerConfig:
    """Tests for Core._build_runner_config()."""

    def test_build_runner_config(self, core, registry):
        """Build runner config adds name to config dict."""
        from llm_gent.runtime import AgentHandle

        handle = AgentHandle(
            name="test-agent",
            config={"directive": {"prompt": "Test"}},
        )

        config = core._build_runner_config(handle)

        assert config["name"] == "test-agent"
        assert config["directive"] == {"prompt": "Test"}
