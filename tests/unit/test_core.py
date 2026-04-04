"""Tests for Runtime Core."""

from unittest.mock import MagicMock, patch

import pytest
from appinfra.service import State

from llm_gent.core.traits.builtin.llm import LLMConfig
from llm_gent.runtime import AgentRegistry, Core


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
def mock_bus():
    return MagicMock()


@pytest.fixture
def core(mock_logger, registry, llm_config, mock_bus):
    """Create Core with mocked log listener and bus."""
    with patch("llm_gent.runtime.core.LogQueueListener"):
        core = Core(
            lg=mock_logger,
            registry=registry,
            llm_config=llm_config,
            bus=mock_bus,
        )
    return core


class TestCoreInit:
    """Tests for Core initialization."""

    def test_init_creates_log_queue(self, mock_logger, registry, llm_config, mock_bus):
        """Core creates log queue and listener on init."""
        with patch("llm_gent.runtime.core.LogQueueListener") as mock_listener_class:
            Core(lg=mock_logger, registry=registry, llm_config=llm_config, bus=mock_bus)
            mock_listener_class.assert_called_once()
            mock_listener_class.return_value.start.assert_called_once()

    def test_init_with_learn_config(self, mock_logger, registry, llm_config, mock_bus):
        """Core accepts optional LearnConfig."""
        mock_learn_config = MagicMock()
        with patch("llm_gent.runtime.core.LogQueueListener"):
            core = Core(
                lg=mock_logger,
                registry=registry,
                llm_config=llm_config,
                bus=mock_bus,
                learn_config=mock_learn_config,
            )
        assert core._learn_config is mock_learn_config

    def test_init_with_variables(self, mock_logger, registry, llm_config, mock_bus):
        """Core accepts optional variables dict."""
        variables = {"API_KEY": "test-key"}
        with patch("llm_gent.runtime.core.LogQueueListener"):
            core = Core(
                lg=mock_logger,
                registry=registry,
                llm_config=llm_config,
                bus=mock_bus,
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

    def test_start_already_running(self, core, registry):
        """Start raises RuntimeError if agent already running."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

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

    def test_stop_running_agent(self, core, registry, mock_bus):
        """Stop sends shutdown via bus and terminates process."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        handle.process = mock_process

        info = core.stop("test")

        assert info.status == "stopped"
        mock_bus.send_to_agent.assert_called_once()


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

    def test_ask_success(self, core, registry, mock_bus):
        """Ask returns response from agent via bus."""
        from llm_gent.bus.protocol import AskResponse

        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        mock_bus.send_to_agent.return_value = AskResponse(id="test-id", response="Test answer")

        response = core.ask("test", "What is the answer?")
        assert response == "Test answer"

    def test_ask_failure(self, core, registry, mock_bus):
        """Ask raises RuntimeError on bus failure."""
        from llm_gent.bus.transport import BusTimeoutError

        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        mock_bus.send_to_agent.side_effect = BusTimeoutError("timeout")

        with pytest.raises(RuntimeError, match="timeout"):
            core.ask("test", "question")


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

    def test_feedback_success(self, core, registry, mock_bus):
        """Feedback sends message to agent via bus."""
        from llm_gent.bus.protocol import FeedbackResponse

        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        mock_bus.send_to_agent.return_value = FeedbackResponse(id="test-id")

        core.feedback("test", "Good job!")  # Should not raise


class TestCoreShutdown:
    """Tests for Core.shutdown()."""

    def test_shutdown_stops_all_running(self, core, registry, mock_bus):
        """Shutdown stops all running agents."""
        registry.register("agent1", {})
        registry.register("agent2", {})

        handle1 = registry.get("agent1")
        handle1.state = State.RUNNING
        handle1.process = MagicMock()
        handle1.process.is_alive.return_value = False

        mock_bus.send_to_agent.return_value = MagicMock(success=True)

        core.shutdown()

        assert handle1.state == State.STOPPED

    def test_shutdown_handles_stop_errors(self, core, registry, mock_logger):
        """Shutdown handles errors when stopping agents."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        with patch.object(core, "stop", side_effect=RuntimeError("Stop failed")):
            core.shutdown()  # Should not raise

        mock_logger.warning.assert_called()
