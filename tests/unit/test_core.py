"""Tests for Runtime Core."""

from unittest.mock import MagicMock

import pytest
from appinfra.service import State

from llm_gent.bus.transport import WorkerBusConfig
from llm_gent.runtime import AgentRegistry, Core


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def llm_config():
    return MagicMock()


@pytest.fixture
def registry(mock_logger):
    return AgentRegistry(lg=mock_logger)


@pytest.fixture
def mock_bus():
    return MagicMock()


@pytest.fixture
def bus_config():
    return WorkerBusConfig()


@pytest.fixture
def core(mock_logger, registry, llm_config, mock_bus, bus_config):
    """Create Core with mocked bus."""
    return Core(
        lg=mock_logger,
        registry=registry,
        llm_config=llm_config,
        bus=mock_bus,
        bus_config=bus_config,
    )


class TestCoreInit:
    """Tests for Core initialization."""

    def test_registry_property(self, core, registry):
        """Core exposes registry property."""
        assert core.registry is registry

    def test_init_with_learn_config(self, mock_logger, registry, llm_config, mock_bus, bus_config):
        """Core accepts optional LearnConfig."""
        mock_learn_config = MagicMock()
        core = Core(
            lg=mock_logger,
            registry=registry,
            llm_config=llm_config,
            bus=mock_bus,
            bus_config=bus_config,
            learn_config=mock_learn_config,
        )
        assert core._learn_config is mock_learn_config


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

    def test_stop_running_agent(self, core, registry):
        """Stop transitions running agent to stopped."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING
        core._runners["test"] = MagicMock()
        core._channels["test"] = MagicMock()

        info = core.stop("test")
        assert info.status == "stopped"


class TestCoreAsk:
    """Tests for Core.ask() via channel."""

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
        """Ask returns response via channel.submit()."""
        from llm_gent.bus.protocol import AskResponse

        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        mock_channel = MagicMock()
        mock_channel.submit.return_value = AskResponse(id="test-id", response="Test answer")
        core._channels["test"] = mock_channel

        response = core.ask("test", "What is the answer?")
        assert response == "Test answer"
        mock_channel.submit.assert_called_once()

    def test_ask_no_channel(self, core, registry):
        """Ask raises RuntimeError if no channel for agent."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        with pytest.raises(RuntimeError, match="No channel"):
            core.ask("test", "question")


class TestCoreFeedback:
    """Tests for Core.feedback() via channel."""

    def test_feedback_not_found(self, core):
        """Feedback raises KeyError for nonexistent agent."""
        with pytest.raises(KeyError, match="not found"):
            core.feedback("nonexistent", "message")

    def test_feedback_success(self, core, registry):
        """Feedback sends message via channel.submit()."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        mock_channel = MagicMock()
        core._channels["test"] = mock_channel

        core.feedback("test", "Good job!")
        mock_channel.submit.assert_called_once()


class TestCoreShutdown:
    """Tests for Core.shutdown()."""

    def test_shutdown_stops_all_running(self, core, registry):
        """Shutdown stops all running agents."""
        registry.register("agent1", {})
        handle1 = registry.get("agent1")
        handle1.state = State.RUNNING
        core._runners["agent1"] = MagicMock()
        core._channels["agent1"] = MagicMock()

        core.shutdown()
        assert handle1.state == State.STOPPED
