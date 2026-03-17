"""Tests for Agent Registry."""

from unittest.mock import MagicMock

import pytest
from appinfra.service import State

from llm_gent.runtime import AgentHandle, AgentInfo, AgentRegistry


pytestmark = pytest.mark.unit


class TestAgentHandle:
    """Tests for AgentHandle dataclass."""

    def test_default_values(self):
        """AgentHandle has correct defaults."""
        handle = AgentHandle(name="test", config={})

        assert handle.name == "test"
        assert handle.config == {}
        assert handle.state == State.CREATED
        assert handle.process is None
        assert handle.channel is None
        assert handle.cycle_count == 0
        assert handle.last_run is None
        assert handle.error is None
        assert handle.schedule_interval is None

    def test_to_info_dict(self):
        """to_info_dict creates serializable dict."""
        handle = AgentHandle(
            name="test-agent",
            config={},
            state=State.RUNNING,
            cycle_count=5,
            schedule_interval=60,
        )

        info = handle.to_info_dict()

        assert info["name"] == "test-agent"
        assert info["status"] == "running"
        assert info["cycle_count"] == 5
        assert info["schedule_interval"] == 60
        assert info["last_run"] is None
        assert info["error"] is None


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_from_handle(self):
        """AgentInfo.from_handle creates snapshot."""
        handle = AgentHandle(
            name="test",
            config={},
            state=State.RUNNING,
            cycle_count=10,
        )

        info = AgentInfo.from_handle(handle)

        assert info.name == "test"
        assert info.status == "running"
        assert info.cycle_count == 10

    def test_immutable_snapshot(self):
        """AgentInfo is a snapshot, not a reference."""
        handle = AgentHandle(name="test", config={}, state=State.RUNNING)
        info = AgentInfo.from_handle(handle)

        # Modify handle
        handle.state = State.STOPPED
        handle.cycle_count = 100

        # Info should not change
        assert info.status == "running"
        assert info.cycle_count == 0


class TestAgentRegistry:
    """Tests for AgentRegistry.

    Note: Registry is now a pure data structure. Lifecycle methods (start, stop,
    ask, feedback, etc.) are on Core, tested separately.
    """

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def registry(self, mock_logger):
        return AgentRegistry(lg=mock_logger)

    def test_register_agent(self, registry):
        """Register creates IDLE handle."""
        config = {"name": "test", "directive": {"prompt": "Test"}}
        handle = registry.register("test-agent", config)

        assert handle.name == "test-agent"
        assert handle.state == State.CREATED
        assert handle.config == config

    def test_register_duplicate_raises(self, registry):
        """Registering same name twice raises."""
        registry.register("test", {})

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", {})

    def test_register_extracts_schedule(self, registry):
        """Register extracts schedule interval from config."""
        config = {"schedule": {"interval": 300}}
        handle = registry.register("scheduled", config)

        assert handle.schedule_interval == 300

    def test_get_existing(self, registry):
        """get returns handle for existing agent."""
        registry.register("test", {})

        handle = registry.get("test")
        assert handle is not None
        assert handle.name == "test"

    def test_get_nonexistent(self, registry):
        """get returns None for nonexistent agent."""
        handle = registry.get("nonexistent")
        assert handle is None

    def test_list_empty(self, registry):
        """list_agents returns empty list when no agents."""
        result = registry.list_agents()
        assert result == []

    def test_list_agents(self, registry):
        """list_agents returns info for all agents."""
        registry.register("agent1", {})
        registry.register("agent2", {})

        result = registry.list_agents()

        assert len(result) == 2
        names = {info.name for info in result}
        assert names == {"agent1", "agent2"}

    def test_unregister_existing(self, registry):
        """unregister removes agent from registry."""
        registry.register("test", {})

        registry.unregister("test")

        assert registry.get("test") is None

    def test_unregister_nonexistent(self, registry):
        """unregister raises KeyError for nonexistent agent."""
        with pytest.raises(KeyError, match="not found"):
            registry.unregister("nonexistent")

    def test_unregister_running_raises(self, registry):
        """unregister raises RuntimeError for running agent."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.RUNNING

        with pytest.raises(RuntimeError, match="Agent must be CREATED or STOPPED"):
            registry.unregister("test")

    def test_unregister_starting_raises(self, registry):
        """unregister raises RuntimeError for agent in STARTING state."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.STARTING

        with pytest.raises(RuntimeError, match="Agent must be CREATED or STOPPED"):
            registry.unregister("test")

    def test_unregister_stopping_raises(self, registry):
        """unregister raises RuntimeError for agent in STOPPING state."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.STOPPING

        with pytest.raises(RuntimeError, match="Agent must be CREATED or STOPPED"):
            registry.unregister("test")

    def test_unregister_error_raises(self, registry):
        """unregister raises RuntimeError for agent in ERROR state."""
        registry.register("test", {})
        handle = registry.get("test")
        handle.state = State.FAILED

        with pytest.raises(RuntimeError, match="Agent must be CREATED or STOPPED"):
            registry.unregister("test")

    def test_handles_returns_all(self, registry):
        """handles returns list of all handles."""
        registry.register("agent1", {})
        registry.register("agent2", {})

        handles = registry.handles()

        assert len(handles) == 2
        names = {h.name for h in handles}
        assert names == {"agent1", "agent2"}
