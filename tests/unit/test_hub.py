"""Tests for the swarm hub coordinator."""

from unittest.mock import MagicMock

import pytest

from llm_gent.bus.protocol import (
    ErrorReport,
    ErrorRequest,
    RegisterRequest,
    UnregisterRequest,
)
from llm_gent.bus.transport import WorkerBusConfig
from llm_gent.hub import Hub, HubConfig
from llm_gent.hub.registry import AgentType


pytestmark = pytest.mark.unit


@pytest.fixture
def lg():
    return MagicMock()


@pytest.fixture
def hub(lg):
    """Create a hub with mocked bus."""
    config = HubConfig(max_restarts=3)
    h = Hub(lg, config, bus_config=WorkerBusConfig())
    h._bus = MagicMock()
    return h


class TestHubRequestHandling:
    """Tests for hub bus message handling."""

    def test_register_request(self, hub):
        """Hub registers agent on register request."""
        req = RegisterRequest(
            agent_id="worker-1",
            capabilities=["fetch", "search"],
            metadata={"version": "1.0"},
        )
        resp = hub._handle_bus_request(req, "zmq-identity")

        assert resp.success is True
        assert resp.agent_id == "worker-1"

        entry = hub.registry.get("worker-1")
        assert entry is not None
        assert entry.agent_type == AgentType.EXTERNAL
        assert entry.capabilities == ["fetch", "search"]

    def test_register_duplicate_updates(self, hub):
        """Re-registering updates entry."""
        req1 = RegisterRequest(agent_id="worker-1", capabilities=["v1"])
        hub._handle_bus_request(req1, "id1")

        req2 = RegisterRequest(agent_id="worker-1", capabilities=["v2"])
        hub._handle_bus_request(req2, "id1")

        entry = hub.registry.get("worker-1")
        assert entry is not None
        assert entry.capabilities == ["v2"]

    def test_unregister_request(self, hub):
        """Hub removes agent on unregister request."""
        hub.registry.register("worker-1")
        req = UnregisterRequest(agent_id="worker-1")
        resp = hub._handle_bus_request(req, "id1")

        assert resp.success is True
        assert hub.registry.get("worker-1") is None

    def test_unregister_unknown_agent(self, hub):
        """Unregistering unknown agent still succeeds."""
        req = UnregisterRequest(agent_id="ghost")
        resp = hub._handle_bus_request(req, "id1")
        assert resp.success is True

    def test_error_request(self, hub):
        """Hub acknowledges error escalation."""
        error = ErrorReport(severity="critical", source="llm", message="rate limited")
        req = ErrorRequest(agent_id="worker-1", error=error, escalation_reason="severity")
        resp = hub._handle_bus_request(req, "id1")

        assert resp.success is True
        assert resp.acknowledged is True

    def test_unknown_request_type(self, hub):
        """Hub returns error for unknown request types."""
        req = MagicMock(spec=["id"])
        req.id = "test-id"
        resp = hub._handle_bus_request(req, "id1")
        assert resp.success is False


class TestHubHeartbeat:
    """Tests for heartbeat handling."""

    def test_heartbeat_updates_registry(self, hub):
        """Heartbeat updates registry stats."""
        from llm_gent.bus.protocol import AgentStats, HeartbeatRequest

        hub.registry.register("worker-1")
        hb = HeartbeatRequest(
            agent_id="worker-1",
            stats=AgentStats(ticks=10, errors=1, llm_tokens_used=500),
        )
        hub._handle_heartbeat(hb)

        entry = hub.registry.get("worker-1")
        assert entry is not None
        assert entry.stats.ticks == 10

    def test_heartbeat_unknown_agent_ignored(self, hub):
        """Heartbeat from unknown agent doesn't crash."""
        from llm_gent.bus.protocol import HeartbeatRequest

        hb = HeartbeatRequest(agent_id="ghost")
        hub._handle_heartbeat(hb)
        assert hub.registry.count == 0

    def test_non_heartbeat_message_ignored(self, hub):
        """Non-heartbeat messages on heartbeat handler are ignored."""
        hub._handle_heartbeat(RegisterRequest(agent_id="x"))


class TestHubAskFeedback:
    """Tests for ask/feedback via channels."""

    def test_ask_uses_channel(self, hub):
        """Ask submits request through channel."""
        from llm_gent.bus.protocol import AskResponse

        mock_channel = MagicMock()
        mock_channel.submit.return_value = AskResponse(id="x", response="hello")
        hub._channels["agent-1"] = mock_channel

        result = hub.ask("agent-1", "question")
        assert result == "hello"
        mock_channel.submit.assert_called_once()

    def test_ask_unknown_agent_raises(self, hub):
        """Ask raises KeyError for unknown agent."""
        with pytest.raises(KeyError, match="No channel"):
            hub.ask("ghost", "question")

    def test_feedback_uses_channel(self, hub):
        """Feedback submits request through channel."""
        mock_channel = MagicMock()
        hub._channels["agent-1"] = mock_channel

        hub.feedback("agent-1", "good job")
        mock_channel.submit.assert_called_once()
