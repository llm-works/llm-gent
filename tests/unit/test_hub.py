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
    """Create a hub with mocked bus (zero grace for fast tests).

    Note: start() is intentionally not called so the heartbeat broadcaster
    thread stays dormant.  Tests that assert broadcast call counts depend
    on this.
    """
    config = HubConfig(max_restarts=3, shutdown_grace_secs=0.0)
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

        # Verify AgentJoined broadcast
        from llm_gent.bus.protocol import AgentJoined

        calls = hub._bus.broadcast.call_args_list
        assert len(calls) == 1
        notice = calls[0][0][0]
        assert isinstance(notice, AgentJoined)
        assert notice.agent_id == "worker-1"
        assert notice.capabilities == ["fetch", "search"]

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
        hub._bus.broadcast.reset_mock()

        req = UnregisterRequest(agent_id="worker-1")
        resp = hub._handle_bus_request(req, "id1")

        assert resp.success is True
        assert hub.registry.get("worker-1") is None

        # Verify AgentLeft broadcast
        from llm_gent.bus.protocol import AgentLeft

        calls = hub._bus.broadcast.call_args_list
        assert len(calls) == 1
        notice = calls[0][0][0]
        assert isinstance(notice, AgentLeft)
        assert notice.agent_id == "worker-1"
        assert notice.reason == "voluntary"

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

    def test_heartbeat_response_updates_registry(self, hub):
        """Heartbeat response from broadcast updates registry stats."""
        from llm_gent.bus.protocol import AgentStats, HeartbeatResponse

        hub.registry.register("worker-1")
        resp = HeartbeatResponse(
            id="r1",
            agent_id="worker-1",
            round_id="abc123",
            stats=AgentStats(ticks=10, errors=1, llm_tokens_used=500),
        )
        hub._handle_heartbeat_response(resp)

        entry = hub.registry.get("worker-1")
        assert entry is not None
        assert entry.stats.ticks == 10

    def test_heartbeat_p2p_updates_registry(self, hub):
        """Agent-initiated p2p heartbeat updates registry and returns response."""
        from llm_gent.bus.protocol import AgentStats, HeartbeatRequest, HeartbeatResponse

        hub.registry.register("worker-1")
        req = HeartbeatRequest(
            agent_id="worker-1",
            stats=AgentStats(ticks=5, errors=0),
        )
        resp = hub._handle_heartbeat_p2p(req)

        assert isinstance(resp, HeartbeatResponse)
        assert resp.agent_id == "worker-1"
        entry = hub.registry.get("worker-1")
        assert entry is not None
        assert entry.stats.ticks == 5

    def test_heartbeat_unknown_agent_ignored(self, hub):
        """Heartbeat from unknown agent doesn't crash."""
        from llm_gent.bus.protocol import HeartbeatResponse

        resp = HeartbeatResponse(id="r1", agent_id="ghost")
        hub._handle_heartbeat_response(resp)
        assert hub.registry.count == 0

    def test_non_heartbeat_message_ignored(self, hub):
        """Non-heartbeat messages on heartbeat handler are ignored."""
        hub._handle_heartbeat_response(RegisterRequest(agent_id="x"))

    def test_heartbeat_p2p_via_bus_request(self, hub):
        """HeartbeatRequest on DEALER is dispatched to p2p handler."""
        from llm_gent.bus.protocol import AgentStats, HeartbeatRequest, HeartbeatResponse

        hub.registry.register("worker-1")
        req = HeartbeatRequest(agent_id="worker-1", stats=AgentStats(ticks=7))
        resp = hub._handle_bus_request(req, "worker-1")

        assert isinstance(resp, HeartbeatResponse)
        assert resp.agent_id == "worker-1"


class TestHubMembership:
    """Tests for membership broadcast behavior."""

    def test_stop_agent_no_duplicate_broadcast(self, hub):
        """stop_agent only broadcasts AgentLeft if agent was still registered."""
        from llm_gent.bus.protocol import AgentLeft

        hub.registry.register("worker-1")
        hub._runners["worker-1"] = MagicMock()
        hub._bus.broadcast.reset_mock()

        # Simulate runner's UnregisterRequest already removed the agent
        hub.registry.unregister("worker-1")
        hub._bus.broadcast.reset_mock()

        hub.stop_agent("worker-1")

        # No AgentLeft broadcast since agent was already unregistered
        left_calls = [
            c for c in hub._bus.broadcast.call_args_list if isinstance(c[0][0], AgentLeft)
        ]
        assert len(left_calls) == 0

    def test_cleanup_dead_agents_broadcasts(self, hub):
        """cleanup_dead_agents broadcasts AgentLeft for each dead agent."""
        from datetime import UTC, datetime, timedelta

        from llm_gent.bus.protocol import AgentLeft

        entry = hub.registry.register("dead-agent")
        # Make agent dead by backdating heartbeat
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)
        hub._bus.broadcast.reset_mock()

        removed = hub.cleanup_dead_agents()

        assert removed == ["dead-agent"]
        calls = hub._bus.broadcast.call_args_list
        assert len(calls) == 1
        notice = calls[0][0][0]
        assert isinstance(notice, AgentLeft)
        assert notice.agent_id == "dead-agent"
        assert notice.reason == "dead"


class TestHubShutdown:
    """Tests for hub shutdown sequence."""

    def test_stop_broadcasts_shutdown_notice(self, hub):
        """Hub broadcasts ShutdownNotice before stopping."""
        from llm_gent.bus.protocol import ShutdownNotice

        hub.stop(reason="test shutdown")

        calls = hub._bus.broadcast.call_args_list
        assert len(calls) == 1
        notice = calls[0][0][0]
        assert isinstance(notice, ShutdownNotice)
        assert notice.reason == "test shutdown"
        assert notice.grace_period_secs == hub._config.shutdown_grace_secs

    def test_stop_with_zero_grace_skips_wait(self, hub):
        """Hub with 0 grace period doesn't sleep."""
        hub._config.shutdown_grace_secs = 0.0
        hub.stop()
        hub._bus.broadcast.assert_called_once()
        hub._bus.stop.assert_called_once()


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
