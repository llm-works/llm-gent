"""Tests for the swarm hub coordinator."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from llm_gent.bus.protocol import (
    ErrorReport,
    ErrorRequest,
    RegisterRequest,
    UnregisterRequest,
)
from llm_gent.bus.registry import AgentType
from llm_gent.hub import Hub, HubConfig


pytestmark = pytest.mark.unit


@pytest.fixture
def lg():
    """Mock logger."""
    return MagicMock()


@pytest.fixture
def hub(lg):
    """Create a hub with mocked bus (no real ZMQ)."""
    config = HubConfig(dead_timeout=90.0, max_restarts=3)
    h = Hub(lg, config)
    # Replace bus with mock to avoid real ZMQ sockets in unit tests
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
            health_url="http://localhost:8080/health",
        )
        resp = hub._handle_request(req, "zmq-identity")

        assert resp.success is True
        assert resp.agent_id == "worker-1"

        info = hub.registry.get("worker-1")
        assert info is not None
        assert info.agent_type == AgentType.EXTERNAL
        assert info.capabilities == ["fetch", "search"]

    def test_register_duplicate_updates(self, hub):
        """Re-registering an agent updates its entry."""
        req1 = RegisterRequest(agent_id="worker-1", capabilities=["v1"])
        hub._handle_request(req1, "id1")

        req2 = RegisterRequest(agent_id="worker-1", capabilities=["v2"])
        hub._handle_request(req2, "id1")

        info = hub.registry.get("worker-1")
        assert info is not None
        assert info.capabilities == ["v2"]

    def test_unregister_request(self, hub):
        """Hub removes agent on unregister request."""
        hub.registry.register("worker-1")
        req = UnregisterRequest(agent_id="worker-1")
        resp = hub._handle_request(req, "id1")

        assert resp.success is True
        assert hub.registry.get("worker-1") is None

    def test_unregister_unknown_agent(self, hub):
        """Unregistering unknown agent still returns success."""
        req = UnregisterRequest(agent_id="ghost")
        resp = hub._handle_request(req, "id1")

        assert resp.success is True

    def test_error_request(self, hub):
        """Hub acknowledges error escalation."""
        error = ErrorReport(
            severity="critical",
            source="llm",
            message="rate limited",
        )
        req = ErrorRequest(
            agent_id="worker-1",
            error=error,
            escalation_reason="severity",
        )
        resp = hub._handle_request(req, "id1")

        assert resp.success is True
        assert resp.acknowledged is True

    def test_unknown_request_type(self, hub):
        """Hub returns error for unknown request types."""
        req = MagicMock(spec=["id"])
        req.id = "test-id"
        # Make isinstance checks fail
        resp = hub._handle_request(req, "id1")

        assert resp.success is False
        assert "unknown" in resp.error


class TestHubHeartbeat:
    """Tests for heartbeat handling via topic subscription."""

    def test_heartbeat_updates_registry(self, hub):
        """Heartbeat from known agent updates registry stats."""
        from llm_gent.bus.protocol import AgentStats, HeartbeatRequest

        hub.registry.register("worker-1")

        hb = HeartbeatRequest(
            agent_id="worker-1",
            stats=AgentStats(ticks=10, errors=1, llm_tokens_used=500),
        )
        hub._handle_heartbeat_topic(hb)

        info = hub.registry.get("worker-1")
        assert info is not None
        assert info.stats.ticks == 10
        assert info.stats.errors == 1

    def test_heartbeat_unknown_agent_ignored(self, hub, lg):
        """Heartbeat from unknown agent is silently ignored."""
        from llm_gent.bus.protocol import HeartbeatRequest

        hb = HeartbeatRequest(agent_id="ghost")
        hub._handle_heartbeat_topic(hb)

        # Should log debug but not crash
        assert hub.registry.count == 0

    def test_non_heartbeat_message_ignored(self, hub):
        """Non-heartbeat messages on heartbeat topic are ignored."""
        from llm_gent.bus.protocol import RegisterRequest

        msg = RegisterRequest(agent_id="worker-1")
        hub._handle_heartbeat_topic(msg)  # should not raise


class TestHubInjectedAgents:
    """Tests for injected agent management."""

    def test_register_injected(self, hub):
        """register_injected adds agent as INJECTED type."""
        hub.register_injected("my-agent", capabilities=["compute"])

        info = hub.registry.get("my-agent")
        assert info is not None
        assert info.agent_type == AgentType.INJECTED
        assert info.capabilities == ["compute"]


class TestHubHealthCheck:
    """Tests for health monitoring logic."""

    def test_dead_external_agent_removed(self, hub):
        """Dead external agent is unregistered during health check."""
        hub.registry.register("ext-agent", AgentType.EXTERNAL)
        info = hub.registry.get("ext-agent")
        assert info is not None
        info.last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)

        hub._check_health()

        assert hub.registry.get("ext-agent") is None

    def test_dead_injected_agent_triggers_restart(self, hub):
        """Dead injected agent increments restart count."""
        hub.register_injected("inj-agent")
        info = hub.registry.get("inj-agent")
        assert info is not None
        info.last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)

        hub._check_health()

        # Agent should still be in registry (restart pending, not removed)
        info = hub.registry.get("inj-agent")
        assert info is not None
        assert info.restart_count == 1

    def test_injected_agent_removed_after_max_restarts(self, hub):
        """Injected agent removed after exceeding max restarts."""
        hub.register_injected("inj-agent")
        info = hub.registry.get("inj-agent")
        assert info is not None

        # Simulate max_restarts + 1 health check cycles
        for _ in range(hub._config.max_restarts + 1):
            info = hub.registry.get("inj-agent")
            if info is None:
                break
            info.last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)
            hub._check_health()

        assert hub.registry.get("inj-agent") is None

    def test_healthy_agents_untouched(self, hub):
        """Healthy agents are not affected by health check."""
        hub.register_injected("healthy-agent")
        hub.registry.register("ext-healthy", AgentType.EXTERNAL)

        hub._check_health()

        assert hub.registry.get("healthy-agent") is not None
        assert hub.registry.get("ext-healthy") is not None
