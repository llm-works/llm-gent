"""Tests for the bus module: protocol and registry (unit tests only)."""

import threading
import time

import pytest

from llm_gent.bus.protocol import (
    MESSAGE_REGISTRY,
    AgentStats,
    Envelope,
    ErrorReport,
    ErrorRequest,
    HeartbeatRequest,
    RegisterRequest,
    RegisterResponse,
    Response,
    UnregisterRequest,
)
from llm_gent.bus.registry import (
    AgentHealth,
    AgentInfo,
    AgentRegistry,
    AgentType,
)


pytestmark = pytest.mark.unit


# =============================================================================
# Protocol tests
# =============================================================================


class TestEnvelope:
    """Tests for Envelope serialization and deserialization."""

    def test_roundtrip_bytes(self):
        """Envelope serializes to bytes and back."""
        env = Envelope(msg_type="test", payload={"key": "value"})
        data = env.to_bytes()
        restored = Envelope.from_bytes(data)

        assert restored.msg_type == "test"
        assert restored.payload == {"key": "value"}
        assert restored.version == 1

    def test_source_and_target(self):
        """Envelope preserves source and target fields."""
        env = Envelope(
            msg_type="test",
            source="agent-a",
            target="agent-b",
            payload={},
        )
        restored = Envelope.from_bytes(env.to_bytes())

        assert restored.source == "agent-a"
        assert restored.target == "agent-b"

    def test_unwrap_known_type(self):
        """Unwrap deserializes payload into correct message class."""
        req = RegisterRequest(agent_id="test-agent", capabilities=["fetch"])
        env = req.to_envelope()
        msg = env.unwrap(MESSAGE_REGISTRY)

        assert isinstance(msg, RegisterRequest)
        assert msg.agent_id == "test-agent"
        assert msg.capabilities == ["fetch"]

    def test_unwrap_unknown_type_raises(self):
        """Unwrap raises ValueError for unknown message type."""
        env = Envelope(msg_type="nonexistent", payload={})

        with pytest.raises(ValueError, match="unknown message type"):
            env.unwrap(MESSAGE_REGISTRY)


class TestMessages:
    """Tests for v1 protocol messages."""

    def test_request_has_id(self):
        """Requests get auto-generated IDs."""
        req = RegisterRequest(agent_id="test")
        assert req.id
        assert len(req.id) == 12

    def test_request_ids_unique(self):
        """Each request gets a unique ID."""
        ids = {RegisterRequest(agent_id="test").id for _ in range(100)}
        assert len(ids) == 100

    def test_register_request_roundtrip(self):
        """RegisterRequest survives envelope roundtrip."""
        req = RegisterRequest(
            agent_id="worker-1",
            capabilities=["search", "fetch"],
            metadata={"version": "1.0"},
            health_url="http://localhost:8080/health",
        )
        env = req.to_envelope()
        restored = env.unwrap(MESSAGE_REGISTRY)

        assert isinstance(restored, RegisterRequest)
        assert restored.agent_id == "worker-1"
        assert restored.capabilities == ["search", "fetch"]
        assert restored.metadata == {"version": "1.0"}
        assert restored.health_url == "http://localhost:8080/health"
        assert restored.id == req.id

    def test_heartbeat_request_with_stats(self):
        """HeartbeatRequest carries agent statistics."""
        stats = AgentStats(ticks=42, errors=1, llm_tokens_used=1500)
        req = HeartbeatRequest(agent_id="worker-1", stats=stats)
        env = req.to_envelope()
        restored = env.unwrap(MESSAGE_REGISTRY)

        assert isinstance(restored, HeartbeatRequest)
        assert restored.stats.ticks == 42
        assert restored.stats.errors == 1
        assert restored.stats.llm_tokens_used == 1500

    def test_error_request_roundtrip(self):
        """ErrorRequest with ErrorReport survives roundtrip."""
        error = ErrorReport(
            severity="critical",
            source="llm_client",
            message="rate limited",
            details={"retry_after": 60},
        )
        req = ErrorRequest(
            agent_id="worker-1",
            error=error,
            escalation_reason="severity",
        )
        env = req.to_envelope()
        restored = env.unwrap(MESSAGE_REGISTRY)

        assert isinstance(restored, ErrorRequest)
        assert restored.error.severity == "critical"
        assert restored.error.details == {"retry_after": 60}

    def test_response_success(self):
        """Response defaults to success=True."""
        resp = RegisterResponse(id="abc", agent_id="worker-1")
        assert resp.success is True
        assert resp.error is None

    def test_response_error(self):
        """Response can carry error information."""
        resp = Response(id="abc", success=False, error="connection refused")
        assert resp.success is False
        assert resp.error == "connection refused"

    def test_message_type_preserved_in_envelope(self):
        """Each message class sets correct message_type in envelope."""
        cases = [
            (RegisterRequest(agent_id="x"), "register_request"),
            (HeartbeatRequest(agent_id="x"), "heartbeat_request"),
            (UnregisterRequest(agent_id="x"), "unregister_request"),
            (ErrorRequest(agent_id="x", error=ErrorReport()), "error_request"),
        ]
        for msg, expected_type in cases:
            env = msg.to_envelope()
            assert env.msg_type == expected_type

    def test_all_registry_types_listed(self):
        """All v1 message types are in the registry."""
        expected = {
            "register_request",
            "register_response",
            "heartbeat_request",
            "heartbeat_response",
            "unregister_request",
            "unregister_response",
            "error_request",
            "error_response",
            "ask_request",
            "ask_response",
            "feedback_request",
            "feedback_response",
            "shutdown_request",
            "shutdown_response",
            "shutdown_notice",
            "agent_joined",
            "agent_left",
            "relay_request",
            "relay_response",
        }
        assert set(MESSAGE_REGISTRY.keys()) == expected

    def test_message_tier_classification(self):
        """Message tiers are correctly assigned per FIX-style classification."""
        from llm_gent.bus.protocol import (
            AgentJoined,
            AgentLeft,
            AskRequest,
            FeedbackResponse,
            HeartbeatRequest,
            MessageTier,
            RelayRequest,
            RelayResponse,
            ShutdownNotice,
            ShutdownRequest,
        )

        # System tier: infrastructure messages
        assert HeartbeatRequest.tier == MessageTier.SYSTEM
        assert ShutdownNotice.tier == MessageTier.SYSTEM
        assert AgentJoined.tier == MessageTier.SYSTEM
        assert AgentLeft.tier == MessageTier.SYSTEM

        # Application tier: business messages
        assert AskRequest.tier == MessageTier.APPLICATION
        assert FeedbackResponse.tier == MessageTier.APPLICATION
        assert ShutdownRequest.tier == MessageTier.APPLICATION

        # Custom tier: agent-defined protocols
        assert RelayRequest.tier == MessageTier.CUSTOM
        assert RelayResponse.tier == MessageTier.CUSTOM


# =============================================================================
# Registry tests
# =============================================================================


class TestAgentInfo:
    """Tests for AgentInfo health calculations."""

    def test_newly_created_is_alive(self):
        """Freshly created AgentInfo is alive."""
        info = AgentInfo(id="test", agent_type=AgentType.EXTERNAL)
        assert info.is_alive()
        assert info.health == AgentHealth.HEALTHY

    def test_stale_agent_is_unhealthy(self):
        """Agent with old heartbeat is unhealthy."""
        from datetime import UTC, datetime, timedelta

        info = AgentInfo(id="test", agent_type=AgentType.EXTERNAL)
        info.last_heartbeat = datetime.now(UTC) - timedelta(seconds=100)
        assert not info.is_alive()
        assert info.health == AgentHealth.UNHEALTHY

    def test_very_stale_agent_is_dead(self):
        """Agent with very old heartbeat is dead."""
        from datetime import UTC, datetime, timedelta

        info = AgentInfo(id="test", agent_type=AgentType.EXTERNAL)
        info.last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)
        assert info.health == AgentHealth.DEAD


class TestAgentRegistry:
    """Tests for AgentRegistry operations."""

    @pytest.fixture
    def registry(self):
        """Create an empty registry."""
        return AgentRegistry(dead_timeout=90.0)

    def test_register_and_get(self, registry):
        """Register an agent and retrieve it."""
        info = registry.register("agent-1", AgentType.INJECTED, capabilities=["fetch"])

        assert info.id == "agent-1"
        assert info.agent_type == AgentType.INJECTED
        assert info.capabilities == ["fetch"]
        assert registry.count == 1

        retrieved = registry.get("agent-1")
        assert retrieved is not None
        assert retrieved.id == "agent-1"

    def test_register_preserves_restart_count(self, registry):
        """Re-registering preserves restart count from previous entry."""
        registry.register("agent-1")
        registry.increment_restart("agent-1")
        registry.increment_restart("agent-1")

        registry.register("agent-1")
        info = registry.get("agent-1")
        assert info is not None
        assert info.restart_count == 2

    def test_unregister(self, registry):
        """Unregister removes agent from registry."""
        registry.register("agent-1")
        assert registry.unregister("agent-1") is True
        assert registry.get("agent-1") is None
        assert registry.count == 0

    def test_unregister_nonexistent(self, registry):
        """Unregistering unknown agent returns False."""
        assert registry.unregister("ghost") is False

    def test_heartbeat_updates_timestamp(self, registry):
        """Heartbeat updates last_heartbeat."""
        registry.register("agent-1")
        info_before = registry.get("agent-1")
        assert info_before is not None
        ts_before = info_before.last_heartbeat

        time.sleep(0.01)
        registry.heartbeat("agent-1")

        info_after = registry.get("agent-1")
        assert info_after is not None
        assert info_after.last_heartbeat > ts_before

    def test_heartbeat_updates_stats(self, registry):
        """Heartbeat can update agent stats."""
        registry.register("agent-1")
        stats = AgentStats(ticks=10, errors=1)
        registry.heartbeat("agent-1", stats)

        info = registry.get("agent-1")
        assert info is not None
        assert info.stats.ticks == 10
        assert info.stats.errors == 1

    def test_heartbeat_unknown_agent(self, registry):
        """Heartbeat for unknown agent returns False."""
        assert registry.heartbeat("ghost") is False

    def test_list_agents(self, registry):
        """List returns all registered agents."""
        registry.register("a")
        registry.register("b")
        registry.register("c")

        agents = registry.list_agents()
        assert len(agents) == 3
        ids = {a.id for a in agents}
        assert ids == {"a", "b", "c"}

    def test_get_healthy_and_dead(self, registry):
        """Healthy/dead filtering works correctly."""
        from datetime import UTC, datetime, timedelta

        registry.register("alive")
        registry.register("dead")

        # Mutate internal state directly (get() returns a copy).
        registry._agents["dead"].last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)

        healthy = registry.get_healthy()
        dead = registry.get_dead()
        assert len(healthy) == 1
        assert healthy[0].id == "alive"
        assert len(dead) == 1
        assert dead[0].id == "dead"

    def test_cleanup_dead(self, registry):
        """Cleanup removes dead agents and returns their IDs."""
        from datetime import UTC, datetime, timedelta

        registry.register("alive")
        registry.register("dead-1")
        registry.register("dead-2")

        # Mutate internal state directly (get() returns a copy).
        for agent_id in ("dead-1", "dead-2"):
            registry._agents[agent_id].last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)

        removed = registry.cleanup_dead()
        assert set(removed) == {"dead-1", "dead-2"}
        assert registry.count == 1

    def test_increment_restart(self, registry):
        """Restart count increments correctly."""
        registry.register("agent-1")
        assert registry.increment_restart("agent-1") == 1
        assert registry.increment_restart("agent-1") == 2
        assert registry.increment_restart("agent-1") == 3

    def test_increment_restart_unknown(self, registry):
        """Restart increment for unknown agent returns 0."""
        assert registry.increment_restart("ghost") == 0

    def test_healthy_count(self, registry):
        """Healthy count reflects current state."""
        from datetime import UTC, datetime, timedelta

        registry.register("a")
        registry.register("b")
        registry.register("c")

        # Mutate internal state directly (get() returns a copy).
        registry._agents["c"].last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)

        assert registry.healthy_count == 2

    def test_thread_safety(self, registry):
        """Concurrent operations don't corrupt state."""
        errors: list[Exception] = []

        def register_many(prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    registry.register(f"{prefix}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many, args=(f"t{t}", 50)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert registry.count == 200
