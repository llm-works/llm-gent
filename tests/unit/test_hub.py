# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the swarm hub coordinator."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.bus.protocol import (
    AgentJoined,
    AgentLeft,
    AskResponse,
    ErrorReport,
    ErrorRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterRequest,
    RelayRequest,
    RelayResponse,
    ShutdownNotice,
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
        """Re-registering updates entry without extra AgentJoined broadcast."""
        from llm_gent.bus.protocol import AgentJoined

        req1 = RegisterRequest(agent_id="worker-1", capabilities=["v1"])
        hub._handle_bus_request(req1, "id1")

        req2 = RegisterRequest(agent_id="worker-1", capabilities=["v2"])
        hub._handle_bus_request(req2, "id1")

        entry = hub.registry.get("worker-1")
        assert entry is not None
        assert entry.capabilities == ["v2"]

        # Only one AgentJoined (from first registration, not re-register)
        joined_calls = [
            c for c in hub._bus.broadcast.call_args_list if isinstance(c[0][0], AgentJoined)
        ]
        assert len(joined_calls) == 1

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
        """Unregistering unknown agent succeeds without phantom broadcast."""
        from llm_gent.bus.protocol import AgentLeft

        hub._bus.broadcast.reset_mock()
        req = UnregisterRequest(agent_id="ghost")
        resp = hub._handle_bus_request(req, "id1")
        assert resp.success is True

        # No AgentLeft broadcast for unknown agent
        left_calls = [
            c for c in hub._bus.broadcast.call_args_list if isinstance(c[0][0], AgentLeft)
        ]
        assert len(left_calls) == 0

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
        from llm_gent.bus.protocol import AgentStats

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
        from llm_gent.bus.protocol import AgentStats, HeartbeatRequest

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

        resp = HeartbeatResponse(id="r1", agent_id="ghost")
        hub._handle_heartbeat_response(resp)
        assert hub.registry.count == 0

    def test_non_heartbeat_message_ignored(self, hub):
        """Non-heartbeat messages on heartbeat handler are ignored."""
        hub._handle_heartbeat_response(RegisterRequest(agent_id="x"))

    def test_heartbeat_p2p_via_bus_request(self, hub):
        """HeartbeatRequest on DEALER is dispatched to p2p handler."""
        from llm_gent.bus.protocol import AgentStats, HeartbeatRequest

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

    def test_ask_failure_raises_runtime_error(self, hub):
        """Ask raises RuntimeError when agent returns failure response."""
        mock_channel = MagicMock()
        mock_channel.submit.return_value = AskResponse(id="x", success=False, error="oops")
        hub._channels["agent-1"] = mock_channel

        with pytest.raises(RuntimeError, match="ask failed: oops"):
            hub.ask("agent-1", "question")

    def test_ask_non_ask_response_returns_empty(self, hub):
        """Ask returns empty string when response is not AskResponse."""
        mock_channel = MagicMock()
        mock_channel.submit.return_value = MagicMock(spec=[])  # not AskResponse
        hub._channels["agent-1"] = mock_channel

        result = hub.ask("agent-1", "question")
        assert result == ""

    def test_feedback_failure_raises_runtime_error(self, hub):
        """Feedback raises RuntimeError when agent returns failure."""
        mock_channel = MagicMock()
        resp = MagicMock()
        resp.success = False
        resp.error = "bad feedback"
        mock_channel.submit.return_value = resp
        hub._channels["agent-1"] = mock_channel

        with pytest.raises(RuntimeError, match="feedback failed"):
            hub.feedback("agent-1", "msg")

    def test_feedback_unknown_agent_raises(self, hub):
        """Feedback raises KeyError for unknown agent."""
        with pytest.raises(KeyError, match="No channel"):
            hub.feedback("ghost", "msg")


class TestHubBroadcastPublish:
    """Tests for broadcast and publish passthrough."""

    def test_broadcast_delegates_to_bus(self, hub):
        """broadcast() passes message to bus.broadcast."""
        msg = ShutdownNotice(reason="test")
        hub.broadcast(msg)
        hub._bus.broadcast.assert_called_once_with(msg)

    def test_publish_delegates_to_bus(self, hub):
        """publish() passes topic and message to bus.publish."""
        msg = ShutdownNotice(reason="test")
        hub.publish("my-topic", msg)
        hub._bus.publish.assert_called_once_with("my-topic", msg)


class TestHubBusProperty:
    """Tests for Hub properties."""

    def test_bus_property_returns_bus(self, hub):
        """bus property returns the internal bus object."""
        assert hub.bus is hub._bus


class TestHubStart:
    """Tests for Hub.start()."""

    def test_start_configures_bus(self, hub):
        """start() calls bus.start and wires up handlers."""
        hub.start()

        hub._bus.start.assert_called_once()
        hub._bus.on_request.assert_called_once()
        hub._bus.set_route_validator.assert_called_once()
        hub._bus.register_async_request.assert_called_once_with("relay_request")
        hub._bus.subscribe.assert_called_once_with("heartbeat", hub._handle_heartbeat_response)

        # Cleanup: stop heartbeat broadcaster thread
        hub._stop_heartbeat_broadcaster()


class TestHubStopExtended:
    """Extended stop tests covering grace period and agent error handling."""

    def test_stop_with_grace_period(self, hub):
        """stop() sleeps when grace > 0."""
        hub._config.shutdown_grace_secs = 0.01  # small but >0
        with patch("llm_gent.hub.hub.time.sleep") as mock_sleep:
            hub.stop()
            mock_sleep.assert_called_once_with(0.01)

    def test_stop_catches_agent_stop_errors(self, hub, lg):
        """stop() logs warning but continues when stop_agent raises."""
        hub._runners["bad-agent"] = MagicMock()
        hub._runners["bad-agent"].stop.side_effect = RuntimeError("boom")
        hub.registry.register("bad-agent")

        hub.stop()

        # Should still have called bus.stop despite the error
        hub._bus.stop.assert_called_once()
        # Warning logged for the failing agent
        lg.warning.assert_called()


class TestBroadcastErrorPaths:
    """Tests for _broadcast_shutdown and _broadcast_membership error paths."""

    def test_broadcast_shutdown_catches_bus_error(self, hub, lg):
        """_broadcast_shutdown logs warning when bus.broadcast raises."""
        hub._bus.broadcast.side_effect = RuntimeError("bus down")
        hub._broadcast_shutdown("reason")

        lg.warning.assert_called()
        assert "failed to broadcast shutdown notice" in lg.warning.call_args[0][0]

    def test_broadcast_membership_catches_bus_error(self, hub, lg):
        """_broadcast_membership logs warning when bus.broadcast raises."""
        hub._bus.broadcast.side_effect = RuntimeError("bus down")
        notice = AgentJoined(agent_id="x", agent_type="external", capabilities=[])
        hub._broadcast_membership(notice)

        lg.warning.assert_called()
        assert "failed to broadcast membership notice" in lg.warning.call_args[0][0]


class TestHubRelay:
    """Tests for _handle_relay and _relay_error."""

    def test_relay_target_not_found(self, hub):
        """Relay returns error when target agent has no channel."""
        req = RelayRequest(
            from_agent="a", to_agent="missing", inner_type="ask_request", inner_payload={}
        )
        resp = hub._handle_relay(req)

        assert resp.success is False
        assert "not found" in resp.error

    def test_relay_unknown_inner_type(self, hub):
        """Relay returns error for unknown inner message type."""
        hub._channels["target"] = MagicMock()
        req = RelayRequest(
            from_agent="a", to_agent="target", inner_type="nonexistent_type", inner_payload={}
        )
        resp = hub._handle_relay(req)

        assert resp.success is False
        assert "Unknown inner message type" in resp.error

    def test_relay_success(self, hub):
        """Relay forwards inner message to target channel and returns response."""
        mock_channel = MagicMock()
        inner_resp = MagicMock()
        inner_resp.message_type = "ask_response"
        inner_resp.model_dump.return_value = {"response": "hi"}
        mock_channel.submit.return_value = inner_resp
        hub._channels["target"] = mock_channel

        req = RelayRequest(
            from_agent="sender",
            to_agent="target",
            inner_type="ask_request",
            inner_payload={"question": "hello"},
        )
        resp = hub._handle_relay(req)

        assert resp.success is True
        assert resp.from_agent == "target"
        assert resp.inner_type == "ask_response"
        assert resp.inner_payload == {"response": "hi"}

    def test_relay_submit_exception(self, hub):
        """Relay catches exception from channel.submit and returns error."""
        mock_channel = MagicMock()
        mock_channel.submit.side_effect = TimeoutError("timed out")
        hub._channels["target"] = mock_channel

        req = RelayRequest(
            from_agent="sender",
            to_agent="target",
            inner_type="ask_request",
            inner_payload={"question": "hello"},
        )
        resp = hub._handle_relay(req)

        assert resp.success is False
        assert "timed out" in resp.error

    def test_relay_dispatched_from_bus_request(self, hub):
        """RelayRequest is dispatched through _handle_bus_request."""
        hub._channels["target"] = MagicMock()
        hub._channels["target"].submit.return_value = MagicMock(
            message_type="ask_response",
            model_dump=MagicMock(return_value={}),
        )
        req = RelayRequest(
            from_agent="a", to_agent="target", inner_type="ask_request", inner_payload={}
        )
        resp = hub._handle_bus_request(req, "sender-id")
        assert isinstance(resp, RelayResponse)

    def test_relay_error_creates_error_response(self, hub):
        """_relay_error creates a RelayResponse with failure fields."""
        req = RelayRequest(from_agent="a", to_agent="b", inner_type="ask_request", inner_payload={})
        resp = hub._relay_error(req, "some error")

        assert resp.success is False
        assert resp.error == "some error"
        assert resp.from_agent == "b"
        assert resp.inner_type == "error"


class TestHeartbeatResponseOldProtocol:
    """Tests for heartbeat response handler edge cases."""

    def test_heartbeat_request_on_pubsub_logs_warning(self, hub, lg):
        """HeartbeatRequest on pub/sub logs old protocol warning."""
        req = HeartbeatRequest(agent_id="old-agent")
        hub._handle_heartbeat_response(req)

        lg.warning.assert_called()
        msg = lg.warning.call_args[0][0]
        assert "old protocol" in msg


class TestHubStartAgent:
    """Tests for start_agent lifecycle."""

    def test_start_agent_registers_and_starts(self, hub):
        """start_agent registers agent, creates channel, starts runner."""
        from appinfra import DotDict

        with (
            patch.object(hub, "_create_channel") as mock_ch,
            patch.object(hub, "_create_runner") as mock_runner_fn,
        ):
            mock_runner = MagicMock()
            mock_runner_fn.return_value = mock_runner

            entry = hub.start_agent("test-agent", DotDict({}))

            assert entry is not None
            assert entry.agent_type == AgentType.INJECTED
            mock_ch.assert_called_once_with("test-agent")
            mock_runner_fn.assert_called_once()
            mock_runner.start.assert_called_once()
            assert hub._runners["test-agent"] is mock_runner

    def test_start_agent_cleanup_on_runner_failure(self, hub):
        """start_agent cleans up on runner creation failure."""
        from appinfra import DotDict

        with (
            patch.object(hub, "_create_channel"),
            patch.object(hub, "_create_runner", side_effect=RuntimeError("no runner")),
            patch.object(hub, "_cleanup_agent_resources") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="no runner"):
                hub.start_agent("bad-agent", DotDict({}))

            mock_cleanup.assert_called_once_with("bad-agent")
            # Agent should be unregistered after failure
            assert hub.registry.get("bad-agent") is None

    def test_start_agent_cleanup_on_channel_failure(self, hub):
        """start_agent cleans up on channel creation failure."""
        from appinfra import DotDict

        with (
            patch.object(hub, "_create_channel", side_effect=RuntimeError("zmq fail")),
            patch.object(hub, "_cleanup_agent_resources") as mock_cleanup,
        ):
            with pytest.raises(RuntimeError, match="zmq fail"):
                hub.start_agent("bad-agent", DotDict({}))

            mock_cleanup.assert_called_once_with("bad-agent")
            assert hub.registry.get("bad-agent") is None


class TestHubCreateRunner:
    """Tests for _create_runner execution modes."""

    def test_create_runner_thread_mode(self, hub):
        """Thread execution mode creates ThreadRunner."""
        from appinfra import DotDict

        with (
            patch("llm_gent.hub.hub.AgentService"),
            patch("llm_gent.hub.hub.ThreadRunner") as mock_tr,
            patch("llm_gent.hub.hub.ProcessRunner"),
        ):
            config = DotDict({"execution": "thread"})
            runner = hub._create_runner("agent-1", config, None, None)
            mock_tr.assert_called_once()
            assert runner is mock_tr.return_value

    def test_create_runner_process_mode_default(self, hub):
        """Default execution mode creates ProcessRunner."""
        from appinfra import DotDict

        with (
            patch("llm_gent.hub.hub.AgentService"),
            patch("llm_gent.hub.hub.ThreadRunner"),
            patch("llm_gent.hub.hub.ProcessRunner") as mock_pr,
        ):
            config = DotDict({})
            runner = hub._create_runner("agent-1", config, None, None)
            mock_pr.assert_called_once()
            assert runner is mock_pr.return_value

    def test_create_runner_explicit_process_mode(self, hub):
        """Explicit 'process' execution mode creates ProcessRunner."""
        from appinfra import DotDict

        with (
            patch("llm_gent.hub.hub.AgentService"),
            patch("llm_gent.hub.hub.ThreadRunner"),
            patch("llm_gent.hub.hub.ProcessRunner") as mock_pr,
        ):
            config = DotDict({"execution": "process"})
            runner = hub._create_runner("agent-1", config, None, None)
            mock_pr.assert_called_once()
            assert runner is mock_pr.return_value


class TestHubCreateChannel:
    """Tests for _create_channel."""

    def test_create_channel_stores_channel(self, hub):
        """_create_channel creates transport + BufferedChannel and stores it."""
        with patch("llm_gent.hub.hub.BufferedChannel") as mock_bc:
            mock_transport = MagicMock()
            hub._bus.create_agent_transport.return_value = mock_transport

            hub._create_channel("agent-1")

            hub._bus.create_agent_transport.assert_called_once_with("agent-1")
            mock_bc.assert_called_once_with(mock_transport)
            assert hub._channels["agent-1"] is mock_bc.return_value


class TestHubStopAgent:
    """Tests for stop_agent."""

    def test_stop_agent_with_runner(self, hub):
        """stop_agent stops runner, cleans up, unregisters, broadcasts."""
        hub.registry.register("agent-1")
        mock_runner = MagicMock()
        hub._runners["agent-1"] = mock_runner
        hub._channels["agent-1"] = MagicMock()
        hub._bus.broadcast.reset_mock()

        hub.stop_agent("agent-1")

        mock_runner.stop.assert_called_once()
        assert "agent-1" not in hub._runners
        assert "agent-1" not in hub._channels
        assert hub.registry.get("agent-1") is None
        # AgentLeft broadcast
        left_calls = [
            c for c in hub._bus.broadcast.call_args_list if isinstance(c[0][0], AgentLeft)
        ]
        assert len(left_calls) == 1
        assert left_calls[0][0][0].reason == "shutdown"

    def test_stop_agent_without_runner(self, hub):
        """stop_agent works even if no runner exists for the agent."""
        hub.registry.register("agent-1")
        hub._bus.broadcast.reset_mock()

        hub.stop_agent("agent-1")

        assert hub.registry.get("agent-1") is None

    def test_stop_agent_cleanup_on_runner_error(self, hub):
        """stop_agent cleans up even if runner.stop() raises."""
        hub.registry.register("agent-1")
        mock_runner = MagicMock()
        mock_runner.stop.side_effect = RuntimeError("stop failed")
        hub._runners["agent-1"] = mock_runner
        hub._bus.broadcast.reset_mock()

        with pytest.raises(RuntimeError, match="stop failed"):
            hub.stop_agent("agent-1")

        # Cleanup still happens despite error (finally block)
        assert "agent-1" not in hub._runners
        assert hub.registry.get("agent-1") is None


class TestHubGetInsights:
    """Tests for get_insights."""

    def test_get_insights_known_agent(self, hub):
        """get_insights returns stats for known agent."""
        hub.registry.register("agent-1")
        insights = hub.get_insights("agent-1")

        assert len(insights) == 1
        assert insights[0]["type"] == "stats"
        assert "ticks" in insights[0]
        assert "errors" in insights[0]
        assert "health" in insights[0]

    def test_get_insights_unknown_agent(self, hub):
        """get_insights returns empty list for unknown agent."""
        insights = hub.get_insights("ghost")
        assert insights == []


class TestRequireChannel:
    """Tests for _require_channel."""

    def test_require_channel_found(self, hub):
        """_require_channel returns channel when it exists."""
        mock_channel = MagicMock()
        hub._channels["agent-1"] = mock_channel
        assert hub._require_channel("agent-1") is mock_channel

    def test_require_channel_not_found(self, hub):
        """_require_channel raises KeyError when channel missing."""
        with pytest.raises(KeyError, match="No channel"):
            hub._require_channel("ghost")
