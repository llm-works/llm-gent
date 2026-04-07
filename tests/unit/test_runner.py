"""Tests for agent runners (ManagedAgentRunner and AgentRunner)."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.bus.protocol import (
    AgentJoined,
    AgentLeft,
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HeartbeatRequest,
    ShutdownNotice,
    ShutdownRequest,
    ShutdownResponse,
)
from llm_gent.bus.transport import WorkerBusConfig
from llm_gent.runtime.handler import Handler
from llm_gent.runtime.runner import AgentRunner, ManagedAgentRunner


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create mock agent."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.cycle_count = 0
    return agent


@pytest.fixture
def bus_config():
    return WorkerBusConfig()


@pytest.fixture
def managed_runner(mock_agent, bus_config):
    """Create ManagedAgentRunner (not connected to bus)."""
    lg = MagicMock()
    return ManagedAgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)


class MockHandler:
    """Test handler implementation."""

    def __init__(self) -> None:
        self.ask_response = "test answer"
        self.ask_calls: list[str] = []
        self.feedback_calls: list[str] = []
        self.shutdown_called = False

    def on_ask(self, question: str) -> str:
        self.ask_calls.append(question)
        return self.ask_response

    def on_feedback(self, message: str) -> None:
        self.feedback_calls.append(message)

    def on_shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def handler():
    return MockHandler()


@pytest.fixture
def ext_runner(handler, bus_config):
    """Create AgentRunner for external agents (not connected to bus)."""
    lg = MagicMock()
    return AgentRunner(
        lg=lg,
        handler=handler,
        agent_id="ext-agent",
        bus_config=bus_config,
        capabilities=["search", "summarize"],
        metadata={"version": "1.0"},
    )


# =============================================================================
# Handler protocol
# =============================================================================


class TestHandlerProtocol:
    """Tests for Handler protocol compliance."""

    def test_mock_handler_is_handler(self, handler):
        """MockHandler satisfies Handler protocol."""
        assert isinstance(handler, Handler)

    def test_object_not_handler(self):
        """Arbitrary object does not satisfy Handler protocol."""
        assert not isinstance(object(), Handler)

    def test_partial_impl_not_handler(self):
        """Class with only some methods does not satisfy Handler."""

        class Partial:
            def on_ask(self, question: str) -> str:
                return ""

        assert not isinstance(Partial(), Handler)


# =============================================================================
# ManagedAgentRunner
# =============================================================================


class TestManagedAgentRunner:
    """Tests for ManagedAgentRunner initialization."""

    def test_runner_init(self, mock_agent, bus_config):
        """Runner initializes with bus config."""
        lg = MagicMock()
        runner = ManagedAgentRunner(lg=lg, agent=mock_agent, bus_config=bus_config)

        assert runner._agent is mock_agent
        assert runner._bus_config is bus_config
        assert runner._ticker is None

    def test_runner_init_with_schedule(self, mock_agent, bus_config):
        """Runner creates Ticker when schedule_interval provided."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=60.0
        )

        assert runner._ticker is not None


class TestManagedRequestHandling:
    """Tests for ManagedAgentRunner request dispatch (via _AgentHandler adapter)."""

    def test_handle_ask(self, managed_runner, mock_agent):
        """Ask request calls agent.ask() and returns response."""
        mock_agent.ask.return_value = "Test answer"
        req = AskRequest(question="What?")

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is True
        assert resp.response == "Test answer"
        mock_agent.ask.assert_called_once_with("What?")

    def test_handle_ask_error(self, managed_runner, mock_agent):
        """Ask request returns error response on exception."""
        mock_agent.ask.side_effect = RuntimeError("LLM failed")
        req = AskRequest(question="What?")

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is False
        assert "LLM failed" in resp.error

    def test_handle_feedback(self, managed_runner, mock_agent):
        """Feedback request calls agent.record_feedback()."""
        req = FeedbackRequest(message="Good job!")

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is True
        mock_agent.record_feedback.assert_called_once_with("Good job!")

    def test_handle_shutdown(self, managed_runner):
        """Shutdown request sets running to False."""
        managed_runner._running = True
        req = ShutdownRequest()

        resp = managed_runner._handle_request(req)

        assert isinstance(resp, ShutdownResponse)
        assert resp.success is True
        assert managed_runner._running is False

    def test_handle_unknown_request(self, managed_runner):
        """Unknown request type returns error response."""
        req = MagicMock(spec=["id"])
        req.id = "test-id"

        resp = managed_runner._handle_request(req)

        assert resp.success is False
        assert "unknown" in resp.error


class TestManagedBroadcastHandling:
    """Tests for ManagedAgentRunner broadcast handling."""

    def test_handle_heartbeat_broadcast(self, managed_runner):
        """HeartbeatRequest triggers heartbeat response."""
        managed_runner._bus = MagicMock()
        req = HeartbeatRequest(round_id="r1")

        managed_runner._handle_broadcast(req)

        managed_runner._bus.publish_heartbeat.assert_called_once()

    def test_handle_shutdown_notice(self, managed_runner):
        """ShutdownNotice sets stop event."""
        notice = ShutdownNotice(reason="test", grace_period_secs=1.0)

        managed_runner._handle_broadcast(notice)

        assert managed_runner._running is False

    def test_handle_agent_joined(self, managed_runner):
        """AgentJoined logs and doesn't crash."""
        msg = AgentJoined(agent_id="peer-1", capabilities=["search"])

        managed_runner._handle_broadcast(msg)

        managed_runner._lg.info.assert_called()
        call_args = managed_runner._lg.info.call_args
        assert "joined" in call_args[0][0]

    def test_handle_agent_left(self, managed_runner):
        """AgentLeft logs and doesn't crash."""
        msg = AgentLeft(agent_id="peer-1", reason="voluntary")

        managed_runner._handle_broadcast(msg)

        managed_runner._lg.info.assert_called()
        call_args = managed_runner._lg.info.call_args
        assert "left" in call_args[0][0]

    def test_handle_unknown_broadcast(self, managed_runner):
        """Unknown broadcast type is silently ignored."""
        msg = MagicMock(spec=[])

        managed_runner._handle_broadcast(msg)  # Should not raise


class TestManagedScheduling:
    """Tests for scheduled execution."""

    def test_should_run_cycle_continuous(self, mock_agent, bus_config):
        """Continuous mode always runs cycle."""
        lg = MagicMock()
        runner = ManagedAgentRunner(
            lg=lg, agent=mock_agent, bus_config=bus_config, schedule_interval=0
        )

        assert runner._should_run_cycle() is True

    def test_should_run_cycle_message_only(self, managed_runner):
        """Message-only mode never runs cycle."""
        assert managed_runner._should_run_cycle() is False

    def test_run_cycle_calls_run_once(self, managed_runner, mock_agent):
        """Cycle calls agent.run_once()."""
        managed_runner._run_cycle()
        mock_agent.run_once.assert_called_once()

    def test_run_cycle_handles_error(self, managed_runner, mock_agent):
        """Cycle handles agent.run_once() exceptions."""
        mock_agent.run_once.side_effect = RuntimeError("cycle failed")
        managed_runner._run_cycle()  # Should not raise


# =============================================================================
# AgentRunner (external)
# =============================================================================


class TestAgentRunner:
    """Tests for AgentRunner initialization."""

    def test_runner_init(self, handler, bus_config):
        """Runner initializes with handler and config."""
        lg = MagicMock()
        runner = AgentRunner(
            lg=lg,
            handler=handler,
            agent_id="my-agent",
            bus_config=bus_config,
            capabilities=["search"],
            metadata={"v": "1"},
        )

        assert runner.agent_id == "my-agent"
        assert runner._capabilities == ["search"]
        assert runner._metadata == {"v": "1"}

    def test_runner_default_capabilities(self, handler, bus_config):
        """Runner defaults to empty capabilities and metadata."""
        lg = MagicMock()
        runner = AgentRunner(lg=lg, handler=handler, agent_id="x", bus_config=bus_config)

        assert runner._capabilities == []
        assert runner._metadata == {}


class TestExtRequestHandling:
    """Tests for AgentRunner request dispatch via Handler."""

    def test_handle_ask(self, ext_runner, handler):
        """Ask request dispatches to handler.on_ask()."""
        handler.ask_response = "42"
        req = AskRequest(question="meaning of life?")

        resp = ext_runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is True
        assert resp.response == "42"
        assert handler.ask_calls == ["meaning of life?"]

    def test_handle_ask_error(self, ext_runner, handler):
        """Ask returns error response when handler raises."""
        handler.on_ask = MagicMock(side_effect=ValueError("bad input"))
        req = AskRequest(question="crash")

        resp = ext_runner._handle_request(req)

        assert isinstance(resp, AskResponse)
        assert resp.success is False
        assert "bad input" in resp.error

    def test_handle_feedback(self, ext_runner, handler):
        """Feedback request dispatches to handler.on_feedback()."""
        req = FeedbackRequest(message="nice work")

        resp = ext_runner._handle_request(req)

        assert isinstance(resp, FeedbackResponse)
        assert resp.success is True
        assert handler.feedback_calls == ["nice work"]

    def test_handle_shutdown(self, ext_runner, handler):
        """Shutdown request calls handler.on_shutdown() and sets stop event."""
        ext_runner._running = True
        req = ShutdownRequest()

        resp = ext_runner._handle_request(req)

        assert isinstance(resp, ShutdownResponse)
        assert resp.success is True
        assert handler.shutdown_called is True
        assert ext_runner._running is False

    def test_handle_unknown_request(self, ext_runner):
        """Unknown request type returns error response."""
        req = MagicMock(spec=["id"])
        req.id = "test-id"

        resp = ext_runner._handle_request(req)

        assert resp.success is False
        assert "unknown" in resp.error


class TestExtBroadcastHandling:
    """Tests for AgentRunner broadcast handling."""

    def test_handle_heartbeat_broadcast(self, ext_runner):
        """HeartbeatRequest triggers heartbeat response."""
        ext_runner._bus = MagicMock()
        req = HeartbeatRequest(round_id="r1")

        ext_runner._handle_broadcast(req)

        ext_runner._bus.publish_heartbeat.assert_called_once()

    def test_handle_shutdown_notice(self, ext_runner):
        """ShutdownNotice sets stop event."""
        notice = ShutdownNotice(reason="test", grace_period_secs=1.0)

        ext_runner._handle_broadcast(notice)

        assert ext_runner._running is False


class TestExtStartStop:
    """Tests for AgentRunner start/stop (background thread)."""

    def _blocking_run(self, runner: AgentRunner) -> None:
        """Replacement for run() that blocks until stop is requested."""
        runner._stop_event.wait()

    def test_start_creates_thread(self, ext_runner):
        """start() creates a background thread."""
        with patch.object(ext_runner, "run", lambda: self._blocking_run(ext_runner)):
            ext_runner.start()
            assert ext_runner._bg_thread is not None
            assert ext_runner._bg_thread.is_alive()
            ext_runner.stop()

    def test_start_raises_if_already_running(self, ext_runner):
        """start() raises if already started."""
        with patch.object(ext_runner, "run", lambda: self._blocking_run(ext_runner)):
            ext_runner.start()
            with pytest.raises(RuntimeError, match="already started"):
                ext_runner.start()
            ext_runner.stop()

    def test_stop_clears_thread(self, ext_runner):
        """stop() joins and clears the background thread."""
        with patch.object(ext_runner, "run", lambda: self._blocking_run(ext_runner)):
            ext_runner.start()
            ext_runner.stop()
            assert ext_runner._bg_thread is None


class TestAgentRunnerConnect:
    """Tests for AgentRunner.connect() classmethod."""

    def test_connect_fetches_bus_config(self, handler):
        """connect() fetches config from hub and creates runner."""
        config_json = b'{"coordinator_host": "10.0.1.5", "router_port": 5555, "pub_port": 5556, "sub_port": 5557}'

        mock_resp = MagicMock()
        mock_resp.read.return_value = config_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            runner = AgentRunner.connect(
                lg=MagicMock(),
                handler=handler,
                agent_id="remote-agent",
                hub_url="http://hub:8080",
                capabilities=["translate"],
            )

            mock_urlopen.assert_called_once_with("http://hub:8080/bus/config", timeout=5)
            assert runner.agent_id == "remote-agent"
            assert runner._bus_config.coordinator_host == "10.0.1.5"
            assert runner._bus_config.router_port == 5555
            assert runner._capabilities == ["translate"]

    def test_connect_strips_trailing_slash(self, handler):
        """connect() strips trailing slash from hub_url."""
        config_json = b'{"coordinator_host": "x", "router_port": 1, "pub_port": 2, "sub_port": 3}'

        mock_resp = MagicMock()
        mock_resp.read.return_value = config_json
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            AgentRunner.connect(
                lg=MagicMock(),
                handler=handler,
                agent_id="x",
                hub_url="http://hub:8080/",
            )
            mock_urlopen.assert_called_once_with("http://hub:8080/bus/config", timeout=5)

    def test_connect_raises_on_failure(self, handler):
        """connect() raises ConnectionError on HTTP failure."""
        with (
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            pytest.raises(ConnectionError, match="failed to fetch"),
        ):
            AgentRunner.connect(
                lg=MagicMock(),
                handler=handler,
                agent_id="x",
                hub_url="http://unreachable:9999",
            )
