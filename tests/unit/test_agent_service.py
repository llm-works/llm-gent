"""Tests for AgentService (runtime/service.py)."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.runtime.service import AgentService


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def lg():
    return MagicMock()


@pytest.fixture
def config():
    return DotDict({})


@pytest.fixture
def llm_config():
    return MagicMock()


@pytest.fixture
def bus_config():
    return MagicMock()


@pytest.fixture
def service(lg, config, llm_config, bus_config):
    return AgentService(
        lg=lg,
        agent_name="test-agent",
        config=config,
        llm_config=llm_config,
        bus_config=bus_config,
    )


# =============================================================================
# __init__
# =============================================================================


class TestInit:
    def test_stores_all_args(self, lg, llm_config, bus_config):
        cfg = DotDict({"key": "val"})
        svc = AgentService(
            lg=lg,
            agent_name="my-agent",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
            learn_config="lc",
            variables={"v": "1"},
            factory_module="my.module",
        )
        assert svc._agent_name == "my-agent"
        assert svc._config is cfg
        assert svc._llm_config is llm_config
        assert svc._bus_config is bus_config
        assert svc._learn_config == "lc"
        assert svc._variables == {"v": "1"}
        assert svc._factory_module == "my.module"

    def test_defaults(self, service):
        assert service._runner is None
        assert service._healthy is False
        assert service._shutdown_event is None
        assert service._variables == {}
        assert service._learn_config is None
        assert service._factory_module == "llm_gent.agents.default"


# =============================================================================
# name property
# =============================================================================


class TestName:
    def test_returns_agent_name(self, service):
        assert service.name == "test-agent"


# =============================================================================
# setup()
# =============================================================================


class TestSetup:
    @patch("llm_gent.runtime.service.ManagedAgentRunner")
    @patch("llm_gent.runtime.service.AgentService._load_factory")
    @patch("llm_gent.core.platform.PlatformContext.from_config")
    def test_creates_platform_factory_agent_runner(
        self, mock_from_config, mock_load_factory, mock_runner_cls, service
    ):
        mock_platform = MagicMock()
        mock_from_config.return_value = mock_platform

        mock_factory = MagicMock()
        mock_agent = MagicMock()
        mock_factory.create.return_value = mock_agent
        mock_load_factory.return_value = mock_factory

        service.setup()

        mock_from_config.assert_called_once_with(
            lg=service._lg,
            llm_config=service._llm_config,
            learn_config=service._learn_config,
        )
        mock_load_factory.assert_called_once_with(mock_platform)
        mock_factory.create.assert_called_once_with(service._config, variables=service._variables)
        mock_agent.start.assert_called_once()
        mock_runner_cls.assert_called_once()
        assert service._runner is mock_runner_cls.return_value

    @patch("llm_gent.runtime.service.ManagedAgentRunner")
    @patch("llm_gent.runtime.service.AgentService._load_factory")
    @patch("llm_gent.core.platform.PlatformContext.from_config")
    def test_passes_schedule_interval(
        self, mock_from_config, mock_load_factory, mock_runner_cls, lg, llm_config, bus_config
    ):
        cfg = DotDict({"schedule": {"interval": "30"}})
        svc = AgentService(
            lg=lg,
            agent_name="a",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
        )
        mock_factory = MagicMock()
        mock_load_factory.return_value = mock_factory

        svc.setup()

        call_kwargs = mock_runner_cls.call_args[1]
        assert call_kwargs["schedule_interval"] == 30.0


# =============================================================================
# execute()
# =============================================================================


class TestExecute:
    def test_raises_when_no_runner(self, service):
        with pytest.raises(RuntimeError, match="setup.*must be called"):
            service.execute()

    def test_calls_runner_run(self, service):
        runner = MagicMock()
        service._runner = runner

        service.execute()

        assert service._healthy is True
        runner.run.assert_called_once()

    def test_sets_healthy_before_run(self, service):
        """Healthy flag is set before runner.run() is called."""
        runner = MagicMock()
        healthy_at_run_time = None

        def capture_healthy():
            nonlocal healthy_at_run_time
            healthy_at_run_time = service._healthy

        runner.run.side_effect = capture_healthy
        service._runner = runner

        service.execute()

        assert healthy_at_run_time is True


# =============================================================================
# teardown()
# =============================================================================


class TestTeardown:
    def test_requests_shutdown_and_sets_unhealthy(self, service):
        runner = MagicMock()
        service._runner = runner
        service._healthy = True

        service.teardown()

        runner.request_shutdown.assert_called_once()
        assert service._healthy is False

    def test_safe_when_no_runner(self, service):
        """teardown() should not raise when runner is None."""
        service.teardown()
        assert service._healthy is False


# =============================================================================
# is_healthy()
# =============================================================================


class TestIsHealthy:
    def test_initially_false(self, service):
        assert service.is_healthy() is False

    def test_true_after_execute(self, service):
        service._runner = MagicMock()
        service.execute()
        assert service.is_healthy() is True

    def test_false_after_teardown(self, service):
        service._runner = MagicMock()
        service.execute()
        service.teardown()
        assert service.is_healthy() is False


# =============================================================================
# _start_shutdown_watcher()
# =============================================================================


class TestStartShutdownWatcher:
    def test_noop_when_no_event(self, service):
        """No thread should be spawned when _shutdown_event is None."""
        with patch("llm_gent.runtime.service.threading.Thread") as mock_thread:
            service._start_shutdown_watcher()
            mock_thread.assert_not_called()

    def test_spawns_thread_when_event_set(self, service):
        service._shutdown_event = MagicMock()
        with patch("llm_gent.runtime.service.threading.Thread") as mock_thread:
            service._start_shutdown_watcher()
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()
            call_kwargs = mock_thread.call_args[1]
            assert call_kwargs["daemon"] is True
            assert "test-agent" in call_kwargs["name"]


# =============================================================================
# _load_factory()
# =============================================================================


class TestLoadFactory:
    def test_success(self, service):
        fake_module = types.ModuleType("fake")
        fake_factory_cls = MagicMock()
        fake_module.Factory = fake_factory_cls
        platform = MagicMock()

        with patch("importlib.import_module", return_value=fake_module) as mock_import:
            result = service._load_factory(platform)

        mock_import.assert_called_once_with(service._factory_module)
        fake_factory_cls.assert_called_once_with(platform=platform)
        assert result is fake_factory_cls.return_value

    def test_uses_module_from_config(self, lg, llm_config, bus_config):
        cfg = DotDict({"module": "custom.agents"})
        svc = AgentService(
            lg=lg,
            agent_name="a",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
        )
        fake_module = types.ModuleType("custom_agents")
        fake_module.Factory = MagicMock()

        with patch("importlib.import_module", return_value=fake_module) as mock_import:
            svc._load_factory(MagicMock())

        mock_import.assert_called_once_with("custom.agents")

    def test_import_error_wrapped(self, service):
        with (
            patch("importlib.import_module", side_effect=ImportError("no module")),
            pytest.raises(RuntimeError, match="Failed to load factory"),
        ):
            service._load_factory(MagicMock())

    def test_attribute_error_wrapped(self, service):
        fake_module = types.ModuleType("fake")
        # No Factory attribute on this module
        with (
            patch("importlib.import_module", return_value=fake_module),
            pytest.raises(RuntimeError, match="Failed to load factory"),
        ):
            service._load_factory(MagicMock())


# =============================================================================
# _extract_schedule()
# =============================================================================


class TestExtractSchedule:
    def test_valid_interval(self, lg, llm_config, bus_config):
        cfg = DotDict({"schedule": {"interval": 60}})
        svc = AgentService(
            lg=lg,
            agent_name="a",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
        )
        assert svc._extract_schedule() == 60.0

    def test_string_interval(self, lg, llm_config, bus_config):
        cfg = DotDict({"schedule": {"interval": "45"}})
        svc = AgentService(
            lg=lg,
            agent_name="a",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
        )
        assert svc._extract_schedule() == 45.0

    def test_no_schedule(self, service):
        assert service._extract_schedule() is None

    def test_schedule_without_interval(self, lg, llm_config, bus_config):
        cfg = DotDict({"schedule": {"other": "value"}})
        svc = AgentService(
            lg=lg,
            agent_name="a",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
        )
        assert svc._extract_schedule() is None

    def test_non_numeric_interval(self, lg, llm_config, bus_config):
        cfg = DotDict({"schedule": {"interval": "not-a-number"}})
        svc = AgentService(
            lg=lg,
            agent_name="a",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
        )
        assert svc._extract_schedule() is None

    def test_schedule_not_dict(self, lg, llm_config, bus_config):
        cfg = DotDict({"schedule": "cron-string"})
        svc = AgentService(
            lg=lg,
            agent_name="a",
            config=cfg,
            llm_config=llm_config,
            bus_config=bus_config,
        )
        assert svc._extract_schedule() is None
