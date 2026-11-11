# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""AgentRunner's background thread: start and stop."""

from unittest.mock import MagicMock, patch

import pytest

from llm_gent.bus.transport import WorkerBusConfig
from llm_gent.runtime.runner import AgentRunner

from ..unit.test_runner import MockHandler


pytestmark = pytest.mark.integration


@pytest.fixture
def ext_runner() -> AgentRunner:
    """AgentRunner for an external agent (not connected to a bus)."""
    return AgentRunner(
        lg=MagicMock(), handler=MockHandler(), agent_id="ext-agent", bus_config=WorkerBusConfig()
    )


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

    def test_stop_without_start_is_noop(self, ext_runner):
        """stop() when not started is safe."""
        ext_runner.stop()  # Should not raise
