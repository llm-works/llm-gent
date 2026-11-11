# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for the Hub with real ZMQ bus."""

from typing import Any

import pytest
from appinfra.service import BufferedChannel

from llm_gent.bus.protocol import RegisterRequest, UnregisterRequest
from llm_gent.bus.transport import ZMQWorkerBus
from llm_gent.hub import Hub, HubConfig
from tests.integration._ports import coordinator_config, start_on_free_ports, worker_config
from tests.integration._zmq import holds, send_until


pytestmark = pytest.mark.integration


def _start_hub(lg: Any, ports: list[int]) -> tuple[Hub, list[int]]:
    """A started Hub bound to ``ports``; no shutdown grace (these tests run no agents)."""
    hub_config = HubConfig(
        bus=coordinator_config(ports),
        health_check_interval=60.0,
        shutdown_grace_secs=0.0,
    )
    hub = Hub(lg, hub_config, bus_config=worker_config(ports))
    hub.start()
    return hub, ports


@pytest.fixture
def hub_and_worker():
    """Start a real Hub + worker bus with channel.

    The worker's DEALER transport queues requests until it has connected.
    """
    from unittest.mock import MagicMock

    lg = MagicMock()
    hub, ports = start_on_free_ports(lambda p: _start_hub(lg, p))
    worker = ZMQWorkerBus(lg, "test-worker", worker_config(ports))
    worker.start()

    assert worker.transport is not None
    channel: BufferedChannel[Any, Any] = BufferedChannel(worker.transport)

    yield hub, worker, channel

    channel.close()
    worker.stop()
    hub.stop()


class TestHubRegistrationFlow:
    """End-to-end registration flow through real bus."""

    def test_worker_registers_with_hub(self, hub_and_worker):
        """Worker registers via channel, appears in hub registry."""
        hub, worker, channel = hub_and_worker

        req = RegisterRequest(agent_id="test-worker", capabilities=["search"])
        channel.submit(req, timeout=5.0)

        info = hub.registry.get("test-worker")
        assert info is not None
        assert info.capabilities == ["search"]

    def test_worker_unregisters_from_hub(self, hub_and_worker):
        """Worker unregisters via channel, removed from hub registry."""
        hub, worker, channel = hub_and_worker

        channel.submit(RegisterRequest(agent_id="test-worker"), timeout=5.0)
        assert hub.registry.count == 1

        channel.submit(UnregisterRequest(agent_id="test-worker"), timeout=5.0)
        assert hub.registry.count == 0


class TestHubHeartbeatFlow:
    """End-to-end heartbeat flow through real bus."""

    def test_worker_heartbeat_updates_registry(self, hub_and_worker):
        """Worker heartbeat via pub/sub updates registry stats."""
        hub, worker, channel = hub_and_worker

        channel.submit(RegisterRequest(agent_id="test-worker"), timeout=5.0)

        def updated() -> bool:
            info = hub.registry.get("test-worker")
            return info is not None and info.stats.ticks == 42

        send_until(lambda: worker.publish_heartbeat({"ticks": 42, "errors": 2}), holds(updated))


class TestHubMultipleWorkers:
    """Hub with multiple workers."""

    @pytest.fixture
    def hub_and_workers(self):
        from unittest.mock import MagicMock

        lg = MagicMock()
        hub, ports = start_on_free_ports(lambda p: _start_hub(lg, p))

        workers = []
        channels = []
        for i in range(3):
            w = ZMQWorkerBus(lg, f"worker-{i}", worker_config(ports))
            w.start()
            workers.append(w)
            assert w.transport is not None
            ch: BufferedChannel[Any, Any] = BufferedChannel(w.transport)
            channels.append(ch)

        yield hub, workers, channels

        for ch in channels:
            ch.close()
        for w in workers:
            w.stop()
        hub.stop()

    def test_all_workers_register(self, hub_and_workers):
        """All workers register and appear in registry."""
        hub, workers, channels = hub_and_workers

        for w, ch in zip(workers, channels, strict=True):
            ch.submit(RegisterRequest(agent_id=w.agent_id, capabilities=["test"]), timeout=5.0)

        assert hub.registry.count == 3
        ids = {a.id for a in hub.registry.list_agents()}
        assert ids == {"worker-0", "worker-1", "worker-2"}
