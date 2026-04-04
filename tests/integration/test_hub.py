"""Integration tests for the Hub with real ZMQ bus."""

import socket
import time
from typing import Any

import pytest
from appinfra.service import BufferedChannel

from llm_gent.bus.protocol import RegisterRequest, UnregisterRequest
from llm_gent.bus.transport import CoordinatorBusConfig, WorkerBusConfig, ZMQWorkerBus
from llm_gent.hub import Hub, HubConfig


pytestmark = pytest.mark.integration


def _find_free_ports(n: int) -> list[int]:
    socks = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


@pytest.fixture
def hub_and_worker():
    """Start a real Hub + worker bus with channel."""
    from unittest.mock import MagicMock

    lg = MagicMock()
    ports = _find_free_ports(3)

    hub_config = HubConfig(
        bus=CoordinatorBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2]),
        health_check_interval=60.0,
    )
    worker_config = WorkerBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2])

    hub = Hub(lg, hub_config)
    worker = ZMQWorkerBus(lg, "test-worker", worker_config)

    hub.start()
    time.sleep(0.1)
    worker.start()
    time.sleep(0.2)

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

        worker.publish_heartbeat({"ticks": 42, "errors": 2})
        time.sleep(0.5)

        info = hub.registry.get("test-worker")
        assert info is not None
        assert info.stats.ticks == 42


class TestHubMultipleWorkers:
    """Hub with multiple workers."""

    @pytest.fixture
    def hub_and_workers(self):
        from unittest.mock import MagicMock

        lg = MagicMock()
        ports = _find_free_ports(3)

        hub_config = HubConfig(
            bus=CoordinatorBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2]),
            health_check_interval=60.0,
        )

        hub = Hub(lg, hub_config)
        hub.start()
        time.sleep(0.1)

        workers = []
        channels = []
        for i in range(3):
            cfg = WorkerBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2])
            w = ZMQWorkerBus(lg, f"worker-{i}", cfg)
            w.start()
            workers.append(w)
            assert w.transport is not None
            ch: BufferedChannel[Any, Any] = BufferedChannel(w.transport)
            channels.append(ch)

        time.sleep(0.3)

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
