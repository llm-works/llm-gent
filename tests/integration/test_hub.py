"""Integration tests for the Hub with real ZMQ bus."""

import socket
import time

import pytest

from llm_gent.bus.protocol import (
    RegisterRequest,
    UnregisterRequest,
)
from llm_gent.bus.transport import CoordinatorBusConfig, WorkerBusConfig, ZMQWorkerBus
from llm_gent.hub import Hub, HubConfig


pytestmark = pytest.mark.integration


def _find_free_ports(n: int) -> list[int]:
    """Allocate n ephemeral ports."""
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
    """Start a real Hub + worker bus pair."""
    from unittest.mock import MagicMock

    lg = MagicMock()
    ports = _find_free_ports(3)

    hub_config = HubConfig(
        bus=CoordinatorBusConfig(
            router_port=ports[0],
            pub_port=ports[1],
            sub_port=ports[2],
        ),
        health_check_interval=60.0,  # long interval so it doesn't interfere
    )

    worker_config = WorkerBusConfig(
        router_port=ports[0],
        pub_port=ports[1],
        sub_port=ports[2],
    )

    hub = Hub(lg, hub_config)
    worker = ZMQWorkerBus(lg, "test-worker", worker_config)

    hub.start()
    time.sleep(0.1)
    worker.start()
    time.sleep(0.2)

    yield hub, worker

    worker.stop()
    hub.stop()


class TestHubRegistrationFlow:
    """End-to-end registration flow through real bus."""

    def test_worker_registers_with_hub(self, hub_and_worker):
        """Worker registers via bus, appears in hub registry."""
        hub, worker = hub_and_worker

        req = RegisterRequest(
            agent_id="test-worker",
            capabilities=["search"],
        )
        resp = worker.send(req, timeout=5.0)

        assert resp.success is True
        info = hub.registry.get("test-worker")
        assert info is not None
        assert info.capabilities == ["search"]

    def test_worker_unregisters_from_hub(self, hub_and_worker):
        """Worker unregisters via bus, removed from hub registry."""
        hub, worker = hub_and_worker

        # Register first
        reg = RegisterRequest(agent_id="test-worker")
        worker.send(reg, timeout=5.0)
        assert hub.registry.count == 1

        # Unregister
        unreg = UnregisterRequest(agent_id="test-worker")
        resp = worker.send(unreg, timeout=5.0)

        assert resp.success is True
        assert hub.registry.count == 0


class TestHubHeartbeatFlow:
    """End-to-end heartbeat flow through real bus."""

    def test_worker_heartbeat_updates_registry(self, hub_and_worker):
        """Worker heartbeat via pub/sub updates registry stats."""
        hub, worker = hub_and_worker

        # Register first (so registry knows the agent)
        reg = RegisterRequest(agent_id="test-worker")
        worker.send(reg, timeout=5.0)

        # Send heartbeat
        worker.publish_heartbeat({"ticks": 42, "errors": 2})
        time.sleep(0.5)  # let pub/sub propagate

        info = hub.registry.get("test-worker")
        assert info is not None
        assert info.stats.ticks == 42
        assert info.stats.errors == 2


class TestHubMultipleWorkers:
    """Hub with multiple workers."""

    @pytest.fixture
    def hub_and_workers(self):
        """Start Hub + 3 workers."""
        from unittest.mock import MagicMock

        lg = MagicMock()
        ports = _find_free_ports(3)

        hub_config = HubConfig(
            bus=CoordinatorBusConfig(
                router_port=ports[0],
                pub_port=ports[1],
                sub_port=ports[2],
            ),
            health_check_interval=60.0,
        )

        hub = Hub(lg, hub_config)
        hub.start()
        time.sleep(0.1)

        workers = []
        for i in range(3):
            cfg = WorkerBusConfig(
                router_port=ports[0],
                pub_port=ports[1],
                sub_port=ports[2],
            )
            w = ZMQWorkerBus(lg, f"worker-{i}", cfg)
            w.start()
            workers.append(w)

        time.sleep(0.3)

        yield hub, workers

        for w in workers:
            w.stop()
        hub.stop()

    def test_all_workers_register(self, hub_and_workers):
        """All workers register and appear in registry."""
        hub, workers = hub_and_workers

        for w in workers:
            req = RegisterRequest(agent_id=w.agent_id, capabilities=["test"])
            resp = w.send(req, timeout=5.0)
            assert resp.success is True

        assert hub.registry.count == 3
        agents = hub.registry.list_agents()
        ids = {a.id for a in agents}
        assert ids == {"worker-0", "worker-1", "worker-2"}
