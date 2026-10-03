# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration tests for the Hub with real ZMQ bus."""

import time
from typing import Any

import pytest
from appinfra.service import BufferedChannel

from llm_gent.bus.protocol import RegisterRequest, UnregisterRequest
from llm_gent.bus.transport import CoordinatorBusConfig, WorkerBusConfig, ZMQWorkerBus
from llm_gent.hub import Hub, HubConfig
from tests.integration._ports import start_on_free_ports


pytestmark = pytest.mark.integration


def _worker_config(ports: list[int]) -> WorkerBusConfig:
    return WorkerBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2])


def _start_hub(lg: Any, ports: list[int]) -> tuple[Hub, list[int]]:
    """A started Hub bound to ``ports``; no shutdown grace (these tests run no agents)."""
    hub_config = HubConfig(
        bus=CoordinatorBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2]),
        health_check_interval=60.0,
        shutdown_grace_secs=0.0,
    )
    hub = Hub(lg, hub_config, bus_config=_worker_config(ports))
    hub.start()
    return hub, ports


def _wait_for_zmq_connect(seconds: float = 0.2) -> None:
    """Wait for ZMQ async connect/bind to settle.

    ZMQ connect is asynchronous and provides no readiness signal.
    A short sleep is the only reliable way to allow the underlying
    TCP handshake and subscription propagation to complete.
    """
    time.sleep(seconds)


@pytest.fixture
def hub_and_worker():
    """Start a real Hub + worker bus with channel."""
    from unittest.mock import MagicMock

    lg = MagicMock()
    hub, ports = start_on_free_ports(lambda p: _start_hub(lg, p))
    _wait_for_zmq_connect(0.1)
    worker = ZMQWorkerBus(lg, "test-worker", _worker_config(ports))
    worker.start()
    _wait_for_zmq_connect(0.2)

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
        _wait_for_zmq_connect(0.5)

        info = hub.registry.get("test-worker")
        assert info is not None
        assert info.stats.ticks == 42


class TestHubMultipleWorkers:
    """Hub with multiple workers."""

    @pytest.fixture
    def hub_and_workers(self):
        from unittest.mock import MagicMock

        lg = MagicMock()
        hub, ports = start_on_free_ports(lambda p: _start_hub(lg, p))
        _wait_for_zmq_connect(0.1)

        workers = []
        channels = []
        for i in range(3):
            w = ZMQWorkerBus(lg, f"worker-{i}", _worker_config(ports))
            w.start()
            workers.append(w)
            assert w.transport is not None
            ch: BufferedChannel[Any, Any] = BufferedChannel(w.transport)
            channels.append(ch)

        _wait_for_zmq_connect(0.3)

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
