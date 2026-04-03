"""Integration tests for ZMQ bus communication.

These tests start real ZMQ sockets on localhost and test actual
message passing between coordinator and worker bus instances.
"""

import socket
import threading
import time

import pytest

from llm_gent.bus.protocol import (
    AgentStats,
    HeartbeatRequest,
    HeartbeatResponse,
    Message,
    RegisterRequest,
    RegisterResponse,
    Response,
    UnregisterRequest,
    UnregisterResponse,
)
from llm_gent.bus.transport import (
    BusTimeoutError,
    CoordinatorBusConfig,
    WorkerBusConfig,
    ZMQCoordinatorBus,
    ZMQWorkerBus,
)


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
def bus_pair():
    """Create and start a coordinator + worker bus pair on ephemeral ports."""
    from unittest.mock import MagicMock

    lg = MagicMock()
    ports = _find_free_ports(3)

    coord_config = CoordinatorBusConfig(
        router_port=ports[0],
        pub_port=ports[1],
        sub_port=ports[2],
    )
    worker_config = WorkerBusConfig(
        router_port=ports[0],
        pub_port=ports[1],
        sub_port=ports[2],
    )

    coord = ZMQCoordinatorBus(lg, coord_config)
    worker = ZMQWorkerBus(lg, "test-worker", worker_config)

    coord.start()
    time.sleep(0.1)
    worker.start()
    time.sleep(0.2)

    yield coord, worker

    worker.stop()
    coord.stop()


class TestRequestResponse:
    """Tests for RPC-style request/response over the bus."""

    def test_worker_register_with_coordinator(self, bus_pair):
        """Worker sends register request, coordinator responds."""
        coord, worker = bus_pair

        def handle_request(req, sender_id):
            if isinstance(req, RegisterRequest):
                return RegisterResponse(id=req.id, agent_id=req.agent_id)
            return Response(id=req.id, success=False, error="unknown")

        coord.on_request(handle_request)

        req = RegisterRequest(agent_id="test-worker", capabilities=["test"])
        resp = worker.send(req, timeout=5.0)

        assert resp.success is True

    def test_worker_unregister(self, bus_pair):
        """Worker sends unregister request, coordinator responds."""
        coord, worker = bus_pair

        def handle_request(req, sender_id):
            if isinstance(req, UnregisterRequest):
                return UnregisterResponse(id=req.id, agent_id=req.agent_id)
            return Response(id=req.id)

        coord.on_request(handle_request)

        req = UnregisterRequest(agent_id="test-worker")
        resp = worker.send(req, timeout=5.0)

        assert resp.success is True

    def test_timeout_when_no_handler(self, bus_pair):
        """Worker times out if coordinator has no request handler."""
        _, worker = bus_pair

        req = RegisterRequest(agent_id="test")
        with pytest.raises(BusTimeoutError):
            worker.send(req, timeout=0.5)


class TestPubSub:
    """Tests for publish/subscribe messaging."""

    def test_coordinator_broadcasts_to_worker(self, bus_pair):
        """Coordinator broadcasts, worker receives via subscription."""
        coord, worker = bus_pair
        received: list[Message] = []
        event = threading.Event()

        def on_broadcast(msg):
            received.append(msg)
            event.set()

        worker.subscribe("broadcast", on_broadcast)
        time.sleep(0.1)

        coord.broadcast(HeartbeatResponse(id="bc", agent_id="hub"))
        event.wait(timeout=5.0)

        assert len(received) == 1

    def test_worker_heartbeat_received_by_coordinator(self, bus_pair):
        """Worker publishes heartbeat, coordinator receives it."""
        coord, worker = bus_pair
        received: list[Message] = []
        event = threading.Event()

        def on_heartbeat(msg):
            received.append(msg)
            event.set()

        coord.subscribe("heartbeat", on_heartbeat)
        time.sleep(0.1)

        worker.publish_heartbeat({"ticks": 5, "errors": 0})
        event.wait(timeout=5.0)

        assert len(received) == 1
        assert isinstance(received[0], HeartbeatRequest)
        assert received[0].stats.ticks == 5

    def test_topic_publish_subscribe(self, bus_pair):
        """Worker publishes to custom topic, coordinator receives."""
        coord, worker = bus_pair
        received: list[Message] = []
        event = threading.Event()

        def on_intel(msg):
            received.append(msg)
            event.set()

        coord.subscribe("intel.news", on_intel)
        time.sleep(0.1)

        msg = HeartbeatRequest(agent_id="news-agent", stats=AgentStats(ticks=1))
        worker.publish("intel.news", msg)
        event.wait(timeout=5.0)

        assert len(received) == 1


class TestMultiWorker:
    """Tests with multiple workers connected to one coordinator."""

    @pytest.fixture
    def multi_bus(self):
        """Create coordinator + 3 workers."""
        from unittest.mock import MagicMock

        lg = MagicMock()
        ports = _find_free_ports(3)

        coord_config = CoordinatorBusConfig(
            router_port=ports[0],
            pub_port=ports[1],
            sub_port=ports[2],
        )
        coord = ZMQCoordinatorBus(lg, coord_config)
        coord.start()
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

        yield coord, workers

        for w in workers:
            w.stop()
        coord.stop()

    def test_all_workers_can_register(self, multi_bus):
        """All workers can register independently."""
        coord, workers = multi_bus
        registered: list[str] = []
        lock = threading.Lock()

        def handle_request(req, sender_id):
            if isinstance(req, RegisterRequest):
                with lock:
                    registered.append(req.agent_id)
                return RegisterResponse(id=req.id, agent_id=req.agent_id)
            return Response(id=req.id)

        coord.on_request(handle_request)

        for w in workers:
            req = RegisterRequest(agent_id=w.agent_id, capabilities=["test"])
            resp = w.send(req, timeout=5.0)
            assert resp.success is True

        assert set(registered) == {"worker-0", "worker-1", "worker-2"}

    def test_broadcast_reaches_all_workers(self, multi_bus):
        """Coordinator broadcast is received by all workers."""
        coord, workers = multi_bus
        counts: dict[str, int] = {}
        lock = threading.Lock()
        all_received = threading.Event()

        def make_handler(agent_id):
            def handler(msg):
                with lock:
                    counts[agent_id] = counts.get(agent_id, 0) + 1
                    if len(counts) == 3:
                        all_received.set()

            return handler

        for w in workers:
            w.subscribe("broadcast", make_handler(w.agent_id))

        time.sleep(0.1)
        coord.broadcast(HeartbeatResponse(id="bc", agent_id="hub"))
        all_received.wait(timeout=5.0)

        assert len(counts) == 3

    def test_multiple_heartbeats_from_different_workers(self, multi_bus):
        """Coordinator receives heartbeats from multiple workers."""
        coord, workers = multi_bus
        received: list[str] = []
        lock = threading.Lock()
        all_received = threading.Event()

        def on_heartbeat(msg):
            if isinstance(msg, HeartbeatRequest):
                with lock:
                    received.append(msg.agent_id)
                    if len(received) == 3:
                        all_received.set()

        coord.subscribe("heartbeat", on_heartbeat)
        time.sleep(0.1)

        for w in workers:
            w.publish_heartbeat({"ticks": 1})

        all_received.wait(timeout=5.0)

        assert set(received) == {"worker-0", "worker-1", "worker-2"}
