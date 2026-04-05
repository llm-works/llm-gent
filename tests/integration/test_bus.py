"""Integration tests for ZMQ bus communication.

These tests start real ZMQ sockets on localhost and test actual
message passing between coordinator and worker bus instances.
"""

import socket
import threading
import time
from typing import Any

import pytest
from appinfra.service import BufferedChannel, ChannelTimeoutError

from llm_gent.bus.protocol import (
    AgentStats,
    HeartbeatRequest,
    HeartbeatResponse,
    Message,
    RegisterRequest,
    RegisterResponse,
    Response,
    UnregisterRequest,
)
from llm_gent.bus.transport import (
    CoordinatorBusConfig,
    WorkerBusConfig,
    ZMQCoordinatorBus,
    ZMQWorkerBus,
)


pytestmark = pytest.mark.integration


def _reserve_ports(n: int) -> tuple[list[int], list[socket.socket]]:
    """Reserve n ephemeral ports, returning ports and open sockets.

    Callers must close the returned sockets immediately before ZMQ binds
    to minimize the TOCTOU race window between port discovery and use.
    """
    socks = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    return ports, socks


def _wait_for_zmq_connect(seconds: float = 0.2) -> None:
    """Wait for ZMQ async connect/bind to settle.

    ZMQ connect is asynchronous and provides no readiness signal.
    A short sleep is the only reliable way to allow the underlying
    TCP handshake and subscription propagation to complete.
    """
    time.sleep(seconds)


@pytest.fixture
def bus_pair():
    """Create coordinator + worker bus pair.

    No parent transport is created -- agent→hub requests go through the
    coordinator's on_request handler. Worker has a BufferedChannel for
    request/response via its DEALER transport.
    """
    from unittest.mock import MagicMock

    lg = MagicMock()
    ports, socks = _reserve_ports(3)

    coord_config = CoordinatorBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2])
    worker_config = WorkerBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2])

    coord = ZMQCoordinatorBus(lg, coord_config)
    worker = ZMQWorkerBus(lg, "test-worker", worker_config)

    for s in socks:
        s.close()
    coord.start()
    _wait_for_zmq_connect(0.1)
    worker.start()
    _wait_for_zmq_connect(0.2)

    assert worker.transport is not None
    child_channel: BufferedChannel[Any, Any] = BufferedChannel(worker.transport)

    yield coord, worker, child_channel

    child_channel.close()
    worker.stop()
    coord.stop()


class TestRequestResponse:
    """Tests for RPC-style request/response over channels backed by ZMQ."""

    def test_worker_register_with_coordinator(self, bus_pair):
        """Worker sends register request via channel, coordinator responds."""
        coord, worker, child_ch = bus_pair

        def handle_request(req, sender_id):
            if isinstance(req, RegisterRequest):
                return RegisterResponse(id=req.id, agent_id=req.agent_id)
            return Response(id=req.id, success=False, error="unknown")

        coord.on_request(handle_request)

        req = RegisterRequest(agent_id="test-worker", capabilities=["test"])
        resp = child_ch.submit(req, timeout=5.0)

        assert isinstance(resp, RegisterResponse)

    def test_worker_unregister(self, bus_pair):
        """Worker sends unregister via channel."""
        from llm_gent.bus.protocol import UnregisterResponse

        coord, worker, child_ch = bus_pair

        def handle_request(req, sender_id):
            if isinstance(req, UnregisterRequest):
                return UnregisterResponse(id=req.id, agent_id=req.agent_id)
            return RegisterResponse(id=req.id, agent_id="unknown")

        coord.on_request(handle_request)

        req = UnregisterRequest(agent_id="test-worker")
        resp = child_ch.submit(req, timeout=5.0)
        assert resp.success is True

    def test_timeout_when_no_handler(self, bus_pair):
        """Channel submit times out if coordinator has no handler."""
        _, _, child_ch = bus_pair

        req = RegisterRequest(agent_id="test")
        with pytest.raises(ChannelTimeoutError):
            child_ch.submit(req, timeout=0.5)


class TestPubSub:
    """Tests for publish/subscribe messaging (no channels needed)."""

    def test_coordinator_broadcasts_to_worker(self, bus_pair):
        """Coordinator broadcasts, worker receives via subscription."""
        coord, worker, _ = bus_pair
        received: list[Message] = []
        event = threading.Event()

        def on_broadcast(msg):
            received.append(msg)
            event.set()

        worker.subscribe("broadcast", on_broadcast)
        _wait_for_zmq_connect(0.1)

        coord.broadcast(HeartbeatResponse(id="bc", agent_id="hub"))
        event.wait(timeout=5.0)

        assert len(received) == 1

    def test_worker_heartbeat_received_by_coordinator(self, bus_pair):
        """Worker publishes heartbeat, coordinator receives it."""
        coord, worker, _ = bus_pair
        received: list[Message] = []
        event = threading.Event()

        def on_heartbeat(msg):
            received.append(msg)
            event.set()

        coord.subscribe("heartbeat", on_heartbeat)
        _wait_for_zmq_connect(0.1)

        worker.publish_heartbeat({"ticks": 5, "errors": 0})
        event.wait(timeout=5.0)

        assert len(received) == 1
        assert isinstance(received[0], HeartbeatRequest)
        assert received[0].stats.ticks == 5

    def test_topic_publish_subscribe(self, bus_pair):
        """Worker publishes to custom topic, coordinator receives."""
        coord, worker, _ = bus_pair
        received: list[Message] = []
        event = threading.Event()

        coord.subscribe("intel.news", lambda msg: (received.append(msg), event.set()))
        _wait_for_zmq_connect(0.1)

        msg = HeartbeatRequest(agent_id="news-agent", stats=AgentStats(ticks=1))
        worker.publish("intel.news", msg)
        event.wait(timeout=5.0)

        assert len(received) == 1


class TestMultiWorker:
    """Tests with multiple workers."""

    @pytest.fixture
    def multi_bus(self):
        """Create coordinator + 3 workers with channels."""
        from unittest.mock import MagicMock

        lg = MagicMock()
        ports, socks = _reserve_ports(3)

        coord_config = CoordinatorBusConfig(
            router_port=ports[0], pub_port=ports[1], sub_port=ports[2]
        )
        coord = ZMQCoordinatorBus(lg, coord_config)

        for s in socks:
            s.close()
        coord.start()
        _wait_for_zmq_connect(0.1)

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

        _wait_for_zmq_connect(0.3)

        yield coord, workers, channels

        for ch in channels:
            ch.close()
        for w in workers:
            w.stop()
        coord.stop()

    def test_all_workers_can_register(self, multi_bus):
        """All workers register via their channels."""
        coord, workers, channels = multi_bus
        registered: list[str] = []
        lock = threading.Lock()

        def handle_request(req, sender_id):
            if isinstance(req, RegisterRequest):
                with lock:
                    registered.append(req.agent_id)
                return RegisterResponse(id=req.id, agent_id=req.agent_id)
            return Response(id=req.id)

        coord.on_request(handle_request)

        for w, ch in zip(workers, channels, strict=True):
            req = RegisterRequest(agent_id=w.agent_id, capabilities=["test"])
            resp = ch.submit(req, timeout=5.0)
            assert isinstance(resp, RegisterResponse)

        assert set(registered) == {"worker-0", "worker-1", "worker-2"}

    def test_broadcast_reaches_all_workers(self, multi_bus):
        """Coordinator broadcast received by all workers."""
        coord, workers, _ = multi_bus
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

        _wait_for_zmq_connect(0.1)
        coord.broadcast(HeartbeatResponse(id="bc", agent_id="hub"))
        all_received.wait(timeout=5.0)

        assert len(counts) == 3


class TestAgentToAgent:
    """Tests for agent-to-agent messaging routed through coordinator."""

    @pytest.fixture
    def two_workers(self):
        """Create coordinator + 2 workers with transports registered."""
        from unittest.mock import MagicMock

        lg = MagicMock()
        ports, socks = _reserve_ports(3)

        coord_config = CoordinatorBusConfig(
            router_port=ports[0], pub_port=ports[1], sub_port=ports[2]
        )
        coord = ZMQCoordinatorBus(lg, coord_config)

        for s in socks:
            s.close()
        coord.start()
        _wait_for_zmq_connect(0.1)

        workers = []
        for name in ("alice", "bob"):
            cfg = WorkerBusConfig(router_port=ports[0], pub_port=ports[1], sub_port=ports[2])
            w = ZMQWorkerBus(lg, name, cfg)
            w.start()
            workers.append(w)
            # Register transport so coordinator can route to this agent
            coord.create_agent_transport(name)

        _wait_for_zmq_connect(0.3)

        yield coord, workers[0], workers[1]

        for w in workers:
            w.stop()
        coord.stop()

    def test_agent_sends_to_agent(self, two_workers):
        """Alice sends a message to Bob through the coordinator."""
        coord, alice, bob = two_workers
        received: list[Message] = []

        # Bob's transport delivers to its inbound queue
        assert bob.transport is not None
        bob_channel: BufferedChannel[Any, Any] = BufferedChannel(bob.transport)

        # Alice sends to Bob
        msg = HeartbeatRequest(agent_id="alice", stats=AgentStats(ticks=99))
        alice.send_to_agent("bob", msg)

        # Bob receives via channel
        try:
            received_msg = bob_channel.recv(timeout=5.0)
            received.append(received_msg)
        except Exception:
            pass

        assert len(received) == 1
        assert isinstance(received[0], HeartbeatRequest)
        assert received[0].stats.ticks == 99
