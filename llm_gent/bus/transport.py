"""ZMQ bus transport for swarm communication.

Three-socket pattern:
    ROUTER/DEALER: Bidirectional RPC between hub and agents (agent-to-agent routed via hub)
    PUB/SUB:       Hub broadcasts to agents (commands, notifications)
    SUB/PUB:       Agents publish to hub (heartbeats, events, topic messages)
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import zmq

from .protocol import Envelope, Message, Request, Response


if TYPE_CHECKING:
    from appinfra.log import Logger


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CoordinatorBusConfig:
    """Configuration for the coordinator (hub) side of the bus."""

    router_port: int = 5555
    pub_port: int = 5556
    sub_port: int = 5557
    bind_host: str = "*"
    recv_timeout_ms: int = 100


@dataclass
class WorkerBusConfig:
    """Configuration for the worker (agent) side of the bus."""

    coordinator_host: str = "localhost"
    router_port: int = 5555
    pub_port: int = 5556
    sub_port: int = 5557
    recv_timeout_ms: int = 100

    @property
    def router_addr(self) -> str:
        return f"tcp://{self.coordinator_host}:{self.router_port}"

    @property
    def pub_addr(self) -> str:
        return f"tcp://{self.coordinator_host}:{self.pub_port}"

    @property
    def sub_addr(self) -> str:
        return f"tcp://{self.coordinator_host}:{self.sub_port}"


# =============================================================================
# Exceptions
# =============================================================================


class BusError(Exception):
    """Base exception for bus operations."""


class BusTimeoutError(BusError):
    """Request timed out waiting for response."""


class BusConnectionError(BusError):
    """Failed to connect or bind."""


# =============================================================================
# Bus protocols
# =============================================================================

RequestHandler = Callable[[Request, str | None], Response]
MessageHandler = Callable[[Message], None]


@runtime_checkable
class AgentBus(Protocol):
    """Protocol for agent-side bus operations."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def send(self, message: Request, timeout: float | None = None) -> Response: ...
    def publish(self, topic: str, message: Message) -> None: ...
    def subscribe(self, topic: str, handler: MessageHandler) -> None: ...
    def on_request(self, handler: RequestHandler) -> None: ...


@runtime_checkable
class CoordinatorBus(AgentBus, Protocol):
    """Protocol for coordinator-side bus operations."""

    def send_to_agent(
        self, agent_id: str, message: Request, timeout: float | None = None
    ) -> Response: ...
    def broadcast(self, message: Message) -> None: ...


@runtime_checkable
class WorkerBus(AgentBus, Protocol):
    """Protocol for worker-side bus operations."""

    @property
    def agent_id(self) -> str: ...
    def publish_heartbeat(self, stats: dict[str, Any]) -> None: ...


# =============================================================================
# Coordinator bus (hub side)
# =============================================================================


class ZMQCoordinatorBus:
    """Hub-side ZMQ bus with three-socket pattern.

    Binds ROUTER, PUB, and SUB sockets. Runs a polling thread
    to dispatch incoming messages to registered handlers.
    """

    def __init__(self, lg: Logger, config: CoordinatorBusConfig | None = None) -> None:
        self._lg = lg
        self._config = config or CoordinatorBusConfig()
        self._ctx: zmq.Context[Any] | None = None
        self._router: zmq.Socket[Any] | None = None
        self._pub: zmq.Socket[Any] | None = None
        self._sub: zmq.Socket[Any] | None = None

        self._request_handler: RequestHandler | None = None
        self._topic_handlers: dict[str, MessageHandler] = {}
        self._pending: dict[str, Response | None] = {}
        self._pending_events: dict[str, threading.Event] = {}
        self._pending_lock = threading.RLock()

        self._poll_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Bind sockets and start polling thread."""
        cfg = self._config
        self._ctx = zmq.Context()

        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(f"tcp://{cfg.bind_host}:{cfg.router_port}")

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://{cfg.bind_host}:{cfg.pub_port}")

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.bind(f"tcp://{cfg.bind_host}:{cfg.sub_port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="bus-coordinator-poll"
        )
        self._poll_thread.start()

        self._lg.info(
            "coordinator bus started",
            extra={
                "router_port": cfg.router_port,
                "pub_port": cfg.pub_port,
                "sub_port": cfg.sub_port,
            },
        )

    def stop(self) -> None:
        """Stop polling and close sockets."""
        self._running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

        for sock in (self._router, self._pub, self._sub):
            if sock is not None:
                sock.close(linger=100)

        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        self._lg.info("coordinator bus stopped")

    def on_request(self, handler: RequestHandler) -> None:
        """Register handler for incoming requests from agents."""
        self._request_handler = handler

    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to messages on a topic (from SUB socket)."""
        self._topic_handlers[topic] = handler

    def send_to_agent(
        self, agent_id: str, message: Request, timeout: float | None = None
    ) -> Response:
        """Send request to a specific agent and wait for response.

        Args:
            agent_id: Target agent's ZMQ identity.
            message: Request to send.
            timeout: Seconds to wait (None = 30s default).

        Returns:
            Response from the agent.

        Raises:
            BusTimeoutError: If no response within timeout.
        """
        timeout = timeout if timeout is not None else 30.0
        event = threading.Event()

        with self._pending_lock:
            self._pending[message.id] = None
            self._pending_events[message.id] = event

        envelope = message.to_envelope()
        envelope.target = agent_id
        assert self._router is not None
        self._router.send_multipart(
            [
                agent_id.encode(),
                b"",
                envelope.to_bytes(),
            ]
        )

        if not event.wait(timeout=timeout):
            with self._pending_lock:
                self._pending.pop(message.id, None)
                self._pending_events.pop(message.id, None)
            raise BusTimeoutError(f"timeout waiting for response to {message.id}")

        with self._pending_lock:
            response = self._pending.pop(message.id)
            self._pending_events.pop(message.id, None)

        assert response is not None
        return response

    def broadcast(self, message: Message) -> None:
        """Broadcast a message to all agents via PUB socket."""
        envelope = message.to_envelope()
        envelope.source = "hub"
        assert self._pub is not None
        self._pub.send_multipart([b"broadcast", envelope.to_bytes()])

    def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a specific topic via PUB socket."""
        envelope = message.to_envelope()
        envelope.source = "hub"
        assert self._pub is not None
        self._pub.send_multipart([topic.encode(), envelope.to_bytes()])

    def send(self, message: Request, timeout: float | None = None) -> Response:
        """Not applicable for coordinator -- use send_to_agent()."""
        raise NotImplementedError("coordinator uses send_to_agent()")

    # -------------------------------------------------------------------------
    # Polling
    # -------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread polling ROUTER and SUB sockets."""
        poller = zmq.Poller()
        assert self._router is not None and self._sub is not None
        poller.register(self._router, zmq.POLLIN)
        poller.register(self._sub, zmq.POLLIN)

        while self._running:
            try:
                sockets = dict(poller.poll(timeout=self._config.recv_timeout_ms))
            except zmq.ZMQError:
                if self._running:
                    continue
                break

            if self._router in sockets:
                self._handle_router()
            if self._sub in sockets:
                self._handle_sub()

    def _handle_router(self) -> None:
        """Process incoming message on ROUTER socket."""
        assert self._router is not None
        try:
            frames = self._router.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(frames) < 3:
            return

        identity = frames[0]
        data = frames[2]
        sender_id = identity.decode(errors="replace")

        try:
            envelope = Envelope.from_bytes(data)
            envelope.source = sender_id
        except Exception as e:
            self._lg.warning("failed to parse envelope", extra={"exception": e})
            return

        if self._resolve_pending(envelope):
            return

        self._dispatch_request(envelope, identity, sender_id)

    def _dispatch_request(self, envelope: Envelope, identity: bytes, sender_id: str) -> None:
        """Dispatch incoming request to handler and send response."""
        if self._request_handler is None:
            return
        try:
            from .protocol import MESSAGE_REGISTRY

            msg = envelope.unwrap(MESSAGE_REGISTRY)
            if isinstance(msg, Request):
                response = self._request_handler(msg, sender_id)
                resp_envelope = response.to_envelope()
                resp_envelope.target = sender_id
                assert self._router is not None
                self._router.send_multipart([identity, b"", resp_envelope.to_bytes()])
        except Exception as e:
            self._lg.warning(
                "error handling request",
                extra={"sender": sender_id, "type": envelope.msg_type, "exception": e},
            )

    def _handle_sub(self) -> None:
        """Process incoming message on SUB socket (heartbeats, events)."""
        assert self._sub is not None
        try:
            frames = self._sub.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(frames) < 2:
            return

        topic = frames[0].decode(errors="replace")
        data = frames[1]
        self._dispatch_topic(topic, data)

    def _resolve_pending(self, envelope: Envelope) -> bool:
        """Check if envelope is a response to a pending request.

        Returns:
            True if the envelope was consumed as a pending response.
        """
        with self._pending_lock:
            if envelope.msg_type.endswith("_response") and "id" in envelope.payload:
                req_id = envelope.payload["id"]
                if req_id in self._pending:
                    self._pending[req_id] = Response.model_validate(envelope.payload)
                    self._pending_events[req_id].set()
                    return True
        return False

    def _dispatch_topic(self, topic: str, data: bytes) -> None:
        """Parse envelope and dispatch to topic handler."""
        try:
            envelope = Envelope.from_bytes(data)
        except Exception as e:
            self._lg.warning("failed to parse sub envelope", extra={"exception": e})
            return

        handler = self._topic_handlers.get(topic) or self._topic_handlers.get("")
        if handler is not None:
            try:
                from .protocol import MESSAGE_REGISTRY

                msg = envelope.unwrap(MESSAGE_REGISTRY)
                handler(msg)
            except Exception as e:
                self._lg.warning(
                    "error in topic handler",
                    extra={"topic": topic, "exception": e},
                )


# =============================================================================
# Worker bus (agent side)
# =============================================================================


class ZMQWorkerBus:
    """Agent-side ZMQ bus connecting to the coordinator.

    Connects DEALER, SUB, and PUB sockets to the coordinator's
    bound sockets. Runs a polling thread for incoming messages.
    """

    def __init__(
        self,
        lg: Logger,
        agent_id: str,
        config: WorkerBusConfig | None = None,
    ) -> None:
        self._lg = lg
        self._agent_id = agent_id
        self._config = config or WorkerBusConfig()
        self._ctx: zmq.Context[Any] | None = None
        self._dealer: zmq.Socket[Any] | None = None
        self._sub: zmq.Socket[Any] | None = None
        self._pub: zmq.Socket[Any] | None = None

        self._request_handler: RequestHandler | None = None
        self._topic_handlers: dict[str, MessageHandler] = {}
        self._pending: dict[str, Response | None] = {}
        self._pending_events: dict[str, threading.Event] = {}
        self._pending_lock = threading.RLock()

        self._poll_thread: threading.Thread | None = None
        self._running = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def start(self) -> None:
        """Connect sockets and start polling thread."""
        cfg = self._config
        self._ctx = zmq.Context()

        # DEALER connects to coordinator ROUTER
        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.setsockopt_string(zmq.IDENTITY, self._agent_id)
        self._dealer.connect(cfg.router_addr)

        # SUB connects to coordinator PUB (receive broadcasts)
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(cfg.pub_addr)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "broadcast")

        # PUB connects to coordinator SUB (send heartbeats/events)
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(cfg.sub_addr)

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name=f"bus-worker-{self._agent_id}"
        )
        self._poll_thread.start()

        self._lg.info("worker bus started", extra={"agent_id": self._agent_id})

    def stop(self) -> None:
        """Stop polling and close sockets."""
        self._running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

        for sock in (self._dealer, self._sub, self._pub):
            if sock is not None:
                sock.close(linger=100)

        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        self._lg.info("worker bus stopped", extra={"agent_id": self._agent_id})

    def on_request(self, handler: RequestHandler) -> None:
        """Register handler for incoming requests from coordinator."""
        self._request_handler = handler

    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to a topic from the coordinator's PUB socket."""
        self._topic_handlers[topic] = handler
        if self._sub is not None:
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    def send(self, message: Request, timeout: float | None = None) -> Response:
        """Send request to coordinator and wait for response.

        Args:
            message: Request to send.
            timeout: Seconds to wait (None = 30s default).

        Returns:
            Response from coordinator.

        Raises:
            BusTimeoutError: If no response within timeout.
        """
        timeout = timeout if timeout is not None else 30.0
        event = threading.Event()

        with self._pending_lock:
            self._pending[message.id] = None
            self._pending_events[message.id] = event

        envelope = message.to_envelope()
        envelope.source = self._agent_id
        assert self._dealer is not None
        self._dealer.send_multipart([b"", envelope.to_bytes()])

        if not event.wait(timeout=timeout):
            with self._pending_lock:
                self._pending.pop(message.id, None)
                self._pending_events.pop(message.id, None)
            raise BusTimeoutError(f"timeout waiting for response to {message.id}")

        with self._pending_lock:
            response = self._pending.pop(message.id)
            self._pending_events.pop(message.id, None)

        assert response is not None
        return response

    def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a topic (sent to coordinator's SUB socket)."""
        envelope = message.to_envelope()
        envelope.source = self._agent_id
        assert self._pub is not None
        self._pub.send_multipart([topic.encode(), envelope.to_bytes()])

    def publish_heartbeat(self, stats: dict[str, Any]) -> None:
        """Publish a heartbeat with agent stats."""
        from .protocol import AgentStats, HeartbeatRequest

        request = HeartbeatRequest(
            agent_id=self._agent_id,
            stats=AgentStats(**stats),
        )
        self.publish("heartbeat", request)

    def broadcast(self, message: Message) -> None:
        """Not applicable for workers -- use publish()."""
        raise NotImplementedError("workers use publish(), not broadcast()")

    # -------------------------------------------------------------------------
    # Polling
    # -------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread polling DEALER and SUB sockets."""
        poller = zmq.Poller()
        assert self._dealer is not None and self._sub is not None
        poller.register(self._dealer, zmq.POLLIN)
        poller.register(self._sub, zmq.POLLIN)

        while self._running:
            try:
                sockets = dict(poller.poll(timeout=self._config.recv_timeout_ms))
            except zmq.ZMQError:
                if self._running:
                    continue
                break

            if self._dealer in sockets:
                self._handle_dealer()
            if self._sub in sockets:
                self._handle_sub()

    def _handle_dealer(self) -> None:
        """Process incoming message on DEALER socket."""
        assert self._dealer is not None
        try:
            frames = self._dealer.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(frames) < 2:
            return

        data = frames[-1]

        try:
            envelope = Envelope.from_bytes(data)
        except Exception as e:
            self._lg.warning("failed to parse dealer envelope", extra={"exception": e})
            return

        if self._resolve_pending(envelope):
            return

        self._dispatch_dealer_request(envelope)

    def _dispatch_dealer_request(self, envelope: Envelope) -> None:
        """Dispatch incoming request from coordinator and send response."""
        if self._request_handler is None:
            return
        try:
            from .protocol import MESSAGE_REGISTRY

            msg = envelope.unwrap(MESSAGE_REGISTRY)
            if isinstance(msg, Request):
                response = self._request_handler(msg, envelope.source)
                resp_envelope = response.to_envelope()
                resp_envelope.target = envelope.source
                assert self._dealer is not None
                self._dealer.send_multipart([b"", resp_envelope.to_bytes()])
        except Exception as e:
            self._lg.warning(
                "error handling dealer request",
                extra={"type": envelope.msg_type, "exception": e},
            )

    def _resolve_pending(self, envelope: Envelope) -> bool:
        """Check if envelope is a response to a pending request.

        Returns:
            True if the envelope was consumed as a pending response.
        """
        with self._pending_lock:
            if envelope.msg_type.endswith("_response") and "id" in envelope.payload:
                req_id = envelope.payload["id"]
                if req_id in self._pending:
                    from .protocol import MESSAGE_REGISTRY

                    msg = envelope.unwrap(MESSAGE_REGISTRY)
                    if isinstance(msg, Response):
                        self._pending[req_id] = msg
                    else:
                        self._pending[req_id] = Response(id=req_id, success=True)
                    self._pending_events[req_id].set()
                    return True
        return False

    def _handle_sub(self) -> None:
        """Process incoming message on SUB socket (broadcasts from coordinator)."""
        assert self._sub is not None
        try:
            frames = self._sub.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(frames) < 2:
            return

        topic = frames[0].decode(errors="replace")
        data = frames[1]
        self._dispatch_topic(topic, data)

    def _dispatch_topic(self, topic: str, data: bytes) -> None:
        """Parse envelope and dispatch to topic handler."""
        try:
            envelope = Envelope.from_bytes(data)
        except Exception as e:
            self._lg.warning("failed to parse sub envelope", extra={"exception": e})
            return

        handler = self._topic_handlers.get(topic) or self._topic_handlers.get("")
        if handler is not None:
            try:
                from .protocol import MESSAGE_REGISTRY

                msg = envelope.unwrap(MESSAGE_REGISTRY)
                handler(msg)
            except Exception as e:
                self._lg.warning(
                    "error in topic handler",
                    extra={"topic": topic, "exception": e},
                )
