"""ZMQ bus transport for swarm communication.

Three-socket pattern:
    ROUTER/DEALER: Bidirectional messaging between hub and agents
    PUB/SUB:       Hub broadcasts to agents (commands, notifications)
    SUB/PUB:       Agents publish to hub (heartbeats, events, topic messages)

The bus handles raw message routing. Request/response correlation is
delegated to appinfra's BufferedChannel which wraps the ZMQ transports.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import zmq

from .protocol import Envelope, Message, Request, Response


if TYPE_CHECKING:
    from appinfra.log import Logger

    from .channel import ZMQDealerTransport, ZMQRouterTransport


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


# =============================================================================
# Coordinator bus (hub side)
# =============================================================================


class ZMQCoordinatorBus:
    """Hub-side ZMQ bus with three-socket pattern.

    Binds ROUTER, PUB, and SUB sockets. Routes incoming ROUTER messages
    to per-agent transports. Handles PUB/SUB for broadcast and topic messaging.

    For request/response with agents, use create_agent_transport() to get
    an appinfra Transport, then wrap in BufferedChannel for correlation.
    """

    def __init__(self, lg: Logger, config: CoordinatorBusConfig | None = None) -> None:
        self._lg = lg
        self._config = config or CoordinatorBusConfig()
        self._ctx: zmq.Context[Any] | None = None
        self._router: zmq.Socket[Any] | None = None
        self._pub: zmq.Socket[Any] | None = None
        self._sub: zmq.Socket[Any] | None = None

        self._request_handler: RequestHandler | None = None
        self._route_validator: Callable[[str], bool] | None = None
        self._topic_handlers: dict[str, MessageHandler] = {}
        self._agent_transports: dict[str, ZMQRouterTransport] = {}

        self._poll_thread: threading.Thread | None = None
        self._running = False
        self._send_lock = threading.Lock()
        self._pub_lock = threading.Lock()
        self._transports_lock = threading.Lock()
        self._async_pool: ThreadPoolExecutor | None = None
        self._async_request_types: set[str] = set()

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

        self._async_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bus-async")
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
        if self._async_pool is not None:
            self._async_pool.shutdown(wait=False)
            self._async_pool = None

        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

        # Close all agent transports
        with self._transports_lock:
            transports = list(self._agent_transports.values())
            self._agent_transports.clear()
        for transport in transports:
            transport.close()

        for sock in (self._router, self._pub, self._sub):
            if sock is not None:
                sock.close(linger=100)

        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        self._lg.info("coordinator bus stopped")

    # -------------------------------------------------------------------------
    # Agent transports (for appinfra channel integration)
    # -------------------------------------------------------------------------

    def create_agent_transport(self, agent_id: str) -> ZMQRouterTransport:
        """Create a transport for communicating with a specific agent.

        The transport routes through the shared ROUTER socket. Incoming
        messages from this agent are delivered to the transport's inbound
        queue by the poll thread.

        Wrap in appinfra's BufferedChannel for request/response correlation.

        Args:
            agent_id: Target agent's ZMQ identity.

        Returns:
            ZMQRouterTransport bound to this agent.
        """
        from .channel import ZMQRouterTransport

        if self._router is None:
            raise RuntimeError("bus not started")
        transport = ZMQRouterTransport(self._router, agent_id, send_lock=self._send_lock)
        with self._transports_lock:
            self._agent_transports[agent_id] = transport
        return transport

    def remove_agent_transport(self, agent_id: str) -> None:
        """Remove and close an agent transport."""
        with self._transports_lock:
            transport = self._agent_transports.pop(agent_id, None)
        if transport is not None:
            transport.close()

    # -------------------------------------------------------------------------
    # Pub/sub and request handling
    # -------------------------------------------------------------------------

    def on_request(self, handler: RequestHandler) -> None:
        """Register handler for incoming requests not routed to a transport."""
        self._request_handler = handler

    def set_route_validator(self, validator: Callable[[str], bool]) -> None:
        """Set a validator for agent-to-agent routing targets.

        The validator receives a target agent ID and returns True if the
        target is a known agent that can receive messages.
        """
        self._route_validator = validator

    def register_async_request(self, msg_type: str) -> None:
        """Register a request type to be dispatched asynchronously.

        Requests of this type are handled in a thread pool so they don't
        block the poll thread. The response is sent back via the ROUTER
        socket from the pool thread.
        """
        self._async_request_types.add(msg_type)

    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to messages on a topic (from SUB socket)."""
        self._topic_handlers[topic] = handler

    def broadcast(self, message: Message) -> None:
        """Broadcast a message to all agents via PUB socket."""
        envelope = message.to_envelope()
        envelope.source = "hub"
        with self._pub_lock:
            if self._pub is None:
                raise RuntimeError("bus not started")
            self._pub.send_multipart([b"broadcast", envelope.to_bytes()])

    def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a specific topic via PUB socket."""
        envelope = message.to_envelope()
        envelope.source = "hub"
        with self._pub_lock:
            if self._pub is None:
                raise RuntimeError("bus not started")
            self._pub.send_multipart([topic.encode(), envelope.to_bytes()])

    # -------------------------------------------------------------------------
    # Polling
    # -------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread polling ROUTER and SUB sockets."""
        poller = zmq.Poller()
        if self._router is None or self._sub is None:
            return
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
        if self._router is None:
            return
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

        if self._route_to_transport(envelope):
            return

        self._dispatch_request(envelope, identity, sender_id)

    def _route_to_transport(self, envelope: Envelope) -> bool:
        """Route envelope to an agent transport if applicable.

        Checks (in order):
        1. Target agent specified -> agent-to-agent routing
        2. Sender has a registered transport AND message is a response -> response routing

        Agent-initiated requests (msg_type ending with ``_request``) always
        fall through to ``_dispatch_request`` even if the sender has a
        pre-registered transport (e.g. injected agents).

        Returns True if the message was consumed.
        """
        if envelope.target and envelope.target != "hub":
            return self._forward_to_agent(envelope)

        # Response from agent: route to sender's transport.
        # Requests (msg_type ending with _request) must always reach
        # _dispatch_request so the hub can process them.
        if envelope.source and not envelope.msg_type.endswith("_request"):
            with self._transports_lock:
                transport = self._agent_transports.get(envelope.source)
            if transport is not None:
                self._deliver_to_transport(transport, envelope)
                return True

        return False

    def _forward_to_agent(self, envelope: Envelope) -> bool:
        """Forward an envelope to a target agent via ROUTER socket."""
        target = envelope.target
        if target is None or self._router is None:
            self._lg.warning("cannot route agent-to-agent: router not bound")
            return False
        if self._route_validator is not None and not self._route_validator(target):
            self._lg.warning(
                "agent-to-agent route rejected: unknown target",
                extra={"source": envelope.source, "target": target},
            )
            return False
        with self._send_lock:
            self._router.send_multipart([target.encode(), b"", envelope.to_bytes()])
        return True

    def _deliver_to_transport(self, transport: ZMQRouterTransport, envelope: Envelope) -> None:
        """Unwrap envelope and deliver to transport."""
        msg = self._unwrap_envelope(envelope)
        if msg is not None:
            transport.deliver(msg)

    def _dispatch_request(self, envelope: Envelope, identity: bytes, sender_id: str) -> None:
        """Dispatch incoming request to handler and send response.

        Requests whose ``msg_type`` is registered via ``register_async_request``
        are dispatched to a thread pool so they don't block the poll thread.
        """
        if self._request_handler is None:
            return
        try:
            from .protocol import MESSAGE_REGISTRY

            msg = envelope.unwrap(MESSAGE_REGISTRY)
            if not isinstance(msg, Request):
                self._lg.debug(
                    "ignoring non-request message",
                    extra={"sender": sender_id, "type": envelope.msg_type},
                )
                return

            if envelope.msg_type in self._async_request_types and self._async_pool is not None:
                self._async_pool.submit(self._handle_and_respond, msg, identity, sender_id)
            else:
                self._handle_and_respond(msg, identity, sender_id)
        except Exception as e:
            self._lg.warning(
                "error handling request",
                extra={"sender": sender_id, "type": envelope.msg_type, "exception": e},
            )

    def _handle_and_respond(self, msg: Request, identity: bytes, sender_id: str) -> None:
        """Call request handler and send response via ROUTER socket."""
        if self._request_handler is None or self._router is None:
            return
        try:
            response = self._request_handler(msg, sender_id)
            resp_envelope = response.to_envelope()
            resp_envelope.target = sender_id
            data = resp_envelope.to_bytes()
            with self._send_lock:
                self._router.send_multipart([identity, b"", data])
        except Exception as e:
            self._lg.warning(
                "error handling request",
                extra={"sender": sender_id, "type": msg.message_type, "exception": e},
            )

    def _handle_sub(self) -> None:
        """Process incoming message on SUB socket (heartbeats, events)."""
        if self._sub is None:
            return
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

    def _unwrap_envelope(self, envelope: Envelope) -> Message | None:
        """Unwrap envelope to typed message, or None on error."""
        try:
            from .protocol import MESSAGE_REGISTRY

            return envelope.unwrap(MESSAGE_REGISTRY)
        except Exception as e:
            self._lg.warning("failed to unwrap envelope", extra={"exception": e})
            return None


# =============================================================================
# Worker bus (agent side)
# =============================================================================


class ZMQWorkerBus:
    """Agent-side ZMQ bus connecting to the coordinator.

    Connects DEALER, SUB, and PUB sockets. The DEALER socket is wrapped
    as a ZMQDealerTransport for appinfra channel integration. PUB/SUB
    handles heartbeats and topic messaging.
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

        self._dealer_transport: ZMQDealerTransport | None = None
        self._topic_handlers: dict[str, MessageHandler] = {}

        self._poll_thread: threading.Thread | None = None
        self._running = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def transport(self) -> ZMQDealerTransport | None:
        """The DEALER transport for appinfra channel integration."""
        return self._dealer_transport

    def start(self) -> None:
        """Connect sockets and start polling thread."""
        from .channel import ZMQDealerTransport

        cfg = self._config
        self._ctx = zmq.Context()

        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.setsockopt_string(zmq.IDENTITY, self._agent_id)
        self._dealer.connect(cfg.router_addr)

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(cfg.pub_addr)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "broadcast")

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(cfg.sub_addr)

        self._dealer_transport = ZMQDealerTransport(self._dealer, self._agent_id)

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name=f"bus-worker-{self._agent_id}"
        )
        self._poll_thread.start()

        self._lg.info("worker bus started", extra={"agent_id": self._agent_id})

    def stop(self) -> None:
        """Stop polling and close sockets."""
        self._running = False
        if self._dealer_transport is not None:
            self._dealer_transport.close()
            self._dealer_transport = None

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

    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to a topic from the coordinator's PUB socket."""
        self._topic_handlers[topic] = handler
        if self._sub is not None:
            self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a topic (sent to coordinator's SUB socket)."""
        envelope = message.to_envelope()
        envelope.source = self._agent_id
        if self._pub is None:
            raise RuntimeError("bus not started")
        self._pub.send_multipart([topic.encode(), envelope.to_bytes()])

    def send_to_agent(self, target_id: str, message: Message) -> None:
        """Send a message to another agent via the coordinator.

        The coordinator routes the message to the target agent's transport.

        Args:
            target_id: Target agent's ID.
            message: Message to send.
        """
        envelope = message.to_envelope()
        envelope.source = self._agent_id
        envelope.target = target_id
        if self._dealer is None:
            raise RuntimeError("bus not started")
        self._dealer.send_multipart([b"", envelope.to_bytes()])

    def publish_heartbeat(
        self,
        stats: dict[str, Any],
        round_id: str = "",
        request_id: str = "",
    ) -> None:
        """Publish a heartbeat response with agent stats.

        Used to respond to hub-initiated heartbeat broadcasts or to
        proactively report liveness on the heartbeat topic.

        Args:
            stats: Agent statistics (ticks, errors, etc.).
            round_id: Round ID from the hub's HeartbeatRequest (if responding
                to a broadcast).
            request_id: ID of the HeartbeatRequest being responded to.
        """
        from .protocol import AgentStats, HeartbeatResponse

        response = HeartbeatResponse(
            id=request_id,
            agent_id=self._agent_id,
            round_id=round_id,
            stats=AgentStats(**stats),
        )
        self.publish("heartbeat", response)

    # -------------------------------------------------------------------------
    # Polling
    # -------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread polling DEALER and SUB sockets."""
        poller = zmq.Poller()
        if self._dealer is None or self._sub is None:
            return
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
        """Process incoming message on DEALER socket.

        All DEALER messages are delivered to the dealer transport
        for appinfra's BufferedChannel to handle correlation.
        """
        if self._dealer is None:
            return
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

        # Deliver to transport for BufferedChannel correlation
        if self._dealer_transport is not None:
            msg = self._unwrap_envelope(envelope)
            if msg is not None:
                self._dealer_transport.deliver(msg)

    def _handle_sub(self) -> None:
        """Process incoming message on SUB socket (broadcasts)."""
        if self._sub is None:
            return
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

    def _unwrap_envelope(self, envelope: Envelope) -> Message | None:
        """Unwrap envelope to typed message, or None on error."""
        try:
            from .protocol import MESSAGE_REGISTRY

            return envelope.unwrap(MESSAGE_REGISTRY)
        except Exception as e:
            self._lg.warning("failed to unwrap envelope", extra={"exception": e})
            return None
