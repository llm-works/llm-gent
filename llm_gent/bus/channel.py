# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""ZMQ transport adapters for appinfra's channel system.

Implements appinfra's Transport protocol over ZMQ sockets, enabling
appinfra's BufferedChannel to handle request/response correlation.

Two transports:
- ZMQRouterTransport: hub-side, routes through ROUTER socket to a specific agent
- ZMQDealerTransport: agent-side, communicates via DEALER socket to hub
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import zmq
from appinfra.service import ChannelClosedError, ChannelTimeoutError

from .protocol import Envelope


if TYPE_CHECKING:
    pass


class ZMQRouterTransport:
    """Hub-side transport routing through a shared ROUTER socket to one agent.

    Implements appinfra's Transport protocol. Multiple ZMQRouterTransports
    share the same ROUTER socket, each routing to a different agent by ID.

    Messages are serialized as Envelope JSON and sent as ZMQ multipart
    frames: [agent_id, empty, payload].
    """

    def __init__(
        self,
        router: zmq.Socket[Any],
        agent_id: str,
        send_lock: threading.Lock | None = None,
    ) -> None:
        self._router = router
        self._agent_id = agent_id
        self._agent_id_bytes = agent_id.encode()
        self._send_lock = send_lock
        self._closed = False
        # Inbound queue: poll thread feeds messages addressed to this agent
        self._inbound: list[Any] = []
        self._inbound_event = threading.Event()
        self._lock = threading.Lock()

    def send(self, message: Any) -> None:
        """Send message to the agent via ROUTER socket."""
        if self._closed:
            raise ChannelClosedError("transport is closed")
        envelope = message.to_envelope() if hasattr(message, "to_envelope") else message
        if isinstance(envelope, Envelope):
            data = envelope.to_bytes()
        else:
            # Raw message -- wrap in envelope
            from .protocol import Message as BusMessage

            if isinstance(envelope, BusMessage):
                data = envelope.to_envelope().to_bytes()
            else:
                raise TypeError(f"cannot send {type(envelope)}")
        if self._send_lock is not None:
            with self._send_lock:
                self._router.send_multipart([self._agent_id_bytes, b"", data])
        else:
            self._router.send_multipart([self._agent_id_bytes, b"", data])

    def recv(self, timeout: float | None = None) -> Any:
        """Receive next message addressed to this agent.

        Called by BufferedChannel's correlation loop.
        """
        if self._closed:
            raise ChannelClosedError("transport is closed")

        # Wait for a message to appear in our inbound queue
        effective_timeout = timeout if timeout is not None else 30.0
        if not self._inbound_event.wait(timeout=effective_timeout):
            raise ChannelTimeoutError(f"recv timeout after {effective_timeout}s")

        with self._lock:
            if self._closed:
                raise ChannelClosedError("transport is closed")
            if not self._inbound:
                raise ChannelTimeoutError("no message available")
            msg = self._inbound.pop(0)
            if not self._inbound:
                self._inbound_event.clear()
            return msg

    def deliver(self, message: Any) -> None:
        """Deliver a message to this transport's inbound queue.

        Called by the coordinator bus poll thread when a message arrives
        from this agent on the ROUTER socket.
        """
        with self._lock:
            self._inbound.append(message)
            self._inbound_event.set()

    def close(self) -> None:
        """Mark transport as closed."""
        self._closed = True
        self._inbound_event.set()  # unblock any waiting recv

    @property
    def is_closed(self) -> bool:
        return self._closed


class ZMQDealerTransport:
    """Agent-side transport communicating via DEALER socket to hub.

    Implements appinfra's Transport protocol. Wraps a DEALER socket
    for sending/receiving messages with the coordinator's ROUTER.
    """

    def __init__(self, dealer: zmq.Socket[Any], agent_id: str) -> None:
        self._dealer = dealer
        self._agent_id = agent_id
        self._closed = False
        self._inbound: list[Any] = []
        self._inbound_event = threading.Event()
        self._lock = threading.Lock()

    def send(self, message: Any) -> None:
        """Send message to coordinator via DEALER socket."""
        if self._closed:
            raise ChannelClosedError("transport is closed")
        envelope = message.to_envelope() if hasattr(message, "to_envelope") else message
        if isinstance(envelope, Envelope):
            envelope.source = self._agent_id
            data = envelope.to_bytes()
        else:
            from .protocol import Message as BusMessage

            if isinstance(envelope, BusMessage):
                env = envelope.to_envelope()
                env.source = self._agent_id
                data = env.to_bytes()
            else:
                raise TypeError(f"cannot send {type(envelope)}")
        self._dealer.send_multipart([b"", data])

    def recv(self, timeout: float | None = None) -> Any:
        """Receive next message from coordinator."""
        if self._closed:
            raise ChannelClosedError("transport is closed")

        effective_timeout = timeout if timeout is not None else 30.0
        if not self._inbound_event.wait(timeout=effective_timeout):
            raise ChannelTimeoutError(f"recv timeout after {effective_timeout}s")

        with self._lock:
            if self._closed:
                raise ChannelClosedError("transport is closed")
            if not self._inbound:
                raise ChannelTimeoutError("no message available")
            msg = self._inbound.pop(0)
            if not self._inbound:
                self._inbound_event.clear()
            return msg

    def deliver(self, message: Any) -> None:
        """Deliver a message to this transport's inbound queue."""
        with self._lock:
            self._inbound.append(message)
            self._inbound_event.set()

    def close(self) -> None:
        """Mark transport as closed."""
        self._closed = True
        self._inbound_event.set()

    @property
    def is_closed(self) -> bool:
        return self._closed
