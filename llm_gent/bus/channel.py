# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""ZMQ transports, each with an appinfra Correlator for request and reply.

Two transports:
- ZMQRouterTransport: hub-side, routes through ROUTER socket to a specific agent
- ZMQDealerTransport: agent-side, communicates via DEALER socket to hub

The bus poll thread hands each message it reads for a transport to
``deliver()``; the transport's ``correlator`` routes a reply to the
``submit()`` / ``stream()`` waiting on its id, and any other message to
``correlator.recv()``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable
from typing import Any

import zmq
from appinfra.runtime import ChanClosedError, Correlator

from .protocol import Envelope
from .protocol import Message as BusMessage


def _key(message: Any) -> Hashable:
    """A reply carries the id of its request."""
    return getattr(message, "id", None)


def _failed(message: Any) -> bool:
    """A reply that sets ``error`` reports a failure."""
    return bool(getattr(message, "error", None))


def _final(message: Any) -> bool:
    """A chunk without ``is_final`` ends its stream."""
    return bool(getattr(message, "is_final", True))


def _correlator(send: Callable[[Any, float | None], None]) -> Correlator[Any]:
    return Correlator(send, key=_key, failed=_failed, final=_final)


def _envelope(message: Any) -> Envelope:
    """The envelope ``message`` travels in: an Envelope as is, a bus Message wrapped."""
    if isinstance(message, Envelope):
        return message
    if isinstance(message, BusMessage):
        return message.to_envelope()
    raise TypeError(f"cannot send {type(message)}")


class ZMQRouterTransport:
    """Hub-side transport routing through a shared ROUTER socket to one agent.

    Multiple ZMQRouterTransports share the same ROUTER socket, each routing
    to a different agent by ID. Messages are serialized as Envelope JSON and
    sent as ZMQ multipart frames: [agent_id, empty, payload].
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
        self.correlator = _correlator(self.send)

    def send(self, message: Any, timeout: float | None = None) -> None:
        """Send message to the agent via ROUTER socket.

        ROUTER drops a message at its high-water mark instead of blocking, so
        the send never waits and ``timeout`` has nothing to bound.
        """
        if self._closed:
            raise ChanClosedError("transport is closed")
        frames = [self._agent_id_bytes, b"", _envelope(message).to_bytes()]
        if self._send_lock is not None:
            with self._send_lock:
                self._router.send_multipart(frames)
        else:
            self._router.send_multipart(frames)

    def deliver(self, message: Any) -> None:
        """Route a message from this agent; called by the coordinator bus poll thread."""
        self.correlator.deliver(message)

    def close(self) -> None:
        """Fail the calls waiting on replies; the ROUTER socket stays the bus's."""
        self._closed = True
        self.correlator.close()

    @property
    def is_closed(self) -> bool:
        return self._closed


class ZMQDealerTransport:
    """Agent-side transport communicating via DEALER socket to hub."""

    def __init__(self, dealer: zmq.Socket[Any], agent_id: str) -> None:
        self._dealer = dealer
        self._agent_id = agent_id
        self._closed = False
        self.correlator = _correlator(self.send)

    def send(self, message: Any, timeout: float | None = None) -> None:
        """Send message to coordinator via DEALER socket, stamped with this agent as source.

        DEALER blocks once its high-water mark is reached. ``None`` waits for
        room; otherwise a full socket raises ``TimeoutError`` after
        ``timeout`` seconds.
        """
        if self._closed:
            raise ChanClosedError("transport is closed")
        envelope = _envelope(message)
        envelope.source = self._agent_id
        frames = [b"", envelope.to_bytes()]
        if timeout is None:
            self._dealer.send_multipart(frames)
        else:
            self._send_within(frames, timeout)

    def _send_within(self, frames: list[bytes], timeout: float) -> None:
        """Send without blocking; on a full socket wait ``timeout`` once, then give up.

        Sleeps rather than polling for POLLOUT: the agent's poll thread already
        polls this socket.
        """
        for attempt in range(2):
            try:
                self._dealer.send_multipart(frames, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                if attempt == 0 and timeout > 0:
                    time.sleep(timeout)
        raise TimeoutError(f"dealer socket full for {timeout}s")

    def deliver(self, message: Any) -> None:
        """Route a message from the coordinator; called by the worker bus poll thread."""
        self.correlator.deliver(message)

    def close(self) -> None:
        """Fail the calls waiting on replies; the DEALER socket stays the bus's."""
        self._closed = True
        self.correlator.close()

    @property
    def is_closed(self) -> bool:
        return self._closed
