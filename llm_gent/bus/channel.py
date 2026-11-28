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


#: Longest a DEALER send waits for room before it checks whether the transport closed.
_SEND_SLICE = 0.1


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
        self._late_replies_dropped = 0
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
        """Route a message from this agent; called by the coordinator bus poll thread.

        Only the agent's replies arrive here (its requests go to the bus's
        request handler), and the hub never reads the correlator's inbox. A
        reply no call waits on, one whose call timed out, is dropped rather
        than kept for the life of the transport.
        """
        self.correlator.deliver(message)
        self._drop_unclaimed()

    def _drop_unclaimed(self) -> None:
        """Empty the inbox: a call's route is registered before its request is sent, so
        whatever reached the inbox has no call waiting on it."""
        while True:
            try:
                self.correlator.recv(timeout=0)
            except (TimeoutError, ChanClosedError):
                return
            self._late_replies_dropped += 1

    @property
    def late_replies_dropped(self) -> int:
        """How many replies arrived after their call stopped waiting."""
        return self._late_replies_dropped

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

        DEALER has no room once its high-water mark is reached. ``None`` waits
        for room; otherwise a full socket raises ``TimeoutError`` after
        ``timeout`` seconds, and ``0`` tries once. Closing the transport ends
        the wait with ``ChanClosedError``.
        """
        if self._closed:
            raise ChanClosedError("transport is closed")
        envelope = _envelope(message)
        envelope.source = self._agent_id
        self._send_until(
            [b"", envelope.to_bytes()], None if timeout is None else time.monotonic() + timeout
        )

    def _send_until(self, frames: list[bytes], deadline: float | None) -> None:
        """Send without blocking, retrying in short slices until there is room, ``deadline``
        passes or the transport closes.

        Sleeps rather than polling for POLLOUT: the agent's poll thread already
        polls this socket.
        """
        while True:
            try:
                self._dealer.send_multipart(frames, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise TimeoutError("dealer socket full") from None
                time.sleep(_SEND_SLICE if left is None else min(_SEND_SLICE, left))
            if self._closed:
                raise ChanClosedError("transport closed while waiting for room")

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
