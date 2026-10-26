# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the ZMQ transports' send timeout under appinfra's BufferedChannel."""

import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import zmq
from appinfra.service import BufferedChannel, ChannelFullError

from llm_gent.bus.channel import ZMQDealerTransport, ZMQRouterTransport
from llm_gent.bus.protocol import Envelope


pytestmark = pytest.mark.unit


@pytest.fixture
def ctx() -> Iterator[zmq.Context[Any]]:
    context: zmq.Context[Any] = zmq.Context()
    yield context
    context.destroy(linger=0)


def _pair(ctx: zmq.Context[Any]) -> tuple[zmq.Socket[Any], zmq.Socket[Any]]:
    """Return a bound ROUTER and a DEALER (identity ``agent-1``) connected to it."""
    endpoint = f"inproc://{uuid.uuid4().hex}"
    router = ctx.socket(zmq.ROUTER)
    router.bind(endpoint)
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt_string(zmq.IDENTITY, "agent-1")
    dealer.connect(endpoint)
    return router, dealer


def _recv_envelope(sock: zmq.Socket[Any]) -> list[bytes]:
    assert sock.poll(1000, zmq.POLLIN), "no message arrived"
    frames: list[bytes] = sock.recv_multipart()
    return frames


class TestDealerSend:
    """DEALER blocks at its high-water mark, so a timeout must bound the wait."""

    def test_unconnected_dealer_raises_full_after_timeout(self, ctx: zmq.Context[Any]) -> None:
        """No peer means no room: the send waits ``timeout`` and raises ChannelFullError."""
        transport = ZMQDealerTransport(ctx.socket(zmq.DEALER), "agent-1")
        start = time.monotonic()
        with pytest.raises(ChannelFullError):
            transport.send(Envelope(msg_type="test"), timeout=0.05)
        assert time.monotonic() - start >= 0.05

    def test_zero_timeout_does_not_wait(self, ctx: zmq.Context[Any]) -> None:
        """``timeout=0`` tries without sleeping."""
        transport = ZMQDealerTransport(ctx.socket(zmq.DEALER), "agent-1")
        start = time.monotonic()
        with pytest.raises(ChannelFullError):
            transport.send(Envelope(msg_type="test"), timeout=0)
        assert time.monotonic() - start < 0.05

    def test_buffered_channel_send_reaches_router(self, ctx: zmq.Context[Any]) -> None:
        """BufferedChannel passes a send slice; the envelope arrives stamped with the source."""
        router, dealer = _pair(ctx)
        channel: BufferedChannel[Any, Any] = BufferedChannel(ZMQDealerTransport(dealer, "agent-1"))
        channel.send(Envelope(msg_type="test", payload={"k": 1}))

        identity, _, data = _recv_envelope(router)
        received = Envelope.from_bytes(data)
        assert identity == b"agent-1"
        assert received.source == "agent-1"
        assert received.payload == {"k": 1}


class TestRouterSend:
    """ROUTER never blocks; the timeout is accepted for the Transport protocol."""

    def test_buffered_channel_send_reaches_dealer(self, ctx: zmq.Context[Any]) -> None:
        """BufferedChannel passes a send slice; the envelope reaches the agent's DEALER."""
        router, dealer = _pair(ctx)
        # The ROUTER learns the DEALER's identity from its first message.
        dealer.send_multipart([b"", Envelope(msg_type="hello").to_bytes()])
        _recv_envelope(router)

        channel: BufferedChannel[Any, Any] = BufferedChannel(ZMQRouterTransport(router, "agent-1"))
        channel.send(Envelope(msg_type="test", payload={"k": 2}))

        _, data = _recv_envelope(dealer)
        assert Envelope.from_bytes(data).payload == {"k": 2}
