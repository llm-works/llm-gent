# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the ZMQ transports' send timeout, sent through their correlators."""

import time
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import zmq
from appinfra.runtime import ChanClosedError

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
        """No peer means no room: the send waits ``timeout`` and raises TimeoutError."""
        transport = ZMQDealerTransport(ctx.socket(zmq.DEALER), "agent-1")
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            transport.send(Envelope(msg_type="test"), timeout=0.05)
        assert time.monotonic() - start >= 0.05

    def test_zero_timeout_does_not_wait(self, ctx: zmq.Context[Any]) -> None:
        """``timeout=0`` tries without sleeping."""
        transport = ZMQDealerTransport(ctx.socket(zmq.DEALER), "agent-1")
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            transport.send(Envelope(msg_type="test"), timeout=0)
        assert time.monotonic() - start < 0.05

    def test_correlator_send_reaches_router(self, ctx: zmq.Context[Any]) -> None:
        """The correlator sends through the transport; the envelope arrives stamped with the source."""
        router, dealer = _pair(ctx)
        ZMQDealerTransport(dealer, "agent-1").correlator.send(
            Envelope(msg_type="test", payload={"k": 1})
        )

        identity, _, data = _recv_envelope(router)
        received = Envelope.from_bytes(data)
        assert identity == b"agent-1"
        assert received.source == "agent-1"
        assert received.payload == {"k": 1}


class TestDealerSendWhileFull:
    """A send waiting for room on a full DEALER sees the transport close, and keeps trying."""

    @staticmethod
    def _full_until_closed(transport: ZMQDealerTransport) -> MagicMock:
        """The socket stays full; its first send attempt closes the transport, as close() would."""

        def send(frames: list[bytes], **_: Any) -> None:
            transport.close()
            raise zmq.Again()

        socket: MagicMock = transport._dealer
        socket.send_multipart.side_effect = send
        return socket

    @pytest.mark.parametrize("timeout", [None, 5.0], ids=["no-timeout", "timeout"])
    def test_closing_ends_the_wait(self, timeout: float | None) -> None:
        transport = ZMQDealerTransport(MagicMock(), "agent-1")
        self._full_until_closed(transport)

        with patch("llm_gent.bus.channel.time.sleep") as sleep, pytest.raises(ChanClosedError):
            transport.send(Envelope(msg_type="test"), timeout=timeout)

        assert all(call.args[0] <= 0.1 for call in sleep.call_args_list)

    def test_a_send_that_finds_room_returns(self) -> None:
        socket = MagicMock()
        socket.send_multipart.side_effect = [zmq.Again(), None]
        transport = ZMQDealerTransport(socket, "agent-1")

        with patch("llm_gent.bus.channel.time.sleep") as sleep:
            transport.send(Envelope(msg_type="test"))

        assert socket.send_multipart.call_count == 2
        assert all(c.kwargs["flags"] == zmq.NOBLOCK for c in socket.send_multipart.call_args_list)
        sleep.assert_called_once_with(0.1)


class TestRouterSend:
    """ROUTER never blocks; the timeout is accepted for the Transport protocol."""

    def test_correlator_send_reaches_dealer(self, ctx: zmq.Context[Any]) -> None:
        """The correlator sends through the transport; the envelope reaches the agent's DEALER."""
        router, dealer = _pair(ctx)
        # The ROUTER learns the DEALER's identity from its first message.
        dealer.send_multipart([b"", Envelope(msg_type="hello").to_bytes()])
        _recv_envelope(router)

        ZMQRouterTransport(router, "agent-1").correlator.send(
            Envelope(msg_type="test", payload={"k": 2})
        )

        _, data = _recv_envelope(dealer)
        assert Envelope.from_bytes(data).payload == {"k": 2}
