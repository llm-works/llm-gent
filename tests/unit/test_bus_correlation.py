# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Each ZMQ transport's correlator matches replies to requests by id."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.runtime import ChanClosedError, ChanResponseError

from llm_gent.bus.channel import ZMQDealerTransport, ZMQRouterTransport
from llm_gent.bus.protocol import (
    MESSAGE_REGISTRY,
    AskProgress,
    AskRequest,
    AskResponse,
    Envelope,
    FeedbackRequest,
    FeedbackResponse,
)


pytestmark = pytest.mark.unit

Transport = ZMQDealerTransport | ZMQRouterTransport


def _dealer() -> ZMQDealerTransport:
    return ZMQDealerTransport(MagicMock(), "agent-1")


def _router() -> ZMQRouterTransport:
    return ZMQRouterTransport(MagicMock(), "agent-1")


def _socket(transport: Transport) -> MagicMock:
    socket: MagicMock = (
        transport._dealer if isinstance(transport, ZMQDealerTransport) else transport._router
    )
    return socket


def _replying(transport: Transport, reply: Callable[[Any], list[Any]]) -> None:
    """The peer answers each sent request at once with ``reply(request)``, in order."""

    def send(frames: list[bytes], **_: Any) -> None:
        request = Envelope.from_bytes(frames[-1]).unwrap(MESSAGE_REGISTRY)
        for message in reply(request):
            transport.deliver(message)

    _socket(transport).send_multipart.side_effect = send


@pytest.fixture(params=[_dealer, _router], ids=["dealer", "router"])
def transport(request: pytest.FixtureRequest) -> Transport:
    make: Callable[[], Transport] = request.param
    return make()


class TestSubmit:
    def test_returns_the_reply_with_the_requests_id(self, transport: Transport) -> None:
        _replying(transport, lambda req: [FeedbackResponse(id=req.id)])
        request = FeedbackRequest(message="good")

        reply = transport.correlator.submit(request, timeout=1.0)

        assert isinstance(reply, FeedbackResponse)
        assert reply.id == request.id

    def test_a_reply_for_another_request_goes_to_recv(self, transport: Transport) -> None:
        stray = FeedbackResponse(id="someone-else")
        _replying(transport, lambda req: [stray, FeedbackResponse(id=req.id)])

        transport.correlator.submit(FeedbackRequest(message="good"), timeout=1.0)

        assert transport.correlator.recv(timeout=0) == stray

    def test_a_reply_with_an_error_raises(self, transport: Transport) -> None:
        _replying(transport, lambda req: [FeedbackResponse(id=req.id, error="agent down")])

        with pytest.raises(ChanResponseError) as raised:
            transport.correlator.submit(FeedbackRequest(message="good"), timeout=1.0)

        assert raised.value.response.error == "agent down"

    def test_no_reply_times_out(self, transport: Transport) -> None:
        with pytest.raises(TimeoutError):
            transport.correlator.submit(FeedbackRequest(message="good"), timeout=0)


class TestStream:
    def test_yields_the_progress_then_the_final_reply(self, transport: Transport) -> None:
        def reply(req: Any) -> list[Any]:
            return [AskProgress(id=req.id, name="think"), AskResponse(id=req.id, response="42")]

        _replying(transport, reply)

        chunks = list(
            transport.correlator.stream(AskRequest(question="q", stream=True), timeout=1.0)
        )

        assert [type(c) for c in chunks] == [AskProgress, AskResponse]
        assert chunks[-1].response == "42"


class TestInbox:
    def test_a_request_from_the_peer_goes_to_recv(self, transport: Transport) -> None:
        request = FeedbackRequest(message="from the peer")

        transport.deliver(request)

        assert transport.correlator.recv(timeout=0) == request

    def test_nothing_delivered_times_out(self, transport: Transport) -> None:
        with pytest.raises(TimeoutError):
            transport.correlator.recv(timeout=0)


class TestClose:
    def test_closing_the_transport_closes_its_correlator(self, transport: Transport) -> None:
        transport.close()

        assert transport.is_closed
        with pytest.raises(ChanClosedError):
            transport.correlator.submit(FeedbackRequest(message="late"), timeout=0)

    def test_a_closed_transport_refuses_to_send(self, transport: Transport) -> None:
        transport.close()

        with pytest.raises(ChanClosedError):
            transport.send(FeedbackRequest(message="late"))
