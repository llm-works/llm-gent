# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""DelayedScriptedBackend: answers like llm-infer's scripted backend, after an awaited delay."""

from __future__ import annotations

import asyncio

import pytest
from llm_infer.client import ChatRequest, ChatResponse, LLMClient

from llm_gent.core.llm.testing import DelayedScriptedBackend, uniform_delay
from tests.helpers import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


def _request(text: str = "hi") -> ChatRequest:
    return ChatRequest(messages=[{"role": "user", "content": text}])


class TestDelay:
    """The delay is drawn per call from the request, and awaited."""

    async def test_each_call_awaits_its_drawn_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        waited: list[float] = []

        async def sleep(seconds: float) -> None:
            waited.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", sleep)
        lengths = DelayedScriptedBackend(LG, lambda r: float(len(r.messages[0]["content"])))
        await lengths.chat_async(_request("abc"))
        await lengths.chat_async(_request("abcde"))
        assert waited == [3.0, 5.0]

    async def test_concurrent_calls_overlap(self) -> None:
        # A zero delay still yields to the loop once: every call is in flight
        # before the first one answers, which the undelayed backend never allows.
        started: list[ChatRequest] = []

        def delay(request: ChatRequest) -> float:
            started.append(request)
            return 0.0

        backend = DelayedScriptedBackend(LG, delay)
        seen: list[int] = []

        async def call() -> None:
            await backend.chat_async(_request())
            seen.append(len(started))

        await asyncio.gather(call(), call(), call())
        assert seen[0] == 3

    async def test_stream_waits_before_the_first_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []

        async def sleep(seconds: float) -> None:
            events.append(f"sleep {seconds}")

        monkeypatch.setattr(asyncio, "sleep", sleep)
        backend = DelayedScriptedBackend(LG, lambda _r: 0.25, [ChatResponse(content="a b")])
        events.extend([chunk async for chunk in backend.chat_stream_async(_request())])
        assert events == ["sleep 0.25", "a ", "b"]


class TestAnswers:
    """Answers and recording stay those of the scripted backend."""

    async def test_answers_through_a_client_from_the_script(self) -> None:
        backend = DelayedScriptedBackend(LG, lambda _r: 0.0, [ChatResponse(content="one")])
        response = await LLMClient(LG, backend).chat_async([{"role": "user", "content": "q"}])
        assert response.content == "one"
        assert len(backend.requests) == 1

    async def test_record_off_keeps_no_requests(self) -> None:
        backend = DelayedScriptedBackend(LG, lambda _r: 0.0, record=False)
        for _ in range(3):
            await backend.chat_async(_request())
        assert len(backend.requests) == 0


class TestUniformDelay:
    def test_same_seed_draws_the_same_delays_within_bounds(self) -> None:
        a, b = uniform_delay(0.1, 0.2, seed=3), uniform_delay(0.1, 0.2, seed=3)
        draws = [a(_request()) for _ in range(20)]
        assert draws == [b(_request()) for _ in range(20)]
        assert all(0.1 <= d <= 0.2 for d in draws)

    @pytest.mark.parametrize(("low", "high"), [(-0.1, 0.2), (0.3, 0.2)])
    def test_rejects_bounds_out_of_order_or_negative(self, low: float, high: float) -> None:
        with pytest.raises(ValueError, match="low <= high"):
            uniform_delay(low, high, seed=0)
