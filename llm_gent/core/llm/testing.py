# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A scripted model that takes time to answer, for experiments and benchmarks.

llm-infer's :class:`~llm_infer.client.ScriptedBackend` answers at once: its
async calls never await, so callers sharing an event loop run one after
another and never overlap. :class:`DelayedScriptedBackend` awaits a delay
drawn per call before answering, as a real model leaves the loop free while
it works. Many runs are then in flight at once, and calls finish out of
order.

Answers, retries and the SAIA adapter path stay llm-infer's: only the wait
is added. Wrap the backend in an ``LLMClient`` like any other::

    rng_delay = uniform_delay(0.05, 0.5, seed=7)
    client = LLMClient(lg, DelayedScriptedBackend(lg, rng_delay, record=False))
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from collections.abc import AsyncIterator, Callable

from appinfra.log import Logger
from llm_infer.client import ChatRequest, ChatResponse, ResponseHolder, Script, ScriptedBackend


Delay = Callable[[ChatRequest], float]
"""Seconds to wait before answering ``request``; drawn once per call."""


def uniform_delay(low: float, high: float, seed: int) -> Delay:
    """A delay drawn uniformly from ``[low, high]`` seconds, reproducible from ``seed``.

    Draws follow call order, so a run on one event loop with the same
    seed and the same calls sees the same delays.
    """
    if not 0 <= low <= high:
        raise ValueError(f"need 0 <= low <= high, got low={low}, high={high}")
    rng = random.Random(seed)
    return lambda _request: rng.uniform(low, high)


class DelayedScriptedBackend(ScriptedBackend):
    """:class:`~llm_infer.client.ScriptedBackend` that awaits ``delay(request)`` before answering.

    A streamed answer waits before its first chunk. ``record=False`` keeps
    no requests: :attr:`requests` stays empty, so memory does not grow
    with the number of calls. A list script that runs out then reports
    "request 0" in its :class:`~llm_infer.client.ScriptExhaustedError`.
    """

    def __init__(
        self,
        lg: Logger,
        delay: Delay,
        script: Script | None = None,
        *,
        record: bool = True,
        default_model: str = "scripted",
    ) -> None:
        """Initialize the backend.

        Args:
            lg: Logger instance.
            delay: Seconds to wait per call, given the request.
            script: As for ``ScriptedBackend``: list of entries, a callable, or
                None to answer from the request.
            record: Keep every request in :attr:`requests`.
            default_model: Model name reported when the request names none.
        """
        super().__init__(lg, script, default_model=default_model)
        self._delay = delay
        if not record:
            self.requests = deque(maxlen=0)  # type: ignore[assignment]

    async def chat_async(self, request: ChatRequest) -> ChatResponse:
        """Wait the drawn delay, then answer with the next script entry."""
        await asyncio.sleep(self._delay(request))
        return await super().chat_async(request)

    async def chat_stream_async(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> AsyncIterator[str]:
        """Wait the drawn delay, then stream the next script entry's content."""
        await asyncio.sleep(self._delay(request))
        async for chunk in super().chat_stream_async(request, holder):
            yield chunk
