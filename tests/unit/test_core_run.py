# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The generic Run: a handle on any execution in a task, Flow or not."""

from __future__ import annotations

import asyncio

import pytest

from llm_gent.core import Environment, Run
from tests.helpers import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


def _run() -> Run:
    return Run(lg=LG, env=Environment())


class TestAwait:
    async def test_awaiting_gives_the_task_s_result(self) -> None:
        async def work() -> int:
            return 7

        run = _run()
        run.task = asyncio.create_task(work())
        assert await run == 7
        assert run.done

    async def test_awaiting_raises_what_the_task_raised(self) -> None:
        async def work() -> None:
            raise ValueError("boom")

        run = _run()
        run.task = asyncio.create_task(work())
        with pytest.raises(ValueError, match="boom"):
            await run
        assert run.done

    async def test_a_run_without_a_task_has_not_started(self) -> None:
        run = _run()
        assert not run.done
        with pytest.raises(RuntimeError, match="has not started"):
            await run

    async def test_a_running_task_is_not_done(self) -> None:
        release = asyncio.Event()

        async def work() -> None:
            await release.wait()

        run = _run()
        run.task = asyncio.create_task(work())
        await asyncio.sleep(0)
        assert not run.done
        release.set()
        await run
        assert run.done


class TestHalt:
    async def test_work_observing_the_halt_event_stops_when_halted(self) -> None:
        run = _run()
        started = asyncio.Event()

        async def work() -> str:
            started.set()
            while run.halt_event is None or not run.halt_event.is_set():
                await asyncio.sleep(0)
            return "halted"

        run.task = asyncio.create_task(work())
        await started.wait()
        run.halt()
        assert await run == "halted"

    def test_halt_sets_the_event_it_was_given(self) -> None:
        event = asyncio.Event()
        run = Run(lg=LG, env=Environment(), halt_event=event)
        run.halt()
        assert run.halt_event is event and event.is_set()

    def test_halt_without_an_event_creates_one_set(self) -> None:
        run = _run()
        run.halt()
        assert run.halt_event is not None and run.halt_event.is_set()
