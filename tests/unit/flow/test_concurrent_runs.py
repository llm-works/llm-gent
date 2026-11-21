# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One Flow, many runs at once: each run keeps its own run state."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import Context, Factory, Flow, verb
from llm_gent.flow.hooks import Hook

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


class _Quiet(Hook):
    """A hook that does nothing; with one attached every run gets a run id."""


def _probe_flow(barrier: asyncio.Barrier, seen: dict[str, list[str | None]]) -> Flow:
    """One step that reads ``ctx.run_id`` before and after both runs reach ``barrier``."""

    @verb
    async def probe(ctx: Context[dict[str, Any]], tag: str) -> None:
        seen[tag].append(ctx.run_id)
        await barrier.wait()
        seen[tag].append(ctx.run_id)

    return Factory(make_test_logger()).create(state={}).call(probe).with_hooks(_Quiet())


class TestConcurrentRuns:
    async def test_each_run_sees_its_own_run_id(self) -> None:
        seen: dict[str, list[str | None]] = {"a": [], "b": []}
        flow = _probe_flow(asyncio.Barrier(2), seen)

        await asyncio.gather(flow.run("a"), flow.run("b"))

        a_before, a_after = seen["a"]
        b_before, b_after = seen["b"]
        assert a_before is not None and b_before is not None
        assert a_before != b_before
        assert a_after == a_before
        assert b_after == b_before
