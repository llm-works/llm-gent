# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""ctx.cost: the run's tracker, its subflows and budgets, and a cap's halt."""

from __future__ import annotations

import asyncio

import pytest

from llm_gent.core.cost import FixedOp, PricingConfig
from llm_gent.core.cost import Tracker as CostTracker
from llm_gent.flow import Context, verb
from tests.helpers import TestFlow, run_flow

from .conftest import ROLE_A, StubFactory, make_test_logger


pytestmark = pytest.mark.unit


def _tracker(*, budget: float = 1.0, halt: asyncio.Event | None = None) -> CostTracker:
    pricing = PricingConfig(ops={"web_search": FixedOp(name="web_search", unit_cost=0.001)})
    return CostTracker(make_test_logger(), pricing, budget=budget, halt=halt)


def _flow(name: str = "") -> TestFlow:
    return TestFlow(make_test_logger(), name, saia_factory=StubFactory())


class TestContextCost:
    """ctx.cost is the run's own tracker, a child of the one the run was started with."""

    @pytest.mark.asyncio
    async def test_verb_sees_the_run_s_tracker(self) -> None:
        tracker = _tracker()
        captured: dict[str, CostTracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["cost"] = ctx.cost
            return "ok"

        await run_flow(_flow().call(peek), cost=tracker)
        cost = captured["cost"]
        assert cost is not None and cost.parent is tracker

    @pytest.mark.asyncio
    async def test_no_cost_gives_none(self) -> None:
        captured: dict[str, CostTracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["cost"] = ctx.cost
            return "ok"

        await run_flow(_flow().call(peek))
        assert captured["cost"] is None

    @pytest.mark.asyncio
    async def test_dispatch_takes_cost(self) -> None:
        tracker = _tracker()
        captured: dict[str, CostTracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["cost"] = ctx.cost
            return "ok"

        flow = _flow()
        flow.register(peek)
        await flow.dispatch("peek", cost=tracker)
        assert captured["cost"] is tracker


class TestSubflows:
    """A subflow shares the run's tracker unless it declares a budget."""

    @pytest.mark.asyncio
    async def test_inherited(self) -> None:
        tracker = _tracker()
        captured: dict[str, CostTracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["cost"] = ctx.cost
            return "ok"

        inner = _flow("inner").call(peek)
        run = await run_flow(_flow("outer").call(inner).call(peek), cost=tracker)
        assert run == "ok"
        cost = captured["cost"]
        assert cost is not None and cost.parent is tracker

    @pytest.mark.asyncio
    async def test_a_budget_is_a_child_of_the_run_s_tracker(self) -> None:
        tracker = _tracker()
        captured: dict[str, CostTracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["cost"] = ctx.cost
            return "ok"

        inner = _flow("inner").with_budget(0.5).call(peek)
        await run_flow(_flow("outer").call(inner), cost=tracker)
        cost = captured["cost"]
        assert cost is not None and cost.budget == 0.5
        assert cost.parent is not None and cost.parent.parent is tracker


class TestEndToEndHalt:
    """Cap trip on a shared halt event stops iterate() at the next boundary."""

    @pytest.mark.asyncio
    async def test_iterate_halts_when_budget_trips(self) -> None:
        halt = asyncio.Event()
        tracker = _tracker(budget=0.002, halt=halt)
        iterations: list[int] = []

        @verb(role=ROLE_A)
        async def burn(ctx: Context, prev: object = None) -> None:
            ctx.cost.track("web_search", count=3)
            iterations.append(len(iterations) + 1)

        body = _flow("body").call(burn)
        await run_flow(_flow().iterate(body, max_iters=10), cost=tracker, halt=halt)
        assert tracker.exceeded is True
        assert halt.is_set()
        assert len(iterations) < 10

    @pytest.mark.asyncio
    async def test_nested_scope_via_ctx_cost_child(self) -> None:
        halt = asyncio.Event()
        root = _tracker(budget=10.0, halt=halt)
        scope_halt = asyncio.Event()

        @verb(role=ROLE_A)
        async def scoped_work(ctx: Context) -> None:
            scope = ctx.cost.child(budget=0.002, halt=scope_halt)
            scope.track("web_search", count=3)

        await run_flow(_flow().call(scoped_work), cost=root)
        assert scope_halt.is_set()
        assert not halt.is_set()
        assert root.spent == pytest.approx(0.003)
