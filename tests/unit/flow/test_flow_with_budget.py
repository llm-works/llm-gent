# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for Flow.with_budget: ctx.budget propagation and subflow inheritance."""

from __future__ import annotations

import asyncio

import pytest

from llm_gent.core.budget import FixedOp, PricingConfig, Tracker
from llm_gent.flow import Context, Flow, verb

from .conftest import ROLE_A, StubFactory, make_test_logger


pytestmark = pytest.mark.unit


def _tracker(*, budget: float = 1.0, halt: asyncio.Event | None = None) -> Tracker:
    pricing = PricingConfig(ops={"web_search": FixedOp(name="web_search", unit_cost=0.001)})
    return Tracker(make_test_logger(), pricing, budget=budget, halt=halt)


class TestContextBudgetPropagation:
    """ctx.budget arrives at verbs when Flow.with_budget was called."""

    @pytest.mark.asyncio
    async def test_verb_sees_tracker(self) -> None:
        tracker = _tracker()
        captured: dict[str, Tracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["budget"] = ctx.budget
            return "ok"

        flow = Flow(make_test_logger(), saia_f=StubFactory()).call(peek).with_budget(tracker)
        await flow.run()
        assert captured["budget"] is tracker

    @pytest.mark.asyncio
    async def test_no_budget_gives_none(self) -> None:
        captured: dict[str, Tracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["budget"] = ctx.budget
            return "ok"

        flow = Flow(make_test_logger(), saia_f=StubFactory()).call(peek)
        await flow.run()
        assert captured["budget"] is None

    @pytest.mark.asyncio
    async def test_dispatch_forwards_default(self) -> None:
        tracker = _tracker()
        captured: dict[str, Tracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["budget"] = ctx.budget
            return "ok"

        flow = Flow(make_test_logger(), saia_f=StubFactory()).with_budget(tracker)
        flow.register(peek)
        await flow.dispatch("peek")
        assert captured["budget"] is tracker


class TestSubflowInheritance:
    """Subflow inherits outer budget unless it overrides via its own .with_budget."""

    @pytest.mark.asyncio
    async def test_inherited(self) -> None:
        outer_tracker = _tracker()
        captured: dict[str, Tracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["budget"] = ctx.budget
            return "ok"

        inner = Flow(make_test_logger(), saia_f=StubFactory(), name="inner").call(peek)
        outer = (
            Flow(make_test_logger(), saia_f=StubFactory(), name="outer")
            .call(inner)
            .with_budget(outer_tracker)
        )
        await outer.run()
        assert captured["budget"] is outer_tracker

    @pytest.mark.asyncio
    async def test_override(self) -> None:
        outer_tracker = _tracker()
        inner_tracker = _tracker()
        captured: dict[str, Tracker | None] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Context) -> str:
            captured["budget"] = ctx.budget
            return "ok"

        inner = (
            Flow(make_test_logger(), saia_f=StubFactory(), name="inner")
            .call(peek)
            .with_budget(inner_tracker)
        )
        outer = (
            Flow(make_test_logger(), saia_f=StubFactory(), name="outer")
            .call(inner)
            .with_budget(outer_tracker)
        )
        await outer.run()
        assert captured["budget"] is inner_tracker


class TestEndToEndHalt:
    """Cap trip on a shared halt event stops iterate() at the next boundary."""

    @pytest.mark.asyncio
    async def test_iterate_halts_when_budget_trips(self) -> None:
        halt = asyncio.Event()
        tracker = _tracker(budget=0.002, halt=halt)
        iterations: list[int] = []

        @verb(role=ROLE_A)
        async def burn(ctx: Context, prev: object = None) -> None:
            ctx.budget.track("web_search", count=3)
            iterations.append(len(iterations) + 1)

        body = Flow(make_test_logger(), saia_f=StubFactory(), name="body").call(burn)
        flow = (
            Flow(make_test_logger(), saia_f=StubFactory())
            .iterate(body, max_iters=10)
            .with_halt(halt)
            .with_budget(tracker)
        )
        await flow.run()
        assert tracker.exceeded is True
        assert halt.is_set()
        assert len(iterations) < 10

    @pytest.mark.asyncio
    async def test_nested_scope_via_ctx_budget_child(self) -> None:
        halt = asyncio.Event()
        root = _tracker(budget=10.0, halt=halt)
        scope_halt = asyncio.Event()

        @verb(role=ROLE_A)
        async def scoped_work(ctx: Context) -> None:
            scope = ctx.budget.child(budget=0.002, halt=scope_halt)
            scope.track("web_search", count=3)

        flow = Flow(make_test_logger(), saia_f=StubFactory()).call(scoped_work).with_budget(root)
        await flow.run()
        assert scope_halt.is_set()
        assert not halt.is_set()
        assert root.spent == pytest.approx(0.003)
