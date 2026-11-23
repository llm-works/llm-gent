# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost tracking in a run — ``start(cost=)`` / ``with_budget``, resume.

A run started with ``cost=tracker`` tracks its cost on a tracker of its
own, a child of ``tracker``: spend rolls up, the run's accounting is the
run's. A ``with_budget(limit)`` flow runs each of its runs (a map item, an
iterate pass, a ``.call``) on a child of the enclosing tracker with that
budget; crossing it latches ``exceeded`` / ``urgent_wrapup`` and the run
carries on — the agent decides. The run's tracker is saved with the
top-level flow's resources (through the completion commit) and a resumed
run restores it into its own new tracker, never into the one it was
started with; a budgeted run's child is saved while that run is in
progress. "A fresh process" here is a new Flow and a new CostTracker over
the same store.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_gent.core import Role
from llm_gent.core.cost import PricingConfig
from llm_gent.core.cost import Tracker as CostTracker
from llm_gent.flow import (
    COST,
    HALTED,
    Context,
    Factory,
    FlowRunner,
    History,
    Interrupted,
    Loop,
    ResourceKey,
    Resume,
    verb,
)
from llm_gent.flow.state.snapshot import RESOURCES
from tests.helpers import TestFactory, memory_store, run_flow

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "budget"
ROLE = Role(name="r", backend="openai", model="none")


def _tracker(cap: float | None = None, halt: asyncio.Event | None = None) -> CostTracker:
    return CostTracker(make_test_logger(), PricingConfig(), cap, halt=halt)


def _spend(ctx: Context[Any], cost: float, op: str = "llm") -> None:
    assert ctx.cost is not None
    ctx.cost.track(op, override_cost=cost)


def _ff(store: Any = None) -> Factory:
    return TestFactory(make_test_logger(), checkpoint_store=store)


class TestRunTracker:
    async def test_the_run_tracks_its_cost_on_a_child_of_the_tracker_it_is_given(self) -> None:
        caller = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.cost)
            _spend(ctx, 0.5)
            return x

        run = FlowRunner(make_test_logger()).start(_ff().create().call(step), 1, cost=caller)
        assert await run == 1
        assert seen == [run.cost]
        assert run.cost is not None and run.cost.parent is caller
        assert run.cost.spent == caller.spent == 0.5

    async def test_each_run_gets_a_tracker_of_its_own(self) -> None:
        caller = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.cost)
            _spend(ctx, x)
            return x

        flow = _ff().create().call(step)
        assert await asyncio.gather(run_flow(flow, 1, cost=caller), run_flow(flow, 2, cost=caller))
        assert seen[0] is not seen[1] and {t.spent for t in seen} == {1.0, 2.0}
        assert caller.spent == 3.0

    async def test_without_cost_the_run_tracks_none(self) -> None:
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.cost)
            return x

        run = FlowRunner(make_test_logger()).start(_ff().create().call(step), 1)
        assert await run == 1
        assert seen == [None] and run.cost is None

    async def test_a_run_under_an_exhausted_caller_sees_exceeded(self) -> None:
        caller = _tracker(1.0)
        caller.track("llm", override_cost=1.0)
        seen: list[tuple[bool, float | None]] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            assert ctx.cost is not None
            seen.append((ctx.cost.exceeded, ctx.cost.remaining))
            return x

        sub = _ff().create().with_budget(5.0).call(step)
        assert await run_flow(_ff().create().call(step).then(sub), 1, cost=caller) == 1
        assert seen == [(True, 0.0), (True, 0.0)]


class TestRunTrackerAcrossResume:
    @staticmethod
    def _flow(store: Any, seen: list[CostTracker]) -> Any:
        """``a`` spends 2 (llm), ``b`` 3 (op) and sets the run's halt, ``c`` 1 (llm)."""

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 2.0)
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 3.0, "op")
            ctx.halt.set()  # type: ignore[union-attr]
            return x

        @verb
        async def c(ctx: Context[Any], x: int) -> int:
            if ctx.halt is not None and ctx.halt.is_set():
                raise Interrupted()
            assert ctx.cost is not None
            seen.append(ctx.cost)
            _spend(ctx, 1.0)
            return x

        return _ff(store).create().with_checkpointer().call(a).then(b).then(c)

    async def test_the_run_s_cost_comes_back_and_the_caller_sees_only_the_new_spend(
        self,
    ) -> None:
        store = memory_store()
        first = _tracker()
        halt = asyncio.Event()
        assert (
            await run_flow(self._flow(store, []), 1, cost=first, halt=halt, history=NAME) is HALTED
        )
        assert first.spent == 5.0

        fresh = _tracker()
        seen: list[CostTracker] = []
        flow = self._flow(store, seen)
        resumed = await run_flow(
            flow, cost=fresh, history=NAME, start=Resume("latest"), halt=asyncio.Event()
        )
        assert resumed == 1
        assert seen[0].parent is fresh
        assert seen[0].spent == 6.0 and seen[0].costs_by_op == {"llm": 3.0, "op": 3.0}
        assert fresh.spent == 1.0 and fresh.costs_by_op == {"llm": 1.0}

    async def test_a_per_session_budget_is_the_session_tracker_s_own(self) -> None:
        """The history's 5.0 so far is the run's; this session's tracker counts only its own."""
        store = memory_store()
        halt = asyncio.Event()
        await run_flow(self._flow(store, []), 1, cost=_tracker(), halt=halt, history=NAME)
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        assert (await history.snapshot(head)).cursors[""][RESOURCES]["cost"]["spent"] == 5.0

        session = _tracker(1.5)
        await run_flow(
            self._flow(store, []),
            cost=session,
            history=NAME,
            start=Resume("latest"),
            halt=asyncio.Event(),
        )
        assert session.spent == 1.0 and not session.exceeded

    async def test_a_hard_stop_is_the_app_s_choice(self) -> None:
        """A tracker built with the run's halt pauses the run when its budget is crossed."""
        store = memory_store()
        halt = asyncio.Event()
        seen: list[float] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 2.0)
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            seen.append(1.0)
            return x

        flow = _ff(store).create().call(a).then(b)
        tracker = _tracker(1.5, halt=halt)
        assert await run_flow(flow, 1, cost=tracker, halt=halt, history=NAME) is HALTED
        assert seen == []
        head = await History(store, NAME).head()
        assert head is not None and head.meta.outcome == "halted"

    async def test_a_finished_run_s_spend_carries_into_the_next_run_of_the_history(
        self,
    ) -> None:
        """The completion commit holds the run's tracker: the next run continues the total."""
        store = memory_store()
        seen: list[CostTracker] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            assert ctx.cost is not None
            seen.append(ctx.cost)
            _spend(ctx, 2.0)
            return x

        flow = _ff(store).create().with_checkpointer().call(a)
        await run_flow(flow, 1, cost=_tracker(), history=NAME)
        history = History(store, NAME)
        assert await history.is_complete()
        head = await history.head()
        assert head is not None
        assert (await history.snapshot(head)).cursors[""][RESOURCES]["cost"]["spent"] == 2.0

        fresh = _tracker()
        await run_flow(flow, 1, cost=fresh, history=NAME, start=Resume("latest"))
        assert seen[1].spent == 4.0 and fresh.spent == 2.0

    async def test_the_run_s_tracker_is_restored_once_where_nested_flows_share_it(self) -> None:
        """Nested flows share the run's tracker: it has one position, at the top.

        Restoring it again where a nested flow starts would overwrite the
        spend recorded since the run's start.
        """
        store = memory_store()
        seen: list[CostTracker] = []

        def build(arm: bool) -> Any:
            ff = _ff(store)

            @verb
            async def a(ctx: Context[Any], x: int) -> int:
                _spend(ctx, 1.0)
                return x

            @verb
            async def b(ctx: Context[Any], x: int) -> int:
                assert ctx.cost is not None
                seen.append(ctx.cost)
                _spend(ctx, 2.0)
                if arm:
                    ctx.halt.set()  # type: ignore[union-attr]
                    raise Interrupted()
                return x

            inner = ff.create().call(b)
            return ff.create().call(a).then(inner)

        halt = asyncio.Event()
        assert await run_flow(build(True), 1, cost=_tracker(), halt=halt, history=NAME) is HALTED
        snapshot = await History(store, NAME).snapshot(await History(store, NAME).head())
        assert [p for p, c in snapshot.cursors.items() if RESOURCES in c] == [""]

        fresh = _tracker()
        resumed = await run_flow(
            build(False), 1, cost=fresh, history=NAME, start=Resume("latest"), halt=asyncio.Event()
        )
        assert resumed == 1
        assert seen[-1].spent == pytest.approx(3.0 + 2.0)  # 3.0 restored, b again: 2.0
        assert fresh.spent == pytest.approx(2.0)


class _Owned(CostTracker):
    """A tracker whose children are of its own class, so the run's tracker and budgets are too."""

    def child(
        self,
        budget: float | None = None,
        *,
        on_cost: Any = None,
        halt: asyncio.Event | None = None,
    ) -> CostTracker:
        return type(self)(self._lg, self._pricing, budget, parent=self, on_cost=on_cost, halt=halt)


class TestTrackerSubclassAcrossResume:
    """What resume does to the run's spend is its tracker's: gent hands back its snapshot()."""

    async def test_a_restore_that_ignores_the_saved_spend_starts_from_its_own(self) -> None:
        """Saved 5.0: ignored by the run's tracker, so the run neither counts nor halts on it."""

        class Ignoring(_Owned):
            def restore(self, data: dict[str, Any]) -> None:
                pass

        store = memory_store()
        flow = TestRunTrackerAcrossResume._flow
        halt = asyncio.Event()
        assert (
            await run_flow(flow(store, []), 1, cost=_tracker(), halt=halt, history=NAME) is HALTED
        )

        halt = asyncio.Event()
        caller = Ignoring(make_test_logger(), PricingConfig(), 4.0, halt=halt)
        seen: list[CostTracker] = []
        resumed = await run_flow(
            flow(store, seen), cost=caller, history=NAME, start=Resume("latest"), halt=halt
        )
        assert resumed == 1
        assert type(seen[0]) is Ignoring and seen[0].spent == 1.0
        assert caller.spent == 1.0 and not halt.is_set()

    async def test_a_child_override_puts_the_run_and_its_budgets_on_the_subclass(self) -> None:
        root = _Owned(make_test_logger(), PricingConfig())
        seen: dict[int, Any] = {}
        costs = {0: [0.2, 0.3], 1: [0.1, 0.1]}
        assert await run_flow(_item_flow(costs, seen), cost=root) == [0, 10]
        assert all(type(t) is _Owned and type(t.parent) is _Owned for t in seen.values())
        assert all(t.parent.parent is root for t in seen.values())

    async def test_a_budgeted_run_s_child_restores_through_the_subclass(self) -> None:
        """Items halted mid-way resume on the subclass's children, through their restore()."""
        restored: list[dict[str, Any]] = []

        class Recording(_Owned):
            def snapshot(self) -> dict[str, Any]:
                is_run = self.parent is not None and self.parent.parent is None
                return {**super().snapshot(), "tag": "run" if is_run else "item"}

            def restore(self, data: dict[str, Any]) -> None:
                restored.append(data)
                super().restore(data)

        def root() -> Recording:
            return Recording(make_test_logger(), PricingConfig())

        store = memory_store()
        costs = {0: [0.3, 0.2], 1: [0.4, 0.1]}
        flow1 = _item_flow(costs, {}, arm=True, store=store)
        halt = asyncio.Event()
        assert await run_flow(flow1, cost=root(), halt=halt, history=NAME) is HALTED

        before: dict[int, float] = {}
        flow2 = _item_flow(costs, {}, before_second=before, store=store)
        resumed = await run_flow(
            flow2, cost=root(), history=NAME, start=Resume("latest"), halt=asyncio.Event()
        )
        assert resumed == [0, 10]
        assert sorted(d["tag"] for d in restored) == ["item", "item", "run"]
        assert before == {0: pytest.approx(0.3), 1: pytest.approx(0.4)}


def _item_flow(
    costs: dict[int, list[float]],
    seen: dict[int, Any],
    *,
    arm: bool = False,
    before_second: dict[int, float] | None = None,
    cap: float = 1.0,
    cooperative: bool = True,
    store: Any = None,
) -> Any:
    """Map over ``costs``' keys; each item runs on its own child (``cap``), two steps.

    Step 1 spends ``costs[item][0]`` and records the item's tracker; with
    ``arm`` the last item to finish it sets the run's halt. Step 2
    stops on the item's halt (unless not ``cooperative``), else records
    the item's spend so far in ``before_second`` and spends ``costs[item][1]``.
    With ``store``, the flow checkpoints its runs' histories there.
    """

    @verb
    async def first(ctx: Context[Any], item: int) -> int:
        seen[item] = ctx.cost
        _spend(ctx, costs[item][0])
        if arm and len(seen) == len(costs):
            ctx.halt.set()  # type: ignore[union-attr]
        return item

    @verb
    async def second(ctx: Context[Any], item: int) -> int:
        await asyncio.sleep(0)  # lets a set run halt reach the item's own halt
        if cooperative and ctx.halt is not None and ctx.halt.is_set():
            raise Interrupted()
        assert ctx.cost is not None
        if before_second is not None:
            before_second[item] = ctx.cost.spent
        _spend(ctx, costs[item][1])
        return item * 10

    flow = _ff(store).create()
    if store is not None:
        flow = flow.with_checkpointer()
    return flow.map(
        lambda b: b.with_budget(cap).call(first).then(second),
        items=lambda _p, _c: sorted(costs),
    )


class TestItemBudgets:
    async def test_each_item_runs_on_its_own_child_and_spend_rolls_up(self) -> None:
        root = _tracker()
        seen: dict[int, Any] = {}
        costs = {0: [0.2, 0.3], 1: [0.1, 0.1]}
        assert await run_flow(_item_flow(costs, seen), cost=root) == [0, 10]
        assert seen[0] is not seen[1]
        assert seen[0].parent is seen[1].parent and seen[0].parent.parent is root
        assert all(t.budget == 1.0 for t in seen.values())
        assert seen[0].spent == pytest.approx(0.5)
        assert root.spent == pytest.approx(0.7)

    async def test_an_item_over_its_budget_carries_on_with_exceeded_latched(self) -> None:
        """Gent tracks; the agent decides: crossing the budget stops nothing."""
        root = _tracker()
        seen: dict[int, Any] = {}
        costs = {0: [0.2, 0.3], 1: [1.5, 9.0], 2: [0.1, 0.1]}
        assert await run_flow(_item_flow(costs, seen), cost=root) == [0, 10, 20]
        assert seen[1].exceeded and seen[1].urgent_wrapup
        assert not seen[0].exceeded
        assert root.spent == pytest.approx(0.5 + 10.5 + 0.2)

    async def test_items_share_the_run_s_tracker_without_a_cap_of_their_own(self) -> None:
        root = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], item: int) -> int:
            seen.append(ctx.cost)
            return item

        flow = _ff().create().map(lambda b: b.call(step), items=lambda _p, _c: [0, 1])
        await run_flow(flow, cost=root)
        assert seen[0] is seen[1] and seen[0].parent is root


class TestItemBudgetsAcrossHalt:
    async def test_budgeted_item_stops_on_the_run_halt_and_resumes_with_its_spend(self) -> None:
        """Each item resumes on a child with its spend; the caller sees only the new spend."""
        store = memory_store()
        costs = {0: [0.3, 0.2], 1: [0.4, 0.1]}

        root1 = _tracker()
        flow1 = _item_flow(costs, {}, arm=True, store=store)
        halt = asyncio.Event()
        assert await run_flow(flow1, cost=root1, halt=halt, history=NAME) is HALTED
        assert root1.spent == pytest.approx(0.7)

        root2 = _tracker()
        seen2: dict[int, Any] = {}
        before: dict[int, float] = {}
        flow2 = _item_flow(costs, seen2, before_second=before, store=store)
        resumed = await run_flow(
            flow2, cost=root2, history=NAME, start=Resume("latest"), halt=asyncio.Event()
        )
        assert resumed == [0, 10]
        assert seen2 == {}  # no first step ran again
        assert before == {0: pytest.approx(0.3), 1: pytest.approx(0.4)}
        assert root2.spent == pytest.approx(0.3)

    async def test_an_item_resumed_over_its_budget_runs_with_exceeded_latched(self) -> None:
        """Item 0 halted at 0.9; resumed under a budget of 0.5 it carries on, exceeded."""
        store = memory_store()
        costs = {0: [0.9, 0.1], 1: [0.1, 0.1]}
        flow1 = _item_flow(costs, {}, arm=True, store=store)
        halt = asyncio.Event()
        assert await run_flow(flow1, cost=_tracker(), halt=halt, history=NAME) is HALTED

        root2 = _tracker()
        before: dict[int, float] = {}
        flow2 = _item_flow(costs, {}, before_second=before, cap=0.5, store=store)
        resumed = await run_flow(
            flow2, cost=root2, history=NAME, start=Resume("latest"), halt=asyncio.Event()
        )
        assert resumed == [0, 10]
        assert before == {0: pytest.approx(0.9), 1: pytest.approx(0.1)}
        assert root2.spent == pytest.approx(0.2)


@dataclass
class _Conv:
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": list(self.messages)}


class _ConvFactory:
    def create(self) -> _Conv:
        return _Conv()

    def create_from_state(self, state: dict[str, Any]) -> _Conv:
        return _Conv(list(state["messages"]))


@dataclass
class _TurnResult:
    paused: bool
    output: str = ""


class _SpendingSAIA:
    """A SAIA turn in two halves, each an LLM call costing ``costs[task]``.

    ``on_executor_ready`` hands it the call's tracker, as an app wires its
    tool executor. After the first half the turn pauses when its
    ``abort_signal`` is set, as a real backend does.
    """

    def __init__(self, costs: dict[str, float]) -> None:
        self.role = ROLE
        self._costs = costs
        self.trackers: dict[str, CostTracker] = {}
        self.paused: list[str] = []

    async def complete(self, task: str, **kwargs: Any) -> _TurnResult:
        self.trackers[task].track("llm", override_cost=self._costs[task])
        await asyncio.sleep(0)
        abort = kwargs.get("abort_signal")
        if abort is not None and abort.is_set():
            self.paused.append(task)
            return _TurnResult(paused=True)
        self.trackers[task].track("llm", override_cost=self._costs[task])
        return _TurnResult(paused=False, output=f"done:{task}")


class TestLoopInABudgetedItem:
    async def test_crossing_the_budget_mid_turn_does_not_pause_the_turn(self) -> None:
        """No stop event: the turn finishes and the agent sees exceeded on its tracker."""
        root = _tracker()
        saia = _SpendingSAIA({"a": 0.3, "b": 1.5})

        @verb
        async def work(ctx: Context[Any], task: str) -> str:
            def hand_over(s: Any, c: Context[Any]) -> None:
                s.trackers[task] = c.cost

            loop = Loop(
                ROLE, saia=saia, conversation_factory=_ConvFactory(), on_executor_ready=hand_over
            )
            result = await loop(ctx, task)
            return str(result.output)

        flow = (
            _ff()
            .create()
            .map(
                lambda b: b.with_budget(1.0).call(work),
                items=lambda _p, _c: ["a", "b"],
                max_concurrency=1,
            )
        )
        assert await run_flow(flow, cost=root) == ["done:a", "done:b"]
        assert saia.paused == []
        assert saia.trackers["b"].exceeded and not saia.trackers["a"].exceeded
        assert root.spent == pytest.approx(0.3 + 0.3 + 1.5 + 1.5)


class TestPassAndCallBudgets:
    async def test_iterate_body_cap_is_per_pass(self) -> None:
        root = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.cost)
            _spend(ctx, 0.5)
            return x + 1

        flow = _ff().create().iterate(lambda b: b.with_budget(1.0).call(step), max_iters=3)
        assert await run_flow(flow, 0, cost=root) == 3
        assert len({id(t) for t in seen}) == 3
        assert all(t.parent.parent is root and t.spent == 0.5 for t in seen)
        assert root.spent == pytest.approx(1.5)

    async def test_call_budget_is_per_call_and_a_crossing_carries_on(self) -> None:
        root = _tracker()
        ran: list[str] = []
        children: list[Any] = []

        @verb
        async def spend(ctx: Context[Any], cost: float) -> float:
            children.append(ctx.cost)
            _spend(ctx, cost)
            return cost

        @verb
        async def check(ctx: Context[Any], cost: float) -> float:
            ran.append("check")
            return cost

        sub = _ff().create().with_budget(1.0).call(spend).then(check)

        @verb
        async def after(ctx: Context[Any], prev: Any) -> Any:
            ran.append(f"after:{prev}")
            return prev

        flow = _ff().create().call(sub).then(after)
        assert await run_flow(flow, 0.5, cost=root) == 0.5
        assert await run_flow(flow, 2.0, cost=root) == 2.0
        assert ran == ["check", "after:0.5", "check", "after:2.0"]
        assert [c.exceeded for c in children] == [False, True]
        assert root.spent == pytest.approx(2.5)

    async def test_the_run_halt_reaches_nested_flows_under_a_budget(self) -> None:
        """The run's halt is every flow's: one set under a budgeted subflow halts the run."""
        halt = asyncio.Event()
        ff = Factory(make_test_logger())
        ran: list[str] = []

        @verb
        async def spend(ctx: Context[Any], cost: float) -> float:
            _spend(ctx, cost)
            halt.set()
            return cost

        @verb
        async def check(ctx: Context[Any], cost: float) -> float:
            ran.append("check")
            return cost

        inner = ff.create().call(spend).then(check)
        sub = ff.create().with_budget(1.0).call(inner)
        flow = ff.create().call(sub)
        assert await run_flow(flow, 0.5, cost=_tracker(), halt=halt) is HALTED
        assert ran == []

    async def test_map_total_through_a_budgeted_enclosing_flow(self) -> None:
        """Items share the map's 1.0; crossing it latches exceeded and every item runs."""
        root = _tracker()
        ran: list[int] = []
        trackers: list[Any] = []

        @verb
        async def item(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            trackers.append(ctx.cost)
            _spend(ctx, 0.4)
            return n

        budgeted_map = (
            _ff()
            .create()
            .with_budget(1.0)
            .map(lambda b: b.call(item), items=lambda _p, _c: [0, 1, 2, 3], max_concurrency=1)
        )
        flow = _ff().create().call(budgeted_map)
        assert await run_flow(flow, cost=root) == [0, 1, 2, 3]
        assert len({id(t) for t in trackers}) == 1 and trackers[0].exceeded
        assert root.spent == pytest.approx(1.6)


class TestCostAcrossShortcut:
    async def test_a_budgeted_region_s_spend_counts_once(self) -> None:
        """Cut after ``a``: nothing runs again, so ``a``'s spend is recorded once."""
        root = _tracker()
        cut = asyncio.Event()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 1.0)
            cut.set()
            await asyncio.sleep(0)
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 9.0)
            return x

        @verb
        async def after(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 0.5)
            return x

        region = _ff().create().with_budget(5.0).with_shortcut("cut").call(a).then(b)
        flow = _ff().create().call(region).then(after)
        assert await run_flow(flow, 1, cost=root, signals={"cut": cut}) == 1
        assert root.spent == pytest.approx(1.5)

    async def test_started_budgeted_items_end_with_their_spend(self) -> None:
        root = _tracker()
        cut = asyncio.Event()
        gate = asyncio.Event()

        @verb
        async def first(ctx: Context[Any], n: int) -> int:
            _spend(ctx, 0.2)
            if n == 1:
                cut.set()
                await asyncio.sleep(0)
                gate.set()
            else:
                await gate.wait()
            return n

        @verb
        async def second(ctx: Context[Any], n: int) -> int:
            _spend(ctx, 5.0)
            return n

        def item(b: Any) -> None:
            b.with_budget(1.0).call(first).then(second)

        region = _ff().create().with_shortcut("cut")
        region.map(item, items=lambda _p, _c: [0, 1], max_concurrency=2)
        flow = _ff().create().call(region)
        assert await run_flow(flow, cost=root, signals={"cut": cut}) == [0, 1]
        assert root.spent == pytest.approx(2 * 0.2)


class TestCostAcrossCrash:
    async def test_a_budgeted_run_resumes_at_its_last_save(self) -> None:
        """Spend after the last save is not in any checkpoint: the redone step records it again."""
        store = memory_store()
        seen: list[float] = []
        run_trackers: list[CostTracker] = []

        def build(fail: bool) -> Any:
            @verb
            async def a(ctx: Context[Any], x: int) -> int:
                _spend(ctx, 1.0)
                return x

            @verb
            async def b(ctx: Context[Any], x: int) -> int:
                assert ctx.cost is not None and ctx.cost.parent is not None
                await ctx.checkpoint()  # the save: b is the step running at it
                seen.append(ctx.cost.spent)
                run_trackers.append(ctx.cost.parent)
                _spend(ctx, 2.0)
                if fail:
                    raise RuntimeError("process died")
                return x

            sub = _ff().create().with_budget(10.0).call(a).then(b)
            return _ff(store).create().with_checkpointer().call(sub)

        with pytest.raises(RuntimeError, match="process died"):
            await run_flow(build(fail=True), 1, cost=_tracker(), history=NAME)

        root2 = _tracker()
        resumed = await run_flow(
            build(fail=False), 1, cost=root2, history=NAME, start=Resume("latest")
        )
        assert resumed == 1
        assert seen == [pytest.approx(1.0), pytest.approx(1.0)]  # b's first spend was lost
        assert run_trackers[-1].spent == pytest.approx(1.0 + 2.0)  # the total at the save, b again
        assert root2.spent == pytest.approx(2.0)


class TestValidation:
    async def test_budget_in_a_run_without_cost_raises_at_run_start(self) -> None:
        ran: list[str] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            ran.append("step")
            return x

        flow = (
            _ff()
            .create()
            .call(step)
            .map(lambda b: b.with_budget(1.0).call(step), items=lambda _p, _c: [1])
        )
        with pytest.raises(RuntimeError, match="the run tracks no cost: start it with"):
            await run_flow(flow, 1)
        assert ran == []

    async def test_a_budget_on_the_top_flow_is_a_child_of_the_run_s_tracker(self) -> None:
        caller = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.cost)
            _spend(ctx, 0.25)
            return x

        flow = _ff().create().with_budget(1.0).call(step)
        assert await run_flow(flow, 1, cost=caller) == 1
        assert seen[0].budget == 1.0 and seen[0].parent.parent is caller
        assert caller.spent == pytest.approx(0.25)

    async def test_a_tracker_on_the_flow_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="a cost tracker is the run's, not the flow's"):
            _ff().create().with_resource(COST, _tracker())

    async def test_a_tracker_on_the_factory_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="a cost tracker is the run's, not the flow's"):
            Factory(make_test_logger()).with_resource(COST, _tracker())

    @staticmethod
    def _flow_with_own_cost_key() -> Any:
        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            return x

        own: ResourceKey[CostTracker] = ResourceKey("cost")
        return _ff().create().with_resource(own, _tracker()).call(step)

    async def test_a_flow_key_named_like_the_run_s_tracker_raises_at_start(self) -> None:
        """Both would be saved under one name in a checkpoint."""
        with pytest.raises(ValueError, match="two resource keys are named 'cost'"):
            FlowRunner(make_test_logger()).start(self._flow_with_own_cost_key(), 1, cost=_tracker())

    async def test_the_name_is_free_in_a_run_without_cost(self) -> None:
        assert await run_flow(self._flow_with_own_cost_key(), 1) == 1

    @pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan")])
    async def test_budget_must_be_finite_and_positive(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite and > 0"):
            _ff().create().with_budget(bad)

    @pytest.mark.parametrize("bad", [True, "1.0", None, _tracker()])
    async def test_budget_must_be_a_number(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="with_budget takes a number"):
            _ff().create().with_budget(bad)

    @pytest.mark.parametrize("bad", [1.0, "tracker"])
    async def test_cost_must_be_a_cost_tracker(self, bad: Any) -> None:
        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            return x

        with pytest.raises(TypeError, match=r"start\(cost=\) takes a CostTracker"):
            await run_flow(_ff().create().call(step), 1, cost=bad)
