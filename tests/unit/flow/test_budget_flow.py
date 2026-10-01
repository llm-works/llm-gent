# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Budgets as flow context — ``with_budget(tracker | cap)``, stops on crossing, resume.

A ``with_budget(cap)`` flow runs each of its runs (a map item, an iterate
pass, a ``.call``) on a child of the enclosing tracker; crossing the cap
stops that run, which ends with ``None``. Every run's own tracker is in the
checkpoints taken while it runs and restored when it resumes. "A fresh
process" here is a new Flow and a new Tracker over the same store.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_gent.core.budget import PricingConfig, Tracker
from llm_gent.flow import Context, FlowFactory, History, Interrupted, Loop, Role, verb
from llm_gent.flow.state.snapshot import BUDGET
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "budget"
ROLE = Role(name="r", backend="openai", model="none")


def _tracker(cap: float | None = None, halt: asyncio.Event | None = None) -> Tracker:
    return Tracker(make_test_logger(), PricingConfig(), cap, halt=halt)


def _spend(ctx: Context[Any], cost: float, op: str = "llm") -> None:
    assert ctx.budget is not None
    ctx.budget.track(op, override_cost=cost)


def _ff() -> FlowFactory:
    return FlowFactory(make_test_logger())


class TestRootTrackerAcrossResume:
    @staticmethod
    def _flow(store: Any, tracker: Tracker, halt: asyncio.Event, seen: list[float]) -> Any:
        """``a`` spends 2 (llm), ``b`` 3 (op) and sets the halt, ``c`` 1 (llm)."""

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 2.0)
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 3.0, "op")
            halt.set()
            return x

        @verb
        async def c(ctx: Context[Any], x: int) -> int:
            if ctx.halt is not None and ctx.halt.is_set():
                raise Interrupted()
            assert ctx.budget is not None
            seen.append(ctx.budget.spent)
            _spend(ctx, 1.0)
            return x

        return (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
            .with_halt(halt)
            .with_budget(tracker)
            .call(a)
            .then(b)
            .then(c)
        )

    async def test_spend_comes_back_in_a_fresh_process(self) -> None:
        store = InMemoryCheckpointStore()
        assert await self._flow(store, _tracker(), asyncio.Event(), []).run(1) is None

        fresh = _tracker(10.0)
        seen: list[float] = []
        assert await self._flow(store, fresh, asyncio.Event(), seen).run(resume="latest") == 1
        assert seen == [5.0]
        assert fresh.spent == 6.0
        assert fresh.costs_by_op == {"llm": 3.0, "op": 3.0}

    async def test_cap_crossing_fires_at_the_same_total(self) -> None:
        store = InMemoryCheckpointStore()
        await self._flow(store, _tracker(), asyncio.Event(), []).run(1)

        crossed = asyncio.Event()
        fresh = _tracker(5.5, halt=crossed)
        await self._flow(store, fresh, asyncio.Event(), []).run(resume="latest")
        assert crossed.is_set() and fresh.urgent_wrapup

    async def test_run_that_ran_out_stays_out_until_its_cap_is_raised(self) -> None:
        """Restored spend over the cap fires the tracker's halt: the resumed run halts again."""
        store = InMemoryCheckpointStore()
        await self._flow(store, _tracker(), asyncio.Event(), []).run(1)

        halt = asyncio.Event()
        seen: list[float] = []
        assert (
            await self._flow(store, _tracker(4.0, halt=halt), halt, seen).run(resume="latest")
            is None
        )
        assert seen == []

        assert (
            await self._flow(store, _tracker(10.0), asyncio.Event(), seen).run(resume="latest") == 1
        )
        assert seen == [5.0]

    async def test_finished_history_restores_no_spend(self) -> None:
        """A finished run's final commit holds no tracker: the next session starts as given."""
        store = InMemoryCheckpointStore()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 2.0)
            return x

        def flow(tracker: Tracker) -> Any:
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
                .with_budget(tracker)
                .call(a)
            )

        await flow(_tracker()).run(1)
        head = await History(store, NAME).head()
        assert head is not None
        assert all(
            BUDGET not in c for c in (await History(store, NAME).snapshot(head)).cursors.values()
        )

        fresh = _tracker()
        await flow(fresh).run(1, resume="latest")
        assert fresh.spent == 2.0


def _item_flow(
    root: Tracker,
    halt: asyncio.Event,
    costs: dict[int, list[float]],
    seen: dict[int, Any],
    *,
    arm: bool = False,
    before_second: dict[int, float] | None = None,
    cap: float = 1.0,
    cooperative: bool = True,
) -> Any:
    """Map over ``costs``' keys; each item runs on its own child (``cap``), two steps.

    Step 1 spends ``costs[item][0]`` and records the item's tracker; with
    ``arm`` the last item to finish it sets the run's ``halt``. Step 2
    stops on the item's halt (unless not ``cooperative``), else records
    the item's spend so far in ``before_second`` and spends ``costs[item][1]``.
    """

    @verb
    async def first(ctx: Context[Any], item: int) -> int:
        seen[item] = ctx.budget
        _spend(ctx, costs[item][0])
        if arm and len(seen) == len(costs):
            halt.set()
        return item

    @verb
    async def second(ctx: Context[Any], item: int) -> int:
        await asyncio.sleep(0)  # lets a set run halt reach the item's own halt
        if cooperative and ctx.halt is not None and ctx.halt.is_set():
            raise Interrupted()
        assert ctx.budget is not None
        if before_second is not None:
            before_second[item] = ctx.budget.spent
        _spend(ctx, costs[item][1])
        return item * 10

    return (
        _ff()
        .create(state={})
        .with_halt(halt)
        .with_budget(root)
        .map(
            lambda b: b.with_budget(cap).call(first).then(second),
            items=lambda _p, _c: sorted(costs),
        )
    )


class TestItemBudgets:
    async def test_each_item_runs_on_its_own_child_and_spend_rolls_up(self) -> None:
        root = _tracker()
        seen: dict[int, Any] = {}
        costs = {0: [0.2, 0.3], 1: [0.1, 0.1]}
        assert await _item_flow(root, asyncio.Event(), costs, seen).run() == [0, 10]
        assert seen[0] is not seen[1]
        assert all(t.parent is root and t.budget == 1.0 for t in seen.values())
        assert seen[0].spent == pytest.approx(0.5)
        assert root.spent == pytest.approx(0.7)

    async def test_item_over_its_cap_ends_with_none_and_the_others_complete(self) -> None:
        root = _tracker()
        costs = {0: [0.2, 0.3], 1: [1.5, 9.0], 2: [0.1, 0.1]}
        assert await _item_flow(root, asyncio.Event(), costs, {}).run() == [0, None, 20]
        # Item 1 spent 1.5 and stopped before its second step.
        assert root.spent == pytest.approx(0.5 + 1.5 + 0.2)

    async def test_items_share_the_tracker_without_a_cap_of_their_own(self) -> None:
        root = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], item: int) -> int:
            seen.append(ctx.budget)
            return item

        flow = (
            _ff()
            .create(state={})
            .with_budget(root)
            .map(lambda b: b.call(step), items=lambda _p, _c: [0, 1])
        )
        await flow.run()
        assert seen == [root, root]


class TestItemBudgetsAcrossHalt:
    async def test_capped_item_stops_on_the_run_halt_and_resumes_with_its_spend(self) -> None:
        """The run's halt stops capped items too; each resumes on a child with its spend."""
        store = InMemoryCheckpointStore()
        costs = {0: [0.3, 0.2], 1: [0.4, 0.1]}

        root1 = _tracker()
        flow1 = _item_flow(root1, asyncio.Event(), costs, {}, arm=True)
        assert await flow1.with_checkpoint_store(store, NAME).with_checkpointer().run() is None
        assert root1.spent == pytest.approx(0.7)

        root2 = _tracker()
        seen2: dict[int, Any] = {}
        before: dict[int, float] = {}
        flow2 = _item_flow(root2, asyncio.Event(), costs, seen2, before_second=before)
        assert await flow2.with_checkpoint_store(store, NAME).with_checkpointer().run(
            resume="latest"
        ) == [0, 10]
        assert seen2 == {}  # no first step ran again
        assert before == {0: pytest.approx(0.3), 1: pytest.approx(0.4)}
        assert root2.spent == pytest.approx(1.0)

    async def test_item_resumed_at_or_over_its_cap_runs_no_step(self) -> None:
        """Item 0 halted at 0.9; resumed under a cap of 0.5 it ends with None, item 1 finishes.

        The resumed steps ignore the halt: only the run's check before its
        first step keeps item 0 from working.
        """
        store = InMemoryCheckpointStore()
        costs = {0: [0.9, 0.1], 1: [0.1, 0.1]}
        flow1 = _item_flow(_tracker(), asyncio.Event(), costs, {}, arm=True)
        assert await flow1.with_checkpoint_store(store, NAME).with_checkpointer().run() is None

        root2 = _tracker()
        before: dict[int, float] = {}
        flow2 = _item_flow(
            root2, asyncio.Event(), costs, {}, before_second=before, cap=0.5, cooperative=False
        )
        assert await flow2.with_checkpoint_store(store, NAME).with_checkpointer().run(
            resume="latest"
        ) == [None, 10]
        assert before == {1: pytest.approx(0.1)}  # item 0 ran no step
        assert root2.spent == pytest.approx(1.1)


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
        self.trackers: dict[str, Tracker] = {}
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


class TestLoopInACappedItem:
    async def test_crossing_the_cap_mid_turn_pauses_the_turn_and_ends_the_item(self) -> None:
        """The item's own stop is the Loop's abort_signal: crossing it pauses the turn."""
        root = _tracker()
        saia = _SpendingSAIA({"a": 0.3, "b": 1.5})

        @verb
        async def research(ctx: Context[Any], task: str) -> str:
            def hand_over(s: Any, c: Context[Any]) -> None:
                s.trackers[task] = c.budget

            loop = Loop(
                ROLE, saia=saia, conversation_factory=_ConvFactory(), on_executor_ready=hand_over
            )
            result = await loop(ctx, task)
            return str(result.output)

        flow = (
            _ff()
            .create(state={})
            .with_budget(root)
            .map(
                lambda b: b.with_budget(1.0).call(research),
                items=lambda _p, _c: ["a", "b"],
                max_concurrency=1,
            )
        )
        assert await flow.run() == ["done:a", None]
        assert saia.paused == ["b"]
        assert root.spent == pytest.approx(0.3 + 0.3 + 1.5)


class TestPassAndCallBudgets:
    async def test_iterate_body_cap_is_per_pass(self) -> None:
        root = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.budget)
            _spend(ctx, 0.5)
            return x + 1

        flow = (
            _ff()
            .create(state={})
            .with_budget(root)
            .iterate(lambda b: b.with_budget(1.0).call(step), max_iters=3)
        )
        assert await flow.run(0) == 3
        assert len({id(t) for t in seen}) == 3
        assert all(t.parent is root and t.spent == 0.5 for t in seen)
        assert root.spent == pytest.approx(1.5)

    async def test_call_cap_is_per_call_and_a_crossing_returns_none(self) -> None:
        root = _tracker()
        ran: list[str] = []

        @verb
        async def spend(ctx: Context[Any], cost: float) -> float:
            _spend(ctx, cost)
            return cost

        @verb
        async def check(ctx: Context[Any], cost: float) -> float:
            if ctx.halt is not None and ctx.halt.is_set():
                raise Interrupted()
            ran.append("check")
            return cost

        sub = _ff().create().with_budget(1.0).call(spend).then(check)

        @verb
        async def after(ctx: Context[Any], prev: Any) -> Any:
            ran.append(f"after:{prev}")
            return prev

        flow = _ff().create(state={}).with_budget(root).call(sub).then(after)
        assert await flow.run(0.5) == 0.5
        assert await flow.run(2.0) is None
        assert ran == ["check", "after:0.5", "after:None"]
        assert root.spent == pytest.approx(2.5)

    async def test_map_total_through_a_capped_enclosing_flow(self) -> None:
        """Items share the map's 1.0; once it is gone the map stops and the run carries on."""
        root = _tracker()
        ran: list[int] = []

        @verb
        async def item(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            _spend(ctx, 0.4)
            return n

        capped_map = (
            _ff()
            .create()
            .with_budget(1.0)
            .map(lambda b: b.call(item), items=lambda _p, _c: [0, 1, 2, 3], max_concurrency=1)
        )

        @verb
        async def after(ctx: Context[Any], prev: Any) -> str:
            return f"after:{prev}"

        flow = _ff().create(state={}).with_budget(root).call(capped_map).then(after)
        assert await flow.run() == "after:None"
        assert ran == [0, 1, 2]
        assert root.spent == pytest.approx(1.2)


class TestValidation:
    async def test_cap_without_an_enclosing_tracker_raises_at_run_start(self) -> None:
        ran: list[str] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            ran.append("step")
            return x

        flow = (
            _ff()
            .create(state={})
            .call(step)
            .map(lambda b: b.with_budget(1.0).call(step), items=lambda _p, _c: [1])
        )
        with pytest.raises(RuntimeError, match="no tracker encloses it"):
            await flow.run(1)
        assert ran == []

    @pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan")])
    async def test_cap_must_be_finite_and_positive(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite and > 0"):
            _ff().create().with_budget(bad)

    @pytest.mark.parametrize("bad", [True, "1.0", None])
    async def test_budget_must_be_a_tracker_or_a_number(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="Tracker or a cap"):
            _ff().create().with_budget(bad)
