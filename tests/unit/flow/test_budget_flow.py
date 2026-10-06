# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost tracking as flow context — ``with_cost_tracker`` / ``with_budget``, resume.

A ``with_budget(limit)`` flow runs each of its runs (a map item, an
iterate pass, a ``.call``) on a child of its cost tracker with that budget;
crossing the budget latches the child's ``exceeded`` / ``urgent_wrapup``
and the run carries on — the agent decides. A budgeted run's child is in
the checkpoints taken while it runs and restored when it resumes; the
tracker a flow declares is saved at that flow's path (the top-level one
through the completion commit) and restored on resume. "A fresh process"
here is a new Flow and a new CostTracker over the same store.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_gent.core.cost import CostTracker, PricingConfig
from llm_gent.flow import HALTED, Context, FlowFactory, History, Interrupted, Loop, Role, verb
from llm_gent.flow.state.snapshot import RESOURCES
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

NAME = "budget"
ROLE = Role(name="r", backend="openai", model="none")


def _tracker(cap: float | None = None, halt: asyncio.Event | None = None) -> CostTracker:
    return CostTracker(make_test_logger(), PricingConfig(), cap, halt=halt)


def _spend(ctx: Context[Any], cost: float, op: str = "llm") -> None:
    assert ctx.cost is not None
    ctx.cost.track(op, override_cost=cost)


def _ff() -> FlowFactory:
    return FlowFactory(make_test_logger())


class TestAppTrackerAcrossResume:
    @staticmethod
    def _flow(store: Any, tracker: CostTracker, halt: asyncio.Event, seen: list[float]) -> Any:
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
            assert ctx.cost is not None
            seen.append(ctx.cost.spent)
            _spend(ctx, 1.0)
            return x

        return (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
            .with_halt(halt)
            .with_cost_tracker(tracker)
            .call(a)
            .then(b)
            .then(c)
        )

    async def test_the_running_cost_comes_back_after_a_halt(self) -> None:
        store = InMemoryCheckpointStore()
        assert await self._flow(store, _tracker(), asyncio.Event(), []).run(1) is HALTED

        fresh = _tracker(10.0)
        seen: list[float] = []
        assert await self._flow(store, fresh, asyncio.Event(), seen).run(resume="latest") == 1
        assert seen == [5.0]
        assert fresh.spent == 6.0
        assert fresh.costs_by_op == {"llm": 3.0, "op": 3.0}

    async def test_a_per_session_budget_is_set_on_the_restored_spend(self) -> None:
        """Spend is cumulative; the app reads it from the history and sets this session's limit."""
        store = InMemoryCheckpointStore()
        await self._flow(store, _tracker(), asyncio.Event(), []).run(1)
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        so_far = (await history.snapshot(head)).cursors[""][RESOURCES]["cost"]["spent"]
        assert so_far == 5.0

        session = _tracker(so_far + 1.5)
        await self._flow(store, session, asyncio.Event(), []).run(resume="latest")
        assert session.spent == 6.0 and not session.exceeded

    async def test_a_hard_stop_is_the_app_s_choice(self) -> None:
        """A tracker built with the run's halt pauses the run when its budget is crossed."""
        store = InMemoryCheckpointStore()
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

        flow = (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_halt(halt)
            .with_cost_tracker(_tracker(1.5, halt=halt))
            .call(a)
            .then(b)
        )
        assert await flow.run(1) is HALTED
        assert seen == []
        head = await History(store, NAME).head()
        assert head is not None and head.meta.outcome == "halted"

    async def test_a_finished_run_s_spend_carries_into_the_next_session(self) -> None:
        """The completion commit holds the tracker: the next session continues the total."""
        store = InMemoryCheckpointStore()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 2.0)
            return x

        def flow(tracker: CostTracker) -> Any:
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
                .with_cost_tracker(tracker)
                .call(a)
            )

        await flow(_tracker()).run(1)
        history = History(store, NAME)
        assert await history.is_complete()
        head = await history.head()
        assert head is not None
        assert (await history.snapshot(head)).cursors[""][RESOURCES]["cost"]["spent"] == 2.0

        fresh = _tracker()
        await flow(fresh).run(1, resume="latest")
        assert fresh.spent == 4.0

    async def test_a_tracker_every_flow_inherits_is_restored_once(self) -> None:
        """A FlowFactory(cost_tracker=) puts the same tracker on every flow: one position, at the top.

        Restoring it again where a nested flow starts would overwrite the
        spend recorded since the run's start.
        """
        store = InMemoryCheckpointStore()
        halt = asyncio.Event()

        def build(tracker: CostTracker, arm: bool) -> Any:
            ff = FlowFactory(make_test_logger(), cost_tracker=tracker)

            @verb
            async def a(ctx: Context[Any], x: int) -> int:
                _spend(ctx, 1.0)
                return x

            @verb
            async def b(ctx: Context[Any], x: int) -> int:
                _spend(ctx, 2.0)
                if arm:
                    halt.set()
                    raise Interrupted()
                return x

            inner = ff.create().call(b)
            return (
                ff.create(state={})
                .with_checkpoint_store(store, NAME)
                .with_halt(halt)
                .call(a)
                .then(inner)
            )

        assert await build(_tracker(), arm=True).run(1) is HALTED
        snapshot = await History(store, NAME).snapshot(await History(store, NAME).head())
        assert [p for p, c in snapshot.cursors.items() if RESOURCES in c] == [""]

        fresh = _tracker()
        assert await build(fresh, arm=False).run(1, resume="latest") == 1
        assert fresh.spent == pytest.approx(3.0 + 2.0)  # 3.0 restored, b again: 2.0


class _SessionTracker(CostTracker):
    """Charges each session only its own spend: keeps the session's baseline in its snapshot."""

    def __init__(self, session: str, allowance: float, halt: asyncio.Event | None = None) -> None:
        super().__init__(make_test_logger(), PricingConfig(), allowance, halt=halt)
        self.session = session
        self.allowance = allowance
        self.baseline = 0.0

    @property
    def session_spent(self) -> float:
        return self.spent - self.baseline

    def snapshot(self) -> dict[str, Any]:
        return {**super().snapshot(), "session": self.session, "baseline": self.baseline}

    def restore(self, data: dict[str, Any]) -> None:
        same = data["session"] == self.session
        self.baseline = data["baseline"] if same else data["spent"]
        self.update_budget(self.baseline + self.allowance)
        super().restore(data)


class TestTrackerSubclassAcrossResume:
    """What resume does to the spend is the tracker's: gent hands back what its snapshot() saved."""

    @staticmethod
    def _flow(
        store: Any,
        tracker: CostTracker,
        halt: asyncio.Event,
        seen: dict[str, float],
        stop_after: str | None = None,
    ) -> Any:
        """Steps ``a``..``d`` record the session's spend so far, then spend 1.0 each.

        The step named ``stop_after`` sets the halt; the next one stops on it.
        """

        def step(ctx: Context[Any], name: str) -> None:
            if ctx.halt is not None and ctx.halt.is_set():
                raise Interrupted()
            assert isinstance(ctx.cost, _SessionTracker)
            seen[name] = ctx.cost.session_spent
            _spend(ctx, 1.0)
            if name == stop_after:
                halt.set()

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            step(ctx, "a")
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            step(ctx, "b")
            return x

        @verb
        async def c(ctx: Context[Any], x: int) -> int:
            step(ctx, "c")
            return x

        @verb
        async def d(ctx: Context[Any], x: int) -> int:
            step(ctx, "d")
            return x

        return (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
            .with_halt(halt)
            .with_cost_tracker(tracker)
            .call(a)
            .then(b)
            .then(c)
            .then(d)
        )

    async def test_a_new_session_mid_flow_is_charged_only_its_own_spend(self) -> None:
        store = InMemoryCheckpointStore()
        first = _SessionTracker("s1", 10.0)
        assert await self._flow(store, first, asyncio.Event(), {}, "b").run(1) is HALTED
        assert first.session_spent == 2.0

        second = _SessionTracker("s2", 10.0)
        seen: dict[str, float] = {}
        assert await self._flow(store, second, asyncio.Event(), seen).run(resume="latest") == 1
        assert seen == {"c": 0.0, "d": 1.0}
        assert second.spent == 4.0 and second.session_spent == 2.0

    async def test_a_resume_in_the_same_session_keeps_its_baseline(self) -> None:
        store = InMemoryCheckpointStore()
        await self._flow(store, _SessionTracker("s1", 10.0), asyncio.Event(), {}, "a").run(1)
        s2 = _SessionTracker("s2", 10.0)
        assert await self._flow(store, s2, asyncio.Event(), {}, "c").run(resume="latest") is HALTED
        assert s2.baseline == 1.0 and s2.session_spent == 2.0

        again = _SessionTracker("s2", 10.0)
        assert await self._flow(store, again, asyncio.Event(), {}).run(resume="latest") == 1
        assert again.baseline == 1.0
        assert again.spent == 4.0 and again.session_spent == 3.0

    async def test_the_session_after_a_finished_run_is_rebased_on_its_total(self) -> None:
        store = InMemoryCheckpointStore()
        await self._flow(store, _SessionTracker("s1", 10.0), asyncio.Event(), {}).run(1)

        second = _SessionTracker("s2", 10.0)
        await self._flow(store, second, asyncio.Event(), {}).run(1, resume="latest")
        assert second.spent == 8.0 and second.session_spent == 4.0

    async def test_a_cap_raised_before_the_base_restore_does_not_fire_the_halt(self) -> None:
        """Restored spend over the session's allowance: the rebased cap keeps the run going."""
        store = InMemoryCheckpointStore()
        await self._flow(store, _SessionTracker("s1", 10.0), asyncio.Event(), {}, "c").run(1)

        halt = asyncio.Event()
        second = _SessionTracker("s2", 2.5, halt=halt)
        assert await self._flow(store, second, halt, {}).run(resume="latest") == 1
        assert not halt.is_set() and not second.exceeded
        assert second.budget == 5.5 and second.session_spent == 1.0

    async def test_a_restore_that_ignores_the_saved_spend_starts_from_its_own(self) -> None:
        """Saved 5.0 over a cap of 4.0: ignored, so the run neither halts nor counts it."""

        class Ignoring(CostTracker):
            def restore(self, data: dict[str, Any]) -> None:
                pass

        store = InMemoryCheckpointStore()
        flow = TestAppTrackerAcrossResume._flow
        assert await flow(store, _tracker(), asyncio.Event(), []).run(1) is HALTED

        halt = asyncio.Event()
        fresh = Ignoring(make_test_logger(), PricingConfig(), 4.0, halt=halt)
        seen: list[float] = []
        assert await flow(store, fresh, halt, seen).run(resume="latest") == 1
        assert seen == [0.0]
        assert fresh.spent == 1.0 and not halt.is_set()

    async def test_a_child_override_puts_budgeted_runs_on_the_subclass(self) -> None:
        class Owned(CostTracker):
            def child(
                self,
                budget: float | None = None,
                *,
                on_cost: Any = None,
                halt: asyncio.Event | None = None,
            ) -> CostTracker:
                return Owned(
                    self._lg, self._pricing, budget, parent=self, on_cost=on_cost, halt=halt
                )

        root = Owned(make_test_logger(), PricingConfig())
        seen: dict[int, Any] = {}
        costs = {0: [0.2, 0.3], 1: [0.1, 0.1]}
        assert await _item_flow(root, asyncio.Event(), costs, seen).run() == [0, 10]
        assert all(type(t) is Owned and t.parent is root for t in seen.values())

    async def test_a_budgeted_run_s_child_restores_through_the_subclass(self) -> None:
        """Items halted mid-way resume on the subclass's children, through their restore()."""
        restored: list[dict[str, Any]] = []

        class Recording(CostTracker):
            def child(
                self,
                budget: float | None = None,
                *,
                on_cost: Any = None,
                halt: asyncio.Event | None = None,
            ) -> CostTracker:
                return Recording(
                    self._lg, self._pricing, budget, parent=self, on_cost=on_cost, halt=halt
                )

            def snapshot(self) -> dict[str, Any]:
                return {**super().snapshot(), "tag": "item" if self.parent else "root"}

            def restore(self, data: dict[str, Any]) -> None:
                restored.append(data)
                super().restore(data)

        def root() -> Recording:
            return Recording(make_test_logger(), PricingConfig())

        store = InMemoryCheckpointStore()
        costs = {0: [0.3, 0.2], 1: [0.4, 0.1]}
        flow1 = _item_flow(root(), asyncio.Event(), costs, {}, arm=True)
        assert await flow1.with_checkpoint_store(store, NAME).with_checkpointer().run() is HALTED

        before: dict[int, float] = {}
        flow2 = _item_flow(root(), asyncio.Event(), costs, {}, before_second=before)
        assert await flow2.with_checkpoint_store(store, NAME).with_checkpointer().run(
            resume="latest"
        ) == [0, 10]
        assert sorted(d["tag"] for d in restored) == ["item", "item", "root"]
        assert before == {0: pytest.approx(0.3), 1: pytest.approx(0.4)}


def _item_flow(
    root: CostTracker,
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
        seen[item] = ctx.cost
        _spend(ctx, costs[item][0])
        if arm and len(seen) == len(costs):
            halt.set()
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

    return (
        _ff()
        .create(state={})
        .with_halt(halt)
        .with_cost_tracker(root)
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

    async def test_an_item_over_its_budget_carries_on_with_exceeded_latched(self) -> None:
        """Gent tracks; the agent decides: crossing the budget stops nothing."""
        root = _tracker()
        seen: dict[int, Any] = {}
        costs = {0: [0.2, 0.3], 1: [1.5, 9.0], 2: [0.1, 0.1]}
        assert await _item_flow(root, asyncio.Event(), costs, seen).run() == [0, 10, 20]
        assert seen[1].exceeded and seen[1].urgent_wrapup
        assert not seen[0].exceeded
        assert root.spent == pytest.approx(0.5 + 10.5 + 0.2)

    async def test_items_share_the_tracker_without_a_cap_of_their_own(self) -> None:
        root = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], item: int) -> int:
            seen.append(ctx.cost)
            return item

        flow = (
            _ff()
            .create(state={})
            .with_cost_tracker(root)
            .map(lambda b: b.call(step), items=lambda _p, _c: [0, 1])
        )
        await flow.run()
        assert seen == [root, root]


class TestItemBudgetsAcrossHalt:
    async def test_budgeted_item_stops_on_the_run_halt_and_resumes_with_its_spend(self) -> None:
        """Each item resumes on a child with its spend; the run's tracker continues its total."""
        store = InMemoryCheckpointStore()
        costs = {0: [0.3, 0.2], 1: [0.4, 0.1]}

        root1 = _tracker()
        flow1 = _item_flow(root1, asyncio.Event(), costs, {}, arm=True)
        assert await flow1.with_checkpoint_store(store, NAME).with_checkpointer().run() is HALTED
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
        # 0.7 restored; the items' restored spend does not roll up a second time.
        assert root2.spent == pytest.approx(0.7 + 0.3)

    async def test_an_item_resumed_over_its_budget_runs_with_exceeded_latched(self) -> None:
        """Item 0 halted at 0.9; resumed under a budget of 0.5 it carries on, exceeded."""
        store = InMemoryCheckpointStore()
        costs = {0: [0.9, 0.1], 1: [0.1, 0.1]}
        flow1 = _item_flow(_tracker(), asyncio.Event(), costs, {}, arm=True)
        assert await flow1.with_checkpoint_store(store, NAME).with_checkpointer().run() is HALTED

        root2 = _tracker()
        before: dict[int, float] = {}
        flow2 = _item_flow(root2, asyncio.Event(), costs, {}, before_second=before, cap=0.5)
        assert await flow2.with_checkpoint_store(store, NAME).with_checkpointer().run(
            resume="latest"
        ) == [0, 10]
        assert before == {0: pytest.approx(0.9), 1: pytest.approx(0.1)}
        assert root2.spent == pytest.approx(1.0 + 0.2)


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
        async def research(ctx: Context[Any], task: str) -> str:
            def hand_over(s: Any, c: Context[Any]) -> None:
                s.trackers[task] = c.cost

            loop = Loop(
                ROLE, saia=saia, conversation_factory=_ConvFactory(), on_executor_ready=hand_over
            )
            result = await loop(ctx, task)
            return str(result.output)

        flow = (
            _ff()
            .create(state={})
            .with_cost_tracker(root)
            .map(
                lambda b: b.with_budget(1.0).call(research),
                items=lambda _p, _c: ["a", "b"],
                max_concurrency=1,
            )
        )
        assert await flow.run() == ["done:a", "done:b"]
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

        flow = (
            _ff()
            .create(state={})
            .with_cost_tracker(root)
            .iterate(lambda b: b.with_budget(1.0).call(step), max_iters=3)
        )
        assert await flow.run(0) == 3
        assert len({id(t) for t in seen}) == 3
        assert all(t.parent is root and t.spent == 0.5 for t in seen)
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

        flow = _ff().create(state={}).with_cost_tracker(root).call(sub).then(after)
        assert await flow.run(0.5) == 0.5
        assert await flow.run(2.0) == 2.0
        assert ran == ["check", "after:0.5", "check", "after:2.0"]
        assert [c.exceeded for c in children] == [False, True]
        assert root.spent == pytest.approx(2.5)

    async def test_the_run_halt_reaches_a_flow_carrying_the_factory_s_halt(self) -> None:
        """Every flow a FlowFactory(halt=...) builds carries the run's halt; nested ones see it."""
        halt = asyncio.Event()
        ff = FlowFactory(make_test_logger(), halt=halt)
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
        flow = ff.create(state={}).with_cost_tracker(_tracker()).call(sub)
        assert await flow.run(0.5) is HALTED
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
        flow = _ff().create(state={}).with_cost_tracker(root).call(budgeted_map)
        assert await flow.run() == [0, 1, 2, 3]
        assert len({id(t) for t in trackers}) == 1 and trackers[0].exceeded
        assert root.spent == pytest.approx(1.6)


class TestCostAcrossShortcut:
    async def test_a_budgeted_flow_keeps_its_tracker_through_the_shortcut(self) -> None:
        """The continuation runs on the same child: spend before and after the cut adds up once."""
        root = _tracker()
        cut = asyncio.Event()
        trackers: list[Any] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            trackers.append(ctx.cost)
            _spend(ctx, 1.0)
            cut.set()
            await asyncio.sleep(0)  # the stop follows the signal
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            _spend(ctx, 9.0)
            return x

        @verb
        async def c(ctx: Context[Any], x: int) -> int:
            trackers.append(ctx.cost)
            _spend(ctx, 0.5)
            return x

        sub = (
            _ff()
            .create()
            .with_budget(5.0)
            .with_shortcut("cut", to="c")
            .call(a)
            .then(b)
            .then(c, name="c")
        )
        flow = _ff().create().with_signal("cut", cut).with_cost_tracker(root).call(sub)
        assert await flow.run(1) == 1
        assert trackers[0] is trackers[1]
        assert trackers[0].spent == pytest.approx(1.5)
        assert root.spent == pytest.approx(1.5)

    @pytest.mark.parametrize("map_cut", [False, True], ids=["items-alone", "map-too"])
    async def test_budgeted_items_continue_with_their_spend(self, map_cut: bool) -> None:
        """Items stopped by the cut continue with their spend; none counts twice.

        Alone, each item's shortcut continues it in place, on its tracker.
        With the map's flow cut too, the items stop with it and run again
        from restaged positions, on a child restored from their spend.
        """
        root = _tracker()
        cut = asyncio.Event()
        gate = asyncio.Event()
        at_extract: dict[int, float] = {}

        @verb
        async def query(ctx: Context[Any], n: int) -> int:
            _spend(ctx, 0.2)
            if n == 1:
                cut.set()
                await asyncio.sleep(0)  # the stop follows the signal
                gate.set()
            else:
                await gate.wait()
            return n

        @verb
        async def explore(ctx: Context[Any], n: int) -> int:
            _spend(ctx, 5.0)
            return n

        @verb
        async def extract(ctx: Context[Any], n: int) -> int:
            assert ctx.cost is not None
            at_extract[n] = ctx.cost.spent
            _spend(ctx, 0.1)
            return n

        def item(b: Any) -> None:
            b.with_budget(1.0).call(query).then(explore).then(extract, name="extract")
            b.with_shortcut("cut", to="extract")

        waves = _ff().create().map(item, items=lambda _p, _c: [0, 1], max_concurrency=2)
        if map_cut:
            waves.with_shortcut("cut")
        flow = _ff().create().with_signal("cut", cut).with_cost_tracker(root).call(waves)
        assert await flow.run() == [0, 1]
        assert at_extract == {0: pytest.approx(0.2), 1: pytest.approx(0.2)}
        assert root.spent == pytest.approx(2 * (0.2 + 0.1))


class TestCostAcrossCrash:
    async def test_a_budgeted_run_resumes_at_its_last_save(self) -> None:
        """Spend after the last save is not in any checkpoint: the redone step records it again."""
        store = InMemoryCheckpointStore()
        seen: list[float] = []

        def build(root: CostTracker, fail: bool) -> Any:
            @verb
            async def a(ctx: Context[Any], x: int) -> int:
                _spend(ctx, 1.0)
                return x

            @verb
            async def b(ctx: Context[Any], x: int) -> int:
                assert ctx.cost is not None
                await ctx.checkpoint()  # the save: b is the step running at it
                seen.append(ctx.cost.spent)
                _spend(ctx, 2.0)
                if fail:
                    raise RuntimeError("process died")
                return x

            sub = _ff().create().with_budget(10.0).call(a).then(b)
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
                .with_cost_tracker(root)
                .call(sub)
            )

        with pytest.raises(RuntimeError, match="process died"):
            await build(_tracker(), fail=True).run(1)

        root2 = _tracker()
        assert await build(root2, fail=False).run(1, resume="latest") == 1
        assert seen == [pytest.approx(1.0), pytest.approx(1.0)]  # b's first spend was lost
        assert root2.spent == pytest.approx(1.0 + 2.0)  # the total at the save, then b again


class TestValidation:
    async def test_budget_without_a_cost_tracker_raises_at_run_start(self) -> None:
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
        with pytest.raises(RuntimeError, match="no cost tracker"):
            await flow.run(1)
        assert ran == []

    async def test_a_budget_on_a_flow_with_its_own_tracker_is_a_child_of_it(self) -> None:
        own = _tracker()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.cost)
            _spend(ctx, 0.25)
            return x

        flow = _ff().create(state={}).with_cost_tracker(own).with_budget(1.0).call(step)
        assert await flow.run(1) == 1
        assert seen[0].parent is own and seen[0].budget == 1.0
        assert own.spent == pytest.approx(0.25)

    @pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan")])
    async def test_budget_must_be_finite_and_positive(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite and > 0"):
            _ff().create().with_budget(bad)

    @pytest.mark.parametrize("bad", [True, "1.0", None, _tracker()])
    async def test_budget_must_be_a_number(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="with_budget takes a number"):
            _ff().create().with_budget(bad)

    @pytest.mark.parametrize("bad", [1.0, None, "tracker"])
    async def test_cost_tracker_must_be_a_cost_tracker(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="with_cost_tracker takes a CostTracker"):
            _ff().create().with_cost_tracker(bad)
