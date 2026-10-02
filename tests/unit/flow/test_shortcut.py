# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""``with_shortcut(signal, to=name)``: on a run signal, a flow continues at a later step.

The flow stops as the run's halt stops it, then continues at once from
where it stopped, in shortcut mode: the interrupted step runs again from
its positions, no new iterate pass or map item starts, a held Loop turn
returns the result SAIA paused it with, and the chain lands at ``to``
(or the flow's end). Signals are declared on the top-level flow
(``with_signal``); which are set is in every checkpoint, and resume sets
them again.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_gent.flow import (
    Context,
    FlowFactory,
    History,
    Interrupted,
    Loop,
    Role,
    Skipped,
    verb,
)
from llm_gent.flow._node_id import _compute_node_ids
from llm_gent.flow.state.snapshot import SIGNALS
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


def _ff() -> FlowFactory:
    return FlowFactory(make_test_logger())


def _top(cut: asyncio.Event) -> Any:
    """A top-level flow declaring the run's signal ``"cut"``."""
    return _ff().create().with_signal("cut", cut)


@verb
async def _echo(ctx: Context[Any], x: Any = None) -> Any:
    return x


class TestDeclaration:
    async def test_to_naming_no_step_raises_at_run_start(self) -> None:
        flow = _top(asyncio.Event()).with_shortcut("cut", to="extract").call(_echo)
        with pytest.raises(RuntimeError, match="no step of its chain is named 'extract'"):
            await flow.run(1)

    async def test_to_naming_two_steps_raises_at_run_start(self) -> None:
        flow = (
            _top(asyncio.Event())
            .with_shortcut("cut", to="x")
            .call(_echo, name="x")
            .then(_echo, name="x")
        )
        with pytest.raises(RuntimeError, match="2 steps of its chain"):
            await flow.run(1)

    async def test_to_may_name_a_step_appended_later_or_a_primitive(self) -> None:
        cut = asyncio.Event()
        call_named = _top(cut).with_shortcut("cut", to="last").call(_echo).then(_echo, name="last")
        assert await call_named.run(1) == 1
        iterate_named = (
            _top(cut)
            .with_shortcut("cut", to="loop")
            .call(_echo)
            .iterate(lambda b: b.call(_echo), max_iters=1, name="loop")
        )
        assert await iterate_named.run(2) == 2

    async def test_a_nested_flow_s_to_is_checked_too(self) -> None:
        inner = _ff().create().with_shortcut("cut", to="missing").call(_echo)
        with pytest.raises(RuntimeError, match="'missing'"):
            await _top(asyncio.Event()).call(inner).run(1)

    async def test_a_shortcut_on_an_undeclared_signal_raises_at_run_start(self) -> None:
        inner = _ff().create().with_shortcut("stop-early").call(_echo)
        with pytest.raises(RuntimeError, match="no such signal"):
            await _top(asyncio.Event()).call(inner).run(1)

    async def test_a_nested_flow_declaring_a_signal_raises_at_run_start(self) -> None:
        inner = _ff().create().with_signal("cut", asyncio.Event()).call(_echo)
        with pytest.raises(RuntimeError, match="declared on its top-level flow"):
            await _ff().create().call(inner).run(1)

    @pytest.mark.parametrize("bad", [None, "event", 1])
    async def test_a_signal_is_an_asyncio_event(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="asyncio.Event"):
            _ff().create().with_signal("cut", bad)

    @pytest.mark.parametrize("bad", ["", 3])
    async def test_names_are_non_empty_strs(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="with_signal"):
            _ff().create().with_signal(bad, asyncio.Event())
        with pytest.raises(ValueError, match="with_shortcut"):
            _ff().create().with_shortcut(bad)
        with pytest.raises(ValueError, match="to="):
            _ff().create().with_shortcut("cut", to=bad)


def _steps(ran: list[str], cut: asyncio.Event, cut_in: str = "", stop_in: str = "") -> Any:
    """Verbs ``a``..``d``: each records itself and passes ``x + name`` on.

    The one named ``cut_in`` sets the signal. The one named ``stop_in``
    also stops before finishing its work the first time it runs (it
    raises :class:`Interrupted` under the stop), so it is the interrupted
    step.
    """
    stopped: set[str] = set()

    def make(name: str) -> Any:
        @verb
        async def step(ctx: Context[Any], x: str) -> str:
            ran.append(name)
            if name == cut_in:
                cut.set()
                await asyncio.sleep(0)  # the stop follows the signal
            if name == stop_in and name not in stopped and ctx.halt is not None:
                stopped.add(name)
                if ctx.halt.is_set():
                    raise Interrupted()
            return x + name

        step.__qualname__ = f"step_{name}"
        return step

    return {name: make(name) for name in "abcd"}


class TestChain:
    async def test_to_none_ends_the_flow_with_the_last_completed_result(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        sub = _ff().create().with_shortcut("cut").call(s["a"]).then(s["b"]).then(s["c"])
        flow = _top(cut).call(sub).then(s["d"])
        assert await flow.run("") == "ad"
        assert ran == ["a", "d"]

    async def test_to_skips_the_steps_before_it(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        flow = (
            _top(cut)
            .with_shortcut("cut", to="c")
            .call(s["a"])
            .then(s["b"])
            .then(s["c"], name="c")
            .then(s["d"])
        )
        assert await flow.run("") == "acd"
        assert ran == ["a", "c", "d"]

    async def test_the_interrupted_step_runs_again_then_the_chain_jumps(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a", stop_in="a")
        flow = (
            _top(cut).with_shortcut("cut", to="c").call(s["a"]).then(s["b"]).then(s["c"], name="c")
        )
        assert await flow.run("") == "ac"
        assert ran == ["a", "a", "c"]

    async def test_a_signal_set_at_or_after_to_does_nothing(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="c")
        flow = (
            _top(cut)
            .with_shortcut("cut", to="b")
            .call(s["a"])
            .then(s["b"], name="b")
            .then(s["c"])
            .then(s["d"])
        )
        assert await flow.run("") == "abcd"
        assert ran == ["a", "b", "c", "d"]

    async def test_a_flow_starting_with_the_signal_set_passes_its_input_through(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        sub = _ff().create().with_shortcut("cut").call(s["b"]).then(s["c"])
        flow = _top(cut).call(s["a"]).then(sub).then(s["d"])
        assert await flow.run("") == "ad"
        assert ran == ["a", "d"]

    async def test_a_subflow_without_a_shortcut_finishes_its_chain(self) -> None:
        """The interrupted step is a subflow: it continues where it stopped, then the jump."""
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        inner = _ff().create().call(s["a"]).then(s["b"])
        flow = (
            _top(cut).with_shortcut("cut", to="d").call(inner).then(s["c"]).then(s["d"], name="d")
        )
        assert await flow.run("") == "abd"
        assert ran == ["a", "b", "d"]


class TestIterate:
    async def test_no_pass_starts_after_the_shortcut(self) -> None:
        ran: list[int] = []
        cut = asyncio.Event()

        @verb
        async def wave(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            if n == 1:
                cut.set()
                await asyncio.sleep(0)
            return n + 1

        flow = _top(cut).with_shortcut("cut").iterate(lambda b: b.call(wave), max_iters=5)
        assert await flow.run(0) == 2
        assert ran == [0, 1]

    @pytest.mark.parametrize(("body_cut", "expected"), [(False, "ab"), (True, "a")])
    async def test_the_running_pass_finishes_by_its_own_rules(
        self, body_cut: bool, expected: str
    ) -> None:
        """The pass the shortcut stopped finishes its body, unless the body declares one too."""
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut)
        passes: list[int] = []

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            passes.append(len(passes))
            if len(passes) == 2:
                cut.set()
                await asyncio.sleep(0)
            return await s["a"](ctx, x)

        def body(b: Any) -> None:
            b.call(a).then(s["b"])
            if body_cut:
                b.with_shortcut("cut")

        flow = _top(cut).with_shortcut("cut").iterate(body, max_iters=5)
        assert await flow.run("") == "ab" + expected
        assert ran == ["a", "b", *expected]


class TestMap:
    async def test_started_items_finish_and_the_rest_are_skipped(self) -> None:
        cut = asyncio.Event()
        gate = asyncio.Event()
        completed: list[Any] = []

        @verb
        async def item(ctx: Context[Any], n: int) -> int:
            if n == 0:
                await gate.wait()
            if n == 1:
                cut.set()
                await asyncio.sleep(0)  # the stop follows the signal
                gate.set()
            return n * 10

        async def on_complete(_item: Any, outcome: Any, ctx: Context[Any]) -> None:
            completed.append(outcome)

        flow = (
            _top(cut)
            .with_shortcut("cut")
            .map(lambda b: b.call(item), items=lambda _p, _c: [0, 1, 2, 3], max_concurrency=2)
            .on_item_complete(on_complete)
        )
        results = await flow.run()
        assert results[:2] == [0, 10]
        assert [type(r).__name__ for r in results[2:]] == ["Skipped", "Skipped"]
        assert [r.item for r in results[2:]] == [2, 3]
        assert sum(isinstance(c, Skipped) for c in completed) == 2

    async def test_an_item_stopped_inside_its_body_continues_there(self) -> None:
        """Items declare their own landing step: an item stopped before it jumps there."""
        cut = asyncio.Event()
        ran: list[str] = []
        gate = asyncio.Event()

        @verb
        async def query(ctx: Context[Any], n: int) -> int:
            ran.append(f"query:{n}")
            if n == 1:
                cut.set()
                await asyncio.sleep(0)  # the stop follows the signal
                gate.set()
            else:
                await gate.wait()
            return n

        @verb
        async def explore(ctx: Context[Any], n: int) -> int:
            ran.append(f"explore:{n}")
            return n

        @verb
        async def extract(ctx: Context[Any], n: int) -> str:
            ran.append(f"extract:{n}")
            return f"x{n}"

        def item_body(b: Any) -> None:
            b.call(query).then(explore).then(extract, name="extract")
            b.with_shortcut("cut", to="extract")

        flow = (
            _top(cut)
            .with_shortcut("cut")
            .map(item_body, items=lambda _p, _c: [0, 1, 2], max_concurrency=2)
        )
        results = await flow.run()
        assert results[:2] == ["x0", "x1"]
        assert isinstance(results[2], Skipped)
        assert sorted(ran) == ["extract:0", "extract:1", "query:0", "query:1"]


ROLE = Role(name="r", backend="openai", model="none")


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
class _Result:
    paused: bool
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"paused": self.paused, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _Result:
        return cls(**data)


@dataclass
class _BareResult:
    """A result :mod:`codec` cannot store."""

    paused: bool
    text: str


class _CuttingSAIA:
    """The first turn sets the signal, then pauses under the stop."""

    def __init__(self, cut: asyncio.Event, result_type: type = _Result) -> None:
        self.role = ROLE
        self._cut = cut
        self._result_type = result_type
        self.calls = 0

    async def complete(self, task: str, **kwargs: Any) -> Any:
        self.calls += 1
        kwargs["conversation"].messages.append(f"explored:{task}")
        self._cut.set()
        await asyncio.sleep(0)  # the stop follows the signal
        abort = kwargs["abort_signal"]
        return self._result_type(paused=abort.is_set(), text=f"partial:{task}")


def _loop_flow(saia: _CuttingSAIA, cut: asyncio.Event, extracted: list[Any]) -> Any:
    loop = Loop(ROLE, saia=saia, conversation_factory=_ConvFactory())

    @verb(role=ROLE)
    async def explore(ctx: Context[Any], topic: str) -> Any:
        return await loop(ctx, topic)

    @verb
    async def extract(ctx: Context[Any], found: Any) -> str:
        extracted.append(found)
        return "extracted"

    return _top(cut).with_shortcut("cut", to="extract").call(explore).then(extract, name="extract")


class TestLoop:
    async def test_a_held_turn_returns_the_result_saia_paused_it_with(self) -> None:
        cut = asyncio.Event()
        saia = _CuttingSAIA(cut)
        extracted: list[Any] = []
        assert await _loop_flow(saia, cut, extracted).run("x") == "extracted"
        assert saia.calls == 1  # the turn was not continued
        assert extracted == [_Result(paused=True, text="partial:x")]

    async def test_a_result_that_cannot_be_kept_comes_back_as_none(self) -> None:
        cut = asyncio.Event()
        saia = _CuttingSAIA(cut, _BareResult)
        extracted: list[Any] = []
        assert await _loop_flow(saia, cut, extracted).run("x") == "extracted"
        assert saia.calls == 1
        assert extracted == [None]


class TestPauseDuringShortcut:
    @staticmethod
    def _flow(
        store: Any, halt: asyncio.Event, cut: asyncio.Event, ran: list[str], arm: bool
    ) -> Any:
        """``a`` sets the signal and stops; run again in shortcut mode, it sets the halt and stops."""
        runs: list[int] = []

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            ran.append("a")
            runs.append(1)
            if arm and len(runs) == 1:
                cut.set()
                await asyncio.sleep(0)
                raise Interrupted()
            if arm and len(runs) == 2:
                halt.set()
                raise Interrupted()
            return x + "a"

        s = _steps(ran, cut)
        sub = (
            _ff().create().with_shortcut("cut", to="c").call(a).then(s["b"]).then(s["c"], name="c")
        )
        return (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, "shortcut-pause")
            .with_halt(halt)
            .with_signal("cut", cut)
            .call(sub)
            .then(s["d"])
        )

    async def test_resume_continues_the_shortcut_and_sets_the_signal(self) -> None:
        store = InMemoryCheckpointStore()
        ran: list[str] = []
        assert (
            await self._flow(store, asyncio.Event(), asyncio.Event(), ran, arm=True).run("") is None
        )
        assert ran == ["a", "a"]
        head = await History(store, "shortcut-pause").head()
        assert head is not None and head.meta.outcome == "halted"

        ran.clear()
        cut = asyncio.Event()
        flow = self._flow(store, asyncio.Event(), cut, ran, arm=False)
        assert await flow.run("", resume="latest") == "acd"
        assert ran == ["a", "c", "d"]  # b is skipped: the shortcut continued
        assert cut.is_set()

    async def test_a_signal_set_before_its_flow_starts_survives_the_pause(self) -> None:
        """Cut in ``a``, halted before the cut flow starts: resume still skips it."""
        store = InMemoryCheckpointStore()

        def build(halt: asyncio.Event, cut: asyncio.Event, ran: list[str], arm: bool) -> Any:
            @verb
            async def a(ctx: Context[Any], x: str) -> str:
                ran.append("a")
                if arm:
                    cut.set()
                    halt.set()
                return x + "a"

            s = _steps(ran, cut)
            sub = _ff().create().with_shortcut("cut").call(s["b"]).then(s["c"])
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "cut-before")
                .with_halt(halt)
                .with_signal("cut", cut)
                .call(a)
                .then(sub)
                .then(s["d"])
            )

        ran: list[str] = []
        assert await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run("") is None
        head = await History(store, "cut-before").head()
        assert head is not None
        assert (await History(store, "cut-before").snapshot(head)).cursors[""][SIGNALS] == ["cut"]

        ran.clear()
        assert (
            await build(asyncio.Event(), asyncio.Event(), ran, arm=False).run("", resume="latest")
            == "ad"
        )
        assert ran == ["d"]

    async def test_a_finished_run_records_no_signal(self) -> None:
        store = InMemoryCheckpointStore()
        cut = asyncio.Event()
        ran: list[str] = []
        s = _steps(ran, cut, cut_in="a")
        flow = (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, "cut-done")
            .with_signal("cut", cut)
            .call(_ff().create().with_shortcut("cut").call(s["a"]).then(s["b"]))
        )
        assert await flow.run("") == "a"
        history = History(store, "cut-done")
        assert await history.is_complete()
        head = await history.head()
        assert head is not None
        assert SIGNALS not in (await history.snapshot(head)).cursors.get("", {})


class TestCampaignShape:
    async def test_nested_shortcuts_on_one_signal(self) -> None:
        """Waves, wave body and items each declare where a cut lands.

        Wave 1 is cut while items 0 and 1 are in ``query``: both go on to
        ``extract`` (``explore`` is not started), item 2 is skipped,
        ``revise`` and further waves do not run, ``synthesis`` does.
        """
        cut = asyncio.Event()
        gate = asyncio.Event()
        ran: list[str] = []
        wave = [0]

        @verb
        async def plan(ctx: Context[Any], carry: Any) -> list[int]:
            ran.append(f"plan:{wave[0]}")
            return [0, 1, 2]

        @verb
        async def query(ctx: Context[Any], n: int) -> int:
            ran.append(f"query:{wave[0]}.{n}")
            if wave[0] == 1 and n == 1:
                cut.set()
                await asyncio.sleep(0)
                gate.set()
            elif wave[0] == 1:
                await gate.wait()
            return n

        @verb
        async def explore(ctx: Context[Any], n: int) -> int:
            ran.append(f"explore:{wave[0]}.{n}")
            return n

        @verb
        async def extract(ctx: Context[Any], n: int) -> int:
            ran.append(f"extract:{wave[0]}.{n}")
            return n

        @verb
        async def revise(ctx: Context[Any], found: list[Any]) -> list[Any]:
            ran.append(f"revise:{wave[0]}")
            wave[0] += 1
            return found

        @verb
        async def synthesis(ctx: Context[Any], found: list[Any]) -> str:
            ran.append("synthesis")
            return f"{len([f for f in found if not isinstance(f, Skipped)])} found"

        def item(b: Any) -> None:
            b.call(query).then(explore).then(extract, name="extract")
            b.with_shortcut("cut", to="extract")

        def wave_body(b: Any) -> None:
            b.call(plan).map(item, max_concurrency=2).then(revise).with_shortcut("cut")

        waves = _ff().create().iterate(wave_body, max_iters=5).with_shortcut("cut")
        campaign = _top(cut).call(waves).then(synthesis)
        assert await campaign.run(None) == "2 found"
        assert ran[:9] == [
            "plan:0",
            "query:0.0",
            "explore:0.0",
            "extract:0.0",
            "query:0.1",
            "explore:0.1",
            "extract:0.1",
            "query:0.2",
            "explore:0.2",
        ]
        wave_1 = ran[ran.index("plan:1") :]
        assert sorted(wave_1[:-1]) == [
            "extract:1.0",
            "extract:1.1",
            "plan:1",
            "query:1.0",
            "query:1.1",
        ]
        assert wave_1[-1] == "synthesis"


class TestSubtreeEndedEarly:
    """A shortcut ending a subtree leaves nothing of it behind for later checkpoints."""

    async def test_a_map_the_shortcut_ended_leaves_the_run_complete(self) -> None:
        store = InMemoryCheckpointStore()
        cut = asyncio.Event()
        ran: list[int] = []

        @verb
        async def item(ctx: Context[dict[str, Any]], x: int) -> int:
            ran.append(x)
            cut.set()
            await asyncio.sleep(0)
            return x

        sub = (
            _ff()
            .create()
            .map(lambda b: b.call(item), items=lambda _p, _c: [1, 2, 3], max_concurrency=1)
            .with_shortcut("cut")
        )
        flow = (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, "cut-map")
            .with_checkpointer()
            .with_signal("cut", cut)
            .call(sub)
        )
        results = await flow.run()
        assert ran == [1]
        assert results[0] == 1 and all(isinstance(r, Skipped) for r in results[1:])
        assert await History(store, "cut-map").is_complete()

    async def test_a_later_halt_checkpoint_holds_none_of_the_ended_subtree(self) -> None:
        store = InMemoryCheckpointStore()
        ran: list[str] = []

        def build(run_halt: asyncio.Event, arm: bool) -> Any:
            cut = asyncio.Event()

            @verb
            async def step(ctx: Context[dict[str, Any]], x: int) -> int:
                ran.append(f"step:{x}")
                if x == 1:
                    cut.set()
                    await asyncio.sleep(0)
                return x + 1

            @verb
            async def last(ctx: Context[dict[str, Any]], x: Any) -> Any:
                ran.append("last")
                if arm:
                    run_halt.set()
                    raise Interrupted()
                return x

            sub = _ff().create().iterate(lambda b: b.call(step), max_iters=3).with_shortcut("cut")
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "cut-sub")
                .with_halt(run_halt)
                .with_signal("cut", cut)
                .call(sub)
                .then(last)
            )

        assert await build(asyncio.Event(), arm=True).run(0) is None
        assert ran == ["step:0", "step:1", "last"]
        history = History(store, "cut-sub")
        head = await history.head()
        assert head is not None
        snapshot = await history.snapshot(head)
        assert sorted([*snapshot.cursors, *snapshot.scopes]) == [""]

        ran.clear()
        assert await build(asyncio.Event(), arm=False).run(0, resume="latest") == 2
        assert ran == ["last"]


class TestStepNames:
    async def test_call_and_then_take_a_name(self) -> None:
        with pytest.raises(ValueError, match=r"\.call\(name=\)"):
            _ff().create().call(_echo, name="")

    async def test_a_name_enters_the_step_id(self) -> None:
        unnamed = _ff().create().call(_echo)
        named = _ff().create().call(_echo, name="a")
        renamed = _ff().create().call(_echo, name="b")
        ids = [_compute_node_ids("", f._nodes)[0] for f in (unnamed, named, renamed)]
        assert len(set(ids)) == 3
