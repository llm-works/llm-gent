# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""``with_shortcut(signal)``: a run signal cuts a region short, fast-forwarding it.

While the signal is set, the region (the declaring flow and every flow
under it) starts no new step, iterate pass or map item, and a Loop turn in
flight returns the result SAIA paused it with. Nothing is stopped or run
again; the step after the region runs as usual. A signal set outside the
region waits until the region starts. A cut is not a halt: no checkpoint,
``ctx.halt`` unset. Signals are declared on the top-level flow
(``with_signal``); which are set is in every checkpoint, and resume sets
them again.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_gent.flow import (
    HALTED,
    Context,
    Factory,
    Failure,
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


def _ff() -> Factory:
    return Factory(make_test_logger())


def _top(cut: asyncio.Event) -> Any:
    """A top-level flow declaring the run's signal ``"cut"``."""
    return _ff().create().with_signal("cut", cut)


def _region() -> Any:
    """A flow the run's ``"cut"`` signal cuts short."""
    return _ff().create().with_shortcut("cut")


@verb
async def _echo(ctx: Context[Any], x: Any = None) -> Any:
    return x


def _steps(ran: list[str], cut: asyncio.Event, cut_in: str = "") -> Any:
    """Verbs ``a``..``e``: each records itself and passes ``x + name`` on; ``cut_in`` sets the signal."""

    def make(name: str) -> Any:
        @verb
        async def step(ctx: Context[Any], x: str) -> str:
            ran.append(name)
            if name == cut_in:
                cut.set()
                await asyncio.sleep(0)
            return x + name

        step.__qualname__ = f"step_{name}"
        return step

    return {name: make(name) for name in "abcde"}


class TestDeclaration:
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

    @pytest.mark.parametrize("where", ["call", "iterate", "map", "branch", "deep"])
    async def test_a_region_inside_one_on_the_same_signal_raises_at_run_start(
        self, where: str
    ) -> None:
        inner = _ff().create(name="inner").with_shortcut("cut").call(_echo)
        outer = _ff().create(name="outer").with_shortcut("cut")
        if where == "call":
            outer.call(inner)
        elif where == "iterate":
            outer.iterate(lambda b: b.call(inner), max_iters=1)
        elif where == "map":
            outer.map(lambda b: b.call(inner), items=lambda *_: [1])
        elif where == "branch":
            outer.branch(when=lambda *_: True, then=inner)
        else:
            outer.call(_ff().create().call(_ff().create().call(inner)))
        with pytest.raises(RuntimeError, match="'inner' has with_shortcut\\('cut'\\) inside"):
            await _top(asyncio.Event()).call(outer).run(1)

    async def test_regions_on_different_signals_may_nest(self) -> None:
        inner = _ff().create().with_shortcut("skip").call(_echo)
        outer = _region().call(inner)
        flow = _top(asyncio.Event()).with_signal("skip", asyncio.Event()).call(outer)
        assert await flow.run(1) == 1

    async def test_sibling_regions_on_one_signal_each_fast_forward(self) -> None:
        """The second region starts with the signal set and ends at once; the rest runs."""
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        first = _region().call(s["a"]).then(s["b"])
        second = _region().call(s["c"])
        assert await _top(cut).call(first).then(second).then(s["d"]).run("") == "ad"
        assert ran == ["a", "d"]


class TestQueuedUntilTheRegion:
    """A -> B -> C (the region) -> D: a cut waits through A and B, ends C, and D runs."""

    @staticmethod
    def _flow(ran: list[str], cut: asyncio.Event, cut_at: str) -> Any:
        def make(name: str) -> Any:
            @verb
            async def step(ctx: Context[Any], x: int) -> int:
                ran.append(f"{name}:{x}")
                if f"{name}:{x}" == cut_at:
                    cut.set()
                    await asyncio.sleep(0)
                return x + 1

            step.__qualname__ = f"queue_{name}"
            return step

        c = _region().iterate(lambda b: b.call(make("c")), max_iters=4)
        return _top(cut).call(make("a")).then(make("b")).then(c).then(make("d"))

    @pytest.mark.parametrize(
        ("cut_at", "expected"),
        [
            ("a:0", ["a:0", "b:1", "d:2"]),
            ("b:1", ["a:0", "b:1", "d:2"]),
            ("c:3", ["a:0", "b:1", "c:2", "c:3", "d:4"]),
            ("d:6", ["a:0", "b:1", "c:2", "c:3", "c:4", "c:5", "d:6"]),
        ],
    )
    async def test_where_the_cut_comes(self, cut_at: str, expected: list[str]) -> None:
        ran: list[str] = []
        await self._flow(ran, asyncio.Event(), cut_at).run(0)
        assert ran == expected

    async def test_steps_outside_the_region_never_see_the_cut(self) -> None:
        cut = asyncio.Event()
        seen: list[tuple[str, bool, bool]] = []

        def make(name: str) -> Any:
            @verb
            async def step(ctx: Context[Any], x: Any) -> Any:
                if name == "a":
                    cut.set()
                    await asyncio.sleep(0)
                seen.append((name, ctx.fast_forward, ctx.halt is not None and ctx.halt.is_set()))
                return x

            step.__qualname__ = f"seen_{name}"
            return step

        flow = _top(cut).with_halt(asyncio.Event()).call(make("a")).then(make("b"))
        await flow.then(_region().call(make("c"))).then(make("d")).run(0)
        assert seen == [("a", False, False), ("b", False, False), ("d", False, False)]


class TestChain:
    async def test_the_region_ends_with_the_running_step_s_result(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        region = _region().call(s["a"]).then(s["b"]).then(s["c"])
        assert await _top(cut).call(region).then(s["d"]).run("") == "ad"
        assert ran == ["a", "d"]

    async def test_a_region_starting_with_the_signal_set_passes_its_input_through(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        region = _region().call(s["b"]).then(s["c"])
        assert await _top(cut).call(s["a"]).then(region).then(s["d"]).run("") == "ad"
        assert ran == ["a", "d"]

    async def test_a_signal_set_after_the_region_ended_does_nothing(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="b")
        region = _region().call(s["a"])
        assert await _top(cut).call(region).then(s["b"]).then(s["c"]).run("") == "abc"
        assert ran == ["a", "b", "c"]

    async def test_flows_under_the_region_end_after_their_running_step(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        inner = _ff().create().call(s["a"]).then(s["b"])
        region = _region().call(inner).then(s["c"])
        assert await _top(cut).call(region).then(s["d"]).run("") == "ad"
        assert ran == ["a", "d"]

    async def test_the_step_running_at_the_cut_finishes_once(self) -> None:
        """It is neither stopped nor run again: it sees the cut and returns what it has."""
        ran: list[tuple[bool, bool]] = []
        cut = asyncio.Event()

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            cut.set()
            await asyncio.sleep(0)
            ran.append((ctx.fast_forward, ctx.halt is not None and ctx.halt.is_set()))
            return x + "a"

        flow = _top(cut).with_halt(asyncio.Event()).call(_region().call(a).then(_echo))
        assert await flow.run("") == "a"
        assert ran == [(True, False)]

    async def test_interrupted_on_a_cut_alone_is_an_error(self) -> None:
        """``Interrupted`` belongs to the halt; a step ending early on a cut returns."""
        cut = asyncio.Event()

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            cut.set()
            raise Interrupted()

        with pytest.raises(RuntimeError, match="raised Interrupted while no halt is set"):
            await _top(cut).call(_region().call(a)).run("")


class TestConclude:
    """``.conclude(step)``: a ``then`` a cut does not skip."""

    async def test_without_a_cut_it_is_then(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut)
        region = _region().call(s["a"]).then(s["b"]).conclude(s["c"]).then(s["d"])
        assert await _top(cut).call(region).run("") == "abcd"
        assert ran == ["a", "b", "c", "d"]

    async def test_a_cut_skips_every_step_but_the_concluding_ones(self) -> None:
        """Cut in ``a``: ``b`` and ``d`` are skipped; ``c`` and ``e`` run, each with the last result."""
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        region = _region().call(s["a"]).then(s["b"]).conclude(s["c"]).then(s["d"])
        region.conclude(s["e"])
        assert await _top(cut).call(region).run("") == "ace"
        assert ran == ["a", "c", "e"]

    async def test_a_region_entered_with_the_signal_set_runs_them_on_its_input(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        region = _region().call(s["b"]).conclude(s["c"])
        assert await _top(cut).call(s["a"]).then(region).then(s["d"]).run("") == "acd"
        assert ran == ["a", "c", "d"]

    async def test_outside_a_region_it_is_then(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        flow = _top(cut).call(s["a"]).then(s["b"]).conclude(s["c"])
        assert await flow.run("") == "abc"
        assert ran == ["a", "b", "c"]

    async def test_it_is_not_a_finally(self) -> None:
        """A step that raises stops the chain; the concluding step does not run."""
        ran: list[str] = []

        @verb
        async def boom(ctx: Context[Any], x: str) -> str:
            raise ValueError("boom")

        s = _steps(ran, asyncio.Event())
        flow = _top(asyncio.Event()).call(_region().call(boom).conclude(s["c"]))
        with pytest.raises(ValueError, match="boom"):
            await flow.run("")
        assert ran == []

    async def test_started_items_run_theirs_and_unstarted_ones_nothing(self) -> None:
        cut = asyncio.Event()
        gate = asyncio.Event()
        ran: list[str] = []

        @verb
        async def first(ctx: Context[Any], n: int) -> int:
            ran.append(f"first:{n}")
            if n == 1:
                cut.set()
                await asyncio.sleep(0)
                gate.set()
            else:
                await gate.wait()
            return n

        @verb
        async def second(ctx: Context[Any], n: int) -> int:
            ran.append(f"second:{n}")
            return n

        @verb
        async def close(ctx: Context[Any], n: int) -> int:
            ran.append(f"close:{n}")
            return n * 10

        item = _ff().create().call(first).then(second).conclude(close)
        region = _region().map(item, items=lambda _p, _c: [0, 1, 2], max_concurrency=2)
        results = await _top(cut).call(region).run()
        assert results[:2] == [0, 10] and isinstance(results[2], Skipped)
        assert sorted(ran) == ["close:0", "close:1", "first:0", "first:1"]

    async def test_a_halt_during_it_resumes_and_finishes_the_chain(self) -> None:
        store = InMemoryCheckpointStore()

        def build(halt: asyncio.Event, cut: asyncio.Event, ran: list[str], arm: bool) -> Any:
            s = _steps(ran, cut, cut_in="a" if arm else "")

            @verb
            async def close(ctx: Context[Any], x: str) -> str:
                ran.append("close")
                if arm:
                    halt.set()
                    raise Interrupted()
                return x + "!"

            region = _region().call(s["a"]).then(s["b"]).conclude(close).then(s["c"])
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "conclude-halt")
                .with_halt(halt)
                .with_signal("cut", cut)
                .call(region)
                .then(s["d"])
            )

        ran: list[str] = []
        assert await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run("") is HALTED
        assert ran == ["a", "close"]

        ran.clear()
        flow = build(asyncio.Event(), asyncio.Event(), ran, arm=False)
        assert await flow.run("", resume="latest") == "a!d"
        assert ran == ["close", "d"]

    async def test_it_does_not_enter_step_ids_or_the_root_hash(self) -> None:
        then = _ff().create().call(_echo).then(_echo)
        conclude = _ff().create().call(_echo).conclude(_echo)
        assert _compute_node_ids("", then._nodes) == _compute_node_ids("", conclude._nodes)
        assert then.root_hash() == conclude.root_hash()

    async def test_halt_after_conclude_skips_project_for_fast_forwarded_step(self) -> None:
        """A halt after a conclude step does not compute inputs for a fast-forwarded step.

        When both the cut (fast-forward) and halt signals are set after a conclude
        step, the next step's inputs are not computed (project is not called) because
        that step will be skipped anyway. This prevents project from throwing before
        the halt checkpoint can be written.
        """
        store = InMemoryCheckpointStore()
        project_calls: list[str] = []

        def build(halt: asyncio.Event, cut: asyncio.Event, ran: list[str], arm: bool) -> Any:

            @verb
            async def a(ctx: Context[Any], x: str) -> str:
                ran.append("a")
                cut.set()
                return x + "a"

            @verb
            async def close(ctx: Context[Any], x: str) -> str:
                ran.append("close")
                if arm:
                    halt.set()
                return x + "!"

            def track_project(_prev: str) -> str:
                project_calls.append("project")
                return _prev

            @verb
            async def b(ctx: Context[Any], x: str) -> str:
                ran.append("b")
                return x + "b"

            region = _region().call(a).conclude(close).then(b, project=track_project)
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "halt-project")
                .with_halt(halt)
                .with_signal("cut", cut)
                .call(region)
            )

        ran: list[str] = []
        result = await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run("")
        assert result is HALTED
        assert ran == ["a", "close"]
        assert project_calls == []

        project_calls.clear()
        ran.clear()
        flow = build(asyncio.Event(), asyncio.Event(), ran, arm=False)
        result = await flow.run("", resume="latest")
        assert result == "a!"
        assert ran == []
        assert project_calls == []


class TestSignalSetInAStep:
    """A step that sets the signal and returns without awaiting: the next boundary sees it."""

    async def test_the_next_step_does_not_start(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            ran.append("a")
            cut.set()  # no await: the boundary comes before any other task runs
            return x + "a"

        @verb
        async def b(ctx: Context[Any], x: str) -> str:
            ran.append("b")
            return x + "b"

        assert await _top(cut).call(_region().call(a).then(b)).then(_echo).run("") == "a"
        assert ran == ["a"]

    async def test_no_new_pass_starts(self) -> None:
        ran: list[int] = []
        cut = asyncio.Event()

        @verb
        async def step(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            if n == 1:
                cut.set()
            return n + 1

        flow = _top(cut).with_shortcut("cut").iterate(lambda b: b.call(step), max_iters=5)
        assert await flow.run(0) == 2
        assert ran == [0, 1]

    async def test_no_new_item_starts(self) -> None:
        ran: list[int] = []
        cut = asyncio.Event()

        @verb
        async def item(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            if n == 0:
                cut.set()
            return n

        flow = (
            _top(cut)
            .with_shortcut("cut")
            .map(lambda b: b.call(item), items=lambda *_: [0, 1, 2], max_concurrency=1)
        )
        result = await flow.run()
        assert ran == [0]
        assert result[0] == 0 and all(isinstance(r, Skipped) for r in result[1:])


class TestContext:
    async def test_halt_is_the_run_s_halt_in_and_out_of_a_region(self) -> None:
        halt, cut = asyncio.Event(), asyncio.Event()
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: Any) -> Any:
            seen.append(ctx.halt)
            return x

        await _top(cut).with_halt(halt).call(step).then(_region().call(step)).run(1)
        assert seen == [halt, halt]

    async def test_setting_the_halt_from_a_step_in_a_region_halts_the_run(self) -> None:
        halt, cut = asyncio.Event(), asyncio.Event()
        ran: list[str] = []

        @verb
        async def pause(ctx: Context[Any], x: str) -> str:
            ran.append("pause")
            assert ctx.halt is not None
            ctx.halt.set()  # e.g. a stop flag stored outside the process
            return x

        @verb
        async def after(ctx: Context[Any], x: str) -> str:
            ran.append("after")
            return x

        flow = _top(cut).with_halt(halt).with_shortcut("cut").call(pause).then(after)
        assert await flow.run("") is HALTED
        assert ran == ["pause"]


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
    """A turn that sets the signal (and the halt, when given), then pauses on its abort."""

    def __init__(
        self, cut: asyncio.Event, halt: asyncio.Event | None = None, result_type: type = _Result
    ) -> None:
        self.role = ROLE
        self._cut = cut
        self._halt = halt
        self._result_type = result_type
        self.calls = 0

    async def complete(self, task: str, **kwargs: Any) -> Any:
        self.calls += 1
        kwargs["conversation"].messages.append(f"half:{task}")
        self._cut.set()
        if self._halt is not None:
            self._halt.set()
        await asyncio.sleep(0)  # the abort follows the signal
        abort = kwargs["abort_signal"]
        return self._result_type(paused=abort.is_set(), text=f"partial:{task}")


def _loop_flow(saia: Any, cut: asyncio.Event, out: dict[str, list[Any]], **top: Any) -> Any:
    """``turn`` (a Loop call, in the region) then ``after``, outside it."""

    async def on_paused(result: Any, ctx: Context[Any]) -> None:
        out["paused"].append(result)

    loop = Loop(ROLE, saia=saia, conversation_factory=_ConvFactory(), on_paused=on_paused)

    @verb(role=ROLE)
    async def turn(ctx: Context[Any], task: str) -> Any:
        out["turn"].append(task)
        return await loop(ctx, task)

    @verb
    async def after(ctx: Context[Any], result: Any) -> str:
        out["after"].append(result)
        return "done"

    flow = _top(cut)
    if "store" in top:
        flow = _ff().create(state={}).with_checkpoint_store(top["store"], "loop-cut")
        flow = flow.with_signal("cut", cut)
    if "halt" in top:
        flow = flow.with_halt(top["halt"])
    return flow.call(_region().call(turn)).then(after)


def _out() -> dict[str, list[Any]]:
    return {"turn": [], "after": [], "paused": []}


class TestLoop:
    async def test_a_turn_in_flight_returns_its_paused_result_in_the_same_call(self) -> None:
        cut = asyncio.Event()
        saia = _CuttingSAIA(cut)
        out = _out()
        assert await _loop_flow(saia, cut, out).run("x") == "done"
        assert saia.calls == 1 and out["turn"] == ["x"]  # nothing ran again
        assert out["after"] == [_Result(paused=True, text="partial:x")]
        assert out["paused"] == [_Result(paused=True, text="partial:x")]

    async def test_a_result_that_cannot_be_kept_still_comes_back(self) -> None:
        """A cut turn is not held, so its result needs no storing."""
        cut = asyncio.Event()
        saia = _CuttingSAIA(cut, result_type=_BareResult)
        out = _out()
        assert await _loop_flow(saia, cut, out).run("x") == "done"
        assert out["after"] == [_BareResult(paused=True, text="partial:x")]

    async def test_with_the_halt_set_too_the_turn_is_held_and_not_continued(self) -> None:
        """Halted mid-turn while cut: resume goes on fast-forwarding, returning the paused result."""
        store = InMemoryCheckpointStore()
        cut, halt = asyncio.Event(), asyncio.Event()
        saia = _CuttingSAIA(cut, halt)
        out = _out()
        assert await _loop_flow(saia, cut, out, store=store, halt=halt).run("x") is HALTED
        assert out["after"] == []

        cut = asyncio.Event()
        resumed = _out()
        flow = _loop_flow(saia, cut, resumed, store=store, halt=asyncio.Event())
        assert await flow.run("x", resume="latest") == "done"
        assert saia.calls == 1  # the held turn was not continued
        assert resumed["after"] == [_Result(paused=True, text="partial:x")]


class TestPauseDuringFastForward:
    async def test_resume_goes_on_fast_forwarding_and_sets_the_signal(self) -> None:
        store = InMemoryCheckpointStore()

        def build(halt: asyncio.Event, cut: asyncio.Event, ran: list[str], arm: bool) -> Any:
            @verb
            async def a(ctx: Context[Any], x: str) -> str:
                ran.append("a")
                if arm:
                    cut.set()
                    await asyncio.sleep(0)
                    halt.set()
                    raise Interrupted()
                return x + "a"

            s = _steps(ran, cut)
            region = _region().call(a).then(s["b"]).then(s["c"])
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "ff-pause")
                .with_halt(halt)
                .with_signal("cut", cut)
                .call(region)
                .then(s["d"])
            )

        ran: list[str] = []
        assert await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run("") is HALTED
        assert ran == ["a"]
        head = await History(store, "ff-pause").head()
        assert head is not None and head.meta.outcome == "halted"

        ran.clear()
        cut = asyncio.Event()
        flow = build(asyncio.Event(), cut, ran, arm=False)
        assert await flow.run("", resume="latest") == "ad"
        assert ran == ["a", "d"]  # the halted step runs again; b and c are skipped
        assert cut.is_set()

    async def test_a_signal_set_before_its_region_starts_survives_the_pause(self) -> None:
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
            region = _region().call(s["b"]).then(s["c"])
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "cut-before")
                .with_halt(halt)
                .with_signal("cut", cut)
                .call(a)
                .then(region)
                .then(s["d"])
            )

        ran: list[str] = []
        assert await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run("") is HALTED
        head = await History(store, "cut-before").head()
        assert head is not None
        assert (await History(store, "cut-before").snapshot(head)).cursors[""][SIGNALS] == ["cut"]

        ran.clear()
        flow = build(asyncio.Event(), asyncio.Event(), ran, arm=False)
        assert await flow.run("", resume="latest") == "ad"
        assert ran == ["d"]

    async def test_a_cut_writes_no_checkpoint_and_a_finished_run_records_no_signal(self) -> None:
        store = InMemoryCheckpointStore()
        cut = asyncio.Event()
        ran: list[str] = []
        s = _steps(ran, cut, cut_in="a")
        flow = (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, "cut-done")
            .with_signal("cut", cut)
            .call(_region().call(s["a"]).then(s["b"]))
        )
        assert await flow.run("") == "a"
        history = History(store, "cut-done")
        assert [c.meta.outcome async for c in history.commits()] == ["ok"]
        head = await history.head()
        assert head is not None
        assert SIGNALS not in (await history.snapshot(head)).cursors.get("", {})


class TestIterate:
    async def test_no_pass_starts_after_the_cut(self) -> None:
        ran: list[int] = []
        cut = asyncio.Event()

        @verb
        async def step(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            if n == 1:
                cut.set()
                await asyncio.sleep(0)
            return n + 1

        flow = _top(cut).with_shortcut("cut").iterate(lambda b: b.call(step), max_iters=5)
        assert await flow.run(0) == 2
        assert ran == [0, 1]

    async def test_the_running_pass_ends_after_its_running_step(self) -> None:
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

        region = _region().iterate(lambda b: b.call(a).then(s["b"]), max_iters=5)
        assert await _top(cut).call(region).run("") == "aba"
        assert ran == ["a", "b", "a"]


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
                await asyncio.sleep(0)
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
        assert [r.item for r in results[2:]] == [2, 3]
        assert sum(isinstance(c, Skipped) for c in completed) == 2

    async def test_a_started_item_s_chain_ends_after_its_running_step(self) -> None:
        cut = asyncio.Event()
        ran: list[str] = []
        gate = asyncio.Event()

        @verb
        async def first(ctx: Context[Any], n: int) -> int:
            ran.append(f"first:{n}")
            if n == 1:
                cut.set()
                await asyncio.sleep(0)
                gate.set()
            else:
                await gate.wait()
            return n

        @verb
        async def second(ctx: Context[Any], n: int) -> int:
            ran.append(f"second:{n}")
            return n

        flow = (
            _top(cut)
            .with_shortcut("cut")
            .map(
                lambda b: b.call(first).then(second),
                items=lambda _p, _c: [0, 1, 2],
                max_concurrency=2,
            )
        )
        results = await flow.run()
        assert results[:2] == [0, 1] and isinstance(results[2], Skipped)
        assert sorted(ran) == ["first:0", "first:1"]

    async def test_failed_and_skipped_items_keep_their_outcome(self) -> None:
        cut = asyncio.Event()
        ran: list[int] = []
        guarded: list[int] = []

        @verb
        async def item(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            if n == 0:
                raise ValueError("bad item 0")
            if n == 2:
                cut.set()  # completes; item 3 does not start
                await asyncio.sleep(0)
            return n * 10

        def guard(n: int, _ctx: Any) -> bool:
            guarded.append(n)
            return n != 1

        flow = (
            _top(cut)
            .with_shortcut("cut")
            .map(
                lambda b: b.call(item),
                items=lambda _p, _c: [0, 1, 2, 3],
                strict=False,
                max_concurrency=1,
            )
            .guard(guard)
        )
        failure, skipped, done, unstarted = await flow.run()
        assert (ran, guarded) == ([0, 2], [0, 1, 2])
        assert isinstance(failure, Failure) and isinstance(failure.exception, ValueError)
        assert (skipped, done, unstarted) == (Skipped(item=1), 20, Skipped(item=3))


class TestNestedShape:
    """``prepare``, then the region (passes of a map over items), then ``finish``."""

    @staticmethod
    def _flow(cut: asyncio.Event, ran: list[str], cut_at: str) -> Any:
        gate = asyncio.Event()
        pass_n = [0]

        def record(what: str) -> bool:
            ran.append(what)
            if what != cut_at:
                return False
            cut.set()
            return True

        @verb
        async def prepare(ctx: Context[Any], x: Any) -> Any:
            if record("prepare"):
                await asyncio.sleep(0)
            return x

        @verb
        async def items(ctx: Context[Any], carry: Any) -> list[int]:
            record(f"items:{pass_n[0]}")
            return [0, 1, 2]

        @verb
        async def first(ctx: Context[Any], n: int) -> int:
            if record(f"first:{pass_n[0]}.{n}"):
                await asyncio.sleep(0)
                gate.set()
            elif pass_n[0] == 1 and n == 0 and cut_at == "first:1.1":
                await gate.wait()
            return n

        @verb
        async def second(ctx: Context[Any], n: int) -> int:
            record(f"second:{pass_n[0]}.{n}")
            return n

        @verb
        async def close(ctx: Context[Any], results: list[Any]) -> list[Any]:
            record(f"close:{pass_n[0]}")
            pass_n[0] += 1
            return results

        @verb
        async def finish(ctx: Context[Any], results: Any) -> str:
            record("finish")
            done = results if isinstance(results, list) else []
            return f"{sum(isinstance(r, int) for r in done)} done"

        def body(b: Any) -> None:
            b.call(items).map(lambda i: i.call(first).then(second), max_concurrency=2).then(close)

        region = _region().iterate(body, max_iters=5)
        return _top(cut).call(prepare).then(region).then(finish)

    async def test_a_cut_during_a_pass_goes_to_the_step_after_the_region(self) -> None:
        """Pass 1 is cut while items 0 and 1 are in ``first``: they end there, item 2 and ``close`` are skipped."""
        cut = asyncio.Event()
        ran: list[str] = []
        assert await self._flow(cut, ran, "first:1.1").run(None) == "2 done"
        pass_0 = [f"{s}:0.{n}" for n in range(3) for s in ("first", "second")]
        assert ran[:2] == ["prepare", "items:0"] and sorted(ran[2:8]) == sorted(pass_0)
        assert ran[8:10] == ["close:0", "items:1"]
        assert sorted(ran[10:12]) == ["first:1.0", "first:1.1"]
        assert ran[12:] == ["finish"]

    async def test_a_cut_before_the_region_waits_and_skips_it(self) -> None:
        cut = asyncio.Event()
        ran: list[str] = []
        assert await self._flow(cut, ran, "prepare").run(None) == "0 done"
        assert ran == ["prepare", "finish"]


class TestSubtreeEndedEarly:
    """A region the cut ended leaves nothing of it behind for later checkpoints."""

    async def test_a_map_the_cut_ended_leaves_the_run_complete(self) -> None:
        store = InMemoryCheckpointStore()
        cut = asyncio.Event()
        ran: list[int] = []

        @verb
        async def item(ctx: Context[dict[str, Any]], x: int) -> int:
            ran.append(x)
            cut.set()
            await asyncio.sleep(0)
            return x

        region = _region().map(
            lambda b: b.call(item), items=lambda _p, _c: [1, 2, 3], max_concurrency=1
        )
        flow = (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, "cut-map")
            .with_checkpointer()
            .with_signal("cut", cut)
            .call(region)
        )
        results = await flow.run()
        assert ran == [1]
        assert results[0] == 1 and all(isinstance(r, Skipped) for r in results[1:])
        assert await History(store, "cut-map").is_complete()

    async def test_a_later_halt_checkpoint_holds_none_of_the_ended_region(self) -> None:
        store = InMemoryCheckpointStore()
        ran: list[str] = []

        def build(halt: asyncio.Event, arm: bool) -> Any:
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
                    halt.set()
                    raise Interrupted()
                return x

            region = _region().iterate(lambda b: b.call(step), max_iters=3)
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "cut-sub")
                .with_halt(halt)
                .with_signal("cut", cut)
                .call(region)
                .then(last)
            )

        assert await build(asyncio.Event(), arm=True).run(0) is HALTED
        assert ran == ["step:0", "step:1", "last"]
        history = History(store, "cut-sub")
        head = await history.head()
        assert head is not None
        snapshot = await history.snapshot(head)
        assert sorted([*snapshot.cursors, *snapshot.scopes]) == [""]

        ran.clear()
        assert await build(asyncio.Event(), arm=False).run(0, resume="latest") == 2
        assert ran == ["last"]


def _drained_items(ran: list[str], seen: list[tuple[str, bool, bool]], cut: asyncio.Event) -> Any:
    """An item body: ``first`` → a flow running ``mid`` → a map of two ``leaf`` items → ``last``.

    Item 1's ``first`` sets ``cut`` while item 0 waits for it; ``last``
    records ``(item, ctx.draining, ctx.fast_forward)``.
    """
    gate = asyncio.Event()

    @verb
    async def first(ctx: Context[Any], n: int) -> int:
        ran.append(f"first:{n}")
        if n == 1:
            cut.set()
            await asyncio.sleep(0)
            gate.set()
        else:
            await gate.wait()
        return n

    @verb
    async def mid(ctx: Context[Any], n: int) -> int:
        ran.append(f"mid:{n}")
        return n

    @verb
    async def leaf(ctx: Context[Any], key: str) -> str:
        ran.append(f"leaf:{key}")
        return key

    @verb
    async def last(ctx: Context[Any], keys: list[str]) -> str:
        item = keys[0][0]
        ran.append(f"last:{item}")
        seen.append((item, ctx.draining, ctx.fast_forward))
        return item

    sub = _ff().create().call(mid)
    leaves = _ff().create().map(lambda b: b.call(leaf), items=lambda n, _c: [f"{n}a", f"{n}b"])
    return lambda b: b.call(first).then(sub).then(leaves).then(last)


class TestDrain:
    """``with_shortcut(signal, drain=True)``: started map items run to their end."""

    def _flow(
        self, cut: asyncio.Event, ran: list[str], seen: list[tuple[str, bool, bool]], drain: bool
    ) -> Any:
        @verb
        async def post(ctx: Context[Any], x: Any) -> Any:
            ran.append("post")
            return x

        @verb
        async def fin(ctx: Context[Any], x: Any) -> Any:
            ran.append("fin")
            seen.append(("fin", ctx.draining, ctx.fast_forward))
            return x

        region = (
            _ff()
            .create()
            .with_shortcut("cut", drain=drain)
            .map(_drained_items(ran, seen, cut), items=lambda *_: [0, 1, 2], max_concurrency=2)
            .then(post)
            .conclude(fin)
        )
        return _top(cut).call(region)

    async def test_started_items_run_to_their_end_and_the_rest_is_cut(self) -> None:
        cut = asyncio.Event()
        ran: list[str] = []
        seen: list[tuple[str, bool, bool]] = []
        results = await self._flow(cut, ran, seen, drain=True).run()

        assert results[:2] == ["0", "1"] and isinstance(results[2], Skipped)
        for n in "01":  # the whole chain, the nested flow and both items of the nested map
            assert {f"first:{n}", f"mid:{n}", f"leaf:{n}a", f"leaf:{n}b", f"last:{n}"} <= set(ran)
        assert "first:2" not in ran
        assert "post" not in ran and ran[-1] == "fin"  # outside the items it fast-forwards
        assert sorted(seen) == [("0", True, False), ("1", True, False), ("fin", False, True)]

    async def test_without_drain_started_items_end_after_their_running_step(self) -> None:
        cut = asyncio.Event()
        ran: list[str] = []
        results = await self._flow(cut, ran, [], drain=False).run()

        assert results[:2] == [0, 1] and isinstance(results[2], Skipped)
        assert sorted(ran) == ["fin", "first:0", "first:1"]

    async def test_an_outer_fast_forward_region_still_cuts_a_started_item(self) -> None:
        cut, stop = asyncio.Event(), asyncio.Event()
        ran: list[str] = []

        @verb
        async def first(ctx: Context[Any], n: int) -> int:
            ran.append("first")
            stop.set()
            await asyncio.sleep(0)
            return n

        @verb
        async def second(ctx: Context[Any], n: int) -> int:
            ran.append("second")
            return n

        drain = (
            _ff()
            .create()
            .with_shortcut("cut", drain=True)
            .map(lambda b: b.call(first).then(second), items=lambda *_: [0])
        )
        outer = _ff().create().with_shortcut("stop").call(drain)
        flow = _top(cut).with_signal("stop", stop).call(outer)
        assert await flow.run() == [0]
        assert ran == ["first"]

    async def test_drain_takes_a_bool(self) -> None:
        with pytest.raises(TypeError, match="drain"):
            _ff().create().with_shortcut("cut", drain="yes")  # type: ignore[arg-type]


def _drained_turns(
    saia: Any,
    cut: asyncio.Event,
    halt: asyncio.Event,
    out: dict[str, list[Any]],
    store: InMemoryCheckpointStore,
) -> Any:
    """A drain region over a map of two items, one at a time: a Loop ``turn``, then ``after``."""
    loop = Loop(ROLE, saia=saia, conversation_factory=_ConvFactory())

    @verb(role=ROLE)
    async def turn(ctx: Context[Any], n: int) -> Any:
        out["turn"].append(n)
        return await loop(ctx, f"t{n}")

    @verb
    async def after(ctx: Context[Any], result: Any) -> Any:
        out["after"].append(result)
        return result

    region = (
        _ff()
        .create()
        .with_shortcut("cut", drain=True)
        .map(lambda b: b.call(turn).then(after), items=lambda *_: [0, 1], max_concurrency=1)
    )
    flow = _ff().create(state={}).with_checkpoint_store(store, "drain-turn")
    return flow.with_checkpointer().with_halt(halt).with_signal("cut", cut).call(region)


class TestDrainLoop:
    async def test_the_signal_does_not_abort_a_turn_in_a_started_item(self) -> None:
        cut, halt = asyncio.Event(), asyncio.Event()
        out = _out()
        saia = _CuttingSAIA(cut)
        flow = _drained_turns(saia, cut, halt, out, InMemoryCheckpointStore())
        results = await flow.run()

        assert out["after"] == [_Result(paused=False, text="partial:t0")]  # not aborted
        assert isinstance(results[1], Skipped) and out["turn"] == [0]

    async def test_a_halt_pauses_the_turn_and_resume_goes_on_draining(self) -> None:
        store = InMemoryCheckpointStore()
        cut, halt = asyncio.Event(), asyncio.Event()
        out = _out()
        saia = _CuttingSAIA(cut, halt)
        assert await _drained_turns(saia, cut, halt, out, store).run() is HALTED
        assert out["after"] == []

        cut = asyncio.Event()
        resumed = _out()
        saia = _CuttingSAIA(cut)
        flow = _drained_turns(saia, cut, asyncio.Event(), resumed, store)
        results = await flow.run(resume="latest")

        assert cut.is_set()  # the drain goes on
        assert saia.calls == 1  # the held turn continued
        assert resumed["after"] == [_Result(paused=False, text="partial:t0")]
        assert isinstance(results[1], Skipped)  # item 1 never started


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
