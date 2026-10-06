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
    HALTED,
    Context,
    Factory,
    Failure,
    History,
    Interrupted,
    Loop,
    RestoredError,
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

    async def test_a_subflow_without_a_shortcut_ends_after_its_interrupted_step(self) -> None:
        """The interrupted step is a subflow: covered by the shortcut, its chain ends, then the jump."""
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        inner = _ff().create().call(s["a"]).then(s["b"])
        flow = (
            _top(cut).with_shortcut("cut", to="d").call(inner).then(s["c"]).then(s["d"], name="d")
        )
        assert await flow.run("") == "ad"
        assert ran == ["a", "d"]

    async def test_a_subflow_lands_where_its_own_shortcut_says(self) -> None:
        """A flow below the declaring one with a ``to`` of its own still reaches that step."""
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        inner = (
            _ff()
            .create()
            .with_shortcut("cut", to="c")
            .call(s["a"])
            .then(s["b"])
            .then(s["c"], name="c")
        )
        flow = (
            _top(cut).with_shortcut("cut", to="d").call(inner).then(s["b"]).then(s["d"], name="d")
        )
        assert await flow.run("") == "acd"
        assert ran == ["a", "c", "d"]


class TestSignalSetInAStep:
    """A step that sets the signal and returns without awaiting: the next boundary still sees it."""

    async def test_the_next_step_does_not_start(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()

        def make(name: str) -> Any:
            @verb
            async def step(ctx: Context[Any], x: str) -> str:
                ran.append(name)
                if name == "a":
                    cut.set()  # no await: the boundary comes before any other task runs
                return x + name

            step.__qualname__ = f"sync_{name}"
            return step

        a, b, c, d = (make(n) for n in "abcd")
        flow = _top(cut).with_shortcut("cut", to="c").call(a).then(b).then(c, name="c").then(d)
        assert await flow.run("") == "acd"
        assert ran == ["a", "c", "d"]

    async def test_the_continuation_is_not_stopped_again_once_taken_over(self) -> None:
        """The stop the signal would set a tick later must not land on the continuation.

        The subflow's step sets the signal and stops at once (no await),
        the outer flow takes the shortcut over before any other task runs,
        and the step runs again in shortcut mode and awaits: the signal's
        stop must not stop it a second time.
        """
        ran: list[str] = []
        cut = asyncio.Event()

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            ran.append("a")
            if not cut.is_set():
                cut.set()  # no await
                raise Interrupted()
            await asyncio.sleep(0)  # lets every pending task run
            if ctx.halt is not None and ctx.halt.is_set():
                raise Interrupted()
            return x + "a"

        inner = _ff().create().call(a).then(_echo)
        flow = _top(cut).with_shortcut("cut", to="d").call(inner).then(_echo).then(_echo, name="d")
        assert await flow.run("") == "a"
        assert ran == ["a", "a"]

    async def test_no_new_pass_starts(self) -> None:
        ran: list[int] = []
        cut = asyncio.Event()

        @verb
        async def wave(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            if n == 1:
                cut.set()
            return n + 1

        flow = _top(cut).with_shortcut("cut").iterate(lambda b: b.call(wave), max_iters=5)
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

    async def test_interrupted_raised_right_after_setting_it_is_accepted(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        stopped: list[bool] = []

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            ran.append("a")
            if not stopped:
                stopped.append(True)
                cut.set()
                raise Interrupted()
            return x + "a"

        @verb
        async def b(ctx: Context[Any], x: str) -> str:
            ran.append("b")
            return x + "b"

        flow = _top(cut).with_shortcut("cut", to="b").call(a).then(b, name="b")
        assert await flow.run("") == "ab"
        assert ran == ["a", "a", "b"]


class TestRunHalt:
    """``ctx.run_halt`` is the run's halt; under a shortcut ``ctx.halt`` is the flow's stop."""

    async def test_under_a_shortcut_run_halt_is_the_run_s_halt_and_halt_the_stop(self) -> None:
        halt, cut = asyncio.Event(), asyncio.Event()
        seen: list[tuple[Any, Any]] = []

        @verb
        async def step(ctx: Context[Any], x: Any) -> Any:
            seen.append((ctx.halt, ctx.run_halt))
            return x

        sub = _ff().create().with_shortcut("cut").call(step)
        await _top(cut).with_halt(halt).call(step).then(sub).run(1)
        (outer_halt, outer_run), (inner_halt, inner_run) = seen
        assert outer_halt is halt and outer_run is halt
        assert inner_run is halt and inner_halt is not halt

    async def test_setting_run_halt_from_a_step_halts_the_run(self) -> None:
        halt, cut = asyncio.Event(), asyncio.Event()
        ran: list[str] = []

        @verb
        async def pause(ctx: Context[Any], x: str) -> str:
            ran.append("pause")
            assert ctx.run_halt is not None
            ctx.run_halt.set()  # e.g. a stop flag stored outside the process
            return x

        @verb
        async def after(ctx: Context[Any], x: str) -> str:
            ran.append("after")
            return x

        flow = _top(cut).with_halt(halt).with_shortcut("cut").call(pause).then(after)
        assert await flow.run("") is HALTED
        assert ran == ["pause"]

    async def test_a_step_tells_a_cut_from_a_halt(self) -> None:
        halt, cut = asyncio.Event(), asyncio.Event()
        stopping: list[tuple[bool, bool]] = []

        @verb
        async def a(ctx: Context[Any], x: str) -> str:
            cut.set()
            await asyncio.sleep(0)
            assert ctx.halt is not None and ctx.run_halt is not None
            stopping.append((ctx.halt.is_set(), ctx.run_halt.is_set()))
            return x + "a"

        flow = _top(cut).with_halt(halt).with_shortcut("cut", to="b").call(a).then(_echo, name="b")
        assert await flow.run("") == "a"
        assert stopping == [(True, False)]  # stopping for the cut, not for a halt

    async def test_without_a_halt_run_halt_is_none(self) -> None:
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: Any) -> Any:
            seen.append(ctx.run_halt)
            return x

        await _ff().create().call(step).run(1)
        assert seen == [None]


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

    @pytest.mark.parametrize("body_cut", [False, True])
    async def test_the_running_pass_ends_after_its_interrupted_step(self, body_cut: bool) -> None:
        """The pass the shortcut stopped ends after ``a``, covered or by its own shortcut."""
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
        assert await flow.run("") == "aba"
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

    async def test_failed_and_skipped_items_do_not_run_again(self) -> None:
        """The continuation takes them from the restaged cursor, as a resume would."""
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
                await asyncio.sleep(0)  # the stop follows the signal
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
        assert isinstance(failure, Failure)
        assert isinstance(failure.exception, RestoredError)
        assert failure.exception.message == "bad item 0"
        assert (skipped, done, unstarted) == (Skipped(item=1), 20, Skipped(item=3))


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
            await self._flow(store, asyncio.Event(), asyncio.Event(), ran, arm=True).run("")
            is HALTED
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
        assert await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run("") is HALTED
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


class TestSubtreeCover:
    """A shortcut covers the flows below it that declare none of their own."""

    async def test_a_nested_iterate_starts_no_new_pass(self) -> None:
        ran: list[Any] = []
        cut = asyncio.Event()

        @verb
        async def wave(ctx: Context[Any], n: int) -> int:
            ran.append(n)
            if n == 1:
                cut.set()
                await asyncio.sleep(0)
            return n + 1

        @verb
        async def last(ctx: Context[Any], n: int) -> int:
            ran.append("last")
            return n

        waves = _ff().create().iterate(lambda b: b.call(wave), max_iters=5)
        flow = _top(cut).with_shortcut("cut", to="last").call(waves).then(last, name="last")
        assert await flow.run(0) == 2
        assert ran == [0, 1, "last"]

    async def test_a_nested_map_skips_its_unstarted_items(self) -> None:
        cut = asyncio.Event()

        @verb
        async def item(ctx: Context[Any], n: int) -> int:
            if n == 1:
                cut.set()
                await asyncio.sleep(0)
            return n * 10

        items = (
            _ff()
            .create()
            .map(lambda b: b.call(item), items=lambda _p, _c: [0, 1, 2, 3], max_concurrency=1)
        )
        results = await _top(cut).with_shortcut("cut").call(items).then(_echo).run()
        assert results[:2] == [0, 10]
        assert [r.item for r in results[2:]] == [2, 3]

    async def test_the_subtree_under_the_landing_step_runs_normally(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        tail = _ff().create().call(s["c"]).then(s["d"])
        flow = _top(cut).with_shortcut("cut", to="tail").call(s["a"]).then(s["b"])
        flow = flow.then(tail, name="tail")
        assert await flow.run("") == "acd"
        assert ran == ["a", "c", "d"]

    @staticmethod
    def _research_steps(ran: list[str], cut: asyncio.Event) -> dict[str, Any]:
        """Verbs recording ``name:item``; ``research`` of item 1 cuts while item 0 is in it."""
        gate = asyncio.Event()

        def make(name: str) -> Any:
            @verb
            async def step(ctx: Context[Any], n: Any) -> Any:
                ran.append(f"{name}:{n}")
                if name == "research" and n == 1:
                    cut.set()
                    await asyncio.sleep(0)  # the stop follows the signal
                    gate.set()
                elif name == "research":
                    await gate.wait()
                return n

            step.__qualname__ = f"step_{name}"
            return step

        names = ("research", "extract", "digest", "more", "close", "finish")
        return {name: make(name) for name in names}

    async def test_what_runs_under_a_landed_flow_runs_normally(self) -> None:
        """Items land on ``extract``; the branch arm after it runs while the batch is still cut."""
        ran: list[str] = []
        cut = asyncio.Event()
        s = self._research_steps(ran, cut)
        digest = _ff().create().call(s["digest"]).then(s["more"])
        item = (
            _ff()
            .create()
            .with_shortcut("cut", to="extract")
            .call(s["research"])
            .then(s["extract"], name="extract")
            .branch(when=lambda _n, _c: True, then=digest)
            .then(s["close"])
        )
        batch = (
            _top(cut)
            .with_shortcut("cut", to="finish")
            .map(item, items=lambda _p, _c: [0, 1, 2], max_concurrency=2)
            .then(s["finish"], name="finish")
        )
        results = await batch.run()
        assert results[:2] == [0, 1] and isinstance(results[2], Skipped)
        names = ("research", "extract", "digest", "more", "close")
        assert sorted(ran[:-1]) == sorted(f"{name}:{n}" for name in names for n in (0, 1))
        assert ran[-1].startswith("finish:")

    async def test_a_subflow_and_an_iterate_under_a_landed_flow_run_fully(self) -> None:
        ran: list[str] = []
        cut = asyncio.Event()
        s = _steps(ran, cut, cut_in="a")
        passes: list[int] = []

        @verb
        async def count(ctx: Context[Any], x: str) -> str:
            passes.append(len(passes))
            return x

        tail = _ff().create().call(s["c"]).iterate(lambda b: b.call(count), max_iters=3)
        inner = (
            _ff()
            .create()
            .with_shortcut("cut", to="b")
            .call(s["a"])
            .then(s["b"], name="b")
            .then(tail)
        )
        flow = _top(cut).with_shortcut("cut", to="d").call(inner).then(s["d"], name="d")
        assert await flow.run("") == "abcd"
        assert ran == ["a", "b", "c", "d"]
        assert passes == [0, 1, 2]

    async def test_a_halt_under_a_landed_flow_resumes_normally(self) -> None:
        """Halted in the arm after the item landed: resume finishes the arm."""
        store = InMemoryCheckpointStore()

        def build(halt: asyncio.Event, cut: asyncio.Event, ran: list[str], arm: bool) -> Any:
            s = self._research_steps(ran, cut)

            @verb
            async def research(ctx: Context[Any], n: int) -> int:
                ran.append(f"research:{n}")
                cut.set()  # completes; item 1 does not start
                await asyncio.sleep(0)  # the stop follows the signal
                return n

            @verb
            async def digest(ctx: Context[Any], n: int) -> int:
                ran.append(f"digest:{n}")
                if arm:
                    halt.set()
                    raise Interrupted()
                return n

            item = (
                _ff()
                .create()
                .with_shortcut("cut", to="extract")
                .call(research)
                .then(s["extract"], name="extract")
                .branch(when=lambda _n, _c: True, then=_ff().create().call(digest).then(s["more"]))
                .then(s["close"])
            )
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "landed-halt")
                .with_halt(halt)
                .with_signal("cut", cut)
                .with_shortcut("cut", to="finish")
                .map(item, items=lambda _p, _c: [0, 1], max_concurrency=1)
                .then(s["finish"], name="finish")
            )

        ran: list[str] = []
        assert await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run() is HALTED
        assert ran == ["research:0", "extract:0", "digest:0"]

        ran.clear()
        flow = build(asyncio.Event(), asyncio.Event(), ran, arm=False)
        await flow.run(resume="latest")
        assert ran[:3] == ["digest:0", "more:0", "close:0"]
        assert ran[3].startswith("finish:") and len(ran) == 4

    async def test_resume_keeps_the_cover(self) -> None:
        """Halted while covered: resume ends the covered chain after the interrupted step."""
        store = InMemoryCheckpointStore()

        def build(halt: asyncio.Event, cut: asyncio.Event, ran: list[str], arm: bool) -> Any:
            runs: list[int] = []

            @verb
            async def a(ctx: Context[Any], x: str) -> str:
                ran.append("a")
                runs.append(1)
                if arm and len(runs) == 1:
                    cut.set()
                    await asyncio.sleep(0)
                    raise Interrupted()
                if arm:
                    halt.set()
                    raise Interrupted()
                return x + "a"

            s = _steps(ran, cut)
            sub = _ff().create().call(a).then(s["b"])
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, "cover-pause")
                .with_halt(halt)
                .with_signal("cut", cut)
                .with_shortcut("cut", to="d")
                .call(sub)
                .then(s["c"])
                .then(s["d"], name="d")
            )

        ran: list[str] = []
        assert await build(asyncio.Event(), asyncio.Event(), ran, arm=True).run("") is HALTED
        assert ran == ["a", "a"]

        ran.clear()
        flow = build(asyncio.Event(), asyncio.Event(), ran, arm=False)
        assert await flow.run("", resume="latest") == "ad"
        assert ran == ["a", "d"]


class TestCampaignShape:
    async def test_nested_shortcuts_on_one_signal(self) -> None:
        """Waves and items each declare where a cut lands; the wave body is covered by the waves'.

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
            b.call(plan).map(item, max_concurrency=2).then(revise)

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
