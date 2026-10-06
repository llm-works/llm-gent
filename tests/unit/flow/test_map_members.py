# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A map over members: each member runs once on the step's input, and the results aggregate.

``.map([a, b, c], aggregate=...)`` is a map whose items are its members —
an ensemble. Every map option applies to it; halted, it resumes at its
members: a finished member does not run again, a paused Loop turn in a
member resumes mid-turn. Members are matched to their records by what
they run, so a reordered list keeps them.
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
    Flow,
    History,
    Interrupted,
    Loop,
    Role,
    Skipped,
    Structure,
    majority,
    verb,
)
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import ROLE_A, ROLE_B, make_ff, make_test_logger


pytestmark = pytest.mark.unit

NAME = "members"


@verb
async def add_one(ctx: Context[Any], x: int) -> int:
    return x + 1


@verb
async def add_two(ctx: Context[Any], x: int) -> int:
    return x + 2


@verb
async def double(ctx: Context[Any], x: int) -> int:
    return x * 2


def _flow() -> Flow:
    return Flow(make_test_logger())


def _store_flow(store: Any, halt: asyncio.Event | None = None) -> Flow:
    flow = Factory(make_test_logger()).create(state={})
    flow.with_checkpoint_store(store, NAME).with_checkpointer()
    return flow.with_halt(halt) if halt is not None else flow


@pytest.mark.asyncio
class TestRunsEachMemberOnTheInput:
    async def test_the_results_in_member_order(self) -> None:
        assert await _flow().map([add_one, add_two, double]).run(10) == [11, 12, 20]

    async def test_aggregate_reduces_them(self) -> None:
        assert await _flow().map([add_one, add_two], aggregate=sum).run(10) == 23

    async def test_a_majority_vote(self) -> None:
        @verb
        async def yes(ctx: Context[Any], _x: Any = None) -> str:
            return "yes"

        @verb
        async def no(ctx: Context[Any], _x: Any = None) -> str:
            return "no"

        flow = _flow().map([yes, no, yes], aggregate=majority)
        assert await flow.run() == "yes"

    async def test_each_member_runs_under_its_own_role(self) -> None:
        @verb(role=ROLE_A)
        async def see_a(ctx: Context[Any], _x: Any = None) -> Role | None:
            return ctx.role

        @verb(role=ROLE_B)
        async def see_b(ctx: Context[Any], _x: Any = None) -> Role | None:
            return ctx.role

        assert await make_ff().create().map([see_a, see_b]).run() == [ROLE_A, ROLE_B]

    async def test_members_share_the_step_s_state_without_a_projection(self) -> None:
        @verb
        async def mark_a(ctx: Context[Any], _x: Any = None) -> str:
            ctx.data["a"] = True
            return "a"

        @verb
        async def mark_b(ctx: Context[Any], _x: Any = None) -> str:
            ctx.data["b"] = True
            return "b"

        @verb
        async def observe(ctx: Context[Any], _x: Any = None) -> dict[str, Any]:
            return dict(ctx.data)

        flow = make_ff().create(state={}).map([mark_a, mark_b]).then(observe)
        assert await flow.run() == {"a": True, "b": True}

    async def test_members_see_the_run_s_halt_from_any_depth(self) -> None:
        halt = asyncio.Event()
        seen: list[asyncio.Event | None] = []

        @verb
        async def capture(ctx: Context[Any], _x: Any = None) -> None:
            seen.append(ctx.halt)

        inner = make_ff().create().map([capture, capture])
        await make_ff().create().with_halt(halt).call(inner).run()
        assert seen == [halt, halt]


class TestDeclaration:
    def test_no_members_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one member"):
            _flow().map([])

    def test_items_with_members_raises(self) -> None:
        with pytest.raises(TypeError, match="items=.*members"):
            _flow().map([add_one], items=lambda _p, _c: [1])

    def test_members_are_in_the_structure(self) -> None:
        old = Structure.of(_flow().map([add_one, add_two], name="vote"))
        new = Structure.of(_flow().map([add_one, add_two, double], name="vote"))
        diff = old.diff(new)
        assert [p[-1][1].target for p in diff.added] == [f"verb:{__name__}.double"]


@pytest.mark.asyncio
class TestMapOptionsApply:
    async def test_strict_false_turns_a_failing_member_into_a_failure(self) -> None:
        @verb
        async def boom(ctx: Context[Any], _x: Any = None) -> int:
            raise ValueError("boom")

        first, failure = await _flow().map([add_one, boom], strict=False).run(1)
        assert first == 2
        assert isinstance(failure, Failure) and failure.item == 1

    async def test_a_strict_failure_raises(self) -> None:
        @verb
        async def boom(ctx: Context[Any], _x: Any = None) -> int:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await _flow().map([add_one, boom]).run(1)

    async def test_a_guard_skips_members(self) -> None:
        flow = _flow().map([add_one, add_two]).guard(lambda x, _c: x > 5)
        assert await flow.run(1) == [Skipped(item=1), Skipped(item=1)]

    async def test_max_concurrency_runs_them_one_at_a_time(self) -> None:
        order: list[str] = []

        def member(name: str) -> Any:
            @verb
            async def run(ctx: Context[Any], _x: Any = None) -> str:
                order.append(f"+{name}")
                await asyncio.sleep(0)
                order.append(f"-{name}")
                return name

            run.__qualname__ = run.__name__ = name
            return run

        flow = _flow().map([member("a"), member("b")], max_concurrency=1)
        assert await flow.run() == ["a", "b"]
        assert order == ["+a", "-a", "+b", "-b"]

    async def test_a_shortcut_skips_members_that_had_not_started(self) -> None:
        cut = asyncio.Event()

        @verb
        async def first(ctx: Context[Any], _x: Any = None) -> str:
            cut.set()
            await asyncio.sleep(0)  # the stop follows the signal
            return "first"

        @verb
        async def second(ctx: Context[Any], _x: Any = None) -> str:
            return "second"

        flow = (
            make_ff()
            .create()
            .with_signal("cut", cut)
            .with_shortcut("cut")
            .map([first, second], max_concurrency=1)
        )
        assert await flow.run() == ["first", Skipped(item=None)]


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
    text: str = ""


class _SAIA:
    """With ``halt``: the turn's first half, then it sets the halt and pauses. Without: completes."""

    def __init__(self, halt: asyncio.Event | None) -> None:
        self.role = ROLE
        self._halt = halt
        self.calls: list[tuple[str, list[str], bool]] = []

    async def complete(self, task: str, **kwargs: Any) -> _Result:
        conv: _Conv = kwargs["conversation"]
        resume = bool(kwargs.get("resume", False))
        self.calls.append((task, list(conv.messages), resume))
        if self._halt is not None:
            conv.messages.append(f"half:{task}")
            self._halt.set()
            return _Result(paused=True)
        return _Result(paused=False, text=f"answer:{task}")


def _judges(store: Any, halt: asyncio.Event | None, saia: _SAIA, ran: list[str]) -> Flow:
    """A map over ``quick`` (a plain verb) and ``talk`` (a Loop turn), joined."""
    loop = Loop(ROLE, saia=saia, conversation_factory=_ConvFactory())

    @verb
    async def quick(ctx: Context[Any], topic: str) -> str:
        ran.append("quick")
        return f"quick:{topic}"

    @verb(role=ROLE)
    async def talk(ctx: Context[Any], topic: str) -> str:
        ran.append("talk")
        return str((await loop(ctx, topic)).text)

    return _store_flow(store, halt).map([quick, talk], aggregate=" + ".join)


@pytest.mark.asyncio
class TestResume:
    async def test_a_finished_member_stays_done_and_a_paused_turn_resumes(self) -> None:
        store = InMemoryCheckpointStore()
        halt = asyncio.Event()
        ran: list[str] = []
        assert await _judges(store, halt, _SAIA(halt), ran).run("x") is HALTED
        assert ran == ["quick", "talk"]

        ran.clear()
        saia = _SAIA(None)
        result = await _judges(store, None, saia, ran).run("x", resume="latest")
        assert ran == ["talk"]
        assert saia.calls == [("x", ["half:x"], True)]
        assert result == "quick:x + answer:x"

    async def test_inside_an_iterate_pass_only_its_unfinished_members_run(self) -> None:
        store = InMemoryCheckpointStore()
        halt = asyncio.Event()
        runs: list[tuple[int, str]] = []

        def voter(name: str, halts_at: int | None) -> Any:
            @verb
            async def vote(ctx: Context[Any], iteration: int) -> int:
                runs.append((iteration, name))
                if iteration == halts_at:
                    halt.set()
                return iteration

            vote.__qualname__ = vote.__name__ = f"voter_{name}"
            return vote

        def build(with_halt: bool) -> Flow:
            voters = [voter("a", 2), voter("b", None), voter("c", None)]
            body = Flow(make_test_logger()).call(_next_pass)
            body.map(voters, aggregate=max, max_concurrency=1)
            flow = _store_flow(store, halt if with_halt else None)
            return flow.iterate(body, max_iters=3)

        assert await build(with_halt=True).run(0) is HALTED
        assert [r for r in runs if r[0] == 2] == [(2, "a")]

        runs.clear()
        assert await build(with_halt=False).run(resume="latest") == 3
        assert [r for r in runs if r[0] == 2] == [(2, "b"), (2, "c")]
        assert sorted(r for r in runs if r[0] == 3) == [(3, "a"), (3, "b"), (3, "c")]

    async def test_members_are_matched_by_what_they_run_not_their_position(self) -> None:
        """Halted after a and b; resumed reordered, b removed, one added: a stays done, the rest run."""
        store = InMemoryCheckpointStore()
        halt = asyncio.Event()
        ran: list[str] = []

        def member(name: str, halts: bool = False) -> Any:
            @verb
            async def run(ctx: Context[Any], x: int) -> str:
                ran.append(name)
                if halts:
                    halt.set()
                return f"{name}:{x}"

            run.__qualname__ = run.__name__ = name
            return run

        first = [member("a"), member("b", halts=True), member("c")]
        assert await _store_flow(store, halt).map(first, max_concurrency=1).run(1) is HALTED
        assert ran == ["a", "b"]

        ran.clear()
        reordered = [member("c"), member("new"), member("a")]
        resumed = _store_flow(store).map(reordered, max_concurrency=1)
        assert await resumed.run(resume="latest") == ["c:1", "new:1", "a:1"]
        assert ran == ["c", "new"]

    async def test_a_checkpoint_inside_a_member_writes_a_commit(self) -> None:
        store = InMemoryCheckpointStore()

        @verb
        async def saves(ctx: Context[Any], x: int) -> int:
            assert await ctx.checkpoint("in-member") is not None
            return x

        assert await _store_flow(store).map([saves, double], aggregate=sum).run(3) == 9
        commit = await History(store, NAME).checkpoint("in-member")
        assert commit is not None
        snapshot = await History(store, NAME).snapshot(commit)
        assert any("/i/" in path for path in snapshot.cursors)

    async def test_a_result_that_cannot_be_checkpointed_raises_naming_the_map(self) -> None:
        store = InMemoryCheckpointStore()
        halt = asyncio.Event()

        @verb
        async def handle(ctx: Context[Any], _x: Any = None) -> object:
            return object()

        @verb
        async def stop(ctx: Context[Any], _x: Any = None) -> int:
            await asyncio.sleep(0)
            halt.set()
            raise Interrupted()

        flow = _store_flow(store, halt).map([handle, stop])
        with pytest.raises(TypeError, match=r"cursor at '.*/done'"):
            await flow.run()


@verb
async def _next_pass(ctx: Context[Any], iteration: int) -> int:
    return iteration + 1
