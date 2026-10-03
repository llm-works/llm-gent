# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A Panel inside a run is in the run's snapshots: halted, it resumes at its verbs.

The Panel runs as a map over its verbs at ``<step>/panel/<k>``: a finished
verb does not run again on resume, a paused Loop turn in a verb resumes
mid-turn, and the aggregate equals the uninterrupted run's.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import (
    HALTED,
    Context,
    FlowFactory,
    History,
    Interrupted,
    Loop,
    Panel,
    Role,
    verb,
)
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

ROLE = Role(name="r", backend="openai", model="none")
NAME = "panel-resume"


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


def _flow(store: Any, halt: asyncio.Event | None, saia: _SAIA, ran: list[str]) -> Any:
    """``judge`` runs a Panel of ``quick`` (a plain verb) and ``talk`` (a Loop turn)."""
    loop = Loop(ROLE, saia=saia, conversation_factory=_ConvFactory())

    @verb
    async def quick(ctx: Context[Any], topic: str) -> str:
        ran.append("quick")
        return f"quick:{topic}"

    @verb(role=ROLE)
    async def talk(ctx: Context[Any], topic: str) -> str:
        ran.append("talk")
        result = await loop(ctx, topic)
        return str(result.text)

    panel = Panel([quick, talk], aggregate=lambda rs: " + ".join(rs))

    @verb
    async def judge(ctx: Context[Any], topic: str) -> str:
        return str(await panel.run(ctx, topic))

    flow = (
        FlowFactory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, NAME)
        .with_checkpointer()
    )
    if halt is not None:
        flow = flow.with_halt(halt)
    return flow.call(judge)


class TestPanelResume:
    async def test_finished_verb_stays_done_and_the_paused_turn_resumes(
        self, tmp_path: Path
    ) -> None:
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        halt = asyncio.Event()
        ran: list[str] = []
        assert await _flow(store, halt, _SAIA(halt), ran).run("x") is HALTED
        assert ran == ["quick", "talk"]

        ran.clear()
        saia = _SAIA(None)
        result = await _flow(store, None, saia, ran).run("x", resume="latest")
        assert ran == ["talk"]  # quick finished before the halt: it does not run again
        assert saia.calls == [("x", ["half:x"], True)]  # the turn resumed mid-turn

        fresh = JsonFileCheckpointStore(make_test_logger(), tmp_path / "fresh")
        assert result == await _flow(fresh, None, _SAIA(None), []).run("x")
        assert result == "quick:x + answer:x"

    async def test_each_panel_in_a_step_resumes_its_own_verbs(self, tmp_path: Path) -> None:
        """The step reruns: its finished first Panel runs again; the second continues."""
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        halt = asyncio.Event()
        ran: list[str] = []

        def flow(run_halt: asyncio.Event | None) -> Any:
            @verb
            async def a(ctx: Context[Any], tag: str) -> str:
                ran.append(f"a:{tag}")
                return f"a:{tag}"

            @verb
            async def b(ctx: Context[Any], tag: str) -> str:
                ran.append(f"b:{tag}")
                if run_halt is not None and tag == "second":
                    run_halt.set()
                return f"b:{tag}"

            @verb
            async def c(ctx: Context[Any], tag: str) -> str:
                ran.append(f"c:{tag}")
                return f"c:{tag}"

            panel = Panel([b, c, a], aggregate=sorted)

            @verb
            async def step(ctx: Context[Any], _x: Any = None) -> list[str]:
                first = await panel.run(ctx, "first")
                second = await panel.run(ctx, "second")
                return [*first, *second]

            f = (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
            )
            return (f.with_halt(run_halt) if run_halt is not None else f).call(step)

        assert await flow(halt).run() is HALTED
        assert ran == ["b:first", "c:first", "a:first", "b:second"]

        ran.clear()
        result = await flow(None).run(resume="latest")
        assert ran == ["b:first", "c:first", "a:first", "c:second", "a:second"]
        assert result == ["a:first", "b:first", "c:first", "a:second", "b:second", "c:second"]

    async def test_checkpoint_inside_a_panel_verb_writes_a_commit(self, tmp_path: Path) -> None:
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            assert await ctx.checkpoint("in-panel") is not None
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            return x * 2

        panel = Panel([a, b], aggregate=sum)

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            return int(await panel.run(ctx, x))

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
        )
        assert await flow.call(step).run(3) == 9
        commit = await History(store, NAME).checkpoint("in-panel")
        assert commit is not None
        snapshot = await History(store, NAME).snapshot(commit)
        assert any(path.endswith("/panel/0") for path in snapshot.cursors)

    async def test_result_that_cannot_be_checkpointed_raises_naming_the_panel(
        self, tmp_path: Path
    ) -> None:
        """A finished verb's result is in the halt checkpoint, like a map item's."""
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        halt = asyncio.Event()

        @verb
        async def handle(ctx: Context[Any], _x: Any = None) -> object:
            return object()

        @verb
        async def stop(ctx: Context[Any], _x: Any = None) -> int:
            await asyncio.sleep(0)
            halt.set()
            raise Interrupted()

        panel = Panel([handle, stop], aggregate=list)

        @verb
        async def step(ctx: Context[Any], _x: Any = None) -> Any:
            return await panel.run(ctx)

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
            .with_halt(halt)
            .call(step)
        )
        with pytest.raises(TypeError, match=r"cursor at '.*/panel/0/done'"):
            await flow.run()
