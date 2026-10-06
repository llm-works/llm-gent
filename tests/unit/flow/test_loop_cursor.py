# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A paused Loop's turn resumes in the Loop call that paused it.

The turn is kept in the run's snapshots at the call's path
(``<step>/t/<k>`` for the step's ``k``-th Loop call): several calls in
one step, or running copies of one Loop in map items, each keep their
own; resume hands each call its own task and conversation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, Factory, History, Loop, Role, verb
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

ROLE = Role(name="r", backend="openai", model="gpt-4o-mini")


@dataclass
class _Conv:
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": list(self.messages)}


class _ConvFactory:
    def create(self) -> _Conv:
        return _Conv()

    def create_from_state(self, state: dict[str, Any]) -> _Conv:
        return _Conv(messages=list(state["messages"]))


@dataclass
class _Result:
    paused: bool = False
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"paused": self.paused, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _Result:
        return cls(**data)


class _PausingSAIA:
    """First run: every turn waits for ``arrivals`` turns, sets the halt, pauses.

    Resume run: records the task and conversation it was handed and completes.
    """

    def __init__(self, halt: asyncio.Event, arrivals: int | None) -> None:
        self.role = ROLE
        self._halt = halt
        self._arrivals = arrivals
        self._arrived = 0
        self._all_in = asyncio.Event()
        self.resumed: list[tuple[str, list[str], bool]] = []

    async def complete(self, task: str, **kwargs: Any) -> _Result:
        conv: _Conv = kwargs["conversation"]
        if self._arrivals is None:
            self.resumed.append((task, list(conv.messages), kwargs.get("resume", False)))
            return _Result(text=f"done:{task}")
        conv.messages.append(f"turn-of-{task}")
        self._arrived += 1
        if self._arrived == self._arrivals:
            self._all_in.set()
        await self._all_in.wait()
        self._halt.set()
        return _Result(paused=True)


def _map_flow(store: Any, halt: asyncio.Event, saia: _PausingSAIA) -> Any:
    loop = Loop(ROLE, saia=saia, conversation_factory=_ConvFactory())

    @verb(role=ROLE)
    async def item(ctx: Context[Any], name: str) -> Any:
        return await loop(ctx, name)

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, "loop-in-map")
        .with_checkpointer()
        .with_halt(halt)
        .map(lambda b: b.call(item), items=lambda _p, _c: ["a", "b"], max_concurrency=2)
    )


class TestLoopInMap:
    async def test_each_item_resumes_its_own_paused_turn(self, tmp_path: Path) -> None:
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        halt1 = asyncio.Event()
        await _map_flow(store, halt1, _PausingSAIA(halt1, arrivals=2)).run()

        halt2 = asyncio.Event()
        saia2 = _PausingSAIA(halt2, arrivals=None)
        await _map_flow(store, halt2, saia2).run(resume="latest")

        assert sorted(saia2.resumed) == [
            ("a", ["turn-of-a"], True),
            ("b", ["turn-of-b"], True),
        ]


class _ScriptedSAIA:
    """Pauses the turns named in ``pause`` (setting the halt); completes the rest.

    Records ``(task, conversation, resume)`` for every call.
    """

    def __init__(self, halt: asyncio.Event, pause: set[str]) -> None:
        self.role = ROLE
        self._halt = halt
        self._pause = pause
        self.calls: list[tuple[str, list[str], bool]] = []

    async def complete(self, task: str, **kwargs: Any) -> _Result:
        conv = kwargs["conversation"]
        resume = kwargs.get("resume", False)
        self.calls.append((task, list(getattr(conv, "messages", [])), resume))
        if hasattr(conv, "messages"):
            conv.messages.append(f"{'more' if resume else 'turn'}-of-{task}")
        if task in self._pause or self._halt.is_set():
            self._halt.set()
            return _Result(paused=True)
        return _Result(text=f"done:{task}")


def _two_calls_flow(store: Any, halt: asyncio.Event, saia: _ScriptedSAIA, factory: Any) -> Any:
    """One step calling two Loops, then a second step."""
    first = Loop(ROLE, name="first", saia=saia, conversation_factory=factory)
    second = Loop(ROLE, name="second", saia=saia, conversation_factory=factory)
    after: list[str] = []

    @verb(role=ROLE)
    async def both(ctx: Context[Any], _p: Any = None) -> Any:
        await first(ctx, "a")
        return await second(ctx, "b")

    @verb(role=ROLE)
    async def then(ctx: Context[Any], _p: Any = None) -> None:
        after.append("then")

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, "two-calls")
        .with_checkpointer()
        .with_halt(halt)
        .call(both)
        .then(then)
    )


class TestTurnsInOneStep:
    async def test_each_call_resumes_its_own_turn(self, tmp_path: Path) -> None:
        """Call 1 pauses and sets the halt; call 2 pauses on the halt. Both resume."""
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        halt1 = asyncio.Event()
        await _two_calls_flow(store, halt1, _ScriptedSAIA(halt1, {"a"}), _ConvFactory()).run()

        halt2 = asyncio.Event()
        saia2 = _ScriptedSAIA(halt2, set())
        await _two_calls_flow(store, halt2, saia2, _ConvFactory()).run(resume="latest")

        assert saia2.calls == [("a", ["turn-of-a"], True), ("b", ["turn-of-b"], True)]

    async def test_uncaptured_turn_reruns_its_step_from_the_start(self, tmp_path: Path) -> None:
        """Without a ConversationFactory the paused step still counts as interrupted."""
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        halt1 = asyncio.Event()
        await _two_calls_flow(store, halt1, _ScriptedSAIA(halt1, {"a"}), None).run()
        history = History(store, "two-calls")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"

        halt2 = asyncio.Event()
        saia2 = _ScriptedSAIA(halt2, set())
        await _two_calls_flow(store, halt2, saia2, None).run(resume="latest")

        assert [(task, resume) for task, _, resume in saia2.calls] == [("a", False), ("b", False)]
        assert await history.is_complete()


class TestTurnPausedAgain:
    async def test_second_pause_saves_the_continued_turn(self, tmp_path: Path) -> None:
        """A resumed turn that pauses again is saved with everything it did so far."""
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        runs = [({"a"}, None), ({"a"}, "latest"), (set(), "latest")]
        saias: list[_ScriptedSAIA] = []
        for pause, resume in runs:
            halt = asyncio.Event()
            saias.append(_ScriptedSAIA(halt, pause))
            flow = _two_calls_flow(store, halt, saias[-1], _ConvFactory())
            await (flow.run() if resume is None else flow.run(resume=resume))

        assert saias[2].calls[0] == ("a", ["turn-of-a", "more-of-a"], True)
        assert await History(store, "two-calls").is_complete()
