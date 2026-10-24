# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The codec for step inputs and carried values: what comes back is what was stored.

Accepted: plain JSON, pydantic models, and objects with ``to_dict()`` plus a
classmethod ``from_dict()``. Everything else fails naming the value's path.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from llm_saia.core.conversation import Message, ToolCall
from llm_saia.core.trace import GuardOutcome, LLMCall, Step, ToolOutcome, VerbTrace
from llm_saia.core.types import LoopScore, TaskResult
from pydantic import BaseModel

from llm_gent.flow import (
    HALTED,
    Context,
    Factory,
    Failure,
    Interrupted,
    RestoredError,
    Skipped,
    verb,
)
from llm_gent.flow.state import StateDataclass
from llm_gent.flow.state.codec import decode, encode
from tests.helpers import memory_store

from .conftest import make_test_logger


pytestmark = pytest.mark.unit


class Note(BaseModel):
    """A pydantic model."""

    title: str
    tags: list[str]


@dataclass
class Point:
    """An object following the to_dict / from_dict protocol."""

    x: int
    y: int

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Point:
        return cls(**data)


@dataclass
class Pairs(StateDataclass):
    """A state dataclass with tuple fields: nested in a list, and top-level."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    fixed: tuple[str, ...] = ()


@dataclass
class OnlyToDict:
    """Has to_dict but no from_dict: cannot be rebuilt."""

    v: int

    def to_dict(self) -> dict[str, Any]:
        return {"v": self.v}


def _round_trip(value: Any) -> Any:
    """Encode, pass through JSON text as a store would, decode."""
    return decode(json.loads(json.dumps(encode(value, "v"))), "v")


class TestRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [None, True, 3, 2.5, "s", [1, "a", None], {"a": {"b": [1, 2]}}, {}, []],
    )
    def test_plain_json_is_stored_as_is(self, value: Any) -> None:
        assert encode(value, "v") == value
        assert _round_trip(value) == value

    def test_pydantic_model(self) -> None:
        note = Note(title="t", tags=["a"])
        back = _round_trip(note)
        assert type(back) is Note and back == note

    def test_to_dict_from_dict_object(self) -> None:
        back = _round_trip({"p": Point(1, 2), "ps": [Point(3, 4)]})
        assert back == {"p": Point(1, 2), "ps": [Point(3, 4)]}

    @pytest.mark.parametrize(
        "value",
        [(1, "a"), (), ((1, 2), [3, (4,)]), {"t": (1, 2)}, [(1, "a"), ("b", 2)]],
    )
    def test_a_tuple_comes_back_a_tuple(self, value: Any) -> None:
        back = _round_trip(value)
        assert back == value
        assert repr(back) == repr(value)  # tuples stay tuples, lists stay lists, at every depth

    def test_a_tuple_is_stored_tagged(self) -> None:
        assert encode((1, "a"), "v") == {"$type": "tuple", "$data": [1, "a"]}

    def test_a_state_dataclass_with_tuples_inside_lists(self) -> None:
        """``to_dict`` keeps a tuple nested in a list; ``from_dict`` gets it back as emitted."""
        value = Pairs(pairs=[("n1", "a"), ("n2", "b")], fixed=("x", "y"))
        back = _round_trip(value)
        assert type(back) is Pairs and back == value

    @pytest.mark.parametrize(
        "value",
        [
            {"$type": "tests.unit.flow.test_codec:Point", "$data": {"x": 1, "y": 2}},
            {"$type": "tuple", "$data": [1, 2]},
            {"$type": "dict", "$data": {}},
        ],
    )
    def test_a_dict_with_its_own_type_key_is_not_mistaken_for_a_typed_value(
        self, value: dict[str, Any]
    ) -> None:
        back = _round_trip(value)
        assert type(back) is dict and back == value

    def test_saia_task_result(self) -> None:
        """A paused Loop result with tool calls, score and trace comes back equal."""
        result = TaskResult(
            completed=False,
            output="partial",
            iterations=3,
            history=[
                Message(role="user", content="hi"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="c1", name="lookup", arguments={"q": "x"})],
                ),
                Message(role="tool", content="res", tool_call_id="c1"),
            ],
            reason="paused",
            paused=True,
            terminal_data={"k": [1, 2]},
            score=LoopScore(
                iterations=3, productive=2, nudges=1, skips=0, total_tokens=90, wasted_tokens=10
            ),
            trace=VerbTrace(
                verb="complete",
                steps=[
                    Step(
                        phase="iteration",
                        llm_call=LLMCall(call_id="a", input_tokens=5),
                        guards=[GuardOutcome(name="g", passed=False)],
                        tools=[ToolOutcome(name="lookup", call_id="c1")],
                    )
                ],
            ),
        )
        back = _round_trip(result)
        assert type(back) is TaskResult and back == result

    def test_map_sentinels_keep_their_type_and_item(self) -> None:
        """A failure's exception comes back a RestoredError with the original type and message."""
        stored = [Skipped(Point(1, 2)), Failure(ValueError("no"), Point(3, 4))]
        skipped, failure = _round_trip(stored)
        assert skipped == Skipped(Point(1, 2))
        assert type(failure) is Failure and failure.item == Point(3, 4)
        exc = failure.exception
        assert isinstance(exc, RestoredError)
        assert (exc.type_name, exc.message) == ("ValueError", "no")

    def test_a_restored_failure_is_stored_again_unchanged(self) -> None:
        failure = Failure(RestoredError("ValueError", "no"), "x")
        assert _round_trip(failure).to_dict() == failure.to_dict()


class TestRejected:
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ({1: "a"}, "a dict with non-str keys"),
            (float("nan"), "nan has no JSON form"),
            (OnlyToDict(1), "a value of type OnlyToDict cannot be checkpointed"),
            (object(), "a value of type object cannot be checkpointed"),
        ],
    )
    def test_encode_names_the_path(self, value: Any, message: str) -> None:
        with pytest.raises(TypeError, match=f"^cursor at 'x': {message}"):
            encode({"inner": [value]}, "cursor at 'x'")

    def test_a_class_defined_in_a_function_cannot_be_stored(self) -> None:
        @dataclass
        class Local:
            def to_dict(self) -> dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Local:
                return cls(**data)

        with pytest.raises(TypeError, match="defined inside a function"):
            encode(Local(), "v")

    def test_decode_uses_only_models_and_from_dict_classes(self) -> None:
        stored = {"$type": "json:JSONDecoder", "$data": {}}
        with pytest.raises(TypeError, match="not a pydantic model and has no from_dict"):
            decode(stored, "v")

    def test_decode_never_imports_a_module(self) -> None:
        stored = {"$type": "a_module_nobody_imported:Thing", "$data": {}}
        with pytest.raises(TypeError, match="in a module that is not imported"):
            decode(stored, "v")

    def test_data_that_no_longer_fits_the_model_names_the_path(self) -> None:
        stored = {"$type": "tests.unit.flow.test_codec:Note", "$data": {"title": 1}}
        with pytest.raises(TypeError, match=r"^v: stored .*Note cannot be rebuilt"):
            decode(stored, "v")


@pytest.mark.asyncio
class TestInAFlow:
    async def test_an_iterate_carrying_tuples_halts_and_resumes(self) -> None:
        """The carry holds tuples nested in a list: the halt checkpoint stores it, resume gets it back."""
        store = memory_store()

        def build(halt: asyncio.Event, arm: bool, seen: list[Pairs]) -> Any:
            @verb
            async def grow(ctx: Context[Any], carry: Pairs) -> Pairs:
                n = len(carry.pairs)
                if arm and n == 1:
                    halt.set()
                    raise Interrupted()
                seen.append(carry)
                return Pairs(pairs=[*carry.pairs, (f"n{n}", "a")], fixed=carry.fixed)

            return (
                Factory(make_test_logger())
                .create(state={})
                .with_checkpoint_store(store, "tuples")
                .with_halt(halt)
                .iterate(lambda b: b.call(grow), max_iters=3)
            )

        start = Pairs(fixed=("x",))
        assert await build(asyncio.Event(), True, []).run(start) is HALTED

        seen: list[Pairs] = []
        result = await build(asyncio.Event(), False, seen).run(start, resume="latest")
        assert seen[0] == Pairs(pairs=[("n0", "a")], fixed=("x",))
        assert result == Pairs(pairs=[("n0", "a"), ("n1", "a"), ("n2", "a")], fixed=("x",))

    async def test_a_map_result_holding_sentinels_is_stored_and_comes_back(self) -> None:
        """A map's result with a Skipped and a Failure, as an iterate's carry and a step's input.

        The ``on_iterate`` commit stores the first pass's result as the
        carry; the halt in the second pass's ``keep`` stores its input;
        resume hands ``keep`` the restored result.
        """
        store = memory_store()

        def build(halt: asyncio.Event, arm: bool, seen: list[Any]) -> Any:
            @verb
            async def work(ctx: Context[Any], item: str) -> str:
                if item == "boom":
                    raise ValueError("no luck")
                return item

            @verb
            async def keep(ctx: Context[Any], results: list[Any]) -> list[Any]:
                if arm and len(seen) == 1:
                    halt.set()
                    raise Interrupted()
                seen.append(results)
                return results

            ff = Factory(make_test_logger())
            batch = (
                ff.create()
                .map(lambda b: b.call(work), items=lambda *_: ["a", "skip", "boom"], strict=False)
                .guard(lambda item, _ctx: item != "skip")
            )
            return (
                ff.create(state={})
                .with_checkpoint_store(store, "sentinels")
                .with_checkpointer()
                .with_checkpoint_policy(on_iterate=True)
                .with_halt(halt)
                .iterate(lambda b: b.call(batch).then(keep), max_iters=2)
            )

        assert await build(asyncio.Event(), True, []).run() is HALTED

        seen: list[Any] = []
        result = await build(asyncio.Event(), False, seen).run(resume="latest")
        assert seen == [result]
        item, skipped, failure = result
        assert item == "a" and skipped == Skipped("skip")
        assert type(failure) is Failure and failure.item == "boom"
        exc = failure.exception
        assert isinstance(exc, RestoredError)
        assert (exc.type_name, exc.message) == ("ValueError", "no luck")
