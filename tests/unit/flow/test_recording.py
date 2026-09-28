# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Execution record written during runs, and carried by the commits a run writes.

Each test reads the record from a commit the run wrote: the ``$failed``
commit of a run whose last step raises, a halt commit, or a
``ctx.checkpoint()`` commit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, Flow, FlowFactory, History, Loop, verb
from llm_gent.flow._node_id import _compute_node_ids
from llm_gent.flow.state.record import ExecutionRecord, RecordError, value_hash
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import ROLE_A, make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

LG = make_test_logger()
NAME = "rec"


@dataclass
class Point:
    x: int


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    return JsonFileCheckpointStore(LG, tmp_path / "cp")


class Boom(Exception):
    """Raised by the last step so the run writes a ``$failed`` commit."""


@verb
async def boom(ctx: Context[dict[str, Any]]) -> None:
    raise Boom("stop")


@verb
async def double(ctx: Context[dict[str, Any]], x: int) -> int:
    return 2 * x


@verb
async def inc(ctx: Context[dict[str, Any]], x: int) -> int:
    return x + 1


@dataclass
class TurnResult:
    """A SAIA-style result; ``paused`` is what Loop checks."""

    value: int
    paused: bool = False


class PausingSAIA:
    """Echoes the task back as the result, paused for the tasks in ``pause_on``."""

    def __init__(self, pause_on: set[int]) -> None:
        self.pause_on = pause_on

    async def complete(self, task: Any, **kwargs: Any) -> TurnResult:
        return TurnResult(value=task, paused=task in self.pause_on)


@verb
async def unwrap(ctx: Context[dict[str, Any]], r: TurnResult) -> int:
    return r.value + 1


def _flow(store: JsonFileCheckpointStore) -> Flow:
    return FlowFactory(LG).create(state={}).with_checkpointer(store, NAME)


async def _head_record(store: JsonFileCheckpointStore) -> ExecutionRecord:
    """The record of the newest commit that carries one.

    A completed run's final-state commit carries none, so a run that
    wrote a ``ctx.checkpoint()`` commit and then completed is read from
    that commit.
    """
    history = History(store, NAME)
    async for commit in history.commits():
        record = await history.record(commit)
        if record is not None:
            return record
    raise AssertionError("no commit carries a record")


async def _run_failing(flow: Flow, *args: Any) -> ExecutionRecord:
    """Run ``flow`` with a raising last step; return the ``$failed`` commit's record."""
    flow.call(boom)
    with pytest.raises(Boom):
        await flow.run(*args)
    return await _head_record(flow._checkpoint_ctx.store)  # type: ignore[union-attr]


def _kinds(record: ExecutionRecord) -> list[str]:
    return [key.split("|", 1)[0] for key, _ in record.items()]


class TestSteps:
    async def test_steps_record_input_hash_and_output(self, store: JsonFileCheckpointStore) -> None:
        flow = _flow(store).call(double).call(inc)
        ids = _compute_node_ids("", flow._nodes)
        record = await _run_failing(flow, 3)
        assert record.get(f"s|{ids[0]}") == {"in": value_hash(((3,), {})), "out": 6}
        assert record.get(f"s|{ids[1]}") == {"in": value_hash(((6,), {})), "out": 7}
        assert _kinds(record) == ["s", "s"]  # the raising step is not recorded

    async def test_unneeded_unstorable_output_is_recorded_without_it(
        self, store: JsonFileCheckpointStore
    ) -> None:
        @verb
        async def handle(ctx: Context[dict[str, Any]]) -> object:
            return object()

        flow = _flow(store).call(handle)
        record = await _run_failing(flow)
        ((_, entry),) = record.items()
        assert "out" not in entry and "in" in entry

    async def test_needed_unstorable_output_raises(self, store: JsonFileCheckpointStore) -> None:
        @verb
        async def handle(ctx: Context[dict[str, Any]]) -> object:
            return object()

        @verb
        async def consume(ctx: Context[dict[str, Any]], h: object) -> None:
            return None

        flow = _flow(store).call(handle).call(consume)
        with pytest.raises(RecordError, match="a later node receives it"):
            await flow.run()

    async def test_rescue_does_not_swallow_a_refusal(self, store: JsonFileCheckpointStore) -> None:
        @verb
        async def handle(ctx: Context[dict[str, Any]]) -> object:
            return object()

        inner = FlowFactory(LG).create().call(handle).call(double)
        flow = _flow(store).call(inner, rescue=lambda _exc, _in, _ctx: 0)
        with pytest.raises(RecordError):
            await flow.run()

    async def test_unstorable_run_input_raises_before_the_step_runs(
        self, store: JsonFileCheckpointStore
    ) -> None:
        ran: list[int] = []

        @verb
        async def take(ctx: Context[dict[str, Any]], x: Any) -> None:
            ran.append(1)

        with pytest.raises(RecordError, match="input of take"):
            await _flow(store).call(take).run(object())
        assert ran == []

    async def test_dropped_input_does_not_need_to_be_storable(
        self, store: JsonFileCheckpointStore
    ) -> None:
        @verb
        async def handle(ctx: Context[dict[str, Any]]) -> object:
            return object()

        @verb
        async def ignore(ctx: Context[dict[str, Any]]) -> int:
            return 1

        await _flow(store).call(handle).call(ignore).run()

    async def test_typed_outputs_are_recorded(self, store: JsonFileCheckpointStore) -> None:
        @verb
        async def make(ctx: Context[dict[str, Any]]) -> Point:
            return Point(1)

        @verb
        async def read(ctx: Context[dict[str, Any]], p: Point) -> int:
            return p.x

        record = await _run_failing(_flow(store).call(make).call(read))
        assert len(record) == 2

    async def test_no_record_without_a_checkpointer(self) -> None:
        @verb
        async def handle(ctx: Context[dict[str, Any]]) -> object:
            return object()

        @verb
        async def consume(ctx: Context[dict[str, Any]], h: object) -> None:
            return None

        flow = FlowFactory(LG).create(state={}).call(handle).call(consume)
        await flow.run()
        assert flow._recorder is None


class TestIterate:
    async def test_passes_and_body_steps_have_their_own_addresses(
        self, store: JsonFileCheckpointStore
    ) -> None:
        flow = _flow(store).iterate(lambda b: b.call(inc), until=lambda r, _c: r >= 3)
        (iterate_id,) = _compute_node_ids("", flow._nodes)
        record = await _run_failing(flow, 0)
        passes = [(k, e) for k, e in record.items() if k.startswith("p|")]
        assert passes == [
            (f"p|{iterate_id}@i0", {"out": 1, "cont": True}),
            (f"p|{iterate_id}@i1", {"out": 2, "cont": True}),
            (f"p|{iterate_id}@i2", {"out": 3, "cont": False}),
        ]
        body_steps = [k for k, _ in record.items() if k.startswith("s|") and "@" in k]
        assert [k.rsplit("@", 1)[1] for k in body_steps] == ["i0", "i1", "i2"]
        assert f"s|{iterate_id}" in record

    async def test_body_result_must_be_storable(self, store: JsonFileCheckpointStore) -> None:
        @verb
        async def handle(ctx: Context[dict[str, Any]], _p: Any = None) -> object:
            return object()

        flow = _flow(store).iterate(lambda b: b.call(handle), max_iters=2)
        with pytest.raises(RecordError):
            await flow.run()


class TestMap:
    async def test_items_and_item_results_are_recorded(
        self, store: JsonFileCheckpointStore
    ) -> None:
        flow = _flow(store).map(
            lambda b: b.call(double),
            items=lambda _p, _c: [1, 2, 3],
            aggregate=sum,
        )
        (map_id,) = _compute_node_ids("", flow._nodes)
        record = await _run_failing(flow)
        assert record.get(f"i|{map_id}") == {"items": [1, 2, 3]}
        items = {k: e for k, e in record.items() if k.startswith("m|")}
        assert items == {
            f"m|{map_id}@n0": {"out": 2},
            f"m|{map_id}@n1": {"out": 4},
            f"m|{map_id}@n2": {"out": 6},
        }
        assert record.get(f"s|{map_id}")["out"] == 12

    async def test_guard_skip_is_recorded(self, store: JsonFileCheckpointStore) -> None:
        flow = _flow(store).map(lambda b: b.call(double), items=lambda _p, _c: [1, 2])
        flow.guard(lambda item, _c: item != 2)
        (map_id,) = _compute_node_ids("", flow._nodes)
        record = await _run_failing(flow)
        assert record.get(f"m|{map_id}@n1") == {"skip": True}

    async def test_unstorable_items_raise(self, store: JsonFileCheckpointStore) -> None:
        flow = _flow(store).map(lambda b: b.call(double), items=lambda _p, _c: [object()])
        with pytest.raises(RecordError, match="items of map"):
            await flow.run()

    async def test_refusal_inside_a_non_strict_map_is_not_a_failure_item(
        self, store: JsonFileCheckpointStore
    ) -> None:
        @verb
        async def handle(ctx: Context[dict[str, Any]], _x: Any) -> object:
            return object()

        flow = _flow(store).map(lambda b: b.call(handle), items=lambda _p, _c: [1], strict=False)
        with pytest.raises(RecordError):
            await flow.run()

    async def test_a_refused_item_result_is_not_merged(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The body's last step refuses the result inside the item, before its merge.

        ``_on_success`` has no rollback for a refusal, so the ``$failed``
        state must not hold the item's merge.
        """

        @verb
        async def handle(ctx: Context[dict[str, Any]], _x: Any) -> object:
            ctx.state.data["n"] = 1
            return object()

        flow = _flow(store).map(
            lambda b: b.call(handle),
            items=lambda _p, _c: [1],
            state=lambda _p: {"n": 0},
            merge=lambda p, c: p.__setitem__("total", c["n"]),
        )
        with pytest.raises(RecordError):
            await flow.run()
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        assert await history.scopes(head) == [{}]


class TestBranch:
    @pytest.mark.parametrize(("value", "arm"), [(1, "then"), (0, "none")])
    async def test_verdict_is_recorded(
        self, store: JsonFileCheckpointStore, value: int, arm: str
    ) -> None:
        flow = _flow(store).branch(when=lambda x, _c: x > 0, then=lambda b: b.call(inc))
        (branch_id,) = _compute_node_ids("", flow._nodes)
        record = await _run_failing(flow, value)
        assert record.get(f"b|{branch_id}") == {"arm": arm}


class TestInterruption:
    async def test_nothing_is_recorded_after_the_halt(self, store: JsonFileCheckpointStore) -> None:
        halt = asyncio.Event()

        @verb
        async def stop(ctx: Context[dict[str, Any]], x: int) -> int:
            halt.set()
            return x

        flow = _flow(store).with_halt(halt).call(double).call(stop).call(inc)
        ids = _compute_node_ids("", flow._nodes)
        await flow.run(1)
        record = await _head_record(store)
        assert [k for k, _ in record.items()] == [f"s|{ids[0]}"]

    async def test_the_paused_step_and_the_steps_after_it_are_not_recorded(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The paused step is incomplete; ``unwrap`` consumed it, so it is not recorded either."""
        loop = Loop(ROLE_A, saia=PausingSAIA(pause_on={2}))
        flow = _flow(store).call(double).call(loop).call(unwrap).call(boom)
        double_id = _compute_node_ids("", flow._nodes)[0]
        with pytest.raises(Boom):
            await flow.run(1)
        record = await _head_record(store)
        assert [k for k, _ in record.items()] == [f"s|{double_id}"]

    async def test_a_pause_leaves_sibling_map_items_recordable(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Item 0's Loop pauses; item 1 completes and is recorded.

        The map step holds the pause, so it is not recorded. Inside item 0
        neither step is recorded; inside item 1 both are.
        """
        loop = Loop(ROLE_A, saia=PausingSAIA(pause_on={0}))
        flow = _flow(store).map(
            lambda b: b.call(loop).call(unwrap), items=lambda _p, _c: [0, 1], max_concurrency=1
        )
        flow.call(boom)
        map_id = _compute_node_ids("", flow._nodes)[0]
        with pytest.raises(Boom):
            await flow.run()
        keys = [k for k, _ in (await _head_record(store)).items()]
        assert f"m|{map_id}@n1" in keys
        assert f"m|{map_id}@n0" not in keys
        assert f"s|{map_id}" not in keys
        step_coords = sorted(k.rsplit("@", 1)[1] for k in keys if k.startswith("s|"))
        assert step_coords == ["n1", "n1"]

    async def test_passes_after_a_paused_pass_are_not_recorded(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Pass 1 pauses; pass 2 completes but was fed by pass 1, so only pass 0 is recorded."""
        loop = Loop(ROLE_A, saia=PausingSAIA(pause_on={1}))
        flow = _flow(store).iterate(lambda b: b.call(loop).call(unwrap), max_iters=3)
        flow.call(boom)
        iterate_id = _compute_node_ids("", flow._nodes)[0]
        with pytest.raises(Boom):
            await flow.run(0)
        passes = [k for k, _ in (await _head_record(store)).items() if k.startswith("p|")]
        assert passes == [f"p|{iterate_id}@i0"]

    async def test_an_unrecorded_step_passes_an_unstorable_value_on(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """After the halt a step's input is not hashed, so a sentinel it receives is not refused."""
        halt = asyncio.Event()
        cancelled = object()

        @verb
        async def work(ctx: Context[dict[str, Any]], _x: Any) -> object:
            halt.set()
            return cancelled

        @verb
        async def finish(ctx: Context[dict[str, Any]], r: object) -> int:
            return 0 if r is cancelled else 1

        flow = _flow(store).with_halt(halt)
        flow.iterate(lambda b: b.call(work).call(finish), max_iters=2)
        assert await flow.run(1) == 0

    async def test_a_subflow_halt_leaves_the_enclosing_step_unrecorded(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A subflow's own halt event stops recording for the rest of the run."""
        local = asyncio.Event()

        @verb
        async def stop(ctx: Context[dict[str, Any]], x: int) -> int:
            local.set()
            return x

        sub = FlowFactory(LG).create(name="sub").with_halt(local).call(stop).call(inc)
        flow = _flow(store).call(double).call(sub).call(inc)
        double_id = _compute_node_ids("", flow._nodes)[0]
        record = await _run_failing(flow, 1)
        assert [k for k, _ in record.items()] == [f"s|{double_id}"]


class TestNestedCheckpointers:
    async def test_a_composed_subflow_with_its_own_checkpointer_is_refused(
        self, store: JsonFileCheckpointStore
    ) -> None:
        sub = FlowFactory(LG).create().with_checkpointer(store, "sub").call(inc)
        with pytest.raises(RuntimeError, match="has its own checkpointer"):
            await _flow(store).call(sub).run(1)

    async def test_refused_without_a_top_level_checkpointer_too(
        self, store: JsonFileCheckpointStore
    ) -> None:
        sub = FlowFactory(LG).create().with_checkpointer(store, "sub").call(inc)
        flow = FlowFactory(LG).create().iterate(lambda b: b.call(sub), max_iters=1)
        with pytest.raises(RuntimeError, match="has its own checkpointer"):
            await flow.run(1)


class TestScopes:
    async def test_open_scopes_are_saved_and_completed_ones_dropped(
        self, store: JsonFileCheckpointStore
    ) -> None:
        @verb
        async def bump(ctx: Context[dict[str, Any]], x: int) -> int:
            ctx.state.data["n"] += x
            if x == 2:
                raise Boom("item 2")
            return x

        flow = _flow(store).map(
            lambda b: b.call(bump),
            items=lambda _p, _c: [1, 2],
            state=lambda _p: {"n": 0},
            merge=lambda _p, _c: None,
            max_concurrency=1,
        )
        (map_id,) = _compute_node_ids("", flow._nodes)
        with pytest.raises(Boom):
            await flow.run()
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        assert await history.open_scopes(head) == {f"{map_id}@n1": {"n": 2}}

    async def test_scope_stack_reads_are_unchanged(self, store: JsonFileCheckpointStore) -> None:
        """History.scopes still returns only the scope stack, root first."""
        flow = _flow(store).call(double)
        await _run_failing(flow, 1)
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        assert await history.scopes(head) == [{}]


class TestCommitWrites:
    async def test_unchanged_record_shards_are_not_rewritten(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A second commit with the same record puts no record blobs again."""
        puts: list[str] = []
        original = store.put_object

        def counting(flow_id: str, kind: Any, content_hash: str, payload: bytes) -> None:
            puts.append(content_hash)
            original(flow_id, kind, content_hash, payload)

        store.put_object = counting  # type: ignore[method-assign]

        @verb
        async def save_twice(ctx: Context[dict[str, Any]], _x: int) -> None:
            await ctx.checkpoint()
            await ctx.checkpoint()

        await _flow(store).call(double).call(save_twice).run(1)
        assert len(puts) == len(set(puts))
