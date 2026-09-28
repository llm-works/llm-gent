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
from llm_gent.flow.state.record import ExecutionRecord, RecordError, decode_value, value_hash
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

        flow = _flow(store).call(make).call(read)
        make_id = _compute_node_ids("", flow._nodes)[0]
        record = await _run_failing(flow)
        assert len(record) == 2
        assert decode_value(record.get(f"s|{make_id}")["out"]) == Point(1)

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

    async def test_an_unlocated_pause_stops_recording_for_the_rest_of_the_run(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A pause reported without a node id cannot be placed: nothing after it is recorded."""

        @verb
        async def pause_without_location(ctx: Context[dict[str, Any]], x: int) -> int:
            assert ctx._env is not None and ctx._env.recorder is not None
            ctx._env.recorder.mark_paused(ctx._env, None)
            return x

        flow = _flow(store).call(double).call(pause_without_location).call(inc).call(boom)
        double_id = _compute_node_ids("", flow._nodes)[0]
        with pytest.raises(Boom):
            await flow.run(1)
        assert [k for k, _ in (await _head_record(store)).items()] == [f"s|{double_id}"]

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


class AsyncStore:
    """Delegates to a sync store, yielding to the event loop before every call.

    In-tree stores are synchronous; an async store lets other tasks run
    in the middle of a commit, which is what these tests need.
    """

    def __init__(self, inner: JsonFileCheckpointStore) -> None:
        self._inner = inner

    @property
    def retention(self) -> Any:
        return self._inner.retention

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        async def call(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return attr(*args, **kwargs)

        return call


def _fail_first_commit(store: JsonFileCheckpointStore) -> None:
    """Make the store's first commit put raise ``OSError("disk full")``."""
    original = store.put_object
    failed: list[str] = []

    def flaky(flow_id: str, kind: Any, content_hash: str, payload: bytes) -> None:
        if kind == "commit" and not failed:
            failed.append(content_hash)
            raise OSError("disk full")
        original(flow_id, kind, content_hash, payload)

    store.put_object = flaky  # type: ignore[method-assign]


@verb
async def add_to_scope(ctx: Context[dict[str, Any]], x: int) -> int:
    ctx.state.data["n"] += x
    return x


def _counting_map(*, strict: bool) -> dict[str, Any]:
    """``.map`` kwargs: items [1, 2], one at a time, each adding itself to a scoped ``n``."""
    return {
        "body": lambda b: b.call(add_to_scope),
        "items": lambda _p, _c: [1, 2],
        "state": lambda _p: {"n": 0},
        "merge": lambda p, c: p.setdefault("total", []).append(c["n"]),
        "max_concurrency": 1,
        "strict": strict,
    }


class TestRecordStateConsistency:
    """A commit never records an instance whose state effects it does not hold."""

    async def test_a_subflow_after_an_unrecorded_step_records_nothing_inside(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The prefix rule crosses descents: the subflow consumed the paused step's result."""
        loop = Loop(ROLE_A, saia=PausingSAIA(pause_on={1}))
        sub = FlowFactory(LG).create().call(unwrap).call(inc)
        flow = _flow(store).call(loop).call(sub).call(boom)
        with pytest.raises(Boom):
            await flow.run(1)
        assert (await _head_record(store)).items() == []

    async def test_a_pass_after_an_unrecorded_pass_records_nothing_inside(
        self, store: JsonFileCheckpointStore
    ) -> None:
        loop = Loop(ROLE_A, saia=PausingSAIA(pause_on={1}))
        flow = _flow(store).iterate(lambda b: b.call(loop).call(unwrap), max_iters=3)
        flow.call(boom)
        with pytest.raises(Boom):
            await flow.run(0)
        keys = [k for k, _ in (await _head_record(store)).items()]
        assert [k for k in keys if k.endswith(("@i1", "@i2"))] == []

    async def test_a_failed_per_item_commit_leaves_the_item_unrecorded(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Item 0's commit fails, so its merge is rolled back: it must not stay recorded.

        The non-strict map step still returns, so it is recorded and item
        0 never runs again: its reopened scope is dropped with the step.
        """
        _fail_first_commit(store)
        flow = _flow(store).with_checkpoint_policy(on_map_item=True)
        flow.map(**_counting_map(strict=False)).call(boom)
        map_id = _compute_node_ids("", flow._nodes)[0]
        with pytest.raises(Boom):
            await flow.run()
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        record = await history.record(head)
        assert record is not None
        assert f"m|{map_id}@n0" not in record
        assert f"m|{map_id}@n1" in record
        assert f"s|{map_id}" in record
        assert (await history.scopes(head))[0]["total"] == [2]
        assert await history.open_scopes(head) == {}

    async def test_a_failed_per_item_commit_in_a_strict_map_keeps_the_item_scope(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A strict map raises past the rolled-back item: the failure commit carries its scope."""
        _fail_first_commit(store)
        flow = _flow(store).with_checkpoint_policy(on_map_item=True)
        flow.map(**_counting_map(strict=True))
        map_id = _compute_node_ids("", flow._nodes)[0]
        with pytest.raises(OSError, match="disk full"):
            await flow.run()
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        record = await history.record(head)
        assert record is not None
        assert f"m|{map_id}@n0" not in record
        assert f"s|{map_id}" not in record
        # Item 1 is not cancelled when the strict map raises, so only item 0 is asserted on.
        assert 1 not in (await history.scopes(head))[0].get("total", [])
        assert (await history.open_scopes(head))[f"{map_id}@n0"] == {"n": 1}

    async def test_commits_with_an_async_store_hold_what_they_record(self, tmp_path: Path) -> None:
        """Items that run while a commit awaits the store must not be recorded without their effects."""
        store = AsyncStore(JsonFileCheckpointStore(LG, tmp_path / "cp"))

        @verb
        async def mark(ctx: Context[dict[str, Any]], x: int) -> int:
            ctx.state.data[f"s{x}"] = True
            return x

        flow = (
            FlowFactory(LG)
            .create(state={})
            .with_checkpointer(store, NAME)  # type: ignore[arg-type]
            .with_checkpoint_policy(on_map_item=True)
            .map(lambda b: b.call(mark), items=lambda _p, _c: [0, 1, 2, 3], max_concurrency=4)
        )
        await flow.run()
        history = History(store, NAME)  # type: ignore[arg-type]
        checked = 0
        async for commit in history.commits():
            record = await history.record(commit)
            if record is None:
                continue
            root = (await history.scopes(commit))[0]
            for key, _ in record.items():
                if key.startswith("s|") and "@n" in key:
                    item = key.rsplit("@n", 1)[1]
                    assert root.get(f"s{item}"), f"{key} recorded, its effect missing from root"
                    checked += 1
        assert checked > 0

    async def test_a_failed_per_item_commit_without_state_keeps_sibling_writes(
        self, tmp_path: Path
    ) -> None:
        """Without ``state=`` items write the shared state; a rollback must not erase siblings."""
        inner = JsonFileCheckpointStore(LG, tmp_path / "cp")
        _fail_first_commit(inner)
        store = AsyncStore(inner)

        @verb
        async def mark(ctx: Context[dict[str, Any]], x: int) -> int:
            if x == 1:
                for _ in range(3):
                    await asyncio.sleep(0)
            ctx.state.data[f"s{x}"] = True
            return x

        state: dict[str, Any] = {}
        flow = (
            FlowFactory(LG)
            .create(state=state)
            .with_checkpointer(store, NAME)  # type: ignore[arg-type]
            .with_checkpoint_policy(on_map_item=True)
            .map(lambda b: b.call(mark), items=lambda _p, _c: [0, 1], strict=False)
        )
        flow.call(boom)
        with pytest.raises(Boom):
            await flow.run()
        assert state == {"s0": True, "s1": True}
        history = History(store, NAME)  # type: ignore[arg-type]
        head = await history.head()
        assert head is not None
        assert (await history.scopes(head))[0] == {"s0": True, "s1": True}


class TestScopeLifecycle:
    async def test_an_unserializable_scope_is_refused_when_opened(
        self, store: JsonFileCheckpointStore
    ) -> None:
        sub = FlowFactory(LG).create().call(inc)
        flow = _flow(store).call(sub, state=lambda _p: {"handle": object()})
        with pytest.raises(RecordError, match="scope"):
            await flow.run(1)

    async def test_a_scope_that_turns_unserializable_still_leaves_a_failure_commit(
        self, store: JsonFileCheckpointStore
    ) -> None:
        @verb
        async def poison(ctx: Context[dict[str, Any]], _x: int) -> None:
            ctx.state.data["handle"] = object()
            raise Boom("after poisoning the scope")

        sub = FlowFactory(LG).create().call(poison)
        flow = _flow(store).call(sub, state=lambda _p: {})
        with pytest.raises(Boom):
            await flow.run(1)
        head = await History(store, NAME).head()
        assert head is not None and History.is_failed(head)

    async def test_a_rescued_owner_closes_its_scope(self, store: JsonFileCheckpointStore) -> None:
        @verb
        async def fail(ctx: Context[dict[str, Any]], _x: int) -> None:
            raise ValueError("inner")

        sub = FlowFactory(LG).create().call(fail)
        flow = _flow(store).call(sub, state=lambda _p: {"n": 0}, rescue=lambda *_: 0)
        flow.call(boom)
        with pytest.raises(Boom):
            await flow.run(1)
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        assert await history.open_scopes(head) == {}

    async def test_scopes_nested_in_a_rescued_step_are_closed(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The rescue returns past an inner iterate that raised with its scope open."""

        @verb
        async def fail(ctx: Context[dict[str, Any]], _x: Any = None) -> None:
            raise ValueError("inner")

        inner = (
            FlowFactory(LG)
            .create()
            .iterate(lambda b: b.call(fail), max_iters=2, state=lambda _p: {"n": 0})
        )
        flow = _flow(store).call(inner, rescue=lambda *_: 0).call(boom)
        with pytest.raises(Boom):
            await flow.run(1)
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        assert await history.open_scopes(head) == {}

    async def test_scopes_after_an_unrecorded_step_are_closed(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A walk that stopped recording (after a pause) still finishes, and merges, its passes."""
        loop = Loop(ROLE_A, saia=PausingSAIA(pause_on={1}))
        sub = FlowFactory(LG).create().call(inc)
        flow = _flow(store).call(loop).call(unwrap)
        flow.iterate(lambda b: b.call(sub, state=lambda _p: {"n": 0}), max_iters=3)
        with pytest.raises(Boom):
            await flow.call(boom).run(1)
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        assert await history.open_scopes(head) == {}

    async def test_subflow_and_iterate_scopes_are_saved_while_open(
        self, store: JsonFileCheckpointStore
    ) -> None:
        @verb
        async def bump_then_fail(ctx: Context[dict[str, Any]], _x: Any = None) -> int:
            ctx.state.data["n"] += 1
            if ctx.state.data["n"] == 2:
                raise Boom("second pass")
            return ctx.state.data["n"]

        inner = (
            FlowFactory(LG)
            .create()
            .iterate(lambda b: b.call(bump_then_fail), max_iters=3, state=lambda _p: {"n": 0})
        )
        flow = _flow(store).call(inner, state=lambda _p: {"outer": True})
        call_id = _compute_node_ids("", flow._nodes)[0]
        with pytest.raises(Boom):
            await flow.run()
        history = History(store, NAME)
        head = await history.head()
        assert head is not None
        scopes = await history.open_scopes(head)
        assert scopes[call_id] == {"outer": True}
        (iterate_scope,) = [v for k, v in scopes.items() if k != call_id]
        assert iterate_scope == {"n": 2}

    async def test_the_recorder_is_dropped_when_the_run_ends(
        self, store: JsonFileCheckpointStore
    ) -> None:
        flow = _flow(store).call(double)
        await flow.run(1)
        assert flow._recorder is None
        failing = _flow(store).call(boom)
        with pytest.raises(Boom):
            await failing.run()
        assert failing._recorder is None
