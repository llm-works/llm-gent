# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A running map's cursor: its items, and its done items with their results.

On resume the map runs over the saved items (``items`` is not evaluated
again): a completed item does not run again and its merge is not applied
again, nor does a failed (``strict=False``) or skipped one, a running item
continues where it was, and the rest run. The
cursor exists only while the map runs, so the next time the map runs, all
its items run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import (
    HALTED,
    Context,
    Factory,
    Failure,
    History,
    Interrupted,
    RestoredError,
    Skipped,
    verb,
)
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


class _Crash(BaseException):
    """The process died: no failure handling runs."""


class _Counts:
    def __init__(self) -> None:
        self.items_fn = 0
        self.bodies: list[int] = []
        self.merges: list[int] = []
        self.completed: list[int] = []


def _map_flow(
    store: Any,
    counts: _Counts,
    halt: asyncio.Event,
    *,
    halt_at: int | None = None,
    items: list[int] | None = None,
) -> Any:
    """Sequential ``.map(state=, merge=)``; the item equal to ``halt_at`` sets the halt."""
    values = [1, 2, 3] if items is None else items

    @verb
    async def work(ctx: Context[dict[str, Any]], item: int) -> int:
        counts.bodies.append(item)
        ctx.state.data["out"] = item * 10
        if item == halt_at:
            halt.set()  # completes: the items after it do not start
        return item * 10

    def items_fn(_prev: Any, _ctx: Any) -> list[int]:
        counts.items_fn += 1
        return values

    def merge(parent: dict[str, Any], child: dict[str, Any]) -> None:
        counts.merges.append(child["out"])
        parent.setdefault("merged", []).append(child["out"])

    def on_item_complete(item: int, _outcome: Any, _ctx: Any) -> None:
        counts.completed.append(item)

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, "map")
        .with_checkpointer()
        .with_halt(halt)
        .map(
            lambda b: b.call(work),
            items=items_fn,
            aggregate=sum,
            max_concurrency=1,
            state=lambda _p: {},
            merge=merge,
        )
        .on_item_complete(on_item_complete)
    )


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


class TestMapCursor:
    async def test_completed_items_do_not_run_or_merge_again(
        self, store: JsonFileCheckpointStore
    ) -> None:
        first = _Counts()
        assert await _map_flow(store, first, asyncio.Event(), halt_at=2).run() is HALTED
        assert (first.bodies, first.merges, first.completed) == ([1, 2], [10, 20], [1, 2])

        resumed = _Counts()
        assert await _map_flow(store, resumed, asyncio.Event()).run(resume="latest") == 60
        assert (resumed.bodies, resumed.merges, resumed.completed) == ([3], [30], [3])
        head = await History(store, "map").head()
        assert head is not None
        assert (await History(store, "map").snapshot(head)).root["merged"] == [10, 20, 30]

    async def test_items_are_not_evaluated_again(self, store: JsonFileCheckpointStore) -> None:
        await _map_flow(store, _Counts(), asyncio.Event(), halt_at=1).run()

        resumed = _Counts()
        result = await _map_flow(store, resumed, asyncio.Event(), items=[7, 8]).run(resume="latest")
        assert resumed.items_fn == 0
        assert resumed.bodies == [2, 3]
        assert result == 60

    async def test_halt_checkpoint_holds_items_and_completed_results(
        self, store: JsonFileCheckpointStore
    ) -> None:
        await _map_flow(store, _Counts(), asyncio.Event(), halt_at=2).run()

        history = History(store, "map")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        [cursor] = [c for c in (await history.snapshot(head)).cursors.values() if "items" in c]
        assert cursor["items"] == [1, 2, 3]
        assert cursor["done"] == {
            "0": {"result": 10, "merged": True},
            "1": {"result": 20, "merged": True},
        }

    async def test_next_run_of_the_map_runs_all_items(self, store: JsonFileCheckpointStore) -> None:
        """Once the map completed, its cursor is gone: running it again runs every item."""
        await _map_flow(store, _Counts(), asyncio.Event(), halt_at=2).run()
        await _map_flow(store, _Counts(), asyncio.Event()).run(resume="latest")

        again = _Counts()
        assert await _map_flow(store, again, asyncio.Event()).run(resume="latest") == 60
        assert again.bodies == [1, 2, 3]


class TestUnmergedItem:
    async def test_item_saved_before_its_merge_merges_without_running(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A checkpoint between an item's body and its merge: resume merges it, once.

        Item 1's merge waits until item 2 has checkpointed; item 2 then dies.
        The checkpoint holds item 1 as completed but not merged, with its
        scope.
        """
        saved = asyncio.Event()
        bodies: list[int] = []
        merges: list[int] = []

        def build(crash: bool) -> Any:
            @verb
            async def work(ctx: Context[dict[str, Any]], item: int) -> int:
                bodies.append(item)
                ctx.state.data["out"] = item
                if item == 2 and crash:
                    await ctx.checkpoint()
                    saved.set()
                    raise _Crash()
                return item

            async def merge(parent: dict[str, Any], child: dict[str, Any]) -> None:
                if crash:
                    await saved.wait()
                merges.append(child["out"])
                parent.setdefault("merged", []).append(child["out"])

            return (
                Factory(make_test_logger())
                .create(state={})
                .with_checkpoint_store(store, "unmerged")
                .with_checkpointer()
                .map(
                    lambda b: b.call(work),
                    items=lambda _p, _c: [1, 2],
                    max_concurrency=2,
                    state=lambda _p: {},
                    merge=merge,
                )
            )

        with pytest.raises(_Crash):
            await build(crash=True).run()
        bodies.clear()
        merges.clear()

        await build(crash=False).run(resume="latest")
        assert bodies == [2]
        assert sorted(merges) == [1, 2]
        head = await History(store, "unmerged").head()
        assert head is not None
        assert sorted((await History(store, "unmerged").snapshot(head)).root["merged"]) == [1, 2]


class TestWaveShape:
    async def test_halt_mid_wave_resumes_without_rerunning_finished_items(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """``.map(strict=False).guard(...)`` in parallel: halt mid-wave, then resume."""
        ran: list[int] = []

        def build(halt: asyncio.Event, halt_on: int | None) -> Any:
            @verb
            async def work(ctx: Context[dict[str, Any]], item: int) -> int:
                await asyncio.sleep(0)
                if ctx.halt is not None and ctx.halt.is_set():
                    raise Interrupted()
                ran.append(item)
                if item == halt_on:
                    halt.set()
                return item * 2

            return (
                Factory(make_test_logger())
                .create(state={})
                .with_checkpoint_store(store, "wave")
                .with_checkpointer()
                .with_halt(halt)
                .map(
                    lambda b: b.call(work),
                    items=lambda _p, _c: list(range(6)),
                    aggregate=lambda rs: sum(r for r in rs if isinstance(r, int)),
                    strict=False,
                    max_concurrency=3,
                )
                .guard(lambda item, _ctx: item != 5)
            )

        assert await build(asyncio.Event(), halt_on=1).run() is HALTED
        first = sorted(ran)
        ran.clear()

        result = await build(asyncio.Event(), halt_on=None).run(resume="latest")
        assert not set(first) & set(ran), f"finished items ran again: {set(first) & set(ran)}"
        assert sorted(first + ran) == [0, 1, 2, 3, 4]  # item 5 is Skipped by the guard
        assert result == sum(i * 2 for i in range(5))


class _Seen:
    def __init__(self) -> None:
        self.bodies: list[int] = []
        self.guards: list[int] = []
        self.errors: list[int] = []
        self.completed: list[int] = []
        self.caps: list[list[int]] = []


def _mixed_flow(
    store: Any,
    seen: _Seen,
    halt: asyncio.Event,
    *,
    halt_at: int | None,
    strict: bool = False,
    save_at: int | None = None,
) -> Any:
    """Items 1-4, one at a time: 1 fails, the guard skips 2, the item equal to ``halt_at`` halts.

    The item equal to ``save_at`` takes the checkpoint named ``"saved"``.
    """

    @verb
    async def work(ctx: Context[dict[str, Any]], item: int) -> int:
        seen.bodies.append(item)
        if item == 1:
            raise ValueError("bad item 1")
        if item == save_at:
            await ctx.checkpoint("saved")
        if item == halt_at:
            halt.set()  # completes: the items after it do not start
        return item * 10

    def guard(item: int, _ctx: Any) -> bool:
        seen.guards.append(item)
        return item != 2

    def cap(items: list[int], _ctx: Any) -> int:
        seen.caps.append(items)
        return 1

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, "mixed")
        .with_checkpointer()
        .with_halt(halt)
        .map(
            lambda b: b.call(work),
            items=lambda _p, _c: [1, 2, 3, 4],
            strict=strict,
            max_concurrency=cap,
        )
        .guard(guard)
        .on_error(lambda _e, item, _c: seen.errors.append(item))
        .on_item_complete(lambda item, _o, _c: seen.completed.append(item))
    )


class TestFailedAndSkippedItemsStayDone:
    """A failed (``strict=False``) or skipped item is done: a resumed map does not run it again."""

    async def test_they_do_not_run_again_and_keep_their_outcome(
        self, store: JsonFileCheckpointStore
    ) -> None:
        first = _Seen()
        assert await _mixed_flow(store, first, asyncio.Event(), halt_at=3).run() is HALTED
        assert (first.bodies, first.guards) == ([1, 3], [1, 2, 3])

        resumed = _Seen()
        result = await _mixed_flow(store, resumed, asyncio.Event(), halt_at=None).run(
            resume="latest"
        )
        assert (resumed.bodies, resumed.guards) == ([4], [4])
        assert (resumed.errors, resumed.completed) == ([], [4])
        failure, skipped, *rest = result
        assert isinstance(failure, Failure) and failure.item == 1
        assert isinstance(failure.exception, RestoredError)
        assert (failure.exception.type_name, failure.exception.message) == (
            "ValueError",
            "bad item 1",
        )
        assert str(failure.exception) == "ValueError: bad item 1"
        assert skipped == Skipped(item=2)
        assert rest == [30, 40]

    async def test_the_halt_checkpoint_records_them(self, store: JsonFileCheckpointStore) -> None:
        await _mixed_flow(store, _Seen(), asyncio.Event(), halt_at=3).run()

        history = History(store, "mixed")
        head = await history.head()
        assert head is not None
        [cursor] = [c for c in (await history.snapshot(head)).cursors.values() if "items" in c]
        assert cursor["done"] == {
            "0": {"failure": {"type": "ValueError", "message": "bad item 1"}},
            "1": {"skipped": True},
            "2": {"result": 30, "merged": True},
        }

    async def test_a_restored_failure_is_stored_again_unchanged(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A checkpoint the resumed run takes holds the first run's failure as it was."""
        await _mixed_flow(store, _Seen(), asyncio.Event(), halt_at=3).run()
        await _mixed_flow(store, _Seen(), asyncio.Event(), halt_at=None, save_at=4).run(
            resume="latest"
        )

        history = History(store, "mixed")
        saved = await history.checkpoint("saved")
        assert saved is not None
        [cursor] = [c for c in (await history.snapshot(saved)).cursors.values() if "items" in c]
        assert cursor["done"]["0"] == {"failure": {"type": "ValueError", "message": "bad item 1"}}

    async def test_the_computed_cap_is_computed_again_on_resume(
        self, store: JsonFileCheckpointStore
    ) -> None:
        first = _Seen()
        await _mixed_flow(store, first, asyncio.Event(), halt_at=3).run()
        resumed = _Seen()
        await _mixed_flow(store, resumed, asyncio.Event(), halt_at=None).run(resume="latest")
        assert first.caps == resumed.caps == [[1, 2, 3, 4]]

    async def test_a_strict_failure_runs_again(self, store: JsonFileCheckpointStore) -> None:
        """Strict: the failure raises out of the run; resume runs the item again.

        Item 3 completes after item 1 failed and saves its commit; that
        commit holds no record of item 1.
        """
        seen = _Seen()
        flow = _mixed_flow(store, seen, asyncio.Event(), halt_at=None, strict=True)
        flow.with_checkpoint_policy(on_map_item=True)
        with pytest.raises(ValueError, match="bad item 1"):
            await flow.run()
        assert seen.bodies == [1, 3, 4]

        resumed = _Seen()
        flow = _mixed_flow(store, resumed, asyncio.Event(), halt_at=None, strict=True)
        with pytest.raises(ValueError, match="bad item 1"):
            await flow.with_checkpoint_policy(on_map_item=True).run(resume="latest")
        assert resumed.bodies == [1]
        assert resumed.guards == [1]  # the skip of item 2 is recorded in strict maps too

    async def test_a_failed_merge_is_recorded(self, store: JsonFileCheckpointStore) -> None:
        """A merge that raises in a ``strict=False`` map: the item failed, and stays failed."""
        bodies: list[int] = []

        def build(halt: asyncio.Event, halt_at: int | None) -> Any:
            @verb
            async def work(ctx: Context[dict[str, Any]], item: int) -> int:
                bodies.append(item)
                ctx.state.data["out"] = item
                if item == halt_at:
                    halt.set()
                return item

            def merge(parent: dict[str, Any], child: dict[str, Any]) -> None:
                if child["out"] == 1:
                    raise KeyError("merge 1")
                parent.setdefault("merged", []).append(child["out"])

            return (
                Factory(make_test_logger())
                .create(state={})
                .with_checkpoint_store(store, "merge-fails")
                .with_checkpointer()
                .with_halt(halt)
                .map(
                    lambda b: b.call(work),
                    items=lambda _p, _c: [1, 2, 3],
                    strict=False,
                    max_concurrency=1,
                    state=lambda _p: {},
                    merge=merge,
                )
            )

        assert await build(asyncio.Event(), halt_at=2).run() is HALTED
        bodies.clear()
        failure, *rest = await build(asyncio.Event(), halt_at=None).run(resume="latest")
        assert bodies == [3]
        assert isinstance(failure, Failure)
        assert isinstance(failure.exception, RestoredError)
        assert failure.exception.type_name == "KeyError"
        assert rest == [2, 3]


class TestSerialization:
    async def test_items_that_cannot_be_stored_raise_naming_the_map(
        self, store: JsonFileCheckpointStore
    ) -> None:
        class _Opaque:
            pass

        @verb
        async def work(ctx: Context[dict[str, Any]], _item: Any) -> int:
            await ctx.checkpoint()
            return 1

        flow = (
            Factory(make_test_logger())
            .create(state={})
            .with_checkpoint_store(store, "opaque")
            .with_checkpointer()
            .map(lambda b: b.call(work), items=lambda _p, _c: [_Opaque()])
        )
        with pytest.raises(TypeError, match=r"/items'.*_Opaque"):
            await flow.run()
