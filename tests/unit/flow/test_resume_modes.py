# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""``Flow.run(resume="latest")``, and what a run that raises leaves behind.

``"latest"`` checks out the newest commit that has state, skipping
stateless commits, and continues every running chain, iterate and branch
at its saved cursor. A run that raises writes no commit: its history's
head is its last save.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, Interrupted, verb
from llm_gent.flow.stores import JsonFileCheckpointStore
from llm_gent.flow.testing.checkpoint import (
    CanonicalCounter,
    build_canonical_flow,
    resume_in_subprocess,
)

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A fresh store under an isolated temp root."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


def _counting_flow(
    store: JsonFileCheckpointStore,
    name: str,
    *,
    max_iters: int,
    fail_at: int | None = None,
    lg: Any = None,
    on_iterate: bool = False,
) -> Any:
    """Iterate a verb that bumps ``n``; raises once ``n`` reaches ``fail_at``."""

    @verb
    async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
        ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
        if fail_at is not None and ctx.state.data["n"] == fail_at:
            raise RuntimeError(f"boom at {fail_at}")
        return ctx.state.data["n"]

    flow = (
        FlowFactory(lg or make_test_logger())
        .create(state={})
        .with_checkpointer(store, name)
        .with_checkpoint_policy(on_iterate=on_iterate)
    )
    return flow.iterate(lambda body: body.call(bump), max_iters=max_iters)


def _capturing_logger() -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """A test logger whose warnings are recorded as ``(message, extra)``."""
    lg = make_test_logger()
    warnings: list[tuple[str, dict[str, Any]]] = []
    lg.warning = lambda msg, *_a, **kw: warnings.append((msg, kw.get("extra", {})))  # type: ignore[method-assign]
    return lg, warnings


async def _head(store: JsonFileCheckpointStore, name: str) -> Any:
    history = History(store, name)
    head = await history.head()
    assert head is not None
    return head, await history.snapshot(head)


class _Crash(BaseException):
    """The process died: not an ``Exception``, so nothing handles it on the way out."""


class TestLatest:
    async def test_after_completion_continues_from_final_state(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Terminal-extend: a finished run's final state is the next run's start."""
        await build_canonical_flow(
            make_test_logger(), max_iters=2, store=store, client_flow_id="extend"
        ).run()

        result = await build_canonical_flow(
            make_test_logger(),
            state=CanonicalCounter(n=100),
            max_iters=2,
            store=store,
            client_flow_id="extend",
        ).run(resume="latest")

        # Restored 2 iterations + 2 more; the fallback state (n=100) is unused.
        assert result["iterations_completed"] == 4
        assert result["log"][0] == 1

    async def test_after_halt_matches_the_uninterrupted_run(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The halted pass continues; max_iters is a bound across runs."""
        baseline = await build_canonical_flow(make_test_logger(), max_iters=5).run()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=asyncio.Event(),
            halt_after_iteration=2,
            store=store,
            client_flow_id="halted",
        ).run()

        result = await build_canonical_flow(
            make_test_logger(), max_iters=5, store=store, client_flow_id="halted"
        ).run(resume="latest")
        assert result == baseline

    async def test_after_failure_continues_from_last_save_point(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A run that raises writes nothing: the head is its last save, where resume continues."""
        with pytest.raises(RuntimeError, match="boom at 3"):
            await _counting_flow(store, "failed", max_iters=5, fail_at=3, on_iterate=True).run()

        lg, warnings = _capturing_logger()
        result = await _counting_flow(store, "failed", max_iters=5, lg=lg).run(
            state={"n": 100}, resume="latest"
        )
        assert result == 5  # the save point after pass 2 (n=2), then passes 2-4
        assert not warnings

    async def test_failure_before_any_save_point_uses_given_state(
        self, store: JsonFileCheckpointStore
    ) -> None:
        with pytest.raises(RuntimeError):
            await _counting_flow(store, "failed-early", max_iters=5, fail_at=2).run()
        assert await History(store, "failed-early").head() is None

        result = await _counting_flow(store, "failed-early", max_iters=1).run(
            state={"n": 10}, resume="latest"
        )
        assert result == 11

    async def test_failure_after_completion_continues_from_the_final_state(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A run that raised after a finished one leaves the finished run's final state as head."""
        await _counting_flow(store, "end-then-fail", max_iters=2).run()
        history = History(store, "end-then-fail")
        finished = await history.head()
        with pytest.raises(RuntimeError):
            await _counting_flow(store, "end-then-fail", max_iters=5, fail_at=1).run()

        head = await history.head()
        assert head is not None and finished is not None
        assert head.content_hash == finished.content_hash

        result = await _counting_flow(store, "end-then-fail", max_iters=1).run(
            state={"n": 10}, resume="latest"
        )
        assert result == 3  # n=2 from the first run's $end, then one pass

    async def test_skips_stateless_final_commit(self, store: JsonFileCheckpointStore) -> None:
        """A final commit without state (unserializable) is walked past, not a reason to reset."""
        ran: list[str] = []

        @verb
        async def bump(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            ran.append("bump")
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            ctx.state.data.pop("handle", None)
            return ctx.state.data["n"]

        @verb
        async def attach(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            ran.append("attach")
            ctx.state.data["handle"] = object()
            return ctx.state.data["n"]

        def _flow(lg: Any = None) -> Any:
            return (
                FlowFactory(lg or make_test_logger())
                .create(state={})
                .with_checkpointer(store, "stateless-end")
                .with_checkpoint_policy(on_iterate=True)
                .iterate(lambda body: body.call(bump), max_iters=2)
                .call(attach)
            )

        await _flow().run()
        head, snapshot = await _head(store, "stateless-end")
        assert History.is_final_state(head) and not snapshot.has_state

        ran.clear()
        lg, warnings = _capturing_logger()
        result = await _flow(lg).run(state={"n": 100}, resume="latest")
        # The last iterate commit: both passes done, so only the step after it runs.
        assert (result, ran) == (2, ["attach"])
        skipped = [extra for msg, extra in warnings if msg.startswith("resume skipped")]
        assert len(skipped) == 1
        assert skipped[0]["skipped"] == ["$end"]
        assert skipped[0]["restored_node_path"] != "$end"

    async def test_newest_commit_restores_without_warning(
        self, store: JsonFileCheckpointStore
    ) -> None:
        await _counting_flow(store, "quiet", max_iters=2).run()
        lg, warnings = _capturing_logger()
        await _counting_flow(store, "quiet", max_iters=1, lg=lg).run(resume="latest")
        assert warnings == []

    async def test_corrupt_history_raises(self, store: JsonFileCheckpointStore) -> None:
        from llm_gent.flow import HistoryCorrupt

        await _counting_flow(store, "corrupt", max_iters=2).run()
        head, _ = await _head(store, "corrupt")
        commits_dir = store._history_dir(head.meta.flow_id) / "objects" / "commit"
        (commits_dir / head.content_hash).unlink()

        with pytest.raises(HistoryCorrupt):
            await _counting_flow(store, "corrupt", max_iters=1).run(resume="latest")

    async def test_empty_history_uses_given_state(self, store: JsonFileCheckpointStore) -> None:
        result = await _counting_flow(store, "empty", max_iters=1).run(
            state={"n": 10}, resume="latest"
        )
        assert result == 11


def _adding_flow(
    store: Any, name: str, log: list[Any], *, wrap: str, crash_after: int | None
) -> Any:
    """An iterate of 4 passes each returning ``prev + 10``, optionally nested; crashes on request."""

    @verb
    async def step(ctx: Context[dict[str, Any]], prev: Any = None) -> Any:
        log.append(prev)
        if crash_after is not None and len(log) == crash_after:
            raise _Crash()
        return prev + 10

    def _iterate(f: Any) -> Any:
        return f.iterate(lambda b: b.call(step), max_iters=4)

    flow = (
        FlowFactory(make_test_logger())
        .create(state={})
        .with_checkpointer(store, name)
        .with_checkpoint_policy(on_iterate=True)
    )
    if wrap == "call":
        return flow.call(_iterate(FlowFactory(make_test_logger()).create()))
    if wrap == "outer-iterate":
        return flow.iterate(_iterate, max_iters=1)
    return _iterate(flow)


class TestCursors:
    async def test_iterate_stopped_by_until_does_not_run_again(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """``until`` stopped the loop after pass 1; resume from its ``on_iterate`` commit ends it."""
        passes: list[int] = []

        def build(crash: bool) -> Any:
            @verb
            async def step(ctx: Context[dict[str, Any]], x: int) -> int:
                passes.append(x)
                return x + 1

            @verb
            async def after(ctx: Context[dict[str, Any]], x: int) -> int:
                if crash:
                    raise _Crash()
                return x

            return (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpointer(store, "until")
                .with_checkpoint_policy(on_iterate=True)
                .iterate(lambda b: b.call(step), max_iters=5, until=lambda r, _c: r == 2)
                .then(after)
            )

        with pytest.raises(_Crash):
            await build(crash=True).run(0)
        assert passes == [0, 1]
        passes.clear()

        assert await build(crash=False).run(0, resume="latest") == 2
        assert passes == []

    @pytest.mark.parametrize("wrap", ["top", "call", "outer-iterate"])
    async def test_iterate_continues_with_its_carried_value(
        self, store: JsonFileCheckpointStore, wrap: str
    ) -> None:
        """Each pass builds on the last; a crash in pass 3 resumes it with pass 2's result."""
        with pytest.raises(_Crash):
            await _adding_flow(store, "carry", [], wrap=wrap, crash_after=3).run(0)
        log: list[Any] = []
        resumed = _adding_flow(store, "carry", log, wrap=wrap, crash_after=None)
        assert await resumed.run(0, resume="latest") == 40
        assert log == [20, 30]  # passes 3 and 4 only

    @pytest.mark.parametrize(
        ("mode", "resumed"),
        [("bail", [10, 20, 30]), ("finish", [20, 30])],
        ids=["interrupted-step-runs-again", "completed-step-does-not"],
    )
    @pytest.mark.parametrize("shape", ["chain", "iterate"])
    async def test_step_during_which_the_halt_arrived(
        self, store: JsonFileCheckpointStore, shape: str, mode: str, resumed: list[int]
    ) -> None:
        """The 2nd step sets the halt, then raises ``Interrupted`` or finishes its work."""

        def build(halt: asyncio.Event | None, seen: list[Any]) -> Any:
            @verb
            async def step(ctx: Context[dict[str, Any]], prev: Any = None) -> Any:
                if ctx.halt is not None and ctx.halt.is_set():
                    raise Interrupted()
                seen.append(prev)
                if halt is not None and len(seen) == 2:
                    halt.set()
                    if mode == "bail":
                        raise Interrupted()
                return prev + 10

            flow = FlowFactory(make_test_logger()).create(state={}).with_checkpointer(store, "h")
            if halt is not None:
                flow = flow.with_halt(halt)
            if shape == "chain":
                return flow.call(step).call(step).call(step).call(step)
            return flow.iterate(lambda b: b.call(step), max_iters=4)

        assert await build(asyncio.Event(), []).run(0) is None
        seen: list[Any] = []
        assert await build(None, seen).run(0, resume="latest") == 40
        assert seen == resumed

    async def test_interrupted_without_a_halt_is_an_error(
        self, store: JsonFileCheckpointStore
    ) -> None:
        @verb
        async def step(ctx: Context[dict[str, Any]], prev: Any = None) -> Any:
            raise Interrupted()

        flow = FlowFactory(make_test_logger()).create(state={}).with_checkpointer(store, "no-halt")
        with pytest.raises(RuntimeError, match="raised Interrupted while no halt is set"):
            await flow.call(step).run(0)

    async def test_branch_takes_its_saved_arm(self, store: JsonFileCheckpointStore) -> None:
        """``when`` is not evaluated again: the state it read has changed since."""
        calls: list[str] = []

        def build(crash: bool) -> Any:
            @verb
            async def then_step(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
                calls.append("then")
                ctx.state.data["flipped"] = True
                if crash:
                    await ctx.checkpoint()
                    raise _Crash()
                return x

            @verb
            async def else_step(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
                calls.append("else")
                return x

            return (
                FlowFactory(make_test_logger())
                .create(state={})
                .with_checkpointer(store, "arm")
                .branch(
                    when=lambda _p, ctx: not ctx.state.data.get("flipped"),
                    then=lambda b: b.call(then_step),
                    else_=lambda b: b.call(else_step),
                )
            )

        with pytest.raises(_Crash):
            await build(crash=True).run(0)
        calls.clear()
        await build(crash=False).run(0, resume="latest")
        assert calls == ["then"]

    async def test_a_removed_step_fails_the_resume(self, store: JsonFileCheckpointStore) -> None:
        """A deploy removed the step the chain was at: there is nowhere to continue."""
        with pytest.raises(_Crash):
            await _scoped_step_flow(store, "deploy", make_test_logger()).run()
        with pytest.raises(RuntimeError, match="which this chain no longer has"):
            await _plain_step_flow(store, "deploy", make_test_logger()).run(resume="latest")


def _scoped_map_flow(
    store: JsonFileCheckpointStore, name: str, seen: list[dict[str, Any]], *, crash: bool
) -> Any:
    """``.map(state=)`` item → ``.call(state=)`` → a leaf recording the scope it runs under.

    With ``crash``, the leaf marks its scope, checkpoints, and the process dies.
    """

    @verb
    async def leaf(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
        seen.append(dict(ctx.state.data))
        if crash:
            ctx.state.data["hit"] = True
            await ctx.checkpoint()
            raise _Crash()
        return x

    sub = FlowFactory(make_test_logger()).create().call(leaf)
    item = (
        FlowFactory(make_test_logger())
        .create()
        .call(sub, state=lambda _p: {"tag": "call"}, merge=lambda _p, _c: None)
    )
    return (
        FlowFactory(make_test_logger())
        .create(state={})
        .with_checkpointer(store, name)
        .map(
            lambda b: b.call(item),
            items=lambda _p, _c: [1],
            max_concurrency=1,
            state=lambda _p: {"tag": "item"},
            merge=lambda _p, _c: None,
        )
    )


@verb
async def _work_then_crash(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
    """Record work in the scope, checkpoint it, then the process dies."""
    ctx.state.data["w"] = 1
    await ctx.checkpoint()
    raise _Crash()


@verb
async def _plain(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
    ctx.state.data["plain"] = True
    return x


def _scoped_step_flow(store: JsonFileCheckpointStore, name: str, lg: Any) -> Any:
    """``.call(sub, state=)`` whose leaf checkpoints its scope and crashes."""
    flow = FlowFactory(lg).create(state={}).with_checkpointer(store, name)
    sub = FlowFactory(lg).create().call(_work_then_crash)
    return flow.call(sub, state=lambda _p: {}, merge=lambda p, c: p.update(c))


def _plain_step_flow(store: JsonFileCheckpointStore, name: str, lg: Any) -> Any:
    """The same history after a deploy that replaced the scoped step with a plain one."""
    return FlowFactory(lg).create(state={}).with_checkpointer(store, name).call(_plain)


def _bounded_iterate_flow(
    store: JsonFileCheckpointStore, name: str, lg: Any, max_iters: int, halt: Any = None
) -> Any:
    """``.iterate`` over a ``.call(state=)`` pass; the pass that gets 2 checkpoints and crashes.

    Run with ``0``: passes 0 and 1 complete, pass 2 saves its scope and
    cursors and crashes. A resume with a lower ``max_iters`` never
    reaches pass 2.
    """

    @verb
    async def work(ctx: Context[dict[str, Any]], x: int) -> int:
        ctx.state.data["x"] = x
        if x == 2:
            await ctx.checkpoint()
            raise _Crash()
        return x + 1

    body = FlowFactory(lg).create().call(work)
    flow = FlowFactory(lg).create(state={}).with_checkpointer(store, name)
    if halt is not None:
        flow = flow.with_halt(halt)
    return flow.iterate(
        lambda b: b.call(body, state=lambda _p: {}, merge=lambda _p, _c: None),
        max_iters=max_iters,
    )


class TestScopesFromSnapshot:
    async def test_scope_under_map_item_gets_its_own_payload(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Each saved scope comes back to the block that owns it, not to the one above it."""
        with pytest.raises(_Crash):
            await _scoped_map_flow(store, "nested", [], crash=True).run()

        seen: list[dict[str, Any]] = []
        await _scoped_map_flow(store, "nested", seen, crash=False).run(resume="latest")
        assert seen == [{"tag": "call", "hit": True}]

    async def test_finished_run_drops_entries_it_never_reached(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The iterate's bound dropped below the saved pass: its scope and cursors go, with a warning."""
        with pytest.raises(_Crash):
            await _bounded_iterate_flow(store, "shrink", make_test_logger(), 5).run(0)
        history = History(store, "shrink")
        head = await history.head()
        assert head is not None and (await history.snapshot(head)).scopes

        lg, warnings = _capturing_logger()
        await _bounded_iterate_flow(store, "shrink", lg, 2).run(resume="latest")

        head = await history.head()
        assert head is not None and await history.is_complete()
        assert (await history.snapshot(head)).scopes == {}
        assert [msg for msg, _ in warnings] == [
            "resumed run finished without reaching saved entries; dropped them"
        ]

    async def test_halted_run_keeps_entries_it_has_not_reached(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A run that halts before reaching a saved scope keeps it for the next resume."""
        with pytest.raises(_Crash):
            await _bounded_iterate_flow(store, "halt-early", make_test_logger(), 5).run(0)

        halt = asyncio.Event()
        halt.set()
        flow = _bounded_iterate_flow(store, "halt-early", make_test_logger(), 5, halt=halt)
        await flow.run(resume="latest")

        history = History(store, "halt-early")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        scopes = (await history.snapshot(head)).scopes
        assert [v for p, v in scopes.items() if "/p/2/" in p] == [{"x": 2}]


class TestFailure:
    async def test_failure_writes_no_commit_and_reraises(
        self, store: JsonFileCheckpointStore
    ) -> None:
        with pytest.raises(RuntimeError, match="boom at 2"):
            await _counting_flow(store, "fail", max_iters=5, fail_at=2).run()
        assert await History(store, "fail").head() is None

    async def test_failure_leaves_the_last_save_as_head(
        self, store: JsonFileCheckpointStore
    ) -> None:
        with pytest.raises(RuntimeError, match="boom at 2"):
            await _counting_flow(store, "fail-saved", max_iters=5, fail_at=2, on_iterate=True).run()

        head, snapshot = await _head(store, "fail-saved")
        assert head.meta.outcome == "ok"
        assert snapshot.root == {"n": 1}  # the save after pass 0, not the half-done pass 1
        assert not await History(store, "fail-saved").is_complete()

    async def test_cancellation_writes_no_commit(self, store: JsonFileCheckpointStore) -> None:
        @verb
        async def cancel(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            raise asyncio.CancelledError()

        flow = (
            FlowFactory(make_test_logger())
            .create(state={})
            .with_checkpointer(store, "cancelled")
            .call(cancel)
        )
        with pytest.raises(asyncio.CancelledError):
            await flow.run()
        assert await History(store, "cancelled").head() is None

    async def test_empty_flow_leaves_no_history(self, store: JsonFileCheckpointStore) -> None:
        flow = FlowFactory(make_test_logger()).create(state={}).with_checkpointer(store, "empty")
        with pytest.raises(RuntimeError, match="no nodes to run"):
            await flow.run()
        assert await History(store, "empty").flow_id() is None


class TestResumeModeValidation:
    @pytest.mark.parametrize("mode", [True, "", "complete"])
    async def test_values_that_are_neither_mode_nor_checkpoint_name_are_rejected(
        self, store: JsonFileCheckpointStore, mode: Any
    ) -> None:
        with pytest.raises(ValueError, match="resume must be one of"):
            await _counting_flow(store, "bad-mode", max_iters=1).run(resume=mode)

    @pytest.mark.parametrize("mode", ["replay", "restart"])
    async def test_removed_modes_name_missing_checkpoints(
        self, store: JsonFileCheckpointStore, mode: str
    ) -> None:
        with pytest.raises(ValueError, match=f"no checkpoint named {mode!r}"):
            await _counting_flow(store, "bad-mode", max_iters=1).run(resume=mode)

    async def test_resume_requires_checkpointer(self) -> None:
        @verb
        async def noop(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            return None

        flow = FlowFactory(make_test_logger()).create(state={}).call(noop)
        with pytest.raises(RuntimeError, match="resume='latest'"):
            await flow.run(resume="latest")


class TestResumeInSubprocess:
    """A resume in a fresh process matches the same resume run in-process."""

    async def _halt_at_two(self, store: JsonFileCheckpointStore) -> None:
        await build_canonical_flow(
            make_test_logger(),
            max_iters=4,
            halt=asyncio.Event(),
            halt_after_iteration=2,
            store=store,
            client_flow_id="xproc",
        ).run()

    async def test_matches_in_process_resume(self, tmp_path: Path) -> None:
        in_proc = JsonFileCheckpointStore(make_test_logger(), tmp_path / "in-proc")
        await self._halt_at_two(in_proc)
        await self._halt_at_two(JsonFileCheckpointStore(make_test_logger(), tmp_path / "xproc"))

        expected = await build_canonical_flow(
            make_test_logger(), max_iters=4, store=in_proc, client_flow_id="xproc"
        ).run(resume="latest")
        result = resume_in_subprocess(
            store_module="llm_gent.flow.stores",
            store_factory="JsonFileCheckpointStore",
            store_kwargs={"root": str(tmp_path / "xproc")},
            flow_builder_kwargs={"max_iters": 4},
            client_flow_id="xproc",
            resume="latest",
        )
        assert result == dict(expected)

    async def test_off_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a resume mode"):
            resume_in_subprocess(
                store_module="llm_gent.flow.stores",
                store_factory="JsonFileCheckpointStore",
                store_kwargs={"root": "unused"},
                resume="off",
            )
