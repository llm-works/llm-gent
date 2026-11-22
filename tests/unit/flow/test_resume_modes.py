# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""``Flow.run(resume="latest")``, ``Flow.run(restart=...)``, and what a run that raises leaves behind.

``"latest"`` checks out the newest commit that has state, skipping
stateless commits, and continues every running chain, iterate and branch
at its saved cursor. ``restart`` checks out a commit the same way but
takes its root state and declared resources alone, so the run starts at
its first step. A run that raises writes no commit: its history's head is
its last save.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from appinfra.cas import MemoryStore

from llm_gent.flow import HALTED, Context, Factory, History, Interrupted, verb
from llm_gent.flow._resume import _restart_snapshot
from llm_gent.flow.state.snapshot import (
    CHAIN,
    RESOURCES,
    RUN_RESOURCES,
    SIGNALS,
    TURN,
    Snapshot,
)
from llm_gent.flow.testing.checkpoint import (
    CanonicalCounter,
    build_canonical_flow,
    resume_in_subprocess,
)
from tests.helpers import memory_store

from .conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def store() -> MemoryStore:
    """A fresh in-memory store."""
    return memory_store()


def _counting_flow(
    store: MemoryStore,
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
        Factory(lg or make_test_logger(), checkpoint_store=store)
        .create()
        .with_checkpointer()
        .with_checkpoint_policy(on_iterate=on_iterate)
    )
    return flow.iterate(lambda body: body.call(bump), max_iters=max_iters)


def _capturing_logger() -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """A test logger whose warnings are recorded as ``(message, extra)``."""
    lg = make_test_logger()
    warnings: list[tuple[str, dict[str, Any]]] = []
    lg.warning = lambda msg, *_a, **kw: warnings.append((msg, kw.get("extra", {})))  # type: ignore[method-assign]
    return lg, warnings


async def _head(store: MemoryStore, name: str) -> Any:
    history = History(store, name)
    head = await history.head()
    assert head is not None
    return head, await history.snapshot(head)


class _Crash(BaseException):
    """The process died: not an ``Exception``, so nothing handles it on the way out."""


class TestLatest:
    async def test_after_completion_continues_from_final_state(self, store: MemoryStore) -> None:
        """Terminal-extend: a finished run's final state is the next run's start."""
        await build_canonical_flow(make_test_logger(), max_iters=2, store=store).run(
            history="extend"
        )

        result = await build_canonical_flow(make_test_logger(), max_iters=2, store=store).run(
            state=CanonicalCounter(n=100), history="extend", resume="latest"
        )

        # Restored 2 iterations + 2 more; the fallback state (n=100) is unused.
        assert result["iterations_completed"] == 4
        assert result["log"][0] == 1

    async def test_after_halt_matches_the_uninterrupted_run(self, store: MemoryStore) -> None:
        """The halted pass continues; max_iters is a bound across runs."""
        baseline = await build_canonical_flow(make_test_logger(), max_iters=5).run()
        await build_canonical_flow(
            make_test_logger(), max_iters=5, halt_after_iteration=2, store=store
        ).run(halt=asyncio.Event(), history="halted")

        result = await build_canonical_flow(make_test_logger(), max_iters=5, store=store).run(
            history="halted", resume="latest"
        )
        assert result == baseline

    async def test_after_failure_continues_from_last_save_point(self, store: MemoryStore) -> None:
        """A run that raises writes nothing: the head is its last save, where resume continues."""
        with pytest.raises(RuntimeError, match="boom at 3"):
            await _counting_flow(store, max_iters=5, fail_at=3, on_iterate=True).run(
                history="failed"
            )

        lg, warnings = _capturing_logger()
        result = await _counting_flow(store, max_iters=5, lg=lg).run(
            state={"n": 100}, history="failed", resume="latest"
        )
        assert result == 5  # the save point after pass 2 (n=2), then passes 2-4
        assert not warnings

    async def test_failure_before_any_save_point_uses_given_state(self, store: MemoryStore) -> None:
        with pytest.raises(RuntimeError):
            await _counting_flow(store, max_iters=5, fail_at=2).run(history="failed-early")
        assert await History(store, "failed-early").head() is None

        result = await _counting_flow(store, max_iters=1).run(
            state={"n": 10}, history="failed-early", resume="latest"
        )
        assert result == 11

    async def test_failure_after_completion_continues_from_the_final_state(
        self, store: MemoryStore
    ) -> None:
        """A run that raised after a finished one leaves the finished run's final state as head."""
        await _counting_flow(store, max_iters=2).run(history="end-then-fail")
        history = History(store, "end-then-fail")
        finished = await history.head()
        with pytest.raises(RuntimeError):
            await _counting_flow(store, max_iters=5, fail_at=1).run(history="end-then-fail")

        head = await history.head()
        assert head is not None and finished is not None
        assert head.id == finished.id

        result = await _counting_flow(store, max_iters=1).run(
            state={"n": 10}, history="end-then-fail", resume="latest"
        )
        assert result == 3  # n=2 from the first run's $end, then one pass

    async def test_skips_stateless_final_commit(self, store: MemoryStore) -> None:
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
                Factory(lg or make_test_logger(), checkpoint_store=store)
                .create()
                .with_checkpointer()
                .with_checkpoint_policy(on_iterate=True)
                .iterate(lambda body: body.call(bump), max_iters=2)
                .call(attach)
            )

        await _flow().run(history="stateless-end")
        head, snapshot = await _head(store, "stateless-end")
        assert History.is_final_state(head) and not snapshot.has_state

        ran.clear()
        lg, warnings = _capturing_logger()
        result = await _flow(lg).run(state={"n": 100}, history="stateless-end", resume="latest")
        # The last iterate commit: both passes done, so only the step after it runs.
        assert (result, ran) == (2, ["attach"])
        skipped = [extra for msg, extra in warnings if msg.startswith("resume skipped")]
        assert len(skipped) == 1
        assert skipped[0]["skipped"] == ["$end"]
        assert skipped[0]["restored_node_path"] != "$end"

    async def test_newest_commit_restores_without_warning(self, store: MemoryStore) -> None:
        await _counting_flow(store, max_iters=2).run(history="quiet")
        lg, warnings = _capturing_logger()
        await _counting_flow(store, max_iters=1, lg=lg).run(history="quiet", resume="latest")
        assert warnings == []

    async def test_corrupt_history_raises(self, store: MemoryStore) -> None:
        from llm_gent.flow import HistoryCorrupt

        await _counting_flow(store, max_iters=2).run(history="corrupt")
        head, _ = await _head(store, "corrupt")
        store.delete_objects("corrupt", [head.id])

        with pytest.raises(HistoryCorrupt):
            await _counting_flow(store, max_iters=1).run(history="corrupt", resume="latest")

    async def test_empty_history_uses_given_state(self, store: MemoryStore) -> None:
        result = await _counting_flow(store, max_iters=1).run(
            state={"n": 10}, history="empty", resume="latest"
        )
        assert result == 11


def _adding_flow(store: Any, log: list[Any], *, wrap: str, crash_after: int | None) -> Any:
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
        Factory(make_test_logger(), checkpoint_store=store)
        .create()
        .with_checkpointer()
        .with_checkpoint_policy(on_iterate=True)
    )
    if wrap == "call":
        return flow.call(_iterate(Factory(make_test_logger()).create()))
    if wrap == "outer-iterate":
        return flow.iterate(_iterate, max_iters=1)
    return _iterate(flow)


class TestCursors:
    async def test_iterate_stopped_by_until_does_not_run_again(self, store: MemoryStore) -> None:
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
                Factory(make_test_logger(), checkpoint_store=store)
                .create()
                .with_checkpointer()
                .with_checkpoint_policy(on_iterate=True)
                .iterate(lambda b: b.call(step), max_iters=5, until=lambda r, _c: r == 2)
                .then(after)
            )

        with pytest.raises(_Crash):
            await build(crash=True).run(0, history="until")
        assert passes == [0, 1]
        passes.clear()

        assert await build(crash=False).run(0, history="until", resume="latest") == 2
        assert passes == []

    @pytest.mark.parametrize("wrap", ["top", "call", "outer-iterate"])
    async def test_iterate_continues_with_its_carried_value(
        self, store: MemoryStore, wrap: str
    ) -> None:
        """Each pass builds on the last; a crash in pass 3 resumes it with pass 2's result."""
        with pytest.raises(_Crash):
            await _adding_flow(store, [], wrap=wrap, crash_after=3).run(0, history="carry")
        log: list[Any] = []
        resumed = _adding_flow(store, log, wrap=wrap, crash_after=None)
        assert await resumed.run(0, history="carry", resume="latest") == 40
        assert log == [20, 30]  # passes 3 and 4 only

    @pytest.mark.parametrize(
        ("mode", "resumed"),
        [("bail", [10, 20, 30]), ("finish", [20, 30])],
        ids=["interrupted-step-runs-again", "completed-step-does-not"],
    )
    @pytest.mark.parametrize("shape", ["chain", "iterate"])
    async def test_step_during_which_the_halt_arrived(
        self, store: MemoryStore, shape: str, mode: str, resumed: list[int]
    ) -> None:
        """The 2nd step sets the halt, then raises ``Interrupted`` or finishes its work."""

        def build(arm: bool, seen: list[Any]) -> Any:
            @verb
            async def step(ctx: Context[dict[str, Any]], prev: Any = None) -> Any:
                if ctx.halt is not None and ctx.halt.is_set():
                    raise Interrupted()
                seen.append(prev)
                if arm and len(seen) == 2:
                    ctx.halt.set()  # type: ignore[union-attr]
                    if mode == "bail":
                        raise Interrupted()
                return prev + 10

            flow = Factory(make_test_logger(), checkpoint_store=store).create()
            flow = flow.with_checkpointer()
            if shape == "chain":
                return flow.call(step).call(step).call(step).call(step)
            return flow.iterate(lambda b: b.call(step), max_iters=4)

        assert await build(True, []).run(0, halt=asyncio.Event(), history="h") is HALTED
        seen: list[Any] = []
        assert await build(False, seen).run(0, history="h", resume="latest") == 40
        assert seen == resumed

    async def test_interrupted_without_a_halt_is_an_error(self, store: MemoryStore) -> None:
        @verb
        async def step(ctx: Context[dict[str, Any]], prev: Any = None) -> Any:
            raise Interrupted()

        flow = Factory(make_test_logger(), checkpoint_store=store).create()
        flow = flow.with_checkpointer()
        with pytest.raises(RuntimeError, match="raised Interrupted while no halt is set"):
            await flow.call(step).run(0, history="no-halt")

    async def test_branch_takes_its_saved_arm(self, store: MemoryStore) -> None:
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
                Factory(make_test_logger(), checkpoint_store=store)
                .create()
                .with_checkpointer()
                .branch(
                    when=lambda _p, ctx: not ctx.state.data.get("flipped"),
                    then=lambda b: b.call(then_step),
                    else_=lambda b: b.call(else_step),
                )
            )

        with pytest.raises(_Crash):
            await build(crash=True).run(0, history="arm")
        calls.clear()
        await build(crash=False).run(0, history="arm", resume="latest")
        assert calls == ["then"]

    async def test_a_removed_step_fails_the_resume(self, store: MemoryStore) -> None:
        """A deploy removed the step the chain was at: there is nowhere to continue."""
        with pytest.raises(_Crash):
            await _scoped_step_flow(store, make_test_logger()).run(history="deploy")
        with pytest.raises(RuntimeError, match="which this chain no longer has"):
            await _plain_step_flow(store, make_test_logger()).run(history="deploy", resume="latest")


def _scoped_map_flow(store: MemoryStore, seen: list[dict[str, Any]], *, crash: bool) -> Any:
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

    sub = Factory(make_test_logger()).create().call(leaf)
    item = (
        Factory(make_test_logger())
        .create()
        .call(sub, state=lambda _p: {"tag": "call"}, merge=lambda _p, _c: None)
    )
    return (
        Factory(make_test_logger(), checkpoint_store=store)
        .create()
        .with_checkpointer()
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


def _scoped_step_flow(store: MemoryStore, lg: Any) -> Any:
    """``.call(sub, state=)`` whose leaf checkpoints its scope and crashes."""
    flow = Factory(lg, checkpoint_store=store).create().with_checkpointer()
    sub = Factory(lg).create().call(_work_then_crash)
    return flow.call(sub, state=lambda _p: {}, merge=lambda p, c: p.update(c))


def _plain_step_flow(store: MemoryStore, lg: Any) -> Any:
    """The same history after a deploy that replaced the scoped step with a plain one."""
    return Factory(lg, checkpoint_store=store).create().with_checkpointer().call(_plain)


def _bounded_iterate_flow(store: MemoryStore, lg: Any, max_iters: int) -> Any:
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

    body = Factory(lg).create().call(work)
    flow = Factory(lg, checkpoint_store=store).create().with_checkpointer()
    return flow.iterate(
        lambda b: b.call(body, state=lambda _p: {}, merge=lambda _p, _c: None),
        max_iters=max_iters,
    )


class TestScopesFromSnapshot:
    async def test_scope_under_map_item_gets_its_own_payload(self, store: MemoryStore) -> None:
        """Each saved scope comes back to the block that owns it, not to the one above it."""
        with pytest.raises(_Crash):
            await _scoped_map_flow(store, [], crash=True).run(history="nested")

        seen: list[dict[str, Any]] = []
        await _scoped_map_flow(store, seen, crash=False).run(history="nested", resume="latest")
        assert seen == [{"tag": "call", "hit": True}]

    async def test_finished_run_drops_entries_it_never_reached(self, store: MemoryStore) -> None:
        """The iterate's bound dropped below the saved pass: its scope and cursors go, with a warning."""
        with pytest.raises(_Crash):
            await _bounded_iterate_flow(store, make_test_logger(), 5).run(0, history="shrink")
        history = History(store, "shrink")
        head = await history.head()
        assert head is not None and (await history.snapshot(head)).scopes

        lg, warnings = _capturing_logger()
        await _bounded_iterate_flow(store, lg, 2).run(history="shrink", resume="latest")

        head = await history.head()
        assert head is not None and await history.is_complete()
        assert (await history.snapshot(head)).scopes == {}
        assert [msg for msg, _ in warnings] == [
            "resumed run finished without reaching saved entries; dropped them"
        ]

    async def test_halted_run_keeps_entries_it_has_not_reached(self, store: MemoryStore) -> None:
        """A run that halts before reaching a saved scope keeps it for the next resume."""
        with pytest.raises(_Crash):
            await _bounded_iterate_flow(store, make_test_logger(), 5).run(0, history="halt-early")

        halt = asyncio.Event()
        halt.set()
        flow = _bounded_iterate_flow(store, make_test_logger(), 5)
        await flow.run(history="halt-early", resume="latest", halt=halt)

        history = History(store, "halt-early")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        scopes = (await history.snapshot(head)).scopes
        assert [v for p, v in scopes.items() if "/p/2/" in p] == [{"x": 2}]


def _counting_chain(store: MemoryStore, arm: bool, seen: list[Any]) -> Any:
    """Four steps, each adding 10 to its input and 1 to state ``n``; armed, the 2nd halts the run."""

    @verb
    async def step(ctx: Context[dict[str, Any]], prev: Any = None) -> Any:
        seen.append(prev)
        ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
        if arm and len(seen) == 2:
            ctx.halt.set()  # type: ignore[union-attr]
            raise Interrupted()
        return prev + 10

    flow = Factory(make_test_logger(), checkpoint_store=store).create()
    return flow.with_checkpointer().call(step).call(step).call(step).call(step)


@verb
async def _record_state(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
    """Append the state the step started with to state ``seen``."""
    before = dict(ctx.state.data)
    ctx.state.data["seen"] = [*ctx.state.data.get("seen", []), before]
    return x


class TestRestart:
    async def test_a_removed_step_does_not_stop_it(self, store: MemoryStore) -> None:
        """Where resume fails on the step the chain was at, a restart starts at the top."""
        with pytest.raises(_Crash):
            await _scoped_step_flow(store, make_test_logger()).run(history="deploy")

        lg, warnings = _capturing_logger()
        await _plain_step_flow(store, lg).run(history="deploy", restart="latest")
        _, snapshot = await _head(store, "deploy")
        assert await History(store, "deploy").is_complete()
        assert snapshot.root == {"plain": True}
        assert warnings == []  # the crashed run's scope and cursors were never handed out

    async def test_starts_at_the_top_on_the_saved_state(self, store: MemoryStore) -> None:
        halted_run = _counting_chain(store, True, []).run(
            0, halt=asyncio.Event(), history="session"
        )
        assert await halted_run is HALTED
        _, halted = await _head(store, "session")
        assert halted.root == {"n": 2}

        seen: list[Any] = []
        restarted = _counting_chain(store, False, seen).run(0, history="session", restart="latest")
        assert await restarted == 40
        assert seen == [0, 10, 20, 30]  # resume would run the 2nd step on: [10, 20, 30]
        _, done = await _head(store, "session")
        assert done.root == {"n": 6}

    async def test_set_signals_stay_behind(self, store: MemoryStore) -> None:
        """The halted run had its signal set; the restarted run does not get it set again."""

        def build(cut: asyncio.Event, arm: bool, seen: list[bool]) -> Any:
            @verb
            async def step(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
                seen.append(cut.is_set())
                if arm:
                    cut.set()
                    ctx.halt.set()  # type: ignore[union-attr]
                    raise Interrupted()
                return x

            flow = Factory(make_test_logger(), checkpoint_store=store).create()
            return flow.with_checkpointer().call(step)

        cut = asyncio.Event()
        halted_run = build(cut, True, []).run(
            halt=asyncio.Event(), signals={"cut": cut}, history="sig"
        )
        assert await halted_run is HALTED
        _, halted = await _head(store, "sig")
        assert halted.cursors[""][SIGNALS] == ["cut"]

        seen: list[bool] = []
        cut = asyncio.Event()
        await build(cut, False, seen).run(history="sig", restart="latest", signals={"cut": cut})
        assert seen == [False]

    @pytest.mark.parametrize("by", ["name", "hash"])
    async def test_a_chosen_commit_seeds_it_and_parents_its_commits(
        self, store: MemoryStore, by: str
    ) -> None:
        marks: list[str | None] = []

        @verb
        async def mark(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
            ctx.state.data["at"] = "mark"
            marks.append(await ctx.checkpoint("mark"))
            return x

        @verb
        async def later(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
            ctx.state.data["at"] = "later"
            return x

        def build() -> Any:
            return Factory(make_test_logger(), checkpoint_store=store).create()

        await build().with_checkpointer().call(mark).then(later).run(history="pick")
        assert marks[0] is not None
        target = "mark" if by == "name" else marks[0]

        await build().with_checkpointer().call(_record_state).run(history="pick", restart=target)
        head, snapshot = await _head(store, "pick")
        assert snapshot.root["seen"] == [{"at": "mark"}]
        assert head.parents == (marks[0],)

    async def test_an_empty_history_runs_from_the_given_state(self, store: MemoryStore) -> None:
        flow = Factory(make_test_logger(), checkpoint_store=store).create()
        flow = flow.with_checkpointer().call(_record_state)
        await flow.run(state={"given": 1}, history="empty", restart="latest")
        _, snapshot = await _head(store, "empty")
        assert snapshot.root["seen"] == [{"given": 1}]

    async def test_takes_the_root_state_and_top_level_declared_resources_only(self) -> None:
        snapshot = Snapshot(
            has_state=True,
            root={"n": 1},
            scopes={"n/a": {"x": 1}},
            cursors={
                "": {
                    CHAIN: {"step": "a", "args": [], "kwargs": {}},
                    SIGNALS: ["cut"],
                    RESOURCES: {"tracker": {"spent": 2.0}},
                    RUN_RESOURCES: {"cost": {"spent": 1.0}},
                },
                "n/a": {TURN: {"task": "t"}, RESOURCES: {"inner": {}}},
            },
        )
        assert _restart_snapshot(snapshot) == Snapshot(
            has_state=True,
            root={"n": 1},
            cursors={"": {RESOURCES: {"tracker": {"spent": 2.0}}}},
        )

    async def test_without_declared_resources_no_cursor_is_left(self) -> None:
        snapshot = Snapshot(has_state=True, root={}, cursors={"": {SIGNALS: ["cut"]}})
        assert _restart_snapshot(snapshot) == Snapshot(has_state=True, root={})


class TestFailure:
    async def test_failure_writes_no_commit_and_reraises(self, store: MemoryStore) -> None:
        with pytest.raises(RuntimeError, match="boom at 2"):
            await _counting_flow(store, max_iters=5, fail_at=2).run(history="fail")
        assert await History(store, "fail").head() is None

    async def test_failure_leaves_the_last_save_as_head(self, store: MemoryStore) -> None:
        with pytest.raises(RuntimeError, match="boom at 2"):
            await _counting_flow(store, max_iters=5, fail_at=2, on_iterate=True).run(
                history="fail-saved"
            )

        head, snapshot = await _head(store, "fail-saved")
        assert head.meta.outcome == "ok"
        assert snapshot.root == {"n": 1}  # the save after pass 0, not the half-done pass 1
        assert not await History(store, "fail-saved").is_complete()

    async def test_cancellation_writes_no_commit(self, store: MemoryStore) -> None:
        @verb
        async def cancel(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            raise asyncio.CancelledError()

        flow = (
            Factory(make_test_logger(), checkpoint_store=store)
            .create()
            .with_checkpointer()
            .call(cancel)
        )
        with pytest.raises(asyncio.CancelledError):
            await flow.run(history="cancelled")
        assert await History(store, "cancelled").head() is None

    async def test_empty_flow_leaves_no_history(self, store: MemoryStore) -> None:
        flow = Factory(make_test_logger(), checkpoint_store=store).create()
        flow = flow.with_checkpointer()
        with pytest.raises(RuntimeError, match="no nodes to run"):
            await flow.run(history="empty")
        assert "empty" not in store.list_repos()


class TestResumeModeValidation:
    @pytest.mark.parametrize("mode", [True, "", "complete"])
    async def test_values_that_are_neither_mode_nor_checkpoint_name_are_rejected(
        self, store: MemoryStore, mode: Any
    ) -> None:
        with pytest.raises(ValueError, match="resume must be one of"):
            await _counting_flow(store, max_iters=1).run(history="bad-mode", resume=mode)

    @pytest.mark.parametrize("mode", ["replay", "restart"])
    async def test_removed_modes_name_missing_checkpoints(
        self, store: MemoryStore, mode: str
    ) -> None:
        with pytest.raises(ValueError, match=f"no checkpoint named {mode!r}"):
            await _counting_flow(store, max_iters=1).run(history="bad-mode", resume=mode)

    async def test_resume_requires_a_history(self, store: MemoryStore) -> None:
        @verb
        async def noop(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            return None

        flow = Factory(make_test_logger(), checkpoint_store=store).create().call(noop)
        with pytest.raises(RuntimeError, match="resume='latest' but no history"):
            await flow.run(resume="latest")
        with pytest.raises(RuntimeError, match="restart='latest' but no history"):
            await flow.run(restart="latest")
        assert store.list_repos() == []

    async def test_a_history_needs_a_checkpoint_store(self) -> None:
        @verb
        async def noop(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            return None

        flow = Factory(make_test_logger()).create("plain").call(noop)
        with pytest.raises(RuntimeError, match="history='h' but has no checkpoint store"):
            await flow.run(history="h")

    async def test_a_checkpointer_needs_a_history(self, store: MemoryStore) -> None:
        @verb
        async def noop(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            return None

        inner = Factory(make_test_logger()).create("inner").with_checkpointer().call(noop)
        flow = Factory(make_test_logger(), checkpoint_store=store).create().call(inner)
        with pytest.raises(RuntimeError, match="'inner' declares with_checkpointer.*no history"):
            await flow.run()

    async def test_a_bad_retention_is_rejected(self, store: MemoryStore) -> None:
        flow = _counting_flow(store, max_iters=1)
        with pytest.raises(ValueError, match="retention must be 'retain' or 'gc_on_success'"):
            await flow.run(history="bad-retention", retention="keep")  # type: ignore[arg-type]
        assert store.list_repos() == []

    async def test_resume_and_restart_are_not_taken_together(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError, match="resume or restart, not both"):
            await _counting_flow(store, max_iters=1).run(
                history="both", resume="latest", restart="latest"
            )

    @pytest.mark.parametrize("target", ["off", "", "complete"])
    async def test_restart_needs_latest_a_hash_or_a_checkpoint_name(
        self, store: MemoryStore, target: str
    ) -> None:
        with pytest.raises(ValueError, match="restart must be 'latest'"):
            await _counting_flow(store, max_iters=1).run(history="bad-target", restart=target)


class TestResumeInSubprocess:
    """Arguments ``resume_in_subprocess`` rejects before it starts a process."""

    async def test_off_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a resume mode"):
            resume_in_subprocess(
                store_module="appinfra.cas",
                store_factory="FileStore",
                store_kwargs={"root": "unused"},
                resume="off",
            )
