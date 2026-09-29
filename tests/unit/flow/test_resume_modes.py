# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""``Flow.run(resume=...)`` modes: restart, the failure commit, and replay past it.

``"restart"`` runs from the first node with the root state of the newest
commit that has usable state, skipping ``$failed`` and stateless commits;
``"replay"`` fast-forwards to the last save point and skips ``$failed``
commits. A run that raises commits its root state at ``$failed`` before
the exception propagates.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, FlowFactory, History, verb
from llm_gent.flow.checkpoint import FAILED_NODE_PATH, FAILURE_PRODUCER, ResumeMode
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


class TestRestart:
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
        ).run(resume="restart")

        # Restored 2 iterations + 2 more; the fallback state (n=100) is unused.
        assert result["iterations_completed"] == 4
        assert result["log"][0] == 1

    async def test_after_halt_starts_at_first_node(self, store: JsonFileCheckpointStore) -> None:
        """Restart keeps the halted state but not the position: counters start at zero."""
        halt = asyncio.Event()
        await build_canonical_flow(
            make_test_logger(),
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            client_flow_id="halted",
        ).run()

        result = await build_canonical_flow(
            make_test_logger(), max_iters=3, store=store, client_flow_id="halted"
        ).run(resume="restart")

        # Replay would stop at the cumulative bound (5); restart runs max_iters=3 afresh.
        assert result["iterations_completed"] == 5

    async def test_after_failure_continues_from_last_save_point(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """The $failed state may be half-updated; restart starts from the commit before it."""
        with pytest.raises(RuntimeError, match="boom at 3"):
            await _counting_flow(store, "failed", max_iters=5, fail_at=3, on_iterate=True).run()

        lg, warnings = _capturing_logger()
        result = await _counting_flow(store, "failed", max_iters=2, lg=lg).run(
            state={"n": 100}, resume="restart"
        )
        assert result == 4  # last save point n=2, then 2 more
        skipped = [extra for msg, extra in warnings if msg.startswith("restart skipped")]
        assert len(skipped) == 1 and skipped[0]["skipped"] == ["$failed"]

    async def test_failure_before_any_save_point_uses_given_state(
        self, store: JsonFileCheckpointStore
    ) -> None:
        with pytest.raises(RuntimeError):
            await _counting_flow(store, "failed-early", max_iters=5, fail_at=2).run()

        lg, warnings = _capturing_logger()
        result = await _counting_flow(store, "failed-early", max_iters=1, lg=lg).run(
            state={"n": 10}, resume="restart"
        )
        assert result == 11
        assert any("no commit with usable state" in msg for msg, _ in warnings)

    async def test_skips_stateless_final_commit(self, store: JsonFileCheckpointStore) -> None:
        """A final commit without state (unserializable) is walked past, not a reason to reset."""

        @verb
        async def bump(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            ctx.state.data.pop("handle", None)
            return ctx.state.data["n"]

        @verb
        async def attach(ctx: Context[dict[str, Any]], _prev: Any = None) -> int:
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

        lg, warnings = _capturing_logger()
        result = await _flow(lg).run(state={"n": 100}, resume="restart")
        assert result == 4  # restored n=2 from the last iterate commit, then 2 more
        skipped = [extra for msg, extra in warnings if msg.startswith("restart skipped")]
        assert len(skipped) == 1
        assert skipped[0]["skipped"] == ["$end"]
        assert skipped[0]["restored_node_path"] != "$end"

    async def test_newest_commit_restores_without_warning(
        self, store: JsonFileCheckpointStore
    ) -> None:
        await _counting_flow(store, "quiet", max_iters=2).run()
        lg, warnings = _capturing_logger()
        await _counting_flow(store, "quiet", max_iters=1, lg=lg).run(resume="restart")
        assert warnings == []

    async def test_corrupt_history_raises(self, store: JsonFileCheckpointStore) -> None:
        from llm_gent.flow import HistoryCorrupt

        await _counting_flow(store, "corrupt", max_iters=2).run()
        head, _ = await _head(store, "corrupt")
        commits_dir = store._history_dir(head.meta.flow_id) / "objects" / "commit"
        (commits_dir / head.content_hash).unlink()

        with pytest.raises(HistoryCorrupt):
            await _counting_flow(store, "corrupt", max_iters=1).run(resume="restart")

    async def test_empty_history_uses_given_state(self, store: JsonFileCheckpointStore) -> None:
        result = await _counting_flow(store, "empty", max_iters=1).run(
            state={"n": 10}, resume="restart"
        )
        assert result == 11


class TestFailureCommit:
    async def test_failure_commits_root_state_and_reraises(
        self, store: JsonFileCheckpointStore
    ) -> None:
        with pytest.raises(RuntimeError, match="boom at 2"):
            await _counting_flow(store, "fail-commit", max_iters=5, fail_at=2).run()

        head, snapshot = await _head(store, "fail-commit")
        assert History.is_failed(head)
        assert head.meta.node_path == FAILED_NODE_PATH
        assert head.meta.outcome == "failed"
        assert head.meta.produced_by.node_id == FAILURE_PRODUCER
        assert (snapshot.root, snapshot.scopes) == ({"n": 2}, {})
        assert not await History(store, "fail-commit").is_complete()

    async def test_store_error_does_not_mask_original_exception(self, tmp_path: Path) -> None:
        """The run's own exception propagates even if the failure commit cannot be written."""

        class _NoCommits(JsonFileCheckpointStore):
            def put_object(
                self, flow_id: str, kind: Any, content_hash: str, payload: bytes
            ) -> None:
                if kind == "commit":
                    raise OSError("store unavailable")
                super().put_object(flow_id, kind, content_hash, payload)

        lg = make_test_logger()
        warnings: list[str] = []
        lg.warning = lambda msg, *_a, **_kw: warnings.append(msg)  # type: ignore[method-assign]
        store = _NoCommits(make_test_logger(), tmp_path / "cp")

        with pytest.raises(RuntimeError, match="boom at 1"):
            await _counting_flow(store, "masked", max_iters=3, fail_at=1, lg=lg).run()
        assert "failure commit could not be written" in warnings

    async def test_cancellation_writes_no_failure_commit(
        self, store: JsonFileCheckpointStore
    ) -> None:
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


class TestReplayPastFailure:
    async def test_replay_resumes_from_save_point_before_failure(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """Replay ignores the $failed head and fast-forwards to the last save point."""
        with pytest.raises(RuntimeError):
            await _counting_flow(
                store, "replay-fail", max_iters=5, fail_at=3, on_iterate=True
            ).run()

        result = await _counting_flow(store, "replay-fail", max_iters=5, on_iterate=True).run(
            resume="replay"
        )
        # Save point after n=2 → three remaining iterations up to the cumulative bound.
        # Restarting from the $failed state (n=3) would have ended at n=8.
        assert result == 5

    async def test_replay_point_skips_failed_head(self, store: JsonFileCheckpointStore) -> None:
        with pytest.raises(RuntimeError):
            await _counting_flow(store, "point", max_iters=5, fail_at=3, on_iterate=True).run()

        history = History(store, "point")
        chain = [c async for c in history.commits()]
        assert History.is_failed(chain[0])
        assert await history.replay_point() == chain[1]

    async def test_failure_after_completion_replays_fresh(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """``$end`` then ``$failed``: no replay point, so replay starts from ``state=``."""
        await _counting_flow(store, "end-then-fail", max_iters=2).run()
        with pytest.raises(RuntimeError):
            await _counting_flow(store, "end-then-fail", max_iters=5, fail_at=1).run()

        history = History(store, "end-then-fail")
        head = await history.head()
        assert head is not None and History.is_failed(head)
        assert not await history.is_complete()
        assert await history.replay_point() is None

        result = await _counting_flow(store, "end-then-fail", max_iters=1).run(
            state={"n": 10}, resume="replay"
        )
        assert result == 11


class _Crash(BaseException):
    """The process died: not an ``Exception``, so no failure commit follows the checkpoint."""


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


def _scoped_step_flow(
    store: JsonFileCheckpointStore, name: str, lg: Any, before: Any = None
) -> Any:
    """``[before] → .call(sub, state=)`` whose leaf checkpoints its scope and crashes."""
    flow = FlowFactory(lg).create(state={}).with_checkpointer(store, name)
    if before is not None:
        flow = flow.call(before)
    sub = FlowFactory(lg).create().call(_work_then_crash)
    return flow.call(sub, state=lambda _p: {}, merge=lambda p, c: p.update(c))


def _plain_step_flow(store: JsonFileCheckpointStore, name: str, lg: Any) -> Any:
    """The same history after a deploy that replaced the scoped step with a plain one."""
    return FlowFactory(lg).create(state={}).with_checkpointer(store, name).call(_plain)


class TestScopesFromSnapshot:
    @pytest.mark.parametrize("mode", ["replay", "restart"])
    async def test_scope_under_map_item_gets_its_own_payload(
        self, store: JsonFileCheckpointStore, mode: ResumeMode
    ) -> None:
        """Each saved scope comes back to the block that owns it, not to the one above it."""
        with pytest.raises(_Crash):
            await _scoped_map_flow(store, "nested", [], crash=True).run()

        seen: list[dict[str, Any]] = []
        await _scoped_map_flow(store, "nested", seen, crash=False).run(resume=mode)
        assert seen == [{"tag": "call", "hit": True}]

    async def test_finished_restart_drops_scopes_it_never_reached(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A deploy removed the scoped step: the final commit holds the root alone."""
        with pytest.raises(_Crash):
            await _scoped_step_flow(store, "deploy", make_test_logger()).run()
        history = History(store, "deploy")
        head = await history.head()
        assert head is not None and (await history.snapshot(head)).scopes

        lg, warnings = _capturing_logger()
        await _plain_step_flow(store, "deploy", lg).run(resume="restart")

        head = await history.head()
        assert head is not None and await history.is_complete()
        assert (await history.snapshot(head)).scopes == {}
        assert [msg for msg, _ in warnings] == [
            "restart finished without reaching saved scopes; dropped them"
        ]

    async def test_halted_restart_keeps_scopes_it_has_not_reached(
        self, store: JsonFileCheckpointStore
    ) -> None:
        """A run that halts before a saved scope's step keeps it for the next restart."""
        with pytest.raises(_Crash):
            await _scoped_step_flow(store, "halt-early", make_test_logger()).run()
        halt = asyncio.Event()

        @verb
        async def stop(ctx: Context[dict[str, Any]], x: Any = None) -> Any:
            halt.set()
            return x

        flow = _scoped_step_flow(store, "halt-early", make_test_logger(), before=stop)
        await flow.with_halt(halt).run(resume="restart")

        history = History(store, "halt-early")
        head = await history.head()
        assert head is not None and head.meta.outcome == "halted"
        assert list((await history.snapshot(head)).scopes.values()) == [{"w": 1}]


class TestResumeModeValidation:
    async def test_boolean_resume_is_rejected(self, store: JsonFileCheckpointStore) -> None:
        with pytest.raises(ValueError, match="resume must be one of"):
            await _counting_flow(store, "bool", max_iters=1).run(resume=True)  # type: ignore[arg-type]

    async def test_resume_mode_requires_checkpointer(self) -> None:
        @verb
        async def noop(ctx: Context[dict[str, Any]], _prev: Any = None) -> None:
            return None

        flow = FlowFactory(make_test_logger()).create(state={}).call(noop)
        with pytest.raises(RuntimeError, match="resume='restart'"):
            await flow.run(resume="restart")


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

    @pytest.mark.parametrize("mode", ["replay", "restart"])
    async def test_matches_in_process_resume(self, tmp_path: Path, mode: ResumeMode) -> None:
        in_proc = JsonFileCheckpointStore(make_test_logger(), tmp_path / "in-proc")
        await self._halt_at_two(in_proc)
        await self._halt_at_two(JsonFileCheckpointStore(make_test_logger(), tmp_path / "xproc"))

        expected = await build_canonical_flow(
            make_test_logger(), max_iters=4, store=in_proc, client_flow_id="xproc"
        ).run(resume=mode)
        result = resume_in_subprocess(
            store_module="llm_gent.flow.stores",
            store_factory="JsonFileCheckpointStore",
            store_kwargs={"root": str(tmp_path / "xproc")},
            flow_builder_kwargs={"max_iters": 4},
            client_flow_id="xproc",
            resume=mode,
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
