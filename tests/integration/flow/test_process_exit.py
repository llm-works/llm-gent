# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Real process deaths: a child process ends in ``os._exit`` mid-run; the parent resumes.

The halt/restart matrix simulates a crash by raising; here the process
really dies — no unwinding, no ``finally``, nothing flushed but what the
store already wrote — and a fresh process resumes from what is on disk.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, Factory, History, verb
from llm_gent.flow.checkpoint import HEAD_REF
from llm_gent.flow.stores import JsonFileCheckpointStore

from ...unit.flow.conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3]
EXIT_CODE = 17
NAME = "exit"


def _iterate_flow(store: Any, ran: list[int], die_at: int | None) -> Any:
    """``iterate(max_iters=4)`` over ``x + 1``; each pass checkpoints after its work."""

    @verb
    async def step(ctx: Context[dict[str, Any]], x: int) -> int:
        ran.append(x)
        ctx.state.data["last"] = x + 1
        await ctx.checkpoint()
        if x == die_at:
            os._exit(EXIT_CODE)
        return x + 1

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, NAME)
        .with_checkpointer()
        .iterate(lambda b: b.call(step), max_iters=4)
    )


def _map_flow(store: Any, ran: list[int], die_at: int | None) -> Any:
    """A parallel map over four items, saving after each; ``die_at``'s item dies mid-body."""

    @verb
    async def item(ctx: Context[dict[str, Any]], x: int) -> int:
        ran.append(x)
        await asyncio.sleep(0.01 * x)
        if x == die_at:
            os._exit(EXIT_CODE)
        return x * 10

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, NAME)
        .with_checkpointer()
        .with_checkpoint_policy(on_map_item=True)
        .map(lambda b: b.call(item), items=lambda _p, _c: [0, 1, 2, 3], aggregate=sum)
    )


class _DiesBeforeSecondHeadMove(JsonFileCheckpointStore):
    """Dies after writing the second commit object, before ``HEAD`` moves to it."""

    moves = 0

    def set_ref(self, flow_id: str, name: str, commit_hash: str, expected: str | None) -> bool:
        if name == HEAD_REF:
            self.moves += 1
            if self.moves == 2:
                os._exit(EXIT_CODE)
        return super().set_ref(flow_id, name, commit_hash, expected)


_SCENARIOS = {"iterate": (_iterate_flow, 2), "map": (_map_flow, 2), "torn": (_iterate_flow, None)}


def child(store_dir: str, scenario: str) -> None:
    """Child process entry: run ``scenario`` until it dies."""
    build, die_at = _SCENARIOS[scenario]
    store_cls = _DiesBeforeSecondHeadMove if scenario == "torn" else JsonFileCheckpointStore
    store = store_cls(make_test_logger(), Path(store_dir))
    asyncio.run(build(store, [], die_at).run(0))


def _run_child(tmp_path: Path, scenario: str) -> Path:
    """Run ``scenario`` in a fresh interpreter; return the store directory it left."""
    store_dir = tmp_path / "cp"
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from tests.integration.flow.test_process_exit import child; "
        f"child({str(store_dir)!r}, {scenario!r})"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == EXIT_CODE, proc.stderr
    return store_dir


async def test_death_after_a_checkpoint_resumes_at_its_step(tmp_path: Path) -> None:
    """The process dies right after pass 2's checkpoint: only that step runs again."""
    store_dir = _run_child(tmp_path, "iterate")
    store = JsonFileCheckpointStore(make_test_logger(), store_dir)
    ran: list[int] = []

    assert await _iterate_flow(store, ran, None).run(0, resume="latest") == 4
    assert ran == [2, 3]
    assert await History(store, NAME).is_complete()


async def test_death_inside_a_parallel_map_reruns_no_saved_item(tmp_path: Path) -> None:
    """Items saved before the death do not run again; the result is the uninterrupted one."""
    store_dir = _run_child(tmp_path, "map")
    store = JsonFileCheckpointStore(make_test_logger(), store_dir)
    history = History(store, NAME)
    head = await history.head()
    assert head is not None
    [cursor] = [c for c in (await history.snapshot(head)).cursors.values() if "done" in c]
    saved = {int(i) for i in cursor["done"]}
    assert saved, "no item was saved before the death"

    ran: list[int] = []
    assert await _map_flow(store, ran, None).run(0, resume="latest") == 60
    assert not saved & set(ran), f"saved items ran again: {saved & set(ran)}"
    assert set(ran) | saved == {0, 1, 2, 3}


async def test_death_between_commit_and_head_move_resumes_from_the_previous_head(
    tmp_path: Path,
) -> None:
    """A commit object written without HEAD moving to it is not part of the history."""
    store_dir = _run_child(tmp_path, "torn")
    store = JsonFileCheckpointStore(make_test_logger(), store_dir)
    history = History(store, NAME)
    commits = [c async for c in history.commits()]
    assert len(commits) == 1  # pass 0's checkpoint; pass 1's commit never became HEAD

    ran: list[int] = []
    assert await _iterate_flow(store, ran, None).run(0, resume="latest") == 4
    assert ran == [0, 1, 2, 3]  # pass 0 reruns: its checkpoint was taken inside it
    assert await history.is_complete()
