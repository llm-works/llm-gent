# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A flow's structure: built from the flow, stored with every commit, compared across versions."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import (
    HALTED,
    Context,
    Factory,
    Flow,
    History,
    Interrupted,
    Structure,
    collect_unreachable,
    verb,
)
from llm_gent.flow.state.snapshot import FLOW
from llm_gent.flow.stores import InMemoryCheckpointStore
from llm_gent.flow.structure import Cycle, StepKey, StepPath, path_label

from .conftest import make_test_logger


pytestmark = pytest.mark.unit


@verb
async def plan(ctx: Context[Any], x: Any = None) -> Any:
    return x


@verb
async def research(ctx: Context[Any], x: Any = None) -> Any:
    return x


@verb
async def explore(ctx: Context[Any], x: Any = None) -> Any:
    return x


@verb
async def synthesize(ctx: Context[Any], x: Any = None) -> Any:
    return x


def _flow() -> Flow:
    return Flow(make_test_logger())


def _key(target: Any, occurrence: int = 0) -> StepKey:
    """The key of an unnamed ``.call`` of ``target``."""
    return StepKey("call", f"verb:{target.__module__}.{target.__qualname__}", occurrence)


def _labels(paths: tuple[StepPath, ...]) -> list[str]:
    return [path_label(p) for p in paths]


class TestFlowStructure:
    def test_its_hash_is_the_flow_s_root_hash(self) -> None:
        flow = _flow().call(plan).map(lambda b: b.call(explore), name="wave")
        assert Structure.of(flow).hash == flow.root_hash()

    def test_round_trips_through_json(self) -> None:
        flow = (
            _flow()
            .call(plan, name="plan")
            .then(research)
            .then(research)
            .map(lambda b: b.call(explore), name="wave")
        )
        structure = Structure.of(flow)
        assert Structure.from_json(structure.to_json()) == structure

    def test_steps_are_addressed_by_key_and_boundary(self) -> None:
        flow = (
            _flow()
            .call(plan, name="plan")
            .then(research)
            .then(research)
            .map(lambda b: b.call(explore), name="wave")
        )
        assert _labels(tuple(Structure.of(flow).steps_by_path())) == [
            "plan",
            f"verb:{__name__}.research",
            f"verb:{__name__}.research[1]",
            "map:wave",
            f"map:wave / map: verb:{__name__}.explore",
        ]

    def test_a_recursive_flow_is_recorded_as_a_cycle(self) -> None:
        recursive = _flow().call(plan)
        recursive.branch(when=lambda *_: False, then=recursive)
        structure = Structure.of(recursive)
        assert structure.steps[1].children == {"then": Cycle(0)}
        assert Structure.from_json(structure.to_json()) == structure


class TestStructureDiff:
    def test_identical_flows_do_not_differ(self) -> None:
        diff = Structure.of(_flow().call(plan).then(research)).diff(
            Structure.of(_flow().call(plan).then(research))
        )
        assert not diff.changed
        assert (diff.added, diff.removed) == ((), ())
        assert len(diff.kept) == 2

    def test_a_step_added_after_another(self) -> None:
        old = Structure.of(_flow().call(plan).then(research))
        new = Structure.of(_flow().call(plan).then(research).then(synthesize))
        diff = old.diff(new)
        assert diff.changed
        assert diff.added == ((("", _key(synthesize)),),)
        assert diff.added_before((("", _key(research)),)) == ()

    def test_a_step_added_before_another(self) -> None:
        old = Structure.of(_flow().call(plan).then(synthesize))
        new = Structure.of(_flow().call(plan).then(research).then(synthesize))
        diff = old.diff(new)
        assert diff.added_before((("", _key(synthesize)),)) == ((("", _key(research)),),)
        assert diff.added_before((("", _key(plan)),)) == ()

    def test_a_removed_step_takes_its_children_with_it(self) -> None:
        old = Structure.of(_flow().call(plan).map(lambda b: b.call(explore), name="wave"))
        new = Structure.of(_flow().call(plan))
        diff = old.diff(new)
        assert _labels(diff.removed) == ["map:wave", f"map:wave / map: verb:{__name__}.explore"]
        assert diff.added_before((("", _key(plan)),)) == ()

    def test_a_named_step_s_verb_can_change(self) -> None:
        old = Structure.of(_flow().call(research, name="research"))
        new = Structure.of(_flow().call(explore, name="research"))
        assert not old.diff(new).changed

    def test_a_change_inside_a_map_body_is_addressed_through_the_map(self) -> None:
        old = Structure.of(_flow().map(lambda b: b.call(explore), name="wave"))
        new = Structure.of(_flow().map(lambda b: b.call(explore).then(research), name="wave"))
        diff = old.diff(new)
        assert _labels(diff.added) == [f"map:wave / map: verb:{__name__}.research"]
        assert _labels(diff.kept) == ["map:wave", f"map:wave / map: verb:{__name__}.explore"]

    def test_a_reorder_changes_the_structure_but_keeps_every_step(self) -> None:
        diff = Structure.of(_flow().call(plan).then(research)).diff(
            Structure.of(_flow().call(research).then(plan))
        )
        assert diff.changed
        assert (diff.added, diff.removed) == ((), ())

    def test_added_before_a_step_the_new_structure_lacks_is_empty(self) -> None:
        diff = Structure.of(_flow().call(plan)).diff(Structure.of(_flow().call(research)))
        assert diff.added_before((("", _key(plan)),)) == ()


def _checkpointed(store: Any, halt: asyncio.Event, *, stop: bool) -> Any:
    """``plan`` (takes the checkpoint ``"saved"``) → ``research`` (halts when ``stop``)."""

    @verb
    async def first(ctx: Context[dict[str, Any]], x: int) -> int:
        await ctx.checkpoint("saved")
        return x

    @verb
    async def second(ctx: Context[dict[str, Any]], x: int) -> int:
        if stop:
            halt.set()
            raise Interrupted()
        return x

    return (
        Factory(make_test_logger())
        .create(state={})
        .with_checkpoint_store(store, "structure")
        .with_checkpointer()
        .with_halt(halt)
        .call(first, name="plan")
        .then(second, name="research")
    )


@pytest.mark.asyncio
class TestStoredStructure:
    async def test_every_commit_holds_the_structure_it_records(self) -> None:
        store = InMemoryCheckpointStore()
        flow = _checkpointed(store, asyncio.Event(), stop=True)
        assert await flow.run(1) is HALTED
        await _checkpointed(store, asyncio.Event(), stop=False).run(resume="latest")

        history = History(store, "structure")
        commits = [c async for c in history.commits()]
        assert [c.meta.outcome for c in commits] == ["ok", "halted", "ok"]  # completion first
        for commit in commits:
            structure = await history.structure(commit)
            assert structure == Structure.of(flow)
            assert structure is not None and structure.hash == commit.meta.flow_root_hash

    async def test_the_snapshot_does_not_read_it(self) -> None:
        store = InMemoryCheckpointStore()
        await _checkpointed(store, asyncio.Event(), stop=True).run(1)
        history = History(store, "structure")
        head = await history.head()
        assert head is not None
        snapshot = await history.snapshot(head)
        assert FLOW not in snapshot.cursors.get("", {})
        assert FLOW not in snapshot.scopes

    async def test_collecting_unreachable_objects_keeps_it(self) -> None:
        """A resume to an earlier checkpoint leaves objects behind; the structure is reachable."""
        store = InMemoryCheckpointStore()
        await _checkpointed(store, asyncio.Event(), stop=True).run(1)
        await _checkpointed(store, asyncio.Event(), stop=False).run(resume="saved")
        assert await collect_unreachable(store, "structure") > 0

        history = History(store, "structure")
        async for commit in history.commits():
            assert await history.structure(commit) is not None

    async def test_a_commit_without_state_has_no_structure(self) -> None:
        """A final state that cannot be serialized commits an empty tree."""
        store = InMemoryCheckpointStore()
        flow = (
            Factory(make_test_logger())
            .create(state={"handle": object()})
            .with_checkpoint_store(store, "stateless")
            .call(plan)
        )
        await flow.run(1)
        history = History(store, "stateless")
        head = await history.head()
        assert head is not None
        assert await history.structure(head) is None
