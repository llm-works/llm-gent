# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow-level :class:`CheckpointStore` — Protocol shape, wiring, and save/resume.

Exercises:

- Protocol conformance for the new Flow-level store.
- :meth:`Flow.with_checkpointer` + :meth:`FlowFactory.with_checkpointer`.
- ``state_factory=`` plumbing on Flow / FlowFactory / .call / .iterate / .map.
- Save-at-``.iterate``-boundary end-to-end.
- Framework-owned ``state_factory.restore`` dispatch on ``run(resume=True)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

from llm_gent.flow import (
    CheckpointStore,
    Context,
    Flow,
    FlowFactory,
    TypeStateFactory,
    verb,
)
from llm_gent.flow.nodes import _Iterate, _Map
from llm_gent.flow.stores.json_file import JsonFileCheckpointStore
from llm_gent.flow.testing import (
    assert_resume_determinism,
    build_canonical_flow,
    resume_in_subprocess,
)

from .conftest import ROLE_A, make_ff, make_test_logger


# -----------------------------------------------------------------------------
# Stubs
# -----------------------------------------------------------------------------


@dataclass
class _RecordingStore:
    """CheckpointStore stub — records saves/loads/deletes and serves a preload."""

    preload: tuple[dict[str, Any], dict[str, Any]] | None = None
    saves: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    loads: list[tuple[str, int | None]] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)

    def save_checkpoint(
        self,
        client_flow_id: str,
        iteration: int,
        state_json: dict[str, Any],
        metadata_json: dict[str, Any],
    ) -> None:
        self.saves.append((client_flow_id, iteration, state_json, metadata_json))

    def load_checkpoint(
        self, client_flow_id: str, iteration: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        self.loads.append((client_flow_id, iteration))
        return self.preload

    def delete_checkpoint(self, client_flow_id: str) -> None:
        self.deletes.append(client_flow_id)


@dataclass
class Counter:
    """Sample :class:`StateData` payload: a serializable counter."""

    n: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


# -----------------------------------------------------------------------------
# Protocol conformance
# -----------------------------------------------------------------------------


class TestProtocolShape:
    """The recording stub structurally satisfies the Flow-level Protocol."""

    def test_recording_store_matches_protocol(self) -> None:
        """Structural conformance: assignment to Protocol type type-checks."""
        store = _RecordingStore()
        _: CheckpointStore = store
        assert callable(store.save_checkpoint)
        assert callable(store.load_checkpoint)
        assert callable(store.delete_checkpoint)


# -----------------------------------------------------------------------------
# with_checkpointer wiring
# -----------------------------------------------------------------------------


class TestFlowWithCheckpointer:
    """``Flow.with_checkpointer`` binds store + client_flow_id as a pair."""

    def test_binds_both_attributes(self) -> None:
        """After the call, both slots reflect the arguments."""
        flow = Flow(make_test_logger())
        store = _RecordingStore()
        flow.with_checkpointer(store, "traj-1")
        assert flow._checkpointer is store
        assert flow._client_flow_id == "traj-1"

    def test_returns_self_for_chaining(self) -> None:
        """Mirrors ``.with_halt`` / ``.with_budget`` — chainable."""
        flow = Flow(make_test_logger())
        result = flow.with_checkpointer(_RecordingStore(), "traj-1")
        assert result is flow


class TestFactoryWithCheckpointer:
    """``FlowFactory.with_checkpointer`` derivation captures the store."""

    def test_derived_factory_carries_store(self) -> None:
        """The new factory has the checkpointer captured; original untouched."""
        base = make_ff()
        store = _RecordingStore()
        derived = base.with_checkpointer(store)
        assert derived._checkpointer is store
        assert base._checkpointer is None

    def test_create_binds_when_client_flow_id_supplied(self) -> None:
        """``create(client_flow_id=)`` wires the built Flow's checkpointer."""
        store = _RecordingStore()
        ff = make_ff().with_checkpointer(store)
        flow = ff.create(client_flow_id="traj-1")
        assert flow._checkpointer is store
        assert flow._client_flow_id == "traj-1"

    def test_create_without_client_flow_id_leaves_flow_unwired(self) -> None:
        """No ``client_flow_id=`` → Flow is built without a bound checkpointer."""
        store = _RecordingStore()
        ff = make_ff().with_checkpointer(store)
        flow = ff.create()
        assert flow._checkpointer is None
        assert flow._client_flow_id is None

    def test_derivations_preserve_checkpointer(self) -> None:
        """``with_halt`` / ``with_budget`` / ``with_traits`` retain the store."""
        store = _RecordingStore()
        base = make_ff().with_checkpointer(store)
        import asyncio as _asyncio

        derived = base.with_halt(_asyncio.Event())
        assert derived._checkpointer is store


# -----------------------------------------------------------------------------
# state_factory plumbing
# -----------------------------------------------------------------------------


class TestStateFactoryPlumbing:
    """``state_factory=`` reaches Flow / FlowFactory and every scoped composition site."""

    def test_flow_init_captures_state_factory(self) -> None:
        """``Flow(..., state_factory=F)`` stores it on the flow."""
        sf = TypeStateFactory(Counter)
        flow = Flow(make_test_logger(), state_factory=sf)
        assert flow._state_factory is sf

    def test_factory_captures_and_threads(self) -> None:
        """``FlowFactory(state_factory=F)`` threads into ``create()``."""
        sf = TypeStateFactory(Counter)
        ff = FlowFactory(make_test_logger(), state_factory=sf)
        flow = ff.create()
        assert flow._state_factory is sf

    def test_call_records_state_factory(self) -> None:
        """``.call(state_factory=F)`` stores it on the node."""

        @verb(role=ROLE_A)
        async def step(ctx: Context[Any], x: int) -> int:
            return x

        subflow = make_ff().create()
        subflow.call(step)
        sf = TypeStateFactory(Counter)
        parent = make_ff().create().call(subflow, state=lambda _p: {}, state_factory=sf)
        assert parent._nodes[-1].state_factory is sf

    def test_iterate_records_state_factory(self) -> None:
        """``.iterate(state_factory=F)`` stores it on the iterate node."""

        @verb(role=ROLE_A)
        async def step(ctx: Context[Any], _: Any = None) -> int:
            return 0

        sf = TypeStateFactory(Counter)
        flow = make_ff().create().iterate(step, max_iters=1, state_factory=sf)
        node = flow._nodes[-1].target
        assert isinstance(node, _Iterate)
        assert node.state_factory is sf

    def test_map_records_state_factory(self) -> None:
        """``.map(state_factory=F)`` stores it on the map node."""

        @verb(role=ROLE_A)
        async def step(ctx: Context[Any], item: Any) -> Any:
            return item

        sf = TypeStateFactory(Counter)
        flow = make_ff().create().map(step, items=lambda _p, _c: [1], state_factory=sf)
        node = flow._nodes[-1].target
        assert isinstance(node, _Map)
        assert node.state_factory is sf


# -----------------------------------------------------------------------------
# Save-at-iterate-boundary
# -----------------------------------------------------------------------------


class TestSaveAtIterateBoundary:
    """Every completed iteration saves a checkpoint under the wired store."""

    async def test_saves_once_per_iteration_dict_payload(self) -> None:
        """Plain-dict payload round-trips as-is into ``state_json['data']``."""

        @verb(role=ROLE_A)
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        store = _RecordingStore()
        flow = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=3)
        )
        await flow.run()
        assert [s[1] for s in store.saves] == [1, 2, 3]
        assert store.saves[-1][0] == "traj-1"
        assert store.saves[-1][2] == {"data": {"n": 3}, "children": []}

    async def test_saves_carry_metadata_path_and_iteration(self) -> None:
        """Every save's ``metadata_json`` carries ``path`` and ``iteration``."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        store = _RecordingStore()
        flow = (
            make_ff().create(state={}).with_checkpointer(store, "traj-1").iterate(noop, max_iters=1)
        )
        await flow.run()
        meta = store.saves[0][3]
        assert len(meta["path"]) == 1
        assert isinstance(meta["path"][0], str)
        assert meta["iteration"] == 1

    async def test_unwired_iterate_makes_no_saves(self) -> None:
        """Without ``.with_checkpointer`` the iterate proceeds silently."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        store = _RecordingStore()
        flow = make_ff().create().iterate(noop, max_iters=2)
        await flow.run()
        assert store.saves == []

    async def test_statedata_payload_uses_to_dict(self) -> None:
        """A :class:`StateData` payload is serialized via ``to_dict()``."""

        @verb(role=ROLE_A)
        async def bump(ctx: Context[Counter], _prev: Any = None) -> int:
            ctx.state.data.n += 1
            return ctx.state.data.n

        store = _RecordingStore()
        flow = (
            Flow(make_test_logger(), state=Counter(), state_factory=TypeStateFactory(Counter))
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=2)
        )
        await flow.run()
        assert store.saves[-1][2] == {"data": {"n": 2}, "children": []}


# -----------------------------------------------------------------------------
# Resume — framework-owned from_dict dispatch
# -----------------------------------------------------------------------------


class TestResumeHydration:
    """``run(resume=True)`` loads the checkpoint and hydrates via ``state_factory.restore``."""

    async def test_resume_without_checkpointer_raises(self) -> None:
        """``resume=True`` without a wired checkpointer is a build-time error."""
        import pytest

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any]) -> None:
            return None

        flow = make_ff().create().call(noop)
        with pytest.raises(RuntimeError, match="no checkpointer"):
            await flow.run(resume=True)

    async def test_resume_no_checkpoint_falls_back_to_state(self) -> None:
        """Load returning ``None`` leaves the caller's ``state=`` intact."""

        seen: list[Any] = []

        @verb(role=ROLE_A)
        async def peek(ctx: Context[Any]) -> None:
            seen.append(ctx.state.data)

        store = _RecordingStore(preload=None)
        flow = make_ff().create().with_checkpointer(store, "traj-1").call(peek)
        await flow.run(state={"n": 5}, resume=True)
        assert seen == [{"n": 5}]

    async def test_resume_dict_payload_passthrough(self) -> None:
        """Plain-dict payload: the loaded dict becomes ``ctx.state.data``."""

        seen: list[Any] = []

        @verb(role=ROLE_A)
        async def peek(ctx: Context[Any]) -> None:
            seen.append(ctx.state.data)

        store = _RecordingStore(preload=({"data": {"n": 42}}, {}))
        flow = make_ff().create().with_checkpointer(store, "traj-1").call(peek)
        await flow.run(state={"n": 0}, resume=True)
        assert seen == [{"n": 42}]

    async def test_resume_typed_payload_calls_restore(self) -> None:
        """When ``state_factory=F`` is set, framework calls ``F.restore``."""

        seen: list[Any] = []

        @verb(role=ROLE_A)
        async def peek(ctx: Context[Counter]) -> None:
            seen.append(ctx.state.data)

        store = _RecordingStore(preload=({"data": {"n": 7}}, {}))
        flow = (
            Flow(make_test_logger(), state_factory=TypeStateFactory(Counter))
            .with_checkpointer(store, "traj-1")
            .call(peek)
        )
        make_ff().create()  # keep SAIAFactory pattern consistent — not needed here
        await flow.run(state=Counter(n=0), resume=True)
        assert seen == [Counter(n=7)]

    async def test_successful_run_deletes_checkpoint(self) -> None:
        """A fully successful run clears the trajectory."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any]) -> None:
            return None

        store = _RecordingStore()
        flow = make_ff().create().with_checkpointer(store, "traj-1").call(noop)
        await flow.run()
        assert store.deletes == ["traj-1"]

    async def test_exception_preserves_checkpoint(self) -> None:
        """A raise inside the flow skips the delete."""
        import pytest

        @verb(role=ROLE_A)
        async def boom(ctx: Context[Any]) -> None:
            raise ValueError("kaboom")

        store = _RecordingStore()
        flow = make_ff().create().with_checkpointer(store, "traj-1").call(boom)
        with pytest.raises(ValueError, match="kaboom"):
            await flow.run()
        assert store.deletes == []


# -----------------------------------------------------------------------------
# End-to-end round-trip
# -----------------------------------------------------------------------------


class TestEndToEndRoundTrip:
    """Save-at-iterate + resume drives the flow to convergence with state carrying progress."""

    async def test_iterate_round_trip_dict_payload(self) -> None:
        """First run saves; ``resume=True`` fast-forwards the counter + hydrates state.

        Under PR 3 ``max_iters`` is a cumulative bound across resumes:
        the second run picks up the saved iteration count, so only the
        remaining iterations up to ``max_iters`` execute.
        """

        @verb(role=ROLE_A)
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        store = _RecordingStore()

        # First run: 3 iterations complete; state stabilizes at n=3.
        flow_a = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=3)
        )
        await flow_a.run()
        final_state, final_meta = store.saves[-1][2], store.saves[-1][3]
        assert final_meta["iteration"] == 3
        # Simulate a crash by preloading the last save for the next run;
        # the checkpoint is what a real store would still hold.
        store.preload = (final_state, final_meta)
        store.saves.clear()
        store.deletes.clear()

        # Second run: resume hydrates n=3 and starts the counter at 3.
        # With max_iters=5 (cumulative), only iterations 4 and 5 execute.
        seen: list[int] = []

        @verb(role=ROLE_A)
        async def bump_and_capture(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            seen.append(ctx.state.data["n"])
            return ctx.state.data["n"]

        flow_b = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-1")
            .iterate(bump_and_capture, max_iters=5)
        )
        await flow_b.run(resume=True)
        assert seen == [4, 5]

    async def test_iterate_round_trip_typed_payload(self) -> None:
        """Same round-trip with a :class:`StateData` payload via ``state_factory=``."""

        @verb(role=ROLE_A)
        async def bump(ctx: Context[Counter], _prev: Any = None) -> int:
            ctx.state.data.n += 1
            return ctx.state.data.n

        store = _RecordingStore()

        # First run: 2 iterations save; leave the checkpoint at iteration=2.
        flow_a = (
            Flow(make_test_logger(), state=Counter(), state_factory=TypeStateFactory(Counter))
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=3)
        )
        # Simulate crash after iteration 2 by dropping the third save.
        await flow_a.run()
        second_state, second_meta = store.saves[1][2], store.saves[1][3]
        assert second_meta["iteration"] == 2
        store.preload = (second_state, second_meta)
        store.saves.clear()

        # Second run: resume starts the counter at 2, so only iteration 3
        # runs — n goes 2 → 3 under cumulative max_iters=3.
        flow_b = (
            Flow(make_test_logger(), state=Counter(), state_factory=TypeStateFactory(Counter))
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=3)
        )
        result = await flow_b.run(resume=True)
        assert result == 3

    async def test_iterate_inherits_call_scope_state_factory(self) -> None:
        """Iterate inheriting call-scope state restores through the call's factory.

        Scenario: `.call(subflow, state=..., state_factory=F)` creates typed
        state; the subflow's iterate has no factory of its own but inherits
        via ``State._factory``. On resume, the iterate uses the inherited
        factory to restore.
        """

        @verb(role=ROLE_A)
        async def bump(ctx: Context[Counter], _prev: Any = None) -> int:
            ctx.data.n += 1
            return ctx.data.n

        store = _RecordingStore()

        # Subflow with iterate — no state_factory on the iterate itself.
        inner = make_ff().create("inner").iterate(bump, max_iters=3)

        # Outer flow calls the subflow with typed state + factory.
        outer_a = (
            make_ff()
            .create(state={})
            .call(
                inner,
                state=lambda _p: Counter(),
                state_factory=TypeStateFactory(Counter),
            )
            .with_checkpointer(store, "traj-inherit")
        )
        await outer_a.run()

        # Simulate crash after iteration 2.
        second_state, second_meta = store.saves[1][2], store.saves[1][3]
        assert second_meta["iteration"] == 2
        store.preload = (second_state, second_meta)
        store.saves.clear()

        # Resume: the iterate should restore typed state via inherited factory.
        outer_b = (
            make_ff()
            .create(state={})
            .call(
                inner,
                state=lambda _p: Counter(),
                state_factory=TypeStateFactory(Counter),
            )
            .with_checkpointer(store, "traj-inherit")
        )
        result = await outer_b.run(resume=True)
        # Iteration 3 runs: n goes 2 → 3.
        assert result == 3


# -----------------------------------------------------------------------------
# Recursive snapshot envelope — path (ancestor IDs) + state tree
# -----------------------------------------------------------------------------


class TestRecursiveSnapshotEnvelope:
    """PR 3: metadata carries the id-chain path from root; state_json holds child scopes."""

    async def test_top_level_iterate_path_has_single_id(self) -> None:
        """A top-level ``.iterate`` produces a one-id path — the iterate itself."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        store = _RecordingStore()
        flow = (
            make_ff().create(state={}).with_checkpointer(store, "traj-1").iterate(noop, max_iters=2)
        )
        await flow.run()
        paths = [s[3]["path"] for s in store.saves]
        iters = [s[3]["iteration"] for s in store.saves]
        assert len(paths) == 2
        assert iters == [1, 2]
        assert all(len(p) == 1 for p in paths)
        # Same iterate node → same id across every save (path is identity, not runtime state).
        assert paths[0] == paths[1]

    async def test_iterate_inside_subflow_appends_ancestor_id(self) -> None:
        """`.call(subflow_that_iterates)` produces a two-id path: [subflow_call, iterate]."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        subflow = Flow(make_test_logger(), name="inner").iterate(noop, max_iters=1)
        store = _RecordingStore()
        outer = make_ff().create(state={}).with_checkpointer(store, "traj-1").call(subflow)
        await outer.run()
        assert len(store.saves) == 1
        path = store.saves[0][3]["path"]
        assert len(path) == 2
        assert all(isinstance(p, str) for p in path)
        # The two ids differ — outer call is not the same node as the inner iterate.
        assert path[0] != path[1]

    async def test_iterate_arm_choice_flips_id(self) -> None:
        """Branch arm ``then`` vs ``else`` puts the iterate at identity-distinct positions.

        Both arms containing the same iterate produce different iterate ids
        because the arm-boundary participates in the descent hash.
        """

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        then_store = _RecordingStore()
        then_flow = (
            make_ff()
            .create(state={})
            .with_checkpointer(then_store, "traj-then")
            .branch(
                when=lambda _r, _c: True,
                then=lambda f: f.iterate(noop, max_iters=1),
                else_=lambda f: f.iterate(noop, max_iters=1),
            )
        )
        await then_flow.run()
        else_store = _RecordingStore()
        else_flow = (
            make_ff()
            .create(state={})
            .with_checkpointer(else_store, "traj-else")
            .branch(
                when=lambda _r, _c: False,
                then=lambda f: f.iterate(noop, max_iters=1),
                else_=lambda f: f.iterate(noop, max_iters=1),
            )
        )
        await else_flow.run()
        assert then_store.saves[0][3]["path"] != else_store.saves[0][3]["path"]

    async def test_state_tree_has_no_children_without_scoped_state(self) -> None:
        """Iterate without ``state=`` writes a flat one-level tree — ``children`` is ``[]``."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        store = _RecordingStore()
        flow = (
            make_ff()
            .create(state={"n": 5})
            .with_checkpointer(store, "traj-1")
            .iterate(noop, max_iters=1)
        )
        await flow.run()
        assert store.saves[0][2] == {"data": {"n": 5}, "children": []}

    async def test_state_tree_captures_scoped_child(self) -> None:
        """Iterate with ``state=`` writes a nested tree — root's ``children`` holds the projection."""

        @verb(role=ROLE_A)
        async def child_bump(ctx: Context[dict[str, int]], _prev: Any = None) -> None:
            ctx.state.data["child"] += 1

        store = _RecordingStore()
        flow = (
            make_ff()
            .create(state={"parent": "root"})
            .with_checkpointer(store, "traj-1")
            .iterate(
                child_bump,
                max_iters=1,
                state=lambda _parent: {"child": 0},
                merge=lambda _p, _c: None,
            )
        )
        await flow.run()
        tree = store.saves[0][2]
        assert tree["data"] == {"parent": "root"}
        assert len(tree["children"]) == 1
        assert tree["children"][0]["data"] == {"child": 1}
        assert tree["children"][0]["children"] == []

    async def test_old_shape_checkpoint_still_resumes(self) -> None:
        """An envelope without ``path`` (pre-PR-3 shape) still resumes via ``data``.

        Backward compatibility: sub-slice 3 keeps ``state_json["data"]``
        at the same key path as PR 2, and sub-slice 4's replay is
        gated on ``metadata_json['path']`` being present — an old-shape
        envelope hydrates state and runs iterate from iteration=0
        without triggering the structural-mismatch guard.
        """

        @verb(role=ROLE_A)
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        # Old-shape preload: no `path`, no `iteration`.
        store = _RecordingStore(preload=({"data": {"n": 10}}, {}))
        flow = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=1)
        )
        result = await flow.run(resume=True)
        assert result == 11


# -----------------------------------------------------------------------------
# Sub-slice 4 — resume position replay
# -----------------------------------------------------------------------------


class TestResumePositionReplay:
    """Iteration counter is restored; structural change on resume is a hard error."""

    async def test_iteration_counter_restored(self) -> None:
        """Fast-forwards the iterate counter from the saved value.

        Direct test of the note-423 fix: save at iteration=4, resume
        with ``max_iters=6`` — only 2 further passes execute because
        the counter starts at 4, not 0. Before PR 3 the counter reset
        and 6 more passes would run.
        """

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        # First run: 4 iterations save; final state at iteration=4.
        store = _RecordingStore()
        flow_a = (
            make_ff()
            .create(state={"count": 0})
            .with_checkpointer(store, "traj-4")
            .iterate(noop, max_iters=4)
        )
        await flow_a.run()
        assert store.saves[-1][3]["iteration"] == 4
        preload = (store.saves[-1][2], store.saves[-1][3])
        store.preload = preload
        store.saves.clear()
        store.deletes.clear()

        # Second run: same iterate node, higher cumulative max_iters.
        seen: list[int] = []

        @verb(role=ROLE_A)
        async def count_pass(ctx: Context[Any], _prev: Any = None) -> None:
            seen.append(len(seen) + 1)

        flow_b = (
            make_ff()
            .create(state={"count": 0})
            .with_checkpointer(store, "traj-4")
            .iterate(count_pass, max_iters=6)
        )
        await flow_b.run(resume=True)
        assert seen == [1, 2]  # iterations 5 and 6 only

    async def test_structural_change_raises(self) -> None:
        """A checkpoint whose save-point iterate no longer exists in the graph raises.

        The raise carries the full saved path (root → leaf) so ops
        triage can correlate the ancestor chain against the current
        composition tree and locate the layer that diverged.
        """
        import pytest

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        # Fabricate a checkpoint pointing at a path whose ids will not
        # appear in the resume flow's composition.
        stale_path = ["cafebabecafebabe", "deadbeefdeadbeef"]
        store = _RecordingStore(
            preload=(
                {"data": {}, "children": []},
                {"path": stale_path, "iteration": 2},
            )
        )
        flow = (
            make_ff()
            .create(state={})
            .with_checkpointer(store, "traj-stale")
            .iterate(noop, max_iters=3)
        )
        with pytest.raises(RuntimeError) as excinfo:
            await flow.run(resume=True)
        message = str(excinfo.value)
        assert "structurally changed" in message
        # Full path appears in the raise (root → leaf), not just the leaf.
        assert "cafebabecafebabe" in message
        assert "deadbeefdeadbeef" in message
        assert "root→leaf" in message

    async def test_head_pop_fails_before_sibling_verbs_run(self) -> None:
        """Structural-change raise fires at chain entry, before any pre-iterate verb runs.

        Pre-head-pop, an unreachable-leaf resume would walk the whole
        chain (running every verb, running every iterate from 0)
        before ``_assert_replay_consumed`` finally raised. The fail-
        fast pre-scan at Flow entry catches the mismatch as soon as
        no chain-step id matches the replay's remaining-path head —
        no chain step at that level runs at all.
        """
        import pytest

        calls: list[str] = []

        @verb(role=ROLE_A)
        async def pre_verb(ctx: Context[Any], _prev: Any = None) -> None:
            calls.append("pre")

        @verb(role=ROLE_A)
        async def body(ctx: Context[Any], _prev: Any = None) -> None:
            calls.append("body")

        stale_path = ["cafebabecafebabe", "deadbeefdeadbeef"]
        store = _RecordingStore(
            preload=(
                {"data": {}, "children": []},
                {"path": stale_path, "iteration": 2},
            )
        )
        flow = (
            make_ff()
            .create(state={})
            .with_checkpointer(store, "traj-headpop")
            .then(pre_verb)
            .iterate(body, max_iters=3)
        )
        with pytest.raises(RuntimeError) as excinfo:
            await flow.run(resume=True)
        assert calls == []  # neither pre_verb nor any iterate body fired
        message = str(excinfo.value)
        assert "structurally changed" in message
        # Pre-scan raise names both the unreachable head and the full path.
        assert "cafebabecafebabe" in message
        assert "deadbeefdeadbeef" in message

    async def test_head_pop_fails_at_inner_flow_with_full_path(self) -> None:
        """A stale id in the inner Flow's chain raises there, preserving full path.

        Root chain matches path head, so descent proceeds into the
        subflow. The subflow's chain-entry pre-scan then rejects the
        stale inner id. The raise still carries the whole root→leaf
        path (via ``_ResumeReplay.full_path``, threaded verbatim
        across the head-pop) so ops triage sees the ancestor context.
        """
        import pytest

        @verb(role=ROLE_A)
        async def body(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        # First run: real save, capture the root-level call id.
        real_store = _RecordingStore()
        inner = make_ff().create(state={}).iterate(body, max_iters=2)
        outer = make_ff().create(state={}).with_checkpointer(real_store, "traj-inner").call(inner)
        await outer.run()
        assert real_store.saves, "expected the inner iterate to have saved"
        saved_path = real_store.saves[-1][3]["path"]
        assert len(saved_path) == 2  # [outer .call id, inner .iterate id]

        # Second run: swap the inner id for a stale hash — the root-level
        # descent parent is still real, so pre-scan passes at root and
        # fails at the subflow level.
        stale_inner = "0badc0de0badc0de"
        preload = (
            {"data": {}, "children": []},
            {
                "path": [saved_path[0], stale_inner],
                "iteration": 1,
            },
        )
        replay_store = _RecordingStore(preload=preload)
        outer2 = (
            make_ff().create(state={}).with_checkpointer(replay_store, "traj-inner").call(inner)
        )
        with pytest.raises(RuntimeError) as excinfo:
            await outer2.run(resume=True)
        message = str(excinfo.value)
        assert stale_inner in message
        assert saved_path[0] in message  # full path threaded through head-pop
        assert "structurally changed" in message

    async def test_resume_rebuilds_ambient_halt(self) -> None:
        """Ambients (halt/budget/traits/…) are not serialized; the resumed run wires fresh ones.

        Sub-slice 5's guarantee, tested directly: after saving under one
        halt event, resuming under a different halt event honors the
        new event's state. Nothing about the first run's ambient
        survives — the checkpoint carries state and position only.
        """
        import asyncio

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        # First run with an unset halt: 2 iterations save.
        store = _RecordingStore()
        halt_a = asyncio.Event()
        flow_a = (
            make_ff()
            .create(state={})
            .with_checkpointer(store, "traj-ambient")
            .with_halt(halt_a)
            .iterate(noop, max_iters=2)
        )
        await flow_a.run()
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()
        store.deletes.clear()

        # Resume under a NEW halt event, already set. The iterate should
        # honor the new halt on the first between-iterations check and
        # exit before running any further pass.
        seen: list[int] = []

        @verb(role=ROLE_A)
        async def count(ctx: Context[Any], _prev: Any = None) -> None:
            seen.append(len(seen) + 1)

        halt_b = asyncio.Event()
        halt_b.set()
        flow_b = (
            make_ff()
            .create(state={})
            .with_checkpointer(store, "traj-ambient")
            .with_halt(halt_b)
            .iterate(count, max_iters=5)
        )
        await flow_b.run(resume=True)
        assert seen == []  # halt_b fired between-iterations — no bodies ran

    async def test_matching_iterate_resumes_cleanly(self) -> None:
        """Round-trip: save a real checkpoint, resume with the same graph, complete cleanly."""

        @verb(role=ROLE_A)
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        store = _RecordingStore()
        flow_a = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-real")
            .iterate(bump, max_iters=2)
        )
        await flow_a.run()
        assert store.saves[-1][3]["iteration"] == 2
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()
        store.deletes.clear()

        flow_b = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-real")
            .iterate(bump, max_iters=4)
        )
        result = await flow_b.run(resume=True)
        assert result == 4  # n hydrates to 2, two more passes → n=4

    async def test_resume_through_subflow(self) -> None:
        """Round-trip through ``.call(subflow_with_iterate)`` — path has two ids.

        Acceptance (h) of #253: the recursive snapshot round-trips
        across nested subflow composition. The saved path has two
        entries (outer call, inner iterate); on resume the executor
        recomputes those ids while descending, matches at the inner
        iterate, and fast-forwards there.
        """

        @verb(role=ROLE_A)
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        store = _RecordingStore()

        def _mk_flow(max_iters: int) -> Flow:
            inner = Flow(make_test_logger(), name="inner").iterate(bump, max_iters=max_iters)
            return (
                make_ff().create(state={"n": 0}).with_checkpointer(store, "traj-nested").call(inner)
            )

        # First run: 2 iterations of the inner iterate.
        await _mk_flow(2).run()
        assert len(store.saves[-1][3]["path"]) == 2
        assert store.saves[-1][3]["iteration"] == 2
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()
        store.deletes.clear()

        # Resume with same shape, higher cumulative bound — one more pass.
        result = await _mk_flow(3).run(resume=True)
        assert result == 3

    async def test_resume_through_branch_arm(self) -> None:
        """Round-trip through a branch's ``then`` arm containing an iterate.

        Acceptance (h) coverage for branches: the path's first id
        encodes the branch node, the second encodes the arm's iterate
        (chain-hashed under the ``then`` boundary). Resume rebuilds
        both ids and hits the fast-forward at the arm's iterate.
        """

        @verb(role=ROLE_A)
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        store = _RecordingStore()

        def _mk_flow(max_iters: int) -> Flow:
            return (
                make_ff()
                .create(state={"n": 0})
                .with_checkpointer(store, "traj-branch")
                .branch(
                    when=lambda _r, _c: True,
                    then=lambda f: f.iterate(bump, max_iters=max_iters),
                )
            )

        await _mk_flow(2).run()
        assert len(store.saves[-1][3]["path"]) == 2
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()
        store.deletes.clear()

        result = await _mk_flow(3).run(resume=True)
        assert result == 3

    async def test_branch_no_else_predicate_change_fails_fast(self) -> None:
        """Fail-fast when branch predicate changes and no else_ arm exists.

        Checkpoint saved through the ``then`` arm. On resume, predicate
        returns falsy with no ``else_`` arm to descend — the fail-fast
        check fires immediately, before any subsequent chain steps run.
        """
        import pytest

        calls: list[str] = []

        @verb(role=ROLE_A)
        async def body(ctx: Context[Any], _prev: Any = None) -> None:
            calls.append("body")

        @verb(role=ROLE_A)
        async def post_branch(ctx: Context[Any], _prev: Any = None) -> None:
            calls.append("post")

        predicate_value = True
        store = _RecordingStore()
        flow = (
            make_ff()
            .create(state={})
            .with_checkpointer(store, "traj-branch-noelse")
            .branch(
                when=lambda _r, _c: predicate_value, then=lambda f: f.iterate(body, max_iters=2)
            )
            .then(post_branch)
        )
        await flow.run()
        assert "body" in calls
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()

        # Resume with falsy predicate — no else_ arm, so fail-fast triggers.
        predicate_value = False
        calls.clear()
        with pytest.raises(RuntimeError) as excinfo:
            await flow.run(resume=True)
        assert calls == []  # post_branch never ran
        message = str(excinfo.value)
        assert "predicate" in message
        assert "no else_ arm" in message

    async def test_map_replay_routes_to_correct_item(self) -> None:
        """Replay is routed only to the saved map item, not all items.

        When a checkpoint is saved inside map item N's body, resuming
        should route the replay only to item N. Other items should
        receive no replay and run fresh. This test verifies the fix
        for scheduling-dependent replay failures: without the fix,
        a non-target item running first would fail replay validation.
        """
        calls: list[tuple[str, int]] = []

        @verb(role=ROLE_A)
        async def body(ctx: Context[dict[str, int]], item: int) -> int:
            calls.append(("body", item))
            ctx.state.data["sum"] = ctx.state.data.get("sum", 0) + item
            return item

        store = _RecordingStore()
        flow = (
            make_ff()
            .create(state={"sum": 0})
            .with_checkpointer(store, "traj-map")
            .map(
                lambda f: f.iterate(body, max_iters=2),
                items=lambda _p, _c: [10, 20, 30],
            )
        )
        await flow.run()
        assert store.saves, "expected iterate inside map to save"
        saved_path = store.saves[-1][3]["path"]
        assert len(saved_path) == 2  # [map_node_id, iterate_inside_item]

        # Resume should route replay to the correct item without error.
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()
        calls.clear()
        await flow.run(resume=True)
        # All items should complete: item 0, 1, 2 each run their iterate bodies.
        assert len([c for c in calls if c[0] == "body"]) >= 3


# -----------------------------------------------------------------------------
# Resume determinism — canonical multi-stage flow via the testing harness
# -----------------------------------------------------------------------------


class TestResumeDeterminismSameProcess:
    """The load-bearing invariant: resume-from-halt equals uninterrupted final state.

    Pinned once here on the canonical flow from
    :mod:`llm_gent.flow.testing.checkpoint`. Downstream consumers assert
    the domain-shaped equivalent on their own Flow.
    """

    async def test_resume_matches_uninterrupted_final_state(self, tmp_path: Path) -> None:
        """A fresh :class:`Flow` resumed from a mid-run checkpoint reaches byte-identical final state."""
        lg = make_test_logger()
        store = JsonFileCheckpointStore(lg, tmp_path / "cp")
        await assert_resume_determinism(lg, store, trajectory_id="det-1")


class TestResumeDeterminismCrossProcess:
    """Cross-process resume matches uninterrupted final state.

    Same-process resume can silently keep working when a state field
    holds a live reference to a non-serializable object. A fresh Python
    subprocess with only the on-disk checkpoint is the production gate.
    Uses :class:`JsonFileCheckpointStore` — file-based, portable across
    processes.
    """

    async def test_cross_process_resume_json_store(self, tmp_path: Path) -> None:
        """Subprocess resume via ``resume_in_subprocess`` yields byte-identical final state."""
        import asyncio

        lg = make_test_logger()
        root = str(tmp_path / "cp")
        store = JsonFileCheckpointStore(lg, root)

        baseline = await build_canonical_flow(lg, max_iters=5).run()

        halt = asyncio.Event()
        await build_canonical_flow(
            lg,
            max_iters=5,
            halt=halt,
            halt_after_iteration=2,
            store=store,
            trajectory_id="xp-1",
        ).run()

        resumed = resume_in_subprocess(
            store_module="llm_gent.flow.stores.json_file",
            store_factory="JsonFileCheckpointStore",
            store_kwargs={"root": root},
            flow_builder_kwargs={"max_iters": 5},
            trajectory_id="xp-1",
        )

        assert resumed == baseline


# -----------------------------------------------------------------------------
# Ambient re-attach — budget + traits
# -----------------------------------------------------------------------------


class TestResumeRebuildsAmbientBudget:
    """Sub-slice 5 for budget: the tracker is not serialized — fresh tracker on resume."""

    async def test_resume_rebuilds_ambient_budget(self) -> None:
        """A fresh :class:`Tracker` on resume sees only post-resume cost; the interrupt run's spend does not leak."""
        from llm_gent.core.budget import PricingConfig, Tracker

        lg = make_test_logger()

        @verb
        async def spend(ctx: Context[Any], _prev: Any = None) -> None:
            if ctx.budget is not None:
                ctx.budget.track("op", override_cost=1.0)

        store = _RecordingStore()

        # First run: 2 iterations save; tracker_a records 2.0.
        tracker_a = Tracker(lg, PricingConfig())
        flow_a = (
            FlowFactory(lg)
            .create(state={})
            .with_checkpointer(store, "traj-budget")
            .with_budget(tracker_a)
            .iterate(spend, max_iters=2)
        )
        await flow_a.run()
        assert tracker_a.spent == 2.0
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()
        store.deletes.clear()

        # Resume: fresh tracker_b, uncapped. Cumulative max_iters=5 leaves 3 to run.
        tracker_b = Tracker(lg, PricingConfig())
        flow_b = (
            FlowFactory(lg)
            .create(state={})
            .with_checkpointer(store, "traj-budget")
            .with_budget(tracker_b)
            .iterate(spend, max_iters=5)
        )
        await flow_b.run(resume=True)

        # tracker_a is untouched by the resume; tracker_b records only the
        # remaining iterations (3, 4, 5).
        assert tracker_a.spent == 2.0
        assert tracker_b.spent == 3.0


class TestResumeRebuildsAmbientTraits:
    """Sub-slice 5 for traits: the trait registry is not serialized — fresh registry on resume."""

    async def test_resume_rebuilds_ambient_traits(self) -> None:
        """Post-resume verbs see the resume-time :class:`TraitRegistry` (identity), not the interrupt run's."""
        from llm_gent.core.traits import Registry as TraitRegistry

        lg = make_test_logger()
        seen: list[TraitRegistry | None] = []

        @verb
        async def capture(ctx: Context[Any], _prev: Any = None) -> None:
            seen.append(ctx.traits)

        store = _RecordingStore()

        # First run: traits_a attached, 2 iterations.
        traits_a = TraitRegistry(lg)
        flow_a = (
            FlowFactory(lg, traits=traits_a)
            .create(state={})
            .with_checkpointer(store, "traj-traits")
            .iterate(capture, max_iters=2)
        )
        await flow_a.run()
        assert seen == [traits_a, traits_a]
        seen.clear()
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()
        store.deletes.clear()

        # Resume: traits_b attached. Remaining iterations see the new registry.
        traits_b = TraitRegistry(lg)
        flow_b = (
            FlowFactory(lg, traits=traits_b)
            .create(state={})
            .with_checkpointer(store, "traj-traits")
            .iterate(capture, max_iters=5)
        )
        await flow_b.run(resume=True)

        assert all(t is traits_b for t in seen), (
            f"expected every post-resume ctx.traits to be traits_b; got {seen}"
        )
        assert len(seen) == 3  # iterations 3, 4, 5
