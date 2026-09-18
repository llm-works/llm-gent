# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow-level :class:`CheckpointStore` — Protocol shape, wiring, and save/resume.

Exercises:

- Protocol conformance for the new Flow-level store.
- :meth:`Flow.with_checkpointer` + :meth:`FlowFactory.with_checkpointer`.
- ``state_type=`` plumbing on Flow / FlowFactory / .call / .iterate / .map.
- Save-at-``.iterate``-boundary end-to-end.
- Framework-owned ``state_type.from_dict`` dispatch on ``run(resume=True)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Self

from llm_gent.flow import (
    CheckpointStore,
    Context,
    Flow,
    FlowFactory,
    verb,
)
from llm_gent.flow.nodes import _Iterate, _Map

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
# state_type plumbing
# -----------------------------------------------------------------------------


class TestStateTypePlumbing:
    """``state_type=`` reaches Flow / FlowFactory and every scoped composition site."""

    def test_flow_init_captures_state_type(self) -> None:
        """``Flow(..., state_type=T)`` stores it on the flow."""
        flow = Flow(make_test_logger(), state_type=Counter)
        assert flow._state_type is Counter

    def test_factory_captures_and_threads(self) -> None:
        """``FlowFactory(state_type=T)`` threads into ``create()``."""
        ff = FlowFactory(make_test_logger(), state_type=Counter)
        flow = ff.create()
        assert flow._state_type is Counter

    def test_call_records_state_type(self) -> None:
        """``.call(state_type=T)`` stores it on the node."""

        @verb(role=ROLE_A)
        async def step(ctx: Context[Any], x: int) -> int:
            return x

        subflow = make_ff().create()
        subflow.call(step)
        parent = make_ff().create().call(subflow, state=lambda _p: {}, state_type=Counter)
        assert parent._nodes[-1].state_type is Counter

    def test_iterate_records_state_type(self) -> None:
        """``.iterate(state_type=T)`` stores it on the iterate node."""

        @verb(role=ROLE_A)
        async def step(ctx: Context[Any], _: Any = None) -> int:
            return 0

        flow = make_ff().create().iterate(step, max_iters=1, state_type=Counter)
        node = flow._nodes[-1].target
        assert isinstance(node, _Iterate)
        assert node.state_type is Counter

    def test_map_records_state_type(self) -> None:
        """``.map(state_type=T)`` stores it on the map node."""

        @verb(role=ROLE_A)
        async def step(ctx: Context[Any], item: Any) -> Any:
            return item

        flow = make_ff().create().map(step, items=lambda _p, _c: [1], state_type=Counter)
        node = flow._nodes[-1].target
        assert isinstance(node, _Map)
        assert node.state_type is Counter


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

    async def test_saves_carry_metadata_schema_version(self) -> None:
        """Every save's ``metadata_json`` carries ``schema_version=1``, ``path``, ``iteration``."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        store = _RecordingStore()
        flow = (
            make_ff().create(state={}).with_checkpointer(store, "traj-1").iterate(noop, max_iters=1)
        )
        await flow.run()
        meta = store.saves[0][3]
        assert meta["schema_version"] == 1
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
            Flow(make_test_logger(), state=Counter(), state_type=Counter)
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=2)
        )
        await flow.run()
        assert store.saves[-1][2] == {"data": {"n": 2}, "children": []}


# -----------------------------------------------------------------------------
# Resume — framework-owned from_dict dispatch
# -----------------------------------------------------------------------------


class TestResumeHydration:
    """``run(resume=True)`` loads the checkpoint and hydrates via ``from_dict``."""

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

        store = _RecordingStore(preload=({"data": {"n": 42}}, {"schema_version": 1}))
        flow = make_ff().create().with_checkpointer(store, "traj-1").call(peek)
        await flow.run(state={"n": 0}, resume=True)
        assert seen == [{"n": 42}]

    async def test_resume_typed_payload_calls_from_dict(self) -> None:
        """When ``state_type=T`` is set, framework calls ``T.from_dict``."""

        seen: list[Any] = []

        @verb(role=ROLE_A)
        async def peek(ctx: Context[Counter]) -> None:
            seen.append(ctx.state.data)

        store = _RecordingStore(preload=({"data": {"n": 7}}, {"schema_version": 1}))
        flow = (
            Flow(make_test_logger(), state_type=Counter)
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
        """Same round-trip with a :class:`StateData` payload via ``state_type=``."""

        @verb(role=ROLE_A)
        async def bump(ctx: Context[Counter], _prev: Any = None) -> int:
            ctx.state.data.n += 1
            return ctx.state.data.n

        store = _RecordingStore()

        # First run: 2 iterations save; leave the checkpoint at iteration=2.
        flow_a = (
            Flow(make_test_logger(), state=Counter(), state_type=Counter)
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
            Flow(make_test_logger(), state=Counter(), state_type=Counter)
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=3)
        )
        result = await flow_b.run(resume=True)
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
        store = _RecordingStore(preload=({"data": {"n": 10}}, {"schema_version": 1}))
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
                {"schema_version": 1, "path": stale_path, "iteration": 2},
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
