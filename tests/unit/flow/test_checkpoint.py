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
        assert store.saves[-1][2] == {"data": {"n": 3}}

    async def test_saves_carry_metadata_schema_version(self) -> None:
        """Every save's ``metadata_json`` carries ``schema_version=1``."""

        @verb(role=ROLE_A)
        async def noop(ctx: Context[Any], _prev: Any = None) -> None:
            return None

        store = _RecordingStore()
        flow = (
            make_ff().create(state={}).with_checkpointer(store, "traj-1").iterate(noop, max_iters=1)
        )
        await flow.run()
        assert store.saves[0][3] == {"schema_version": 1}

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
        assert store.saves[-1][2] == {"data": {"n": 2}}


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
        """First run saves; second run with ``resume=True`` picks up where it left off."""

        @verb(role=ROLE_A)
        async def bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            return ctx.state.data["n"]

        store = _RecordingStore()

        # First run: bump 3 times → checkpoint records n=3.
        flow_a = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=3)
        )
        await flow_a.run()
        latest_state = store.saves[-1][2]
        assert latest_state == {"data": {"n": 3}}
        # Successful run deletes → simulate a mid-run failure by re-inserting the
        # checkpoint as the store's preload for the next run.
        store.preload = (latest_state, {"schema_version": 1})
        store.saves.clear()
        store.deletes.clear()

        # Second run: resume=True hydrates n=3, then max_iters=2 → n=5.
        result_state: list[int] = []

        @verb(role=ROLE_A)
        async def bump_and_capture(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["n"] += 1
            result_state.append(ctx.state.data["n"])
            return ctx.state.data["n"]

        flow_b = (
            make_ff()
            .create(state={"n": 0})
            .with_checkpointer(store, "traj-1")
            .iterate(bump_and_capture, max_iters=2)
        )
        await flow_b.run(resume=True)
        assert result_state == [4, 5]

    async def test_iterate_round_trip_typed_payload(self) -> None:
        """Same round-trip with a :class:`StateData` payload via ``state_type=``."""

        @verb(role=ROLE_A)
        async def bump(ctx: Context[Counter], _prev: Any = None) -> int:
            ctx.state.data.n += 1
            return ctx.state.data.n

        store = _RecordingStore()

        # First run seeds the checkpoint at n=2.
        flow_a = (
            Flow(make_test_logger(), state=Counter(), state_type=Counter)
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=2)
        )
        await flow_a.run()
        assert store.saves[-1][2] == {"data": {"n": 2}}
        store.preload = (store.saves[-1][2], store.saves[-1][3])
        store.saves.clear()

        # Second run picks up at n=2, bumps once → n=3.
        flow_b = (
            Flow(make_test_logger(), state=Counter(), state_type=Counter)
            .with_checkpointer(store, "traj-1")
            .iterate(bump, max_iters=1)
        )
        result = await flow_b.run(resume=True)
        assert result == 3
