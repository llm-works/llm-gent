# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for Panel + aggregation helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, Flow, Panel, Role, verb
from llm_gent.flow.panel import majority, mean, unanimous, weighted
from llm_gent.flow.stores.json_file import JsonFileCheckpointStore

from .conftest import ROLE_A, ROLE_B, make_ff, make_test_logger


class TestAggregators:
    """Aggregation helpers exposed by the panel module."""

    def test_majority_returns_most_common(self) -> None:
        """majority picks the most frequently occurring value."""
        assert majority(["a", "b", "a", "c", "a"]) == "a"

    def test_majority_tie_first_seen_wins(self) -> None:
        """On a tie, the first-inserted value wins (Counter semantics)."""
        assert majority(["a", "b", "a", "b"]) == "a"

    def test_majority_empty_raises(self) -> None:
        """majority on an empty list is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            majority([])

    def test_unanimous_all_agree(self) -> None:
        """unanimous returns the shared value when every vote matches."""
        assert unanimous(["yes", "yes", "yes"]) == "yes"

    def test_unanimous_disagree_returns_none(self) -> None:
        """unanimous returns None when any vote diverges."""
        assert unanimous(["yes", "no", "yes"]) is None

    def test_unanimous_empty_returns_none(self) -> None:
        """unanimous on an empty list returns None (no value to agree on)."""
        assert unanimous([]) is None

    def test_mean(self) -> None:
        """mean returns the arithmetic mean of the votes."""
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty(self) -> None:
        """mean on an empty list returns 0.0 (avoids ZeroDivisionError)."""
        assert mean([]) == 0.0

    def test_weighted_normal(self) -> None:
        """weighted returns a properly weight-normalized average."""
        assert weighted([(1.0, 0.5), (3.0, 0.5)]) == pytest.approx(2.0)

    def test_weighted_uneven_weights(self) -> None:
        """weighted respects unequal weights."""
        assert weighted([(1.0, 0.25), (5.0, 0.75)]) == pytest.approx(4.0)

    def test_weighted_zero_weight_returns_zero(self) -> None:
        """weighted with total weight of 0 returns 0.0 (safe on empty / all-zero)."""
        assert weighted([]) == 0.0
        assert weighted([(1.0, 0.0), (5.0, 0.0)]) == 0.0


class TestPanel:
    """Panel fans out via ctx.flow and aggregates results."""

    def test_empty_verbs_raises(self) -> None:
        """Constructing a Panel with no verbs is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            Panel(verbs=[], aggregate=majority)

    @pytest.mark.asyncio
    async def test_panel_fans_out_and_sums(self) -> None:
        """Each verb runs in parallel and its result feeds the aggregate."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def add_one(ctx: Context, x: int) -> int:
            """Return x + 1."""
            return x + 1

        @verb(role=ROLE_A)
        async def add_two(ctx: Context, x: int) -> int:
            """Return x + 2."""
            return x + 2

        flow.register(add_one)
        flow.register(add_two)

        panel = Panel([add_one, add_two], aggregate=sum)

        @verb(role=ROLE_A)
        async def outer(ctx: Context, x: int) -> int:
            """Run the panel and return its aggregate."""
            return await panel.run(ctx, x)

        flow.register(outer)
        result = await flow.dispatch("outer", 10)
        # add_one(10)=11, add_two(10)=12, sum=23
        assert result == 23

    @pytest.mark.asyncio
    async def test_panel_with_custom_registered_names(self) -> None:
        """Panel dispatches correctly when verbs are registered with custom names."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def impl_a(ctx: Context, x: int) -> int:
            """Return x * 2."""
            return x * 2

        @verb(role=ROLE_A)
        async def impl_b(ctx: Context, x: int) -> int:
            """Return x * 3."""
            return x * 3

        # Register with custom names different from __name__
        flow.register(impl_a, name="custom_doubler")
        flow.register(impl_b, name="custom_tripler")

        panel = Panel([impl_a, impl_b], aggregate=sum)

        @verb(role=ROLE_A)
        async def outer(ctx: Context, x: int) -> int:
            """Run the panel."""
            return await panel.run(ctx, x)

        flow.register(outer)
        result = await flow.dispatch("outer", 5)
        # impl_a(5)=10, impl_b(5)=15, sum=25
        assert result == 25

    @pytest.mark.asyncio
    async def test_panel_with_majority(self) -> None:
        """A 3-judge panel returning majority verdict works end-to-end."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def yes_a(ctx: Context) -> str:
            """Vote yes."""
            return "yes"

        @verb(role=ROLE_A)
        async def yes_b(ctx: Context) -> str:
            """Vote yes."""
            return "yes"

        @verb(role=ROLE_A)
        async def no_one(ctx: Context) -> str:
            """Vote no."""
            return "no"

        for v in (yes_a, yes_b, no_one):
            flow.register(v)

        panel = Panel([yes_a, yes_b, no_one], aggregate=majority)

        @verb(role=ROLE_A)
        async def outer(ctx: Context) -> str:
            """Run the panel."""
            return await panel.run(ctx)

        flow.register(outer)
        assert await flow.dispatch("outer") == "yes"

    @pytest.mark.asyncio
    async def test_panel_routes_per_verb_role(self) -> None:
        """Each inner verb receives a saia bound to its own role, not the caller's."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def see_a(ctx: Context) -> Role:
            """Return the ctx role."""
            return ctx.role

        @verb(role=ROLE_B)
        async def see_b(ctx: Context) -> Role:
            """Return the ctx role."""
            return ctx.role

        flow.register(see_a)
        flow.register(see_b)

        panel = Panel([see_a, see_b], aggregate=list)

        @verb(role=ROLE_A)
        async def outer(ctx: Context) -> list[Role]:
            """Run the panel and return the two roles each verb saw."""
            return await panel.run(ctx)

        flow.register(outer)
        roles = await flow.dispatch("outer")
        assert set(roles) == {ROLE_A, ROLE_B}

    @pytest.mark.asyncio
    async def test_panel_forwards_live_scope_state_to_inner_verbs(self) -> None:
        """Inner verbs see the caller's live ``ctx.state``, not the runtime construction state.

        Scenario: outer flow's construction state is ``{"outer": True}``;
        it calls a subflow projected to ``{"scoped": True, "outer": False}``;
        that subflow runs a verb that fires a Panel. Without state
        forwarding, the Panel's inner verbs would see the runtime flow's
        construction state through ``ctx.flow.dispatch``; with it they
        see the caller's projected scope.
        """
        observed: list[dict[str, Any]] = []

        @verb(role=ROLE_A)
        async def peek_a(ctx: Context) -> str:
            """Record ``ctx.state.data`` and return a marker."""
            observed.append(dict(ctx.state.data))
            return "a"

        @verb(role=ROLE_A)
        async def peek_b(ctx: Context) -> str:
            """Record ``ctx.state.data`` and return a marker."""
            observed.append(dict(ctx.state.data))
            return "b"

        panel = Panel([peek_a, peek_b], aggregate=list)

        @verb(role=ROLE_A)
        async def run_panel(ctx: Context, _prev: object) -> list[str]:
            """Fire the Panel from inside the projected scope."""
            return await panel.run(ctx)

        inner = make_ff().create().call(run_panel)
        outer = make_ff().create(state={"outer": True})
        outer.register(peek_a)
        outer.register(peek_b)
        outer.call(inner, state=lambda _p: {"scoped": True, "outer": False})

        await outer.run(())
        assert observed == [
            {"scoped": True, "outer": False},
            {"scoped": True, "outer": False},
        ]

    @pytest.mark.asyncio
    async def test_panel_state_forwarding_independent_of_nested_resumable_flow(
        self, tmp_path: Path
    ) -> None:
        """Panel state forwarding doesn't interfere with nested Flow resume.

        Scenario: Panel dispatches a verb that creates and resumes a nested
        Flow with its own checkpointer. The Panel's forwarded state should
        reach the inner verb; the nested Flow's resume should hydrate its
        own checkpoint independently.
        """
        from dataclasses import dataclass, field

        @dataclass
        class _RecordingStore:
            saves: list[tuple[str, str, int]] = field(default_factory=list)
            loaded: dict[str, tuple[dict, dict]] = field(default_factory=dict)

            def save_checkpoint(
                self,
                client_flow_id: str,
                node_path: str,
                iteration: int,
                state_json: dict[str, Any],
                metadata_json: dict[str, Any],
            ) -> None:
                self.saves.append((client_flow_id, node_path, iteration))
                self.loaded[client_flow_id] = (state_json, metadata_json)

            def load_checkpoint(
                self,
                client_flow_id: str,
                node_path: str | None = None,
                iteration: int | None = None,
            ) -> tuple[dict[str, Any], dict[str, Any]] | None:
                return self.loaded.get(client_flow_id)

            def delete_checkpoint(self, client_flow_id: str) -> None:
                self.loaded.pop(client_flow_id, None)

        panel_state_observed: list[dict[str, Any]] = []
        nested_flow_result: list[int] = []
        store = _RecordingStore()

        @verb(role=ROLE_A)
        async def nested_bump(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            ctx.state.data["nested_n"] += 1
            return ctx.state.data["nested_n"]

        @verb(role=ROLE_A)
        async def run_nested_resumable(ctx: Context, _prev: Any = None) -> str:
            """Records Panel-forwarded state, then runs a nested resumable Flow."""
            panel_state_observed.append(dict(ctx.state.data))

            # Create and prime a nested flow with checkpointing.
            halt = asyncio.Event()

            @verb(role=ROLE_A)
            async def bump_then_halt(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
                ctx.state.data["nested_n"] += 1
                if ctx.state.data["nested_n"] >= 2:
                    halt.set()
                return ctx.state.data["nested_n"]

            primed = (
                make_ff()
                .create(state={"nested_n": 0})
                .with_checkpointer(store, "nested-traj")
                .iterate(bump_then_halt, max_iters=5)
                .with_halt(halt)
            )
            await primed.run()  # Runs 2 iterations, saves checkpoint, halts.

            # Resume the nested flow — hydrates its checkpoint independently.
            resumed = (
                make_ff()
                .create(state={"nested_n": 0})
                .with_checkpointer(store, "nested-traj")
                .iterate(nested_bump, max_iters=5)
            )
            result = await resumed.run(resume=True)
            nested_flow_result.append(result)
            return "done"

        panel = Panel([run_nested_resumable], aggregate=lambda x: x[0])

        @verb(role=ROLE_A)
        async def fire_panel(ctx: Context, _prev: object) -> str:
            return await panel.run(ctx)

        inner = make_ff().create().call(fire_panel)
        outer = make_ff().create(state={"outer_marker": "present"})
        outer.register(run_nested_resumable)
        outer.call(inner, state=lambda _p: {"panel_scope": "forwarded"})

        await outer.run(())

        # Panel state forwarding worked: inner verb saw the projected scope.
        assert panel_state_observed == [{"panel_scope": "forwarded"}]
        # Nested Flow resume worked independently: 2 from checkpoint + 3 more = 5.
        assert nested_flow_result == [5]

    @pytest.mark.asyncio
    async def test_panel_inner_verb_mutations_visible_to_caller(self) -> None:
        """Mutations by inner verbs are visible to the calling verb's state.

        When Panel dispatches with ``state=ctx.state``, inner verbs share
        the same State object. A mutation by one inner verb should be
        visible to subsequent inner verbs and to the caller after Panel
        returns.
        """

        @verb(role=ROLE_A)
        async def mutate_a(ctx: Context) -> str:
            ctx.state.data["a_ran"] = True
            ctx.state.data["counter"] = ctx.state.data.get("counter", 0) + 1
            return "a"

        @verb(role=ROLE_A)
        async def mutate_b(ctx: Context) -> str:
            ctx.state.data["b_ran"] = True
            ctx.state.data["counter"] = ctx.state.data.get("counter", 0) + 1
            return "b"

        panel = Panel([mutate_a, mutate_b], aggregate=list)
        caller_observed: list[dict[str, Any]] = []

        @verb(role=ROLE_A)
        async def run_panel_and_observe(ctx: Context, _prev: object) -> list[str]:
            result = await panel.run(ctx)
            caller_observed.append(dict(ctx.state.data))
            return result

        inner = make_ff().create().call(run_panel_and_observe)
        outer = make_ff().create(state={"initial": True})
        outer.register(mutate_a)
        outer.register(mutate_b)
        outer.call(inner, state=lambda _p: {"counter": 0})

        await outer.run(())

        # Both inner verbs' mutations should be visible after Panel returns.
        assert caller_observed == [{"counter": 2, "a_ran": True, "b_ran": True}]

    @pytest.mark.asyncio
    async def test_panel_propagates_local_halt_in_subflow(self) -> None:
        """Panel.run passes ctx.halt to dispatched verbs, not the outer flow's halt.

        Scenario: outer.with_halt(root) → middle.with_halt(local) → Panel.run
        The Panel's verbs should observe ``local``, not ``root``.
        """
        root_halt = asyncio.Event()
        local_halt = asyncio.Event()
        observed: list[asyncio.Event | None] = []

        @verb(role=ROLE_A)
        async def capture_a(ctx: Context) -> str:
            """Record the halt event we received."""
            observed.append(ctx.halt)
            return "a"

        @verb(role=ROLE_A)
        async def capture_b(ctx: Context) -> str:
            """Record the halt event we received."""
            observed.append(ctx.halt)
            return "b"

        panel = Panel([capture_a, capture_b], aggregate=list)

        @verb(role=ROLE_A)
        async def run_panel(ctx: Context, _: object) -> list[str]:
            """Run Panel from within a subflow with a local halt."""
            return await panel.run(ctx)

        # Build a middle subflow that has its own local halt event.
        middle = make_ff().create().with_halt(local_halt)
        middle.call(run_panel)

        # Build the outer/root flow with a different (root) halt event.
        # Register verbs on outer since ctx.flow points to the runtime.
        outer = make_ff().create().with_halt(root_halt)
        outer.register(capture_a)
        outer.register(capture_b)
        outer.call(middle)

        results = await outer.run(())
        assert set(results) == {"a", "b"}
        # Panel verbs received the middle flow's local_halt, not root_halt.
        assert observed == [local_halt, local_halt]


class TestPanelInsideIterateResumeBoundary:
    """The iterate iteration is the checkpoint boundary around a Panel.

    Panel has no checkpoint boundary of its own. When a halt fires
    during a Panel's ``asyncio.gather``, siblings are not cancelled —
    each dispatched verb observes ``ctx.halt`` on its own if it
    chooses (saia-backed verbs observe halt at call entry and mid-stream,
    so they abort fast; verbs that don't poll halt run to completion).
    The gather returns whatever the verbs returned; control goes back
    to the enclosing iterate, which honors halt at its next
    between-iterations check.

    The preserved checkpoint reflects the *completion* of the
    iteration containing the Panel. On resume, iterate proceeds at
    the next iteration and dispatches a fresh Panel — the halted
    iteration's Panel is never partially re-dispatched.
    """

    @pytest.mark.asyncio
    async def test_halted_iteration_does_not_partially_redispatch_on_resume(
        self, tmp_path: Path
    ) -> None:
        """Halt set from inside one Panel verb: iteration completes; resume proceeds at K+1 with a fresh Panel."""
        lg = make_test_logger()
        store = JsonFileCheckpointStore(lg, tmp_path / "cp")
        halt = asyncio.Event()
        runs: list[tuple[int, str]] = []

        # Voters receive their iteration number positionally — Flow.dispatch
        # rebuilds ctx.state from the flow's construction state, not the
        # in-flight iterate state, so ctx.state.data is not usable here.

        @verb(role=ROLE_A)
        async def voter_a(ctx: Context[Any], iteration: int) -> int:
            """Vote; the first invocation at iteration 2 also fires halt."""
            runs.append((iteration, "a"))
            if iteration == 2:
                halt.set()
            return 1

        @verb(role=ROLE_A)
        async def voter_b(ctx: Context[Any], iteration: int) -> int:
            """Vote."""
            runs.append((iteration, "b"))
            return 1

        @verb(role=ROLE_A)
        async def voter_c(ctx: Context[Any], iteration: int) -> int:
            """Vote."""
            runs.append((iteration, "c"))
            return 1

        panel = Panel([voter_a, voter_b, voter_c], aggregate=sum)

        @verb(role=ROLE_A)
        async def body(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            """Increment the iteration counter, run the Panel with it, return the aggregate."""
            ctx.state.data["i"] += 1
            return await panel.run(ctx, ctx.state.data["i"])

        def build(with_halt: bool) -> Flow:
            # Voters must live on the top-level runtime flow — Panel dispatches
            # by name via ctx.flow, which resolves to the outer runtime, not
            # the iterate body's subflow.
            flow = make_ff().create(state={"i": 0}).with_checkpointer(store, "panel-iter-1")
            if with_halt:
                flow.with_halt(halt)
            for v in (voter_a, voter_b, voter_c):
                flow.register(v)
            flow.iterate(body, max_iters=5)
            return flow

        # First run: halt fires during iteration 2's Panel; iterate exits
        # at the next between-iterations check.
        await build(with_halt=True).run()

        iter1 = [r for r in runs if r[0] == 1]
        iter2 = [r for r in runs if r[0] == 2]
        past_2 = [r for r in runs if r[0] >= 3]
        assert sorted(iter1) == [(1, "a"), (1, "b"), (1, "c")]
        # Iteration 2 completes fully despite halt-set from inside voter_a.
        assert sorted(iter2) == [(2, "a"), (2, "b"), (2, "c")]
        assert not past_2, f"iterations past 2 must not run: {past_2}"

        runs.clear()

        # Resume: iterate continues at iteration 3 with a fresh Panel each pass.
        await build(with_halt=False).run(resume=True)

        seen_iters = sorted({r[0] for r in runs})
        assert seen_iters == [3, 4, 5], "iteration 2's Panel must not re-dispatch"
        for i in (3, 4, 5):
            assert sorted(r for r in runs if r[0] == i) == [(i, "a"), (i, "b"), (i, "c")]
