# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for Tracker: hierarchy, cap enforcement, halt integration."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from appinfra.log import quick_console_logger

from llm_gent.core.budget import FixedOp, LLMOp, PricingConfig, Tracker


pytestmark = pytest.mark.unit


def _lg() -> Any:
    return quick_console_logger("test", config={"level": "error"})


def _pricing() -> PricingConfig:
    return PricingConfig(
        ops={
            "some-model": LLMOp(name="some-model", input_per_mtok=1.0, output_per_mtok=5.0),
            "web_search": FixedOp(name="web_search", unit_cost=0.001),
        }
    )


class TestInitValidation:
    """budget must be > 0 when set; None means uncapped."""

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            Tracker(_lg(), _pricing(), budget=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            Tracker(_lg(), _pricing(), budget=-1)

    def test_uncapped_allowed(self) -> None:
        t = Tracker(_lg(), _pricing())
        assert t.budget is None
        assert t.exceeded is False
        assert t.remaining is None


class TestTrackDispatch:
    """.track() forwards kwargs to the Op's .cost() signature."""

    def test_llm_op_receives_token_kwargs(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        cost = t.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(1.0)
        assert t.spent == pytest.approx(1.0)

    def test_fixed_op_receives_count_kwarg(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        cost = t.track("web_search", count=3)
        assert cost == pytest.approx(0.003)

    def test_unknown_op_returns_zero(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        assert t.track("nothing-here") == 0.0
        assert t.spent == 0.0

    def test_provider_cost_wins_over_computed(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        cost = t.track(
            "some-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider_cost=0.42,
        )
        assert cost == pytest.approx(0.42)


class TestCap:
    """spent / remaining / exceeded on a capped tracker."""

    def test_starts_at_zero(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        assert t.spent == 0.0
        assert t.remaining == pytest.approx(1.0)
        assert t.exceeded is False
        assert t.urgent_wrapup is False

    def test_remaining_clamps_at_zero(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=0.001)
        t.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert t.remaining == 0.0
        assert t.exceeded is True

    def test_urgent_wrapup_latches_on_cross(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=0.001)
        assert t.urgent_wrapup is False
        t.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert t.urgent_wrapup is True

    def test_uncapped_never_exceeds(self) -> None:
        t = Tracker(_lg(), _pricing())
        t.track("some-model", input_tokens=1_000_000, output_tokens=1_000_000)
        assert t.exceeded is False
        assert t.urgent_wrapup is False


class TestPerOpAggregation:
    """costs_by_op accumulates per invocation and reports up."""

    def test_totals_and_per_op(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=10.0)
        t.track("some-model", input_tokens=1_000_000, output_tokens=0)
        t.track("web_search", count=5)
        assert t.spent == pytest.approx(1.005)
        assert t.costs_by_op == {
            "some-model": pytest.approx(1.0),
            "web_search": pytest.approx(0.005),
        }

    def test_costs_by_op_returns_copy(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        t.track("web_search", count=1)
        snapshot = t.costs_by_op
        snapshot["web_search"] = 999.0
        assert t.costs_by_op["web_search"] == pytest.approx(0.001)


class TestHierarchy:
    """Costs recorded at any level propagate up the parent chain."""

    def test_child_spent_reports_to_parent(self) -> None:
        root = Tracker(_lg(), _pricing(), budget=10.0)
        child = root.child(budget=1.0)
        child.track("web_search", count=5)
        assert child.spent == pytest.approx(0.005)
        assert root.spent == pytest.approx(0.005)

    def test_arbitrary_depth(self) -> None:
        root = Tracker(_lg(), _pricing(), budget=10.0)
        wave = root.child(budget=3.0)
        run = wave.child(budget=1.0)
        run.track("web_search", count=2)
        assert run.spent == pytest.approx(0.002)
        assert wave.spent == pytest.approx(0.002)
        assert root.spent == pytest.approx(0.002)

    def test_sibling_children_independent(self) -> None:
        root = Tracker(_lg(), _pricing(), budget=10.0)
        a = root.child(budget=1.0)
        b = root.child(budget=1.0)
        a.track("web_search", count=3)
        b.track("web_search", count=2)
        assert a.spent == pytest.approx(0.003)
        assert b.spent == pytest.approx(0.002)
        assert root.spent == pytest.approx(0.005)

    def test_parent_property(self) -> None:
        root = Tracker(_lg(), _pricing(), budget=1.0)
        child = root.child(budget=0.5)
        assert child.parent is root
        assert root.parent is None

    def test_child_inherits_pricing(self) -> None:
        pricing = _pricing()
        root = Tracker(_lg(), pricing, budget=1.0)
        child = root.child(budget=0.5)
        cost = child.track("web_search", count=1)
        assert cost == pytest.approx(0.001)

    def test_costs_by_op_aggregates_up(self) -> None:
        root = Tracker(_lg(), _pricing(), budget=10.0)
        a = root.child()
        b = root.child()
        a.track("web_search", count=1)
        b.track("web_search", count=2)
        assert root.costs_by_op == {"web_search": pytest.approx(0.003)}
        assert a.costs_by_op == {"web_search": pytest.approx(0.001)}
        assert b.costs_by_op == {"web_search": pytest.approx(0.002)}


class TestHaltIntegration:
    """Each level's halt event fires when its OWN cap crosses."""

    def test_root_halt_fires_on_root_cross(self) -> None:
        halt = asyncio.Event()
        root = Tracker(_lg(), _pricing(), budget=0.001, halt=halt)
        assert not halt.is_set()
        root.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert halt.is_set()

    def test_child_halt_fires_on_child_cross(self) -> None:
        root_halt = asyncio.Event()
        child_halt = asyncio.Event()
        root = Tracker(_lg(), _pricing(), budget=10.0, halt=root_halt)
        child = root.child(budget=0.001, halt=child_halt)
        child.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert child_halt.is_set()
        assert not root_halt.is_set()

    def test_root_halt_fires_when_descendant_pushes_root_over(self) -> None:
        root_halt = asyncio.Event()
        root = Tracker(_lg(), _pricing(), budget=0.001, halt=root_halt)
        child = root.child(budget=1.0)
        child.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert root_halt.is_set()

    def test_halt_only_set_on_transition(self) -> None:
        halt = asyncio.Event()
        t = Tracker(_lg(), _pricing(), budget=0.001, halt=halt)
        t.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert halt.is_set()
        halt.clear()
        t.track("web_search", count=1)
        assert not halt.is_set()

    def test_sibling_children_have_independent_halts(self) -> None:
        halt_a = asyncio.Event()
        halt_b = asyncio.Event()
        root = Tracker(_lg(), _pricing(), budget=10.0)
        a = root.child(budget=0.001, halt=halt_a)
        root.child(budget=0.001, halt=halt_b)
        a.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert halt_a.is_set()
        assert not halt_b.is_set()


class TestCallback:
    """on_cost fires at each level whose scope saw the cost."""

    def test_callback_receives_cost_and_context(self) -> None:
        events: list[tuple[float, dict[str, Any]]] = []

        def cb(cost: float, ctx: dict[str, Any]) -> None:
            events.append((cost, ctx))

        t = Tracker(_lg(), _pricing(), budget=1.0, on_cost=cb)
        t.track("web_search", count=2, context={"phase": "x"})
        assert len(events) == 1
        cost, ctx = events[0]
        assert cost == pytest.approx(0.002)
        assert ctx == {"phase": "x"}

    def test_context_passed_through_unchanged(self) -> None:
        events: list[dict[str, Any]] = []

        def cb(cost: float, ctx: dict[str, Any]) -> None:
            events.append(ctx)

        t = Tracker(_lg(), _pricing(), budget=1.0, on_cost=cb)
        t.track("web_search", count=1, context={"op": "outer", "phase": "x"})
        assert events[0] == {"op": "outer", "phase": "x"}

    def test_callback_fires_at_every_ancestor(self) -> None:
        seen: list[str] = []

        def make_cb(label: str):
            def cb(cost: float, ctx: dict[str, Any]) -> None:
                seen.append(label)

            return cb

        root = Tracker(_lg(), _pricing(), budget=10.0, on_cost=make_cb("root"))
        wave = root.child(budget=1.0, on_cost=make_cb("wave"))
        run = wave.child(budget=0.1, on_cost=make_cb("run"))
        run.track("web_search", count=1)
        assert seen == ["root", "wave", "run"]

    def test_halt_fires_before_callback(self) -> None:
        halt = asyncio.Event()
        observed: list[bool] = []

        def cb(cost: float, ctx: dict[str, Any]) -> None:
            observed.append(halt.is_set())

        t = Tracker(_lg(), _pricing(), budget=0.001, on_cost=cb, halt=halt)
        t.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert observed == [True]

    def test_halt_set_when_callback_raises(self) -> None:
        halt = asyncio.Event()

        def bad_cb(cost: float, ctx: dict[str, Any]) -> None:
            raise RuntimeError("intentional")

        t = Tracker(_lg(), _pricing(), budget=0.001, on_cost=bad_cb, halt=halt)
        with pytest.raises(RuntimeError):
            t.track("some-model", input_tokens=1_000_000, output_tokens=0)
        assert halt.is_set()

    def test_all_accounting_completes_before_callbacks(self) -> None:
        """A raising callback leaves the whole tree's spend intact."""

        def bad_cb(cost: float, ctx: dict[str, Any]) -> None:
            raise RuntimeError("intentional")

        root = Tracker(_lg(), _pricing(), budget=10.0, on_cost=bad_cb)
        child = root.child(budget=1.0)
        with pytest.raises(RuntimeError):
            child.track("web_search", count=5)
        assert child.spent == pytest.approx(0.005)
        assert root.spent == pytest.approx(0.005)


class TestUpdateBudget:
    """update_budget replaces the cap."""

    def test_amend(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        t.update_budget(2.0)
        assert t.budget == pytest.approx(2.0)

    def test_zero_raises(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        with pytest.raises(ValueError):
            t.update_budget(0)

    def test_amend_below_spent_makes_exceeded(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=10.0)
        t.track("web_search", count=10)
        t.update_budget(0.005)
        assert t.exceeded is True


class TestRestoreSpent:
    """restore_spent is this-level-only and unconditional."""

    def test_restore(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        t.restore_spent(0.75)
        assert t.spent == pytest.approx(0.75)
        assert t.remaining == pytest.approx(0.25)

    def test_overwrites_live_spend(self) -> None:
        t = Tracker(_lg(), _pricing(), budget=1.0)
        t.track("web_search", count=5)
        t.restore_spent(0.5)
        assert t.spent == pytest.approx(0.5)

    def test_does_not_walk_up(self) -> None:
        root = Tracker(_lg(), _pricing(), budget=10.0)
        child = root.child(budget=1.0)
        child.track("web_search", count=1)
        child.restore_spent(0.5)
        assert child.spent == pytest.approx(0.5)
        assert root.spent == pytest.approx(0.001)
