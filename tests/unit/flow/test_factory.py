# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for llm_gent.flow.factory."""

from __future__ import annotations

from typing import Any

import pytest

from llm_gent.core import Role, SAIAFactory
from llm_gent.core.cost import Tracker as CostTracker
from llm_gent.flow import Context, Factory, Flow, FlowRunner, TypeStateFactory, verb
from tests.helpers import make_platform, memory_store, run_flow

from .conftest import ROLE_A, make_test_logger


class _StubSAIA:
    """Minimal saia stand-in — a factory just needs to return something."""

    def __init__(self, role: Role) -> None:
        """Track the role the factory bound."""
        self.role = role


class _StubFactory:
    """Test SAIAFactory impl that echoes the role back inside a stub saia."""

    def __init__(self) -> None:
        """Record every build() call."""
        self.built_for: list[Role] = []

    def build(self, role: Role) -> Any:
        """Return a stub saia and record the role."""
        self.built_for.append(role)
        return _StubSAIA(role)


class TestSAIAFactoryProtocol:
    """SAIAFactory is a structural Protocol — any conforming class satisfies it."""

    def test_stub_conforms_structurally(self) -> None:
        """A class with a matching build() satisfies the SAIAFactory protocol."""
        factory: SAIAFactory = _StubFactory()
        role = Role(name="x", backend="y", model="z")
        saia = factory.build(role)
        assert isinstance(saia, _StubSAIA)
        assert saia.role is role

    def test_factory_called_per_role(self) -> None:
        """Each build(role) call is independent — factory sees every request."""
        factory = _StubFactory()
        a = Role(name="a", backend="openai", model="gpt-4o-mini")
        b = Role(name="b", backend="anthropic", model="claude-3-5")
        factory.build(a)
        factory.build(b)
        factory.build(a)
        assert factory.built_for == [a, b, a]


class TestFlowFactory:
    """Factory captures lg + state; .create builds Flows."""

    def test_create_returns_flow(self) -> None:
        """The value from .create() is a fresh :class:`Flow`."""
        ff = Factory(make_test_logger())
        flow = ff.create("grade")
        assert isinstance(flow, Flow)
        assert flow.name == "grade"

    def test_create_default_name_is_empty(self) -> None:
        """.create() without a name yields an anonymous flow."""
        assert Factory(make_test_logger()).create().name == ""

    def test_with_state_factory_returns_new_factory(self) -> None:
        """with_state_factory() derives a new Factory; the original keeps its own."""
        state_factory = TypeStateFactory(dict)
        ff = Factory(make_test_logger())
        derived = ff.with_state_factory(state_factory)
        assert derived is not ff
        assert derived.create()._state_factory is state_factory
        assert ff.create()._state_factory is None


class TestRunnerStore:
    """Where runs keep their histories is the runner's store, not the flow's."""

    @pytest.mark.asyncio
    async def test_a_run_s_history_goes_to_the_runner_s_store(self) -> None:
        @verb
        async def noop(ctx: Context[Any]) -> None:
            return None

        store = memory_store()
        runner = FlowRunner(
            make_test_logger(), platform=make_platform(make_test_logger()), store=store
        )
        assert runner.store is store
        await runner.start(Factory(make_test_logger()).create("named").call(noop), history="h")
        assert store.list_repos() == ["h"]

    @pytest.mark.asyncio
    async def test_a_history_on_a_runner_without_a_store_raises_at_start(self) -> None:
        @verb
        async def noop(ctx: Context[Any]) -> None:
            return None

        flow = Factory(make_test_logger()).create("plain").call(noop)
        with pytest.raises(RuntimeError, match="history='h' but the runner has no store"):
            FlowRunner(make_test_logger(), platform=make_platform(make_test_logger())).start(
                flow, history="h"
            )


class TestPricingProviderSwap:
    """A stub PricingProvider on the run's tracker reaches ctx.cost.track."""

    @pytest.mark.asyncio
    async def test_verb_sees_stub_pricing(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        class StubProvider:
            def compute(self, op_name: str, /, **usage: Any) -> float:
                calls.append((op_name, dict(usage)))
                return 0.42

        tracker = CostTracker(make_test_logger(), StubProvider(), budget=10.0)
        recorded: dict[str, float] = {}

        @verb(role=ROLE_A)
        async def spend(ctx: Context) -> None:
            recorded["cost"] = ctx.cost.track("some-model", input_tokens=1000, output_tokens=100)

        await run_flow(Factory(make_test_logger()).create().call(spend), cost=tracker)
        assert recorded["cost"] == pytest.approx(0.42)
        assert tracker.spent == pytest.approx(0.42)
        assert calls == [("some-model", {"input_tokens": 1000, "output_tokens": 100})]
