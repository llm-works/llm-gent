# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for llm_gent.flow.factory."""

from __future__ import annotations

from typing import Any

import pytest

from llm_gent.core.budget import Tracker
from llm_gent.flow import Context, Flow, FlowFactory, Role, SAIAFactory, verb

from .conftest import ROLE_A, StubFactory, make_test_logger


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
    """FlowFactory captures lg + saia + state; .create builds Flows."""

    def test_create_returns_flow(self) -> None:
        """The value from .create() is a fresh :class:`Flow`."""
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory())
        flow = ff.create("grade")
        assert isinstance(flow, Flow)
        assert flow.name == "grade"

    def test_create_default_name_is_empty(self) -> None:
        """.create() without a name yields an anonymous flow."""
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory())
        assert ff.create().name == ""

    def test_create_threads_saia_to_flow(self) -> None:
        """The saia captured on the factory is used by the built flow."""
        sf = StubFactory()
        ff = FlowFactory(make_test_logger(), saia_factory=sf)
        flow = ff.create()
        # Reach into the private slot — this test's point is the wiring itself.
        assert flow._saia_factory is sf

    def test_create_threads_state_default(self) -> None:
        """The factory's state default is used unless overridden on create()."""
        default_state = {"scope": "app"}
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory(), state=default_state)
        assert ff.create().state is default_state

    def test_create_state_override(self) -> None:
        """Passing state= on create() overrides the factory default for that Flow."""
        default_state = {"scope": "app"}
        per_flow = {"scope": "grade"}
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory(), state=default_state)
        assert ff.create(state=per_flow).state is per_flow

    def test_create_state_override_with_none(self) -> None:
        """state=None explicit is honored (distinct from the UNSET default)."""
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory(), state={"x": 1})
        flow = ff.create(state=None)
        assert flow.state is None

    def test_with_saia_returns_new_factory(self) -> None:
        """with_saia_factory() derives a new FlowFactory (immutable-style swap)."""
        a, b = StubFactory(), StubFactory()
        ff = FlowFactory(make_test_logger(), saia_factory=a)
        derived = ff.with_saia_factory(b)
        assert derived is not ff
        assert derived.create()._saia_factory is b
        # Original untouched.
        assert ff.create()._saia_factory is a

    def test_with_saia_preserves_state(self) -> None:
        """with_saia_factory() carries state and lg forward untouched."""
        state = {"scope": "shared"}
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory(), state=state)
        derived = ff.with_saia_factory(StubFactory())
        assert derived.create().state is state


class TestCreateHaltAndCheckpointer:
    """FlowFactory.create() accepts halt= and checkpointer= for per-flow wiring."""

    def test_create_halt_kwarg_binds_flow_halt(self) -> None:
        """A ``halt=`` on create() supersedes the factory's captured halt."""
        import asyncio

        from llm_gent.flow import FlowFactory

        factory_halt = asyncio.Event()
        create_halt = asyncio.Event()
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory(), halt=factory_halt)
        flow = ff.create("named", halt=create_halt)
        assert flow._halt_event is create_halt

    def test_create_halt_kwarg_supplies_when_factory_has_none(self) -> None:
        """A ``halt=`` on create() applies even when the factory has none."""
        import asyncio

        from llm_gent.flow import FlowFactory

        create_halt = asyncio.Event()
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory())
        flow = ff.create(halt=create_halt)
        assert flow._halt_event is create_halt

    def test_create_halt_absent_inherits_factory(self) -> None:
        """Without ``halt=``, the factory's captured halt reaches the flow."""
        import asyncio

        from llm_gent.flow import FlowFactory

        factory_halt = asyncio.Event()
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory(), halt=factory_halt)
        flow = ff.create()
        assert flow._halt_event is factory_halt

    def test_create_checkpointer_pair_binds_flow(self, tmp_path: Any) -> None:
        """``checkpointer=(store, id)`` wires the pair via .with_checkpointer."""
        from llm_gent.flow import FlowFactory
        from llm_gent.flow.stores import JsonFileCheckpointStore

        store = JsonFileCheckpointStore(make_test_logger(), tmp_path)
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory())
        flow = ff.create(checkpointer=(store, "flow-1"))
        assert flow._checkpointer is store
        assert flow._client_flow_id == "flow-1"

    def test_create_checkpointer_supersedes_factory_pair(self, tmp_path: Any) -> None:
        """``checkpointer=`` wins over the factory-level store + client_flow_id."""
        from llm_gent.flow import FlowFactory
        from llm_gent.flow.stores import JsonFileCheckpointStore

        factory_store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "a")
        create_store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "b")
        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory(), checkpointer=factory_store)
        flow = ff.create(client_flow_id="ignored", checkpointer=(create_store, "flow-override"))
        assert flow._checkpointer is create_store
        assert flow._client_flow_id == "flow-override"


class TestPricingProviderSwap:
    """A stub PricingProvider injected via FlowFactory.with_budget reaches ctx.budget.track."""

    @pytest.mark.asyncio
    async def test_verb_sees_stub_pricing(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        class StubProvider:
            def compute(self, op_name: str, /, **usage: Any) -> float:
                calls.append((op_name, dict(usage)))
                return 0.42

        tracker = Tracker(make_test_logger(), StubProvider(), budget=10.0)
        recorded: dict[str, float] = {}

        @verb(role=ROLE_A)
        async def spend(ctx: Context) -> None:
            recorded["cost"] = ctx.budget.track("some-model", input_tokens=1000, output_tokens=100)

        ff = FlowFactory(make_test_logger(), saia_factory=StubFactory()).with_budget(tracker)
        await ff.create().call(spend).run()
        assert recorded["cost"] == pytest.approx(0.42)
        assert tracker.spent == pytest.approx(0.42)
        assert calls == [("some-model", {"input_tokens": 1000, "output_tokens": 100})]
