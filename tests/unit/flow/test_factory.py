"""Tests for llm_gent.flow.factory."""

from __future__ import annotations

from typing import Any

from llm_gent.flow import Flow, FlowFactory, Role, SAIAFactory

from .conftest import StubFactory, make_test_logger


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
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory())
        flow = ff.create("grade")
        assert isinstance(flow, Flow)
        assert flow.name == "grade"

    def test_create_default_name_is_empty(self) -> None:
        """.create() without a name yields an anonymous flow."""
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory())
        assert ff.create().name == ""

    def test_create_threads_saia_to_flow(self) -> None:
        """The saia captured on the factory is used by the built flow."""
        sf = StubFactory()
        ff = FlowFactory(make_test_logger(), saia_f=sf)
        flow = ff.create()
        # Reach into the private slot — this test's point is the wiring itself.
        assert flow._saia_f is sf

    def test_create_threads_state_default(self) -> None:
        """The factory's state default is used unless overridden on create()."""
        default_state = {"scope": "app"}
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), state=default_state)
        assert ff.create().state is default_state

    def test_create_state_override(self) -> None:
        """Passing state= on create() overrides the factory default for that Flow."""
        default_state = {"scope": "app"}
        per_flow = {"scope": "grade"}
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), state=default_state)
        assert ff.create(state=per_flow).state is per_flow

    def test_create_state_override_with_none(self) -> None:
        """state=None explicit is honored (distinct from the UNSET default)."""
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), state={"x": 1})
        flow = ff.create(state=None)
        assert flow.state is None

    def test_with_saia_returns_new_factory(self) -> None:
        """with_saia_f() derives a new FlowFactory (immutable-style swap)."""
        a, b = StubFactory(), StubFactory()
        ff = FlowFactory(make_test_logger(), saia_f=a)
        derived = ff.with_saia_f(b)
        assert derived is not ff
        assert derived.create()._saia_f is b
        # Original untouched.
        assert ff.create()._saia_f is a

    def test_with_saia_preserves_state(self) -> None:
        """with_saia_f() carries state and lg forward untouched."""
        state = {"scope": "shared"}
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), state=state)
        derived = ff.with_saia_f(StubFactory())
        assert derived.create().state is state
