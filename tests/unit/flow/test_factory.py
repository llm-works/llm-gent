"""Tests for llm_gent.flow.factory."""

from __future__ import annotations

from typing import Any

from llm_gent.flow import Role, SAIAFactory


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
