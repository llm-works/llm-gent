"""Shared test fixtures for flow tests."""

from __future__ import annotations

from appinfra.log import Logger, quick_console_logger

from llm_gent.flow import Role


ROLE_A = Role(name="a", backend="openai", model="gpt-4o-mini")
ROLE_B = Role(name="b", backend="anthropic", model="claude-3-5")


def make_test_logger() -> Logger:
    """Return a logger for tests (suppressed output)."""
    return quick_console_logger("test", config={"level": "error"})


class StubSAIA:
    """Minimal saia stand-in — tests only need identity."""

    def __init__(self, role: Role) -> None:
        """Track which role this saia was built for."""
        self.role = role


class StubFactory:
    """SAIAFactory impl that records every build() call."""

    def __init__(self) -> None:
        """Initialize an empty build log."""
        self.built_for: list[Role] = []

    def build(self, role: Role) -> StubSAIA:
        """Record and return a fresh stub saia for the role."""
        self.built_for.append(role)
        return StubSAIA(role)
