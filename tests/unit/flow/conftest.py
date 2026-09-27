# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shared test fixtures for flow tests."""

from __future__ import annotations

import uuid
from typing import Any

from appinfra.log import Logger, quick_console_logger

from llm_gent.flow import FlowFactory, Role, SAIAFactory


ROLE_A = Role(name="a", backend="openai", model="gpt-4o-mini")
ROLE_B = Role(name="b", backend="anthropic", model="claude-3-5")


def make_test_logger() -> Logger:
    """Return a logger for tests (suppressed output)."""
    return quick_console_logger("test", config={"level": "error"})


def flow_id_for(store: Any, client_flow_id: str) -> str:
    """Return the ``flow_id`` bound to ``client_flow_id`` in a sync store.

    Binds a fresh one when the name is unbound, so tests can also seed
    objects / refs under a name before any flow ran. Stores key every
    object and ref by ``flow_id``; tests address histories by name.
    """
    flow_id: str = store.bind_flow_id(client_flow_id, str(uuid.uuid4()))
    return flow_id


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


def make_ff(saia_factory: SAIAFactory | None = None) -> FlowFactory:
    """Return a fresh :class:`FlowFactory` with a test logger + SAIAFactory.

    Defaults to a fresh :class:`StubFactory` per call so tests that
    introspect the factory get an isolated instance. Pass ``saia_factory=``
    when the test needs to keep a reference to inspect after the run.
    """
    return FlowFactory(make_test_logger(), saia_factory=saia_factory or StubFactory())
