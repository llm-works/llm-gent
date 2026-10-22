# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shared test fixtures for flow tests."""

from __future__ import annotations

from appinfra.cas import Commit, Store

from llm_gent.flow import Factory, FlowCommit, Role, SAIAFactory
from llm_gent.flow.checkpoint import HEAD_REF
from tests.helpers import make_test_logger


ROLE_A = Role(name="a", backend="openai", model="gpt-4o-mini")
ROLE_B = Role(name="b", backend="anthropic", model="claude-3-5")


def read_commit(store: Store, client_flow_id: str, commit_id: str) -> FlowCommit:
    """The history's commit ``commit_id``, with its metadata parsed; it must exist."""
    commit = store.repo(client_flow_id).get(commit_id, Commit)
    assert commit is not None, f"{client_flow_id}: no commit {commit_id}"
    return FlowCommit.from_commit(commit)


def head_commit(store: Store, client_flow_id: str) -> FlowCommit:
    """The commit the history's ``HEAD`` points at; the history must have one."""
    head = store.repo(client_flow_id).get_ref(HEAD_REF)
    assert head is not None, f"{client_flow_id}: no HEAD"
    return read_commit(store, client_flow_id, head)


def commit_ids(store: Store, client_flow_id: str) -> set[str]:
    """Ids of every commit the history's repository holds, on its line or off it."""
    return {
        oid
        for oid in store.list_objects(client_flow_id)
        if store.object_type(client_flow_id, oid) == "commit"
    }


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


def make_ff(saia_factory: SAIAFactory | None = None) -> Factory:
    """Return a fresh :class:`Factory` with a test logger + SAIAFactory.

    Defaults to a fresh :class:`StubFactory` per call so tests that
    introspect the factory get an isolated instance. Pass ``saia_factory=``
    when the test needs to keep a reference to inspect after the run.
    """
    return Factory(make_test_logger(), saia_factory=saia_factory or StubFactory())
