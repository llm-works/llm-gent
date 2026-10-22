# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow commits — CommitMeta as a commit payload, FlowCommit round-trip, what the id covers.

Hashing, object serialization and tree ordering are :mod:`appinfra.cas`'s
and tested there; these tests cover what gent puts in a commit.
"""

from __future__ import annotations

from appinfra.cas import Blob, Commit

from llm_gent.flow.state.cas import CommitMeta, FlowCommit, ProducedBy, TraceRef


ROOT = "a" * 64
"""A tree id: commits only need the form of one."""

PARENT = "b" * 64


def _meta(
    *,
    node_path: str = "n1",
    node_id: str = "n1",
    verb_name: str | None = "verb_a",
    trace_ref: tuple[TraceRef, ...] = (),
    timestamp_iso: str = "2026-09-22T00:00:00+00:00",
    framework_version: str = "0.4.0",
) -> CommitMeta:
    """Build a CommitMeta for tests without repeating every field."""
    return CommitMeta(
        node_path=node_path,
        iteration=1,
        produced_by=ProducedBy(
            node_id=node_id, verb_name=verb_name, role="role_a", result_hash=None
        ),
        trace_ref=trace_ref,
        outcome="ok",
        flow_root_hash="c" * 64,
        timestamp_iso=timestamp_iso,
        framework_version=framework_version,
        scope_path="n/x",
    )


def _id(meta: CommitMeta, parents: tuple[str, ...] = ()) -> str:
    return FlowCommit.build(ROOT, parents, meta)[0].id


class TestCommitMeta:
    def test_payload_round_trips(self) -> None:
        meta = _meta(trace_ref=(TraceRef("verb_trace", "vt-1"), TraceRef("otel_span", "s-1")))
        assert CommitMeta.from_payload(meta.to_payload()) == meta

    def test_payload_is_canonical(self) -> None:
        """Equal metadata gives equal bytes, so an equal commit id."""
        assert _meta().to_payload() == _meta().to_payload()

    def test_external_produced_by_round_trips(self) -> None:
        """Direct-save commits with a $external/* producer survive the round trip."""
        meta = _meta(node_id="$external/initial_plan", verb_name=None)
        back = CommitMeta.from_payload(meta.to_payload())
        assert back.produced_by.node_id == "$external/initial_plan"
        assert back.produced_by.verb_name is None


class TestFlowCommit:
    def test_build_returns_the_stored_commit(self) -> None:
        meta = _meta()
        flow_commit, commit = FlowCommit.build(ROOT, (PARENT,), meta)
        assert flow_commit.id == commit.id
        assert (commit.tree, commit.parents, commit.payload) == (ROOT, (PARENT,), meta.to_payload())

    def test_from_commit_parses_the_metadata(self) -> None:
        meta = _meta()
        flow_commit, commit = FlowCommit.build(ROOT, (PARENT,), meta)
        assert FlowCommit.from_commit(Commit.deserialize(commit.serialize())) == flow_commit

    def test_parents_change_the_id(self) -> None:
        assert _id(_meta()) != _id(_meta(), (PARENT,))

    def test_timestamp_changes_the_id(self) -> None:
        """Timepoint semantics: the same state saved at another time is another commit."""
        assert _id(_meta(timestamp_iso="2026-09-22T00:00:00+00:00")) != _id(
            _meta(timestamp_iso="2026-09-22T00:00:01+00:00")
        )

    def test_trace_ref_changes_the_id(self) -> None:
        assert _id(_meta()) != _id(_meta(trace_ref=(TraceRef("verb_trace", "vt-1"),)))

    def test_framework_version_changes_the_commit_not_the_blobs(self) -> None:
        """A version bump makes new commits identifiable; blob ids never see the version."""
        assert _id(_meta(framework_version="0.4.0")) != _id(_meta(framework_version="0.5.0"))
        assert Blob(b'{"count":1}').id == Blob(b'{"count":1}').id
