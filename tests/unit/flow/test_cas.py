# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CAS object model — hash determinism, tree ordering, commit round-trip.

Covers Blob / Tree / TreeEntry / ProducedBy / TraceRef / CommitMeta / Commit
shapes, blake2b content hashing with canonical serialization, sorted tree
entries, ``$external/*`` produced_by round-trip, hash stability of unchanged
bytes across framework_version bumps.

The framework_version test is the load-bearing invariant: existing blobs
must NOT be invalidated when the framework's canonicalization moves
forward. Old commits stay resolvable under their own recorded version.
"""

from __future__ import annotations

import pytest

from llm_gent.flow.state.cas import (
    Blob,
    Commit,
    CommitMeta,
    ProducedBy,
    TraceRef,
    Tree,
    TreeEntry,
    canonical_json,
    content_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(
    *,
    client_flow_id: str = "flow-1",
    node_path: str = "n1",
    iteration: int = 1,
    node_id: str = "n1",
    verb_name: str | None = "verb_a",
    role: str | None = "role_a",
    result_hash: str | None = None,
    trace_ref: tuple[TraceRef, ...] = (),
    outcome: str = "ok",
    flow_root_id: str = "root-hash-xx",
    timestamp_iso: str = "2026-09-22T00:00:00+00:00",
    framework_version: str = "0.4.0",
) -> CommitMeta:
    """Build a CommitMeta for tests without repeating every field."""
    return CommitMeta(
        client_flow_id=client_flow_id,
        node_path=node_path,
        iteration=iteration,
        produced_by=ProducedBy(
            node_id=node_id,
            verb_name=verb_name,
            role=role,
            result_hash=result_hash,
        ),
        trace_ref=trace_ref,
        outcome=outcome,  # type: ignore[arg-type]
        flow_root_id=flow_root_id,
        timestamp_iso=timestamp_iso,
        framework_version=framework_version,
    )


# ---------------------------------------------------------------------------
# content_hash + canonical_json
# ---------------------------------------------------------------------------


class TestContentHash:
    """blake2b hex digest — deterministic, content-only, 32-byte."""

    def test_same_bytes_same_hash(self) -> None:
        assert content_hash(b"payload-A") == content_hash(b"payload-A")

    def test_different_bytes_different_hash(self) -> None:
        assert content_hash(b"payload-A") != content_hash(b"payload-B")

    def test_hash_length_is_64_hex_chars(self) -> None:
        # blake2b digest_size=32 → 64 hex chars
        assert len(content_hash(b"x")) == 64
        assert all(c in "0123456789abcdef" for c in content_hash(b"x"))


class TestCanonicalJson:
    """Deterministic ordering for equal-shaped inputs."""

    def test_dict_key_order_does_not_affect_bytes(self) -> None:
        a = canonical_json({"a": 1, "b": 2})
        b = canonical_json({"b": 2, "a": 1})
        assert a == b

    def test_nested_dict_stable(self) -> None:
        a = canonical_json({"outer": {"a": 1, "b": 2}})
        b = canonical_json({"outer": {"b": 2, "a": 1}})
        assert a == b

    def test_compact_separators(self) -> None:
        raw = canonical_json({"a": 1, "b": 2}).decode("utf-8")
        # sort_keys + compact separators → no whitespace between tokens
        assert raw == '{"a":1,"b":2}'


# ---------------------------------------------------------------------------
# Blob
# ---------------------------------------------------------------------------


class TestBlob:
    def test_from_bytes_hashes_payload(self) -> None:
        b = Blob.from_bytes(b"hello")
        assert b.content_hash == content_hash(b"hello")
        assert b.payload == b"hello"

    def test_same_bytes_same_blob_hash(self) -> None:
        assert Blob.from_bytes(b"x").content_hash == Blob.from_bytes(b"x").content_hash

    def test_different_bytes_different_blob_hash(self) -> None:
        assert Blob.from_bytes(b"x").content_hash != Blob.from_bytes(b"y").content_hash


# ---------------------------------------------------------------------------
# Tree — sorted entries + canonical hash
# ---------------------------------------------------------------------------


class TestTree:
    def test_entries_sorted_by_scope_id(self) -> None:
        entries = [
            TreeEntry(scope_id="b", kind="blob", child_hash="hb"),
            TreeEntry(scope_id="a", kind="blob", child_hash="ha"),
            TreeEntry(scope_id="c", kind="tree", child_hash="hc"),
        ]
        tree = Tree.from_entries(entries)
        assert [e.scope_id for e in tree.entries] == ["a", "b", "c"]

    def test_same_entries_different_input_order_same_hash(self) -> None:
        e_ab = [
            TreeEntry(scope_id="a", kind="blob", child_hash="ha"),
            TreeEntry(scope_id="b", kind="tree", child_hash="hb"),
        ]
        e_ba = list(reversed(e_ab))
        assert Tree.from_entries(e_ab).content_hash == Tree.from_entries(e_ba).content_hash

    def test_different_entries_different_hash(self) -> None:
        t1 = Tree.from_entries([TreeEntry("a", "blob", "ha")])
        t2 = Tree.from_entries([TreeEntry("a", "blob", "hb")])
        assert t1.content_hash != t2.content_hash

    def test_different_kind_different_hash(self) -> None:
        t_blob = Tree.from_entries([TreeEntry("a", "blob", "h")])
        t_tree = Tree.from_entries([TreeEntry("a", "tree", "h")])
        assert t_blob.content_hash != t_tree.content_hash

    def test_duplicate_scope_id_rejected(self) -> None:
        import pytest

        entries = [
            TreeEntry(scope_id="a", kind="blob", child_hash="h1"),
            TreeEntry(scope_id="a", kind="blob", child_hash="h2"),
        ]
        with pytest.raises(ValueError, match="unique scope_id"):
            Tree.from_entries(entries)


# ---------------------------------------------------------------------------
# Commit — build + hash + $external/ round-trip + timepoint semantics
# ---------------------------------------------------------------------------


class TestCommit:
    def test_build_sets_content_hash_from_body(self) -> None:
        meta = _meta()
        c = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=meta)
        assert c.root_tree_hash == "rt"
        assert c.parent_hashes == ()
        assert c.meta is meta
        # Hash is populated (non-empty, correct length)
        assert len(c.content_hash) == 64

    def test_identical_inputs_same_commit_hash(self) -> None:
        meta = _meta()
        c1 = Commit.build(root_tree_hash="rt", parent_hashes=("p",), meta=meta)
        c2 = Commit.build(root_tree_hash="rt", parent_hashes=("p",), meta=meta)
        assert c1.content_hash == c2.content_hash

    def test_different_root_tree_different_commit_hash(self) -> None:
        meta = _meta()
        c1 = Commit.build(root_tree_hash="rt-1", parent_hashes=(), meta=meta)
        c2 = Commit.build(root_tree_hash="rt-2", parent_hashes=(), meta=meta)
        assert c1.content_hash != c2.content_hash

    def test_different_parents_different_commit_hash(self) -> None:
        meta = _meta()
        c_none = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=meta)
        c_p = Commit.build(root_tree_hash="rt", parent_hashes=("p",), meta=meta)
        assert c_none.content_hash != c_p.content_hash

    def test_different_timestamp_different_commit_hash(self) -> None:
        """Timepoint semantics: same state at different times → different commits."""
        m1 = _meta(timestamp_iso="2026-09-22T00:00:00+00:00")
        m2 = _meta(timestamp_iso="2026-09-22T00:00:01+00:00")
        c1 = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m1)
        c2 = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m2)
        assert c1.content_hash != c2.content_hash

    def test_external_produced_by_round_trips(self) -> None:
        """Direct-save commits with $external/* prefix survive build round-trip."""
        meta = _meta(node_id="$external/initial_plan", verb_name=None, result_hash=None)
        c = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=meta)
        assert c.meta.produced_by.node_id == "$external/initial_plan"
        assert c.meta.produced_by.verb_name is None
        # Hash is stable — rebuild produces the same content_hash.
        c2 = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=meta)
        assert c.content_hash == c2.content_hash

    def test_trace_ref_participates_in_hash(self) -> None:
        m_empty = _meta(trace_ref=())
        m_verb = _meta(trace_ref=(TraceRef(kind="verb_trace", id="vt-1"),))
        c_empty = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m_empty)
        c_verb = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m_verb)
        assert c_empty.content_hash != c_verb.content_hash

    def test_trace_ref_multiple_kinds_participate(self) -> None:
        m_a = _meta(trace_ref=(TraceRef(kind="verb_trace", id="vt-1"),))
        m_b = _meta(
            trace_ref=(
                TraceRef(kind="verb_trace", id="vt-1"),
                TraceRef(kind="otel_span", id="span-1"),
            )
        )
        c_a = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m_a)
        c_b = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m_b)
        assert c_a.content_hash != c_b.content_hash


# ---------------------------------------------------------------------------
# framework_version bump invariant
# ---------------------------------------------------------------------------


class TestFrameworkVersionInvariant:
    """A framework_version bump MUST NOT rehash existing blobs.

    Blobs are hashed over their raw bytes only — no framework_version
    salt. Commits (which do carry framework_version in meta) get a new
    hash under a new version, but that's expected: a commit is
    version-stamped. The load-bearing invariant is that a blob's identity
    is stable across framework revisions.
    """

    def test_blob_hash_stable_across_version_change(self) -> None:
        # A "framework version bump" is just: the framework starts stamping
        # a different value in Commit.meta.framework_version. Blobs are
        # produced from raw bytes and never see framework_version at all.
        payload = b'{"count":1}'
        h_before = Blob.from_bytes(payload).content_hash
        # ... framework_version moves from "0.4.0" to "0.5.0" ...
        h_after = Blob.from_bytes(payload).content_hash
        assert h_before == h_after

    def test_commit_hash_differs_across_framework_version(self) -> None:
        """framework_version is in Commit.meta — bump → different commit hash.

        This is the correct direction: new commits under a new framework
        version are identifiable as such. The invariant that matters is
        that OLD commits under OLD version stay resolvable — asserted
        indirectly via :meth:`test_blob_hash_stable_across_version_change`
        (blob identity survives) plus the fact that Commit is a plain
        frozen dataclass so an already-built old commit's content_hash
        is untouched.
        """
        m_v1 = _meta(framework_version="0.4.0")
        m_v2 = _meta(framework_version="0.5.0")
        c_v1 = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m_v1)
        c_v2 = Commit.build(root_tree_hash="rt", parent_hashes=(), meta=m_v2)
        assert c_v1.content_hash != c_v2.content_hash


# ---------------------------------------------------------------------------
# In-memory object store fixture
# ---------------------------------------------------------------------------


class InMemoryObjectStore:
    """Dict-backed :class:`CheckpointStore` for testing.

    Implements the full Protocol surface — object store (put/get/has),
    ref store (put_ref / resolve_ref), gc_trajectory, and the retention
    attribute — so tests can wire it in place of a real store without
    touching the filesystem.

    Objects and refs are trajectory-scoped by ``client_flow_id`` under
    the same keyspace policy as the reference stores. ``resolve_ref``
    with both filters ``None`` returns the latest ref by insertion order
    (matches Postgres's ``ORDER BY created_at DESC LIMIT 1`` semantics).
    """

    def __init__(self, retention: str = "retain") -> None:
        self.retention = retention
        self._objects: dict[tuple[str, str, str], bytes] = {}
        self._refs: dict[tuple[str, str, int], str] = {}
        self._ref_order: list[tuple[str, str, int]] = []

    def put_object(self, client_flow_id: str, kind: str, content_hash: str, blob: bytes) -> None:
        # Idempotent: same hash MUST have same bytes.
        existing = self._objects.get((client_flow_id, kind, content_hash))
        if existing is not None and existing != blob:
            raise ValueError(
                f"hash collision for ({client_flow_id}, {kind}, {content_hash}) — differing bytes"
            )
        self._objects[(client_flow_id, kind, content_hash)] = blob

    def get_object(self, client_flow_id: str, kind: str, content_hash: str) -> bytes | None:
        return self._objects.get((client_flow_id, kind, content_hash))

    def has_object(self, client_flow_id: str, kind: str, content_hash: str) -> bool:
        return (client_flow_id, kind, content_hash) in self._objects

    def put_ref(
        self, client_flow_id: str, node_path: str, iteration: int, commit_hash: str
    ) -> None:
        key = (client_flow_id, node_path, iteration)
        self._refs[key] = commit_hash
        if key in self._ref_order:
            self._ref_order.remove(key)
        self._ref_order.append(key)

    def resolve_ref(
        self,
        client_flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None:
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        if node_path is None:
            for key in reversed(self._ref_order):
                if key[0] == client_flow_id:
                    return self._refs[key]
            return None
        if iteration is not None:
            return self._refs.get((client_flow_id, node_path, iteration))
        best_iter = -1
        best_hash: str | None = None
        for (c, n, i), h in self._refs.items():
            if c == client_flow_id and n == node_path and i > best_iter:
                best_iter = i
                best_hash = h
        return best_hash

    def gc_trajectory(self, client_flow_id: str) -> None:
        self._objects = {k: v for k, v in self._objects.items() if k[0] != client_flow_id}
        self._refs = {k: v for k, v in self._refs.items() if k[0] != client_flow_id}
        self._ref_order = [k for k in self._ref_order if k[0] != client_flow_id]


class TestInMemoryObjectStore:
    """Object surface — trajectory-scoped keyspace + idempotency + dedup."""

    def test_put_get_round_trip(self) -> None:
        store = InMemoryObjectStore()
        blob = Blob.from_bytes(b"payload")
        store.put_object("traj-1", "blob", blob.content_hash, blob.payload)
        assert store.get_object("traj-1", "blob", blob.content_hash) == b"payload"

    def test_has_object_reflects_presence(self) -> None:
        store = InMemoryObjectStore()
        assert not store.has_object("traj-1", "blob", "missing")
        blob = Blob.from_bytes(b"x")
        store.put_object("traj-1", "blob", blob.content_hash, blob.payload)
        assert store.has_object("traj-1", "blob", blob.content_hash)

    def test_put_idempotent_same_bytes(self) -> None:
        store = InMemoryObjectStore()
        blob = Blob.from_bytes(b"x")
        store.put_object("traj-1", "blob", blob.content_hash, blob.payload)
        store.put_object("traj-1", "blob", blob.content_hash, blob.payload)  # no raise
        assert store.get_object("traj-1", "blob", blob.content_hash) == b"x"

    def test_kind_scopes_the_keyspace(self) -> None:
        """Same hash under different kind slots does not collide."""
        store = InMemoryObjectStore()
        store.put_object("traj-1", "blob", "h", b"blob-bytes")
        store.put_object("traj-1", "tree", "h", b"tree-bytes")
        assert store.get_object("traj-1", "blob", "h") == b"blob-bytes"
        assert store.get_object("traj-1", "tree", "h") == b"tree-bytes"

    def test_client_flow_id_scopes_the_keyspace(self) -> None:
        """Same (kind, hash) under two trajectories store separately."""
        store = InMemoryObjectStore()
        store.put_object("traj-a", "blob", "h", b"a-bytes")
        store.put_object("traj-b", "blob", "h", b"b-bytes")
        assert store.get_object("traj-a", "blob", "h") == b"a-bytes"
        assert store.get_object("traj-b", "blob", "h") == b"b-bytes"


class TestInMemoryRefStore:
    """Ref surface — exact / by-node_path / latest-across-trajectory lookups."""

    def test_put_resolve_exact_key(self) -> None:
        store = InMemoryObjectStore()
        store.put_ref("traj-1", "node/x", 5, "commit-hash-5")
        assert store.resolve_ref("traj-1", "node/x", 5) == "commit-hash-5"

    def test_resolve_none_when_absent(self) -> None:
        store = InMemoryObjectStore()
        assert store.resolve_ref("traj-1") is None
        assert store.resolve_ref("traj-1", "node/x", 5) is None

    def test_resolve_latest_across_trajectory(self) -> None:
        """Both filters None → newest ref by insertion order."""
        store = InMemoryObjectStore()
        store.put_ref("traj-1", "node/a", 1, "hash-1")
        store.put_ref("traj-1", "node/b", 1, "hash-2")
        assert store.resolve_ref("traj-1") == "hash-2"

    def test_resolve_latest_under_node_path(self) -> None:
        """node_path only → highest iteration under that path."""
        store = InMemoryObjectStore()
        store.put_ref("traj-1", "node/x", 1, "hash-1")
        store.put_ref("traj-1", "node/x", 3, "hash-3")
        store.put_ref("traj-1", "node/x", 2, "hash-2")
        assert store.resolve_ref("traj-1", "node/x") == "hash-3"

    def test_resolve_iteration_without_node_path_raises(self) -> None:
        store = InMemoryObjectStore()
        with pytest.raises(ValueError, match="iteration requires node_path"):
            store.resolve_ref("traj-1", None, 5)

    def test_put_ref_overwrites_same_key(self) -> None:
        store = InMemoryObjectStore()
        store.put_ref("traj-1", "node/x", 5, "hash-first")
        store.put_ref("traj-1", "node/x", 5, "hash-second")
        assert store.resolve_ref("traj-1", "node/x", 5) == "hash-second"


class TestInMemoryGcTrajectory:
    def test_removes_all_objects_and_refs(self) -> None:
        store = InMemoryObjectStore()
        store.put_object("traj-1", "blob", "h1", b"payload")
        store.put_ref("traj-1", "node/x", 1, "commit-h")
        store.gc_trajectory("traj-1")
        assert store.get_object("traj-1", "blob", "h1") is None
        assert store.resolve_ref("traj-1", "node/x", 1) is None

    def test_leaves_other_trajectories_intact(self) -> None:
        store = InMemoryObjectStore()
        store.put_object("traj-a", "blob", "h", b"a-bytes")
        store.put_object("traj-b", "blob", "h", b"b-bytes")
        store.gc_trajectory("traj-a")
        assert store.get_object("traj-a", "blob", "h") is None
        assert store.get_object("traj-b", "blob", "h") == b"b-bytes"


class TestInMemoryRetention:
    def test_default_is_retain(self) -> None:
        assert InMemoryObjectStore().retention == "retain"

    def test_explicit_gc_on_success(self) -> None:
        assert InMemoryObjectStore(retention="gc_on_success").retention == "gc_on_success"
