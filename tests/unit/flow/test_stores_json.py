# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.flow.stores.JsonFileCheckpointStore`.

Direct surface tests: object put/get/has round-trip, ref put/resolve
across the three key forms, gc_trajectory idempotency, path-traversal
guards, retention default. End-to-end resume behavior is covered in
:mod:`tests.unit.flow.test_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A fresh store under an isolated temp root."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "checkpoints")


# ---------------------------------------------------------------------------
# Object store surface
# ---------------------------------------------------------------------------


class TestObjectStore:
    def test_put_get_round_trip(self, store: JsonFileCheckpointStore) -> None:
        store.put_object("traj-1", "blob", "hash-a", b"payload-a")
        assert store.get_object("traj-1", "blob", "hash-a") == b"payload-a"

    def test_get_returns_none_when_absent(self, store: JsonFileCheckpointStore) -> None:
        assert store.get_object("traj-1", "blob", "missing") is None

    def test_has_object_reflects_presence(self, store: JsonFileCheckpointStore) -> None:
        assert not store.has_object("traj-1", "blob", "hash-a")
        store.put_object("traj-1", "blob", "hash-a", b"payload")
        assert store.has_object("traj-1", "blob", "hash-a")

    def test_put_idempotent_same_hash(self, store: JsonFileCheckpointStore) -> None:
        store.put_object("traj-1", "blob", "hash-a", b"payload")
        store.put_object("traj-1", "blob", "hash-a", b"payload")  # no raise
        assert store.get_object("traj-1", "blob", "hash-a") == b"payload"

    def test_kinds_do_not_collide(self, store: JsonFileCheckpointStore) -> None:
        """Same hash under different kinds addresses different objects."""
        store.put_object("traj-1", "blob", "hash", b"blob-bytes")
        store.put_object("traj-1", "tree", "hash", b"tree-bytes")
        store.put_object("traj-1", "commit", "hash", b"commit-bytes")
        assert store.get_object("traj-1", "blob", "hash") == b"blob-bytes"
        assert store.get_object("traj-1", "tree", "hash") == b"tree-bytes"
        assert store.get_object("traj-1", "commit", "hash") == b"commit-bytes"

    def test_trajectories_do_not_collide(self, store: JsonFileCheckpointStore) -> None:
        """Same (kind, hash) under two trajectories store separately."""
        store.put_object("traj-a", "blob", "hash", b"a-bytes")
        store.put_object("traj-b", "blob", "hash", b"b-bytes")
        assert store.get_object("traj-a", "blob", "hash") == b"a-bytes"
        assert store.get_object("traj-b", "blob", "hash") == b"b-bytes"


# ---------------------------------------------------------------------------
# Ref store surface
# ---------------------------------------------------------------------------


class TestRefStore:
    def test_put_resolve_exact_key(self, store: JsonFileCheckpointStore) -> None:
        store.put_ref("traj-1", "node/x", 5, "commit-hash-5")
        assert store.resolve_ref("traj-1", "node/x", 5) == "commit-hash-5"

    def test_resolve_returns_none_when_absent(self, store: JsonFileCheckpointStore) -> None:
        assert store.resolve_ref("traj-1") is None
        assert store.resolve_ref("traj-1", "node/x") is None
        assert store.resolve_ref("traj-1", "node/x", 5) is None

    def test_resolve_latest_across_node_paths(self, store: JsonFileCheckpointStore) -> None:
        """resolve_ref with both None returns the newest ref across the trajectory."""
        store.put_ref("traj-1", "node/a", 1, "hash-1")
        store.put_ref("traj-1", "node/b", 1, "hash-2")
        # hash-2 was put last → latest.
        assert store.resolve_ref("traj-1") == "hash-2"

    def test_resolve_latest_under_node_path(self, store: JsonFileCheckpointStore) -> None:
        """resolve_ref with node_path returns highest iteration under that path."""
        store.put_ref("traj-1", "node/x", 1, "hash-1")
        store.put_ref("traj-1", "node/x", 3, "hash-3")
        store.put_ref("traj-1", "node/x", 2, "hash-2")
        assert store.resolve_ref("traj-1", "node/x") == "hash-3"

    def test_resolve_iteration_without_node_path_raises(
        self, store: JsonFileCheckpointStore
    ) -> None:
        with pytest.raises(ValueError, match="iteration requires node_path"):
            store.resolve_ref("traj-1", None, 5)

    def test_put_ref_overwrites_same_key(self, store: JsonFileCheckpointStore) -> None:
        """Same (client_flow_id, node_path, iteration) re-put replaces."""
        store.put_ref("traj-1", "node/x", 5, "hash-first")
        store.put_ref("traj-1", "node/x", 5, "hash-second")
        assert store.resolve_ref("traj-1", "node/x", 5) == "hash-second"

    def test_refs_do_not_leak_across_trajectories(self, store: JsonFileCheckpointStore) -> None:
        store.put_ref("traj-a", "node/x", 1, "hash-a")
        store.put_ref("traj-b", "node/x", 1, "hash-b")
        assert store.resolve_ref("traj-a", "node/x", 1) == "hash-a"
        assert store.resolve_ref("traj-b", "node/x", 1) == "hash-b"


# ---------------------------------------------------------------------------
# gc_trajectory
# ---------------------------------------------------------------------------


class TestGcTrajectory:
    def test_removes_all_objects_and_refs(self, store: JsonFileCheckpointStore) -> None:
        store.put_object("traj-1", "blob", "h1", b"payload")
        store.put_ref("traj-1", "node/x", 1, "commit-h")
        store.gc_trajectory("traj-1")
        assert store.get_object("traj-1", "blob", "h1") is None
        assert store.resolve_ref("traj-1", "node/x", 1) is None

    def test_idempotent_when_absent(self, store: JsonFileCheckpointStore) -> None:
        store.gc_trajectory("never-existed")  # no raise

    def test_leaves_other_trajectories_intact(self, store: JsonFileCheckpointStore) -> None:
        store.put_object("traj-a", "blob", "h", b"a-bytes")
        store.put_object("traj-b", "blob", "h", b"b-bytes")
        store.gc_trajectory("traj-a")
        assert store.get_object("traj-a", "blob", "h") is None
        assert store.get_object("traj-b", "blob", "h") == b"b-bytes"


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetention:
    def test_default_is_retain(self, tmp_path: Path) -> None:
        s = JsonFileCheckpointStore(make_test_logger(), tmp_path)
        assert s.retention == "retain"

    def test_explicit_retain(self, tmp_path: Path) -> None:
        s = JsonFileCheckpointStore(make_test_logger(), tmp_path, retention="retain")
        assert s.retention == "retain"

    def test_explicit_gc_on_success(self, tmp_path: Path) -> None:
        s = JsonFileCheckpointStore(make_test_logger(), tmp_path, retention="gc_on_success")
        assert s.retention == "gc_on_success"


# ---------------------------------------------------------------------------
# Path-traversal guards
# ---------------------------------------------------------------------------


class TestPathTraversalGuards:
    @pytest.mark.parametrize("bad_id", ["", ".", ".."])
    def test_rejects_adversarial_client_flow_id(
        self, store: JsonFileCheckpointStore, bad_id: str
    ) -> None:
        with pytest.raises(ValueError):
            store.put_object(bad_id, "blob", "h", b"x")
        with pytest.raises(ValueError):
            store.get_object(bad_id, "blob", "h")

    @pytest.mark.parametrize("bad_path", ["", ".", ".."])
    def test_rejects_adversarial_node_path(
        self, store: JsonFileCheckpointStore, bad_path: str
    ) -> None:
        with pytest.raises(ValueError):
            store.put_ref("traj-1", bad_path, 1, "hash")

    def test_slash_and_special_chars_supported(self, store: JsonFileCheckpointStore) -> None:
        """URL-quoting round-trips arbitrary caller strings through path segments."""
        weird_id = "campaign/2026-09-22:15h30 "
        weird_path = "cafebabe/deadbeef"
        store.put_object(weird_id, "blob", "h", b"x")
        store.put_ref(weird_id, weird_path, 1, "commit-h")
        assert store.get_object(weird_id, "blob", "h") == b"x"
        assert store.resolve_ref(weird_id, weird_path, 1) == "commit-h"
