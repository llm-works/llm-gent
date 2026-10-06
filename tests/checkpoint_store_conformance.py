# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Behaviour every :class:`~llm_gent.flow.CheckpointStore` must share.

:class:`CheckpointStoreConformance` holds the Protocol-level tests. A
store's test module subclasses it as ``Test...`` and provides a
``store`` fixture returning a fresh, empty store; tests specific to one
backend (file layout, SQL clock handling) stay in that module.

The class name has no ``Test`` prefix, so pytest collects it only
through its subclasses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest


class CheckpointStoreConformance:
    """Protocol tests; subclasses supply the ``store`` fixture."""

    # --- object store ---

    def test_put_get_round_trip(self, store: Any) -> None:
        store.put_object("history-1", "blob", "hash-a", b"payload-a")
        assert store.get_object("history-1", "blob", "hash-a") == b"payload-a"

    def test_get_returns_none_when_absent(self, store: Any) -> None:
        assert store.get_object("history-1", "blob", "missing") is None

    def test_has_object_reflects_presence(self, store: Any) -> None:
        assert not store.has_object("history-1", "blob", "hash-a")
        store.put_object("history-1", "blob", "hash-a", b"payload")
        assert store.has_object("history-1", "blob", "hash-a")

    def test_put_idempotent_same_hash(self, store: Any) -> None:
        store.put_object("history-1", "blob", "hash-a", b"payload")
        store.put_object("history-1", "blob", "hash-a", b"payload")
        assert store.get_object("history-1", "blob", "hash-a") == b"payload"

    def test_kinds_do_not_collide(self, store: Any) -> None:
        """Same hash under different kinds addresses different objects."""
        store.put_object("history-1", "blob", "hash", b"blob-bytes")
        store.put_object("history-1", "tree", "hash", b"tree-bytes")
        store.put_object("history-1", "commit", "hash", b"commit-bytes")
        assert store.get_object("history-1", "blob", "hash") == b"blob-bytes"
        assert store.get_object("history-1", "tree", "hash") == b"tree-bytes"
        assert store.get_object("history-1", "commit", "hash") == b"commit-bytes"

    def test_histories_do_not_collide(self, store: Any) -> None:
        """Same (kind, hash) under two histories store separately."""
        store.put_object("history-a", "blob", "hash", b"a-bytes")
        store.put_object("history-b", "blob", "hash", b"b-bytes")
        assert store.get_object("history-a", "blob", "hash") == b"a-bytes"
        assert store.get_object("history-b", "blob", "hash") == b"b-bytes"

    # --- refs ---

    def test_get_ref_returns_none_when_absent(self, store: Any) -> None:
        assert store.get_ref("history-1", "HEAD") is None

    def test_set_ref_creates_when_expected_absent(self, store: Any) -> None:
        assert store.set_ref("history-1", "HEAD", "commit-1", None) is True
        assert store.get_ref("history-1", "HEAD") == "commit-1"

    def test_create_refused_when_ref_exists(self, store: Any) -> None:
        store.set_ref("history-1", "HEAD", "commit-1", None)
        assert store.set_ref("history-1", "HEAD", "commit-2", None) is False
        assert store.get_ref("history-1", "HEAD") == "commit-1"

    def test_set_ref_moves_from_expected(self, store: Any) -> None:
        store.set_ref("history-1", "HEAD", "commit-1", None)
        assert store.set_ref("history-1", "HEAD", "commit-2", "commit-1") is True
        assert store.get_ref("history-1", "HEAD") == "commit-2"

    def test_move_refused_from_stale_expected(self, store: Any) -> None:
        """A writer whose view of the ref is stale changes nothing."""
        store.set_ref("history-1", "HEAD", "commit-1", None)
        store.set_ref("history-1", "HEAD", "commit-2", "commit-1")
        assert store.set_ref("history-1", "HEAD", "commit-3", "commit-1") is False
        assert store.get_ref("history-1", "HEAD") == "commit-2"

    def test_move_refused_when_ref_absent(self, store: Any) -> None:
        assert store.set_ref("history-1", "HEAD", "commit-2", "commit-1") is False
        assert store.get_ref("history-1", "HEAD") is None

    def test_ref_names_are_independent(self, store: Any) -> None:
        """A name with a slash is one ref, not a hierarchy."""
        store.set_ref("history-1", "HEAD", "commit-1", None)
        store.set_ref("history-1", "tags/complete", "commit-0", None)
        assert store.get_ref("history-1", "HEAD") == "commit-1"
        assert store.get_ref("history-1", "tags/complete") == "commit-0"
        assert store.get_ref("history-1", "tags") is None

    def test_refs_do_not_leak_across_histories(self, store: Any) -> None:
        store.set_ref("history-a", "HEAD", "hash-a", None)
        store.set_ref("history-b", "HEAD", "hash-b", None)
        assert store.get_ref("history-a", "HEAD") == "hash-a"
        assert store.get_ref("history-b", "HEAD") == "hash-b"

    def test_concurrent_moves_from_one_parent_have_one_winner(self, store: Any) -> None:
        store.set_ref("history-1", "HEAD", "commit-0", None)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: store.set_ref("history-1", "HEAD", f"commit-{i}", "commit-0"),
                    range(1, 17),
                )
            )
        assert results.count(True) == 1
        assert store.get_ref("history-1", "HEAD") == f"commit-{results.index(True) + 1}"

    # --- name map ---

    def test_bind_get_round_trip(self, store: Any) -> None:
        assert store.bind_flow_id("client-1", "history-1") == "history-1"
        assert store.get_flow_id("client-1") == "history-1"

    def test_get_flow_id_returns_none_when_unbound(self, store: Any) -> None:
        assert store.get_flow_id("client-1") is None

    def test_second_bind_returns_existing(self, store: Any) -> None:
        store.bind_flow_id("client-1", "history-1")
        assert store.bind_flow_id("client-1", "history-2") == "history-1"
        assert store.get_flow_id("client-1") == "history-1"

    def test_flow_id_cannot_name_two_histories(self, store: Any) -> None:
        store.bind_flow_id("client-1", "history-1")
        with pytest.raises(ValueError, match="already names"):
            store.bind_flow_id("client-2", "history-1")

    def test_bind_after_unbound_writes(self, store: Any) -> None:
        """A history written to before its name was bound can still be bound."""
        store.put_object("history-1", "blob", "h", b"x")
        assert store.bind_flow_id("client-1", "history-1") == "history-1"
        assert store.get_object("history-1", "blob", "h") == b"x"

    def test_concurrent_binds_agree_on_one_winner(self, store: Any) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda i: store.bind_flow_id("client-1", f"history-{i}"), range(16))
            )
        assert len(set(results)) == 1
        assert store.get_flow_id("client-1") == results[0]

    # --- gc_history ---

    def test_gc_removes_objects_and_refs(self, store: Any) -> None:
        store.put_object("history-1", "blob", "h1", b"payload")
        store.set_ref("history-1", "HEAD", "commit-h", None)
        store.set_ref("history-1", "tags/complete", "commit-h", None)
        store.gc_history("history-1")
        assert store.get_object("history-1", "blob", "h1") is None
        assert store.get_ref("history-1", "HEAD") is None
        assert store.get_ref("history-1", "tags/complete") is None

    def test_gc_frees_the_name(self, store: Any) -> None:
        store.bind_flow_id("client-1", "history-1")
        store.gc_history("history-1")
        assert store.get_flow_id("client-1") is None
        store.bind_flow_id("client-1", "history-2")
        assert store.get_flow_id("client-1") == "history-2"

    def test_gc_idempotent_when_absent(self, store: Any) -> None:
        store.gc_history("never-existed")

    def test_gc_leaves_other_histories_intact(self, store: Any) -> None:
        store.bind_flow_id("client-a", "history-a")
        store.bind_flow_id("client-b", "history-b")
        store.put_object("history-a", "blob", "h", b"a-bytes")
        store.put_object("history-b", "blob", "h", b"b-bytes")
        store.set_ref("history-b", "HEAD", "hash-b", None)
        store.gc_history("history-a")
        assert store.get_object("history-a", "blob", "h") is None
        assert store.get_object("history-b", "blob", "h") == b"b-bytes"
        assert store.get_ref("history-b", "HEAD") == "hash-b"
        assert store.get_flow_id("client-b") == "history-b"

    # --- listing and deleting objects ---

    def test_list_objects_returns_every_key_of_the_history(self, store: Any) -> None:
        store.put_object("history-1", "blob", "b", b"x")
        store.put_object("history-1", "tree", "t", b"y")
        store.put_object("history-1", "commit", "c", b"z")
        store.put_object("history-2", "blob", "other", b"w")
        assert sorted(store.list_objects("history-1")) == [
            ("blob", "b"),
            ("commit", "c"),
            ("tree", "t"),
        ]
        assert store.list_objects("never-existed") == []

    def test_delete_objects_removes_only_the_given_keys(self, store: Any) -> None:
        store.put_object("history-1", "blob", "keep", b"x")
        store.put_object("history-1", "blob", "drop", b"y")
        store.put_object("history-1", "tree", "drop", b"z")
        store.put_object("history-2", "blob", "drop", b"w")
        store.delete_objects("history-1", [("blob", "drop"), ("tree", "drop"), ("commit", "gone")])
        assert store.list_objects("history-1") == [("blob", "keep")]
        assert store.get_object("history-2", "blob", "drop") == b"w"

    def test_list_refs_returns_every_ref_of_the_history(self, store: Any) -> None:
        store.set_ref("history-1", "HEAD", "commit-2", None)
        store.set_ref("history-1", "tags/complete", "commit-1", None)
        store.set_ref("history-2", "HEAD", "commit-x", None)
        assert store.list_refs("history-1") == {"HEAD": "commit-2", "tags/complete": "commit-1"}
        assert store.list_refs("never-existed") == {}

    def test_list_refs_includes_refs_ending_in_tmp(self, store: Any) -> None:
        store.set_ref("history-1", "tags/archive.tmp", "commit-1", None)
        store.set_ref("history-1", "HEAD", "commit-2", None)
        assert store.list_refs("history-1") == {
            "tags/archive.tmp": "commit-1",
            "HEAD": "commit-2",
        }

    # --- retention ---

    def test_default_retention_is_retain(self, store: Any) -> None:
        assert store.retention == "retain"
