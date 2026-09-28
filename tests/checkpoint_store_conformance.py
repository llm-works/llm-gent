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

    # --- ref store ---

    def test_put_resolve_exact_key(self, store: Any) -> None:
        store.put_ref("history-1", "node/x", 5, "commit-hash-5")
        assert store.resolve_ref("history-1", "node/x", 5) == "commit-hash-5"

    def test_resolve_ref_returns_none_when_absent(self, store: Any) -> None:
        assert store.resolve_ref("history-1") is None
        assert store.resolve_ref("history-1", "node/x") is None
        assert store.resolve_ref("history-1", "node/x", 5) is None

    def test_resolve_latest_across_node_paths(self, store: Any) -> None:
        """No node_path: the newest write across the history."""
        store.put_ref("history-1", "node/a", 1, "hash-1")
        store.put_ref("history-1", "node/b", 1, "hash-2")
        assert store.resolve_ref("history-1") == "hash-2"

    def test_re_put_becomes_latest(self, store: Any) -> None:
        """Re-putting an existing key counts as a new write, so it becomes the newest."""
        store.put_ref("history-1", "node/a", 1, "hash-a")
        store.put_ref("history-1", "node/b", 1, "hash-b")
        store.put_ref("history-1", "node/a", 1, "hash-a2")
        assert store.resolve_ref("history-1") == "hash-a2"

    def test_resolve_latest_under_node_path(self, store: Any) -> None:
        """node_path alone: its highest iteration, not its newest write."""
        store.put_ref("history-1", "node/x", 1, "hash-1")
        store.put_ref("history-1", "node/x", 3, "hash-3")
        store.put_ref("history-1", "node/x", 2, "hash-2")
        assert store.resolve_ref("history-1", "node/x") == "hash-3"

    def test_resolve_iteration_without_node_path_raises(self, store: Any) -> None:
        with pytest.raises(ValueError, match="iteration requires node_path"):
            store.resolve_ref("history-1", None, 5)

    def test_put_ref_overwrites_same_key(self, store: Any) -> None:
        store.put_ref("history-1", "node/x", 5, "hash-first")
        store.put_ref("history-1", "node/x", 5, "hash-second")
        assert store.resolve_ref("history-1", "node/x", 5) == "hash-second"

    def test_refs_do_not_leak_across_histories(self, store: Any) -> None:
        store.put_ref("history-a", "node/x", 1, "hash-a")
        store.put_ref("history-b", "node/x", 1, "hash-b")
        assert store.resolve_ref("history-a", "node/x", 1) == "hash-a"
        assert store.resolve_ref("history-b", "node/x", 1) == "hash-b"
        assert store.resolve_ref("history-a") == "hash-a"

    # --- name map ---

    def test_bind_get_round_trip(self, store: Any) -> None:
        assert store.bind_flow_id("campaign-1", "history-1") == "history-1"
        assert store.get_flow_id("campaign-1") == "history-1"

    def test_get_flow_id_returns_none_when_unbound(self, store: Any) -> None:
        assert store.get_flow_id("campaign-1") is None

    def test_second_bind_returns_existing(self, store: Any) -> None:
        store.bind_flow_id("campaign-1", "history-1")
        assert store.bind_flow_id("campaign-1", "history-2") == "history-1"
        assert store.get_flow_id("campaign-1") == "history-1"

    def test_flow_id_cannot_name_two_histories(self, store: Any) -> None:
        store.bind_flow_id("campaign-1", "history-1")
        with pytest.raises(ValueError, match="already names"):
            store.bind_flow_id("campaign-2", "history-1")

    def test_bind_after_unbound_writes(self, store: Any) -> None:
        """A history written to before its name was bound can still be bound."""
        store.put_object("history-1", "blob", "h", b"x")
        assert store.bind_flow_id("campaign-1", "history-1") == "history-1"
        assert store.get_object("history-1", "blob", "h") == b"x"

    def test_concurrent_binds_agree_on_one_winner(self, store: Any) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda i: store.bind_flow_id("campaign-1", f"history-{i}"), range(16))
            )
        assert len(set(results)) == 1
        assert store.get_flow_id("campaign-1") == results[0]

    # --- tags ---

    def test_tag_put_resolve_round_trip(self, store: Any) -> None:
        store.put_tag("history-1", "complete", "commit-h")
        assert store.resolve_tag("history-1", "complete") == "commit-h"

    def test_resolve_tag_returns_none_when_absent(self, store: Any) -> None:
        assert store.resolve_tag("history-1", "complete") is None

    def test_re_put_moves_the_tag(self, store: Any) -> None:
        store.put_tag("history-1", "complete", "commit-1")
        store.put_tag("history-1", "complete", "commit-2")
        assert store.resolve_tag("history-1", "complete") == "commit-2"

    def test_tags_do_not_leak_across_histories(self, store: Any) -> None:
        store.put_tag("history-a", "complete", "hash-a")
        assert store.resolve_tag("history-b", "complete") is None

    # --- gc_history ---

    def test_gc_removes_objects_refs_and_tags(self, store: Any) -> None:
        store.put_object("history-1", "blob", "h1", b"payload")
        store.put_ref("history-1", "node/x", 1, "commit-h")
        store.put_tag("history-1", "complete", "commit-h")
        store.gc_history("history-1")
        assert store.get_object("history-1", "blob", "h1") is None
        assert store.resolve_ref("history-1", "node/x", 1) is None
        assert store.resolve_ref("history-1") is None
        assert store.resolve_tag("history-1", "complete") is None

    def test_gc_frees_the_name(self, store: Any) -> None:
        store.bind_flow_id("campaign-1", "history-1")
        store.gc_history("history-1")
        assert store.get_flow_id("campaign-1") is None
        store.bind_flow_id("campaign-1", "history-2")
        assert store.get_flow_id("campaign-1") == "history-2"

    def test_gc_idempotent_when_absent(self, store: Any) -> None:
        store.gc_history("never-existed")

    def test_gc_leaves_other_histories_intact(self, store: Any) -> None:
        store.bind_flow_id("campaign-a", "history-a")
        store.bind_flow_id("campaign-b", "history-b")
        store.put_object("history-a", "blob", "h", b"a-bytes")
        store.put_object("history-b", "blob", "h", b"b-bytes")
        store.put_ref("history-b", "node/x", 1, "hash-b")
        store.gc_history("history-a")
        assert store.get_object("history-a", "blob", "h") is None
        assert store.get_object("history-b", "blob", "h") == b"b-bytes"
        assert store.resolve_ref("history-b") == "hash-b"
        assert store.get_flow_id("campaign-b") == "history-b"

    # --- retention ---

    def test_default_retention_is_retain(self, store: Any) -> None:
        assert store.retention == "retain"
