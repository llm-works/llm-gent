# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.flow.stores.JsonFileCheckpointStore`.

Exercises the Protocol contract end-to-end against a local directory:
save/load/delete round-trip, latest-vs-explicit-iteration reads,
idempotent-overwrite on same-iteration re-save, atomic writes, and
absent-record semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import make_test_logger


pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A store rooted at ``tmp_path/checkpoints``. Root is created lazily."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "checkpoints")


def _state(n: int) -> dict[str, Any]:
    """Small deterministic ``state_json`` fixture."""
    return {"data": {"n": n}, "children": []}


def _meta(iteration: int, path: list[str] | None = None) -> dict[str, Any]:
    """Small deterministic ``metadata_json`` fixture."""
    return {"path": path or ["node-a"], "iteration": iteration}


class TestJsonFileCheckpointStore:
    """Save / load / delete Protocol conformance against the local FS."""

    def test_round_trip_single_iteration(self, store: JsonFileCheckpointStore) -> None:
        """Save one record, load it back verbatim."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(42), _meta(1))
        loaded = store.load_checkpoint("traj-1")
        assert loaded is not None
        state_json, meta_json = loaded
        assert state_json == _state(42)
        assert meta_json == _meta(1)

    def test_load_returns_none_when_absent(self, store: JsonFileCheckpointStore) -> None:
        """A never-saved trajectory reads as ``None``."""
        assert store.load_checkpoint("nobody") is None

    def test_load_specific_iteration(self, store: JsonFileCheckpointStore) -> None:
        """``iteration=N`` fetches exactly that record when it exists."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 2, _state(2), _meta(2))
        loaded = store.load_checkpoint("traj-1", node_path="node-a", iteration=1)
        assert loaded is not None
        assert loaded[0] == _state(1)

    def test_load_specific_iteration_missing(self, store: JsonFileCheckpointStore) -> None:
        """A non-existent iteration under a live trajectory reads as ``None``."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        assert store.load_checkpoint("traj-1", node_path="node-a", iteration=99) is None

    def test_load_latest_picks_last_save(self, store: JsonFileCheckpointStore) -> None:
        """Both filters ``None`` returns the most recently written record (last save wins)."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 3, _state(3), _meta(3))
        store.save_checkpoint("traj-1", "node-a", 2, _state(2), _meta(2))
        loaded = store.load_checkpoint("traj-1")
        assert loaded is not None
        assert loaded[1]["iteration"] == 2

    def test_same_iteration_resave_overwrites(self, store: JsonFileCheckpointStore) -> None:
        """A second save at the same iteration replaces the earlier record."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 1, _state(99), _meta(1))
        loaded = store.load_checkpoint("traj-1", node_path="node-a", iteration=1)
        assert loaded is not None
        assert loaded[0] == _state(99)

    def test_delete_wipes_trajectory(self, store: JsonFileCheckpointStore) -> None:
        """Every iteration under the trajectory is gone after delete."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", "node-a", 2, _state(2), _meta(2))
        store.delete_checkpoint("traj-1")
        assert store.load_checkpoint("traj-1") is None
        assert store.load_checkpoint("traj-1", node_path="node-a", iteration=1) is None
        assert store.load_checkpoint("traj-1", node_path="node-a", iteration=2) is None

    def test_delete_idempotent_when_absent(self, store: JsonFileCheckpointStore) -> None:
        """Deleting an unknown trajectory is a no-op, not an error."""
        store.delete_checkpoint("nobody")

    def test_delete_leaves_other_trajectories(self, store: JsonFileCheckpointStore) -> None:
        """Deleting one trajectory does not affect a sibling under the same root."""
        store.save_checkpoint("traj-a", "node-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-b", "node-a", 1, _state(2), _meta(1))
        store.delete_checkpoint("traj-a")
        assert store.load_checkpoint("traj-a") is None
        loaded = store.load_checkpoint("traj-b")
        assert loaded is not None
        assert loaded[0] == _state(2)

    def test_trajectory_ids_with_path_unsafe_chars(self, store: JsonFileCheckpointStore) -> None:
        """URL-quoting keeps arbitrary caller ids path-safe (slashes, colons, spaces)."""
        weird_id = "team/agent-1:run 42"
        store.save_checkpoint(weird_id, "node-a", 1, _state(7), _meta(1))
        loaded = store.load_checkpoint(weird_id)
        assert loaded is not None
        assert loaded[0] == _state(7)

    def test_creates_root_lazily(self, tmp_path: Path) -> None:
        """The root directory does not need to exist at construction time."""
        deep = tmp_path / "does" / "not" / "exist"
        assert not deep.exists()
        s = JsonFileCheckpointStore(make_test_logger(), deep)
        s.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        assert deep.is_dir()

    def test_unreadable_file_treated_as_absent(
        self, store: JsonFileCheckpointStore, tmp_path: Path
    ) -> None:
        """A truncated / corrupt file logs and reads as ``None`` rather than crashing."""
        store.save_checkpoint("traj-1", "node-a", 1, _state(1), _meta(1))
        # Corrupt the on-disk file.
        target = tmp_path / "checkpoints" / "traj-1" / "save-1.json"
        target.write_text("{not valid json", encoding="utf-8")
        assert store.load_checkpoint("traj-1", node_path="node-a", iteration=1) is None

    def test_distinct_node_paths_do_not_collide(self, store: JsonFileCheckpointStore) -> None:
        """Two iterates' iteration=1 records under one trajectory coexist without overwrite.

        Under the old (client_flow_id, iteration) key, an inner iterate's
        iteration=1 save would upsert on top of an outer's iteration=1.
        With node_path in the key, both records live on disk and each
        is retrievable by its own (node_path, iteration).
        """
        store.save_checkpoint("traj-1", "outer", 1, _state(11), _meta(1, ["outer"]))
        store.save_checkpoint("traj-1", "outer/inner", 1, _state(99), _meta(1, ["outer", "inner"]))

        outer = store.load_checkpoint("traj-1", node_path="outer", iteration=1)
        assert outer is not None
        assert outer[0] == _state(11)

        inner = store.load_checkpoint("traj-1", node_path="outer/inner", iteration=1)
        assert inner is not None
        assert inner[0] == _state(99)


class TestJsonFileCheckpointStoreAdversarialIds:
    """Path-traversal / malformed-id rejection at :meth:`_trajectory_dir`.

    :func:`urllib.parse.quote` leaves the RFC-3986 "unreserved" set
    unencoded, which includes ``.`` — so a naive ``quote(id, safe="")``
    round-trips ``"."`` and ``".."`` verbatim and lets a caller escape
    the store root. Every public method routes through
    :meth:`_trajectory_dir`, so validating there is sufficient.
    """

    @pytest.mark.parametrize("bad_id", ["", ".", ".."])
    def test_save_rejects_adversarial_id(self, store: JsonFileCheckpointStore, bad_id: str) -> None:
        """Every save-time rejection surfaces as :exc:`ValueError`, not a silent write."""
        with pytest.raises(ValueError):
            store.save_checkpoint(bad_id, "node-a", 1, _state(1), _meta(1))

    @pytest.mark.parametrize("bad_id", ["", ".", ".."])
    def test_load_rejects_adversarial_id(self, store: JsonFileCheckpointStore, bad_id: str) -> None:
        """Load through the same validator; no silent read from outside root."""
        with pytest.raises(ValueError):
            store.load_checkpoint(bad_id)

    @pytest.mark.parametrize("bad_id", ["", ".", ".."])
    def test_delete_rejects_adversarial_id(
        self, store: JsonFileCheckpointStore, bad_id: str
    ) -> None:
        """Delete through the same validator; no silent wipe outside root."""
        with pytest.raises(ValueError):
            store.delete_checkpoint(bad_id)

    def test_slash_and_special_chars_still_supported(self, store: JsonFileCheckpointStore) -> None:
        """Slashes / colons / spaces / backslashes / NUL round-trip via URL-quote.

        :func:`urllib.parse.quote` encodes every path-relevant char that
        isn't in the RFC-3986 unreserved set, so these safely collapse
        to a single directory name inside the root. Only ``.`` and
        ``..`` need explicit rejection (they are unreserved).
        """
        for weird_id in [
            "team/agent-1:run 42",
            "back\\slash",
            "nul\x00byte",
            "..foo",
            "foo..",
        ]:
            store.save_checkpoint(weird_id, "node-a", 1, _state(1), _meta(1))
            loaded = store.load_checkpoint(weird_id)
            assert loaded is not None
            assert loaded[0] == _state(1)

    def test_dotdot_would_have_escaped_root(
        self, tmp_path: Path, store: JsonFileCheckpointStore
    ) -> None:
        """Sanity check: a sibling of the root MUST NOT be reachable via ``..``.

        Plants a marker file at ``tmp_path/marker.json`` (a sibling of the
        store root), attempts a rejected save/delete with ``client_flow_id="..``,
        then asserts the marker survives. Locks the fix in against a future
        regression that would let ``..`` through the validator.
        """
        marker = tmp_path / "marker.json"
        marker.write_text("keep me", encoding="utf-8")
        with pytest.raises(ValueError):
            store.save_checkpoint("..", "node-a", 1, _state(1), _meta(1))
        with pytest.raises(ValueError):
            store.delete_checkpoint("..")
        assert marker.read_text(encoding="utf-8") == "keep me"
