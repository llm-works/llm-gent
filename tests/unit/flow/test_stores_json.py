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
        store.save_checkpoint("traj-1", 1, _state(42), _meta(1))
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
        store.save_checkpoint("traj-1", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", 2, _state(2), _meta(2))
        loaded = store.load_checkpoint("traj-1", iteration=1)
        assert loaded is not None
        assert loaded[0] == _state(1)

    def test_load_specific_iteration_missing(self, store: JsonFileCheckpointStore) -> None:
        """A non-existent iteration under a live trajectory reads as ``None``."""
        store.save_checkpoint("traj-1", 1, _state(1), _meta(1))
        assert store.load_checkpoint("traj-1", iteration=99) is None

    def test_load_latest_picks_highest_iteration(self, store: JsonFileCheckpointStore) -> None:
        """``iteration=None`` returns the record with the highest ``iteration``."""
        store.save_checkpoint("traj-1", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", 3, _state(3), _meta(3))
        store.save_checkpoint("traj-1", 2, _state(2), _meta(2))
        loaded = store.load_checkpoint("traj-1")
        assert loaded is not None
        assert loaded[1]["iteration"] == 3

    def test_same_iteration_resave_overwrites(self, store: JsonFileCheckpointStore) -> None:
        """A second save at the same iteration replaces the earlier record."""
        store.save_checkpoint("traj-1", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", 1, _state(99), _meta(1))
        loaded = store.load_checkpoint("traj-1", iteration=1)
        assert loaded is not None
        assert loaded[0] == _state(99)

    def test_delete_wipes_trajectory(self, store: JsonFileCheckpointStore) -> None:
        """Every iteration under the trajectory is gone after delete."""
        store.save_checkpoint("traj-1", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-1", 2, _state(2), _meta(2))
        store.delete_checkpoint("traj-1")
        assert store.load_checkpoint("traj-1") is None
        assert store.load_checkpoint("traj-1", iteration=1) is None
        assert store.load_checkpoint("traj-1", iteration=2) is None

    def test_delete_idempotent_when_absent(self, store: JsonFileCheckpointStore) -> None:
        """Deleting an unknown trajectory is a no-op, not an error."""
        store.delete_checkpoint("nobody")

    def test_delete_leaves_other_trajectories(self, store: JsonFileCheckpointStore) -> None:
        """Deleting one trajectory does not affect a sibling under the same root."""
        store.save_checkpoint("traj-a", 1, _state(1), _meta(1))
        store.save_checkpoint("traj-b", 1, _state(2), _meta(1))
        store.delete_checkpoint("traj-a")
        assert store.load_checkpoint("traj-a") is None
        loaded = store.load_checkpoint("traj-b")
        assert loaded is not None
        assert loaded[0] == _state(2)

    def test_trajectory_ids_with_path_unsafe_chars(self, store: JsonFileCheckpointStore) -> None:
        """URL-quoting keeps arbitrary caller ids path-safe (slashes, colons, spaces)."""
        weird_id = "team/agent-1:run 42"
        store.save_checkpoint(weird_id, 1, _state(7), _meta(1))
        loaded = store.load_checkpoint(weird_id)
        assert loaded is not None
        assert loaded[0] == _state(7)

    def test_creates_root_lazily(self, tmp_path: Path) -> None:
        """The root directory does not need to exist at construction time."""
        deep = tmp_path / "does" / "not" / "exist"
        assert not deep.exists()
        s = JsonFileCheckpointStore(make_test_logger(), deep)
        s.save_checkpoint("traj-1", 1, _state(1), _meta(1))
        assert deep.is_dir()

    def test_unreadable_file_treated_as_absent(
        self, store: JsonFileCheckpointStore, tmp_path: Path
    ) -> None:
        """A truncated / corrupt file logs and reads as ``None`` rather than crashing."""
        store.save_checkpoint("traj-1", 1, _state(1), _meta(1))
        # Corrupt the on-disk file.
        target = tmp_path / "checkpoints" / "traj-1" / "iter-1.json"
        target.write_text("{not valid json", encoding="utf-8")
        assert store.load_checkpoint("traj-1", iteration=1) is None
