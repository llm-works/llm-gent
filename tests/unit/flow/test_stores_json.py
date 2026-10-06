# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.flow.stores.JsonFileCheckpointStore`.

Protocol behaviour comes from :class:`CheckpointStoreConformance`; this
module adds the file store's own concerns: directory cleanup, gc retry
after a partial failure, and path-traversal guards. End-to-end resume
behavior is covered in :mod:`tests.unit.flow.test_checkpoint`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llm_gent.flow.stores import JsonFileCheckpointStore
from tests.checkpoint_store_conformance import CheckpointStoreConformance

from .conftest import make_test_logger


pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    """A fresh store under an isolated temp root."""
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "checkpoints")


class TestConformance(CheckpointStoreConformance):
    """The Protocol behaviour every store shares."""


class TestDirectories:
    def test_second_bind_leaves_no_directory(self, store: JsonFileCheckpointStore) -> None:
        store.bind_flow_id("client-1", "history-1")
        store.bind_flow_id("client-1", "history-2")
        assert not store._history_dir("history-2").exists()

    def test_concurrent_bind_losers_leave_no_directory(
        self, store: JsonFileCheckpointStore
    ) -> None:
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(
                pool.map(lambda i: store.bind_flow_id("client-1", f"history-{i}"), range(32))
            )
        assert [p.name for p in (store._root / "histories").iterdir()] == [results[0]]

    def test_failed_gc_keeps_name_for_retry(
        self, store: JsonFileCheckpointStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gc that fails partway leaves the name bound, so the retry finds and finishes it."""
        from llm_gent.flow.stores import json_file

        store.bind_flow_id("client-1", "history-1")
        store.put_object("history-1", "blob", "h1", b"payload")
        real_rmtree = json_file.shutil.rmtree

        def failing_rmtree(path: object, *args: object, **kwargs: object) -> None:
            raise OSError("disk went away")

        monkeypatch.setattr(json_file.shutil, "rmtree", failing_rmtree)
        with pytest.raises(OSError):
            store.gc_history("history-1")
        assert store.get_flow_id("client-1") == "history-1"

        monkeypatch.setattr(json_file.shutil, "rmtree", real_rmtree)
        store.gc_history("history-1")
        assert store.get_flow_id("client-1") is None
        assert not store._history_dir("history-1").exists()


class TestRetention:
    def test_explicit_retain(self, tmp_path: Path) -> None:
        s = JsonFileCheckpointStore(make_test_logger(), tmp_path, retention="retain")
        assert s.retention == "retain"

    def test_explicit_gc_on_success(self, tmp_path: Path) -> None:
        s = JsonFileCheckpointStore(make_test_logger(), tmp_path, retention="gc_on_success")
        assert s.retention == "gc_on_success"


class TestPathTraversalGuards:
    @pytest.mark.parametrize("bad_id", ["", ".", ".."])
    def test_rejects_adversarial_client_flow_id(
        self, store: JsonFileCheckpointStore, bad_id: str
    ) -> None:
        with pytest.raises(ValueError):
            store.put_object(bad_id, "blob", "h", b"x")
        with pytest.raises(ValueError):
            store.get_object(bad_id, "blob", "h")

    @pytest.mark.parametrize("bad_name", ["", ".", ".."])
    def test_rejects_adversarial_names(self, store: JsonFileCheckpointStore, bad_name: str) -> None:
        with pytest.raises(ValueError):
            store.set_ref("history-1", bad_name, "hash", None)
        with pytest.raises(ValueError):
            store.get_ref("history-1", bad_name)
        with pytest.raises(ValueError):
            store.bind_flow_id(bad_name, "history-1")

    def test_slash_and_special_chars_supported(self, store: JsonFileCheckpointStore) -> None:
        """URL-quoting round-trips arbitrary caller strings through path segments."""
        weird_id = "client/2026-09-22:15h30 "
        weird_ref = "tags/complete: v1"
        store.put_object(weird_id, "blob", "h", b"x")
        store.set_ref(weird_id, weird_ref, "commit-h", None)
        assert store.get_object(weird_id, "blob", "h") == b"x"
        assert store.get_ref(weird_id, weird_ref) == "commit-h"
        refs_dir = store._history_dir(weird_id) / "refs"
        assert sorted(p.name for p in refs_dir.iterdir() if p.name != ".lock") == [
            "tags%2Fcomplete%3A%20v1"
        ]
