# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.flow.stores.InMemoryCheckpointStore`.

Protocol behaviour comes from :class:`CheckpointStoreConformance`; this
module adds what only the in-memory store does: refusing different
bytes under an existing hash, empty-key checks, and the Flow round trip
through a store that outlives the flow.
"""

from __future__ import annotations

import pytest

from llm_gent.flow.stores import InMemoryCheckpointStore
from tests.checkpoint_store_conformance import CheckpointStoreConformance

from .conftest import make_test_logger


pytestmark = pytest.mark.unit


@pytest.fixture
def store() -> InMemoryCheckpointStore:
    """A fresh, empty store."""
    return InMemoryCheckpointStore()


class TestConformance(CheckpointStoreConformance):
    """The Protocol behaviour every store shares."""


class TestMemoryStore:
    def test_different_bytes_under_an_existing_hash_raise(
        self, store: InMemoryCheckpointStore
    ) -> None:
        store.put_object("history-1", "blob", "h", b"one")
        with pytest.raises(ValueError, match="different bytes"):
            store.put_object("history-1", "blob", "h", b"two")

    def test_stored_payload_is_a_copy(self, store: InMemoryCheckpointStore) -> None:
        payload = bytearray(b"abc")
        store.put_object("history-1", "blob", "h", payload)  # type: ignore[arg-type]
        payload[0] = ord("z")
        assert store.get_object("history-1", "blob", "h") == b"abc"

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.bind_flow_id("", "history-1"),
            lambda s: s.bind_flow_id("campaign-1", ""),
            lambda s: s.put_object("", "blob", "h", b"x"),
            lambda s: s.put_ref("history-1", "", 1, "c"),
            lambda s: s.put_tag("history-1", "", "c"),
        ],
        ids=["client-flow-id", "flow-id", "object-flow-id", "node-path", "tag-name"],
    )
    def test_empty_keys_are_rejected(self, store: InMemoryCheckpointStore, call: object) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            call(store)  # type: ignore[operator]

    def test_explicit_gc_on_success(self) -> None:
        assert InMemoryCheckpointStore(retention="gc_on_success").retention == "gc_on_success"

    async def test_restart_continues_from_a_completed_run(
        self, store: InMemoryCheckpointStore
    ) -> None:
        """A second flow over the same store continues from the first run's final state."""
        from typing import Any

        from llm_gent.flow import Context, FlowFactory, verb

        @verb
        async def bump(ctx: Context[dict[str, Any]]) -> int:
            ctx.state.data["n"] = ctx.state.data.get("n", 0) + 1
            return int(ctx.state.data["n"])

        def build() -> Any:
            return FlowFactory(make_test_logger()).create(state={}).with_checkpointer(store, "c")

        assert await build().call(bump).run() == 1
        assert await build().call(bump).run(resume="restart") == 2
