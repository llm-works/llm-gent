# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""In-process :class:`CheckpointStore` — dicts in memory, gone with the process.

For tests and short-lived local runs. Payloads are kept as the bytes the
framework serialized, so state goes through the same serialize / restore
path as with the file and Postgres stores: a state that does not
round-trip fails here too.

One instance holds any number of histories and outlives the flows that
use it, so a restart is a new :class:`~llm_gent.flow.Flow` run against
the same store instance.

Methods are synchronous and never yield to the event loop, so each call
is atomic with respect to other tasks. Binds, ref writes and gc also
hold a lock, so they stay atomic when called from several threads.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..checkpoint import Kind, Retention


@dataclass
class _History:
    """Everything stored under one ``flow_id``."""

    client_flow_id: str
    """The name bound to this history; empty while written to without a binding."""
    objects: dict[tuple[Kind, str], bytes] = field(default_factory=dict)
    refs: dict[tuple[str, int], tuple[str, int]] = field(default_factory=dict)
    """``(node_path, iteration)`` → ``(commit_hash, write sequence)``."""
    tags: dict[str, str] = field(default_factory=dict)


def _require_non_empty(value: str, field_name: str) -> None:
    """Reject an empty key, as the file store does."""
    if not value:
        raise ValueError(f"{field_name} must not be empty")


class InMemoryCheckpointStore:
    """Dict-backed :class:`CheckpointStore`; see the module docstring."""

    retention: Retention

    def __init__(self, *, retention: Retention = "retain") -> None:
        """Create an empty store.

        Args:
            retention: ``"retain"`` (default) keeps a successful run's
                history; ``"gc_on_success"`` removes it on a fully
                successful :meth:`Flow.run`.
        """
        self.retention = retention
        self._names: dict[str, str] = {}
        self._histories: dict[str, _History] = {}
        self._seq = 0
        self._lock = threading.Lock()

    # --- name map ---

    def get_flow_id(self, client_flow_id: str) -> str | None:
        """Return the ``flow_id`` bound to ``client_flow_id``, or ``None``."""
        _require_non_empty(client_flow_id, "client_flow_id")
        return self._names.get(client_flow_id)

    def bind_flow_id(self, client_flow_id: str, flow_id: str) -> str:
        """Bind ``client_flow_id`` → ``flow_id`` unless already bound; return the bound id."""
        _require_non_empty(client_flow_id, "client_flow_id")
        _require_non_empty(flow_id, "flow_id")
        with self._lock:
            existing = self._names.get(client_flow_id)
            if existing is not None:
                return existing
            history = self._histories.get(flow_id)
            if history is None:
                history = self._histories[flow_id] = _History(client_flow_id)
            elif history.client_flow_id not in ("", client_flow_id):
                raise ValueError(f"flow_id {flow_id!r} already names {history.client_flow_id!r}")
            history.client_flow_id = client_flow_id
            self._names[client_flow_id] = flow_id
            return flow_id

    # --- object store ---

    def put_object(self, flow_id: str, kind: Kind, content_hash: str, payload: bytes) -> None:
        """Store ``payload``; a re-put of the same bytes is a no-op.

        Raises:
            ValueError: Different bytes arrive under an existing hash —
                a hash collision or corrupted serialization.
        """
        objects = self._history(flow_id).objects
        existing = objects.get((kind, content_hash))
        if existing is None:
            objects[(kind, content_hash)] = bytes(payload)
        elif existing != payload:
            raise ValueError(f"{kind} {content_hash} re-put with different bytes")

    def get_object(self, flow_id: str, kind: Kind, content_hash: str) -> bytes | None:
        """Return the payload stored under the key, or ``None``."""
        history = self._histories.get(flow_id)
        return None if history is None else history.objects.get((kind, content_hash))

    def has_object(self, flow_id: str, kind: Kind, content_hash: str) -> bool:
        """Return ``True`` when the object exists."""
        return self.get_object(flow_id, kind, content_hash) is not None

    # --- ref store ---

    def put_ref(self, flow_id: str, node_path: str, iteration: int, commit_hash: str) -> None:
        """Point ``(node_path, iteration)`` at ``commit_hash``; a re-put is the newest write."""
        _require_non_empty(node_path, "node_path")
        with self._lock:
            self._seq += 1
            self._history(flow_id).refs[(node_path, iteration)] = (commit_hash, self._seq)

    def resolve_ref(
        self,
        flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None:
        """Return the commit hash for the key; see :meth:`CheckpointStore.resolve_ref`.

        No ``node_path``: the newest write in the history. ``node_path``
        alone: its highest iteration. Both: the exact ref.
        """
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        history = self._histories.get(flow_id)
        if history is None or not history.refs:
            return None
        if node_path is None:
            return max(history.refs.values(), key=lambda ref: ref[1])[0]
        if iteration is not None:
            ref = history.refs.get((node_path, iteration))
            return None if ref is None else ref[0]
        under = [(it, ref) for (path, it), ref in history.refs.items() if path == node_path]
        return max(under, key=lambda entry: entry[0])[1][0] if under else None

    # --- tags ---

    def put_tag(self, flow_id: str, name: str, commit_hash: str) -> None:
        """Point tag ``name`` at ``commit_hash``, moving it if it exists."""
        _require_non_empty(name, "tag name")
        self._history(flow_id).tags[name] = commit_hash

    def resolve_tag(self, flow_id: str, name: str) -> str | None:
        """Return the commit hash tag ``name`` points at, or ``None``."""
        history = self._histories.get(flow_id)
        return None if history is None else history.tags.get(name)

    # --- history cleanup ---

    def gc_history(self, flow_id: str) -> None:
        """Remove the history and its name binding; a no-op when absent."""
        with self._lock:
            history = self._histories.pop(flow_id, None)
            if history is not None and self._names.get(history.client_flow_id) == flow_id:
                del self._names[history.client_flow_id]

    def _history(self, flow_id: str) -> _History:
        """The history a write goes to; created unbound when the caller never bound a name."""
        _require_non_empty(flow_id, "flow_id")
        return self._histories.setdefault(flow_id, _History(client_flow_id=""))
