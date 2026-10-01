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

Every method holds one lock for its whole body, so each call is atomic
with respect to other threads as well as other tasks.
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
    refs: dict[str, str] = field(default_factory=dict)
    """Ref name → commit hash."""


def _require_non_empty(**keys: str | None) -> None:
    """Reject an empty key on reads and writes alike, as the file store does."""
    for name, value in keys.items():
        if value is not None and not value:
            raise ValueError(f"{name} must not be empty")


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
        self._lock = threading.Lock()

    # --- name map ---

    def get_flow_id(self, client_flow_id: str) -> str | None:
        """Return the ``flow_id`` bound to ``client_flow_id``, or ``None``."""
        _require_non_empty(client_flow_id=client_flow_id)
        with self._lock:
            return self._names.get(client_flow_id)

    def bind_flow_id(self, client_flow_id: str, flow_id: str) -> str:
        """Bind ``client_flow_id`` → ``flow_id`` unless already bound; return the bound id."""
        _require_non_empty(client_flow_id=client_flow_id, flow_id=flow_id)
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
        _require_non_empty(flow_id=flow_id, content_hash=content_hash)
        with self._lock:
            objects = self._history(flow_id).objects
            existing = objects.get((kind, content_hash))
            if existing is None:
                objects[(kind, content_hash)] = bytes(payload)
            elif existing != payload:
                raise ValueError(f"{kind} {content_hash} re-put with different bytes")

    def get_object(self, flow_id: str, kind: Kind, content_hash: str) -> bytes | None:
        """Return the payload stored under the key, or ``None``."""
        _require_non_empty(flow_id=flow_id, content_hash=content_hash)
        with self._lock:
            history = self._histories.get(flow_id)
            return None if history is None else history.objects.get((kind, content_hash))

    def has_object(self, flow_id: str, kind: Kind, content_hash: str) -> bool:
        """Return ``True`` when the object exists."""
        return self.get_object(flow_id, kind, content_hash) is not None

    def list_objects(self, flow_id: str) -> list[tuple[Kind, str]]:
        """Return the ``(kind, content_hash)`` of every object under ``flow_id``."""
        _require_non_empty(flow_id=flow_id)
        with self._lock:
            history = self._histories.get(flow_id)
            return [] if history is None else list(history.objects)

    def delete_objects(self, flow_id: str, keys: list[tuple[Kind, str]]) -> None:
        """Delete the objects ``keys``; keys that do not exist are skipped."""
        _require_non_empty(flow_id=flow_id)
        with self._lock:
            history = self._histories.get(flow_id)
            if history is None:
                return
            for key in keys:
                history.objects.pop(key, None)

    # --- refs ---

    def get_ref(self, flow_id: str, name: str) -> str | None:
        """Return the commit hash ref ``name`` points at, or ``None``."""
        _require_non_empty(flow_id=flow_id, ref_name=name)
        with self._lock:
            history = self._histories.get(flow_id)
            return None if history is None else history.refs.get(name)

    def set_ref(self, flow_id: str, name: str, commit_hash: str, expected: str | None) -> bool:
        """Point ref ``name`` at ``commit_hash`` if it points at ``expected`` (``None``: absent)."""
        _require_non_empty(flow_id=flow_id, ref_name=name, commit_hash=commit_hash)
        with self._lock:
            refs = self._history(flow_id).refs
            if refs.get(name) != expected:
                return False
            refs[name] = commit_hash
            return True

    def list_refs(self, flow_id: str) -> dict[str, str]:
        """Return every ref under ``flow_id``: name → commit hash."""
        _require_non_empty(flow_id=flow_id)
        with self._lock:
            history = self._histories.get(flow_id)
            return {} if history is None else dict(history.refs)

    # --- history cleanup ---

    def gc_history(self, flow_id: str) -> None:
        """Remove the history and its name binding; a no-op when absent."""
        _require_non_empty(flow_id=flow_id)
        with self._lock:
            history = self._histories.pop(flow_id, None)
            if history is not None and self._names.get(history.client_flow_id) == flow_id:
                del self._names[history.client_flow_id]

    def _history(self, flow_id: str) -> _History:
        """The history a write goes to, created unbound if new; the caller holds the lock."""
        return self._histories.setdefault(flow_id, _History(client_flow_id=""))
