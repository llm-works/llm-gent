# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CheckpointStore — content-addressed persistence for Flow trajectories.

The store is a two-surface Protocol:

- **Object store** — content-addressed put / get / has for opaque bytes,
  keyed by ``(client_flow_id, kind, content_hash)``. ``kind`` is one of
  ``"blob"`` / ``"tree"`` / ``"commit"``, the object triad from
  :mod:`llm_gent.flow.state.cas`. Objects are trajectory-scoped: each
  ``client_flow_id`` owns its objects, so gc at trajectory boundaries is
  self-contained. Content-addressing still holds within a trajectory —
  identical byte payloads produce identical blob hashes, so a resume's
  reconstruction is byte-exact.

- **Ref store** — points ``(client_flow_id, node_path, iteration)`` at a
  commit hash. ``put_ref`` records "this trajectory reached this commit
  at this iterate boundary"; ``resolve_ref`` returns the commit hash for
  a full or partial key (``node_path=None, iteration=None`` returns the
  latest commit across the trajectory — the resume entry point).

- **Trajectory cleanup** — :meth:`gc_trajectory` removes every object
  and ref under one ``client_flow_id``. The framework calls it on a
  fully successful :meth:`Flow.run` when the store's retention policy is
  ``"gc_on_success"``; the default ``"retain"`` keeps successful
  trajectories on disk for audit, cross-run diff, and downstream
  provenance exporters. Consumers who need explicit cleanup call
  :meth:`gc_trajectory` themselves.

Every method may be declared ``def`` (returning the value directly) or
``async def`` (returning a coroutine). The framework awaits the return
value when it is awaitable; a synchronous store keeps working, an
async-native store gains first-class support without blocking the event
loop. The :func:`maybe_await` helper below is used at every call site.

Retention policy
----------------
The reference stores accept a ``retention`` argument at construction:

- ``"retain"`` (default): a successful :meth:`Flow.run` does NOT call
  :meth:`gc_trajectory`. Provenance framing — the successful record is
  the one most often needed for audit, cross-run diff, and motif
  extraction across good runs.
- ``"gc_on_success"``: on a fully successful run the framework calls
  :meth:`gc_trajectory`, matching the pre-CAS delete-on-clean-exit
  behavior for consumers who don't want the trajectory to accumulate.

Halt / cancellation / unhandled exceptions preserve the trajectory
regardless of retention so a later ``resume=True`` run can pick up.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, Literal, Protocol


Kind = Literal["blob", "tree", "commit"]
"""Object kind — one of the three CAS object types.

Values match :mod:`llm_gent.flow.state.cas`:
- ``"blob"`` — serialized scope-state payload bytes.
- ``"tree"`` — canonical JSON of ordered ``TreeEntry`` list.
- ``"commit"`` — canonical JSON of ``(root_tree_hash, parent_hashes,
  meta)``.
"""


Retention = Literal["retain", "gc_on_success"]
"""Store retention policy for successful :meth:`Flow.run` completion.

See the module docstring's retention section for semantics.
"""


async def maybe_await(value: Any) -> Any:
    """Await ``value`` if awaitable; return it as-is otherwise.

    Use this helper at every checkpoint-store call site so sync and
    async stores are handled uniformly::

        result = await maybe_await(store.get_object("commit", h))

    Uses :func:`inspect.isawaitable`, which returns ``True`` only for
    coroutines and objects with ``__await__``. Generators and async
    generators return ``False`` and pass through unchanged — they are
    iterable, not awaitable.
    """
    if inspect.isawaitable(value):
        return await value
    return value


class CheckpointStore(Protocol):
    """Content-addressed persistence for Flow trajectories.

    Two surfaces on one Protocol: object store (put / get / has for
    opaque bytes keyed by content hash) and ref store (points a
    trajectory key at a commit hash). See the module docstring for the
    object model and retention policy.

    Each method may be declared ``def`` (returning its value directly)
    or ``async def`` (returning a coroutine). The framework awaits the
    return value when it is awaitable.

    Consumers implementing this Protocol declare a construction-time
    ``retention: Retention`` argument (default ``"retain"``) and expose
    it via :attr:`retention` for the framework to consult on the clean-
    exit path.
    """

    retention: Retention
    """Store's retention policy for successful runs.

    Read by :meth:`Flow.run`'s clean-exit path to decide whether to call
    :meth:`gc_trajectory`. Set at store construction; the Protocol does
    not dictate the construction shape but every reference implementation
    accepts a ``retention=`` keyword argument.
    """

    # Object store

    def put_object(
        self,
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
        payload: bytes,
    ) -> None | Awaitable[None]:
        """Persist ``payload`` under ``(client_flow_id, kind, content_hash)``.

        Idempotent — the same ``(client_flow_id, kind, content_hash)`` re-
        put with the same bytes MUST be a no-op. Because ``content_hash``
        is derived from ``payload``, differing bytes under the same hash
        indicate either a hash collision or corruption; implementations
        MAY raise on that condition.

        May be declared ``async def``.
        """
        ...

    def get_object(
        self,
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bytes | None | Awaitable[bytes | None]:
        """Return the payload for ``(client_flow_id, kind, content_hash)``, or ``None``.

        Absence is not an error — callers routinely probe for objects
        that may not exist yet (dedup shortcut on save).

        May be declared ``async def``.
        """
        ...

    def has_object(
        self,
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bool | Awaitable[bool]:
        """Return ``True`` when ``(client_flow_id, kind, content_hash)`` exists.

        Cheaper than :meth:`get_object` when the caller only needs to
        know whether to skip a put — the dedup shortcut on save.

        May be declared ``async def``.
        """
        ...

    # Ref store

    def put_ref(
        self,
        client_flow_id: str,
        node_path: str,
        iteration: int,
        commit_hash: str,
    ) -> None | Awaitable[None]:
        """Point ``(client_flow_id, node_path, iteration)`` at ``commit_hash``.

        Idempotent overwrite: the same key re-put with a different
        commit_hash replaces the earlier record. Callers rely on this
        to record "this iterate boundary reached this commit."

        May be declared ``async def``.
        """
        ...

    def resolve_ref(
        self,
        client_flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None | Awaitable[str | None]:
        """Return the ``commit_hash`` for the trajectory key, or ``None``.

        Argument combinations:

        - Both ``None`` (default) — return the latest commit across every
          ``node_path`` under ``client_flow_id``. This is what
          :meth:`Flow.run` ``resume=True`` calls to find the resume
          entry point.
        - ``node_path`` set, ``iteration=None`` — latest iteration under
          that specific ``node_path``.
        - Both set — exact record, or ``None``.

        ``iteration`` without ``node_path`` is invalid — implementations
        raise :class:`ValueError`.

        "Latest" is by save order — implementations use their
        write-order signal (Postgres row created_at, JsonFile ref
        directory mtime).

        May be declared ``async def``.
        """
        ...

    # Trajectory cleanup

    def gc_trajectory(self, client_flow_id: str) -> None | Awaitable[None]:
        """Remove every object and ref under ``client_flow_id``.

        Idempotent: absence is not an error. Called by :meth:`Flow.run`'s
        clean-exit path when :attr:`retention` is ``"gc_on_success"``;
        also callable directly by consumers who want to prune a
        trajectory.

        May be declared ``async def``.
        """
        ...
