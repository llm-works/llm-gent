# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CheckpointStore — content-addressed persistence for Flow histories.

A *history* is the line of commits one flow instance writes. Two ids name
it:

- ``client_flow_id`` — the agent's name for the history, supplied through
  the public API (``with_checkpointer`` / ``FlowFactory.create``). Used only
  to look the history up.
- ``flow_id`` — gent's internal identity for the history: an opaque UUID
  generated on the first save. Every object, ref and commit is keyed by
  ``flow_id``; the agent's name never enters stored objects.

The store is a Protocol with four surfaces:

- **Name map** — :meth:`get_flow_id` / :meth:`put_flow_id` map a
  ``client_flow_id`` to its ``flow_id``. At most one history per name.

- **Object store** — content-addressed put / get / has for opaque bytes,
  keyed by ``(flow_id, kind, content_hash)``. ``kind`` is one of
  ``"blob"`` / ``"tree"`` / ``"commit"``, the object triad from
  :mod:`llm_gent.flow.state.cas`. Objects are history-scoped: each
  ``flow_id`` owns its objects, so gc at history boundaries is
  self-contained. Content-addressing still holds within a history —
  identical byte payloads produce identical blob hashes, so a resume's
  reconstruction is byte-exact.

- **Ref store** — points ``(flow_id, node_path, iteration)`` at a
  commit hash. ``put_ref`` records "this history reached this commit
  at this iterate boundary"; ``resolve_ref`` returns the commit hash for
  a full or partial key (``node_path=None, iteration=None`` returns the
  latest commit across the history — the resume entry point).

- **History cleanup** — :meth:`gc_history` removes every object, ref
  and the name mapping of one ``flow_id``. The framework calls it on a
  fully successful :meth:`Flow.run` when the store's retention policy is
  ``"gc_on_success"``; the default ``"retain"`` keeps successful
  histories on disk for audit, cross-run diff, and downstream
  provenance exporters. Consumers who need explicit cleanup call
  :meth:`gc_history` themselves. A name whose history was collected
  starts a new history (new ``flow_id``) on its next save.

Every method may be declared ``def`` (returning the value directly) or
``async def`` (returning a coroutine). The framework awaits the return
value when it is awaitable; a synchronous store keeps working, an
async-native store gains first-class support without blocking the event
loop. The :func:`maybe_await` helper below is used at every call site.

Retention policy
----------------
The reference stores accept a ``retention`` argument at construction:

- ``"retain"`` (default): a successful :meth:`Flow.run` does NOT call
  :meth:`gc_history`. Provenance framing — the successful record is
  the one most often needed for audit, cross-run diff, and motif
  extraction across good runs.
- ``"gc_on_success"``: on a fully successful run the framework calls
  :meth:`gc_history`, matching the pre-CAS delete-on-clean-exit
  behavior for consumers who don't want the history to accumulate.

Halt / cancellation / unhandled exceptions preserve the history
regardless of retention so a later ``resume=True`` run can pick up.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CheckpointPolicy:
    """Governs implicit framework saves inside a Flow run.

    Two save triggers are always on and NOT gated by this policy:

    - Halt observation — :meth:`HaltSaveObserver.save_if_signaled`
      fires whenever the executor observes the halt event set at
      any of its save sites (iterate boundary, chain between-step,
      chain tail), provided a checkpointer is wired. This is the durability guarantee that makes
      ``run(resume=True)`` reach a halted history.
    - Explicit ``ctx.checkpoint()`` — the verb-level trigger fires
      regardless of policy; when the verb asks to save, we save.

    What this policy DOES gate is automatic, unconditional save
    points that would otherwise fire per composition step:

    - :attr:`on_iterate` — save at every successful iterate body
      boundary. Off by default; halt-only histories skip every
      per-iteration commit and only stamp on halt + on explicit
      calls.

    Attached to a :class:`Flow` via :meth:`Flow.with_checkpoint_policy`;
    subflows inherit the outer runtime's policy unless they attach
    their own. Instances are frozen so a policy value can be shared
    across flows without accidental mutation.

    Room to grow: additional gates for :meth:`Flow.map` per-item
    saves, branch-arm saves, or a custom predicate can land as new
    dataclass fields without changing the type of the parameter.
    """

    on_iterate: bool = False
    """Save a commit after every successful iterate body iteration.

    Default ``False``: iterate boundaries do NOT auto-save. Set
    ``True`` when each iteration boundary should be a resumable
    anchor — the consumer pays the write cost per iteration.
    """

    on_map_item: bool = False
    """Save a commit after every successful map item body + merge.

    Default ``False``: map items do NOT auto-save. Set ``True`` for
    long-running maps with expensive per-item bodies where observing
    per-item progress is valuable. Each save writes at
    ``iteration=item_index`` under the map's node_path, so distinct
    items land in distinct ref slots. Items complete in parallel;
    save order is not guaranteed to match item order.

    **Limitation:** per-item commits are currently observability-only.
    Resume does not yet skip completed items — the whole map re-runs,
    applying each item's merge again. If merge accumulates state
    (append, increment, dict update), completed items will double-count.
    Until per-item resume is implemented, callers relying on
    ``on_map_item`` should ensure their merge function is idempotent.

    Failed items (:class:`Failure`), guard-skipped items
    (:class:`Skipped`), and cancelled items do NOT save regardless of
    this flag — only successful body-plus-merge triggers the save.
    """


async def maybe_await(value: Any) -> Any:
    """Await ``value`` if awaitable; return it as-is otherwise.

    Use this helper at every checkpoint-store call site so sync and
    async stores are handled uniformly::

        result = await maybe_await(store.get_object(flow_id, "commit", h))

    Uses :func:`inspect.isawaitable`, which returns ``True`` only for
    coroutines and objects with ``__await__``. Generators and async
    generators return ``False`` and pass through unchanged — they are
    iterable, not awaitable.
    """
    if inspect.isawaitable(value):
        return await value
    return value


class CheckpointStore(Protocol):
    """Content-addressed persistence for Flow histories.

    Two surfaces on one Protocol: object store (put / get / has for
    opaque bytes keyed by content hash) and ref store (points a
    history key at a commit hash). See the module docstring for the
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
    :meth:`gc_history`. Set at store construction; the Protocol does
    not dictate the construction shape but every reference implementation
    accepts a ``retention=`` keyword argument.
    """

    # Name map

    def get_flow_id(self, client_flow_id: str) -> str | None | Awaitable[str | None]:
        """Return the ``flow_id`` of the history named ``client_flow_id``, or ``None``.

        ``None`` means no history exists under that name (never saved, or
        collected by :meth:`gc_history`).

        May be declared ``async def``.
        """
        ...

    def put_flow_id(self, client_flow_id: str, flow_id: str) -> None | Awaitable[None]:
        """Record that ``client_flow_id`` names the history ``flow_id``.

        Called once per history, by the framework, on its first save —
        after :meth:`get_flow_id` returned ``None``. Under the
        single-writer contract a name is never bound twice concurrently;
        implementations MAY raise if the name is already bound.

        May be declared ``async def``.
        """
        ...

    # Object store

    def put_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
        payload: bytes,
    ) -> None | Awaitable[None]:
        """Persist ``payload`` under ``(flow_id, kind, content_hash)``.

        Idempotent — the same ``(flow_id, kind, content_hash)`` re-put
        with the same bytes MUST be a no-op. Because ``content_hash``
        is derived from ``payload``, differing bytes under the same hash
        indicate either a hash collision or corruption; implementations
        MAY raise on that condition.

        May be declared ``async def``.
        """
        ...

    def get_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bytes | None | Awaitable[bytes | None]:
        """Return the payload for ``(flow_id, kind, content_hash)``, or ``None``.

        Absence is not an error — callers routinely probe for objects
        that may not exist yet (dedup shortcut on save).

        May be declared ``async def``.
        """
        ...

    def has_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bool | Awaitable[bool]:
        """Return ``True`` when ``(flow_id, kind, content_hash)`` exists.

        Cheaper than :meth:`get_object` when the caller only needs to
        know whether to skip a put — the dedup shortcut on save.

        May be declared ``async def``.
        """
        ...

    # Ref store

    def put_ref(
        self,
        flow_id: str,
        node_path: str,
        iteration: int,
        commit_hash: str,
    ) -> None | Awaitable[None]:
        """Point ``(flow_id, node_path, iteration)`` at ``commit_hash``.

        Idempotent overwrite: the same key re-put with a different
        commit_hash replaces the earlier record. Callers rely on this
        to record "this iterate boundary reached this commit."

        May be declared ``async def``.
        """
        ...

    def resolve_ref(
        self,
        flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None | Awaitable[str | None]:
        """Return the ``commit_hash`` for the history key, or ``None``.

        Argument combinations:

        - Both ``None`` (default) — return the latest commit across every
          ``node_path`` under ``flow_id``. This is what
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

    # History cleanup

    def gc_history(self, flow_id: str) -> None | Awaitable[None]:
        """Remove every object and ref under ``flow_id``, and its name mapping.

        Idempotent: absence is not an error. Called by :meth:`Flow.run`'s
        clean-exit path when :attr:`retention` is ``"gc_on_success"``;
        also callable directly by consumers who want to prune a
        history.

        May be declared ``async def``.
        """
        ...
