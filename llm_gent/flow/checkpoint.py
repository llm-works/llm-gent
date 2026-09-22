# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CheckpointStore — persistence seam for :class:`Flow` pause/resume.

Flow-level composition checkpointer. Distinct from
:class:`llm_gent.flow.LoopCheckpointStore`, which persists SAIA per-turn
state inside a single :class:`Loop` dispatch. This Protocol persists the
composition-graph state at ``.iterate`` boundaries, keyed by an agent-owned
identifier the consumer supplies via :meth:`Flow.with_checkpointer`.

Record identity
---------------
Each save is uniquely identified by ``(client_flow_id, node_path,
iteration)`` — a record is one iterate's one iteration under one
trajectory. ``node_path`` is the ``/``-joined content-addressed node id
chain from the run root down to the saving iterate (e.g.
``"cafebabe/deadbeef"`` for an inner iterate nested one level). Two
iterates in one chain, nested iterates, or an iterate inside a
``.map`` body each have distinct ``node_path`` values and never collide.
Load-latest returns the most recently saved record across every
``node_path`` under a ``client_flow_id`` — the resume entry consults
``metadata_json`` for the exact save-point.

Snapshot shape
--------------
Save writes two JSON-compatible dicts per iteration:

- ``state_json`` — the framework-owned payload snapshot. Its shape is
  internal to :mod:`llm_gent.flow`; consumers do not construct it. Its
  contract is that on resume the framework reads it back and, if the flow
  was constructed with a ``state_factory=``, calls
  ``state_factory.restore(state_json['data'])`` to reconstruct the user
  payload. Plain-``dict`` payloads round-trip as-is.
- ``metadata_json`` — framework-owned bookkeeping (``path``,
  ``iteration``) that the resume path consults to locate the
  save-point iterate in the composition graph. Also JSON-compatible.

Both are handed to :meth:`save_checkpoint` inline on the event loop — an
implementation performing disk or database I/O should avoid blocking
(back a non-blocking driver, or run the blocking work off the loop
thread). Every method may be declared ``def`` (returning the result
directly) or ``async def`` (returning a coroutine). The framework
awaits the return value when it is awaitable; a synchronous store
returning ``None`` keeps working unchanged. A store using
:func:`asyncio.to_thread` internally declares its methods ``async
def`` and awaits the ``to_thread`` call there.

Save timing
-----------
The framework saves at :meth:`Flow.iterate` boundaries (once per successful
body iteration). Iteration numbering starts at 1 for the first saved
iteration within one iterate's ``node_path``. Consumers do not schedule
saves; wiring :meth:`Flow.with_checkpointer` is the entire opt-in.

Delete policy
-------------
On a fully successful :meth:`Flow.run` (no exception, no cancellation),
the framework calls :meth:`delete_checkpoint` so the resumable trajectory
does not leak past its own completion. Cancellation, halt-triggered exit,
and unhandled exceptions preserve the checkpoint so a later ``resume=True``
run can pick up.

Resume semantics
----------------
On :meth:`Flow.run` ``resume=True``:

- ``state.data`` hydrates from ``state_json``'s root ``data`` slot (via
  ``state_factory.restore`` when a ``state_factory`` is bound, else
  passthrough for plain dicts).
- The iteration counter is restored from ``metadata_json['iteration']``,
  so ``max_iters`` is an absolute cumulative bound across resumes — a
  save at iteration N with ``max_iters=M`` runs at most ``max(0, M - N)``
  further passes, and a counter that already meets the bound exits
  without re-running the body.
- Deadline is not restored — the wall clock starts fresh each run.
- Ambients (halt, budget, saia, traits, logger, checkpointer itself)
  are never serialized; they reattach from the current runtime.

Schema evolution
----------------
Framework changes to the on-disk shape of ``state_json`` /
``metadata_json`` land as additive extensions wherever possible so
prior-shape checkpoints keep resuming under a newer framework (PR 3
did this when the flat payload became a ``{data, children}`` tree).
When a non-additive change is unavoidable, it ships as an alembic
data migration on the Postgres backend
(:mod:`llm_gent.migrations`); the JSON-file backend is a dev/local
tool and old checkpoints there are expected to be discarded across a
non-additive change.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol


LoadResult = tuple[dict[str, Any], dict[str, Any]] | None
"""Return payload of :meth:`CheckpointStore.load_checkpoint`."""


class CheckpointStore(Protocol):
    """Flow-level pause/resume Protocol — persists composition-graph state.

    Records are keyed by ``(client_flow_id, node_path, iteration)`` —
    the ``client_flow_id`` is the trajectory identifier supplied at
    :meth:`Flow.with_checkpointer` time; ``node_path`` scopes the record
    to one iterate in the composition graph (see module docstring);
    ``iteration`` is the framework-managed count starting at 1 for that
    iterate's first saved iteration.

    Each method may be declared ``def`` (returning its value directly)
    or ``async def`` (returning a coroutine). The framework awaits the
    return value when it is awaitable — a synchronous store keeps
    working; an async-native store gains first-class support without
    blocking the event loop.
    """

    def save_checkpoint(
        self,
        client_flow_id: str,
        node_path: str,
        iteration: int,
        state_json: dict[str, Any],
        metadata_json: dict[str, Any],
    ) -> None | Awaitable[None]:
        """Persist one record at ``(client_flow_id, node_path, iteration)``.

        Called by the framework once per successful body iteration inside
        a ``.iterate`` block on a Flow with :meth:`Flow.with_checkpointer`
        wired. Same ``(client_flow_id, node_path, iteration)`` from a later
        save MUST replace the earlier record (idempotent overwrite —
        an iterate re-saving iteration N under a running trajectory).

        May be declared ``async def``.
        """
        ...

    def load_checkpoint(
        self,
        client_flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> LoadResult | Awaitable[LoadResult]:
        """Return the ``(state_json, metadata_json)`` pair, or ``None`` if absent.

        Argument combinations:

        - Both ``None`` (default) — return the most recently saved record
          across every ``node_path`` under ``client_flow_id``. This is what
          :meth:`Flow.run` ``resume=True`` calls.
        - ``node_path`` set, ``iteration=None`` — latest iteration under
          that specific ``node_path``.
        - Both set — exact record, or ``None``.

        ``iteration`` without ``node_path`` is invalid — stores raise
        :exc:`ValueError`.

        "Most recently saved" is by save order (Postgres uses the row's
        autoincrement id; JsonFile uses a monotonic save sequence in the
        filename).

        May be declared ``async def``.
        """
        ...

    def delete_checkpoint(self, client_flow_id: str) -> None | Awaitable[None]:
        """Remove every record under ``client_flow_id`` (all node_paths).

        Called by the framework on fully successful :meth:`Flow.run`
        completion (no exception, no cancellation). Idempotent: absence
        is not an error — stores must not raise when nothing matches.

        May be declared ``async def``.
        """
        ...
