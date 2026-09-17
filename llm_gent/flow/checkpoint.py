# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""CheckpointStore — persistence seam for :class:`Flow` pause/resume.

Flow-level composition checkpointer. Distinct from
:class:`llm_gent.flow.LoopCheckpointStore`, which persists SAIA per-turn
state inside a single :class:`Loop` dispatch. This Protocol persists the
composition-graph state at ``.iterate`` boundaries, keyed by an agent-owned
identifier the consumer supplies via :meth:`Flow.with_checkpointer`.

Snapshot shape
--------------
Save writes two JSON-compatible dicts per iteration:

- ``state_json`` — the framework-owned payload snapshot. Its shape is
  internal to :mod:`llm_gent.flow`; consumers do not construct it. Its
  contract is that on resume the framework reads it back and, if the flow
  was constructed with ``state_type=T``, calls ``T.from_dict(state_json['data'])``
  to reconstruct the user payload. Plain-``dict`` payloads round-trip as-is.
- ``metadata_json`` — checkpointer-owned metadata (a
  ``schema_version`` key for future discrimination). Also JSON-compatible.

Both are handed to :meth:`save_checkpoint` inline on the event loop — an
implementation performing disk or database I/O should avoid blocking
(e.g. wrap disk writes in :func:`asyncio.to_thread` inside the store, or
back a non-blocking driver).

Save timing
-----------
The framework saves at :meth:`Flow.iterate` boundaries (once per successful
body iteration). Iteration numbering starts at 1 for the first saved
iteration. Consumers do not schedule saves; wiring
:meth:`Flow.with_checkpointer` is the entire opt-in.

Delete policy
-------------
On a fully successful :meth:`Flow.run` (no exception, no cancellation),
the framework calls :meth:`delete_checkpoint` so the resumable trajectory
does not leak past its own completion. Cancellation, halt-triggered exit,
and unhandled exceptions preserve the checkpoint so a later ``resume=True``
run can pick up.
"""

from __future__ import annotations

from typing import Any, Protocol


class CheckpointStore(Protocol):
    """Flow-level pause/resume Protocol — persists composition-graph state.

    Consumers implement three synchronous methods keyed by an opaque
    ``client_flow_id`` chosen at :meth:`Flow.with_checkpointer` time.
    ``iteration`` is the framework-managed save index (see module docstring).

    Methods are called inline from the executor; implementations doing
    I/O should keep them non-blocking (see module docstring).
    """

    def save_checkpoint(
        self,
        client_flow_id: str,
        iteration: int,
        state_json: dict[str, Any],
        metadata_json: dict[str, Any],
    ) -> None:
        """Persist ``state_json`` + ``metadata_json`` at ``iteration`` under ``client_flow_id``.

        Called by the framework once per successful body iteration inside
        a ``.iterate`` block on a Flow with :meth:`Flow.with_checkpointer`
        wired. Same ``(client_flow_id, iteration)`` from a later save MUST
        replace the earlier record (idempotent overwrite semantics — the
        framework does not maintain iteration history itself).
        """
        ...

    def load_checkpoint(
        self,
        client_flow_id: str,
        iteration: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Return the ``(state_json, metadata_json)`` pair, or ``None`` if absent.

        ``iteration=None`` (default) returns the latest checkpoint under
        ``client_flow_id`` per the store's convention; a specific
        ``iteration`` returns exactly that record or ``None``.

        The framework calls this once at ``Flow.run(resume=True)`` start.
        """
        ...

    def delete_checkpoint(self, client_flow_id: str) -> None:
        """Remove every checkpoint under ``client_flow_id``.

        Called by the framework on fully successful :meth:`Flow.run`
        completion (no exception, no cancellation). Idempotent: absence
        is not an error — stores must not raise when nothing matches.
        """
        ...
