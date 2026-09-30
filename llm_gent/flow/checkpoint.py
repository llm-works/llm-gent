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

- **Name map** — :meth:`get_flow_id` / :meth:`bind_flow_id` map a
  ``client_flow_id`` to its ``flow_id``. At most one history per name;
  binding is atomic bind-if-absent, so concurrent first saves agree.

- **Object store** — content-addressed put / get / has for opaque bytes,
  keyed by ``(flow_id, kind, content_hash)``. ``kind`` is one of
  ``"blob"`` / ``"tree"`` / ``"commit"``, the object triad from
  :mod:`llm_gent.flow.state.cas`. Objects are history-scoped: each
  ``flow_id`` owns its objects, so gc at history boundaries is
  self-contained. Content-addressing still holds within a history —
  identical byte payloads produce identical blob hashes, so a resume's
  reconstruction is byte-exact.

- **Refs** — named pointers to commits, as in git. :meth:`get_ref` reads
  one; :meth:`set_ref` moves it with compare-and-set: the write lands only
  while the ref still points where the writer expects, so a second writer
  on the same history is detected instead of silently forking it.
  :data:`HEAD_REF` is the newest commit of the history, where every new
  commit is parented. Tags are refs under ``tags/``: on a clean exit the
  framework commits the final state at :data:`END_NODE_PATH` and moves
  :data:`COMPLETE_TAG` to it. The history is complete while ``HEAD`` is
  that final-state commit; the tag keeps pointing at the last finished
  run's final state after later runs append past it. A named checkpoint
  (``ctx.checkpoint(name)``) is the tag ``tags/<name>``.

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
regardless of retention so a later resuming run can pick up.
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


HEAD_REF = "HEAD"
"""Ref naming a history's newest commit: the parent of the next commit and
the commit ``resume="latest"`` starts its walk from."""


class ConcurrentWriteError(RuntimeError):
    """A ref moved under a writer: a second writer is committing to the same history.

    A history has one writer at a time. Every commit moves :data:`HEAD_REF`
    by compare-and-set from its parent, so a second writer is detected at
    its next commit instead of silently forking the history.
    """

    def __init__(self, client_flow_id: str, ref: str, expected: str | None) -> None:
        super().__init__(
            f"history {client_flow_id!r}: ref {ref!r} no longer points at {expected!r}; "
            f"another writer is committing to this history"
        )
        self.client_flow_id = client_flow_id
        self.ref = ref
        self.expected = expected


COMPLETE_TAG = "tags/complete"
"""Ref the framework moves to a history's final-state commit on clean exit.

It always points at the final state of the most recent run that finished,
including after a later run appended commits past it.
"""


RESERVED_CHECKPOINT_NAMES = frozenset({"off", "latest", "complete"})
"""Names a checkpoint cannot take: the :data:`ResumeMode` values and the
framework's own ``complete`` tag."""


def checkpoint_tag(name: str) -> str:
    """Ref of the named checkpoint ``name``: ``tags/<name>``.

    Raises:
        ValueError: ``name`` is not a non-empty ``str``, or is one of
            :data:`RESERVED_CHECKPOINT_NAMES`.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"a checkpoint name must be a non-empty str; got {name!r}")
    if name in RESERVED_CHECKPOINT_NAMES:
        raise ValueError(
            f"{name!r} cannot name a checkpoint; reserved: {sorted(RESERVED_CHECKPOINT_NAMES)}"
        )
    return f"tags/{name}"


END_NODE_PATH = "$end"
"""Reserved ``node_path`` of the final-state commit written on clean exit.

Not a node id: the commit sits after the last top-level node, and the
``$`` prefix cannot collide with a blake2b hex node id. A history whose
head is at this path is complete.
"""


FAILED_NODE_PATH = "$failed"
"""Reserved ``node_path`` of the commit written when a top-level run raises.

Records the root state at the moment of failure (``outcome="failed"``)
for inspection. It is never a starting point: that state may be
half-updated, so ``resume="latest"`` skips such commits and continues
from the last commit before them.
"""


COMPLETION_PRODUCER = "$framework/completion"
"""``produced_by.node_id`` of the :data:`END_NODE_PATH` final-state commit."""


FAILURE_PRODUCER = "$framework/failure"
"""``produced_by.node_id`` of the :data:`FAILED_NODE_PATH` commit.

The framework writes both itself — no node produced them — following the
``$external/*`` convention for producers that are not flow nodes.
"""


ResumeMode = Literal["off", "latest"]
"""How :meth:`Flow.run` starts from a checkpointed history.

- ``"off"`` — run from ``state=`` as given; new commits still append to
  the history.
- ``"latest"`` — check out the newest commit that has usable state
  (halted, ok or final; ``$failed`` and stateless commits are skipped) and
  continue from it: every scope comes back, and every chain, iterate and
  branch that was running continues where its cursor was — at the same
  step with the same input, in the same pass with the same carried value,
  on the same arm. Only the step that was running when the checkpoint was
  taken runs again, and a Loop that paused mid-turn resumes that turn.
  Starts from ``state=`` when the history is empty; raises
  :class:`~llm_gent.flow.history.HistoryCorrupt` on a corrupt history.

Any other string names a checkpoint taken with ``ctx.checkpoint(name)``
(see :func:`checkpoint_tag`): the run checks it out the same way and
moves ``HEAD`` back to it, so its commits continue from there.
"""


Retention = Literal["retain", "gc_on_success"]
"""Store retention policy for successful :meth:`Flow.run` completion.

See the module docstring's retention section for semantics.
"""


@dataclass(frozen=True)
class CheckpointPolicy:
    """Governs implicit framework saves inside a Flow run.

    Two save triggers are always on and NOT gated by this policy:

    - The halt checkpoint — when the run's halt stops it, :meth:`Flow.run`
      commits the run's position once everything has stopped, provided a
      checkpointer is wired. This is the durability guarantee that makes
      ``run(resume="latest")`` reach a halted history.
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
    per-item progress is valuable. Each save's commit records
    ``iteration=item_index`` under the map's node_path. Items complete
    in parallel; save order is not guaranteed to match item order.

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

    Name map, object store (put / get / has for opaque bytes keyed by
    content hash), named refs with compare-and-set, and cleanup on one
    Protocol. See the module docstring for the
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

    def bind_flow_id(self, client_flow_id: str, flow_id: str) -> str | Awaitable[str]:
        """Bind ``client_flow_id`` to ``flow_id`` unless already bound; return the bound id.

        Called by the framework on a history's first save, with a fresh
        ``flow_id``. MUST be atomic: of concurrent binds of one name exactly
        one wins, and every caller gets the winner's ``flow_id`` back (the
        losers then write into the winner's history). A ``flow_id`` names
        at most one ``client_flow_id``; binding it to a second name raises
        :class:`ValueError`.

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

    # Refs

    def get_ref(self, flow_id: str, name: str) -> str | None | Awaitable[str | None]:
        """Return the commit hash ref ``name`` points at, or ``None`` when it does not exist.

        Ref names are non-empty strings such as :data:`HEAD_REF` or
        ``"tags/complete"``; a ``/`` is part of the name, not a hierarchy.

        May be declared ``async def``.
        """
        ...

    def set_ref(
        self,
        flow_id: str,
        name: str,
        commit_hash: str,
        expected: str | None,
    ) -> bool | Awaitable[bool]:
        """Point ref ``name`` at ``commit_hash`` if it currently points at ``expected``.

        ``expected=None`` means the ref must not exist yet. Returns
        ``True`` when the write landed, ``False`` when the ref points
        elsewhere — nothing is written then. MUST be atomic: of concurrent
        writers passing the same ``expected``, at most one succeeds.

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
