# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Checkpoint histories — the refs, names and policies of a flow's history.

A *history* is the line of commits one flow instance writes. It lives in
one repository of an :class:`appinfra.cas.Store` (``MemoryStore``,
``FileStore``, or gent's :class:`~llm_gent.flow.stores.PgStore`), named by
the agent's ``client_flow_id`` (``with_checkpoint_store(store,
client_flow_id)``). The repository holds the history's objects — snapshot
blobs and trees, and commits whose payload is
:class:`~llm_gent.flow.state.cas.CommitMeta` — and its refs:

- :data:`HEAD_REF` is the newest commit on the history's line. Each
  commit moves it by compare-and-set, so a second writer on the same
  history is detected instead of silently forking it.
- Tags are refs under ``tags/``: on a clean exit the framework commits
  the final state at :data:`END_NODE_PATH` and moves :data:`COMPLETE_TAG`
  to it. The history is complete while ``HEAD`` is that final-state
  commit; the tag keeps pointing at the last finished run's final state
  after later runs append past it. A named checkpoint
  (``ctx.checkpoint(name)``) is the tag ``tags/<name>``.

Cleanup is the repository's: ``store.repo(client_flow_id).prune(expire)``
deletes the objects no ref reaches — commits off the history's line after
a run resumed with ``resume=<name>`` or ``resume=<hash>`` committed, or
written by a process that died before moving ``HEAD`` — once they are
older than ``expire`` seconds. ``store.delete_repo(client_flow_id)``
removes the whole history; the next save starts a new one.

Retention policy
----------------
``with_checkpoint_store(..., retention=...)`` decides what a fully
successful :meth:`Flow.run` leaves behind:

- ``"retain"`` (default): the history stays, ending in the final-state
  commit. The successful record is the one most often needed for audit,
  cross-run diff, and motif extraction across good runs.
- ``"gc_on_success"``: the framework deletes the history's repository,
  for consumers who don't want histories to accumulate.

Halt / cancellation / unhandled exceptions preserve the history
regardless of retention so a later resuming run can pick up.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

from appinfra.cas import is_object_id


HEAD_REF = "HEAD"
"""Ref naming the newest commit on a history's line."""


class ConcurrentWriteError(RuntimeError):
    """A ref moved under a writer: a second writer is committing to the same history.

    A history has one writer at a time. Each commit moves :data:`HEAD_REF`
    by compare-and-set, so a concurrent writer is detected at its next
    commit instead of silently forking the history.
    """

    def __init__(self, client_flow_id: str, ref: str, expected: str | None) -> None:
        super().__init__(
            f"history {client_flow_id!r}: ref {ref!r} no longer points at {expected!r}; "
            f"another writer is committing to this history"
        )
        self.client_flow_id = client_flow_id
        self.ref = ref
        self.expected = expected


TAG_PREFIX = "tags/"
"""Prefix of every tag ref: :data:`COMPLETE_TAG` and the named checkpoints."""


COMPLETE_TAG = f"{TAG_PREFIX}complete"
"""Ref the framework moves to a history's final-state commit on clean exit.

It always points at the final state of the most recent run that finished,
including after a later run appended commits past it.
"""


RESERVED_CHECKPOINT_NAMES = frozenset({"off", "latest", "complete"})
"""Names a checkpoint cannot take: the :data:`ResumeMode` values and the
framework's own ``complete`` tag."""


def is_commit_hash(value: object) -> bool:
    """True when ``value`` has the form of a commit hash: 64 lowercase hex characters."""
    return is_object_id(value)


def checkpoint_tag(name: str) -> str:
    """Ref of the named checkpoint ``name``: ``tags/<name>``.

    Raises:
        ValueError: ``name`` is not a non-empty ``str``, is one of
            :data:`RESERVED_CHECKPOINT_NAMES`, or has the form of a commit
            hash (``run(resume=<hash>)`` takes those as commits).
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"a checkpoint name must be a non-empty str; got {name!r}")
    if name in RESERVED_CHECKPOINT_NAMES:
        raise ValueError(
            f"{name!r} cannot name a checkpoint; reserved: {sorted(RESERVED_CHECKPOINT_NAMES)}"
        )
    if is_commit_hash(name):
        raise ValueError(f"{name!r} cannot name a checkpoint; it has the form of a commit hash")
    return f"{TAG_PREFIX}{name}"


END_NODE_PATH = "$end"
"""Reserved ``node_path`` of the final-state commit written on clean exit.

Not a node id: the commit sits after the last top-level node, and the
``$`` prefix cannot collide with a hex node id. A history whose head is
at this path is complete.
"""


COMPLETION_PRODUCER = "$framework/completion"
"""``produced_by.node_id`` of the :data:`END_NODE_PATH` final-state commit.

The framework writes it — no node produced it — following the
``$external/*`` convention for producers that are not flow nodes.
"""


ResumeMode = Literal["off", "latest"]
"""How :meth:`Flow.run` starts from a checkpointed history.

- ``"off"`` — run from ``state=`` as given; new commits still append to
  the history.
- ``"latest"`` — check out the newest commit that has state (commits
  without state are skipped) and continue from it — a run that raised
  wrote no commit, so this is the last save before it: every scope comes back, and every chain, iterate and
  branch that was running continues where its cursor was — at the same
  step with the same input, in the same pass with the same carried value,
  on the same arm. Only the step that was running when the checkpoint was
  taken runs again, and a Loop that paused mid-turn resumes that turn.
  Starts from ``state=`` when the history is empty; raises
  :class:`~llm_gent.flow.history.HistoryCorrupt` on a corrupt history.

A commit hash (:func:`is_commit_hash`; ``ctx.checkpoint()`` returns the
one it wrote) checks out that commit of the history, and any other string
names a checkpoint taken with ``ctx.checkpoint(name)`` (see
:func:`checkpoint_tag`). Either way the run checks the commit out the same
way and its commits continue from there. Once the run commits, the commits
written after it leave the history's line but stay resumable by hash until
a prune of the repository deletes them; a run that fails before its first
commit leaves ``HEAD`` where it was.
"""


Retention = Literal["retain", "gc_on_success"]
"""Retention policy for successful :meth:`Flow.run` completion.

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

    Each commit holds the map's cursor (its items and completed items
    with their results): resuming from it does not run a completed item
    again and does not apply its merge again.

    Failed items (:class:`Failure`), guard-skipped items
    (:class:`Skipped`), and cancelled items do NOT save regardless of
    this flag — only successful body-plus-merge triggers the save.
    """


async def maybe_await(value: Any) -> Any:
    """Await ``value`` if awaitable; return it as-is otherwise.

    Lets a hook be declared ``def`` or ``async def``::

        await maybe_await(hook(ctx))

    Uses :func:`inspect.isawaitable`, which returns ``True`` only for
    coroutines and objects with ``__await__``. Generators and async
    generators return ``False`` and pass through unchanged — they are
    iterable, not awaitable.
    """
    if inspect.isawaitable(value):
        return await value
    return value
