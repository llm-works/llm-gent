# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Composition-graph node types and per-run environment.

Internal to :mod:`llm_gent.flow`: the private dataclasses (:class:`_Node`,
:class:`_Branch`, :class:`_Iterate`, :class:`_Map`, :class:`_RunEnv`), and
the type aliases used by the fluent builder's callback slots. Three public
symbols are routed through this module as well: :class:`Failure`, the
sentinel returned in place of a failed item by :meth:`Flow.map` when
``strict=False``; :class:`Skipped`, the sentinel returned in place of an
item whose :meth:`Flow.guard` predicate returned falsy; and :data:`UNSET`
(with its :class:`Unset` type), the "no value here" sentinel used by
:meth:`Flow.run`'s ``state=`` default and by rescue policies'
``pending_input`` positional.

Depends only on :mod:`.context` and :mod:`.state`; :class:`Flow` is
referenced solely inside string-form annotations (via
``from __future__ import annotations``), so this module imports cleanly
without pulling in :mod:`.flow`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from appinfra.log import Logger

from ..core.budget import Tracker
from .context import Context
from .state import State, StateFactory


if TYPE_CHECKING:
    from .flow import Flow
    from .state.saia_turn import PendingSaiaTurns, ResumeSaiaTurns

from .checkpoint import CheckpointPolicy, CheckpointStore


class Unset:
    """Singleton sentinel type used by :data:`UNSET`.

    Distinct from ``None`` — appears where ``None`` is a legitimate value
    that must be distinguished from "no value was supplied here". The
    canonical comparison is identity (``x is UNSET``); ``isinstance(x, Unset)``
    works too and is what tooling checks against the ``Any | Unset`` union
    in :type:`RescuePolicy`.
    """

    _instance: Unset | None = None

    def __new__(cls) -> Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final[Unset] = Unset()
"""Sentinel meaning "no value here" — distinct from ``None``.

Surfaces in two places on the public API:

- :meth:`Flow.run` — ``state=UNSET`` is the sentinel default that resolves
  to the flow's construction ``state`` (or a fresh empty ``dict``); passing
  ``state=None`` explicitly is honored as "payload is ``None``".
- :type:`RescuePolicy` — the ``pending_input`` positional is :data:`UNSET`
  when the failing node is the chain's first node and :meth:`Flow.run` was
  called with no positional argument.
"""


RescuePolicy = Callable[[BaseException, Any, Context[Any]], Any]
"""Failure hook: ``(exception, pending_input, ctx) -> fallback``. May be async.

``pending_input`` is the value that would have been passed into the failing
node (post ``project=`` if one was set) — carried through so a rescue can
fall back to a prior result without a preceding ``.after``-hook stash. When
the failing node is the chain's first node and :meth:`Flow.run` had no
positional argument, ``pending_input`` is :data:`UNSET`.
"""

AfterHook = Callable[[Any, Context[Any]], Any]
"""Success hook: ``(result, ctx) -> None`` (return value ignored). May be async."""

ProjectFn = Callable[[Any], Any]
"""Data-flow projection: transforms the previous node's result into the next input."""

WhenFn = Callable[[Any, Context[Any]], Any]
"""Branch predicate: ``(prev_result, ctx) -> bool``. May be async."""

UntilFn = Callable[[Any, Context[Any]], Any]
"""Iterate stop predicate: ``(result, ctx) -> bool``. May be async.

``result`` is the last iteration's body return value; ``ctx.state`` carries
mutations made during the body. Either signal — or both — can drive
termination.
"""

ItemsFn = Callable[[Any, Context[Any]], Any]
"""Map item source: ``(prev_result, ctx) -> iterable``. May be async. Consumed eagerly to a list."""

AggregateFn = Callable[[list[Any]], Any]
"""Map result reducer: ``list[R] -> R'``. May be async. If omitted, .map returns the list as-is."""

GuardFn = Callable[[Any, Context[Any]], Any]
"""Map per-item skip predicate: ``(item, ctx) -> bool``. May be async.

Falsy return skips the item; a :class:`Skipped` sentinel lands in that
position of the result list. The predicate runs after per-item state
projection so it can read ``ctx.state``.
"""

OnErrorFn = Callable[[BaseException, Any, Context[Any]], Any]
"""Map per-item error hook: ``(exception, item, ctx) -> None``. Return value ignored.

Fires in both ``strict`` modes for side-effect narration (logging,
tracing). Does not alter control flow — in ``strict=True`` the exception
still propagates after ``on_error`` returns; in ``strict=False`` the item
is still replaced by a :class:`Failure` sentinel.
"""

OnItemCompleteFn = Callable[[Any, Any, Context[Any]], Any]
"""Map per-item completion hook: ``(item, outcome, ctx) -> None``. Return value ignored.

Fires exactly once per item after it reaches a terminal state, for
observability (progress, streaming, adaptive throttling). ``outcome``
is the value that lands in the map's result list: the body's return
for successful items, :class:`Failure` for items whose body raised
(both ``strict`` modes), or :class:`Skipped` for guard- or halt-skipped
items. Success-path hooks fire after per-item merge; ``ctx.state``
is the child state, and the merged parent is reachable via
``ctx.state.root()`` when projection is used.

Does not fire on :class:`asyncio.CancelledError`; cancellation
propagates unconditionally. A hook exception is logged and swallowed
so a broken observer never masks the item outcome — mirrors
:type:`OnErrorFn`.
"""

StateProject = Callable[[Any], Any]
"""Scoped-state projection: ``(parent_state) -> child_state``. May be async.

Runs once around a :meth:`Flow.call` subflow, once around a :meth:`Flow.iterate`
(before the first iteration), and once per item for :meth:`Flow.map`.
"""

StateMerge = Callable[[Any, Any], Any]
"""Scoped-state merge: ``(parent_state, child_state) -> None``. May be async.

Runs only when the isolated block completes successfully. Return value is
ignored — mutate ``parent_state`` in place.
"""


@dataclass(frozen=True)
class Failure:
    """Placeholder for a failed item in :meth:`Flow.map` when ``strict=False``.

    Exposes the raised exception and the input item that produced it so
    downstream aggregators can partition successes from failures without
    losing either.
    """

    exception: BaseException
    """The exception the item's subflow raised (never :class:`asyncio.CancelledError`)."""

    item: Any
    """The input item whose subflow run failed."""


@dataclass(frozen=True)
class Skipped:
    """Placeholder for an item whose :meth:`Flow.guard` predicate returned falsy.

    Occupies the same positional slot in :meth:`Flow.map`'s result list
    that a successful or failed item would, so aggregators can partition
    ``list[R | Failure | Skipped]`` by isinstance without losing position.
    """

    item: Any
    """The input item that was gated out before the map body ran."""


@dataclass(frozen=True)
class _ResumeReplay:
    """Threaded through the executor when :meth:`Flow.run` resumes.

    ``remaining_path`` is the ancestor chain of node IDs still to match
    on descent — a tuple of content-addressed hashes assembled from
    root to the save-point iterate (the leaf ID is included). Each
    Flow entry pre-scans its chain-step ids: exactly one must equal
    ``remaining_path[0]`` (the on-path descent parent), or the entry
    raises structural-change immediately — no chain step runs at a
    level whose path head is unreachable. On the matched step's
    descent, the head is popped and the tail is threaded into the
    child Flow; every off-path sibling descent threads ``None`` (its
    subtree cannot contain the leaf). When the matched step is itself
    the save-point iterate (``len(remaining_path) == 1``), the iterate
    consumes the replay and fast-forwards to ``iteration``.

    ``full_path`` is the un-popped path from root to leaf, kept
    verbatim across descent for triage — a pre-scan raise at depth N
    can still show the full ancestor chain the checkpoint was written
    against.

    ``child_state_data`` carries the innermost scoped state from the
    checkpoint tree. When the target iterate is reached, this data is
    used instead of projecting fresh — restoring child mutations that
    occurred before the checkpoint was saved.

    ``intermediate_scope_data`` carries the middle-scope payloads —
    every scope between root and leaf. Consumed head-first at each
    scope-creating descent along the replay path: a ``.call(state=)``
    or ``.iterate(state=)`` on the path pops the first entry and uses
    it as the child scope, in place of re-projecting via the state
    factory. Empty when the checkpointed stack was only root + leaf.
    """

    remaining_path: tuple[str, ...]
    full_path: tuple[str, ...] = ()
    iteration: int = 0
    child_state_data: Any = None
    intermediate_scope_data: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _RunEnv:
    """Per-run environment threaded through the execution helpers.

    Bundles the runtime flow (factory + saia cache + logger source), the
    currently active :class:`State`, and any ambient halt event or budget
    tracker so helpers do not each need to carry them as separate positional
    arguments. ``lg`` is cached off ``runtime`` at the top of
    :meth:`Flow.run` for brevity in the debug/warning call sites. ``halt``
    is the ambient :class:`asyncio.Event` attached via :meth:`Flow.with_halt`
    (or inherited from the outer runtime); ``None`` when no halt is in
    scope. ``budget`` is the ambient session tracker attached via
    :meth:`Flow.with_budget` (or inherited); ``None`` when no budget is in
    scope.

    Composition-graph position is threaded via a pair of content-addressed
    hashes: ``chain_context`` is the hash used to compute this Flow's own
    chain-step node IDs, and ``ancestor_chain`` is the tuple of ancestor
    ``_Node`` IDs from the run's root down to the ``_Node`` whose descent
    entered this Flow. Extended pairwise on every subflow / branch-arm /
    iterate-body descent. The pair is what the checkpoint layer walks to
    address any point in the composition tree.
    """

    runtime: Flow
    state: State[Any]
    lg: Logger
    halt: asyncio.Event | None = None
    budget: Tracker | None = None
    checkpointer: CheckpointStore | None = None
    client_flow_id: str | None = None
    chain_context: str = ""
    ancestor_chain: tuple[str, ...] = ()
    replay: _ResumeReplay | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)

    @property
    def pending_saia_turns(self) -> PendingSaiaTurns:
        """Typed accessor for the runtime's pending SAIA turn container.

        Callers use this instead of reaching through
        ``env.runtime._pending_saia_turns``. The container itself lives
        on the top-level Flow (which is what ``runtime`` points at);
        this property is the typed public interface across the module
        boundary.
        """
        return self.runtime._pending_saia_turns

    @property
    def resume_saia_turns(self) -> ResumeSaiaTurns:
        """Typed accessor for the runtime's resume SAIA turn container.

        Companion to :attr:`pending_saia_turns` on the read side.
        """
        return self.runtime._resume_saia_turns


@dataclass
class _Branch:
    """Composition-graph node: run one of two subflows based on a predicate."""

    when: WhenFn
    then_flow: Flow
    else_flow: Flow | None


@dataclass
class _Iterate:
    """Composition-graph node: iterate a subflow until a bound or predicate fires."""

    body: Flow
    until: UntilFn | None
    max_iters: int | None
    deadline: float | None
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None
    state_factory: StateFactory[Any] | None = None


@dataclass
class _Map:
    """Composition-graph node: fan out a subflow over items and (optionally) reduce."""

    body: Flow
    items: ItemsFn | None
    aggregate: AggregateFn | None
    strict: bool
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None
    guard: GuardFn | None = None
    on_error: OnErrorFn | None = None
    on_item_complete: OnItemCompleteFn | None = None
    max_concurrency: int | None = None
    state_factory: StateFactory[Any] | None = None


@dataclass
class _Node:
    """One step in a Flow composition chain.

    Identity is content-addressed but derived lazily, not stored: the
    executor computes each node's runtime ``node_id`` from the enclosing
    :attr:`_RunEnv.chain_context` plus the node's local key
    (kind + target qualname + chain position) on descent. The hash chain
    from root gives every node in the composition tree a globally-unique
    identifier the checkpoint layer uses; see
    :func:`llm_gent.flow.flow._compute_node_id`.
    """

    target: Any
    project: ProjectFn | None = None
    rescue: RescuePolicy | None = None
    after: AfterHook | None = None
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None
    state_factory: StateFactory[Any] | None = None
