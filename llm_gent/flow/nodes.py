"""Composition-graph node types and per-run environment.

Internal to :mod:`llm_gent.flow`: the private dataclasses (:class:`_Node`,
:class:`_Branch`, :class:`_Loop`, :class:`_Map`, :class:`_RunEnv`), the
sentinel :data:`_UNSET`, the type aliases used by the fluent builder's
callback slots, and the one public export routed through this module —
:class:`Failure`, the sentinel returned in place of a failed item by
:meth:`Flow.map` when ``strict=False``.

Depends only on :mod:`.context` and :mod:`.state`; :class:`Flow` is
referenced solely inside string-form annotations (via
``from __future__ import annotations``), so this module imports cleanly
without pulling in :mod:`.flow`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from .context import Context
from .state import State


if TYPE_CHECKING:
    from .flow import Flow


RescuePolicy = Callable[[BaseException, Context], Any]
"""Failure hook: ``(exception, ctx) -> fallback``. May be async."""

AfterHook = Callable[[Any, Context], Any]
"""Success hook: ``(result, ctx) -> None`` (return value ignored). May be async."""

ProjectFn = Callable[[Any], Any]
"""Data-flow projection: transforms the previous node's result into the next input."""

WhenFn = Callable[[Any, Context], Any]
"""Branch predicate: ``(prev_result, ctx) -> bool``. May be async."""

UntilFn = Callable[[Context], Any]
"""Loop stop predicate: ``(ctx) -> bool``. May be async. State lives on ``ctx.state``."""

ItemsFn = Callable[[Any, Context], Any]
"""Map item source: ``(prev_result, ctx) -> iterable``. May be async. Consumed eagerly to a list."""

AggregateFn = Callable[[list[Any]], Any]
"""Map result reducer: ``list[R] -> R'``. May be async. If omitted, .map returns the list as-is."""

StateProject = Callable[[Any], Any]
"""Scoped-state projection: ``(parent_state) -> child_state``. May be async.

Runs once around a :meth:`Flow.call` subflow, once around a :meth:`Flow.loop`
(before the first iteration), and once per item for :meth:`Flow.map`.
"""

StateMerge = Callable[[Any, Any], Any]
"""Scoped-state merge: ``(parent_state, child_state) -> None``. May be async.

Runs only when the isolated block completes successfully. Return value is
ignored — mutate ``parent_state`` in place.
"""


_UNSET: Any = object()
"""Sentinel used to distinguish "kwarg omitted" from "kwarg = None"."""


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
class _RunEnv:
    """Per-run environment threaded through the execution helpers.

    Bundles the runtime flow (factory + saia cache + logger source) and the
    currently active :class:`State` so helpers do not each need to carry
    them as separate positional arguments. ``lg`` is cached off ``runtime``
    at the top of :meth:`Flow.run` for brevity in the debug/warning call sites.
    """

    runtime: Flow
    state: State
    lg: Logger


@dataclass
class _Branch:
    """Composition-graph node: run one of two subflows based on a predicate."""

    when: WhenFn
    then_flow: Flow
    else_flow: Flow | None


@dataclass
class _Loop:
    """Composition-graph node: iterate a subflow until a bound or predicate fires."""

    body: Flow
    until: UntilFn | None
    max_iters: int | None
    deadline: float | None
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None


@dataclass
class _Map:
    """Composition-graph node: fan out a subflow over items and (optionally) reduce."""

    body: Flow
    items: ItemsFn | None
    aggregate: AggregateFn | None
    strict: bool
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None


@dataclass
class _Node:
    """One step in a Flow composition chain."""

    target: Any
    project: ProjectFn | None = None
    rescue: RescuePolicy | None = None
    after: AfterHook | None = None
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None
