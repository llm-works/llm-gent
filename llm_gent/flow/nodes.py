# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Composition-graph node types and per-run environment.

Internal to :mod:`llm_gent.flow`: the private dataclasses (:class:`_Node`,
:class:`_Branch`, :class:`_Loop`, :class:`_Map`, :class:`_RunEnv`), and the
type aliases used by the fluent builder's callback slots. Three public
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from appinfra.log import Logger

from .context import Context
from .state import State


if TYPE_CHECKING:
    from .flow import Flow


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


RescuePolicy = Callable[[BaseException, Any, Context], Any]
"""Failure hook: ``(exception, pending_input, ctx) -> fallback``. May be async.

``pending_input`` is the value that would have been passed into the failing
node (post ``project=`` if one was set) — carried through so a rescue can
fall back to a prior result without a preceding ``.after``-hook stash. When
the failing node is the chain's first node and :meth:`Flow.run` had no
positional argument, ``pending_input`` is :data:`UNSET`.
"""

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

GuardFn = Callable[[Any, Context], Any]
"""Map per-item skip predicate: ``(item, ctx) -> bool``. May be async.

Falsy return skips the item; a :class:`Skipped` sentinel lands in that
position of the result list. The predicate runs after per-item state
projection so it can read ``ctx.state``.
"""

OnErrorFn = Callable[[BaseException, Any, Context], Any]
"""Map per-item error hook: ``(exception, item, ctx) -> None``. Return value ignored.

Fires in both ``strict`` modes for side-effect narration (logging,
tracing). Does not alter control flow — in ``strict=True`` the exception
still propagates after ``on_error`` returns; in ``strict=False`` the item
is still replaced by a :class:`Failure` sentinel.
"""

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
    guard: GuardFn | None = None
    on_error: OnErrorFn | None = None
    max_concurrency: int | None = None
    halt_event: asyncio.Event | None = None


@dataclass
class _Node:
    """One step in a Flow composition chain."""

    target: Any
    project: ProjectFn | None = None
    rescue: RescuePolicy | None = None
    after: AfterHook | None = None
    state_fn: StateProject | None = None
    merge_fn: StateMerge | None = None
