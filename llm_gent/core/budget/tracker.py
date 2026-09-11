# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Hierarchical cost accounting.

:class:`Tracker` is the sole accounting primitive. A tracker holds a
scope-local cap, `spent` counter, `costs_by_op` breakdown, and (when
configured) a halt event that fires when this tracker's own cap
crosses. Trackers form a tree via :meth:`Tracker.child`: costs
recorded at any level propagate up the parent chain, so an ancestor's
`spent` reflects every descendant's activity and an ancestor's halt
fires when *its* cap crosses.

Arbitrary depth is supported — session → wave → run, or any other
nesting a consumer needs. Every level has the same capabilities.

Halt-observation contract:

- Halt at tracker level N fires when N's own cap crosses.
- Cascading halt observation across scopes is a Flow-layer concern —
  a nested Flow that wants to observe both a parent and a local halt
  either uses :meth:`Flow.with_halt` inheritance (the default) or the
  consumer wires the events externally (e.g. mirror one event into
  another with a small async task).

This separation of concerns is deliberate: the tracker records and
fires; the Flow decides what to observe.
"""

from __future__ import annotations

import asyncio
import copy
import math
from typing import Any, Protocol

from appinfra.log import Logger

from .pricing import PricingProvider


class CostCallback(Protocol):
    """Callback invoked on every cost recorded at a tracker."""

    def __call__(self, cost: float, context: dict[str, Any], *, overridden: bool) -> None:
        """Handle one recorded cost event.

        ``overridden`` is True when the cost came from an explicit
        ``override_cost`` argument to :meth:`Tracker.track` rather
        than being computed by the pricing provider.
        """
        ...


class Tracker:
    """Hierarchical cost accounting.

    A root tracker is constructed directly; children come from
    :meth:`child` and share the pricing config of their root. Every
    cost recorded via :meth:`track` (or reported from a descendant)
    updates this tracker's ``spent`` and ``costs_by_op`` and, when
    this tracker's own cap crosses on that record, fires ``halt`` (if
    any) and latches ``urgent_wrapup``.

    Single-writer expectation: :meth:`_record_cost` is not
    synchronized. Concurrent :meth:`track` calls under one tracker
    tree require the caller to serialize them (run under one asyncio
    task or guard with a lock).

    Example — the substrate imposes no particular hierarchy; the
    "session → wave → run" shape below is one common pattern, not the
    only one. Consumers pick depth and names to fit their model::

        outer_halt = asyncio.Event()
        outer = Tracker(lg, pricing, budget=10.0, halt=outer_halt)

        mid_halt = asyncio.Event()
        mid = outer.child(budget=3.0, halt=mid_halt)

        leaf = mid.child(budget=1.0)
        leaf.track("some-model", input_tokens=1000, output_tokens=100)

        if leaf.urgent_wrapup:
            ...   # leaf has hit its own cap
        if mid.exceeded:
            ...   # mid crossed its $3 — mid_halt already set
    """

    def __init__(
        self,
        lg: Logger,
        pricing: PricingProvider,
        budget: float | None = None,
        *,
        parent: Tracker | None = None,
        on_cost: CostCallback | None = None,
        halt: asyncio.Event | None = None,
    ) -> None:
        """Initialize a tracker.

        Args:
            lg: appinfra Logger.
            pricing: :class:`PricingProvider` used by this tracker and
                every descendant created via :meth:`child`. Substrate
                ships :class:`PricingConfig` as the default static
                implementation; consumers plug their own for dynamic
                pricing.
            budget: Optional cap. When set, must be > 0. When ``None``
                this tracker is uncapped — ``exceeded`` is always
                False and ``urgent_wrapup`` never latches from cap
                crossing.
            parent: Optional parent tracker; costs recorded here also
                update the parent (recursively).
            on_cost: Optional per-tracker callback fired on every cost
                recorded at this level (whether via :meth:`track` here
                or reported up from a descendant).
            halt: Optional event set on the cap-crossing transition at
                this level.

        Raises:
            ValueError: When ``budget`` is set and ``<= 0``.
        """
        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be > 0 when set, got {budget}")
        self._lg = lg
        self._pricing = pricing
        self._budget = budget
        self._parent = parent
        self._on_cost = on_cost
        self._halt = halt
        self._spent = 0.0
        self._costs_by_op: dict[str, float] = {}
        self._urgent_wrapup = False

    @property
    def budget(self) -> float | None:
        """Current cap, or ``None`` when uncapped."""
        return self._budget

    @property
    def parent(self) -> Tracker | None:
        """The parent tracker if this is a child, else ``None``."""
        return self._parent

    @property
    def spent(self) -> float:
        """Sum of every cost recorded at this level or below."""
        return self._spent

    @property
    def remaining(self) -> float | None:
        """``budget - spent`` clamped at zero, or ``None`` when uncapped."""
        if self._budget is None:
            return None
        return max(0.0, self._budget - self._spent)

    @property
    def exceeded(self) -> bool:
        """Whether this tracker's spend has reached or crossed its cap."""
        if self._budget is None:
            return False
        return self._spent >= self._budget

    @property
    def urgent_wrapup(self) -> bool:
        """Soft signal — this tracker's own cap has been crossed."""
        return self._urgent_wrapup

    @urgent_wrapup.setter
    def urgent_wrapup(self, value: bool) -> None:
        """Force the wrap-up signal — e.g. caller-driven abort."""
        self._urgent_wrapup = value

    @property
    def costs_by_op(self) -> dict[str, float]:
        """Cost breakdown by op name at this level (defensive copy)."""
        return self._costs_by_op.copy()

    def child(
        self,
        budget: float | None = None,
        *,
        on_cost: CostCallback | None = None,
        halt: asyncio.Event | None = None,
    ) -> Tracker:
        """Create a child tracker whose costs report up to this one.

        Pricing is inherited from this tracker; ``halt`` and
        ``on_cost`` attach independently — a child's halt fires when
        the child's own cap crosses (not when this tracker's does),
        and a child's callback observes only the costs recorded on
        that child or its descendants.
        """
        return Tracker(
            self._lg,
            self._pricing,
            budget,
            parent=self,
            on_cost=on_cost,
            halt=halt,
        )

    def track(
        self,
        op_name: str,
        *,
        override_cost: float | None = None,
        context: dict[str, Any] | None = None,
        **usage: Any,
    ) -> float:
        """Record a usage event at this tracker level.

        Delegates cost computation to the tracker's
        :class:`PricingProvider` — ``op_name`` and ``usage`` are
        forwarded verbatim to :meth:`PricingProvider.compute`. When
        ``override_cost`` is supplied, the provider is skipped and
        the given value is recorded directly — useful for invoice
        reconciliation, provider-reported costs, or test ergonomics.

        Cost propagates up the parent chain: every ancestor's
        ``spent`` advances, and any ancestor whose cap crosses on
        THIS call fires its halt event and latches its
        ``urgent_wrapup``.

        ``context`` is passed through to :class:`CostCallback`s at
        every level in the chain (defensively deep-copied per
        callback). Callers must treat ``context`` as immutable after
        ``track()`` returns. In an N-deep tree with a callback at
        every level, a single :meth:`track` triggers N deep-copies;
        hot-path callers with observed deep trees should keep
        ``context`` shallow (or pre-frozen) to keep the copy cheap.

        Returns the cost recorded (``0.0`` when the op is unknown
        under the default :class:`PricingConfig` and no override was
        supplied).
        """
        if override_cost is not None:
            cost = override_cost
            overridden = True
        else:
            cost = self._pricing.compute(op_name, **usage)
            overridden = False
        if not math.isfinite(cost):
            raise ValueError(f"cost must be finite, got {cost}")
        self._record_cost(
            cost, op_name, context if context is not None else {}, overridden=overridden
        )
        return cost

    def update_budget(self, new_budget: float) -> None:
        """Amend the live cap.

        Loose contract — no comparison against ``spent``. When
        ``new_budget < spent`` ``exceeded`` becomes True immediately,
        but the halt event does NOT fire (halt only fires on the
        transition from not-exceeded to exceeded during a
        :meth:`track` call). Wallet-side floors (e.g. billable >=
        spent) belong at the caller.

        Raises:
            ValueError: When ``new_budget <= 0``.
        """
        if new_budget <= 0:
            raise ValueError(f"new_budget must be > 0, got {new_budget}")
        old = self._budget
        self._budget = new_budget
        self._lg.info(
            "budget amended",
            extra={"old_budget": old, "new_budget": new_budget, "spent": self._spent},
        )

    def restore_spent(self, amount: float) -> None:
        """Restore previously recorded spend (for pause/resume).

        Unconditional and this-level-only: does not walk up the
        parent chain. Callers restoring a full tree drive each level
        explicitly. Callers guarding against overwrite of live spend
        check ``tracker.spent == 0`` before calling.
        """
        self._spent = amount
        self._lg.debug("restored spend", extra={"spent": amount})

    def _record_cost(
        self,
        cost: float,
        op_name: str,
        context: dict[str, Any],
        *,
        overridden: bool,
    ) -> None:
        """Record one cost event and propagate up.

        Order at each level is load-bearing: spend increment →
        ``costs_by_op`` update → halt-on-cross + ``urgent_wrapup``
        latch → parent walk → ``on_cost`` callback. All bookkeeping
        (including at every ancestor) completes before any callback
        fires, so a raising callback cannot corrupt spend or suppress
        halt anywhere in the tree. Callbacks fire top-down (root
        first) as recursion unwinds; a raising callback interrupts
        callbacks below it in that top-down order but leaves all
        prior accounting intact.
        """
        was_exceeded = self.exceeded
        self._spent += cost
        self._costs_by_op[op_name] = self._costs_by_op.get(op_name, 0.0) + cost
        if not was_exceeded and self.exceeded:
            if self._halt is not None:
                self._halt.set()
            self._urgent_wrapup = True
        if self._parent is not None:
            self._parent._record_cost(cost, op_name, context, overridden=overridden)
        if self._on_cost is not None:
            self._on_cost(cost, copy.deepcopy(context), overridden=overridden)


__all__ = ["CostCallback", "Tracker"]
