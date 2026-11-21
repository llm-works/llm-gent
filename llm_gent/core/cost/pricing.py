# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Pricing configuration — currency-agnostic op cost lookup.

:class:`PricingProvider` is the plug seam :class:`Tracker` calls on
every recorded usage event. Substrate ships :class:`PricingConfig` as
the default static implementation — a name-indexed :class:`Op`
registry with prefix/default fallback. Consumers that need dynamic or
provider-aware pricing (per-request surcharges, negotiated rates,
invoice reconciliation) implement their own :class:`PricingProvider`
and inject it on :class:`Tracker` construction.

An :class:`Op` is a named, billable operation. Each Op exposes a
``cost()`` method whose signature is implementation-defined; callers
pass matching kwargs when tracking usage.

Substrate ships two built-in Ops:

- :class:`LLMOp` — token-based (input / output / cached) with per-mtok
  rates and an optional cache discount.
- :class:`FixedOp` — flat unit cost per invocation.

Consumers can define their own Op shapes for domain-specific billing
(compute hours, bytes transferred, quality-scored ops, ...). Any object
with a ``name: str`` attribute and a compatible ``cost()`` method
satisfies the :class:`Op` protocol.

The tracker is currency-agnostic — rates and returned values carry
whatever unit the consumer uses. All Ops in one :class:`PricingConfig`
must share that unit; Python floats can't enforce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class Op(Protocol):
    """A billable operation known to a :class:`PricingConfig`.

    Implementations expose:

    - ``name: str`` — identifier used for lookup.
    - ``cost(**usage) -> float`` — computes cost from usage kwargs.

    The ``cost()`` signature is implementation-defined. Callers of
    :meth:`Tracker.track` pass usage kwargs matching the Op they're
    tracking; the tracker forwards them to ``op.cost(**usage)``.
    """

    name: str

    def cost(self, **usage: Any) -> float:
        """Compute cost from usage kwargs; signature is impl-defined."""
        ...


@dataclass
class LLMOp:
    """Token-based LLM billing.

    ``cached_input_per_mtok`` is the discounted rate for the subset of
    ``input_tokens`` served from a provider prompt cache. When
    ``None``, cached tokens bill at ``input_per_mtok``.
    """

    name: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    def cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> float:
        """Compute cost from token counts. Negative inputs clamp to zero."""
        input_tokens = max(input_tokens, 0)
        output_tokens = max(output_tokens, 0)

        if self.cached_input_per_mtok is None:
            uncached_input = input_tokens
            cached_cost = 0.0
        else:
            billable_cached = min(max(cached_tokens, 0), input_tokens)
            uncached_input = input_tokens - billable_cached
            cached_cost = billable_cached * self.cached_input_per_mtok / 1_000_000

        return (
            uncached_input * self.input_per_mtok / 1_000_000
            + cached_cost
            + output_tokens * self.output_per_mtok / 1_000_000
        )


@dataclass
class FixedOp:
    """Flat unit cost per invocation, scaled by ``count``."""

    name: str
    unit_cost: float

    def cost(self, count: int = 1) -> float:
        """Compute ``unit_cost * count``. Negative count clamps to zero."""
        return self.unit_cost * max(count, 0)


class PricingProvider(Protocol):
    """Cost-computation seam :class:`Tracker` calls on every event.

    Implementations own the pricing math. The default
    :class:`PricingConfig` computes from a static per-op registry;
    downstream implementations plug in dynamic or provider-aware
    pricing (Anthropic cache-creation multipliers, Gemini service-tier
    surcharges, negotiated rates, invoice reconciliation) without
    changing the tracker or verb-body call shape.

    Wiring: pass the implementation as the ``pricing`` argument to
    :class:`Tracker`. Every :meth:`Tracker.track` call forwards
    ``op_name`` and the usage kwargs directly to :meth:`compute`.
    """

    def compute(self, op_name: str, /, **usage: Any) -> float:
        """Compute cost for one usage event.

        The kwargs are whatever the caller passed to
        :meth:`Tracker.track` alongside ``op_name`` (token counts,
        cached-token counts, provider signals, ...). Implementations
        return the cost in the consumer's chosen currency.

        Unknown ops should return ``0.0`` — matches the passive
        default of :class:`PricingConfig` and lets consumers keep
        tracking calls in place while pricing rules roll out.
        """
        ...


@dataclass
class PricingConfig:
    """Default :class:`PricingProvider` — static per-op registry.

    Composes an :class:`Op`-per-name table with prefix/default
    fallback. Suitable for tests, simple consumers, and any workload
    where per-op rates are known ahead of time. Consumers needing
    dynamic pricing implement :class:`PricingProvider` directly.

    Example::

        pricing = PricingConfig(ops={
            "some-fast-model": LLMOp(
                name="some-fast-model",
                input_per_mtok=0.30,
                output_per_mtok=2.50,
                cached_input_per_mtok=0.03,
            ),
            "web_search": FixedOp(name="web_search", unit_cost=0.001),
        })
        cost = pricing.compute("some-fast-model", input_tokens=1000, output_tokens=100)
    """

    ops: dict[str, Op] = field(default_factory=dict)

    def get(self, name: str) -> Op | None:
        """Look up an Op by name.

        Order: exact match → longest-prefix match (excluding a
        ``"default"`` key) → the ``"default"`` key if present.
        Returns ``None`` when no rule applies.
        """
        if name in self.ops:
            return self.ops[name]

        matches = [k for k in self.ops if k != "default" and name.startswith(k)]
        if matches:
            return self.ops[max(matches, key=len)]

        return self.ops.get("default")

    def compute(self, op_name: str, /, **usage: Any) -> float:
        """Look up ``op_name`` and forward ``usage`` to ``op.cost``.

        Returns ``0.0`` when no rule applies for ``op_name``.
        """
        op = self.get(op_name)
        return op.cost(**usage) if op is not None else 0.0


__all__ = ["FixedOp", "LLMOp", "Op", "PricingConfig", "PricingProvider"]
