# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Panel — fan-out N verbs in parallel and aggregate their results.

A :class:`Panel` is a composition helper (not a verb). Called from inside a
verb, it dispatches each inner verb through the calling context's flow so
per-verb role routing applies as normal, then reduces the results via a
pluggable aggregation function.

Common aggregators are included as module-level helpers (``majority``,
``unanimous``, ``mean``, ``weighted``); any callable ``list[R] -> R'`` works.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from typing import Any

from .context import Context


AggregateFn = Callable[[list[Any]], Any]
"""A function that reduces a list of verb results to a single value."""


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------


def majority(votes: list[Any]) -> Any:
    """Return the most common vote.

    Ties are broken by first-occurrence order. Raises :class:`ValueError` on
    an empty list. Falls back to equality-based counting for unhashable types.
    """
    if not votes:
        raise ValueError("majority requires at least one vote")
    try:
        return Counter(votes).most_common(1)[0][0]
    except TypeError:
        # Fallback for unhashable types (dicts, lists)
        counts: list[tuple[Any, int]] = []
        for v in votes:
            for i, (existing, count) in enumerate(counts):
                if v == existing:
                    counts[i] = (existing, count + 1)
                    break
            else:
                counts.append((v, 1))
        return max(counts, key=lambda x: x[1])[0]


def unanimous(votes: list[Any]) -> Any | None:
    """Return the common value if all votes agree, else ``None`` (also for empty)."""
    if not votes:
        return None
    first = votes[0]
    return first if all(v == first for v in votes) else None


def mean(votes: list[float]) -> float:
    """Return the arithmetic mean of numeric votes; ``0.0`` for an empty list."""
    return sum(votes) / len(votes) if votes else 0.0


def weighted(items: list[tuple[float, float]]) -> float:
    """Return the weight-normalized total of ``(value, weight)`` pairs.

    Returns ``0.0`` when the total weight is zero (including the empty case).
    """
    total_weight = sum(w for _, w in items)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in items) / total_weight


# -----------------------------------------------------------------------------
# Panel
# -----------------------------------------------------------------------------


class Panel:
    """Fan-out N verbs in parallel and aggregate their results.

    ``Panel`` is a helper called from inside a verb (or user code that
    already holds a :class:`Context`). Each inner verb is dispatched via
    ``ctx.flow`` so per-verb role routing applies.

    Example::

        panel = Panel(
            verbs=[verify_bio, verify_tweets, verify_projects],
            aggregate=majority,
        )

        @verb(role=EVALUATOR)
        async def evaluate(ctx, candidate):
            verdict = await panel.run(ctx, candidate)
            ...
    """

    def __init__(
        self,
        verbs: list[Any],
        aggregate: AggregateFn = majority,
    ) -> None:
        """Initialize with the verbs to fan out and the aggregation function."""
        if not verbs:
            raise ValueError("Panel requires at least one verb")
        self.verbs = list(verbs)
        self.aggregate = aggregate

    async def run(self, ctx: Context, *args: Any, **kwargs: Any) -> Any:
        """Dispatch each inner verb in parallel and aggregate the results.

        Each inner verb must have been registered with the flow referenced by
        ``ctx.flow``. Positional and keyword args are forwarded to every verb.
        """
        results = await asyncio.gather(
            *[
                ctx.flow.dispatch(getattr(v, "_registered_name", v.__name__), *args, **kwargs)
                for v in self.verbs
            ]
        )
        return self.aggregate(results)
