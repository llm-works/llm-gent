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

    async def run(self, ctx: Context[Any], *args: Any, **kwargs: Any) -> Any:
        """Dispatch each inner verb in parallel and aggregate the results.

        Each inner verb must have been registered with the flow referenced by
        ``ctx.flow``. Positional and keyword args are forwarded to every verb.

        Ambient forwarding — each dispatch receives ``scope_state=ctx.state``,
        ``halt=ctx.halt``, ``budget=ctx.budget``, and ``extra=ctx.extra``
        so inner verbs see the caller's live scope, not the flow's
        construction defaults.
        Inside a ``.map(state=...)`` or ``.call(state=...)`` block this
        makes ``ctx.data`` on the inner verb the same object the calling
        verb sees. Without this forwarding an inner verb reading
        ``ctx.data`` would silently observe the flow's construction
        state instead.

        Halt semantics — cooperative per verb, as for map items. Each verb
        receives ``ctx.halt`` and decides on its own whether to poll it;
        verbs backed by saia observe it at call entry and mid-stream (a
        Loop's turn pauses), so a halted Panel of saia verbs stops in
        about the longest single in-flight LLM call — not the sum across
        N. Verbs that don't poll halt run to completion.

        Inside a run (``ctx`` from a step of a running flow), the Panel is
        in the run's snapshots: it runs as a map over its verbs at
        ``<step>/panel/<k>``, the ``k``-th Panel the step starts, verb
        ``i`` as map item ``i``. A verb's result is recorded when it
        finishes — and must then be checkpointable, like a map item's: plain
        JSON, a pydantic model, or an object with ``to_dict()`` and a
        classmethod ``from_dict()``. A Loop in a verb keeps its paused turn,
        and ``ctx.checkpoint()`` in a verb writes a commit. When the halt
        stops a verb before it finishes, the Panel raises
        :class:`~llm_gent.flow.Interrupted` once every verb has stopped, so
        the calling step is interrupted; resume runs the step again, and
        its Panel continues: finished verbs don't run again, a running one
        continues where it was (a paused turn resumes mid-turn), and the
        aggregate equals the uninterrupted run's. The first verb exception
        propagates; siblings run on and their results are discarded.

        Outside a run (``Flow.dispatch`` used standalone, or a hook
        context) there is nothing to checkpoint: the verbs are dispatched
        through ``ctx.flow`` and gathered.

        Raises:
            Interrupted: Inside a run, the halt stopped a verb before it
                finished.
        """
        env, node_id = ctx._env, ctx._node_id
        if env is not None and node_id is not None:
            from ._panel_map import run_panel

            return await run_panel(env, node_id, self.verbs, self.aggregate, args, kwargs)
        return await self._gather(ctx, args, kwargs)

    async def _gather(
        self, ctx: Context[Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Dispatch every verb through ``ctx.flow`` concurrently and aggregate (outside a run)."""
        results = await asyncio.gather(
            *[
                ctx.flow.dispatch(
                    getattr(v, "_registered_name", v.__name__),
                    *args,
                    scope_state=ctx.state,
                    halt=ctx.halt,
                    budget=ctx.budget,
                    extra=ctx.extra,
                    **kwargs,
                )
                for v in self.verbs
            ]
        )
        return self.aggregate(results)
