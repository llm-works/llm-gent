# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Aggregators for a map's results — mostly an ensemble's votes.

``.map([judge_a, judge_b, judge_c], aggregate=majority)`` runs each judge on
the step's input and reduces their results. Any callable ``list[R] -> R'``
works as ``aggregate``; these cover the common votes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


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
