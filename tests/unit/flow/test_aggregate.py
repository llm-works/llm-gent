# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Aggregators for a map's results: ``majority``, ``unanimous``, ``mean``, ``weighted``."""

from __future__ import annotations

import pytest

from llm_gent.flow import majority, mean, unanimous, weighted


pytestmark = pytest.mark.unit


class TestAggregators:
    def test_majority_returns_most_common(self) -> None:
        """majority picks the most frequently occurring value."""
        assert majority(["a", "b", "a", "c", "a"]) == "a"

    def test_majority_tie_first_seen_wins(self) -> None:
        """On a tie, the first-inserted value wins (Counter semantics)."""
        assert majority(["a", "b", "a", "b"]) == "a"

    def test_majority_of_unhashable_votes(self) -> None:
        """Unhashable votes are counted by equality."""
        assert majority([{"v": 1}, {"v": 2}, {"v": 1}]) == {"v": 1}

    def test_majority_empty_raises(self) -> None:
        """majority on an empty list is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            majority([])

    def test_unanimous_all_agree(self) -> None:
        """unanimous returns the shared value when every vote matches."""
        assert unanimous(["yes", "yes", "yes"]) == "yes"

    def test_unanimous_disagree_returns_none(self) -> None:
        """unanimous returns None when any vote diverges."""
        assert unanimous(["yes", "no", "yes"]) is None

    def test_unanimous_empty_returns_none(self) -> None:
        """unanimous on an empty list returns None (no value to agree on)."""
        assert unanimous([]) is None

    def test_mean(self) -> None:
        """mean returns the arithmetic mean of the votes."""
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty(self) -> None:
        """mean on an empty list returns 0.0 (avoids ZeroDivisionError)."""
        assert mean([]) == 0.0

    def test_weighted_normal(self) -> None:
        """weighted returns a properly weight-normalized average."""
        assert weighted([(1.0, 0.5), (3.0, 0.5)]) == pytest.approx(2.0)

    def test_weighted_uneven_weights(self) -> None:
        """weighted respects unequal weights."""
        assert weighted([(1.0, 0.25), (5.0, 0.75)]) == pytest.approx(4.0)

    def test_weighted_zero_weight_returns_zero(self) -> None:
        """weighted with total weight of 0 returns 0.0 (safe on empty / all-zero)."""
        assert weighted([]) == 0.0
        assert weighted([(1.0, 0.0), (5.0, 0.0)]) == 0.0
