# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for star-based preference pairing algorithms."""

import pytest

from llm_gent.core.training.pairing.stars import (
    StarFilter,
    _build_pools,
    _find_chosen_match,
    _generate_pairs,
    _track_used_items,
    _validate_pairing_params,
    pair_by_margin,
    pair_by_threshold,
)
from llm_gent.core.training.pairing.types import PreferencePair, RatedItem


pytestmark = pytest.mark.unit


def _item(id: int, score: int, content: str = "x") -> RatedItem[int]:
    return RatedItem(id=id, content=content, score=score)


# ---------------------------------------------------------------------------
# StarFilter
# ---------------------------------------------------------------------------


class TestStarFilterMatches:
    def test_eq(self):
        assert StarFilter(3, "==").matches(3) is True
        assert StarFilter(3, "==").matches(4) is False

    def test_gte(self):
        assert StarFilter(3, ">=").matches(3) is True
        assert StarFilter(3, ">=").matches(4) is True
        assert StarFilter(3, ">=").matches(2) is False

    def test_lte(self):
        assert StarFilter(3, "<=").matches(3) is True
        assert StarFilter(3, "<=").matches(2) is True
        assert StarFilter(3, "<=").matches(4) is False

    def test_gt(self):
        assert StarFilter(3, ">").matches(4) is True
        assert StarFilter(3, ">").matches(3) is False

    def test_lt(self):
        assert StarFilter(3, "<").matches(2) is True
        assert StarFilter(3, "<").matches(3) is False

    def test_unsupported_op(self):
        with pytest.raises(ValueError, match="Unsupported operator"):
            StarFilter(3, "!=").matches(3)


class TestStarFilterParse:
    def test_none(self):
        assert StarFilter.parse(None) is None

    def test_empty_string(self):
        assert StarFilter.parse("") is None

    def test_whitespace(self):
        assert StarFilter.parse("  ") is None

    def test_int(self):
        f = StarFilter.parse(3)
        assert f.value == 3
        assert f.op == "=="

    def test_string_exact(self):
        f = StarFilter.parse("3")
        assert f.value == 3
        assert f.op == "=="

    def test_string_gte(self):
        f = StarFilter.parse(">=4")
        assert f.value == 4
        assert f.op == ">="

    def test_string_lte(self):
        f = StarFilter.parse("<=2")
        assert f.value == 2
        assert f.op == "<="

    def test_string_gt(self):
        f = StarFilter.parse(">1")
        assert f.value == 1
        assert f.op == ">"

    def test_string_lt(self):
        f = StarFilter.parse("<5")
        assert f.value == 5
        assert f.op == "<"

    def test_whitespace_stripped(self):
        f = StarFilter.parse("  >=3  ")
        assert f.value == 3
        assert f.op == ">="

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid star filter"):
            StarFilter.parse("abc")


# ---------------------------------------------------------------------------
# _validate_pairing_params
# ---------------------------------------------------------------------------


class TestValidatePairingParams:
    def test_valid(self):
        _validate_pairing_params(1, None, None)
        _validate_pairing_params(3, 5, 10)

    def test_zero_margin(self):
        with pytest.raises(ValueError, match="margin must be >= 1"):
            _validate_pairing_params(0, None, None)

    def test_negative_min(self):
        with pytest.raises(ValueError, match="min_pairs cannot be negative"):
            _validate_pairing_params(1, -1, None)

    def test_negative_max(self):
        with pytest.raises(ValueError, match="max_pairs cannot be negative"):
            _validate_pairing_params(1, None, -1)

    def test_min_gt_max(self):
        with pytest.raises(ValueError, match="min_pairs.*cannot be greater than max_pairs"):
            _validate_pairing_params(1, 10, 5)


# ---------------------------------------------------------------------------
# _build_pools
# ---------------------------------------------------------------------------


class TestBuildPools:
    def test_empty(self):
        assert _build_pools([], 1, None, None) == ([], [])

    def test_margin_split(self):
        items = [_item(1, 1), _item(2, 3), _item(3, 5)]
        chosen, rejected = _build_pools(items, 2, None, None)
        # chosen: score - margin >= min_score (1), so score >= 3
        assert all(c.score >= 3 for c in chosen)
        # rejected: score + margin <= max_score (5), so score <= 3
        assert all(r.score <= 3 for r in rejected)

    def test_margin_too_large(self):
        items = [_item(1, 3), _item(2, 4)]
        chosen, rejected = _build_pools(items, 3, None, None)
        # margin=3, range is only 1 star, no valid pools
        assert chosen == [] or rejected == []

    def test_chosen_filter(self):
        items = [_item(1, 1), _item(2, 3), _item(3, 5)]
        f = StarFilter(5, "==")
        chosen, rejected = _build_pools(items, 1, chosen_filter=f, rejected_filter=None)
        assert all(c.score == 5 for c in chosen)

    def test_rejected_filter(self):
        items = [_item(1, 1), _item(2, 3), _item(3, 5)]
        f = StarFilter(1, "==")
        chosen, rejected = _build_pools(items, 1, chosen_filter=None, rejected_filter=f)
        assert all(r.score == 1 for r in rejected)


# ---------------------------------------------------------------------------
# _find_chosen_match
# ---------------------------------------------------------------------------


class TestFindChosenMatch:
    def test_valid_match(self):
        chosen_pool = [_item(1, 5, "hello")]
        rejected = _item(2, 1, "world")
        result = _find_chosen_match(rejected, chosen_pool, set(), 1, False, 0)
        assert result is not None
        assert result[0].id == 1

    def test_same_id_skipped(self):
        chosen_pool = [_item(1, 5, "hello")]
        rejected = _item(1, 1, "world")  # same ID
        result = _find_chosen_match(rejected, chosen_pool, set(), 1, False, 0)
        assert result is None

    def test_insufficient_margin(self):
        chosen_pool = [_item(1, 3, "hello")]
        rejected = _item(2, 3, "world")
        result = _find_chosen_match(rejected, chosen_pool, set(), 1, False, 0)
        assert result is None

    def test_no_reuse_skips_used(self):
        chosen_pool = [_item(1, 5, "hello")]
        rejected = _item(2, 1, "world")
        result = _find_chosen_match(rejected, chosen_pool, {1}, 1, True, 0)
        assert result is None

    def test_reuse_allows_used(self):
        chosen_pool = [_item(1, 5, "hello")]
        rejected = _item(2, 1, "world")
        result = _find_chosen_match(rejected, chosen_pool, {1}, 1, False, 0)
        assert result is not None

    def test_length_epsilon_filters(self):
        chosen_pool = [_item(1, 5, "a" * 100)]
        rejected = _item(2, 1, "b")  # length diff = 99
        result = _find_chosen_match(rejected, chosen_pool, set(), 1, False, 0, length_epsilon=5)
        assert result is None

    def test_length_epsilon_allows(self):
        chosen_pool = [_item(1, 5, "hello")]
        rejected = _item(2, 1, "world")
        result = _find_chosen_match(rejected, chosen_pool, set(), 1, False, 0, length_epsilon=5)
        assert result is not None

    def test_wrap_around(self):
        chosen_pool = [_item(1, 5, "a"), _item(2, 5, "b")]
        rejected = _item(3, 1, "c")
        # start_idx=1, should try idx 1 first, then wrap to 0
        result = _find_chosen_match(rejected, chosen_pool, set(), 1, False, 1)
        assert result is not None
        assert result[0].id == 2  # starts at idx 1

    def test_empty_pool(self):
        result = _find_chosen_match(_item(1, 1, "x"), [], set(), 1, False, 0)
        assert result is None


# ---------------------------------------------------------------------------
# _track_used_items
# ---------------------------------------------------------------------------


class TestTrackUsedItems:
    def test_tracks_rejected_always(self):
        pairs = []
        new_pairs = [PreferencePair(chosen=_item(1, 5), rejected=_item(2, 1))]
        used_chosen = set()
        used_rejected = set()
        _track_used_items(pairs, new_pairs, used_chosen, used_rejected, no_reuse=False)
        assert 2 in used_rejected
        assert 1 not in used_chosen  # no_reuse=False

    def test_tracks_chosen_with_no_reuse(self):
        pairs = []
        new_pairs = [PreferencePair(chosen=_item(1, 5), rejected=_item(2, 1))]
        used_chosen = set()
        used_rejected = set()
        _track_used_items(pairs, new_pairs, used_chosen, used_rejected, no_reuse=True)
        assert 1 in used_chosen
        assert 2 in used_rejected


# ---------------------------------------------------------------------------
# _generate_pairs
# ---------------------------------------------------------------------------


class TestGeneratePairs:
    def test_empty_pools(self):
        assert _generate_pairs([], [_item(1, 1)], 1, 5, False) == []
        assert _generate_pairs([_item(1, 5)], [], 1, 5, False) == []

    def test_single_pass_without_epsilons(self):
        chosen = [_item(1, 5, "hello")]
        rejected = [_item(2, 1, "world")]
        pairs = _generate_pairs(chosen, rejected, 1, 10, False, length_epsilons=None)
        assert len(pairs) == 1

    def test_multi_pass_with_epsilons(self):
        chosen = [_item(1, 5, "hello")]
        rejected = [_item(2, 1, "world")]
        pairs = _generate_pairs(chosen, rejected, 1, 10, False, length_epsilons=[0, None])
        assert len(pairs) >= 1


# ---------------------------------------------------------------------------
# pair_by_margin
# ---------------------------------------------------------------------------


class TestPairByMargin:
    def test_empty(self):
        result = pair_by_margin([])
        assert result.pairs == []
        assert result.total_rated == 0
        assert result.strategy == "margin"

    def test_basic_pairing(self):
        items = [_item(1, 5, "good"), _item(2, 1, "bad")]
        result = pair_by_margin(items, margin=1)
        assert len(result.pairs) >= 1
        assert result.total_rated == 2

    def test_no_valid_pairs(self):
        items = [_item(1, 3, "a"), _item(2, 3, "b")]
        result = pair_by_margin(items, margin=2)
        assert len(result.pairs) == 0

    def test_max_pairs(self):
        items = [_item(i, 5 if i % 2 == 0 else 1, "x") for i in range(20)]
        result = pair_by_margin(items, margin=1, max_pairs=3)
        assert len(result.pairs) <= 3

    def test_min_pairs(self):
        items = [_item(1, 5, "good"), _item(2, 1, "bad")]
        result = pair_by_margin(items, margin=1, min_pairs=1)
        assert len(result.pairs) >= 1

    def test_no_reuse(self):
        items = [_item(1, 5, "a"), _item(2, 1, "b"), _item(3, 1, "c")]
        result = pair_by_margin(items, margin=1, no_reuse=True)
        chosen_ids = [p.chosen.id for p in result.pairs]
        assert len(chosen_ids) == len(set(chosen_ids))

    def test_chosen_filter(self):
        items = [_item(1, 5, "a"), _item(2, 4, "b"), _item(3, 1, "c")]
        f = StarFilter(5, "==")
        result = pair_by_margin(items, margin=1, chosen_filter=f)
        for p in result.pairs:
            assert p.chosen.score == 5

    def test_length_epsilons(self):
        items = [_item(1, 5, "hello"), _item(2, 1, "world")]
        result = pair_by_margin(items, margin=1, length_epsilons=[0, None])
        assert len(result.pairs) >= 1

    def test_validation_error(self):
        with pytest.raises(ValueError, match="margin must be >= 1"):
            pair_by_margin([], margin=0)

    def test_empty_pools_after_filter(self):
        items = [_item(1, 5, "a"), _item(2, 4, "b")]
        f = StarFilter(1, "==")
        result = pair_by_margin(items, margin=1, chosen_filter=f)
        assert result.pairs == []


# ---------------------------------------------------------------------------
# pair_by_threshold
# ---------------------------------------------------------------------------


class TestPairByThreshold:
    def test_empty(self):
        result = pair_by_threshold([])
        assert result.pairs == []
        assert result.total_rated == 0
        assert result.strategy == "threshold"

    def test_basic(self):
        items = [_item(1, 5, "good"), _item(2, 1, "bad")]
        result = pair_by_threshold(items, high_threshold=4, low_threshold=2)
        assert len(result.pairs) == 1
        assert result.pairs[0].chosen.score >= 4
        assert result.pairs[0].rejected.score <= 2

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError, match="must be greater than"):
            pair_by_threshold([], high_threshold=3, low_threshold=3)

    def test_reversed_thresholds_raises(self):
        with pytest.raises(ValueError, match="must be greater than"):
            pair_by_threshold([], high_threshold=2, low_threshold=4)

    def test_negative_max_pairs(self):
        with pytest.raises(ValueError, match="max_pairs cannot be negative"):
            pair_by_threshold([], max_pairs=-1)

    def test_max_pairs_caps(self):
        items = [_item(i, 5 if i < 5 else 1, "x") for i in range(10)]
        result = pair_by_threshold(items, high_threshold=4, low_threshold=2, max_pairs=2)
        assert len(result.pairs) <= 2

    def test_no_chosen(self):
        items = [_item(1, 1, "a"), _item(2, 2, "b")]
        result = pair_by_threshold(items, high_threshold=4, low_threshold=2)
        assert result.pairs == []

    def test_no_rejected(self):
        items = [_item(1, 5, "a"), _item(2, 4, "b")]
        result = pair_by_threshold(items, high_threshold=4, low_threshold=2)
        assert result.pairs == []

    def test_id_exclusion(self):
        """Same item can't appear as both chosen and rejected."""
        # item 3 (score=3) is >= high=3 (chosen) AND <= low=2 is False, so not in rejected
        # This tests the chosen_ids exclusion in the rejected pool filter
        items = [_item(1, 5, "a"), _item(2, 1, "b"), _item(3, 3, "c")]
        result = pair_by_threshold(items, high_threshold=3, low_threshold=2)
        for p in result.pairs:
            assert p.chosen.id != p.rejected.id
