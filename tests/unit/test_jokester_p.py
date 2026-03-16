"""Tests for jokester-p agent components."""

from unittest.mock import MagicMock

import pytest
from appinfra import DotDict

from llm_gent.agents.jokester_p.history import JokeHistory
from llm_gent.agents.jokester_p.novelty import NoveltyCheck, NoveltyChecker
from llm_gent.agents.jokester_p.variety import VarietyChecker
from llm_gent.core.training import StarPreferencePair, StarRatedItem
from llm_gent.core.training.pairing.stars import StarFilter, pair_by_margin


pytestmark = pytest.mark.unit


def _novelty_config(threshold: float = 0.85, mode: str | None = None) -> DotDict:
    """Create novelty config for tests."""
    config = DotDict({"similarity": {"threshold": threshold}})
    if mode:
        config["mode"] = mode
    return config


class TestNoveltyCheck:
    """Tests for NoveltyCheck dataclass."""

    def test_novel_joke(self):
        """NoveltyCheck for a novel joke."""
        check = NoveltyCheck(is_novel=True, max_similarity=0.3, similar_joke=None)
        assert check.is_novel is True
        assert check.max_similarity == 0.3
        assert check.similar_joke is None

    def test_duplicate_joke(self):
        """NoveltyCheck for a duplicate joke."""
        check = NoveltyCheck(is_novel=False, max_similarity=0.95, similar_joke="existing joke")
        assert check.is_novel is False
        assert check.max_similarity == 0.95
        assert check.similar_joke == "existing joke"


class TestNoveltyChecker:
    """Tests for NoveltyChecker."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    def mock_learn_trait(self):
        """Create mock learn trait."""
        trait = MagicMock()
        trait.has_embedder = True
        return trait

    def test_has_embedder_delegates_to_trait(self, mock_logger, mock_learn_trait):
        """has_embedder property delegates to learn trait."""
        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        assert checker.has_embedder is True

        mock_learn_trait.has_embedder = False
        assert checker.has_embedder is False

    def test_check_novel_no_existing_jokes(self, mock_logger, mock_learn_trait):
        """Check returns novel when no existing jokes found."""
        mock_learn_trait.recall.return_value = []
        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())

        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is True
        assert result.max_similarity == 0.0
        assert result.similar_joke is None

    def test_check_novel_below_threshold(self, mock_logger, mock_learn_trait):
        """Check returns novel when similarity is below threshold."""
        similar = MagicMock()
        similar.score = 0.5
        similar.entity.content = "Different joke"
        mock_learn_trait.recall.return_value = [similar]

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is True
        assert result.max_similarity == 0.5
        assert result.similar_joke == "Different joke"

    def test_check_duplicate_above_threshold(self, mock_logger, mock_learn_trait):
        """Check returns not novel when similarity exceeds threshold."""
        similar = MagicMock()
        similar.score = 0.92
        similar.entity.content = "Very similar joke"
        mock_learn_trait.recall.return_value = [similar]

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is False
        assert result.max_similarity == 0.92
        assert result.similar_joke == "Very similar joke"

    def test_check_fails_closed_on_error(self, mock_logger, mock_learn_trait):
        """Check returns not novel on error (fail closed)."""
        mock_learn_trait.recall.side_effect = Exception("Database error")

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is False
        assert result.max_similarity == 1.0

    def test_check_novel_when_table_not_exists(self, mock_logger, mock_learn_trait):
        """Check returns novel when metadata table doesn't exist (lazy creation)."""
        # Simulate PostgreSQL "relation does not exist" error
        mock_learn_trait.recall.side_effect = Exception(
            'relation "playground.agent_jokester_training" does not exist'
        )

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        # Should treat as novel (no existing jokes), not fail closed
        assert result.is_novel is True
        assert result.max_similarity == 0.0
        assert result.similar_joke is None

    def test_check_novel_when_table_not_exists_sqlite(self, mock_logger, mock_learn_trait):
        """Check returns novel on SQLite 'no such table' error."""
        mock_learn_trait.recall.side_effect = Exception("no such table: agent_jokester_training")

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is True
        assert result.max_similarity == 0.0

    def test_check_rejects_empty_text(self, mock_logger, mock_learn_trait):
        """Check rejects empty joke text without calling embedder."""
        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())

        # Empty string
        result = checker.check("")
        assert result.is_novel is False
        assert result.max_similarity == 1.0
        mock_learn_trait.recall.assert_not_called()

        # Whitespace only
        result = checker.check("   ")
        assert result.is_novel is False
        mock_learn_trait.recall.assert_not_called()


class TestPreferencePair:
    """Tests for preference pairing types."""

    def test_rated_item(self):
        """StarRatedItem stores item data."""
        item = StarRatedItem(id=1, content="Funny joke", score=4)
        assert item.id == 1
        assert item.content == "Funny joke"
        assert item.score == 4

    def test_preference_pair_margin(self):
        """PreferencePair margin is score difference."""
        chosen = StarRatedItem(id=1, content="Great joke", score=5)
        rejected = StarRatedItem(id=2, content="Bad joke", score=1)
        pair = StarPreferencePair(chosen=chosen, rejected=rejected)

        assert pair.chosen.score - pair.rejected.score == 4

    def test_preference_pair_zero_margin(self):
        """PreferencePair with same score has zero margin."""
        chosen = StarRatedItem(id=1, content="Joke A", score=3)
        rejected = StarRatedItem(id=2, content="Joke B", score=3)
        pair = StarPreferencePair(chosen=chosen, rejected=rejected)

        assert pair.chosen.score - pair.rejected.score == 0


class TestJokeHistory:
    """Tests for JokeHistory."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    def test_contains_empty_history(self, mock_logger):
        """Contains returns False for empty history."""
        history = JokeHistory(mock_logger)
        assert history.contains("any joke") is False

    def test_contains_after_record(self, mock_logger):
        """Contains returns True after recording joke."""
        history = JokeHistory(mock_logger)
        history.record("Why did the chicken cross the road?")
        assert history.contains("Why did the chicken cross the road?") is True
        assert history.contains("Different joke") is False

    def test_contains_bumps_frequency(self, mock_logger):
        """Recording same joke multiple times bumps frequency."""
        history = JokeHistory(mock_logger)
        history.record("Same joke")
        history.record("Same joke")
        history.record("Same joke")
        # Frequency should be 3, check via _frequency (internal but useful for test)
        assert history._frequency["Same joke"] == 3


class TestVarietyChecker:
    """Tests for VarietyChecker."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    def test_disabled_when_no_config(self, mock_logger):
        """Checker is disabled with empty config."""
        checker = VarietyChecker(mock_logger, DotDict({}))
        assert checker.enabled is False
        result = checker.check("any joke")
        assert result.is_varied is True

    def test_ngram_rejects_same_opening(self, mock_logger):
        """N-gram check rejects jokes with same opening."""
        config = DotDict({"ngram": {"enabled": True, "n": 2, "history": 10}})
        checker = VarietyChecker(mock_logger, config)

        checker.record("Why don't scientists trust atoms?")
        result = checker.check("Why don't programmers like nature?")
        assert result.is_varied is False
        assert "why don't" in result.reason.lower()

    def test_ngram_allows_different_opening(self, mock_logger):
        """N-gram check allows jokes with different openings."""
        config = DotDict({"ngram": {"enabled": True, "n": 3, "history": 10}})
        checker = VarietyChecker(mock_logger, config)

        checker.record("Why don't scientists trust atoms?")
        result = checker.check("I told my wife she was drawing her eyebrows too high.")
        assert result.is_varied is True

    def test_levenshtein_rejects_similar(self, mock_logger):
        """Levenshtein check rejects very similar jokes."""
        # Only enable levenshtein (no ngram) to test levenshtein specifically
        config = DotDict({"levenshtein": {"enabled": True, "threshold": 0.7, "history": 10}})
        checker = VarietyChecker(mock_logger, config)

        checker.record("The chicken crossed the road to get to the other side.")
        result = checker.check("The chicken crossed the street to get to the other side.")
        assert result.is_varied is False
        assert "similar" in result.reason.lower()

    def test_levenshtein_allows_different(self, mock_logger):
        """Levenshtein check allows sufficiently different jokes."""
        config = DotDict({"levenshtein": {"enabled": True, "threshold": 0.8, "history": 10}})
        checker = VarietyChecker(mock_logger, config)

        checker.record("Why did the chicken cross the road?")
        result = checker.check("I used to hate facial hair, but then it grew on me.")
        assert result.is_varied is True


class TestStarsPairing:
    """Tests for star-based preference pairing algorithm."""

    def _make_items(self, scores: list[int]) -> list[StarRatedItem]:
        """Create rated items from a list of scores."""
        return [StarRatedItem(id=i, content=f"joke_{i}", score=s) for i, s in enumerate(scores)]

    def test_pair_by_margin_empty_list(self):
        """Empty list returns empty result."""
        result = pair_by_margin([], margin=2)
        assert result.pairs == []
        assert result.total_rated == 0

    def test_pair_by_margin_basic(self):
        """Basic pairing with margin=2."""
        items = self._make_items([5, 4, 3, 2, 1])
        result = pair_by_margin(items, margin=2)

        assert len(result.pairs) > 0
        for pair in result.pairs:
            assert pair.chosen.score - pair.rejected.score >= 2

    def test_pair_by_margin_respects_margin(self):
        """Pairs respect minimum margin requirement."""
        items = self._make_items([5, 4, 3, 2, 1])
        result = pair_by_margin(items, margin=3)

        for pair in result.pairs:
            assert pair.chosen.score - pair.rejected.score >= 3

    def test_pair_by_margin_no_valid_pairs(self):
        """Returns empty when no pairs meet margin."""
        items = self._make_items([3, 3, 3])  # All same score
        result = pair_by_margin(items, margin=2)

        assert result.pairs == []
        assert result.total_rated == 3

    def test_pair_by_margin_no_reuse(self):
        """Each chosen item used once with no_reuse=True."""
        items = self._make_items([5, 5, 4, 1, 1, 1])
        result = pair_by_margin(items, margin=3, no_reuse=True)

        chosen_ids = [p.chosen.id for p in result.pairs]
        assert len(chosen_ids) == len(set(chosen_ids))  # No duplicates

    def test_pair_by_margin_with_reuse(self):
        """Chosen items can be reused with no_reuse=False."""
        items = self._make_items([5, 1, 1, 1, 1])
        result = pair_by_margin(items, margin=3, min_pairs=4, no_reuse=False)

        # The single 5-star item should be reused to pair with multiple 1-star items
        assert len(result.pairs) >= 1

    def test_pair_by_margin_max_pairs(self):
        """Max pairs caps output."""
        items = self._make_items([5, 5, 5, 1, 1, 1])
        result = pair_by_margin(items, margin=3, max_pairs=2)

        assert len(result.pairs) <= 2

    def test_pair_by_margin_with_chosen_filter(self):
        """Chosen filter restricts chosen pool."""
        items = self._make_items([5, 4, 3, 2, 1])
        result = pair_by_margin(items, margin=2, chosen_filter=StarFilter(5, "=="))

        for pair in result.pairs:
            assert pair.chosen.score == 5

    def test_pair_by_margin_with_rejected_filter(self):
        """Rejected filter restricts rejected pool."""
        items = self._make_items([5, 4, 3, 2, 1])
        result = pair_by_margin(items, margin=2, rejected_filter=StarFilter(2, "<="))

        for pair in result.pairs:
            assert pair.rejected.score <= 2

    def test_pair_by_margin_each_rejected_used_once(self):
        """Each rejected item is used at most once (O(n) algorithm)."""
        items = self._make_items([5, 5, 5, 1, 1, 1])
        result = pair_by_margin(items, margin=3)

        rejected_ids = [p.rejected.id for p in result.pairs]
        assert len(rejected_ids) == len(set(rejected_ids))  # No duplicates

    def test_pair_by_margin_invalid_margin(self):
        """Raises ValueError for invalid margin."""
        items = self._make_items([5, 1])
        with pytest.raises(ValueError, match="margin must be >= 1"):
            pair_by_margin(items, margin=0)

    def test_pair_by_margin_invalid_min_max(self):
        """Raises ValueError when min_pairs > max_pairs."""
        items = self._make_items([5, 1])
        with pytest.raises(ValueError, match="min_pairs.*cannot be greater than max_pairs"):
            pair_by_margin(items, margin=2, min_pairs=10, max_pairs=5)


class TestStarFilter:
    """Tests for StarFilter parsing and matching."""

    def test_parse_exact(self):
        """Parse exact match filter."""
        f = StarFilter.parse("3")
        assert f.value == 3
        assert f.op == "=="
        assert f.matches(3) is True
        assert f.matches(4) is False

    def test_parse_greater_equal(self):
        """Parse >= filter."""
        f = StarFilter.parse(">=3")
        assert f.value == 3
        assert f.op == ">="
        assert f.matches(3) is True
        assert f.matches(4) is True
        assert f.matches(2) is False

    def test_parse_less_equal(self):
        """Parse <= filter."""
        f = StarFilter.parse("<=2")
        assert f.value == 2
        assert f.op == "<="
        assert f.matches(2) is True
        assert f.matches(1) is True
        assert f.matches(3) is False

    def test_parse_none(self):
        """Parse None returns None."""
        assert StarFilter.parse(None) is None

    def test_parse_int(self):
        """Parse int returns exact match."""
        f = StarFilter.parse(4)
        assert f.value == 4
        assert f.op == "=="
