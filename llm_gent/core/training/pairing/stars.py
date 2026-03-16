"""Star-based preference pairing.

Specialization for integer star ratings (1-5 scale).
Provides margin-based and threshold-based pairing algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import PairingResult, PreferencePair, RatedItem


# Type alias for star-rated items
StarRatedItem = RatedItem[int]
StarPreferencePair = PreferencePair[int]
StarPairingResult = PairingResult[int]


@dataclass
class StarFilter:
    """Filter for star ratings with comparison operator.

    Examples:
        StarFilter.parse("3")   -> exact match (== 3)
        StarFilter.parse(">=3") -> minimum (>= 3)
        StarFilter.parse("<=2") -> maximum (<= 2)
        StarFilter.parse(">1")  -> greater than (> 1)
        StarFilter.parse("<5")  -> less than (< 5)
    """

    value: int
    op: str = "=="  # "==", ">=", "<=", ">", "<"

    def matches(self, stars: int) -> bool:
        """Check if stars value matches this filter."""
        if self.op == "==":
            return stars == self.value
        elif self.op == ">=":
            return stars >= self.value
        elif self.op == "<=":
            return stars <= self.value
        elif self.op == ">":
            return stars > self.value
        elif self.op == "<":
            return stars < self.value
        raise ValueError(f"Unsupported operator for StarFilter: {self.op!r}")

    @classmethod
    def parse(cls, value: str | int | None) -> StarFilter | None:
        """Parse a star filter from string or int.

        Args:
            value: "3" (exact), ">=3" (min), "<=2" (max), or int for exact match

        Returns:
            StarFilter instance or None if value is None/empty

        Raises:
            ValueError: If value is not a valid star filter format
        """
        if value is None:
            return None
        if isinstance(value, int):
            return cls(value=value, op="==")

        value = value.strip()
        if not value:
            return None

        try:
            for op in (">=", "<=", ">", "<"):
                if value.startswith(op):
                    num = int(value[len(op) :])
                    return cls(value=num, op=op)
            return cls(value=int(value), op="==")
        except ValueError:
            raise ValueError(
                f"Invalid star filter: '{value}'. "
                f"Use a number (3) or operator with number (>=3, <=2, >1, <5)"
            ) from None


def _validate_pairing_params(margin: int, min_pairs: int | None, max_pairs: int | None) -> None:
    """Validate pairing parameters."""
    if margin < 1:
        raise ValueError(f"margin must be >= 1, got {margin}")
    if min_pairs is not None and min_pairs < 0:
        raise ValueError(f"min_pairs cannot be negative, got {min_pairs}")
    if max_pairs is not None and max_pairs < 0:
        raise ValueError(f"max_pairs cannot be negative, got {max_pairs}")
    if min_pairs is not None and max_pairs is not None and min_pairs > max_pairs:
        raise ValueError(f"min_pairs ({min_pairs}) cannot be greater than max_pairs ({max_pairs})")


def pair_by_margin(
    rated_items: list[StarRatedItem],
    margin: int = 1,
    min_pairs: int | None = None,
    max_pairs: int | None = None,
    no_reuse: bool = False,
    chosen_filter: StarFilter | None = None,
    rejected_filter: StarFilter | None = None,
    length_epsilons: list[int | None] | None = None,
) -> StarPairingResult:
    """Pair highest-rated with lowest-rated, respecting minimum margin.

    This is the "relative" pairing strategy - pairs are formed based on
    the relative difference in ratings rather than fixed thresholds.

    Args:
        rated_items: List of star-rated items to pair.
        margin: Minimum star difference for valid pairs (default: 1).
        min_pairs: Target number of pairs (reuses chosen if needed).
        max_pairs: Maximum pairs to return (caps output).
        no_reuse: If True, each chosen item is used at most once (1:1 pairing).
        chosen_filter: Filter for chosen pool (e.g., StarFilter(4, ">=")).
        rejected_filter: Filter for rejected pool (e.g., StarFilter(2, "<=")).
        length_epsilons: Multi-pass length balancing. List of max length differences
            (in chars) for each pass. E.g., [5, 15, 30, None] means:
            - Pass 1: only pairs where |len(chosen) - len(rejected)| <= 5
            - Pass 2: remaining items with <= 15 char difference
            - Pass 3: remaining items with <= 30 char difference
            - Pass 4: any remaining valid pairs (no length constraint)
            This prevents DPO from learning "longer = better" as spurious correlation.

    Returns:
        PairingResult with pairs and metadata.

    Raises:
        ValueError: If margin < 1 or min/max_pairs are invalid.
    """
    _validate_pairing_params(margin, min_pairs, max_pairs)

    if not rated_items:
        return StarPairingResult(
            pairs=[], total_rated=0, strategy="margin", params={"margin": margin}
        )

    chosen_pool, rejected_pool = _build_pools(rated_items, margin, chosen_filter, rejected_filter)

    if not chosen_pool or not rejected_pool:
        return StarPairingResult(
            pairs=[],
            total_rated=len(rated_items),
            strategy="margin",
            params={"margin": margin},
        )

    target = min_pairs if min_pairs is not None else len(chosen_pool)
    pairs = _generate_pairs(chosen_pool, rejected_pool, margin, target, no_reuse, length_epsilons)

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    return StarPairingResult(
        pairs=pairs,
        total_rated=len(rated_items),
        strategy="margin",
        params={"margin": margin, "length_epsilons": length_epsilons},
    )


def pair_by_threshold(
    rated_items: list[StarRatedItem],
    high_threshold: int = 4,
    low_threshold: int = 2,
    max_pairs: int | None = None,
) -> StarPairingResult:
    """Pair based on fixed star thresholds.

    Chosen items must be at or above high_threshold, rejected items
    must be at or below low_threshold.

    Args:
        rated_items: List of star-rated items to pair.
        high_threshold: Minimum stars for chosen pool (default: 4).
        low_threshold: Maximum stars for rejected pool (default: 2).
        max_pairs: Maximum pairs to return (caps output).

    Returns:
        PairingResult with pairs and metadata.

    Raises:
        ValueError: If high_threshold <= low_threshold.
    """
    if high_threshold <= low_threshold:
        raise ValueError(
            f"high_threshold ({high_threshold}) must be greater than "
            f"low_threshold ({low_threshold})"
        )
    if max_pairs is not None and max_pairs < 0:
        raise ValueError(f"max_pairs cannot be negative, got {max_pairs}")

    params = {"high_threshold": high_threshold, "low_threshold": low_threshold}

    if not rated_items:
        return StarPairingResult(pairs=[], total_rated=0, strategy="threshold", params=params)

    chosen_pool = [item for item in rated_items if item.score >= high_threshold]
    chosen_ids = {item.id for item in chosen_pool}
    rejected_pool = [
        item for item in rated_items if item.score <= low_threshold and item.id not in chosen_ids
    ]

    pairs: list[StarPreferencePair] = []
    for chosen, rejected in zip(chosen_pool, rejected_pool, strict=False):
        pairs.append(PreferencePair(chosen=chosen, rejected=rejected))

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    return StarPairingResult(
        pairs=pairs, total_rated=len(rated_items), strategy="threshold", params=params
    )


def _build_pools(
    sorted_items: list[StarRatedItem],
    margin: int,
    chosen_filter: StarFilter | None,
    rejected_filter: StarFilter | None,
) -> tuple[list[StarRatedItem], list[StarRatedItem]]:
    """Build chosen and rejected pools based on margin requirement."""
    if not sorted_items:
        return [], []

    min_score = min(item.score for item in sorted_items)
    max_score = max(item.score for item in sorted_items)

    chosen_pool = [item for item in sorted_items if min_score <= item.score - margin]
    rejected_pool = [item for item in sorted_items if max_score >= item.score + margin]

    if chosen_filter is not None:
        chosen_pool = [item for item in chosen_pool if chosen_filter.matches(item.score)]
    if rejected_filter is not None:
        rejected_pool = [item for item in rejected_pool if rejected_filter.matches(item.score)]

    chosen_pool.sort(key=lambda x: (-x.score, x.id))
    rejected_pool.sort(key=lambda x: (x.score, x.id))

    return chosen_pool, rejected_pool


def _generate_pairs(
    chosen_pool: list[StarRatedItem],
    rejected_pool: list[StarRatedItem],
    margin: int,
    target: int,
    no_reuse: bool,
    length_epsilons: list[int | None] | None = None,
) -> list[StarPreferencePair]:
    """Generate pairs, optionally using multi-pass length balancing.

    If length_epsilons is provided, uses multi-pass pairing to minimize length bias.
    """
    if not chosen_pool or not rejected_pool:
        return []

    if length_epsilons is None:
        return _generate_pairs_single_pass(
            chosen_pool, rejected_pool, margin, target, no_reuse, length_epsilon=None
        )

    return _generate_pairs_multi_pass(
        chosen_pool, rejected_pool, margin, target, no_reuse, length_epsilons
    )


def _generate_pairs_multi_pass(
    chosen_pool: list[StarRatedItem],
    rejected_pool: list[StarRatedItem],
    margin: int,
    target: int,
    no_reuse: bool,
    length_epsilons: list[int | None],
) -> list[StarPreferencePair]:
    """Multi-pass pairing with progressively looser length constraints."""
    pairs: list[StarPreferencePair] = []
    used_chosen: set[int] = set()
    used_rejected: set[int] = set()

    for epsilon in length_epsilons:
        if len(pairs) >= target:
            break

        available_chosen = [c for c in chosen_pool if c.id not in used_chosen or not no_reuse]
        available_rejected = [r for r in rejected_pool if r.id not in used_rejected]

        if not available_chosen or not available_rejected:
            break

        pass_pairs = _generate_pairs_single_pass(
            available_chosen,
            available_rejected,
            margin,
            target - len(pairs),
            no_reuse,
            length_epsilon=epsilon,
            used_chosen=used_chosen if no_reuse else None,
        )

        _track_used_items(pairs, pass_pairs, used_chosen, used_rejected, no_reuse)

    return pairs


def _track_used_items(
    pairs: list[StarPreferencePair],
    new_pairs: list[StarPreferencePair],
    used_chosen: set[int],
    used_rejected: set[int],
    no_reuse: bool,
) -> None:
    """Track used items from new pairs."""
    for pair in new_pairs:
        pairs.append(pair)
        used_rejected.add(pair.rejected.id)
        if no_reuse:
            used_chosen.add(pair.chosen.id)


def _generate_pairs_single_pass(
    chosen_pool: list[StarRatedItem],
    rejected_pool: list[StarRatedItem],
    margin: int,
    target: int,
    no_reuse: bool,
    length_epsilon: int | None,
    used_chosen: set[int] | None = None,
) -> list[StarPreferencePair]:
    """Generate pairs in a single pass with optional length constraint."""
    if not chosen_pool or not rejected_pool:
        return []

    pairs: list[StarPreferencePair] = []
    local_used_chosen: set[int] = used_chosen.copy() if used_chosen else set()
    chosen_idx = 0

    for rejected in rejected_pool:
        if len(pairs) >= target:
            break

        match = _find_chosen_match(
            rejected, chosen_pool, local_used_chosen, margin, no_reuse, chosen_idx, length_epsilon
        )
        if match:
            chosen, new_idx = match
            pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
            if no_reuse:
                local_used_chosen.add(chosen.id)
            chosen_idx = new_idx

    return pairs


def _find_chosen_match(
    rejected: StarRatedItem,
    chosen_pool: list[StarRatedItem],
    used: set[int],
    margin: int,
    no_reuse: bool,
    start_idx: int,
    length_epsilon: int | None = None,
) -> tuple[StarRatedItem, int] | None:
    """Find a chosen item that can pair with this rejected item.

    Args:
        rejected: The rejected item to find a match for.
        chosen_pool: Pool of potential chosen items.
        used: Set of already-used chosen IDs.
        margin: Minimum star difference required.
        no_reuse: If True, skip items in used set.
        start_idx: Index to start searching from (for round-robin).
        length_epsilon: If set, only match if |len(chosen) - len(rejected)| <= epsilon.

    Returns (chosen, next_idx) or None if no match found.
    """
    pool_size = len(chosen_pool)
    rejected_len = len(rejected.content)

    for offset in range(pool_size):
        idx = (start_idx + offset) % pool_size
        chosen = chosen_pool[idx]

        # Basic validity checks
        if chosen.score - rejected.score < margin:
            continue
        if chosen.id == rejected.id:
            continue
        if no_reuse and chosen.id in used:
            continue

        # Length constraint check
        if length_epsilon is not None:
            length_diff = abs(len(chosen.content) - rejected_len)
            if length_diff > length_epsilon:
                continue

        return chosen, (idx + 1) % pool_size

    return None
