# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Preference pairing infrastructure for DPO training.

Provides generic types for preference pairing and specializations
for common rating schemes.

Generic types (work with any score type):
- RatedItem[S]: An item with a score of type S
- PreferencePair[S]: A chosen/rejected pair
- PairingResult[S]: Result of a pairing operation

Star-based specialization (integer 1-5 ratings):
- StarRatedItem, StarPreferencePair, StarPairingResult: Type aliases
- StarFilter: Filter with comparison operators (>=, <=, etc.)
- pair_by_margin: Pair based on relative score difference
- pair_by_threshold: Pair based on fixed score thresholds

Example:
    from llm_gent.core.training.pairing import (
        StarRatedItem,
        StarFilter,
        pair_by_margin,
    )

    # Build rated items from your data source
    items = [
        StarRatedItem(id=1, content="Great", score=5),
        StarRatedItem(id=2, content="Okay", score=3),
        StarRatedItem(id=3, content="Bad", score=1),
    ]

    # Pair with margin-based algorithm
    result = pair_by_margin(items, margin=2)
"""

# Generic types
# Star-based specialization
from .stars import (
    StarFilter,
    StarPairingResult,
    StarPreferencePair,
    StarRatedItem,
    pair_by_margin,
    pair_by_threshold,
)
from .types import PairingResult, PreferencePair, RatedItem


__all__ = [
    # Generic types
    "RatedItem",
    "PreferencePair",
    "PairingResult",
    # Star-based types
    "StarRatedItem",
    "StarPreferencePair",
    "StarPairingResult",
    "StarFilter",
    # Star-based algorithms
    "pair_by_margin",
    "pair_by_threshold",
]
