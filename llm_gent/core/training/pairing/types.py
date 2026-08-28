# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Generic types for preference pairing.

These types are generic over the score type - agents define what "rating" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


# Score type - could be int (stars), float, or any comparable
S = TypeVar("S")


@dataclass
class RatedItem(Generic[S]):
    """A rated item with a generic score type.

    Agents populate this from their own data sources. The framework doesn't
    care how you got the data or what the score represents.

    Attributes:
        id: Unique identifier for this item.
        content: The content text (e.g., joke, response, summary).
        score: Rating value (type defined by agent - int, float, etc.).
        metadata: Agent-specific data (e.g., model used, adapter info).
    """

    id: int
    content: str
    score: S
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair(Generic[S]):
    """A preference pair for DPO training.

    Pairs a higher-rated ("chosen") item with a lower-rated ("rejected") item.

    Attributes:
        chosen: The preferred (higher-rated) content.
        rejected: The less preferred (lower-rated) content.
    """

    chosen: RatedItem[S]
    rejected: RatedItem[S]


@dataclass
class PairingResult(Generic[S]):
    """Result of a pairing operation.

    Attributes:
        pairs: List of preference pairs generated.
        total_rated: Total number of rated items in the input.
        strategy: Pairing strategy name (e.g., "margin", "threshold").
        params: Strategy-specific parameters used.
    """

    pairs: list[PreferencePair[S]]
    total_rated: int
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
