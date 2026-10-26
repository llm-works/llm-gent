# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Provider cost + cached-token extraction from raw LLM responses.

Some providers ship the authoritative cost directly in the response
payload (xAI reports ``usage.cost_in_usd_ticks``, one tick = 1e-10 USD).
When present, that value beats any rate-card estimate — the invoice will
match it, not the rate card. Cached-token counts also live in different
shapes per provider (OpenAI-compat under ``prompt_tokens_details``;
Anthropic under ``cache_read_input_tokens``); this module normalizes the
read.
"""

from __future__ import annotations

import math
from typing import Any


__all__ = ["ProviderCostExtractor"]


def _to_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _to_valid_cost(value: float) -> float | None:
    if not math.isfinite(value) or value < 0:
        return None
    return value


class ProviderCostExtractor:
    """Extract provider-reported cost + cached tokens from raw responses.

    Both methods accept the raw response dict (``ChatResponse.raw`` in
    llm-infer parlance, or any equivalent provider-shape mapping) and
    return ``None`` / ``0`` when the field isn't present or isn't
    parseable. The per-provider ``_extract_*`` helpers are chained
    through :meth:`extract` so a new provider that starts reporting cost
    lands as one added branch.
    """

    @staticmethod
    def extract(raw: dict[str, Any] | None) -> float | None:
        """Return the USD cost reported by the provider, or ``None`` if absent.

        Only xAI reports cost in-band today (``usage.cost_in_usd_ticks``,
        1 tick = 1e-10 USD). Add per-provider branches here when others
        follow.
        """
        if raw is None:
            return None
        return ProviderCostExtractor._extract_xai(raw)

    @staticmethod
    def _extract_xai(raw: dict[str, Any]) -> float | None:
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return None
        cost_ticks = usage.get("cost_in_usd_ticks")
        if cost_ticks is None or isinstance(cost_ticks, bool):
            return None
        try:
            cost = float(cost_ticks) / 10_000_000_000
        except (TypeError, ValueError):
            return None
        return _to_valid_cost(cost)

    @staticmethod
    def extract_cached_tokens(raw: dict[str, Any] | None) -> int:
        """Return the cached-token count from raw, or ``0`` if absent.

        Reads OpenAI-compat's ``usage.prompt_tokens_details.cached_tokens``
        first, then Anthropic's ``usage.cache_read_input_tokens``.
        """
        if raw is None:
            return 0
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return 0
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            parsed = _to_non_negative_int(details.get("cached_tokens"))
            if parsed is not None:
                return parsed
        parsed = _to_non_negative_int(usage.get("cache_read_input_tokens"))
        if parsed is not None:
            return parsed
        return 0
