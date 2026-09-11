# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Batteries-included LLM pricing: opinionated helpers on top of the
substrate :class:`~llm_gent.core.budget.PricingProvider` Protocol.

- :class:`LLMPricingProvider` — publisher-strip + boundary-safe lookup +
  per-provider dispatch (Anthropic / OpenAI-compat / Gemini) + strict
  ``on_missing`` default.
- :func:`load_pricing_config` — parse a yaml-shape rate dict into a
  :class:`~llm_gent.core.budget.PricingConfig` of :class:`LLMOp` entries.
- :class:`ProviderCostExtractor` — read provider-reported costs (xAI) and
  cached-token counts from raw ChatResponse payloads.

The substrate primitives (``LLMOp``, ``PricingConfig``, ``PricingProvider``
Protocol) stay in :mod:`llm_gent.core.budget` — this subpackage adds the
opinionated pieces LLM cost tracking needs.
"""

from .extractor import ProviderCostExtractor
from .loader import load_pricing_config
from .pricing_provider import LLMPricingProvider, MissingModelCostError, OnMissing


__all__ = [
    "LLMPricingProvider",
    "MissingModelCostError",
    "OnMissing",
    "ProviderCostExtractor",
    "load_pricing_config",
]
