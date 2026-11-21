# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Cost accounting for agent runs.

Currency-agnostic, hierarchical cost accounting, checked against
budgets, with optional halt integration. Consumers wire pricing from their own
config (yaml, environment, ...) and choose the currency; the tracker
multiplies rates by usage and returns the same unit.

Public shape:

- :class:`PricingProvider` — the plug seam :class:`Tracker` calls on
  every recorded event. Downstream implementations own the pricing
  math (dynamic rates, provider-specific surcharges, negotiated
  contracts, invoice reconciliation).
- :class:`PricingConfig` — default static implementation of
  :class:`PricingProvider`; a name-indexed :class:`Op` registry.
- :class:`Op` protocol + built-in :class:`LLMOp` and :class:`FixedOp`.
  Consumers can define their own Op shapes for domain-specific billing.
- :class:`Tracker` — single accounting primitive. A tracker holds a
  scope-local budget and reports costs up to its parent (if any); trees
  of arbitrary depth are supported. Each level has independent halt
  and observer callback semantics.
- :class:`CostCallback` — per-tracker hook consumers use to build
  observability (aggregations, audit logs, dashboards).

The top-level ``llm_gent`` package exports :class:`Tracker` as ``CostTracker``.
"""

from .llm import (
    LLMPricingProvider,
    MissingModelCostError,
    OnMissing,
    ProviderCostExtractor,
    load_pricing_config,
)
from .pricing import FixedOp, LLMOp, Op, PricingConfig, PricingProvider
from .tracker import CostCallback, Tracker


__all__ = [
    "CostCallback",
    "FixedOp",
    "LLMOp",
    "LLMPricingProvider",
    "MissingModelCostError",
    "Op",
    "OnMissing",
    "PricingConfig",
    "PricingProvider",
    "ProviderCostExtractor",
    "Tracker",
    "load_pricing_config",
]
