# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Budget accounting for agent runs.

Currency-agnostic, hierarchical cost accounting with cap enforcement
and optional halt integration. Consumers wire pricing from their own
config (yaml, environment, ...) and choose the currency; the tracker
multiplies rates by usage and returns the same unit.

Public shape:

- :class:`Op` protocol + built-in :class:`LLMOp` and :class:`FixedOp`.
  Consumers can define their own Op shapes for domain-specific billing.
- :class:`PricingConfig` — name-indexed :class:`Op` registry.
- :class:`Tracker` — single accounting primitive. A tracker holds a
  scope-local cap and reports costs up to its parent (if any); trees
  of arbitrary depth are supported. Each level has independent halt
  and observer callback semantics.
- :class:`CostCallback` — per-tracker hook consumers use to build
  observability (aggregations, audit logs, dashboards).
"""

from .pricing import FixedOp, LLMOp, Op, PricingConfig
from .tracker import CostCallback, Tracker


__all__ = [
    "CostCallback",
    "FixedOp",
    "LLMOp",
    "Op",
    "PricingConfig",
    "Tracker",
]
