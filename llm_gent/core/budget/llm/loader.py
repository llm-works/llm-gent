# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Load a :class:`PricingConfig` from a yaml-shaped costs dict.

Lets any OSS agent consume its own rate card (yaml, env, wherever) without
re-implementing the parsing shape. Values stay with the consumer's config;
only the mapping ``dict → PricingConfig`` is shared.

Expected input shape::

    {
        "grok-4": {
            "input_per_mtok": 2.0,
            "output_per_mtok": 10.0,
            "cached_input_per_mtok": 0.5,   # optional
        },
        "claude-haiku-4-5": {
            "input_per_mtok": 0.80,
            "output_per_mtok": 4.00,
            "cached_input_per_mtok": 0.08,
        },
    }

``input_per_mtok`` and ``output_per_mtok`` are required per entry;
``cached_input_per_mtok`` is optional (when unset, cached tokens bill at
the input rate — matches :class:`LLMOp` semantics).

Callers wanting op-typed entries beyond LLM models can construct a
:class:`PricingConfig` directly and mix in :class:`FixedOp` or custom
:class:`Op` implementations.
"""

from __future__ import annotations

import math
from typing import Any, cast

from ..pricing import LLMOp, Op, PricingConfig


__all__ = ["load_pricing_config"]


def load_pricing_config(costs: dict[str, Any]) -> PricingConfig:
    """Build a :class:`PricingConfig` of :class:`LLMOp` entries from ``costs``.

    Rejects unknown top-level types and negative / non-finite rates.
    """
    if not isinstance(costs, dict):
        raise TypeError(f"costs must be a dict, got {type(costs).__name__}")

    ops: dict[str, Op] = {}
    for model_name, entry in costs.items():
        if not isinstance(model_name, str):
            raise TypeError(f"model name must be a string, got {type(model_name).__name__}")
        ops[model_name] = cast(Op, _parse_llm_op(model_name, entry))
    return PricingConfig(ops=ops)


def _parse_llm_op(model_name: str, entry: Any) -> LLMOp:
    if not isinstance(entry, dict):
        raise ValueError(f"cost entry for '{model_name}' must be a dict")
    if "input_per_mtok" not in entry or "output_per_mtok" not in entry:
        raise ValueError(
            f"cost entry for '{model_name}' requires input_per_mtok and output_per_mtok"
        )
    input_raw, output_raw = entry["input_per_mtok"], entry["output_per_mtok"]
    if isinstance(input_raw, bool) or isinstance(output_raw, bool):
        raise TypeError(f"rates for '{model_name}' must be numbers, not bool")
    input_rate = float(input_raw)
    output_rate = float(output_raw)
    if not math.isfinite(input_rate) or input_rate < 0:
        raise ValueError(f"input_per_mtok for '{model_name}' must be finite and >= 0")
    if not math.isfinite(output_rate) or output_rate < 0:
        raise ValueError(f"output_per_mtok for '{model_name}' must be finite and >= 0")
    cached_raw = entry.get("cached_input_per_mtok")
    if isinstance(cached_raw, bool):
        raise TypeError(f"cached_input_per_mtok for '{model_name}' must be a number, not bool")
    cached_rate = float(cached_raw) if cached_raw is not None else None
    if cached_rate is not None and (not math.isfinite(cached_rate) or cached_rate < 0):
        raise ValueError(f"cached_input_per_mtok for '{model_name}' must be finite and >= 0")
    return LLMOp(
        name=model_name,
        input_per_mtok=input_rate,
        output_per_mtok=output_rate,
        cached_input_per_mtok=cached_rate,
    )
