# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Merging llm-infer router configs."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def merge_llm_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``override`` merged into a copy of ``base``.

    Backends merge one level deep: a backend named in both keeps ``base``'s
    settings and takes ``override``'s on top; a backend set to ``None`` in
    ``override`` (a YAML block with its content commented out) is skipped.
    Other top-level keys in ``override`` replace ``base``'s.
    """
    result = copy.deepcopy(dict(base))
    backends = override.get("backends")
    if backends is not None:
        merged: dict[str, Any] = result.get("backends") or {}
        for name, settings in backends.items():
            if settings is None:
                continue
            current = merged.get(name)
            if current is None:
                merged[name] = copy.deepcopy(settings)
            elif settings:
                current.update(copy.deepcopy(settings))
        result["backends"] = merged
    for key, value in override.items():
        if key != "backends":
            result[key] = copy.deepcopy(value)
    return result
