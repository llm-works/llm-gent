# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Configuration helper functions for agents."""

from __future__ import annotations

import os
import re
from typing import Any


def _substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Substitute {{VAR}} patterns in text with variable values.

    Falls back to environment variables if not in variables dict.

    Args:
        text: Text containing {{VAR}} patterns to substitute.
        variables: Dictionary of variable name to value mappings.

    Returns:
        Text with all {{VAR}} patterns replaced.

    Raises:
        ValueError: If a variable is not found in variables dict or environment.
    """
    pattern = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in variables:
            return variables[var_name]
        if var_name in os.environ:
            return os.environ[var_name]
        raise ValueError(f"Variable {{{{{var_name}}}}} not found in variables or environment")

    return pattern.sub(replacer, text)


def _substitute_in_dict(data: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute variables in a dict/list structure.

    Args:
        data: Data structure to process (dict, list, str, or other).
        variables: Dictionary of variable name to value mappings.

    Returns:
        Data with all {{VAR}} patterns in strings replaced.
    """
    if isinstance(data, str):
        return _substitute_variables(data, variables)
    if isinstance(data, dict):
        return {k: _substitute_in_dict(v, variables) for k, v in data.items()}
    if isinstance(data, list):
        return [_substitute_in_dict(item, variables) for item in data]
    return data
