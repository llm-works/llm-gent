# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Default agent - config-driven, no learning."""

from .agent import Agent
from .factory import Factory


__all__ = [
    "Agent",
    "Factory",
]
