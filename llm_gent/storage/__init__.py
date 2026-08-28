# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent storage layer for custom relational schemas.

Provides AgentTable base class and AgentStorage client for agents to define
and query their own relational data structures.
"""

from .client import AgentStorage
from .schema import AgentTable


__all__ = [
    "AgentStorage",
    "AgentTable",
]
