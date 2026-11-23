# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent execution result."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result from agent execution.

    ``session`` is the id of the session the execution ran in: pass it to the
    next ``ask`` to continue that conversation. Empty when it ran in none
    (``run_once``).
    """

    success: bool
    content: str
    iterations: int = 1
    tokens_used: int = 0
    latency_ms: int = 0
    trace_id: str = ""
    session: str = ""
