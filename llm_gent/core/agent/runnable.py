# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Runnable agent — the executable specialization of :class:`Agent`.

Bridges the container (:class:`Agent`) with the runtime-runner contract
(:class:`Runnable`). Consumers that need ``run_once`` / ``ask`` / feedback
subclass :class:`RunnableAgent`; consumers that only need traits +
lifecycle keep using :class:`Agent` directly.
"""

from __future__ import annotations

from abc import abstractmethod

from ..runnable import Runnable
from .agent import Agent
from .types import ExecutionResult


__all__ = ["RunnableAgent"]


class RunnableAgent(Agent, Runnable):
    """Agent that also implements the :class:`Runnable` contract.

    Concrete subclasses must implement :meth:`run_once`, :meth:`ask`,
    :meth:`record_feedback`, and :meth:`get_recent_results`. The
    container / lifecycle / trait surface is inherited from :class:`Agent`
    unchanged.
    """

    @abstractmethod
    def run_once(self) -> ExecutionResult:
        """Execute one cycle of the default task."""
        ...

    @abstractmethod
    def ask(self, question: str) -> str:
        """Answer one interactive question and return the response text."""
        ...

    @abstractmethod
    def record_feedback(self, message: str) -> None:
        """Record feedback from the runtime about a prior execution."""
        ...

    @abstractmethod
    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Return the most recent execution results, newest last."""
        ...
