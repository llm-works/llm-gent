# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Runnable interface for executable agents.

Defines the interface that the runtime runner expects from anything it can run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .agent.types import ExecutionResult


__all__ = ["Runnable", "ExecutionResult"]


class Runnable(ABC):
    """Interface for anything the runner can execute.

    Defines lifecycle and execution methods that the runtime runner needs.
    Agent extends this with trait management capabilities.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for this runnable."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Start the runnable. Must be called before execution methods."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the runnable. Clean up resources."""
        ...

    @abstractmethod
    def run_once(self) -> ExecutionResult:
        """Execute one cycle of the default task.

        Returns:
            ExecutionResult with success status and content.
        """
        ...

    @abstractmethod
    def ask(self, question: str) -> str:
        """Ask a question and get a response.

        Args:
            question: The question to ask.

        Returns:
            Response string.
        """
        ...

    @property
    @abstractmethod
    def cycle_count(self) -> int:
        """Number of execution cycles completed."""
        ...
