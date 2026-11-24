# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Chat: an agent answers questions, each in a session it issues.

An agent supports chat by mixing :class:`Chat` in (``class Reviewer(Agent, Chat)``);
the runtime refuses an ask to an agent that does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .types import ExecutionResult


__all__ = ["AskStep", "Chat", "OnAskStep"]


@dataclass(frozen=True)
class AskStep:
    """Progress while an ask runs: ``name`` says what happened, ``data`` the details."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)


OnAskStep = Callable[[AskStep], None]
"""Takes the steps of an ask as the agent reports them."""


class Chat(ABC):
    """An agent that answers questions, each in a session it issues.

    An agent that takes more than a question adds keyword-only parameters
    of its own to :meth:`ask`; callers over the wire send them as a
    ``kwargs`` dict, passed through unread.
    """

    @abstractmethod
    def ask(
        self, question: str, session: str | None = None, *, on_step: OnAskStep | None = None
    ) -> ExecutionResult:
        """Answer ``question`` in a session.

        Args:
            question: The question to answer.
            session: The id of the session to continue, as an earlier answer
                returned it; ``None`` opens a new session.
            on_step: Called with each step of the ask while it runs.

        Returns:
            The answer; its ``session`` continues the conversation.

        Raises:
            SessionNotFoundError: ``session`` is not a session of this agent.
        """
        ...
