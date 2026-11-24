# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Handler protocol for agent runner dispatch.

Defines the callback interface that runners use to dispatch incoming
requests. Implementations handle the actual business logic; the runner
handles bus connectivity and protocol plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..bus.protocol import AskResponse
from ..core.agent.chat import OnAskStep


@dataclass(frozen=True)
class Answer:
    """What an ask returns: the response text and the id of the session it ran in."""

    response: str
    session: str


def answer(name: str, reply: Any) -> Answer:
    """The answer in agent ``name``'s reply to an ask.

    Raises:
        RuntimeError: The reply is a failure, or no answer to an ask.
    """
    if isinstance(reply, AskResponse):
        if not reply.success:
            raise RuntimeError(f"Agent {name} ask failed: {reply.error}")
        return Answer(response=reply.response, session=reply.session)
    raise RuntimeError(f"Agent {name} did not answer the ask: {getattr(reply, 'error', reply)}")


@runtime_checkable
class Handler(Protocol):
    """Protocol for handling swarm requests.

    Implement this to define how an agent responds to incoming messages
    from the hub. The runner calls these methods when it receives the
    corresponding request type on the bus.
    """

    def on_ask(
        self,
        question: str,
        session: str | None,
        kwargs: dict[str, Any],
        on_step: OnAskStep | None,
    ) -> Answer:
        """Handle an ask request.

        Args:
            question: The question text from the hub or another agent.
            session: The session to continue, as an earlier answer returned
                it; ``None`` opens a new one.
            kwargs: Keyword arguments of the agent's own ``ask``.
            on_step: For a streamed ask, takes each step while it runs.

        Returns:
            The answer and its session.
        """
        ...

    def on_feedback(self, message: str) -> None:
        """Handle a feedback message.

        Args:
            message: Feedback text from the hub or another agent.
        """
        ...

    def on_shutdown(self) -> None:
        """Handle a shutdown request.

        Called when the hub requests this agent to shut down.
        Perform any cleanup before the runner disconnects from the bus.
        """
        ...
