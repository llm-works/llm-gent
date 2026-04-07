"""Handler protocol for agent runner dispatch.

Defines the callback interface that runners use to dispatch incoming
requests. Implementations handle the actual business logic; the runner
handles bus connectivity and protocol plumbing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Handler(Protocol):
    """Protocol for handling swarm requests.

    Implement this to define how an agent responds to incoming messages
    from the hub. The runner calls these methods when it receives the
    corresponding request type on the bus.
    """

    def on_ask(self, question: str) -> str:
        """Handle an ask request.

        Args:
            question: The question text from the hub or another agent.

        Returns:
            Response text.
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
