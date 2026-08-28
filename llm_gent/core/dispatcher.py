# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Event dispatcher for agent execution.

Provides event routing and handler registration for declarative agent behavior.
Separate from Agent base class - this is a composable component that agents can use.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


EventHandler = Callable[..., Awaitable[Any]]


class Dispatcher:
    """Event dispatcher for agent execution.

    Enables declarative agent behavior through event routing:
    - Register handlers for events (schedule, question, etc.)
    - Trigger events with context-specific data
    - Handlers can use different memory strategies per event

    This is a composable component that agents use internally,
    NOT a replacement for the Agent base class.

    Example:
        dispatcher = Dispatcher()

        async def on_schedule(task: str, **ctx):
            # Use chronological recall for scheduled tasks
            past = recall_chronological(learn_trait, agent_name, limit=5)
            context = format_solutions_context(past)
            return await saia.complete(f"{context}\\n\\n{task}")

        dispatcher.on("schedule", on_schedule)
        result = await dispatcher.trigger("schedule", task="Tell a joke")
    """

    def __init__(self) -> None:
        """Initialize dispatcher."""
        self._handlers: dict[str, EventHandler] = {}

    def on(self, event: str, handler: EventHandler) -> None:
        """Register event handler.

        Args:
            event: Event name (e.g., "schedule", "question").
            handler: Async function to call when event is triggered.
        """
        self._handlers[event] = handler

    async def trigger(self, event: str, **kwargs: Any) -> Any:
        """Trigger event and execute handler.

        Args:
            event: Event name to trigger.
            **kwargs: Arguments to pass to the handler.

        Returns:
            Result from the handler.

        Raises:
            ValueError: If no handler is registered for the event.
        """
        handler = self._handlers.get(event)
        if not handler:
            raise ValueError(f"No handler registered for event: {event}")
        return await handler(**kwargs)

    def has_handler(self, event: str) -> bool:
        """Check if a handler is registered for an event.

        Args:
            event: Event name to check.

        Returns:
            True if a handler is registered.
        """
        return event in self._handlers
