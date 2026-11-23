# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One run: what any execution a runner starts has, whether it walks a flow or not.

A run executes in a task of its own; the caller awaits it for its result,
holds it, or halts it. Every run reports to a logger and uses an
:class:`~llm_gent.core.env.Environment`. What else it carries is its
kind's: a flow run (:class:`llm_gent.flow.FlowRun`) adds its signals, its
history writer and its scopes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger

from .env import Environment


@dataclass(eq=False, kw_only=True)
class Run:
    """One execution: await it for its result, ``halt()`` it, check ``done``.

    ``lg`` is the logger of the runner that started it and ``env`` what it
    may use. ``halt_event`` is the event it observes to stop: set it, or call
    :meth:`halt`. ``task`` executes the run once it has started. Cancelling
    the task that awaits the run cancels the run.
    """

    lg: Logger
    env: Environment
    halt_event: asyncio.Event | None = None
    task: asyncio.Task[Any] | None = None

    def halt(self) -> None:
        """Halt the run: it stops at the next place it looks, as when its halt event is set.

        A run started without a halt event gets one now; work already
        running reads it from its next check on.
        """
        if self.halt_event is None:
            self.halt_event = asyncio.Event()
        self.halt_event.set()

    @property
    def done(self) -> bool:
        """The run has ended: returned, raised or been cancelled."""
        return self.task is not None and self.task.done()

    def __await__(self) -> Generator[Any, None, Any]:
        if self.task is None:
            raise RuntimeError("the run has not started")
        return self.task.__await__()
