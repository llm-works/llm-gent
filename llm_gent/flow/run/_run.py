# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One execution of a top-level Flow: the state that belongs to it, not to the flow.

A :class:`~llm_gent.flow.Flow` is the structure of the work and can run
many times at once. Everything one :meth:`~llm_gent.flow.Flow.run` call
writes while it goes lives on its :class:`Run`: the live scopes and cursors
its checkpoints are built from, the history writer, the halt and signals it
observes, where its halt was first seen and its hook bookkeeping. The run
reaches every step through the walk's frame (``_Frame.run``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..flow import Flow
from ..state.snapshot import ScopeRegistry


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ..hooks import RunInfo
    from ._checkpoint_ctx import CheckpointContext
    from ._halt_observer import HaltPoint
    from ._hook_points import RunHooks


@dataclass(eq=False)
class Run:
    """The state of one :meth:`~llm_gent.flow.Flow.run` call.

    ``flow`` is the top-level flow being run. ``halt`` and ``signals`` are
    the events the run observes; ``checkpoint_ctx`` writes its history, or
    is ``None`` without a checkpoint store. ``scopes`` holds the run's live
    scopes and cursors. ``halt_at`` records where the halt was first
    observed, for the halt commit. ``hooks`` and ``info`` are the run's
    hook events and identity when any flow of the run has hooks;
    ``resumed_from`` is the commit the run checked out.
    """

    flow: Flow
    halt: asyncio.Event | None
    signals: Mapping[str, asyncio.Event]
    checkpoint_ctx: CheckpointContext | None
    scopes: ScopeRegistry = field(default_factory=ScopeRegistry)
    halt_at: HaltPoint | None = None
    hooks: RunHooks | None = None
    info: RunInfo | None = None
    resumed_from: str | None = None

    @property
    def lg(self) -> Logger:
        """The top-level flow's logger."""
        return self.flow._lg
