# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The runner: executes flows, one :class:`~._run.Run` per execution.

A flow is the structure of the work and never runs itself. A runner holds
what its runs share and starts them; each run is given its own inputs and
keeps everything it writes while it goes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from appinfra.cas import Store
from appinfra.log import Logger

from ...core.env import Environment
from ..checkpoint import Retention
from ..flow import Flow
from ..nodes import UNSET
from ._flow_run import Wiring, start_run
from ._run import Run
from .start import FRESH, Start


class Runner:
    """Executes flows: :meth:`start` begins a run and returns it.

    Exported as :class:`llm_gent.flow.FlowRunner`. ``lg`` is the logger
    every run it starts reports to; ``env`` is what every one of them may
    use (``ctx.env``, ``ctx.saia``, ``ctx.traits``), and runs sharing it
    share its SAIA instances, one per role. Without one, the runs have an
    empty environment of this runner's: no traits, and ``ctx.saia`` raises
    for a role-bound verb. ``store`` is the :class:`appinfra.cas.Store` the
    runs keep their histories in, each in the repository its ``history=``
    names; without one, a run given a history raises.
    """

    def __init__(
        self,
        lg: Logger,
        *,
        env: Environment | None = None,
        store: Store | None = None,
    ) -> None:
        self._wiring = Wiring(lg, env if env is not None else Environment(), store)

    @property
    def env(self) -> Environment:
        """What the runs this runner starts may use."""
        return self._wiring.env

    @property
    def store(self) -> Store | None:
        """The store the runs this runner starts keep their histories in."""
        return self._wiring.store

    def start(
        self,
        flow: Flow,
        *args: Any,
        state: Any = UNSET,
        halt: asyncio.Event | None = None,
        signals: Mapping[str, asyncio.Event] | None = None,
        history: str | None = None,
        retention: Retention = "retain",
        start: Start = FRESH,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Run:
        """Start a run of ``flow`` and return it; ``await`` the run for its result.

        Needs a running event loop: the run executes in a task of its own.
        The first node receives ``*args`` / ``**kwargs``; each later node
        receives the previous node's result, optionally transformed by its
        ``project`` callable. Cancelling the task that awaits the run
        cancels the run; :class:`asyncio.CancelledError` raised by any node
        propagates out unchanged, past any ``rescue`` policy.

        Args:
            flow: The flow to run. A flow can run many times, at once too.
            *args: Positional inputs to the first node.
            state: The payload ``ctx.state`` exposes for this run, wrapped as
                :class:`~llm_gent.flow.State` (``ctx.state.data`` reaches
                the payload; ``ctx.state.root()`` the outermost scope from
                any subflow). Omitted → a fresh state of this run's own:
                ``state_factory.new()`` when the flow has a state factory,
                else an empty ``dict``. A run that starts from a commit
                takes its state from the commit; ``state`` is used only on
                an empty history. Explicit ``None`` is honored (verbs must
                guard). Not forwarded to the first node.
            halt: The run's halt: set it, or call :meth:`Run.halt`, to pause
                the whole run. Every step sees it as ``ctx.halt``; a chain
                checks it after each step, an iterate before each pass, a
                map before each item. In-flight bodies are not interrupted.
                A halted run returns :data:`~llm_gent.flow.HALTED`; its
                state is in the halt checkpoint.
            signals: The run's signals by name, events the app sets for
                :meth:`Flow.with_shortcut` regions. Which are set is
                recorded in every checkpoint the run takes, and a resumed
                run sets them again before the first step.
            history: The name of the run's history: the repository of the
                runner's ``store`` its commits go to, and the one
                ``start`` reads. Owned by the caller; the framework never
                assigns one. With a history the run writes its halt
                checkpoint and, on a clean exit, its final state; saves
                inside the run (``ctx.checkpoint()``, the checkpoint
                policy) need a :meth:`Flow.with_checkpointer` on the saving
                flow or above it. ``None``: the run keeps no history. One
                history has one writer at a time.
            retention: What a fully successful run does with its history:
                ``"retain"`` keeps it and commits the final state tagged
                ``complete``; ``"gc_on_success"`` deletes it. Cancellation,
                a halt and an unhandled exception keep it either way.
            start: :class:`~llm_gent.flow.Fresh` (the default),
                :class:`~llm_gent.flow.Resume` or
                :class:`~llm_gent.flow.Restart`: whether the run starts
                from ``state`` or from a commit of its history. Not
                forwarded to the first node.
            extra: Caller-supplied opaque dict reachable as ``ctx.extra`` on
                every dispatch inside the run. The framework neither
                inspects nor persists it; a resumed run is given it again.
            **kwargs: Keyword inputs to the first node.

        Returns:
            The run. Awaiting it gives the last step's result, or
            :data:`~llm_gent.flow.HALTED` when the halt stopped the run
            before its end. A halt set during the last step, which
            completed, does not stop the run. A run that raises writes no
            commit: its history's head stays its last save.

        Raises:
            RuntimeError: No running event loop; the flow has no nodes; a
                ``history`` on a runner without a ``store``; a
                ``start`` from a commit without a ``history``; a flow that
                declares :meth:`Flow.with_checkpointer` in a run without a
                ``history``; a shortcut's signal that is not one of the
                run's ``signals``, or a region holding another on the same
                signal; a ``with_budget(...)`` with no cost tracker on its
                flow or above it.
            ValueError: A signal's name is not a non-empty str;
                ``retention`` is not ``"retain"`` or ``"gc_on_success"``.
            TypeError: ``start`` is not a :data:`~llm_gent.flow.Start`;
                ``signals`` is not a mapping, or a signal is not an
                :class:`asyncio.Event`.
            appinfra.cas.InvalidNameError: ``history`` cannot name a
                repository.

            Awaiting the run raises what the run raised, among them
            ``ValueError`` for a ``start`` naming a commit or checkpoint
            the history does not have, or a commit without state.
        """
        return start_run(
            self._wiring,
            flow,
            args,
            kwargs,
            state=state,
            halt=halt,
            signals=signals,
            history=history,
            retention=retention,
            start=start,
            extra=extra,
        )
