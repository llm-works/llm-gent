# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The inputs in effect at one place in a run, and how a flow derives them.

A run's inputs travel through the walk in three ways:

- fixed for the whole run: the :class:`~llm_gent.flow.run._run.Run` (halt,
  signals, history writer, live scopes) and ``extra``;
- layered: the :class:`Scope` (state, resources, checkpointer, checkpoint
  policy, hooks, shortcut regions), derived at every flow the walk enters;
- computed by the walk: the :class:`Position` (snapshot path, chain
  context, ancestor chain, activity).

The scope in effect at a place is a fold over the path to it. A node the
walk descends through (a subflow call, a branch arm, an iterate pass, a map
item) moves the position and may change the scope: it hands the child its
state, and a map item leaves its drain regions. The flow entered then
applies its :class:`~llm_gent.flow.decl.Decl` with :meth:`Scope.child`, one
rule per field:

- override: the checkpointer and the checkpoint policy; the inner
  declaration wins, an unset one inherits;
- accumulate: the shortcut regions; the flow's region adds to the
  enclosing ones;
- derive: the hooks are the enclosing hooks' children at the flow's place,
  then the flow's own, or the flow's own alone when it replaces them; the
  resources are the enclosing ones overridden by key with the flow's,
  plus their per-run children (derived and registered on entry by
  :func:`~llm_gent.flow.resource._runtime.run_resources`).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from .._node_id import _descend_context
from ..checkpoint import CheckpointPolicy
from ..decl import Decl
from ..flow import Flow
from ..hooks import Hooks, Where
from ..nodes import Checkpointer
from ..resource import NO_RESOURCES, ResourceKey
from ..state import State
from ..state.snapshot import ScopePath, ScopeRegistry
from ._checkpoint_ctx import CheckpointContext
from ._halt_observer import is_run_halted
from ._run import Run
from ._shortcut import ShortcutRun, is_draining, is_fast_forward, region_run, turn_abort


if TYPE_CHECKING:
    from ..context import Context
    from ..state.cas import CommitOutcome, FlowCommit


@dataclass(frozen=True)
class Position:
    """Where the walk is: computed as it descends, never given by the caller.

    ``path`` is the snapshot path of the flow being walked: node ids plus
    the iterate pass and map item coordinates that tell repeated executions
    apart (:mod:`llm_gent.flow.state.snapshot`); scopes opened under the
    flow register at ``path`` extended by their owner. ``chain_context`` is
    the hash the flow's chain-step node ids are computed against and
    ``ancestor_chain`` the ids of the nodes from the run's root down to the
    one whose descent entered the flow: the pair the checkpoint layer walks
    to address a point in the composition tree. ``activity_id`` is the
    invocation events raised here belong to: the step running this flow,
    the run at the top level.
    """

    path: ScopePath = ()
    chain_context: str = ""
    ancestor_chain: tuple[str, ...] = ()
    activity_id: str | None = None

    def descend(self, node_id: str, boundary: str, path: ScopePath) -> Position:
        """The position of the flow a node enters through ``boundary``, at ``path``.

        ``boundary`` names the node's slot the flow fills (``"call"``, an
        arm, ``"body"``, ``"map:<key>"``); the activity carries over.
        """
        return Position(
            path=path,
            chain_context=_descend_context(node_id, boundary),
            ancestor_chain=(*self.ancestor_chain, node_id),
            activity_id=self.activity_id,
        )


@dataclass(frozen=True)
class Scope:
    """The layered inputs in effect at one place in a run.

    ``resources`` are what ``ctx.resource`` reads (the cost tracker,
    ``ctx.cost``, among them). ``checkpointer`` is the innermost
    :meth:`Flow.with_checkpointer` enclosing the place: saves write commits
    only under one. ``hooks`` are ``None`` when none is in effect, so a run
    without hooks builds no events. ``shortcuts`` are the
    :meth:`Flow.with_shortcut` regions the place is in, outermost first.
    """

    state: State[Any]
    # Through a factory: Python 3.11's dataclasses reject a mappingproxy default.
    resources: Mapping[ResourceKey[Any], Any] = field(default_factory=lambda: NO_RESOURCES)
    checkpointer: Checkpointer | None = None
    policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    hooks: Hooks | None = None
    shortcuts: tuple[ShortcutRun, ...] = ()

    def child(
        self,
        decl: Decl,
        *,
        resources: Mapping[ResourceKey[Any], Any],
        where: Where,
        signals: Mapping[str, Any],
        lg: Logger,
    ) -> Scope:
        """The scope of a flow entered here that declares ``decl``.

        ``resources`` come derived (they register with the run on entry),
        ``where`` is the flow's place for its inherited hooks' children,
        ``signals`` the run's, ``lg`` the logger its own hooks report to.
        """
        return Scope(
            state=self.state,
            resources=resources,
            checkpointer=decl.checkpointer if decl.checkpointer is not None else self.checkpointer,
            policy=decl.policy if decl.policy is not None else self.policy,
            hooks=_derived_hooks(self.hooks, decl, where, lg),
            shortcuts=_accumulated(self.shortcuts, region_run(decl.shortcut, signals)),
        )


def _derived_hooks(parent: Hooks | None, decl: Decl, where: Where, lg: Logger) -> Hooks | None:
    """The inherited hooks' children at ``where``, then the flow's; ``None`` when none applies."""
    inherited = None if parent is None else parent.child(where)
    if not decl.hooks:
        return None if decl.hooks_replace else inherited
    if inherited is None or decl.hooks_replace:
        return Hooks(lg, decl.hooks)
    return inherited.extend(decl.hooks)


def _accumulated(
    regions: tuple[ShortcutRun, ...], region: ShortcutRun | None
) -> tuple[ShortcutRun, ...]:
    """``regions`` with the flow's region added innermost, when it declares one."""
    return regions if region is None else (*regions, region)


@dataclass(frozen=True)
class _Frame:
    """What the walk carries at one place in a run: the run, the scope, the position.

    ``run`` holds what is fixed for the whole run (its halt, signals,
    history writer and live scopes; :attr:`runtime`, :attr:`halt`,
    :attr:`checkpoint_ctx` and :attr:`scopes` read it), as does ``extra``,
    the caller's opaque dict. ``lg`` is the run's logger.
    """

    run: Run
    lg: Logger
    scope: Scope
    position: Position = field(default_factory=Position)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def runtime(self) -> Flow:
        """The run's top-level Flow."""
        return self.run.flow

    @property
    def halt(self) -> asyncio.Event | None:
        """The run's halt event; ``None`` when the run has none."""
        return self.run.halt_event

    @property
    def checkpoint_ctx(self) -> CheckpointContext | None:
        """The run's history writer; ``None`` without a checkpoint store."""
        return self.run.checkpoint_ctx

    @property
    def scopes(self) -> ScopeRegistry:
        """The run's live-scope registry."""
        return self.run.scopes

    def owner_path(self, node_id: str) -> ScopePath:
        """Snapshot path of the step ``node_id`` of the flow being walked."""
        return (*self.position.path, "n", node_id)

    def descend(
        self,
        node_id: str,
        boundary: str,
        path: ScopePath,
        state: State[Any],
        *,
        shortcuts: tuple[ShortcutRun, ...] | None = None,
    ) -> _Frame:
        """What node ``node_id`` hands the flow it enters through ``boundary``.

        The position moves to ``path``; the scope takes the node's
        ``state`` and, when given, its ``shortcuts`` (a map item leaves its
        drain regions). The flow entered applies its declarations on top
        (:meth:`Scope.child`).
        """
        scope = dataclasses.replace(
            self.scope,
            state=state,
            shortcuts=self.scope.shortcuts if shortcuts is None else shortcuts,
        )
        return dataclasses.replace(
            self, scope=scope, position=self.position.descend(node_id, boundary, path)
        )

    def at_activity(self, activity_id: str) -> _Frame:
        """This place, with events raised here belonging to ``activity_id``."""
        return dataclasses.replace(
            self, position=dataclasses.replace(self.position, activity_id=activity_id)
        )

    # What a step reaches of its run, through ``ctx`` (the structure never imports the engine).

    def fast_forward(self) -> bool:
        """This place is in a region whose signal is set."""
        return is_fast_forward(self)

    def draining(self) -> bool:
        """This place is in a started map item of a drain region whose signal is set."""
        return is_draining(self)

    def run_halted(self) -> bool:
        """The run's halt is set."""
        return is_run_halted(self)

    def turn_abort(self) -> contextlib.AbstractAsyncContextManager[asyncio.Event | None]:
        """What a Loop turn here aborts on, for the duration of the turn."""
        return turn_abort(self)

    def llm_call_hook(
        self, ctx: Context[Any], conversation: Any, verb: str, role: str | None
    ) -> Callable[[Any], Awaitable[None]] | None:
        """An ``after_llm_call`` caller for one Loop call here, or ``None`` when no hook wants it."""
        from ._hook_points import llm_call_hook

        return llm_call_hook(ctx, conversation, verb, role)

    async def save_commit(
        self, node_id: str, state: State[Any], outcome: CommitOutcome
    ) -> FlowCommit | None:
        """Save the whole run as a commit from step ``node_id``; ``None`` when nothing saves here."""
        from ._executor import _save_scope_commit

        return await _save_scope_commit(self, 0, node_id, state, outcome)
