# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Context — runtime environment injected into every verb dispatch.

A :class:`Context` is built by the :class:`Flow` at dispatch time and passed
as the first argument to every verb. It exposes:

- ``saia`` — the role-bound saia instance for this dispatch, resolved lazily
  on first access
- ``role`` — the :class:`Role` under which this verb is running
- ``state`` — the enclosing scope's :class:`State` wrapper (user-owned payload
  reached via ``ctx.state.data`` or the shorter alias ``ctx.data``; run-wide
  payload via ``ctx.state.root().data``)
- ``flow`` — back-reference to the dispatching flow (enables a verb to
  dispatch sibling verbs)
- ``lg`` — the dispatching flow's :class:`~appinfra.log.Logger`, so verbs
  written as module-level ``async def`` (rather than :class:`Verb` classes
  that capture ``lg`` at ``__init__``) can trace without threading it
  through state
- ``extra`` — caller-supplied per-invocation opaque dict, escape hatch
  for handles the framework does not type (tenant IDs, correlation IDs,
  per-run callbacks); supplied at :meth:`FlowRunner.start` via ``extra=`` and
  never checkpointed
- ``resource(key)`` — a resource the run runs with
  (:meth:`Flow.with_resource`), typed by its :class:`ResourceKey`

Verbs read from this and (typically) mutate ``state.data`` in place.

Generic in the payload type :data:`T`. Annotating a verb parameter as
``ctx: Context[MyState]`` narrows ``ctx.state.data`` to ``MyState`` for
type-checked access; unparameterized ``Context`` remains valid and treats the
payload as :data:`Any`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, overload

from appinfra.log import Logger

from ..core.cost import Tracker as CostTracker
from ..core.env import Environment
from ..core.role import Role
from ..core.traits import Registry as TraitRegistry
from .checkpoint import checkpoint_tag
from .resource import COST, NO_RESOURCES, R, ResourceKey
from .state import State


T = TypeVar("T")
"""Payload type — threads to ``ctx.state.data``. See module docstring."""

S = TypeVar("S")
"""SAIA type — narrows :meth:`Context.saia_as` return."""

D = TypeVar("D")
"""Default type of :meth:`Context.resource`."""

H = TypeVar("H")
"""Hook type of :meth:`Context.hook`."""

_NO_DEFAULT: Any = object()


@dataclass(frozen=True)
class Context(Generic[T]):
    """Runtime environment injected into every verb.

    Constructed by the flow at dispatch time. Verbs receive it as their first
    positional argument. Generic in the payload type; ``Context[MyState]``
    narrows ``ctx.state.data`` for type-checked access.
    """

    role: Role | None
    """The role the current verb declared, or ``None`` for composition hooks and pure-Python verbs.

    ``None`` appears on composition hook contexts (``rescue`` / ``after``
    attached to a subflow node) and on pure-Python verb contexts created
    from ``@verb`` without ``role=``.
    """

    state: State[T]
    """The enclosing scope's :class:`State` wrapper.

    ``ctx.state.data`` is the scope's payload (user-owned; the flow does not
    inspect it), typed as :data:`T`. Shared with the parent by reference by
    default; the ``state=`` / ``merge=`` kwargs on :meth:`Flow.call`,
    :meth:`Flow.iterate`, and :meth:`Flow.map` project an isolated child
    payload for the subflow they contain. Verbs reach run-wide state via
    ``ctx.state.root().data`` (typed :data:`Any` because a child's payload
    type has no static relationship to its ancestors').
    """

    flow: Any
    """Back-reference to the :class:`Flow` that built this context.

    A verb uses this to dispatch sibling verbs. Typed as ``Any`` to avoid a
    circular import — ``.dispatch(name, *args, **kwargs)`` is the only
    method used.
    """

    env: Environment = field(default_factory=Environment)
    """What the run may use: the SAIA per role (:attr:`saia`) and the traits (:attr:`traits`).

    The run's :class:`~llm_gent.core.env.Environment`, given to the runner
    that started it; every flow of the run, nested ones included, sees the
    same one. Pass ``env=ctx.env`` to :meth:`Flow.dispatch` to hand it to a
    sibling.
    """

    halt: asyncio.Event | None = None
    """The run's halt — :meth:`FlowRunner.start` ``halt=`` — or ``None``.

    Verbs that expose their own inner loop (SAIA turn-by-turn, long-running
    external calls) can observe ``ctx.halt`` to short-circuit gracefully:
    a step that stops before finishing raises :class:`Interrupted` and
    runs again on resume. :meth:`Flow.map` and :meth:`Flow.iterate`
    observe it at their natural boundaries automatically. Setting it from
    a step pauses the whole run (awaiting the run gives ``HALTED``).

    A cut (:meth:`Flow.with_shortcut`) does not set it; see
    :attr:`fast_forward`.
    """

    extra: dict[str, Any] = field(default_factory=dict)
    """Caller-supplied per-invocation opaque data.

    Escape hatch for handles the framework does not type (tenant IDs,
    correlation IDs, request-scoped audit hooks, per-run callbacks).
    Supplied at :meth:`FlowRunner.start` via the ``extra=`` kwarg; propagates
    unchanged to every dispatch inside the run — subflows, iterate
    bodies and map items all see the same dict.

    Framework does not inspect the contents, does not type-check the
    values, and never persists them: ``extra`` never enters a
    checkpoint Blob, Tree, or node content hash. On resume the caller
    re-supplies at :meth:`FlowRunner.start`; identity across resume is not
    preserved. Default is a fresh empty dict.
    """

    resources: Mapping[ResourceKey[Any], Any] = field(default_factory=lambda: NO_RESOURCES)
    """The resources the run runs with, by key; read one with :meth:`resource`."""

    _frame: Any = None
    """Private: the executor's :class:`_Frame` for this dispatch.

    Threaded from :func:`_build_ctx` so verbs can invoke framework
    save primitives (:meth:`checkpoint`). ``None`` on ctx surfaces
    that do not originate from a live run (e.g. :meth:`Flow.dispatch`
    used standalone, until predicates before an iterate started).
    Not part of the verb-facing API — the leading underscore marks it
    as an executor-only slot.
    """

    _node_id: str | None = None
    """Private: the node id under which this ctx was built.

    Threaded from :func:`_build_ctx` so :meth:`checkpoint` can address
    the currently-executing composition position. ``None`` when the
    ctx was built outside a chain-walk step (standalone
    :meth:`Flow.dispatch`, until predicates, on_error hooks).
    """

    @property
    def data(self) -> T:
        """Shortcut for ``ctx.state.data`` typed as :data:`T`.

        Every verb that reads or mutates the scope's payload does so
        through ``ctx.state.data``; this alias returns the same object at
        the shorter spelling. Reads (``ctx.data.field``) and mutations
        (``ctx.data.field = ...``) both work — the alias returns the
        payload, not a copy.

        Reaches the local scope's payload, not the root. Verbs that need
        run-wide state stay on ``ctx.state.root().data`` — the alias does
        not shortcut that traversal.
        """
        return self.state.data

    @property
    def traits(self) -> TraitRegistry | None:
        """The run's trait registry, or ``None`` when its environment has none.

        Verbs reach mounted platform capabilities (memory, storage, tools,
        custom traits) via ``ctx.traits.get(SomeTrait)`` or
        ``ctx.traits.require(SomeTrait)``; a verb that needs a trait handles
        its absence, or the run is started with an environment that has
        one. The same class is exported as ``Registry`` from
        :mod:`llm_gent.core.traits`.
        """
        return self.env.traits

    @property
    def saia(self) -> Any:
        """The role-bound saia instance for this dispatch.

        Built on first access by the run's environment
        (:meth:`Environment.saia <llm_gent.core.env.Environment.saia>`),
        which keeps one per role: repeated reads, from this run or any
        other sharing the environment, get the same instance.

        Returns ``None`` when :attr:`role` is ``None`` — subflow-node ctx,
        hook ctx, and pure-Python verbs (``@verb`` with no ``role=``)
        have no role to bind against, so there is no saia to build.

        Raises :class:`RuntimeError` if the environment has no SAIAFactory
        and this ctx has a role. The error surfaces at access, not at
        construction, so verbs that never touch ``ctx.saia`` (e.g. those
        routing LLM calls through their own configuration) run without one.
        """
        if self.role is None:
            return None
        return self.env.saia(self.role)

    def saia_as(self, cls: type[S]) -> S | None:
        """Return :attr:`saia` typed as ``cls`` — cast helper for verb authors.

        :attr:`saia` is typed :data:`Any` because the framework has no
        static knowledge of the concrete class an application's
        :class:`SAIAFactory` returns. Verbs that know the type write::

            saia = ctx.saia_as(MySAIA)
            answer = await saia.complete_structured(prompt, schema)

        Runtime returns exactly what :attr:`saia` returns; ``cls`` is
        used only for type inference, not for a runtime instance check.
        The framework's ``S`` TypeVar carries the annotation through so
        IDEs and mypy narrow the return type at the call site.

        Best suited to concrete SAIA classes. mypy's ``type-abstract``
        check rejects passing a :class:`typing.Protocol` here — for a
        protocol-typed handle, use the annotation form instead::

            saia: MyProto = ctx.saia

        When several verbs in a file need the same Protocol-typed
        handle, keep the annotation in one file-scoped helper rather
        than repeating it per verb::

            def _saia(ctx: Context[MyState]) -> MyProto:
                saia: MyProto = ctx.saia
                return saia

            @verb(role=R)
            async def do_thing(ctx: Context[MyState], prev: X) -> Y:
                return await _saia(ctx).complete_structured(...)

        The helper assumes the flow was configured with a SAIAFactory;
        role-bound verbs (``@verb(role=...)``) require one.
        """
        return self.saia  # type: ignore[no-any-return]

    @overload
    def resource(self, key: ResourceKey[R]) -> R: ...

    @overload
    def resource(self, key: ResourceKey[R], default: D) -> R | D: ...

    def resource(self, key: ResourceKey[Any], default: Any = _NO_DEFAULT) -> Any:
        """The ``key`` resource the run runs with (:meth:`Flow.with_resource`), typed by ``key``.

        Its own per-run child when the run's flow asks for one, else the
        resource declared on the run's flow or the nearest flow enclosing
        it. ``default`` when none is in scope.

        Raises:
            KeyError: No ``key`` resource is in scope and no ``default``
                was given.
        """
        try:
            return self.resources[key]
        except KeyError:
            if default is _NO_DEFAULT:
                raise KeyError(
                    f"no {key.name!r} resource in scope: declare one with "
                    f"with_resource(key, resource) on this flow or an enclosing one"
                ) from None
            return default

    @property
    def activity_id(self) -> str | None:
        """The invocation this step runs as, as hook events name it; ``None`` without hooks.

        Set when hooks are in effect (:meth:`Flow.with_hooks`): the step's
        own ``activity_id``, which the events raised inside it name as their
        parent.
        """
        return None if self._frame is None else self._frame.position.activity_id

    @property
    def run_id(self) -> str | None:
        """The id of the :meth:`FlowRunner.start` call this step runs in; ``None`` without hooks."""
        if self._frame is None:
            return None
        info = self._frame.run.info
        return None if info is None else info.run_id

    def hook(self, cls: type[H]) -> H | None:
        """The innermost hook of type ``cls`` in effect at this step, or ``None``.

        Hooks are attached with :meth:`Flow.with_hooks`; a verb uses this to
        call one directly, e.g. to record something of its own. ``None``
        outside a run.
        """
        frame = self._frame
        if frame is None or frame.scope.hooks is None:
            return None
        found: H | None = frame.scope.hooks.find(cls)
        return found

    @property
    def fast_forward(self) -> bool:
        """True when this step runs in a :meth:`Flow.with_shortcut` region whose signal is set.

        The region is being cut short: a step that can end early returns
        what it has now — it completed, and nothing runs it again. (A
        step stopping for the run's :attr:`halt` raises
        :class:`Interrupted` instead.) ``False`` outside a run.
        """
        return self._frame is not None and self._frame.fast_forward()

    @property
    def draining(self) -> bool:
        """True when this step runs in a started map item of a draining region whose signal is set.

        The region (``with_shortcut(signal, drain=True)``) starts nothing
        new, but this item runs to its end: a step that can wrap up — a
        long LLM turn, a search — finishes what it has now and lets the
        item's remaining steps run. ``False`` outside a run.
        """
        return self._frame is not None and self._frame.draining()

    @property
    def cost(self) -> CostTracker | None:
        """The run's cost tracker — ``resource(COST, None)`` — or ``None``.

        Set with :meth:`Flow.with_cost_tracker`; a run of a flow with
        :meth:`Flow.with_budget` gets a child of it with that budget. Verbs
        record LLM and operation costs via ``ctx.cost.track(...)``;
        ``ctx.cost.spent`` is the cost so far and ``ctx.cost.budget`` the
        limit it is checked against. The tracker enforces its budget and,
        when configured with a halt event, trips it on the first cross into
        ``exceeded``; ancestors in the tracker chain do the same on their own
        budgets. Subflows inherit the outer runtime's tracker.
        """
        return self.resource(COST, None)

    @property
    def lg(self) -> Logger:
        """The dispatching flow's :class:`~appinfra.log.Logger`.

        Delegates to the flow's construction-time ``lg``. Module-level
        ``@verb`` functions (which can't capture ``lg`` at ``__init__``
        the way :class:`Verb` classes can) reach the ambient logger via
        ``ctx.lg`` instead of threading it through ``state``.
        """
        lg: Logger = self.flow._lg
        return lg

    async def checkpoint(self, name: str | None = None) -> str | None:
        """Explicit save trigger — writes a snapshot of the whole run as a commit; returns its hash.

        Fires regardless of the flow's :class:`CheckpointPolicy`; the
        policy governs implicit framework-driven saves only. The commit
        holds every live scope and every running structure's cursor, like
        any checkpoint: the chain running this step is at this step, so
        resuming from it runs this step again (with the input it had),
        and every enclosing iterate continues in its current pass. The
        commit's metadata records the current node, iteration ``0`` and
        ``outcome="ok"``.

        The commit goes to the run's history (:meth:`~llm_gent.flow.FlowRunner.start`
        ``history=``), wherever in the flow tree this step runs. Returns the commit's hash:
        ``start=Resume(<hash>)`` checks it out later, even after a later run
        leaves it off the history's line. With ``name``, the checkpoint is also
        tagged ``tags/<name>``: ``start=Resume(name)`` checks it out. Taking
        it again (at this step or elsewhere) moves the tag to the new
        commit. A named :meth:`~llm_gent.flow.Flow.with_checkpointer` this
        save belongs to moves its tag too.

        No-op returning ``None`` when:
        - No ``with_checkpointer()`` is on this step's flow or above it, or
          the run has no checkpoint store.
        - The ctx has no live executor frame (e.g. built by
          :meth:`Flow.dispatch` used standalone).

        Raises:
            ValueError: ``name`` cannot name a checkpoint (empty, one of
                ``"off"``, ``"latest"``, ``"complete"``, or a commit hash's
                form) — also without a checkpointer.
        """
        tag = None if name is None else checkpoint_tag(name)
        frame = self._frame
        node_id = self._node_id
        if frame is None or node_id is None:
            return None
        commit = await frame.save_commit(node_id, self.state, "ok")
        if commit is None:
            return None
        if tag is not None:
            assert frame.checkpoint_ctx is not None  # a commit was written to it
            await frame.checkpoint_ctx.put_tag(tag, commit.id)
        commit_id: str = commit.id
        return commit_id
