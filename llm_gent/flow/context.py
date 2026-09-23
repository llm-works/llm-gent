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
- ``flow`` — back-reference to the dispatching flow (enables inner verb calls
  from composition helpers like :class:`Panel`)
- ``lg`` — the dispatching flow's :class:`~appinfra.log.Logger`, so verbs
  written as module-level ``async def`` (rather than :class:`Verb` classes
  that capture ``lg`` at ``__init__``) can trace without threading it
  through state

Verbs read from this and (typically) mutate ``state.data`` in place.

Generic in the payload type :data:`T`. Annotating a verb parameter as
``ctx: Context[MyState]`` narrows ``ctx.state.data`` to ``MyState`` for
type-checked access; unparameterized ``Context`` remains valid and treats the
payload as :data:`Any`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from appinfra.log import Logger

from ..core.budget import Tracker
from ..core.traits import Registry as TraitRegistry
from .role import Role
from .state import State


T = TypeVar("T")
"""Payload type — threads to ``ctx.state.data``. See module docstring."""

S = TypeVar("S")
"""SAIA type — narrows :meth:`Context.saia_as` return."""


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

    Composition helpers (:class:`Panel`, etc.) use this to dispatch sibling
    verbs with their own role-bound saia. Typed as ``Any`` to avoid a circular
    import — ``.dispatch(name, *args, **kwargs)`` is the only method used.
    Also the resolver for :attr:`saia`.
    """

    traits: TraitRegistry | None = None
    """Trait registry the dispatching flow was constructed with, or ``None``.

    Verbs reach mounted platform capabilities (memory, storage, tools,
    custom traits) via ``ctx.traits.get(SomeTrait)`` or
    ``ctx.traits.require(SomeTrait)``. ``None`` when the flow was
    constructed without a registry — verbs that need a trait must handle
    absence, or the flow must be constructed with one. Imported as
    ``TraitRegistry`` to disambiguate from other registry types in
    consumer codebases; the same class is exported as ``Registry`` from
    :mod:`llm_gent.core.traits`.
    """

    halt: asyncio.Event | None = None
    """Ambient halt event attached via :meth:`Flow.with_halt`, or ``None``.

    Verbs that expose their own inner loop (SAIA turn-by-turn, long-running
    external calls) can observe ``ctx.halt`` to short-circuit gracefully.
    :meth:`Flow.map` and :meth:`Flow.iterate` observe this at their natural
    boundaries automatically; verbs are free to poll it when useful.
    Subflows inherit the outer runtime's halt unless they declare their own.
    """

    budget: Tracker | None = None
    """Cost tracker attached via :meth:`Flow.with_budget`, or ``None``.

    Verbs record LLM and operation costs via ``ctx.budget.track(...)`` (or
    against a child obtained via ``ctx.budget.child(budget=...)`` for
    per-scope caps). The tracker enforces its cap and, when configured
    with a halt event, trips it on the first cross into ``exceeded``;
    ancestors in the tracker chain do the same on their own caps.
    Subflows inherit the outer runtime's budget unless they declare their
    own via :meth:`Flow.with_budget`.
    """

    extra: dict[str, Any] = field(default_factory=dict)
    """Caller-supplied per-invocation opaque data.

    Escape hatch for handles the framework does not type (tenant IDs,
    correlation IDs, request-scoped audit hooks, per-run callbacks).
    Supplied at :meth:`Flow.run` via the ``extra=`` kwarg; propagates
    unchanged to every dispatch inside the run — subflows, iterate
    bodies, map items, and Panel arms all see the same dict.

    Framework does not inspect the contents, does not type-check the
    values, and never persists them: ``extra`` never enters a
    checkpoint Blob, Tree, or node content hash. On resume the caller
    re-supplies at :meth:`Flow.run`; identity across resume is not
    preserved. Default is a fresh empty dict.
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
    def saia(self) -> Any:
        """The role-bound saia instance for this dispatch.

        Resolved lazily on first access via the dispatching flow's
        SAIAFactory. The flow caches per role, so repeated reads on the
        same or sibling contexts hit the same instance.

        Returns ``None`` when :attr:`role` is ``None`` — subflow-node ctx,
        hook ctx, and pure-Python verbs (``@verb`` with no ``role=``)
        have no role to bind against, so there is no saia to build.

        Raises :class:`RuntimeError` if the flow was constructed with no
        SAIAFactory and this ctx has a role. The error surfaces at access,
        not at construction, so verbs that never touch ``ctx.saia`` (e.g.
        those routing LLM calls through their own configuration) can run
        under a factoryless flow.
        """
        if self.role is None:
            return None
        return self.flow._saia_for(self.role)

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
