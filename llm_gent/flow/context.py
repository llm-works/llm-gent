# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Context — runtime environment injected into every verb dispatch.

A :class:`Context` is built by the :class:`Flow` at dispatch time and passed
as the first argument to every verb. It exposes:

- ``saia`` — the role-bound saia instance for this dispatch, resolved lazily
  on first access
- ``role`` — the :class:`Role` under which this verb is running
- ``state`` — the enclosing scope's :class:`State` wrapper (user-owned payload
  reached via ``ctx.state.data``; run-wide payload via ``ctx.state.root().data``)
- ``flow`` — back-reference to the dispatching flow (enables inner verb calls
  from composition helpers like :class:`Panel`)

Verbs read from this and (typically) mutate ``state.data`` in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.traits import Registry as TraitRegistry
from .role import Role
from .state import State


@dataclass(frozen=True)
class Context:
    """Runtime environment injected into every verb.

    Constructed by the flow at dispatch time. Verbs receive it as their first
    positional argument.
    """

    role: Role | None
    """The role the current verb declared, or ``None`` for a subflow-node ctx.

    ``None`` only appears on the ambient ctx passed to ``rescue`` / ``after``
    hooks attached to a subflow node — those hooks fire at the composition
    layer, above any single role. Verb-level contexts always carry a role.
    """

    state: State
    """The enclosing scope's :class:`State` wrapper.

    ``ctx.state.data`` is the scope's payload (user-owned; the flow does not
    inspect it). Shared with the parent by reference by default; the
    ``state=`` / ``merge=`` kwargs on :meth:`Flow.call`, :meth:`Flow.loop`,
    and :meth:`Flow.map` project an isolated child payload for the subflow
    they contain. Verbs reach run-wide state via ``ctx.state.root().data``.
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

    @property
    def saia(self) -> Any:
        """The role-bound saia instance for this dispatch.

        Resolved lazily on first access via the dispatching flow's
        SAIAFactory. The flow caches per role, so repeated reads on the
        same or sibling contexts hit the same instance.

        Returns ``None`` when :attr:`role` is ``None`` — subflow-node ctx
        and hook ctx have no single role, so no saia to bind.

        Raises :class:`RuntimeError` if the flow was constructed with no
        SAIAFactory and this ctx has a role. The error surfaces at access,
        not at construction, so verbs that never touch ``ctx.saia`` (e.g.
        those routing LLM calls through their own configuration) can run
        under a factoryless flow.
        """
        if self.role is None:
            return None
        return self.flow._saia_for(self.role)
