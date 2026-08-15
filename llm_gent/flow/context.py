"""Context — runtime environment injected into every verb dispatch.

A :class:`Context` is built by the :class:`Flow` at dispatch time and passed
as the first argument to every verb. It exposes:

- ``saia`` — the role-bound saia instance for this dispatch
- ``role`` — the :class:`Role` under which this verb is running
- ``state`` — the enclosing subflow's scoped state (user-owned; opaque to the flow)
- ``global_state`` — the run-wide state provided at the outermost :meth:`Flow.run`
- ``flow`` — back-reference to the dispatching flow (enables inner verb calls
  from composition helpers like :class:`Panel`)

Verbs read from this and (typically) mutate ``state`` in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .role import Role


@dataclass(frozen=True)
class Context:
    """Runtime environment injected into every verb.

    Constructed by the flow at dispatch time. Verbs receive it as their first
    positional argument.
    """

    saia: Any
    """The role-bound saia instance for this dispatch.

    Optional in one narrow case: when a fluent-builder node is itself a
    subflow, the ``rescue`` / ``after`` hooks attached to that node run with
    ``saia=None`` because the outer node has no single role. Verb-level
    contexts always carry a real saia.
    """

    role: Role | None
    """The role the current verb declared, or ``None`` for a subflow-node ctx.

    ``None`` only appears on the ambient ctx passed to ``rescue`` / ``after``
    hooks attached to a subflow node — those hooks fire at the composition
    layer, above any single role. Verb-level contexts always carry a role.
    """

    state: Any
    """The enclosing subflow's scoped state (user-owned; the flow does not inspect it).

    Shared with the parent flow by reference by default; the ``state=`` /
    ``merge=`` kwargs on :meth:`Flow.call`, :meth:`Flow.loop`, and
    :meth:`Flow.map` opt into an isolated child state for the subflow they
    contain.
    """

    global_state: Any
    """The run-wide state supplied at the outermost :meth:`Flow.run` invocation.

    Every node in the composition tree — including subflows, loop iterations,
    and map items — sees the same container. Defaults to an empty ``dict``
    when the caller does not supply one, so verbs guard on the KEY they need
    (``ctx.global_state.get("budget")``) rather than on the container.
    """

    flow: Any
    """Back-reference to the :class:`Flow` that built this context.

    Composition helpers (:class:`Panel`, etc.) use this to dispatch sibling
    verbs with their own role-bound saia. Typed as ``Any`` to avoid a circular
    import — ``.dispatch(name, *args, **kwargs)`` is the only method used.
    """
