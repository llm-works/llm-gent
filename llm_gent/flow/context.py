"""Context — runtime environment injected into every verb dispatch.

A :class:`Context` is built by the :class:`Flow` at dispatch time and passed
as the first argument to every verb. It exposes:

- ``saia`` — the role-bound saia instance for this dispatch
- ``role`` — the :class:`Role` under which this verb is running
- ``state`` — the flow's shared state object (user-owned; opaque to the flow)
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
    """The role-bound saia instance for this dispatch."""

    role: Role
    """The role the current verb declared."""

    state: Any
    """The flow's shared state object (user-owned; the flow does not inspect it)."""

    flow: Any
    """Back-reference to the :class:`Flow` that built this context.

    Composition helpers (:class:`Panel`, etc.) use this to dispatch sibling
    verbs with their own role-bound saia. Typed as ``Any`` to avoid a circular
    import — ``.dispatch(name, *args, **kwargs)`` is the only method used.
    """
