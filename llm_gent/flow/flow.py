"""Flow — verb registry with role-routed dispatch.

A :class:`Flow` holds a set of registered verbs, a :class:`SAIAFactory` for
turning roles into saia instances, and a user-owned shared ``state`` object.
When a verb is dispatched by name the flow:

1. Looks up the verb in its registry.
2. Resolves the verb's role to a saia instance (cached per role).
3. Builds a :class:`Context` exposing ``saia`` / ``role`` / ``state``.
4. Invokes the verb, awaiting its result.

The flow is unopinionated about how verbs are authored — plain functions
decorated with :func:`verb`, class instances with a ``role`` attribute and
async ``__call__``, or any other callable that satisfies the shape.
"""

from __future__ import annotations

from typing import Any

from .context import Context
from .factory import SAIAFactory
from .role import Role


class Flow:
    """Verb registry + role-routed dispatch.

    Construction requires a :class:`SAIAFactory`; the shared ``state`` object
    is user-owned and opaque (dict, ``appinfra.FieldDict`` subclass, custom
    dataclass — anything).
    """

    def __init__(self, factory: SAIAFactory, state: Any = None) -> None:
        """Initialize a flow with a saia factory and optional shared state."""
        self._factory = factory
        self._state = state
        self._verbs: dict[str, Any] = {}
        self._saia_by_role: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # State access
    # -------------------------------------------------------------------------

    @property
    def state(self) -> Any:
        """The flow's shared state object (user-owned)."""
        return self._state

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, verb: Any, name: str | None = None) -> None:
        """Register a verb under a name (default: the verb's ``__name__``).

        The verb must carry a ``role`` attribute of type :class:`Role`.
        """
        if not hasattr(verb, "role"):
            raise TypeError(
                f"verb must carry a .role attribute; got {type(verb).__name__} without one"
            )
        if not isinstance(verb.role, Role):
            raise TypeError(f"verb.role must be a Role instance; got {type(verb.role).__name__}")
        resolved_name = name or getattr(verb, "__name__", None)
        if not resolved_name:
            raise TypeError("verb has no __name__ and no explicit name was provided")
        self._verbs[resolved_name] = verb

    def registered(self, name: str) -> bool:
        """Return True if a verb is registered under ``name``."""
        return name in self._verbs

    # -------------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------------

    async def dispatch(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a registered verb by name, awaiting its result.

        The verb receives a fresh :class:`Context` as its first argument,
        followed by ``*args`` / ``**kwargs`` from the caller.
        """
        if name not in self._verbs:
            raise KeyError(f"no verb registered under name {name!r}")
        verb = self._verbs[name]
        saia = self._saia_for(verb.role)
        ctx = Context(saia=saia, role=verb.role, state=self._state)
        return await verb(ctx, *args, **kwargs)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _saia_for(self, role: Role) -> Any:
        """Return a cached saia for ``role``, building it on first request."""
        cached = self._saia_by_role.get(role.name)
        if cached is not None:
            return cached
        built = self._factory.build(role)
        self._saia_by_role[role.name] = built
        return built
