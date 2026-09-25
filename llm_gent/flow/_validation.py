# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Builder-side validators for Flow composition.

Called from :meth:`Flow.call` / :meth:`Flow.iterate` / :meth:`Flow.map`
/ :meth:`Flow.branch` at build time so bad compositions surface where
the graph is authored, not at :meth:`Flow.run` time. All helpers are
module-private; the :class:`Flow` isinstance checks resolve through a
localized late import (``from .flow import Flow``) to break the
circular dependency between this module and the class it validates
against.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from .nodes import StateMerge, StateProject
from .role import Role


if TYPE_CHECKING:
    from .flow import Flow


def _validate_target(target: Any) -> None:
    """Reject anything that isn't a verb (callable with .role) or a Flow.

    ``.role`` may be ``None`` — the pure-Python-verb form produced by
    ``@verb`` without a role. The framework still dispatches such verbs;
    they just cannot read ``ctx.saia`` (which stays ``None``).
    """
    from .flow import Flow

    if isinstance(target, Flow):
        return
    if not callable(target):
        raise TypeError(
            f"target must be a verb (callable with .role) or a Flow; got {type(target).__name__}"
        )
    if not hasattr(target, "role"):
        raise TypeError(f"verb target must carry a .role attribute; got {type(target).__name__}")
    if target.role is not None and not isinstance(target.role, Role):
        raise TypeError(
            f"verb target .role must be a Role instance or None; got {type(target.role).__name__}"
        )
    _reject_reserved_kwarg(target, "state")
    _reject_reserved_kwarg(target, "runtime")
    _reject_reserved_kwarg(target, "resume")


def _reject_reserved_kwarg(verb: Any, name: str) -> None:
    """Reject a verb whose signature declares a :meth:`Flow.run`-bound kwarg.

    ``Flow.run(state=...)`` binds ``state`` to seed the top-level payload
    and strips it before forwarding kwargs to the first node. A verb whose
    signature declares ``state`` as a keyword-visible parameter would never
    see a value passed via ``.run()`` — silent misbehavior. Rejected at
    build time so the collision surfaces where the graph is authored.

    The first positional is conventionally ``ctx`` and is skipped: the
    framework always binds it, so its name doesn't collide with any run
    kwarg. Ignores verbs whose signature cannot be introspected (C
    callables, some partials) — those cannot statically collide with a
    bound kwarg either. ``**kwargs``-only verbs are allowed: :meth:`run`
    already strips the reserved name before forwarding, so the verb's
    ``**kwargs`` never sees it.
    """
    try:
        sig = inspect.signature(verb)
    except (TypeError, ValueError):
        return
    for param in list(sig.parameters.values())[1:]:
        if param.name != name:
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            verb_name = getattr(verb, "__name__", type(verb).__name__)
            raise TypeError(
                f"verb {verb_name!r} declares reserved parameter {name!r} — "
                f"Flow.run({name}=...) binds this name and it would never "
                f"reach the verb; rename the parameter"
            )
        return


def _require_state_for_merge(
    state: StateProject | None, merge: StateMerge | None, method: str
) -> None:
    """Enforce that ``merge=`` is only legal with ``state=`` (nothing to merge otherwise)."""
    if merge is not None and state is None:
        raise ValueError(
            f"{method}(merge=...) requires state= "
            "(nothing to merge back without an isolated child state)"
        )


def _materialize(buildable: Any, lg: Logger, name: str) -> Flow:
    """Turn a :data:`Buildable` (Flow, verb, or ``lambda f: ...`` callback) into a Flow.

    A ``Flow`` is returned as-is. A callable carrying a :class:`Role` on
    ``.role`` (a verb — whether a module-level ``@verb`` function or a
    bound instance method) is wrapped as a single-node flow calling that
    verb. Any other callable is invoked against a fresh Flow it may mutate
    (the return value, if any, is ignored). Anything else is a
    :class:`TypeError` — bad Buildables fail eagerly at build time, not at
    :meth:`Flow.run` time.
    """
    from .flow import Flow

    if isinstance(buildable, Flow):
        return buildable
    if not callable(buildable):
        raise TypeError(
            f"expected a Flow or a lambda f: f.call(...) callback for {name!r}; "
            f"got {type(buildable).__name__}"
        )
    if hasattr(buildable, "role"):
        # A verb — role attribute may be a Role (role-bound) or None
        # (pure-Python verb). Either way, wrap as a single-node subflow.
        # Defense-in-depth: validate role here (also checked in fresh.call).
        role = buildable.role
        if role is not None and not isinstance(role, Role):
            raise TypeError(
                f"verb target .role must be a Role instance or None; got {type(role).__name__}"
            )
        fresh = Flow(lg=lg, name=name)
        fresh.call(buildable)
        return fresh
    fresh = Flow(lg=lg, name=name)
    buildable(fresh)
    return fresh
