"""Archetype decorators — semantic tags for verbs in the standard agent shape.

These decorators behave identically to :func:`verb` but also tag the wrapped
function with an ``.archetype`` attribute for introspection. The tags make
the semantic role of a verb explicit in code and allow tooling to filter or
route verbs by kind (e.g. "list all planners in this flow").

The standard agent shape:

- **planner** — task → plan
- **extractor** — evidence → structured data
- **grader** — result → verdict / score
- **synthesizer** — accumulated state → composed final result

Users are not required to use archetypes; plain :func:`verb` works for any
role. Archetypes exist to make the common shapes explicit.
"""

from __future__ import annotations

from collections.abc import Callable

from .role import Role
from .verb import VerbCallable, verb


def _archetype(role: Role, name: str) -> Callable[[VerbCallable], VerbCallable]:
    """Return a decorator that applies ``@verb(role)`` and tags with ``name``."""

    def decorator(func: VerbCallable) -> VerbCallable:
        wrapped = verb(role)(func)
        wrapped.archetype = name  # type: ignore[attr-defined]
        return wrapped

    return decorator


def planner(role: Role) -> Callable[[VerbCallable], VerbCallable]:
    """Mark a verb as a planner (task → plan)."""
    return _archetype(role, "planner")


def extractor(role: Role) -> Callable[[VerbCallable], VerbCallable]:
    """Mark a verb as an extractor (evidence → structured data)."""
    return _archetype(role, "extractor")


def grader(role: Role) -> Callable[[VerbCallable], VerbCallable]:
    """Mark a verb as a grader (result → verdict / score)."""
    return _archetype(role, "grader")


def synthesizer(role: Role) -> Callable[[VerbCallable], VerbCallable]:
    """Mark a verb as a synthesizer (accumulated state → composed final result)."""
    return _archetype(role, "synthesizer")
