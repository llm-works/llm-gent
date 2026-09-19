# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Verb — an async callable dispatched by a :class:`Flow`.

A verb is any async callable whose first argument is a :class:`Context`.
When it needs a role-bound saia, it also carries a ``role`` attribute of
type :class:`Role`; pure-Python verbs (utility steps that never touch
``ctx.saia``) can omit the role entirely and declare only ``ctx``:

    @verb(role=BIO_JUDGE)
    async def verify_bio(ctx, candidate):
        return await ctx.saia.verify(claim=candidate.bio, criterion="indie voice")

    @verb
    async def tick(ctx):
        ctx.state.data.count += 1

Class-based verbs also work: any object with a ``role`` attribute and an
async ``__call__`` (or ``run``) method satisfies the same shape. The
framework dispatches whatever the flow's registry holds under a name.

The decorator has three call shapes — bare (``@verb``), empty
(``@verb()``), and role-bound (``@verb(role=R)``) — so pure-Python and
role-bound verbs read the same at the call site.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, overload

from .role import Role


VerbCallable = Callable[..., Awaitable[Any]]
"""An async callable that takes ``(ctx, *args, **kwargs)`` and returns any result."""


@overload
def verb(fn: VerbCallable, /) -> VerbCallable: ...


@overload
def verb(*, role: Role | None = None) -> Callable[[VerbCallable], VerbCallable]: ...


def verb(
    fn: VerbCallable | None = None,
    /,
    *,
    role: Role | None = None,
) -> VerbCallable | Callable[[VerbCallable], VerbCallable]:
    """Mark an async function as a verb, optionally role-bound.

    Three call shapes:

    - ``@verb`` — bare decorator; the verb has no role. ``ctx.saia``
      returns ``None`` on such a dispatch, so the verb must not touch
      it. Use for pure-Python steps (counters, tick loops, plain data
      transforms) that share the flow's dispatch machinery without
      needing a backend.
    - ``@verb()`` — same as bare; the empty-call form exists for
      readers who reach for parentheses reflexively.
    - ``@verb(role=R)`` — role-bound verb; the framework builds
      ``ctx.saia`` for role ``R`` via the flow's :class:`SAIAFactory`.

    In every form the wrapped function is otherwise unchanged — it
    remains an ordinary async callable.

    Example::

        BIO_JUDGE = Role(name="bio_judge", backend="openai", model="gpt-4o-mini")

        @verb(role=BIO_JUDGE)
        async def verify_bio(ctx, candidate):
            return await ctx.saia.verify(
                claim=candidate.bio, criterion="indie voice",
            )

        @verb
        async def tick(ctx):
            ctx.state.data.count += 1
    """
    if fn is not None:
        # Bare @verb form: the single positional is the target function.
        fn.role = None  # type: ignore[attr-defined]
        return fn

    def _decorator(func: VerbCallable) -> VerbCallable:
        func.role = role  # type: ignore[attr-defined]
        return func

    return _decorator
