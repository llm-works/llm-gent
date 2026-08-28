# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Verb — a role-bound async callable dispatched by a :class:`Flow`.

A verb is any async callable whose first argument is a :class:`Context` and
which carries a ``role`` attribute of type :class:`Role`. The framework does
not require subclassing — a plain function decorated with :func:`verb` is
the canonical shape:

    @verb(role=BIO_JUDGE)
    async def verify_bio(ctx, candidate):
        return await ctx.saia.verify(
            claim=candidate.bio, criterion="indie voice",
        )

Class-based verbs also work: any object with a ``role`` attribute and an
async ``__call__`` (or ``run``) method satisfies the same shape. The
framework dispatches whatever the flow's registry holds under a name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .role import Role


VerbCallable = Callable[..., Awaitable[Any]]
"""An async callable that takes ``(ctx, *args, **kwargs)`` and returns any result."""


def verb(role: Role) -> Callable[[VerbCallable], VerbCallable]:
    """Mark an async function as a role-bound verb.

    Attaches the given :class:`Role` to the function so the flow can dispatch
    it under a role-bound saia. The wrapped function is otherwise unchanged
    — it remains an ordinary async callable.

    Example::

        BIO_JUDGE = Role(name="bio_judge", backend="openai", model="gpt-4o-mini")

        @verb(role=BIO_JUDGE)
        async def verify_bio(ctx, candidate):
            return await ctx.saia.verify(
                claim=candidate.bio, criterion="indie voice",
            )
    """

    def _decorator(func: VerbCallable) -> VerbCallable:
        func.role = role  # type: ignore[attr-defined]
        return func

    return _decorator
