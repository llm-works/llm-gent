# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resources — run-scoped objects a flow carries, checkpointed with the run.

A resource is an object with behaviour (a counter, a stats collector, a
rate limiter) whose accounting must survive pause, resume and shortcut.
It implements the :class:`Resource` protocol: ``snapshot()`` returns its
accounting as plain data, ``restore(data)`` sets it back from a
snapshot. A resource that gives each run (a map item, an iterate pass) a
child of its own also has ``child(...)``, returning a resource that
reports to it. Subclassing :class:`Resource` makes the contract explicit
and lets type checkers check the methods against it; a class that has
the methods without subclassing works too::

    class Stats(Resource):
        def snapshot(self) -> dict[str, Any]: ...
        def restore(self, data: dict[str, Any]) -> None: ...
        def child(self, per_item: bool = True) -> Stats: ...  # optional

The app declares a :class:`ResourceKey` once — the typed handle and the
name the resource is stored under — and attaches resources with
:meth:`Flow.with_resource`::

    STATS = ResourceKey[Stats]("stats")

    flow.with_resource(STATS, Stats())                  # this flow and below
    body.with_resource(STATS, per_item=True)            # each run: a child of the enclosing one
    body.with_resource(STATS, Stats(), per_item=True)   # each run: a child of its own

Verbs reach it with ``ctx.resource(STATS)``, typed ``Stats``.

A fluent name of the app's own, ``flow.with_stats(...)``, is
:func:`resource_method`: on a :class:`Flow` subclass (type-checked), or
added to every flow a factory builds with
:meth:`Factory.with_resource_method`.
"""

from __future__ import annotations

from abc import abstractmethod
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar, overload, runtime_checkable


F = TypeVar("F")
F_co = TypeVar("F_co", covariant=True)


@runtime_checkable
class Resource(Protocol):
    """A run-scoped object whose accounting is checkpointed with the run.

    ``snapshot()`` returns the accounting as JSON-serializable data;
    :meth:`restore` is its inverse, called with what ``snapshot()``
    saved before the run's first step on resume. What resume means for
    the resource is its own decision: continue, rebase, ignore. Handles
    and wiring (loggers, connections, callbacks) are not accounting and
    stay out of the snapshot.

    A class implements it by subclassing it (its methods are then checked
    against it, and one left out keeps the class from being
    instantiated) or by having the methods.

    Optional: ``child(...)``. :meth:`Flow.with_resource` with child
    arguments calls it once per run of the flow (a map item, an iterate
    pass, a ``.call``) with those arguments as keywords; it returns a
    resource — what that run's verbs read, typically reporting to the
    resource it came from. Its parameters are the resource's own choice.
    A resource without it cannot take child arguments: the flow raises
    :class:`TypeError` naming it. (It is not declared here: a protocol's
    members are all required.)
    """

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """This resource's accounting as plain data."""

    @abstractmethod
    def restore(self, data: dict[str, Any]) -> None:
        """Set the accounting back from a :meth:`snapshot` taken earlier."""


R = TypeVar("R", bound=Resource)
"""A resource type: what a :class:`ResourceKey` is for."""

R_contra = TypeVar("R_contra", bound=Resource, contravariant=True)


def has_child(resource: object) -> bool:
    """True when ``resource`` has the optional ``child()`` (see :class:`Resource`)."""
    return callable(getattr(resource, "child", None))


class ResourceKey(Generic[R]):
    """Typed handle for a resource: declared once, used to attach and to read it.

    The type a key is for implements :class:`Resource` —
    ``ResourceKey[Stats]`` is a type error when ``Stats`` does not.
    ``name`` is what the resource is stored under in a checkpoint, so it
    must stay stable across releases of the app that declares it. Keys
    compare by identity: two keys with the same name in one flow's
    composition tree are an error at run start.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        """Declare a resource key.

        Raises:
            TypeError: ``name`` is not a string.
            ValueError: ``name`` is empty.
        """
        if not isinstance(name, str):
            raise TypeError(f"a resource key's name must be a str; got {type(name).__name__}")
        if not name:
            raise ValueError("a resource key's name must not be empty")
        self.name = name

    def __repr__(self) -> str:
        return f"ResourceKey({self.name!r})"


NO_RESOURCES: MappingProxyType[ResourceKey[Any], Any] = MappingProxyType({})
"""The resources of a run with none in scope."""


def check_resource(key: object, value: object, child_args: dict[str, Any]) -> None:
    """Raise unless :meth:`Flow.with_resource` accepts ``key``, ``value`` and ``child_args``.

    Raises:
        TypeError: ``key`` is not a :class:`ResourceKey`; ``value`` does
            not implement :class:`Resource`; child arguments with a
            ``value`` that has no ``child()``.
        ValueError: neither a value nor child arguments.
    """
    if not isinstance(key, ResourceKey):
        raise TypeError(f"with_resource takes a ResourceKey; got {type(key).__name__}")
    if value is None:
        if not child_args:
            raise ValueError(
                f"with_resource({key.name!r}) needs a resource, child arguments, or both"
            )
        return
    if not isinstance(value, Resource):
        raise TypeError(
            f"resource {key.name!r} must implement snapshot() and restore(data); "
            f"got {type(value).__name__}"
        )
    if child_args and not has_child(value):
        raise TypeError(
            f"resource {key.name!r} ({type(value).__name__}) has no child(); "
            f"it cannot take child arguments"
        )


class _BoundResourceMethod(Protocol[R_contra, F_co]):
    """A :func:`resource_method` read from a flow: ``with_resource`` with the key filled in."""

    def __call__(self, value: R_contra | None = None, /, **child_args: Any) -> F_co: ...


class _ResourceMethod(Generic[R]):
    """Descriptor :func:`resource_method` returns; see there."""

    def __init__(self, key: ResourceKey[R]) -> None:
        self.key = key

    @overload
    def __get__(self, obj: None, owner: type[Any]) -> _ResourceMethod[R]: ...

    @overload
    def __get__(self, obj: F, owner: type[Any]) -> _BoundResourceMethod[R, F]: ...

    def __get__(self, obj: Any, owner: type[Any]) -> Any:
        if obj is None:
            return self
        key = self.key

        def bound(value: R | None = None, /, **child_args: Any) -> Any:
            return obj.with_resource(key, value, **child_args)

        bound.__name__ = f"with_{key.name}"
        bound.__doc__ = f"``with_resource({key.name!r}, ...)``: see :meth:`Flow.with_resource`."
        return bound


def resource_method(key: ResourceKey[R]) -> _ResourceMethod[R]:
    """A fluent method of the app's own for ``key``: ``with_resource(key, ...)`` by another name.

    Assigned on a :class:`Flow` subclass, it is typed — the value is
    checked against ``key``'s type and the chain keeps the subclass::

        class MyFlow(Flow):
            with_stats = resource_method(STATS)

        Factory(lg, flow_class=MyFlow).create().with_stats(Stats()).then(step)

    :meth:`Factory.with_resource_method` adds one to every flow a
    factory builds, without a subclass of the app's own (untyped).
    """
    return _ResourceMethod(key)
