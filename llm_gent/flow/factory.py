# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""App-scoped :class:`Factory` for flows.

:class:`Factory` bundles what every flow an application builds shares as
structure (the logger, the state factory, the flow class, declared
resources), so construction sites read as ``f.create("grade").call(...)``.
What a run may use (SAIA factory, traits) and where it keeps its history
are not structure: they are the runner's (its
:class:`~llm_gent.core.env.Environment` and its store).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from appinfra.log import Logger

from ..core.cost import Tracker as CostTracker
from .resource import R, ResourceKey, check_resource, resource_method
from .state import StateFactory


if TYPE_CHECKING:
    from .flow import Flow


F = TypeVar("F", bound="Flow")
"""The class of the flows a :class:`Factory` builds."""


class Factory(Generic[F]):
    """App-scoped factory for :class:`Flow` — captures ``lg`` and the shared declarations once.

    :meth:`create` builds a Flow with the captured settings; every
    ``with_*`` method returns a new factory with one setting swapped.

    ``flow_class=`` builds a :class:`Flow` subclass instead — typed:
    ``Factory(lg, flow_class=MyFlow).create()`` is a ``MyFlow``, and
    so are the subflows its ``lambda b: ...`` bodies build.
    """

    @overload
    def __init__(
        self: Factory[Flow],
        lg: Logger,
        *,
        cost_tracker: CostTracker | None = None,
        state_factory: StateFactory[Any] | None = None,
        flow_class: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        lg: Logger,
        *,
        cost_tracker: CostTracker | None = None,
        state_factory: StateFactory[Any] | None = None,
        flow_class: type[F],
    ) -> None: ...

    def __init__(
        self,
        lg: Logger,
        *,
        cost_tracker: CostTracker | None = None,
        state_factory: StateFactory[Any] | None = None,
        flow_class: type[Any] | None = None,
    ) -> None:
        """Capture the shared settings for subsequent :meth:`create` calls.

        Args:
            lg: Logger threaded into every :class:`Flow` this factory builds.
            cost_tracker: Optional :class:`CostTracker` attached via
                :meth:`Flow.with_cost_tracker` on every built flow. Wire once at
                the factory to thread the same cost tracker through an
                entire agent shape.
            state_factory: Optional :class:`StateFactory` threaded into
                every :class:`Flow`'s ``state_factory=`` slot: ``new()``
                builds the state of a run given no ``state=``, ``restore()``
                rebuilds it from a checkpoint. Wrap a stateless type in
                :class:`TypeStateFactory`; implement :class:`StateFactory`
                directly for state that binds runtime handles. ``None``
                (default): a run given no state gets a fresh empty ``dict``.
            flow_class: A :class:`Flow` subclass to build instead of
                :class:`Flow`; it keeps :class:`Flow`'s constructor
                signature. ``None`` (default) builds :class:`Flow`.
        """
        self._lg = lg
        self._cost_tracker = cost_tracker
        self._state_factory = state_factory
        self._flow_class = flow_class
        self._resources: dict[ResourceKey[Any], Any] = {}
        self._resource_methods: dict[str, ResourceKey[Any]] = {}
        self._built_class: type[F] | None = None

    def create(self, name: str = "") -> F:
        """Return a :class:`Flow` (``flow_class``) built with this factory's settings.

        The flow is structure: it can run many times, at once too. What
        belongs to one run — its state, halt, signals, history and
        environment — goes to the runner that starts it.

        Args:
            name: Optional identifier — used in error messages and traces.
                Also lets the flow serve as a named node inside a parent
                chain.
        """
        flow = self._class()(self._lg, name, state_factory=self._state_factory)
        if self._cost_tracker is not None:
            flow.with_cost_tracker(self._cost_tracker)
        for key, value in self._resources.items():
            flow.with_resource(key, value)
        return flow

    def _class(self) -> type[F]:
        """The class :meth:`create` builds: ``flow_class``, with this factory's resource methods.

        Without resource methods it is ``flow_class`` itself; with them, a
        subclass of it made once per factory, so :class:`Flow` and other
        factories' flows are unchanged.
        """
        if self._built_class is None:
            from .flow import Flow

            base: type[Any] = self._flow_class if self._flow_class is not None else Flow
            if self._resource_methods:
                methods = {n: resource_method(k) for n, k in self._resource_methods.items()}
                attrs = {"__module__": base.__module__, "__qualname__": base.__qualname__}
                base = type(base.__name__, (base,), {**attrs, **methods})
            self._built_class = base
        return self._built_class

    def _replace(self, **slots: Any) -> Factory[F]:
        """A copy of this factory with ``slots`` (attribute names without ``_``) replaced."""
        new = copy.copy(self)
        for name, value in slots.items():
            setattr(new, f"_{name}", value)
        new._built_class = None
        return new

    def with_resource(self, key: ResourceKey[R], value: R) -> Factory[F]:
        """Return a new :class:`Factory` that declares ``value`` under ``key`` on every flow.

        Every subsequently created :class:`Flow` gets
        :meth:`Flow.with_resource` ``(key, value)`` — the same object on
        every flow, so it is kept once in the run's checkpoints, at the
        top-level flow. Every other captured slot carries over.

        Raises:
            TypeError: ``key`` is not a :class:`ResourceKey`, or ``value``
                does not implement ``snapshot()`` and ``restore(data)``.
            ValueError: ``value`` is ``None``.
        """
        check_resource(key, value, {})
        return self._replace(resources={**self._resources, key: value})

    def with_resource_method(self, name: str, key: ResourceKey[Any]) -> Factory[F]:
        """Return a new :class:`Factory` whose flows have ``name``: ``with_resource(key, ...)``.

        ``factory.with_resource_method("with_stats", STATS)`` gives every
        flow the new factory builds — and every subflow its ``lambda b:
        ...`` bodies build — a ``with_stats(value, **child_args)`` that is
        :meth:`Flow.with_resource` for ``STATS``. The method lives on a
        subclass the factory makes once; :class:`Flow` itself and other
        factories' flows are unchanged. Type checkers do not see it: for a
        typed method, assign :func:`~llm_gent.flow.resource_method` on a
        :class:`Flow` subclass and pass it as ``flow_class=``.

        Raises:
            TypeError: ``key`` is not a :class:`ResourceKey`.
            ValueError: ``name`` is not an identifier, the flow class
                already has an attribute ``name``, or this factory already
                maps ``name`` to another key.
        """
        if not isinstance(key, ResourceKey):
            raise TypeError(f"with_resource_method takes a ResourceKey; got {type(key).__name__}")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"a resource method's name must be an identifier; got {name!r}")
        current = self._resource_methods.get(name)
        if current is not None and current is not key:
            raise ValueError(f"this factory already maps {name!r} to resource {current.name!r}")
        if current is None and hasattr(self._class(), name):
            raise ValueError(f"{name!r} is already an attribute of the flow class")
        return self._replace(resource_methods={**self._resource_methods, name: key})

    def with_cost_tracker(self, tracker: CostTracker) -> Factory[F]:
        """Return a new :class:`Factory` whose cost tracker is swapped.

        Every other captured slot carries over. Every subsequently created
        :class:`Flow` gets ``tracker`` attached via
        :meth:`Flow.with_cost_tracker` — one wiring reaches every layer that
        observes ``ctx.cost``.
        """
        return self._replace(cost_tracker=tracker)

    def with_state_factory(self, state_factory: StateFactory[Any] | None) -> Factory[F]:
        """Return a new :class:`Factory` whose state factory is swapped.

        Every other captured slot carries over. Useful for subsystems that
        need a different state restore strategy (e.g., a plugin with its
        own state type).
        """
        return self._replace(state_factory=state_factory)
