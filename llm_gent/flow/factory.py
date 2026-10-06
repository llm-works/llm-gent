# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Factory surfaces for the flow substrate.

Two factories live here, at different scopes:

- :class:`SAIAFactory` — protocol that turns a :class:`Role` into a saia
  instance. Typically one per application; the wiring (backend, tools,
  system prompt) is deployment-specific so the framework only names the
  contract.
- :class:`Factory` — app-scoped bundle of the ambient ``lg`` and (by
  convention) a single ``SAIAFactory``. Provides :meth:`create` for
  building Flows without repeating those two arguments at every
  construction site, and :meth:`with_saia_factory` for deriving a
  factory that swaps the SAIAFactory (e.g. a plugin subsystem).
"""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, overload

from appinfra.log import Logger

from ..core.cost import CostTracker
from ..core.traits import Registry as TraitRegistry
from .checkpoint import CheckpointStore
from .nodes import UNSET
from .resource import R, ResourceKey, check_resource, resource_method
from .role import Role
from .state import StateFactory


if TYPE_CHECKING:
    from .flow import Flow


F = TypeVar("F", bound="Flow")
"""The class of the flows a :class:`Factory` builds."""


class SAIAFactory(Protocol):
    """Constructs a role-bound saia instance.

    Implementations decide how a :class:`Role` maps to a backend + tools +
    system prompt + call options. The framework's only expectation is that
    :meth:`build` returns an object supporting saia's public surface — the
    verb calling code uses ``.verify(...)`` / ``.complete(...)`` / etc.

    Implementations MAY read :attr:`Role.params` for per-run parameters
    (max_iterations, cost trackers, session identifiers, task state —
    anything the consumer needs at build time that isn't captured in the
    typed Role fields). Key naming inside ``params`` is a contract between
    a factory and its callers; gent itself imposes none.

    Reference sketch (users typically write one specific to their yaml
    config)::

        class MySAIAFactory:
            def __init__(self, lg, llm_yaml, tools=None):
                self._lg = lg
                self._llm_yaml = llm_yaml
                self._tools = tools or []

            def build(self, role):
                backend = build_backend_from_config(self._llm_yaml, role)
                builder = SAIA.builder().backend(backend).logger(self._lg)
                if "max_iterations" in role.params:
                    builder = builder.max_iterations(role.params["max_iterations"])
                if self._tools:
                    builder = builder.tools(self._tools, executor)
                if role.style:
                    builder = builder.system(role.style)
                return builder.build()
    """

    def build(self, role: Role) -> Any:
        """Return a saia instance configured for ``role``."""
        ...


class Factory(Generic[F]):
    """App-scoped factory for :class:`Flow` — captures ``lg`` and ``saia`` once.

    An application typically has one logger and one :class:`SAIAFactory`
    covering every :class:`Flow` it constructs. Repeating both at every
    Flow-construction site is noise; :class:`Factory` bundles them
    once so subsystem builders read as ``f.create("grade").call(...)``.

    :meth:`create` builds a Flow with the captured defaults;
    :meth:`with_saia_factory` returns a new :class:`Factory` whose
    SAIAFactory is swapped (for subsystems that need a different saia
    builder). Every ``with_*`` method returns a new factory the same way.

    ``flow_class=`` builds a :class:`Flow` subclass instead — typed:
    ``Factory(lg, flow_class=MyFlow).create()`` is a ``MyFlow``, and
    so are the subflows its ``lambda b: ...`` bodies build.
    """

    @overload
    def __init__(
        self: Factory[Flow],
        lg: Logger,
        *,
        saia_factory: SAIAFactory | None = None,
        state: Any = UNSET,
        traits: TraitRegistry | None = None,
        halt: asyncio.Event | None = None,
        cost_tracker: CostTracker | None = None,
        state_factory: StateFactory[Any] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        flow_class: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        lg: Logger,
        *,
        saia_factory: SAIAFactory | None = None,
        state: Any = UNSET,
        traits: TraitRegistry | None = None,
        halt: asyncio.Event | None = None,
        cost_tracker: CostTracker | None = None,
        state_factory: StateFactory[Any] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        flow_class: type[F],
    ) -> None: ...

    def __init__(
        self,
        lg: Logger,
        *,
        saia_factory: SAIAFactory | None = None,
        state: Any = UNSET,
        traits: TraitRegistry | None = None,
        halt: asyncio.Event | None = None,
        cost_tracker: CostTracker | None = None,
        state_factory: StateFactory[Any] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        flow_class: type[Any] | None = None,
    ) -> None:
        """Capture the ambient environment for subsequent :meth:`create` calls.

        Args:
            lg: Logger threaded into every :class:`Flow` this factory builds.
            saia_factory: A :class:`SAIAFactory` that builds role-bound
                saia instances. Threaded into every :class:`Flow` this
                factory builds.
            state: Default construction ``state`` for built flows. Per-Flow
                overrides go through :meth:`create`; per-run overrides go
                through :meth:`Flow.run`.
            traits: Optional trait registry propagated to every :class:`Flow`
                built by this factory. Verbs reach mounted capabilities via
                ``ctx.traits``. ``None`` yields flows with ``ctx.traits is
                None``.
            halt: Optional :class:`asyncio.Event` attached via
                :meth:`Flow.with_halt` on every built flow. Wire once at
                the factory to thread the same halt handle through an
                entire agent shape.
            cost_tracker: Optional :class:`CostTracker` attached via
                :meth:`Flow.with_cost_tracker` on every built flow. Wire once at
                the factory to thread the same cost tracker through an
                entire agent shape.
            state_factory: Optional :class:`StateFactory` threaded into
                every :class:`Flow`'s ``state_factory=`` slot. Consumed by
                :meth:`Flow.run` ``resume="latest"`` to reconstruct
                ``ctx.state.data`` from a loaded checkpoint via
                ``state_factory.restore(...)``. Wrap a stateless type in
                :class:`TypeStateFactory`; implement :class:`StateFactory`
                directly for state that binds runtime handles. ``None``
                (default) treats the payload as a plain dict.
            checkpoint_store: Optional :class:`CheckpointStore` captured for
                subsequent :meth:`create` calls. Only bound to a built
                :class:`Flow` (:meth:`Flow.with_checkpoint_store`) when
                :meth:`create` is passed a ``client_flow_id=`` — the id
                scopes the history and is agent-owned per Flow instance.
                Saves inside the run also need
                :meth:`Flow.with_checkpointer`.
            flow_class: A :class:`Flow` subclass to build instead of
                :class:`Flow`; it keeps :class:`Flow`'s constructor
                signature. ``None`` (default) builds :class:`Flow`.
        """
        self._lg = lg
        self._saia_factory = saia_factory
        self._state = state
        self._traits = traits
        self._halt = halt
        self._cost_tracker = cost_tracker
        self._state_factory = state_factory
        self._checkpoint_store = checkpoint_store
        self._flow_class = flow_class
        self._resources: dict[ResourceKey[Any], Any] = {}
        self._resource_methods: dict[str, ResourceKey[Any]] = {}
        self._built_class: type[F] | None = None

    def create(
        self,
        name: str = "",
        *,
        state: Any = UNSET,
        client_flow_id: str | None = None,
        halt: asyncio.Event | None = None,
        checkpointer: tuple[CheckpointStore, str] | None = None,
    ) -> F:
        """Return a :class:`Flow` (``flow_class``) using this factory's captured environment.

        Args:
            name: Optional identifier — used in error messages and traces.
                Also lets the flow serve as a named node inside a parent
                chain.
            state: Per-Flow construction ``state`` override. Passing
                :data:`UNSET` (default) inherits the factory's ``state``;
                passing ``None`` explicitly is honored as "payload is
                ``None``"; any other value replaces the factory default.
            client_flow_id: Per-Flow history identifier for the
                captured :class:`CheckpointStore`. Required to bind the
                store — the built Flow gets
                :meth:`Flow.with_checkpoint_store` called with
                ``(store, client_flow_id)`` only when both this argument
                is supplied AND the factory carries a store.
                ``None`` (default) leaves the built Flow unwired even
                when the factory carries a store. Ignored when
                ``checkpointer=`` is passed (that argument carries its
                own id).
            halt: Per-Flow halt event override. Wires
                :meth:`Flow.with_halt` on the built flow. ``None``
                (default) falls back to the factory's captured
                ``halt=``; passing an event here supersedes it. Use to
                give each concurrent flow instance its own halt handle
                while keeping one shared factory.
            checkpointer: Per-Flow ``(store, client_flow_id)`` pair —
                atomically binds both via :meth:`Flow.with_checkpoint_store`.
                Supersedes the factory's captured ``checkpoint_store`` and
                the ``client_flow_id`` argument above; use when the
                store differs from the factory's default or when a
                shared factory hands each flow its own history id.
                ``None`` (default) inherits the factory's store paired
                with ``client_flow_id``.
        """
        resolved_state = self._state if state is UNSET else state
        flow = self._class()(
            self._lg,
            name,
            saia_factory=self._saia_factory,
            state=resolved_state,
            traits=self._traits,
            state_factory=self._state_factory,
        )
        effective_halt = halt if halt is not None else self._halt
        if effective_halt is not None:
            flow.with_halt(effective_halt)
        if self._cost_tracker is not None:
            flow.with_cost_tracker(self._cost_tracker)
        for key, value in self._resources.items():
            flow.with_resource(key, value)
        self._bind_store(flow, client_flow_id, checkpointer)
        return flow

    def _bind_store(
        self, flow: F, client_flow_id: str | None, checkpointer: tuple[CheckpointStore, str] | None
    ) -> None:
        """Bind ``flow``'s repo per :meth:`create`'s ``client_flow_id`` / ``checkpointer``."""
        if checkpointer is not None:
            store, flow_id = checkpointer
            flow.with_checkpoint_store(store, flow_id)
        elif self._checkpoint_store is not None and client_flow_id is not None:
            flow.with_checkpoint_store(self._checkpoint_store, client_flow_id)

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

    def with_saia_factory(self, saia_factory: SAIAFactory) -> Factory[F]:
        """Return a new :class:`Factory` whose :class:`SAIAFactory` is swapped.

        Every other captured slot (``lg``, ``state``, ``traits``, ``halt``,
        ``cost_tracker``, ``state_factory``, ``checkpoint_store``, the flow
        class, resources and resource methods) carries over. Useful for
        subsystems that share the app's logger but need a different saia
        builder (e.g. a plugin with its own model wiring).
        """
        return self._replace(saia_factory=saia_factory)

    def with_traits(self, traits: TraitRegistry | None) -> Factory[F]:
        """Return a new :class:`Factory` whose trait registry is swapped.

        Every other captured slot carries over. Mirrors
        :meth:`with_saia_factory` for the trait dimension.
        """
        return self._replace(traits=traits)

    def with_halt(self, event: asyncio.Event) -> Factory[F]:
        """Return a new :class:`Factory` whose halt event is swapped.

        Every other captured slot carries over. Every subsequently created
        :class:`Flow` gets ``event`` attached via :meth:`Flow.with_halt` —
        one wiring reaches every layer that observes ``ctx.halt``.
        """
        return self._replace(halt=event)

    def with_cost_tracker(self, tracker: CostTracker) -> Factory[F]:
        """Return a new :class:`Factory` whose cost tracker is swapped.

        Every other captured slot carries over. Every subsequently created
        :class:`Flow` gets ``tracker`` attached via
        :meth:`Flow.with_cost_tracker` — one wiring reaches every layer that
        observes ``ctx.cost``.
        """
        return self._replace(cost_tracker=tracker)

    def with_checkpoint_store(self, store: CheckpointStore) -> Factory[F]:
        """Return a new :class:`Factory` whose checkpoint store is swapped.

        Every other captured slot carries over. The store binds to each
        built :class:`Flow` only when :meth:`create` is called with a
        ``client_flow_id=`` — the history identifier is agent-owned
        per Flow instance, so the factory captures the store once and
        the id is chosen at construction time.
        """
        return self._replace(checkpoint_store=store)

    def with_state_factory(self, state_factory: StateFactory[Any] | None) -> Factory[F]:
        """Return a new :class:`Factory` whose state factory is swapped.

        Every other captured slot carries over. Useful for subsystems that
        need a different state restore strategy (e.g., a plugin with its
        own state type).
        """
        return self._replace(state_factory=state_factory)
