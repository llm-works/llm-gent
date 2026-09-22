# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""State — scope-aware wrapper around a user-owned payload.

Every :meth:`Flow.run` wraps its incoming state as a :class:`State`; sub-flows
opened with ``state=`` on ``.call`` / ``.loop`` / ``.map`` produce a child
:class:`State` whose ``_parent`` links back to the enclosing scope. Verbs
access their scope's payload via :attr:`data` and reach run-wide state via
:meth:`root`.

The payload is user-owned and opaque to the framework — dict, dataclass,
Pydantic model, arbitrary object. The framework wraps but never inspects.

Checkpoint serialization contract
---------------------------------
Payloads that need to round-trip through a checkpoint MUST satisfy
:class:`StateData` — implement ``to_dict()`` and
``classmethod from_dict(cls, data)`` — OR be a plain ``dict``. Dicts pass
through the checkpointer as-is.

Serialization is **consumer-owned**: the framework cannot call ``from_dict``
because it doesn't know the payload's concrete type. Consumers call
``state.data.to_dict()`` in their save hook and
``PayloadClass.from_dict(checkpoint["data"])`` in their resume hook.

The contract only surfaces when ``.with_checkpointer(...)`` is wired on
the enclosing flow. Consumers who never checkpoint don't need to conform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, Self, TypeVar, cast, runtime_checkable

from .serialization import state_converter


T = TypeVar("T")
"""Payload type carried by a :class:`State`.

Consumers who want typed payload access annotate the enclosing
:class:`~llm_gent.flow.context.Context` as ``Context[MyState]``; the parameter
threads through to ``ctx.state.data``. Unparameterized usage remains valid
and treats the payload as :data:`Any`.
"""


T_co = TypeVar("T_co", covariant=True)
"""Covariant payload variant used where the type only appears in return
position — notably :class:`StateFactory`, whose sole method produces a
``T`` but never consumes one. A ``StateFactory[Subclass]`` naturally
satisfies ``StateFactory[Parent]``.
"""


@runtime_checkable
class StateData(Protocol):
    """Serialization contract for :class:`State.data` payloads under checkpoint.

    Implementations MUST provide:

    - ``to_dict(self) -> dict[str, Any]`` — serialize instance state to a
      JSON-compatible dict.
    - ``from_dict(cls, data: dict[str, Any]) -> Self`` — reconstruct an
      instance from a serialized dict.

    Structural: any class (Pydantic, dataclass, TypedDict wrapper, custom)
    with both methods satisfies. No inheritance required.

    **Consumer-owned:** the framework exposes the contract but does not call
    these methods automatically — it cannot call ``from_dict`` without knowing
    the concrete class. Consumers serialize in their checkpoint hooks.

    Plain dicts pass through the checkpointer as-is and are NOT required to
    implement :class:`StateData`.
    """

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...


class StateDataclass:
    """Mixin that satisfies :class:`StateData` for dataclass state payloads.

    Both methods delegate to :data:`state_converter`, a module-level
    :class:`cattrs.Converter` built from the JSON preconf preset. Out of
    the box the converter walks:

    - :class:`~pydantic.BaseModel` — ``model_dump(mode="json")`` /
      ``model_validate(...)`` registered as a hook factory; every
      BaseModel subclass is handled without per-class setup.
    - Nested :class:`StateDataclass` — as an ordinary dataclass; cattrs
      recurses uniformly.
    - Plain nested dataclass — recurses field-by-field.
    - ``T | None`` / :class:`typing.Optional` — unwrap None, decode ``T``.
    - ``list[T]`` / ``tuple[T, ...]`` — element-wise recursion; tuple
      type preserved on decode.
    - ``dict[K, V]`` — recurses over values (JSON-native keys only).
    - :class:`~enum.Enum` — ``.value`` / ``EnumType(value)``.
    - :class:`~datetime.datetime`, :class:`~uuid.UUID`,
      :class:`~decimal.Decimal`, :class:`~pathlib.Path`, ``set`` /
      ``frozenset`` — via the JSON preconf hooks.
    - :class:`~typing.Any` — pass-through; the caller owns the runtime shape.

    Usage::

        @dataclass
        class Counter(StateDataclass):
            count: int = 0
            log: list[int] = field(default_factory=list)

    Escalation
    ----------
    Handled fields with unsupported nested values raise
    :class:`cattrs.errors.ClassValidationError` with the full field path;
    types without a registered structure handler (e.g., ambiguous unions)
    raise :class:`cattrs.errors.StructureHandlerNotFoundError` directly.
    Two escape hatches:

    1. Register a hook on :data:`state_converter` — framework-wide,
       covers every state class carrying the type.
    2. Override :meth:`to_dict` / :meth:`from_dict` on the state class.
       Overrides always win. Prefer this when the shape is class-local.

    Ambiguous unions ``A | B`` (two non-None arms) need
    :func:`cattrs.strategies.configure_tagged_union` on the converter, or
    an override on the state class.
    """

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], state_converter.unstructure(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return state_converter.structure(data, cls)


@dataclass(frozen=True)
class State(Generic[T]):
    """Scope-aware wrapper around a user-owned payload.

    ``data`` is the payload the caller supplied at :meth:`Flow.run` (or the
    child payload produced by a ``state=`` projection); the framework carries
    it verbatim without inspecting or dictating its shape.

    Generic in the payload type :data:`T`. Unparameterized ``State`` treats
    the payload as :data:`Any`; ``State[MyState]`` narrows ``data`` to
    ``MyState`` for type-checked access. Runtime shape is unchanged either
    way — the framework never inspects the payload.

    Payloads that need checkpoint round-tripping MUST be plain dicts or
    satisfy :class:`StateData`; see the module docstring.
    """

    data: T = cast(T, None)
    """User-owned payload (dict, dataclass, Pydantic model, arbitrary object).

    Typed via the generic parameter :data:`T` — unparameterized ``State`` is
    ``State[Any]``, so verbs retain rich payload access (dict indexing,
    attribute access, etc.). Defaults to ``None`` for backwards compatibility
    with zero-argument ``State()`` construction. The checkpoint serialization
    contract is :class:`StateData`, enforced only at snapshot time.
    """

    _parent: State[Any] | None = field(default=None, repr=False)
    """Link to the enclosing scope's :class:`State`, or ``None`` at the root.

    Private on purpose — public traversal is via :meth:`root` /
    :attr:`is_root`. Nothing else needs raw parent access today.
    """

    _factory: StateFactory[Any] | None = field(default=None, repr=False)
    """Factory that created this state's payload, for checkpoint restore.

    When a scoped child state is restored from a checkpoint, the framework
    calls ``_factory.restore(data)`` to reconstruct the typed payload.
    Threading: passthrough projection inherits the parent's factory;
    explicit projection attaches the node's factory (or inherits if none).
    ``None`` at the root unless the top-level :class:`Flow` was constructed
    with ``state_factory=``.
    """

    @property
    def is_root(self) -> bool:
        """True when this state has no parent — the outermost scope of a run."""
        return self._parent is None

    def root(self) -> State[Any]:
        """Walk the parent chain to the outermost :class:`State`.

        Returns ``self`` when already at the root. Verbs reading run-wide
        state (budgets, deadlines, shared registries) reach it via
        ``ctx.state.root().data``. The return type is ``State[Any]``
        because a scoped child's payload type has no static relationship
        to its ancestors' — callers know their runtime's top-level type
        and can annotate the read site accordingly.
        """
        node: State[Any] = self
        while node._parent is not None:
            node = node._parent
        return node


class StateFactory(Protocol[T_co]):
    """Framework-facing state construction on the checkpoint restore path.

    The framework calls :meth:`restore` at ``.with_checkpointer(...)``
    resume time to rebuild state from the serialized dict. Runtime
    handles that cannot be serialized (Logger, storage backends,
    connections) are captured at factory construction and threaded
    through :meth:`restore`; the framework passes no runtime context of
    its own.

    Fresh construction is user-owned — the protocol declares no method
    for it. Implementations typically ship a ``new(**kwargs)`` alongside
    for centralized construction (see :class:`TypeStateFactory`), but
    the framework never calls it.

    ``StateFactory[T]`` narrows the payload type; :meth:`restore` returns
    ``T`` so the restore path in :class:`~llm_gent.flow.flow.Flow` is
    statically typed against the caller's state class. ``T`` is
    covariant here because it only appears in return position.
    """

    def restore(self, data: dict[str, Any]) -> T_co: ...


class TypeStateFactory(Generic[T]):
    """:class:`StateFactory` adapter for stateless state types.

    Wraps a bare class satisfying :class:`StateData` in the factory shape
    for the common case where state carries no runtime handles. The
    framework calls :meth:`restore` on resume, which delegates to
    ``state_type.from_dict(data)``.

    :meth:`new` is a user-facing convenience for fresh construction —
    the framework never calls it. It forwards kwargs to the type
    constructor, centralizing state construction alongside restore.

    Usage::

        ff = FlowFactory(lg, state_factory=TypeStateFactory(Counter))

    State that needs runtime bindings should implement
    :class:`StateFactory` directly and inject handles in :meth:`restore`.
    """

    def __init__(self, state_type: type[T]) -> None:
        self._state_type = state_type

    def new(self, **kwargs: Any) -> T:
        """Construct a fresh instance by forwarding kwargs to the type."""
        return self._state_type(**kwargs)

    def restore(self, data: dict[str, Any]) -> T:
        """Restore an instance via ``state_type.from_dict(data)``."""
        return cast(T, self._state_type.from_dict(data))  # type: ignore[attr-defined]
