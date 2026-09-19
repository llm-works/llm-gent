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

from dataclasses import asdict, dataclass, field
from typing import Any, Generic, Protocol, Self, TypeVar, cast, runtime_checkable


T = TypeVar("T")
"""Payload type carried by a :class:`State`.

Consumers who want typed payload access annotate the enclosing
:class:`~llm_gent.flow.context.Context` as ``Context[MyState]``; the parameter
threads through to ``ctx.state.data``. Unparameterized usage remains valid
and treats the payload as :data:`Any`.
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
    """Mixin that satisfies :class:`StateData` for flat dataclasses.

    Opt-in convenience for the common case: a dataclass whose fields are
    JSON-native scalars, lists, or dicts. Inheriting this mixin provides
    :meth:`to_dict` via :func:`dataclasses.asdict` and
    :meth:`from_dict` via ``cls(**data)`` — no per-field boilerplate.

    Usage::

        @dataclass
        class Counter(StateDataclass):
            count: int = 0
            log: list[int] = field(default_factory=list)

    Limitations
    -----------
    :func:`dataclasses.asdict` recurses into nested dataclasses and
    produces plain dicts for them; the naive :meth:`from_dict` does
    **not** recurse — nested-dataclass fields land as dicts, not
    reconstructed instances. Consumers with nested dataclass state
    override :meth:`from_dict` (or both) with a shape-aware version.

    Type coercion is likewise absent: fields are assigned as-is from the
    input dict. Sources that may deliver values with drifted types
    (e.g., ints as strings) must override :meth:`from_dict` to coerce.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize via :func:`dataclasses.asdict`.

        Requires the subclass to be a dataclass. Recurses into nested
        dataclass fields — see the class docstring on the asymmetric
        :meth:`from_dict` behavior.
        """
        return asdict(cast(Any, self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct via ``cls(**data)``.

        Flat-dataclass path only; nested-dataclass fields are NOT
        reconstructed — see the class docstring.
        """
        return cls(**data)


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
