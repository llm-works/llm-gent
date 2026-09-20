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
from decimal import Decimal
from pathlib import PurePath
from typing import (
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    cast,
    runtime_checkable,
)
from uuid import UUID

from cattrs import Converter
from cattrs.preconf.json import make_converter as _make_json_converter
from pydantic import BaseModel


T = TypeVar("T")
"""Payload type carried by a :class:`State`.

Consumers who want typed payload access annotate the enclosing
:class:`~llm_gent.flow.context.Context` as ``Context[MyState]``; the parameter
threads through to ``ctx.state.data``. Unparameterized usage remains valid
and treats the payload as :data:`Any`.
"""


def _is_basemodel_class(cls: Any) -> bool:
    """Predicate for the cattrs BaseModel hook factory.

    Guards against non-class arguments (generic aliases, TypeVars) that
    ``issubclass`` would reject with :class:`TypeError`.
    """
    try:
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except TypeError:
        return False


def _build_state_converter() -> Converter:
    """Build the module-level converter with the pydantic hook wired.

    Starts from :func:`cattrs.preconf.json.make_converter`, which is the
    JSON-compatible preset — dataclass / Enum / Optional / list / tuple /
    dict / TypedDict / NamedTuple / datetime / :class:`~uuid.UUID` /
    :class:`~decimal.Decimal` / :class:`~pathlib.Path` / set / frozenset
    all have JSON-native round-trip hooks registered.

    :class:`~pydantic.BaseModel` is not one of the preconf hooks, so a
    factory dispatches every BaseModel subclass to
    ``model_dump(mode="json")`` / ``model_validate(...)``.
    """
    conv = _make_json_converter()
    conv.register_unstructure_hook_factory(
        _is_basemodel_class,
        lambda _cls: lambda inst: inst.model_dump(mode="json"),
    )
    conv.register_structure_hook_factory(
        _is_basemodel_class,
        lambda cls: lambda raw, _: cls.model_validate(raw),
    )
    # UUID, Decimal, and pathlib types are not in the JSON preconf's
    # default hooks (verified through cattrs 26.x). Encode as strings; decode via the
    # class constructor. Decimal round-trips through str exactly (float
    # would lose precision).
    conv.register_unstructure_hook(UUID, str)
    conv.register_structure_hook(UUID, lambda raw, _: UUID(raw))
    conv.register_unstructure_hook(Decimal, str)
    conv.register_structure_hook(Decimal, lambda raw, _: Decimal(raw))
    conv.register_unstructure_hook_factory(
        lambda cls: isinstance(cls, type) and issubclass(cls, PurePath),
        lambda _cls: str,
    )
    conv.register_structure_hook_factory(
        lambda cls: isinstance(cls, type) and issubclass(cls, PurePath),
        lambda cls: lambda raw, _: cls(raw),
    )
    return conv


state_converter: Converter = _build_state_converter()
"""Module-level :class:`cattrs.Converter` backing :class:`StateDataclass`.

Built from :func:`cattrs.preconf.json.make_converter` so JSON-native
round-trip is the default — dataclass / Enum / Optional / list / tuple /
dict / TypedDict / NamedTuple / :class:`~datetime.datetime` /
:class:`~uuid.UUID` / :class:`~decimal.Decimal` / :class:`~pathlib.Path` /
set / frozenset all round-trip. :class:`~pydantic.BaseModel` is bridged
via a hook factory registered at import time.

Consumers whose state shape lands outside the converter's built-ins
(heterogeneous ``dict[str, Any]`` with discriminated values, exception
fields, custom sentinels) extend it in two ways:

1. Register a hook on the converter, local to the module that owns the
   shape::

       from llm_gent.flow.state import state_converter

       state_converter.register_unstructure_hook(MyType, _to_dict)
       state_converter.register_structure_hook(MyType, _from_dict)

2. Override :meth:`StateDataclass.to_dict` / :meth:`StateDataclass.from_dict`
   on the state class. Overrides always win — the mixin never intercepts a
   method the subclass provides. Prefer this when the shape is specific to
   one class and doesn't compose across the codebase.
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
