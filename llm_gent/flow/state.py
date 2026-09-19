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

import dataclasses
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from pydantic import BaseModel


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
    """Mixin that satisfies :class:`StateData` for dataclass state payloads.

    :meth:`to_dict` walks the dataclass fields and encodes each value;
    :meth:`from_dict` walks the resolved type hints and reconstructs each
    field. Both recurse into the shapes that appear naturally in state
    payloads:

    - :class:`~pydantic.BaseModel` — ``model_dump(mode="json")`` /
      ``model_validate(...)``, including nested BaseModels which pydantic
      handles internally.
    - Nested :class:`StateDataclass` — delegates to the inner mixin.
    - Plain nested dataclass — recurses field-by-field.
    - ``T | None`` / :class:`typing.Optional` — unwrap None and decode ``T``.
    - ``list[T]`` / ``tuple[T, ...]`` — element-wise recursion.
    - ``dict[K, V]`` — recurses over values (keys pass through; they must
      be JSON-native).
    - :class:`~enum.Enum` — ``.value`` / ``Enum(value)``.
    - :class:`~typing.Any` — pass-through on both sides; the caller owns
      the runtime shape.
    - JSON primitives — passed through.

    Usage::

        @dataclass
        class Counter(StateDataclass):
            count: int = 0
            log: list[int] = field(default_factory=list)

    Escalation
    ----------
    A field whose value (on encode) or annotation (on decode) does not
    match any of the recognized shapes raises :class:`TypeError` with the
    field name and the offending type. The subclass then overrides
    :meth:`to_dict` and/or :meth:`from_dict` with hand-written logic.
    This is the deliberate escape hatch — Union of multiple non-None arms,
    ``dict[str, Any]`` whose values need discriminated reconstruction, and
    fields wrapping non-JSON-native objects (exceptions, custom sentinels)
    all land here. Overrides always win: the mixin never intercepts a
    method the subclass provides.

    Ambiguous unions specifically: ``A | B`` where both are non-None
    types cannot be reconstructed without a discriminator, so the mixin
    raises rather than guessing. ``T | None`` is fine — the None arm is
    trivially discriminated by value.
    """

    def to_dict(self) -> dict[str, Any]:
        """Encode the dataclass fields to a JSON-native dict.

        Iterates ``dataclasses.fields(self)`` and encodes each value via
        the recursive rules described in the class docstring. Fields
        whose value cannot be auto-encoded raise :class:`TypeError` with
        the field name attached — see the class docstring's Escalation
        section for the override path.
        """
        result: dict[str, Any] = {}
        for f in dataclasses.fields(cast(Any, self)):
            value = getattr(self, f.name)
            try:
                result[f.name] = _encode_value(value)
            except TypeError as e:
                raise TypeError(
                    f"{type(self).__name__}.to_dict(): field {f.name!r} "
                    f"(value type {type(value).__name__!r}) is not "
                    f"auto-serializable — override to_dict()/from_dict(). "
                    f"Cause: {e}"
                ) from e
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct the dataclass from a JSON-native dict.

        Iterates ``dataclasses.fields(cls)`` and decodes each field's raw
        value against its resolved type hint. Missing keys fall back to
        the dataclass field default (or default_factory) — the mixin
        never overrides a default with ``None``. Fields whose annotation
        cannot be auto-decoded raise :class:`TypeError` with the field
        name attached — see the class docstring's Escalation section.
        """
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cast(Any, cls)):
            if f.name not in data:
                continue
            raw = data[f.name]
            annotation = hints.get(f.name, Any)
            try:
                kwargs[f.name] = _decode_value(raw, annotation)
            except TypeError as e:
                raise TypeError(
                    f"{cls.__name__}.from_dict(): field {f.name!r} "
                    f"(annotation {annotation!r}) is not "
                    f"auto-reconstructable — override from_dict(). "
                    f"Cause: {e}"
                ) from e
        return cls(**kwargs)


def _encode_value(value: Any) -> Any:
    """Recursively encode a :class:`StateDataclass` field value.

    Contract lives on :class:`StateDataclass`; this helper implements the
    dispatch. Unrecognized value types raise a bare :class:`TypeError`
    that the caller reraises with the field name attached.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, StateDataclass):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    raise TypeError(f"cannot auto-encode value of type {type(value).__name__!r}")


def _decode_value(raw: Any, annotation: Any) -> Any:
    """Recursively decode a JSON-native ``raw`` value under ``annotation``.

    Contract lives on :class:`StateDataclass`; this helper implements the
    dispatch. Unrecognized annotations raise a bare :class:`TypeError`
    that the caller reraises with the field name attached.
    """
    origin = get_origin(annotation)
    if origin is types.UnionType or origin is Union:
        return _decode_union(raw, annotation)
    if annotation is Any:
        return raw
    if origin is list:
        args = get_args(annotation)
        item_t: Any = args[0] if args else Any
        return [_decode_value(v, item_t) for v in raw]
    if origin is tuple:
        args = get_args(annotation)
        item_t = args[0] if args else Any
        return tuple(_decode_value(v, item_t) for v in raw)
    if origin is dict:
        args = get_args(annotation)
        val_t: Any = args[1] if len(args) == 2 else Any
        return {k: _decode_value(v, val_t) for k, v in raw.items()}
    if isinstance(annotation, type):
        if annotation in (str, int, float, bool):
            return raw
        if issubclass(annotation, BaseModel):
            return annotation.model_validate(raw)
        if issubclass(annotation, StateDataclass):
            return annotation.from_dict(raw)
        if issubclass(annotation, Enum):
            return annotation(raw)
        if dataclasses.is_dataclass(annotation):
            return _decode_plain_dataclass(annotation, raw)
    raise TypeError(f"cannot auto-decode annotation {annotation!r}")


def _decode_union(raw: Any, annotation: Any) -> Any:
    """Decode a Union / ``T | None`` annotation.

    Only ``T | None`` (exactly one non-None arm) is auto-decodable —
    multi-arm unions need a discriminator the mixin cannot invent. Split
    from :func:`_decode_value` to keep that dispatcher small.
    """
    non_none = tuple(a for a in get_args(annotation) if a is not type(None))
    if len(non_none) != 1:
        raise TypeError(
            f"cannot decode union {annotation!r} — more than one non-None "
            f"arm; use a discriminated union or override from_dict()"
        )
    return None if raw is None else _decode_value(raw, non_none[0])


def _decode_plain_dataclass(cls: type, raw: dict[str, Any]) -> Any:
    """Reconstruct a plain dataclass (not a :class:`StateDataclass`) from ``raw``.

    Split from :func:`_decode_value` to keep that dispatcher small.
    Resolves the nested class's own type hints once, then decodes each
    field against them. Missing keys fall back to the field default.
    """
    nested_hints = get_type_hints(cls)
    return cls(
        **{
            f.name: _decode_value(raw[f.name], nested_hints.get(f.name, Any))
            for f in dataclasses.fields(cls)
            if f.name in raw
        }
    )


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
