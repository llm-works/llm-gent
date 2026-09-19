# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""StateData — the checkpoint serialization contract.

Any class implementing ``to_dict()`` and ``classmethod from_dict()``
structurally satisfies :class:`StateData`; no inheritance is required. The
framework checks conformance at checkpoint save/load time via isinstance
against the runtime-checkable Protocol.

Fixture classes live at module level because :func:`typing.get_type_hints`
cannot resolve annotations that reference names local to a test function.
Real state classes are always module-level, so the test surface matches
production shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Self

import pytest
from pydantic import BaseModel

from llm_gent.flow import StateData, StateDataclass


# ── fixture classes ──────────────────────────────────────────────────


class _BMScalar(BaseModel):
    k: str
    v: int = 0


class _BMSmall(BaseModel):
    k: str


class _BMItem(BaseModel):
    n: int


class _BMLeft(BaseModel):
    k: str


class _BMRight(BaseModel):
    n: int


@dataclass
class _PlainInner:
    x: int


class _Color(StrEnum):
    RED = "red"
    BLUE = "blue"


@dataclass
class _InnerStateDC(StateDataclass):
    k: str
    v: int = 0


# ── protocol conformance ─────────────────────────────────────────────


class TestPydanticConformance:
    """Pydantic models bridged to to_dict / from_dict satisfy StateData."""

    def test_pydantic_model_satisfies(self) -> None:
        """A BaseModel that bridges model_dump / model_validate satisfies."""

        class MyState(BaseModel):
            name: str
            count: int = 0

            def to_dict(self) -> dict[str, Any]:
                return self.model_dump(mode="json")

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                return cls.model_validate(data)

        assert isinstance(MyState(name="foo"), StateData)

    def test_pydantic_round_trip(self) -> None:
        """to_dict output feeds from_dict back to an equal instance."""

        class MyState(BaseModel):
            name: str
            count: int = 0

            def to_dict(self) -> dict[str, Any]:
                return self.model_dump(mode="json")

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                return cls.model_validate(data)

        original = MyState(name="foo", count=3)
        loaded = MyState.from_dict(original.to_dict())
        assert loaded == original


class TestDataclassConformance:
    """Dataclasses with hand-written serializers satisfy StateData."""

    def test_dataclass_satisfies(self) -> None:
        """A frozen dataclass with to_dict / from_dict satisfies."""

        @dataclass(frozen=True)
        class MyState:
            name: str
            count: int = 0

            def to_dict(self) -> dict[str, Any]:
                return asdict(self)

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                return cls(**data)

        assert isinstance(MyState(name="foo"), StateData)

    def test_dataclass_round_trip(self) -> None:
        """Dataclass to_dict/from_dict pair round-trips."""

        @dataclass(frozen=True)
        class MyState:
            name: str
            count: int = 0

            def to_dict(self) -> dict[str, Any]:
                return asdict(self)

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                return cls(**data)

        original = MyState(name="foo", count=3)
        loaded = MyState.from_dict(original.to_dict())
        assert loaded == original


class TestCustomClassConformance:
    """Plain classes with the two methods also satisfy structurally."""

    def test_custom_class_satisfies(self) -> None:
        """No BaseModel, no dataclass — just the two methods."""

        class MyState:
            def __init__(self, name: str) -> None:
                self.name = name

            def to_dict(self) -> dict[str, Any]:
                return {"name": self.name}

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                return cls(name=data["name"])

        assert isinstance(MyState("foo"), StateData)


class TestNonConformance:
    """Values that don't structurally match StateData don't satisfy."""

    def test_bare_object_does_not_satisfy(self) -> None:
        """A class with neither method fails the runtime check."""

        class NoMethods:
            def __init__(self, x: int) -> None:
                self.x = x

        assert not isinstance(NoMethods(1), StateData)

    def test_partial_conformance_rejected(self) -> None:
        """to_dict alone (no from_dict) is not enough."""

        class HalfWay:
            def to_dict(self) -> dict[str, Any]:
                return {}

        assert not isinstance(HalfWay(), StateData)

    def test_plain_dict_does_not_satisfy(self) -> None:
        """Dicts pass through the framework's dict path, not the Protocol.

        The checkpoint layer will accept dicts as passthrough serialization
        without requiring them to satisfy :class:`StateData`.
        """
        assert not isinstance({"any": "thing"}, StateData)


# ── StateDataclass mixin: flat and nested-dataclass cases ────────────


@dataclass
class _FlatPayload(StateDataclass):
    name: str
    count: int = 0
    log: list[int] = field(default_factory=list)
    meta: dict[str, int] = field(default_factory=dict)


@dataclass
class _PlainNestedOuter(StateDataclass):
    inner: _PlainInner
    n: int = 0


class TestStateDataclassMixin:
    """The :class:`StateDataclass` mixin — flat + nested-dataclass round-trips."""

    def test_mixin_satisfies_state_data(self) -> None:
        """A flat dataclass inheriting the mixin satisfies :class:`StateData`."""

        @dataclass
        class Counter(StateDataclass):
            count: int = 0

        assert isinstance(Counter(), StateData)

    def test_flat_round_trip(self) -> None:
        """Scalars, lists, and dict fields round-trip through the mixin."""
        original = _FlatPayload(name="foo", count=3, log=[1, 2], meta={"k": 9})
        loaded = _FlatPayload.from_dict(original.to_dict())
        assert loaded == original

    def test_nested_plain_dataclass_round_trip(self) -> None:
        """Plain nested dataclass encodes AND decodes — annotation-driven."""
        original = _PlainNestedOuter(inner=_PlainInner(x=7), n=1)
        payload = original.to_dict()
        assert payload == {"inner": {"x": 7}, "n": 1}

        loaded = _PlainNestedOuter.from_dict(payload)
        assert loaded == original
        assert isinstance(loaded.inner, _PlainInner)

    def test_subclass_can_override_from_dict(self) -> None:
        """A subclass override always wins — the mixin never intercepts it."""

        @dataclass
        class OuterOverride(StateDataclass):
            inner: _PlainInner
            n: int = 0

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                # Force a non-default reconstruction so the override is
                # observable; verifies subclass logic runs, not the mixin.
                return cls(inner=_PlainInner(x=data["inner"]["x"] + 100), n=int(data["n"]))

        loaded = OuterOverride.from_dict({"inner": {"x": 7}, "n": 1})
        assert loaded == OuterOverride(inner=_PlainInner(x=107), n=1)


# ── StateDataclass mixin: Pydantic field auto-recursion ──────────────


@dataclass
class _PydRequired(StateDataclass):
    inner: _BMScalar


@dataclass
class _PydOptional(StateDataclass):
    inner: _BMSmall | None = None


@dataclass
class _PydList(StateDataclass):
    items: list[_BMItem] = field(default_factory=list)


@dataclass
class _PydDict(StateDataclass):
    by_key: dict[str, _BMItem] = field(default_factory=dict)


class TestStateDataclassPydanticFields:
    """Auto-recursion into :class:`~pydantic.BaseModel` fields — the load-bearing case."""

    def test_pydantic_field_round_trip(self) -> None:
        """State with a required BaseModel field round-trips."""
        original = _PydRequired(inner=_BMScalar(k="foo", v=3))
        loaded = _PydRequired.from_dict(original.to_dict())
        assert loaded == original
        assert isinstance(loaded.inner, _BMScalar)

    def test_optional_pydantic_field_populated(self) -> None:
        """``T | None`` with a value round-trips as T."""
        original = _PydOptional(inner=_BMSmall(k="foo"))
        loaded = _PydOptional.from_dict(original.to_dict())
        assert loaded == original
        assert isinstance(loaded.inner, _BMSmall)

    def test_optional_pydantic_field_none(self) -> None:
        """``T | None`` with None round-trips as None."""
        original = _PydOptional(inner=None)
        loaded = _PydOptional.from_dict(original.to_dict())
        assert loaded == original
        assert loaded.inner is None

    def test_list_of_pydantic_round_trip(self) -> None:
        """``list[BaseModel]`` recurses element-wise on both sides."""
        original = _PydList(items=[_BMItem(n=1), _BMItem(n=2), _BMItem(n=3)])
        loaded = _PydList.from_dict(original.to_dict())
        assert loaded == original
        assert all(isinstance(i, _BMItem) for i in loaded.items)

    def test_dict_of_pydantic_round_trip(self) -> None:
        """``dict[str, BaseModel]`` recurses over values."""
        original = _PydDict(by_key={"a": _BMItem(n=1), "b": _BMItem(n=2)})
        loaded = _PydDict.from_dict(original.to_dict())
        assert loaded == original
        assert all(isinstance(v, _BMItem) for v in loaded.by_key.values())


# ── StateDataclass mixin: Enum, nested StateDataclass, Optional scalar ─


@dataclass
class _EnumOuter(StateDataclass):
    color: _Color = _Color.RED


@dataclass
class _NestedStateDCOuter(StateDataclass):
    inner: _InnerStateDC


@dataclass
class _OptionalScalar(StateDataclass):
    count: int | None = None


class TestStateDataclassOtherRecursion:
    """Auto-recursion into other supported shapes: Enum, nested StateDataclass, Optional scalar."""

    def test_enum_field_round_trip(self) -> None:
        """``StrEnum`` fields encode via ``.value`` and decode via ``T(value)``."""
        original = _EnumOuter(color=_Color.BLUE)
        loaded = _EnumOuter.from_dict(original.to_dict())
        assert loaded == original
        assert isinstance(loaded.color, _Color)

    def test_nested_state_dataclass_round_trip(self) -> None:
        """A :class:`StateDataclass` field delegates to the inner mixin."""
        original = _NestedStateDCOuter(inner=_InnerStateDC(k="foo", v=7))
        loaded = _NestedStateDCOuter.from_dict(original.to_dict())
        assert loaded == original
        assert isinstance(loaded.inner, _InnerStateDC)

    def test_optional_scalar_round_trip(self) -> None:
        """``int | None`` decodes as-is for either arm."""
        assert _OptionalScalar.from_dict(_OptionalScalar(count=5).to_dict()) == _OptionalScalar(
            count=5
        )
        assert _OptionalScalar.from_dict(_OptionalScalar(count=None).to_dict()) == _OptionalScalar(
            count=None
        )


# ── StateDataclass mixin: escalation ─────────────────────────────────


@dataclass
class _BadEncode(StateDataclass):
    error: Any = None


@dataclass
class _AmbiguousUnion(StateDataclass):
    payload: _BMLeft | _BMRight = field(default_factory=lambda: _BMLeft(k=""))


class TestStateDataclassEscalation:
    """Unsupported shapes raise :class:`TypeError` naming the field."""

    def test_unsupported_value_type_raises(self) -> None:
        """A field holding a non-JSON-native object (an exception) raises."""
        with pytest.raises(TypeError, match="field 'error'"):
            _BadEncode(error=RuntimeError("boom")).to_dict()

    def test_ambiguous_union_raises(self) -> None:
        """``A | B`` with two non-None types cannot be reconstructed."""
        payload = _AmbiguousUnion(payload=_BMLeft(k="foo")).to_dict()
        with pytest.raises(TypeError, match="field 'payload'"):
            _AmbiguousUnion.from_dict(payload)
