# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""StateData — the checkpoint serialization contract.

Any class implementing ``to_dict()`` and ``classmethod from_dict()``
structurally satisfies :class:`StateData`; no inheritance is required. The
framework checks conformance at checkpoint save/load time via isinstance
against the runtime-checkable Protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Self

from pydantic import BaseModel

from llm_gent.flow import StateData, StateDataclass


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


class TestStateDataclassMixin:
    """The :class:`StateDataclass` mixin — asdict / cls(**data) convenience."""

    def test_mixin_satisfies_state_data(self) -> None:
        """A flat dataclass inheriting the mixin satisfies :class:`StateData`."""

        @dataclass
        class Counter(StateDataclass):
            count: int = 0
            log: list[int] = field(default_factory=list)

        assert isinstance(Counter(), StateData)

    def test_flat_round_trip(self) -> None:
        """Scalars, lists, and dict fields round-trip through the mixin."""

        @dataclass
        class Payload(StateDataclass):
            name: str
            count: int = 0
            log: list[int] = field(default_factory=list)
            meta: dict[str, int] = field(default_factory=dict)

        original = Payload(name="foo", count=3, log=[1, 2], meta={"k": 9})
        loaded = Payload.from_dict(original.to_dict())
        assert loaded == original

    def test_to_dict_recurses_into_nested_dataclass(self) -> None:
        """:func:`dataclasses.asdict` recurses — nested fields land as dicts.

        Documents the asymmetry with :meth:`from_dict`, which does NOT
        reconstruct nested instances (see the next test).
        """

        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer(StateDataclass):
            inner: Inner
            n: int = 0

        original = Outer(inner=Inner(x=7), n=1)
        payload = original.to_dict()
        assert payload == {"inner": {"x": 7}, "n": 1}

    def test_from_dict_does_not_reconstruct_nested_dataclass(self) -> None:
        """Nested-dataclass limit: :meth:`from_dict` leaves the field as dict.

        Consumers with nested state override :meth:`from_dict`. This test
        pins the naive behavior so a future implementation swap surfaces
        the change deliberately.
        """

        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer(StateDataclass):
            inner: Inner
            n: int = 0

        loaded = Outer.from_dict({"inner": {"x": 7}, "n": 1})
        assert loaded.inner == {"x": 7}
        assert not isinstance(loaded.inner, Inner)

    def test_subclass_can_override_from_dict_for_nested(self) -> None:
        """Consumers with nested state can override :meth:`from_dict`."""

        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer(StateDataclass):
            inner: Inner
            n: int = 0

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> Self:
                return cls(inner=Inner(**data["inner"]), n=int(data["n"]))

        original = Outer(inner=Inner(x=7), n=1)
        loaded = Outer.from_dict(original.to_dict())
        assert loaded == original
