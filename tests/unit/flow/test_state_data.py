# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""StateData — the checkpoint serialization contract.

Any class implementing ``to_dict()`` and ``classmethod from_dict()``
structurally satisfies :class:`StateData`; no inheritance is required. The
framework checks conformance at checkpoint save/load time via isinstance
against the runtime-checkable Protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Self

from pydantic import BaseModel

from llm_gent.flow import StateData


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
