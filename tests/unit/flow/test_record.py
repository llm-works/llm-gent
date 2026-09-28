# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Execution record: value codec, round-trip refusal, input hashes, addresses, shards."""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gent.flow.state.record import (
    ExecutionRecord,
    RecordError,
    decode_value,
    encode_value,
    index_coord,
    instance_address,
    iteration_coord,
    key_coord,
    storable,
    value_hash,
)


pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass
class Outcome:
    run_id: str
    points: list[Point] = field(default_factory=list)


@dataclass(frozen=True)
class Tagged:
    tags: frozenset[str]


class Color(enum.Enum):
    RED = "red"


class Level(enum.IntEnum):
    LOW = 1


class Model(BaseModel):
    name: str
    score: float


class Loose(BaseModel):
    parsed: Any = None


def _round_trip(value: Any) -> Any:
    import json

    from llm_gent.flow.state.cas import canonical_json

    return decode_value(json.loads(canonical_json(storable(value))))


class TestRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            0,
            -3,
            1.5,
            "text",
            [1, "a", None],
            (1, 2),
            {"a": [1, (2, 3)], "b": {"c": None}},
            {1, 2, 3},
            frozenset({"x"}),
            [],
            {},
        ],
    )
    def test_json_shapes_keep_their_types(self, value: Any) -> None:
        decoded = _round_trip(value)
        assert decoded == value
        assert type(decoded) is type(value)

    def test_nested_tuple_stays_a_tuple(self) -> None:
        assert _round_trip({"k": [(1, 2)]}) == {"k": [(1, 2)]}
        assert type(_round_trip({"k": [(1, 2)]})["k"][0]) is tuple

    @pytest.mark.parametrize(
        "value",
        [
            Point(1, 2),
            Outcome("r1", [Point(0, 0)]),
            Color.RED,
            Level.LOW,
            Model(name="m", score=0.5),
            [Point(1, 2), Model(name="n", score=1.0)],
        ],
    )
    def test_converter_types_round_trip(self, value: Any) -> None:
        decoded = _round_trip(value)
        assert decoded == value
        assert type(decoded) is type(value)

    def test_int_enum_is_not_flattened_to_int(self) -> None:
        assert encode_value(Level.LOW) != 1
        assert _round_trip(Level.LOW) is Level.LOW


class TestRefusal:
    def test_local_class_is_refused(self) -> None:
        @dataclass
        class Local:
            a: int

        with pytest.raises(RecordError, match="inside a function"):
            storable(Local(1))

    def test_non_str_dict_key_is_refused(self) -> None:
        with pytest.raises(RecordError, match="not a str"):
            storable({1: "a"})

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_float_is_refused(self, value: float) -> None:
        with pytest.raises(RecordError, match="non-finite"):
            storable(value)

    def test_any_field_that_changes_type_is_refused(self) -> None:
        """A pydantic ``Any`` field holding a model decodes as a dict — not exact."""
        with pytest.raises(RecordError, match="does not round-trip"):
            storable(Loose(parsed=Model(name="m", score=1.0)))

    def test_any_field_holding_json_round_trips(self) -> None:
        assert _round_trip(Loose(parsed={"a": 1})) == Loose(parsed={"a": 1})

    def test_exception_is_refused(self) -> None:
        with pytest.raises(RecordError):
            storable(ValueError("boom"))

    def test_callable_is_refused(self) -> None:
        with pytest.raises(RecordError):
            storable(len)


class TestMalformed:
    @pytest.mark.parametrize(
        "encoded",
        [
            {"$o": "tests.unit.flow.test_record:Point"},
            {"$o": "tests.unit.flow.test_record:Missing", "v": {}},
            {"$l": 5},
            {"$l": "abc"},
            {"$l": {"k": 1}},
            {"$d": [1]},
            {"$zz": []},
            {"$o": "tests.unit.flow.test_record:Point", "v": {"x": 1, "y": 2}, "extra": 1},
            {"a": 1, "b": 2},
            [1, 2],
            {"$l": [[1]]},
        ],
    )
    def test_malformed_value_raises_record_error(self, encoded: Any) -> None:
        with pytest.raises(RecordError, match="malformed"):
            decode_value(encoded)

    def test_decoding_never_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stored type path in a module not yet imported is refused, not imported."""
        module = "wsgiref.simple_server"
        monkeypatch.delitem(sys.modules, module, raising=False)
        with pytest.raises(RecordError, match="not imported"):
            decode_value({"$o": f"{module}:WSGIServer", "v": {}})
        assert module not in sys.modules


class TestValueHash:
    def test_equal_values_hash_equal(self) -> None:
        assert value_hash({"a": 1, "b": [2]}) == value_hash({"b": [2], "a": 1})

    def test_set_hash_is_order_independent(self) -> None:
        assert value_hash({"b", "a", "c"}) == value_hash({"c", "a", "b"})

    def test_set_in_object_encodes_sorted(self) -> None:
        """Converter output for a nested set is sorted, not in per-process hash order."""
        encoded = encode_value(Tagged(frozenset({"delta", "alpha", "gamma", "beta"})))
        assert encoded["v"]["tags"] == ["alpha", "beta", "delta", "gamma"]

    def test_container_types_hash_differently(self) -> None:
        assert value_hash([1, 2]) != value_hash((1, 2))
        assert value_hash(1) != value_hash(Level.LOW)

    def test_value_without_encoding_raises(self) -> None:
        with pytest.raises(RecordError):
            value_hash({1: "a"})

    def test_lossy_value_has_no_hash(self) -> None:
        """A model in an ``Any`` field dumps like the equal-looking dict; it gets no hash."""
        with pytest.raises(RecordError, match="does not round-trip"):
            value_hash(Loose(parsed=Model(name="m", score=1.0)))
        with pytest.raises(RecordError, match="does not round-trip"):
            key_coord(Loose(parsed=Model(name="m", score=1.0)))
        assert value_hash(Loose(parsed={"name": "m", "score": 1.0}))


class TestAddresses:
    def test_top_level_address_is_the_node_id(self) -> None:
        assert instance_address("abc", ()) == "abc"

    def test_coords_join_outermost_first(self) -> None:
        coords = (iteration_coord(2), index_coord(0))
        assert instance_address("abc", coords) == "abc@i2/n0"

    def test_key_coord_depends_on_the_key_only(self) -> None:
        assert key_coord("q1") == key_coord("q1")
        assert key_coord("q1") != key_coord("q2")
        assert key_coord(("run", 1)) != key_coord(["run", 1])


class TestExecutionRecord:
    def test_shards_round_trip(self) -> None:
        record = ExecutionRecord({})
        for i in range(50):
            record.put(f"step:{i}", {"out": i})
        rebuilt = ExecutionRecord.from_shards(record.shards().values())
        assert len(rebuilt) == 50
        assert rebuilt.get("step:7") == {"out": 7}
        assert "step:49" in rebuilt and "step:50" not in rebuilt

    def test_shard_ids_are_hash_prefixes(self) -> None:
        record = ExecutionRecord({f"k{i}": i for i in range(1000)})
        shards = record.shards()
        assert 1 < len(shards) <= 256
        assert all(len(s) == 2 for s in shards)

    def test_unchanged_shards_keep_their_bytes(self) -> None:
        """Adding one entry rewrites one shard; the others are byte-identical (CAS dedupe)."""
        record = ExecutionRecord({f"k{i}": i for i in range(100)})
        before = record.shards()
        record.put("new-entry", 1)
        after = record.shards()
        changed = [s for s in after if before.get(s) != after[s]]
        assert len(changed) == 1

    @pytest.mark.parametrize(
        "payload",
        [b"[]", b'[["k", 1]]', b"not json", b"\xff\xfe", b"42"],
        ids=["empty-array", "pairs", "invalid-json", "invalid-utf8", "scalar"],
    )
    def test_non_object_shard_is_refused(self, payload: bytes) -> None:
        with pytest.raises(RecordError, match="record shard"):
            ExecutionRecord.from_shards([b'{"a": 1}', payload])

    def test_key_in_two_shards_is_refused(self) -> None:
        with pytest.raises(RecordError, match="more than one shard"):
            ExecutionRecord.from_shards([b'{"a": 1}', b'{"a": 2}'])
