# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Execution record — what a run has completed, keyed by instance address.

A node instance is one execution of a composition node: a chain step,
one pass of an iterate body, one map item. Its address is the node's
structural id plus the dynamic coordinates of every iterate pass and map
item enclosing it (:func:`instance_address`). The record maps addresses
(and control decisions made at them) to JSON entries; a resumed run
returns a recorded instance's stored output instead of executing it.

Values the record stores — outputs a later node receives, map items —
follow the state serialization contract: JSON primitives, ``list`` /
``tuple`` / ``set`` / ``frozenset`` / ``dict`` with ``str`` keys, and
any type :data:`~llm_gent.flow.state.state_converter` round-trips
(dataclasses, attrs, enums, pydantic models, ...). A value is storable
when it decodes back equal and of the same type (:func:`storable`);
anything else raises :class:`RecordError`. Object types are recorded by
import path, so they must be module-level classes, and decoding finds
them only in modules the process has already imported.

The record is persisted as up to 256 shards (:meth:`ExecutionRecord.shards`),
grouped by a hash of the entry key, so a commit rewrites only the shards
whose entries changed and unchanged shards dedupe in the CAS.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from .cas import canonical_json, content_hash
from .serialization import state_converter


class RecordError(TypeError):
    """A value the execution record must store does not round-trip."""


_LIST, _TUPLE, _SET, _FROZENSET, _DICT, _OBJECT = "$l", "$t", "$s", "$fs", "$d", "$o"
_PRIMITIVES = (type(None), bool, int, str)


def encode_value(value: Any) -> Any:
    """Return ``value`` as tagged JSON; raise :class:`RecordError` when it has no encoding.

    Primitives pass through; containers and objects become one-key
    dicts tagged by kind (``$l`` list, ``$t`` tuple, ``$s`` / ``$fs``
    set / frozenset, ``$d`` dict, ``$o`` object), so decoding restores
    the exact container types. Exact type checks: subclasses of the
    primitives and containers (``IntEnum``, ``OrderedDict``,
    ``NamedTuple``) take the object path and keep their type.
    """
    kind = type(value)
    if kind in _PRIMITIVES:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise RecordError(f"non-finite float {value!r} has no JSON encoding")
        return value
    if kind is list or kind is tuple:
        return {_LIST if kind is list else _TUPLE: [encode_value(v) for v in value]}
    if kind is set or kind is frozenset:
        return {_SET if kind is set else _FROZENSET: _encode_members(value)}
    if kind is dict:
        return {_DICT: _encode_dict(value)}
    return _encode_object(value)


def _encode_members(members: Iterable[Any]) -> list[Any]:
    """Encode set members in a canonical order, so equal sets encode (and hash) equally."""
    encoded = [encode_value(m) for m in members]
    return sorted(encoded, key=canonical_json)


def _encode_dict(value: dict[Any, Any]) -> dict[str, Any]:
    """Encode a plain dict; keys must be strings (JSON object keys)."""
    for key in value:
        if type(key) is not str:
            raise RecordError(f"dict key {key!r} is not a str; JSON object keys must be strings")
    return {k: encode_value(v) for k, v in value.items()}


def _encode_object(value: Any) -> dict[str, Any]:
    """Encode an object through the state converter, tagged with its type's import path."""
    path = _type_path(type(value))
    try:
        raw = state_converter.unstructure(value)
    except Exception as e:
        raise RecordError(f"cannot serialize {path}: {e}") from e
    return {_OBJECT: path, "v": raw}


def _type_path(cls: type) -> str:
    """``module:qualname`` for a class decoding can import; raise for local classes."""
    qualname = cls.__qualname__
    if "<locals>" in qualname:
        raise RecordError(
            f"{cls.__module__}.{qualname} is defined inside a function; not importable"
        )
    return f"{cls.__module__}:{qualname}"


def decode_value(encoded: Any) -> Any:
    """Inverse of :func:`encode_value` over the parsed JSON form.

    Raises :class:`RecordError` for anything :func:`encode_value` cannot
    have produced, including an object whose recorded type no longer
    imports or no longer structures from the stored fields.
    """
    try:
        return _decode(encoded)
    except RecordError:
        raise
    except Exception as e:
        raise RecordError(f"malformed record value: {e!r}") from e


def _decode(encoded: Any) -> Any:
    """Recursive body of :func:`decode_value`: untagged scalars, tagged everything else."""
    kind = type(encoded)
    if kind in _PRIMITIVES:
        return encoded
    if kind is float:
        if not math.isfinite(encoded):
            raise RecordError(f"malformed record value: non-finite float {encoded!r}")
        return encoded
    if kind is not dict:
        raise RecordError(f"malformed record value: untagged {kind.__name__}")
    if encoded.keys() == {_OBJECT, "v"}:
        return state_converter.structure(encoded["v"], _load_type(encoded[_OBJECT]))
    if len(encoded) == 1:
        ((tag, body),) = encoded.items()
        return _decode_tagged(tag, body)
    raise RecordError(f"malformed record value: keys {sorted(encoded)}")


_SEQUENCE_TAGS: dict[str, Callable[[Iterable[Any]], Any]] = {
    _LIST: list,
    _TUPLE: tuple,
    _SET: set,
    _FROZENSET: frozenset,
}
"""Tag → constructor for the containers encoded as a JSON array body."""


def _decode_tagged(tag: str, body: Any) -> Any:
    """Decode a one-key tagged container; the body must have the tag's JSON shape."""
    if tag == _DICT and type(body) is dict:
        return {k: _decode(v) for k, v in body.items()}
    build = _SEQUENCE_TAGS.get(tag)
    if build is not None and type(body) is list:
        return build(_decode(v) for v in body)
    raise RecordError(f"malformed record value: tag {tag!r} with a {type(body).__name__} body")


def _load_type(path: str) -> Any:
    """The class at ``module:qualname``, from a module this process has already imported.

    Decoding never imports: a path read from the store cannot pull in a
    module and run its import-time code. The flow reading the record
    imports its own output types before it runs.
    """
    module_name, _, qualname = path.partition(":")
    target: Any = sys.modules.get(module_name)
    if target is None:
        raise RecordError(f"{path}: module {module_name!r} is not imported; decoding never imports")
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def storable(value: Any) -> Any:
    """Return ``value``'s stored (JSON) form; raise :class:`RecordError` unless it round-trips.

    Round-trip means the decoded value equals ``value`` and has its
    exact type — a pydantic model whose ``Any`` field comes back as a
    dict, or a class whose converter hooks drop a field, is refused.
    Below the top level the check is equality only: an untyped field
    holding a value equal to its JSON form (an ``IntEnum`` member in an
    ``Any`` field decodes as ``int``) passes and comes back as that form.
    """
    encoded = encode_value(value)
    try:
        payload = canonical_json(encoded)
        decoded = decode_value(json.loads(payload))
        same = type(decoded) is type(value) and bool(decoded == value)
    except RecordError:
        raise
    except Exception as e:
        raise RecordError(f"{type(value).__qualname__} does not round-trip: {e}") from e
    if not same:
        raise RecordError(f"{type(value).__qualname__} does not round-trip through the record")
    return encoded


def value_hash(value: Any) -> str:
    """Content hash of ``value``'s stored form — how the record compares inputs and map keys.

    Only a value :func:`storable` accepts has a hash; anything else raises
    :class:`RecordError`. A lossy encoding would give unequal values one
    hash (a model in an ``Any`` field dumps like the equivalent dict), and
    a matching input hash must mean the same input. Sets hash
    order-independently, including sets nested in objects the state
    converter unstructures. Exception: a set field of a pydantic model is
    dumped by pydantic in iteration order, so its hash can differ across
    processes (a miss, never a false match).
    """
    return content_hash(canonical_json(storable(value)))


# --- addresses ---------------------------------------------------------------


def instance_address(node_id: str, coords: tuple[str, ...]) -> str:
    """Address of one execution of ``node_id`` under the enclosing dynamic ``coords``.

    ``coords`` holds one coordinate per enclosing iterate pass or map
    item, outermost first. The node id already fixes which constructs
    enclose the node, so the pair is unique across the run.
    """
    return node_id if not coords else f"{node_id}@{'/'.join(coords)}"


def iteration_coord(iteration: int) -> str:
    """Coordinate of the ``iteration``-th pass of an iterate body (0-based)."""
    return f"i{iteration}"


def index_coord(index: int) -> str:
    """Coordinate of the map item at ``index`` (unkeyed maps)."""
    return f"n{index}"


def key_coord(key: Any) -> str:
    """Coordinate of the map item whose app key is ``key`` (keyed maps)."""
    return "k" + value_hash(key)[:32]


# --- entry keys --------------------------------------------------------------


RecordKind = Literal["s", "p", "i", "m", "b"]
"""Kind of a record entry: ``s`` chain step, ``p`` iterate pass, ``i`` a map's item
list, ``m`` one map item, ``b`` a branch verdict."""

STEP: RecordKind = "s"
PASS: RecordKind = "p"
ITEMS: RecordKind = "i"
ITEM: RecordKind = "m"
BRANCH: RecordKind = "b"

_KINDS: frozenset[str] = frozenset({STEP, PASS, ITEMS, ITEM, BRANCH})


def record_key(kind: RecordKind, address: str) -> str:
    """Key of the entry of ``kind`` at ``address``: ``"<kind>|<address>"``."""
    return f"{kind}|{address}"


def parse_record_key(key: str) -> tuple[RecordKind, str]:
    """``(kind, address)`` of a :func:`record_key`; :class:`RecordError` for anything else."""
    kind, sep, address = key.partition("|")
    if not sep or kind not in _KINDS or not address:
        raise RecordError(f"malformed record key: {key!r}")
    return cast(RecordKind, kind), address


# --- the record --------------------------------------------------------------


_SHARD_PREFIX_LEN = 2
"""Hex chars of the key hash that pick an entry's shard — 256 shards."""


class ExecutionRecord:
    """Entries for completed node instances and control decisions, keyed by address.

    Entries are JSON values; their shape belongs to the executor that
    writes them. Not safe for concurrent writers across threads — the
    executor writes from one event loop.

    :meth:`shards` serializes only the shards whose entries changed since
    the previous call and returns cached bytes for the rest, so a run
    that commits often pays for what it recorded since the last commit,
    not for the whole record.
    """

    def __init__(self, entries: dict[str, Any]) -> None:
        self._shards: dict[str, dict[str, Any]] = {}
        for key, entry in entries.items():
            self._shards.setdefault(_shard_of(key), {})[key] = entry
        self._bytes: dict[str, bytes] = {}
        self._dirty: set[str] = set(self._shards)

    def get(self, key: str) -> Any | None:
        """The entry at ``key``, or ``None``."""
        return self._shards.get(_shard_of(key), {}).get(key)

    def put(self, key: str, entry: Any) -> None:
        """Set the entry at ``key``."""
        shard = _shard_of(key)
        self._shards.setdefault(shard, {})[key] = entry
        self._dirty.add(shard)

    def remove(self, key: str) -> None:
        """Drop the entry at ``key``, if any."""
        shard = _shard_of(key)
        entries = self._shards.get(shard)
        if entries is not None and key in entries:
            del entries[key]
            if not entries:
                del self._shards[shard]
                self._bytes.pop(shard, None)
                self._dirty.discard(shard)
            else:
                self._dirty.add(shard)

    def items(self) -> list[tuple[str, Any]]:
        """Every ``(key, entry)`` pair, sorted by key."""
        pairs = [pair for entries in self._shards.values() for pair in entries.items()]
        return sorted(pairs)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._shards.get(_shard_of(key), {})

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._shards.values())

    def shards(self) -> dict[str, bytes]:
        """Canonical JSON bytes per shard id, one shard per populated key-hash prefix."""
        for shard in self._dirty:
            self._bytes[shard] = canonical_json(self._shards[shard])
        self._dirty.clear()
        return dict(self._bytes)

    @classmethod
    def from_shards(cls, payloads: Iterable[bytes]) -> ExecutionRecord:
        """Rebuild a record from :meth:`shards` payloads.

        Raises :class:`RecordError` when a payload is not a JSON object, or
        holds a key an earlier payload already held — :meth:`shards` puts
        each key in exactly one shard.
        """
        entries: dict[str, Any] = {}
        for payload in payloads:
            shard = _parse_shard(payload)
            overlap = entries.keys() & shard.keys()
            if overlap:
                raise RecordError(f"record keys in more than one shard: {sorted(overlap)[:3]}")
            entries.update(shard)
        return cls(entries)


def _shard_of(key: str) -> str:
    """Shard id of ``key``: the leading hex chars of its content hash."""
    return content_hash(key.encode("utf-8"))[:_SHARD_PREFIX_LEN]


def _parse_shard(payload: bytes) -> dict[str, Any]:
    """One shard payload as a JSON object; raise :class:`RecordError` for any other shape.

    Non-finite numbers (``NaN``, ``Infinity``, literals that overflow to
    infinity) are refused anywhere in the payload: the record only ever
    writes finite JSON.
    """
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            parse_constant=_refuse_non_finite,
            parse_float=_finite_float,
        )
    except ValueError as e:  # JSONDecodeError and UnicodeDecodeError
        raise RecordError(f"record shard is not JSON: {e}") from e
    if type(parsed) is not dict:
        raise RecordError(f"record shard is a {type(parsed).__name__}, not a JSON object")
    return parsed


def _refuse_non_finite(literal: str) -> float:
    """``parse_constant`` hook: ``NaN`` / ``Infinity`` / ``-Infinity`` are not record values."""
    raise RecordError(f"record shard holds a non-finite number: {literal}")


def _finite_float(literal: str) -> float:
    """``parse_float`` hook: refuse float literals that overflow to infinity (``1e999``)."""
    value = float(literal)
    if not math.isfinite(value):
        raise RecordError(f"record shard holds a non-finite number: {literal}")
    return value
