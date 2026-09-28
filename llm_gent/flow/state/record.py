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
from collections.abc import Iterable
from typing import Any

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
    """Recursive body of :func:`decode_value`."""
    if not isinstance(encoded, dict):
        return encoded
    if len(encoded) == 1:
        ((tag, body),) = encoded.items()
        if tag in (_LIST, _TUPLE):
            items = [_decode(v) for v in body]
            return items if tag == _LIST else tuple(items)
        if tag in (_SET, _FROZENSET):
            members = (_decode(v) for v in body)
            return set(members) if tag == _SET else frozenset(members)
        if tag == _DICT:
            return {k: _decode(v) for k, v in body.items()}
    if _OBJECT in encoded:
        return state_converter.structure(encoded["v"], _load_type(encoded[_OBJECT]))
    raise RecordError(f"malformed record value: {sorted(encoded)}")


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
    """Content hash of ``value``'s encoding — how the record compares inputs.

    Needs an encoding, not a round trip. Raises :class:`RecordError` when
    ``value`` has none. Sets hash order-independently, including sets
    nested in objects the state converter unstructures. Exception: a set
    field of a pydantic model is dumped by pydantic in iteration order,
    so its hash can differ across processes (a miss, never a false match).
    """
    try:
        return content_hash(canonical_json(encode_value(value)))
    except RecordError:
        raise
    except (TypeError, ValueError) as e:
        raise RecordError(f"{type(value).__qualname__} has no JSON encoding: {e}") from e


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


# --- the record --------------------------------------------------------------


_SHARD_PREFIX_LEN = 2
"""Hex chars of the key hash that pick an entry's shard — 256 shards."""


class ExecutionRecord:
    """Entries for completed node instances and control decisions, keyed by address.

    Entries are JSON values; their shape belongs to the executor that
    writes them. Not safe for concurrent writers across threads — the
    executor writes from one event loop.
    """

    def __init__(self, entries: dict[str, Any]) -> None:
        self._entries: dict[str, Any] = dict(entries)

    def get(self, key: str) -> Any | None:
        """The entry at ``key``, or ``None``."""
        return self._entries.get(key)

    def put(self, key: str, entry: Any) -> None:
        """Set the entry at ``key``."""
        self._entries[key] = entry

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def shards(self) -> dict[str, bytes]:
        """Canonical JSON bytes per shard id, one shard per populated key-hash prefix."""
        grouped: dict[str, dict[str, Any]] = {}
        for key, entry in self._entries.items():
            shard = content_hash(key.encode("utf-8"))[:_SHARD_PREFIX_LEN]
            grouped.setdefault(shard, {})[key] = entry
        return {shard: canonical_json(entries) for shard, entries in grouped.items()}

    @classmethod
    def from_shards(cls, payloads: Iterable[bytes]) -> ExecutionRecord:
        """Rebuild a record from :meth:`shards` payloads."""
        entries: dict[str, Any] = {}
        for payload in payloads:
            entries.update(json.loads(payload.decode("utf-8")))
        return cls(entries)
