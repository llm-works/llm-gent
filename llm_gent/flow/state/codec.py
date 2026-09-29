# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Codec for values a checkpoint holds besides scopes: step inputs, carried values.

Such a value must come back on resume as the value it was, so three forms
are accepted:

- plain JSON: ``dict`` with ``str`` keys, ``list``, ``str``, ``int``,
  ``float``, ``bool``, ``None`` — stored as is;
- a pydantic model — stored as ``model_dump(mode="json")``, rebuilt with
  ``model_validate``;
- an object with ``to_dict()`` and a classmethod ``from_dict()`` (the
  :class:`~llm_gent.flow.state.StateData` pattern) — rebuilt with
  ``from_dict``.

A typed value is stored as ``{"$type": "<module>:<qualname>", "$data": ...}``;
a plain dict that has a ``"$type"`` key of its own is wrapped the same way
with type ``"dict"`` so it is never mistaken for one. Anything else raises
:class:`TypeError` naming the value's path. On decode a class is found only
in modules already imported, and used only when it is a pydantic model or
has ``from_dict``: a stored type name cannot make the codec import or call
anything else.
"""

from __future__ import annotations

import math
import sys
from typing import Any

from pydantic import BaseModel


TYPE = "$type"
DATA = "$data"
_DICT = "dict"


def encode(value: Any, where: str) -> Any:
    """Return the JSON form of ``value``; ``where`` names it in errors.

    Raises:
        TypeError: ``value`` (or something inside it) is none of the three
            accepted forms.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{where}: {value!r} has no JSON form")
        return value
    if isinstance(value, list):
        return [encode(item, where) for item in value]
    if isinstance(value, dict):
        return _encode_dict(value, where)
    if isinstance(value, BaseModel):
        return _typed(value, value.model_dump(mode="json"), where)
    if callable(getattr(value, "to_dict", None)) and callable(
        getattr(type(value), "from_dict", None)
    ):
        return _typed(value, encode(value.to_dict(), where), where)
    raise TypeError(
        f"{where}: a value of type {type(value).__name__} cannot be checkpointed; use plain "
        f"JSON, a pydantic model, or an object with to_dict() and a classmethod from_dict()"
    )


def decode(data: Any, where: str) -> Any:
    """Rebuild the value :func:`encode` turned into ``data``.

    Raises:
        TypeError: A stored type cannot be loaded, is not a pydantic model and
            has no ``from_dict``, or its data no longer fits it.
    """
    if isinstance(data, list):
        return [decode(item, where) for item in data]
    if not isinstance(data, dict):
        return data
    if TYPE not in data:
        return {k: decode(v, where) for k, v in data.items()}
    if data[TYPE] == _DICT:
        return {k: decode(v, where) for k, v in data[DATA].items()}
    cls = _load(data[TYPE], where)
    try:
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            return cls.model_validate(data[DATA])
        return cls.from_dict(decode(data[DATA], where))
    except Exception as e:
        raise TypeError(f"{where}: stored {data[TYPE]} cannot be rebuilt: {e}") from e


def _encode_dict(value: dict[Any, Any], where: str) -> Any:
    """A plain dict, wrapped when it has a ``"$type"`` key of its own."""
    if not all(isinstance(k, str) for k in value):
        raise TypeError(f"{where}: a dict with non-str keys cannot be checkpointed")
    encoded = {k: encode(v, where) for k, v in value.items()}
    return {TYPE: _DICT, DATA: encoded} if TYPE in value else encoded


def _typed(value: Any, data: Any, where: str) -> dict[str, Any]:
    """``{"$type": ..., "$data": data}`` for ``value``'s class, which must be importable."""
    cls = type(value)
    if "<locals>" in cls.__qualname__:
        raise TypeError(
            f"{where}: {cls.__qualname__} is defined inside a function and cannot be "
            f"loaded on resume; define it at module level"
        )
    return {TYPE: f"{cls.__module__}:{cls.__qualname__}", DATA: data}


def _load(name: str, where: str) -> Any:
    """The class stored as ``name``; only a pydantic model or a class with ``from_dict``.

    Looked up in modules the process has already imported, never imported
    here: a stored name must not be able to run a module's import-time code.
    The flow that resumes has imported the types its steps use.
    """
    module_name, _, qualname = name.partition(":")
    obj: Any = sys.modules.get(module_name)
    if obj is None:
        raise TypeError(f"{where}: stored type {name} is in a module that is not imported")
    try:
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except AttributeError as e:
        raise TypeError(f"{where}: stored type {name} cannot be loaded: {e}") from e
    is_model = isinstance(obj, type) and issubclass(obj, BaseModel)
    if not is_model and not callable(getattr(obj, "from_dict", None)):
        raise TypeError(f"{where}: stored type {name} is not a pydantic model and has no from_dict")
    return obj
