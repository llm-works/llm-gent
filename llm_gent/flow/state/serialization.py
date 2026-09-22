# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""JSON-native cattrs converter backing :class:`StateDataclass`.

Extracted from :mod:`llm_gent.flow.state` at the state-cas package split
so :mod:`.base` (State + protocols) can stay focused on the scope-tree
runtime shape and :mod:`.cas` (content-addressed object model) can rest
on a small, stable serialization surface.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import PurePath
from typing import Any
from uuid import UUID

from cattrs import Converter
from cattrs.preconf.json import make_converter as _make_json_converter
from pydantic import BaseModel


def _is_basemodel_class(cls: Any) -> bool:
    """Predicate for the cattrs BaseModel hook factory.

    Guards against non-class arguments (generic aliases, TypeVars) that
    :func:`issubclass` would reject with :class:`TypeError`.
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
    # UUID, Decimal, and pathlib types are not in the JSON preconf's default
    # hooks (verified through cattrs 26.x). Encode as strings; decode via the
    # class constructor. Decimal round-trips through str exactly (float would
    # lose precision).
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
