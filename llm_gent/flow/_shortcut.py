# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""What :meth:`Flow.with_shortcut` declares: the signal that cuts a flow short, and how.

How a cut region runs is :mod:`llm_gent.flow.run._shortcut`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shortcut:
    """What :meth:`Flow.with_shortcut` declares: the signal that cuts the flow short, and how."""

    signal: str
    drain: bool
    """Started map items run to their end instead of fast-forwarding."""


def check_shortcut(signal: object, drain: object) -> Shortcut:
    """The :class:`Shortcut` :meth:`Flow.with_shortcut` declares, when its arguments are valid.

    Raises:
        ValueError: ``signal`` is not a non-empty str.
        TypeError: ``drain`` is not a bool.
    """
    _check_name(signal, ".with_shortcut(signal)")
    if not isinstance(drain, bool):
        raise TypeError(f"with_shortcut(drain=) takes a bool; got {type(drain).__name__}")
    return Shortcut(signal, drain)  # type: ignore[arg-type]


def _check_name(name: object, what: str) -> None:
    """Raise ``ValueError`` naming ``what`` unless ``name`` is a non-empty str."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} must be a non-empty str; got {name!r}")
