# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Halt observation: detection, and where the run's halt was first observed.

A halt stops the run at the next place the framework looks: a chain
after each step, an iterate before each pass, a map item before it
starts. An LLM call in flight finishes (or its SAIA turn pauses) and
its step ends. A structure that stops leaves its position registered
and raises :class:`~llm_gent.flow.Interrupted` to the step running it;
once everything has stopped, :meth:`Flow.run` writes the run's one halt
checkpoint from those positions (:func:`~llm_gent.flow.run._resume.commit_halt`).

A run has one halt, given to :meth:`~llm_gent.flow.Flow.run` (``halt=``).
:func:`note_halt` records where it was first observed, for that commit's
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..state import State
    from .scope import _Frame


@dataclass(frozen=True)
class HaltPoint:
    """Where the run's halt was first observed: the halt commit's metadata.

    ``node_id`` is the step a chain stopped at (the interrupted step, or
    the next one after a completed step), the iterate or the map that
    stopped; ``ancestor_chain`` leads to it from the run's root, and
    ``iteration`` is the stopped iterate's pass (``0`` elsewhere).
    ``state`` is the scope it ran under — still registered when the halt
    commit is written, which records its path.
    """

    ancestor_chain: tuple[str, ...]
    iteration: int
    node_id: str
    state: State[Any]


def is_halt_signaled(frame: _Frame) -> bool:
    """True when the halt in effect under ``frame`` — the run's — is set.

    A cut is not a halt: a region its signal cuts short fast-forwards
    instead (:func:`~._shortcut.is_fast_forward`).
    """
    if frame.halt is not None and frame.halt.is_set():
        return True
    return is_run_halted(frame)


def is_run_halted(frame: _Frame) -> bool:
    """True when the run's halt is set."""
    halt = frame.run.halt_event
    return halt is not None and halt.is_set()


def note_halt(frame: _Frame, iteration: int, node_id: str) -> None:
    """Record where the run's halt was observed, unless an earlier observation was recorded."""
    run = frame.run
    halt = run.halt_event
    if halt is not None and halt.is_set() and run.halt_at is None:
        run.halt_at = HaltPoint(
            frame.position.ancestor_chain, iteration, node_id, frame.scope.state
        )
