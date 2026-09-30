# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Halt observation: detection, and where the run's halt was first observed.

A halt stops the run at the next place the framework looks: a chain
after each step, an iterate before each pass, a map item before it
starts. An LLM call in flight finishes (or its SAIA turn pauses) and
its step ends. A structure that stops leaves its position registered
and raises :class:`~llm_gent.flow.Interrupted` to the step running it;
once everything has stopped, :meth:`Flow.run` writes the run's one halt
checkpoint from those positions (:func:`~llm_gent.flow._resume.commit_halt`).

:func:`note_halt` records where the run's halt was first observed, for
that commit's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .nodes import _RunEnv
    from .state import State


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


def is_halt_signaled(env: _RunEnv) -> bool:
    """True when the halt event in effect under ``env`` is set."""
    return env.halt is not None and env.halt.is_set()


def note_halt(env: _RunEnv, iteration: int, node_id: str) -> None:
    """Record where the run's halt was observed, unless an earlier observation was recorded.

    A subflow's own ``.with_halt`` is not the run's halt: it stops that
    subtree, which then ends the interruption without a halt checkpoint.
    """
    runtime = env.runtime
    if env.halt is runtime._halt_event and runtime._halt_at is None:
        runtime._halt_at = HaltPoint(env.ancestor_chain, iteration, node_id, env.state)
