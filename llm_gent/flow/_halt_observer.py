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

A run has one halt, set on its top-level flow (:func:`check_one_halt`).
:func:`note_halt` records where it was first observed, for that commit's
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .flow import Flow
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


def check_one_halt(root: Flow) -> None:
    """Raise when a flow inside ``root``'s run sets a halt other than ``root``'s.

    The halt pauses the whole run, so a run has one, on its top-level
    flow. A nested flow carrying the same event (as every flow a
    ``FlowFactory(halt=...)`` builds does) sets no other halt.

    Raises:
        RuntimeError: A nested flow's ``with_halt`` event is not the
            top-level flow's.
    """
    from ._node_id import iter_flows

    for flow in iter_flows(root)[1:]:
        if flow._halt_event is not None and flow._halt_event is not root._halt_event:
            raise RuntimeError(
                f"Flow {_label(flow)} sets a halt but runs inside {_label(root)}: a run "
                f"has one halt, set on its top-level flow with with_halt(event)"
            )


def _label(flow: Flow) -> str:
    return repr(flow._name or "<anonymous>")


def is_halt_signaled(env: _RunEnv) -> bool:
    """True when the halt event in effect under ``env`` is set, or the run's halt is.

    Under a capped run or a shortcut, ``env.halt`` is a stop event that
    follows the run's halt one loop tick later; the run's halt is checked
    directly so it is never missed in between.
    """
    return (env.halt is not None and env.halt.is_set()) or is_run_halted(env)


def is_run_halted(env: _RunEnv) -> bool:
    """True when the run's halt (the top-level flow's) is set."""
    halt = env.runtime._halt_event
    return halt is not None and halt.is_set()


def note_halt(env: _RunEnv, iteration: int, node_id: str) -> None:
    """Record where the run's halt was observed, unless an earlier observation was recorded.

    Only the run's halt is recorded: a capped run stopped by its own
    budget, with the run's halt not set, ends that run without a halt
    checkpoint.
    """
    runtime = env.runtime
    halt = runtime._halt_event
    if halt is not None and halt.is_set() and runtime._halt_at is None:
        runtime._halt_at = HaltPoint(env.ancestor_chain, iteration, node_id, env.state)
