# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Framework state for halt anchors — cut-short steps and resumed-step inputs.

The chain's halt-observation sites read both when writing a halt
commit: a step the halt cut short is where the commit anchors, so
resume re-runs it, and the top-level chain step the commit lands on
gets back the input it was handed, so it re-runs with the same
arguments instead of none.

Not re-exported from :mod:`llm_gent.flow.state` — internal callers
import from :mod:`llm_gent.flow.state.halt_anchor` explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .cas import Blob, Commit, TraceRef, canonical_json


if TYPE_CHECKING:
    from .._checkpoint_ctx import CheckpointContext


STEP_INPUT_KIND = "step_input"
""":class:`TraceRef` kind carrying a resumed chain step's input blob."""


@dataclass
class CutShortSteps:
    """Nodes whose work was cut short during the current run.

    Marked by a map that skipped items on halt, a :class:`Loop` whose
    SAIA turn paused, and verbs via :meth:`Context.mark_cut_short`.
    Each entry keeps the node's ancestor chain so :meth:`owns` matches
    a chain step that is the node itself or any ancestor of it (a map
    or Loop nested inside a subflow or iterate body).
    """

    _ancestors: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def mark(self, node_id: str, ancestors: tuple[str, ...]) -> None:
        """Record ``node_id``, reached through ``ancestors``, as cut short."""
        self._ancestors[node_id] = ancestors

    def owns(self, node_id: str) -> bool:
        """True when ``node_id`` or a node beneath it was cut short."""
        if node_id in self._ancestors:
            return True
        return any(node_id in chain for chain in self._ancestors.values())

    def clear(self) -> None:
        """Reset. Called at the top of :meth:`Flow.run`."""
        self._ancestors.clear()


@dataclass
class StepInputs:
    """The input the top-level chain handed each step this run.

    A step resumed at chain index > 0 has no predecessor result to
    receive, so the halt commit carries the resumed step's input as a
    ``step_input`` :class:`TraceRef`. Only values that survive a JSON
    round trip unchanged are storable; anything else (tuples, objects,
    non-string keys) would come back as a different value.
    """

    _values: dict[str, Any] = field(default_factory=dict)

    def record(self, step_id: str, value: Any) -> None:
        """Remember the input handed to the step at ``step_id``."""
        self._values[step_id] = value

    def storable(self, step_id: str) -> bool:
        """True when ``step_id`` has a recorded input that round-trips through JSON."""
        return step_id in self._values and _round_trips(self._values[step_id])

    def clear(self) -> None:
        """Reset. Called at the top of :meth:`Flow.run`."""
        self._values.clear()

    async def stash_to_ctx(self, ctx: CheckpointContext, step_id: str) -> tuple[TraceRef, ...]:
        """Persist ``step_id``'s input as a blob; return its :class:`TraceRef`, or ``()``.

        The ref id is ``f"{step_id}:{blob_hash}"`` so the resume side
        can check the input belongs to the step it resumes at.
        """
        if not self.storable(step_id):
            return ()
        blob = Blob.from_bytes(canonical_json(self._values[step_id]))
        await ctx.put_blob(blob.content_hash, blob.payload)
        return (TraceRef(kind=STEP_INPUT_KIND, id=f"{step_id}:{blob.content_hash}"),)


async def load_step_input(ctx: CheckpointContext, commit: Commit) -> tuple[str, Any] | None:
    """Return the ``(step_id, input)`` stamped on ``commit``, or ``None``.

    A missing blob or a malformed ref id yields ``None``: the resumed
    step then runs with no input, as it would without the ref.
    """
    for ref in commit.meta.trace_ref:
        if ref.kind != STEP_INPUT_KIND:
            continue
        step_id, sep, blob_hash = ref.id.partition(":")
        if not sep or not step_id or not blob_hash:
            continue
        payload = await ctx.get_object("blob", blob_hash)
        if payload is None:
            continue
        return step_id, json.loads(payload)
    return None


def _round_trips(value: Any) -> bool:
    """True when ``value`` encodes to canonical JSON and decodes back equal."""
    try:
        return bool(json.loads(canonical_json(value)) == value)
    except (TypeError, ValueError):
        return False
