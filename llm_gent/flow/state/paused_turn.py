# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Paused turns — SAIA turns a Loop stopped mid-turn, kept in the run's snapshots.

A Loop call whose ``saia.complete`` paused registers a :class:`PausedTurn`
cursor at its path (``<step>/t/<k>``, see
:meth:`~llm_gent.flow.state.snapshot.ScopeRegistry.next_turn`); every
checkpoint taken while it is registered stores the turn's
:class:`PausedTurnEnvelope` there. On resume the step runs again, and the
same Loop call takes the envelope back from the same path.

Not re-exported from :mod:`llm_gent.flow.state` — internal callers
import from :mod:`llm_gent.flow.state.paused_turn` explicitly.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from . import codec
from .cas import canonical_json
from .snapshot import TURN


@dataclass(frozen=True)
class PausedTurnEnvelope:
    """Canonical ``{"task", "conversation"}`` envelope for a paused SAIA turn.

    ``conversation`` is the ``to_dict()`` payload of the paused
    ``SerializableConversationLike``; rebuilding a ``Conversation`` from
    it is the :class:`ConversationFactory`'s job, not this envelope's.
    ``result`` is what ``saia.complete`` returned when it paused the turn:
    a call that finishes the turn without continuing it (a shortcut)
    returns it. It is stored when :mod:`.codec` can store it (SAIA's
    ``TaskResult`` can), else left out (``None``).
    """

    task: str
    conversation: dict[str, Any]
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        """The envelope as a JSON-compatible dict; ``result`` encoded, when it can be."""
        data = {"task": self.task, "conversation": self.conversation}
        if self.result is not None:
            with contextlib.suppress(TypeError):
                data["result"] = codec.encode(self.result, "paused turn result")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PausedTurnEnvelope:
        """Inverse of :meth:`to_dict`; raises on missing keys."""
        raw = data.get("result")
        result = None if raw is None else codec.decode(raw, "paused turn result")
        return cls(task=data["task"], conversation=data["conversation"], result=result)

    def to_bytes(self) -> bytes:
        """Canonical JSON bytes of the task and conversation."""
        return canonical_json({"task": self.task, "conversation": self.conversation})


class PausedTurn:
    """Cursor of a Loop call whose turn is paused, or is resuming a paused turn.

    ``envelope`` is ``None`` when the paused turn could not be captured (no
    :class:`ConversationFactory`, or a conversation without ``to_dict``):
    the step still counts as interrupted, and its rerun starts the turn
    over.
    """

    def __init__(self, envelope: PausedTurnEnvelope | None) -> None:
        self.envelope = envelope

    def cursor(self) -> dict[str, Any]:
        """``{"turn": <envelope dict or None>}``."""
        return {TURN: None if self.envelope is None else self.envelope.to_dict()}
