# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Integration test: real :class:`llm_saia.SAIA` + gent Loop mid-turn pause/resume.

Exercises the full save-side (halt-observation stamps a saia_turn Blob +
TraceRef on the halt commit) and resume-side (framework loads the blob and
Loop dispatches SAIA with the reconstructed Conversation + ``resume=True``)
against an actual ``llm_saia.SAIA`` driven by a Backend that halts mid-turn
via ``PauseRequested`` and completes on the resumed dispatch.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from llm_saia import SAIA
from llm_saia.core.backend import Backend
from llm_saia.core.config import Config
from llm_saia.core.errors import PauseRequested
from llm_saia.core.logger import NullLogger
from llm_saia.core.types import ChatResponse, Message, ToolDef

from llm_gent.flow import Context, FlowFactory, Loop, Role, verb
from llm_gent.flow.stores import JsonFileCheckpointStore

from ...unit.flow.conftest import make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class _RecordingBackend(Backend):
    """First chat() halts mid-stream; subsequent chats return a final answer.

    Simulates a real inference backend that observes the abort_signal and
    raises PauseRequested to trigger SAIA's paused-TaskResult path.
    """

    def __init__(self, halt: asyncio.Event) -> None:
        self._halt = halt
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: dict[str, Any] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "response_schema": response_schema})
        # First non-structured call: set halt + raise PauseRequested so SAIA's
        # Complete verb returns paused. Structured-response calls (SAIA runs a
        # task-state classifier around each turn) are answered with a valid
        # JSON payload so they don't derail before the pause point.
        if response_schema is not None:
            return ChatResponse(
                content=json.dumps(
                    {"category": "in_progress", "confidence": 0.9, "reason": "test"}
                ),
                tool_calls=[],
                finish_reason="end_turn",
            )
        real_calls = [c for c in self.calls if c.get("response_schema") is None]
        if len(real_calls) == 1:
            self._halt.set()
            raise PauseRequested()
        # Resume call: return a final answer.
        return ChatResponse(content="final answer", tool_calls=[], finish_reason="end_turn")


class _SerializableConv:
    """Minimal SerializableConversationLike + to_dict/from_dict round-trip."""

    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages: list[Message] = list(messages or [])

    def append(self, msg: Message) -> None:
        self._messages.append(msg)

    def as_messages(self) -> list[Message]:
        return list(self._messages)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [m.to_dict() for m in self._messages]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _SerializableConv:
        return cls(messages=[Message.from_dict(m) for m in data.get("messages", [])])


class _ConvFactory:
    def create(self) -> _SerializableConv:
        return _SerializableConv()

    def create_from_state(self, state: dict[str, Any]) -> _SerializableConv:
        return _SerializableConv.from_dict(state)


class _SaiaFactory:
    """Framework SAIAFactory that hands the SAME real SAIA to every role bind."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def build(self, role: Role) -> SAIA:
        # Complete requires a tool + executor even though we halt before any
        # tool call fires. The executor is a no-op sentinel.
        async def _noop_executor(name: str, args: dict[str, Any]) -> str:
            return "unused"

        tools = [ToolDef(name="noop", description="unused in test", parameters={"type": "object"})]
        config = Config(
            lg=NullLogger(), backend=self._backend, tools=tools, executor=_noop_executor
        )
        return SAIA(config)


@pytest.fixture
def store(tmp_path: Path) -> JsonFileCheckpointStore:
    return JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")


async def test_real_saia_pause_resume_round_trip(store: JsonFileCheckpointStore) -> None:
    """Real SAIA halts mid-turn on run 1; run 2 with resume=True completes the turn."""
    role = Role(name="r", backend="openai", model="gpt-4o-mini")

    # ---- Run 1: SAIA's first chat() sets halt + raises PauseRequested.
    halt1 = asyncio.Event()
    backend1 = _RecordingBackend(halt1)
    loop = Loop(role, conversation_factory=_ConvFactory())

    @verb(role=role)
    async def run_loop(ctx: Context, _prev: Any = None) -> Any:
        return await loop(ctx, ctx.extra["task"], conversation=ctx.extra["conv"])

    body_ff = FlowFactory(make_test_logger())
    body = body_ff.create()
    body.call(run_loop)

    ff1 = FlowFactory(make_test_logger(), saia_factory=_SaiaFactory(backend1))
    flow1 = (
        ff1.create(state={})
        .with_checkpointer(store, "real-saia-resume")
        .with_halt(halt1)
        .iterate(body, max_iters=2)
    )
    await flow1.run(extra={"task": "please answer", "conv": _SerializableConv()})

    # SAIA's real complete() was called exactly once — it hit PauseRequested and
    # returned paused before the second turn could start.
    assert len(backend1.calls) == 1

    # The halt commit was written with a saia_turn TraceRef pointing at a blob
    # that encodes the paused task + conversation-state envelope.
    from llm_gent.flow.state.cas import Commit

    halted_hash = store.resolve_ref("real-saia-resume")
    assert halted_hash is not None
    commit = Commit.from_bytes(store.get_object("real-saia-resume", "commit", halted_hash) or b"")
    assert commit.meta.outcome == "halted"
    saia_refs = [r for r in commit.meta.trace_ref if r.kind == "saia_turn"]
    assert len(saia_refs) == 1

    # ---- Run 2: framework loads the envelope, Loop dispatches SAIA with
    # resume=True and the reconstructed Conversation. Backend2's halt is never
    # set (halt is caller-controlled and left clear for the resume), so it
    # takes the return-answer branch when SAIA calls chat().
    halt2 = asyncio.Event()
    backend2 = _RecordingBackend(halt2)
    # Pre-mark a fake non-structured call so backend2's first real chat falls
    # through the halt arm — the first-call arm only fires when there is
    # exactly one non-structured call recorded.
    backend2.calls.append({"messages": [], "response_schema": None})

    ff2 = FlowFactory(make_test_logger(), saia_factory=_SaiaFactory(backend2))
    flow2 = (
        ff2.create(state={})
        .with_checkpointer(store, "real-saia-resume")
        .with_halt(halt2)
        .iterate(body, max_iters=2)
    )
    await flow2.run(
        resume=True,
        # Caller supplies a DIFFERENT task + conv to prove the resume path
        # forwards the SAVED values from the envelope, not the caller's.
        extra={"task": "caller-task", "conv": _SerializableConv()},
    )

    # Backend2 saw at least one real (non-structured) chat call carrying the
    # user message threaded from the saved task, not the caller's task.
    real_calls = [
        c for c in backend2.calls[1:] if c.get("response_schema") is None and c["messages"]
    ]
    assert real_calls, "resume path did not reach the backend"
    resumed_msgs = real_calls[0]["messages"]
    assert any(m.role == "user" and m.content == "please answer" for m in resumed_msgs), (
        f"resumed dispatch should carry the SAVED task 'please answer', not "
        f"the caller's 'caller-task'; got {[(m.role, m.content) for m in resumed_msgs]}"
    )
