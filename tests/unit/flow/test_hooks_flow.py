# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Hooks in a running Flow: which events fire where, attachment, verdicts, checkpoints, Loop calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from llm_gent.flow import HALTED, Context, Factory, Flow, Interrupted, Loop, Role, verb
from llm_gent.flow.hooks import (
    CheckpointEvent,
    Hook,
    HookDenied,
    HookEvent,
    LLMCallEvent,
    Pause,
    RunEvent,
    StepEvent,
    Where,
)
from tests.helpers import memory_store

from .conftest import make_ff, make_test_logger


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


class Recorder(Hook):
    """Keeps every event it gets."""

    def __init__(self) -> None:
        self.events: list[HookEvent] = []

    def before_run(self, event: RunEvent) -> None:
        self.events.append(event)

    def after_run(self, event: RunEvent) -> None:
        self.events.append(event)

    def before_step(self, event: StepEvent) -> None:
        self.events.append(event)

    def after_step(self, event: StepEvent) -> None:
        self.events.append(event)

    def after_llm_call(self, event: LLMCallEvent) -> None:
        self.events.append(event)

    def after_checkpoint(self, event: CheckpointEvent) -> None:
        self.events.append(event)

    def steps(self, name: str = "after_step") -> list[StepEvent]:
        return [e for e in self.events if isinstance(e, StepEvent) and e.name == name]

    def names(self) -> list[str]:
        """Each event as ``name`` or ``name:step``."""
        return [
            f"{e.name}:{e.where.step_name}" if isinstance(e, StepEvent) else e.name
            for e in self.events
        ]


class Gate(Hook):
    """``before_step`` returns ``verdicts[step name]`` (allow when absent)."""

    def __init__(self, verdicts: dict[str, Any]) -> None:
        self.verdicts = verdicts
        self.asked: list[str | None] = []

    def before_step(self, event: StepEvent) -> Any:
        self.asked.append(event.where.step_name)
        return self.verdicts.get(event.where.step_name or "")


@verb
async def add_one(ctx: Context[dict[str, Any]], x: int) -> int:
    ctx.state.data.setdefault("ran", []).append("add_one")
    return x + 1


@verb
async def double(ctx: Context[dict[str, Any]], x: int) -> int:
    ctx.state.data.setdefault("ran", []).append("double")
    return x * 2


def _flow(**kw: Any) -> Flow:
    return Factory(make_test_logger()).create(state={}, **kw)


def _two_steps() -> Flow:
    return _flow().call(add_one, name="a").then(double, name="b")


class TestEvents:
    async def test_a_run_calls_run_and_step_events_in_order(self) -> None:
        rec = Recorder()
        assert await _two_steps().with_hooks(rec).run(1) == 4
        assert rec.names() == [
            "before_run",
            "before_step:a",
            "after_step:a",
            "before_step:b",
            "after_step:b",
            "after_run",
        ]

    async def test_step_events_carry_inputs_result_and_outcome(self) -> None:
        rec = Recorder()
        await _two_steps().with_hooks(rec).run(1)
        a, b = rec.steps()
        assert (a.args, a.result, a.outcome) == ((1,), 2, "ok")
        assert (b.args, b.result, b.outcome) == ((2,), 4, "ok")
        assert a.where.kind == "verb" and a.actor.verb == "add_one"
        assert isinstance(a.ctx, Context)

    async def test_steps_belong_to_the_run_and_nested_steps_to_their_step(self) -> None:
        rec = Recorder()
        inner = _flow().call(add_one, name="inner")
        await _flow().call(inner, name="outer").with_hooks(rec).run(1)
        run = rec.events[0]
        outer = next(e for e in rec.steps() if e.where.step_name == "outer")
        nested = next(e for e in rec.steps() if e.where.step_name == "inner")
        assert outer.parent_activity_id == run.run.run_id
        assert nested.parent_activity_id == outer.activity_id
        assert outer.where.kind == "subflow"
        assert nested.where.node_path[:-1] == outer.where.node_path

    async def test_each_execution_is_its_own_activity(self) -> None:
        rec = Recorder()
        await (
            _flow().iterate(_flow().call(add_one, name="body"), max_iters=3).with_hooks(rec).run(0)
        )
        bodies = [e for e in rec.steps() if e.where.step_name == "body"]
        assert len({e.activity_id for e in bodies}) == 3
        assert len({e.where.scope_path for e in bodies}) == 3

    async def test_a_failing_step_ends_failed_and_so_does_the_run(self) -> None:
        @verb
        async def boom(ctx: Context[Any], x: int) -> int:
            raise ValueError("boom")

        rec = Recorder()
        with pytest.raises(ValueError):
            await _flow().call(boom, name="boom").with_hooks(rec).run(1)
        (step,) = rec.steps()
        assert step.outcome == "failed" and isinstance(step.error, ValueError)
        assert rec.events[-1].name == "after_run"
        assert isinstance(rec.events[-1], RunEvent) and rec.events[-1].outcome == "failed"

    async def test_a_cancelled_step_ends_cancelled_and_so_does_the_run(self) -> None:
        @verb
        async def cancelled(ctx: Context[Any], x: int) -> int:
            raise asyncio.CancelledError

        rec = Recorder()
        with pytest.raises(asyncio.CancelledError):
            await _flow().call(cancelled, name="c").with_hooks(rec).run(1)
        (step,) = rec.steps()
        assert step.outcome == "cancelled" and isinstance(step.error, asyncio.CancelledError)
        run = rec.events[-1]
        assert isinstance(run, RunEvent) and run.name == "after_run"
        assert run.outcome == "cancelled" and isinstance(run.error, asyncio.CancelledError)

    async def test_a_step_interrupted_without_a_halt_ends_failed(self) -> None:
        @verb
        async def stray(ctx: Context[Any], x: int) -> int:
            raise Interrupted()

        rec = Recorder()
        with pytest.raises(RuntimeError, match="while no halt is set") as raised:
            await _flow().call(stray, name="s").with_hooks(rec).run(1)
        (step,) = rec.steps()
        assert step.outcome == "failed" and step.error is raised.value

    async def test_a_run_whose_finalization_raises_ends_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def raises(*_: Any) -> None:
            raise OSError("store unavailable")

        monkeypatch.setattr("llm_gent.flow.flow.apply_clean_exit_retention", raises)
        rec = Recorder()
        with pytest.raises(OSError, match="store unavailable"):
            await _two_steps().with_hooks(rec).run(1)
        run = rec.events[-1]
        assert isinstance(run, RunEvent) and run.name == "after_run"
        assert run.outcome == "failed" and isinstance(run.error, OSError)


class TestAttachment:
    async def test_hooks_on_a_subflow_see_only_its_steps(self) -> None:
        rec = Recorder()
        inner = _flow().call(add_one, name="inner").with_hooks(rec)
        await _flow().call(inner, name="outer").then(double, name="after").run(1)
        assert rec.names() == ["before_step:inner", "after_step:inner"]

    async def test_inherited_hooks_run_outside_the_local_ones(self) -> None:
        order: list[str] = []

        class Named(Hook):
            def __init__(self, name: str) -> None:
                self.name = name

            def before_step(self, event: StepEvent) -> None:
                if event.where.step_name == "inner":
                    order.append(f"before {self.name}")

            def after_step(self, event: StepEvent) -> None:
                if event.where.step_name == "inner":
                    order.append(f"after {self.name}")

        inner = _flow().call(add_one, name="inner").with_hooks(Named("local"))
        await _flow().call(inner).with_hooks(Named("inherited")).run(1)
        assert order == ["before inherited", "before local", "after local", "after inherited"]

    async def test_replace_drops_the_inherited_hooks(self) -> None:
        outer_rec, inner_rec = Recorder(), Recorder()
        inner = _flow().call(add_one, name="inner").with_hooks(inner_rec, replace=True)
        await _flow().call(inner, name="outer").with_hooks(outer_rec).run(1)
        assert "before_step:inner" not in outer_rec.names()
        assert "before_step:inner" in inner_rec.names()

    async def test_child_is_asked_at_each_map_item(self) -> None:
        wheres: list[Where] = []

        class PerItem(Hook):
            def child(self, where: Where) -> Hook:
                wheres.append(where)
                return self

        body = _flow().call(add_one, name="body")
        await _flow().map(body, items=lambda _prev, _ctx: [1, 2, 3]).with_hooks(PerItem()).run(0)
        assert len(wheres) == 3
        assert len({w.scope_path for w in wheres}) == 3

    async def test_ctx_hook_finds_the_attached_hook(self) -> None:
        rec = Recorder()
        found: list[Any] = []

        @verb
        async def look(ctx: Context[Any], x: int) -> int:
            found.extend([ctx.hook(Recorder), ctx.hook(Gate)])
            return x

        await _flow().call(look).with_hooks(rec).run(1)
        assert found == [rec, None]


class TestVerdicts:
    async def test_a_denied_step_does_not_run_and_fails_the_run(self) -> None:
        rec, gate = Recorder(), Gate({"b": "not allowed"})
        flow = _two_steps().with_hooks(gate, rec)
        with pytest.raises(HookDenied, match="not allowed"):
            await flow.run(1)
        b = next(e for e in rec.steps() if e.where.step_name == "b")
        assert b.outcome == "denied" and b.decision is not None
        assert b.decision.reason == "not allowed"
        assert flow.state.get("ran") is None or "double" not in flow.state["ran"]

    async def test_a_rescue_policy_handles_a_denied_step(self) -> None:
        rec, gate = Recorder(), Gate({"a": False})
        flow = (
            _flow()
            .call(add_one, name="a", rescue=lambda _exc, _pending, _ctx: -1)
            .with_hooks(gate, rec)
        )
        assert await flow.run(1) == -1
        (a,) = rec.steps()
        assert a.outcome == "denied" and a.result == -1

    async def test_a_denied_step_whose_rescue_is_cancelled_ends_cancelled(self) -> None:
        def cancel(_exc: Any, _pending: Any, _ctx: Any) -> int:
            raise asyncio.CancelledError

        rec, gate = Recorder(), Gate({"a": False})
        flow = _flow().call(add_one, name="a", rescue=cancel).with_hooks(gate, rec)
        with pytest.raises(asyncio.CancelledError):
            await flow.run(1)
        (a,) = rec.steps()
        assert a.outcome == "cancelled" and a.decision is not None

    async def test_a_paused_step_halts_the_run_and_resume_asks_again(self) -> None:
        store, halt = memory_store(), asyncio.Event()
        gate = Gate({"b": Pause("approval")})

        def flow() -> Flow:
            return _two_steps().with_checkpoint_store(store, "paused").with_hooks(gate)

        rec = Recorder()
        assert await flow().with_hooks(rec).run(1, halt=halt) is HALTED
        b = next(e for e in rec.steps() if e.where.step_name == "b")
        assert b.outcome == "paused"
        halt.clear()
        gate.verdicts.clear()  # approved
        assert await flow().run(resume="latest", halt=halt) == 4
        assert gate.asked == ["a", "b", "b"]

    async def test_a_pause_without_a_halt_raises(self) -> None:
        flow = _two_steps().with_hooks(Gate({"a": Pause()}))
        with pytest.raises(RuntimeError, match="no halt"):
            await flow.run(1)

    async def test_before_run_can_deny_or_pause_the_run(self) -> None:
        class RunGate(Hook):
            def __init__(self, verdict: Any) -> None:
                self.verdict = verdict

            def before_run(self, event: RunEvent) -> Any:
                return self.verdict

        rec = Recorder()
        with pytest.raises(HookDenied):
            await _two_steps().with_hooks(RunGate("closed"), rec).run(1)
        assert rec.names() == ["after_run"]
        assert await _two_steps().with_hooks(RunGate(Pause())).run(1) is HALTED


class TestCheckpoints:
    async def test_save_points_and_the_final_commit_are_reported(self) -> None:
        store = memory_store()
        hashes: list[str | None] = []

        @verb
        async def save(ctx: Context[Any], x: int) -> int:
            hashes.append(await ctx.checkpoint())
            return x

        rec = Recorder()
        flow = (
            _flow()
            .with_checkpoint_store(store, "cp")
            .with_checkpointer()
            .call(save)
            .with_hooks(rec)
        )
        await flow.run(1)
        commits = [e for e in rec.events if isinstance(e, CheckpointEvent)]
        assert [c.commit_id for c in commits][0] == hashes[0]
        assert [c.outcome for c in commits] == ["ok", "ok"]  # save point, final state
        assert commits[-1].parent_activity_id == commits[-1].run.run_id

    async def test_the_halt_commit_is_reported(self) -> None:
        store, halt = memory_store(), asyncio.Event()

        @verb
        async def stop(ctx: Context[Any], x: int) -> int:
            halt.set()
            return x

        rec = Recorder()
        flow = _flow().with_checkpoint_store(store, "halt").call(stop).then(add_one).with_hooks(rec)
        assert await flow.run(1, halt=halt) is HALTED
        (commit,) = [e for e in rec.events if isinstance(e, CheckpointEvent)]
        assert commit.outcome == "halted"
        assert commit.parent_activity_id == commit.run.run_id
        assert isinstance(rec.events[-1], RunEvent) and rec.events[-1].outcome == "halted"

    async def test_a_resumed_run_names_the_commit_it_started_from(self) -> None:
        store, halt = memory_store(), asyncio.Event()

        @verb
        async def stop(ctx: Context[Any], x: int) -> int:
            if not ctx.state.data.get("stopped"):
                ctx.state.data["stopped"] = True
                halt.set()
            return x

        def flow(rec: Recorder) -> Flow:
            return (
                _flow()
                .with_checkpoint_store(store, "resumed")
                .call(stop)
                .then(add_one)
                .with_hooks(rec)
            )

        first, second = Recorder(), Recorder()
        await flow(first).run(1, halt=halt)
        halt.clear()
        await flow(second).run(resume="latest", halt=halt)
        (halt_commit,) = [e for e in first.events if isinstance(e, CheckpointEvent)]
        assert second.events[0].run.resumed_from == halt_commit.commit_id
        assert first.events[0].run.resumed_from is None


@dataclass
class _Response:
    content: str
    model: str = "stub-model"
    call_id: str = ""


class _SAIA:
    """Calls ``on_iteration`` once per response, appending to the conversation after it."""

    def __init__(self, role: Role, responses: int) -> None:
        self.role, self.responses = role, responses

    async def complete(
        self, task: str, *, on_iteration: Any = None, conversation: Any = None, **_: Any
    ) -> Any:
        for i in range(self.responses):
            response = _Response(f"turn-{i}", call_id=f"call-{i}")
            if on_iteration is not None:
                await on_iteration(i, response)
            if conversation is not None:
                conversation.append(response.content)
        return None


class _Factory:
    def build(self, role: Role) -> _SAIA:
        return _SAIA(role, responses=2)


class TestLLMCalls:
    async def test_each_model_call_of_a_loop_is_reported_in_its_step(self) -> None:
        rec = Recorder()
        seen: list[list[str]] = []

        class Snapshot(Hook):
            def after_llm_call(self, event: LLMCallEvent) -> None:
                seen.append(list(event.conversation))

        class Conversations:
            def create(self) -> list[str]:
                return []

        loop = Loop(Role(name="worker"), conversation_factory=Conversations())  # type: ignore[arg-type]
        flow = make_ff(_Factory()).create(state={})
        await flow.call(loop, name="work").with_hooks(rec, Snapshot()).run("go")
        calls = [e for e in rec.events if isinstance(e, LLMCallEvent)]
        step = next(e for e in rec.steps() if e.where.step_name == "work")
        assert [c.response.content for c in calls] == ["turn-0", "turn-1"]
        assert [c.trace.call_id for c in calls] == ["call-0", "call-1"]
        assert {c.parent_activity_id for c in calls} == {step.activity_id}
        assert calls[0].actor.model == "stub-model" and calls[0].actor.role == "worker"
        # Each hook saw the context as sent, before that call's response.
        assert seen == [[], ["turn-0"]]
