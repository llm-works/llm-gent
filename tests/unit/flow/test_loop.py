# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the Loop primitive and LoopFactory.

Covers:

- ``.role`` presence (Buildable contract) and drop-in use as a
  ``Flow.call`` / ``.iterate`` / ``.map`` body.
- Halt resolution rule (explicit Loop(halt=X) wins over ctx.halt;
  ctx.halt is fallback).
- Lifecycle hook ordering: on_start / on_resume, on_executor_ready,
  on_iteration bridge, on_cost, on_complete.
- Checkpointer seam: load-at-start decides start/resume path;
  delete-on-non-paused only.
- LoopFactory: default inheritance, per-create overrides, ``with_halt`` /
  ``with_checkpointer`` derivations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_gent.flow import (
    CheckpointStore,
    Context,
    Flow,
    Loop,
    LoopFactory,
    Role,
    verb,
)

from .conftest import ROLE_A, make_ff, make_test_logger


# -----------------------------------------------------------------------------
# Stubs
# -----------------------------------------------------------------------------


@dataclass
class _StubResult:
    """Minimal stand-in for SAIA's ``TaskResult``."""

    paused: bool = False
    reason: str = "done"


@dataclass
class _StubResponse:
    """Minimal stand-in for SAIA's per-turn ``ChatResponse``."""

    text: str = ""


class _CompleteSAIA:
    """SAIA stub whose ``.complete`` is recording + configurable."""

    def __init__(
        self,
        role: Role,
        *,
        result: _StubResult | None = None,
        iterations: int = 0,
    ) -> None:
        self.role = role
        self._result = result or _StubResult()
        self._iterations = iterations
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        task: str,
        *,
        on_iteration: Any = None,
        conversation: Any = None,
        abort_signal: Any = None,
        resume: bool = False,
    ) -> _StubResult:
        call = {
            "task": task,
            "on_iteration": on_iteration,
            "conversation": conversation,
            "abort_signal": abort_signal,
            "resume": resume,
        }
        self.calls.append(call)
        if on_iteration is not None:
            for i in range(self._iterations):
                await on_iteration(i, _StubResponse(text=f"turn-{i}"))
        return self._result


class _CompleteFactory:
    """SAIAFactory that hands out ``_CompleteSAIA`` instances."""

    def __init__(
        self,
        *,
        result: _StubResult | None = None,
        iterations: int = 0,
    ) -> None:
        self._result = result
        self._iterations = iterations
        self.built: list[_CompleteSAIA] = []

    def build(self, role: Role) -> _CompleteSAIA:
        saia = _CompleteSAIA(role, result=self._result, iterations=self._iterations)
        self.built.append(saia)
        return saia


@dataclass
class _RecordingStore:
    """CheckpointStore stub recording every call.

    ``preload`` is what :meth:`load_checkpoint` returns for
    ``(scope_id, run_id)`` matches (any scope_id + any run_id → the same
    dict); ``None`` means "no checkpoint here" and drives the on_start
    path in Loop.
    """

    preload: dict[str, Any] | None = None
    saves: list[tuple[str, int, dict[str, Any]]] = field(default_factory=list)
    loads: list[tuple[str, int | None]] = field(default_factory=list)
    deletes: list[tuple[str, int | None]] = field(default_factory=list)

    def save_checkpoint(self, scope_id: str, run_id: int, state: dict[str, Any]) -> None:
        self.saves.append((scope_id, run_id, state))

    def load_checkpoint(self, scope_id: str, run_id: int | None = None) -> dict[str, Any] | None:
        self.loads.append((scope_id, run_id))
        return self.preload

    def delete_checkpoint(self, scope_id: str, run_id: int | None = None) -> None:
        self.deletes.append((scope_id, run_id))


# -----------------------------------------------------------------------------
# Buildable contract
# -----------------------------------------------------------------------------


class TestBuildable:
    """Loop exposes ``.role: Role`` — drops into any Flow body= slot."""

    def test_role_property_returns_construction_role(self) -> None:
        """``loop.role`` is exactly the role passed to the constructor."""
        loop = Loop(ROLE_A)
        assert loop.role is ROLE_A

    def test_flow_call_accepts_loop_as_target(self) -> None:
        """``Flow.call(loop)`` passes the target validator (has ``.role``)."""
        flow = make_ff().create()
        loop = Loop(ROLE_A)
        flow.call(loop)  # no raise → valid Buildable

    def test_flow_iterate_accepts_loop_body(self) -> None:
        """A Loop drops into ``.iterate(body=)`` without wrapping."""
        flow = make_ff().create()
        loop = Loop(ROLE_A)
        flow.iterate(loop, max_iters=1)  # no raise

    def test_flow_map_accepts_loop_body(self) -> None:
        """A Loop drops into ``.map(body=)`` without wrapping."""
        flow = make_ff().create()
        loop = Loop(ROLE_A)
        flow.map(loop)  # no raise


# -----------------------------------------------------------------------------
# saia + Flow-body integration
# -----------------------------------------------------------------------------


class TestFlowBodyIntegration:
    """Loop runs under the enclosing Flow's SAIA + ctx wiring."""

    @pytest.mark.asyncio
    async def test_call_forwards_task_to_saia_complete(self) -> None:
        """The chain's positional reaches ``saia.complete(task, ...)``."""
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).create().call(Loop(ROLE_A))
        result = await flow.run("hello")
        assert isinstance(result, _StubResult)
        assert factory.built[0].calls == [
            {
                "task": "hello",
                "on_iteration": None,
                "conversation": None,
                "abort_signal": None,
                "resume": False,
            }
        ]

    @pytest.mark.asyncio
    async def test_ctx_saia_missing_raises_informative_error(self) -> None:
        """No SAIAFactory → ``ctx.saia`` accessor raises before Loop touches it."""
        # Loop needs ctx.saia; a factoryless Flow raises at the accessor.
        flow = Flow(make_test_logger()).call(Loop(ROLE_A))
        with pytest.raises(RuntimeError, match="SAIAFactory"):
            await flow.run("x")

    def test_require_saia_raises_when_saia_is_none(self) -> None:
        """``_require_saia`` raises with informative message when ctx.saia is None."""
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.saia = None
        ctx.role = ROLE_A
        with pytest.raises(RuntimeError, match="Loop requires ctx.saia"):
            Loop(ROLE_A)._require_saia(ctx)


# -----------------------------------------------------------------------------
# Halt resolution
# -----------------------------------------------------------------------------


class TestSAIAResolution:
    """Explicit ``Loop(saia=X)`` wins over ambient ``ctx.saia`` (halt precedent)."""

    @pytest.mark.asyncio
    async def test_explicit_saia_bypasses_ctx_saia(self) -> None:
        """``Loop(saia=X)`` uses X; the enclosing flow's SAIAFactory is not consulted."""
        explicit = _CompleteSAIA(ROLE_A)
        factory = _CompleteFactory()
        loop = Loop(ROLE_A, saia=explicit)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert explicit.calls[0]["task"] == "t"
        assert factory.built == []  # factory never consulted

    @pytest.mark.asyncio
    async def test_explicit_saia_works_without_saia_factory(self) -> None:
        """Loop with explicit ``saia=`` runs even under a factoryless Flow."""
        explicit = _CompleteSAIA(ROLE_A)
        loop = Loop(ROLE_A, saia=explicit)
        flow = Flow(make_test_logger()).call(loop)
        await flow.run("t")
        assert explicit.calls[0]["task"] == "t"

    @pytest.mark.asyncio
    async def test_no_explicit_saia_falls_back_to_ctx_saia(self) -> None:
        """Without ``Loop(saia=)``, ctx.saia (from the flow's SAIAFactory) is used."""
        factory = _CompleteFactory()
        loop = Loop(ROLE_A)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert factory.built and factory.built[0].calls[0]["task"] == "t"

    def test_loop_factory_create_forwards_explicit_saia(self) -> None:
        """``LoopFactory.create(role, saia=X)`` pins X on the built Loop."""
        explicit = _CompleteSAIA(ROLE_A)
        lf = LoopFactory(make_test_logger())
        loop = lf.create(ROLE_A, saia=explicit)
        assert loop._saia is explicit


class TestHaltResolution:
    """Explicit Loop(halt=X) wins over ambient ctx.halt (ctx.saia precedent)."""

    @pytest.mark.asyncio
    async def test_explicit_halt_becomes_abort_signal(self) -> None:
        """``Loop(halt=X)`` forwards ``X`` to ``saia.complete(abort_signal=)``."""
        explicit = asyncio.Event()
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).create().call(Loop(ROLE_A, halt=explicit))
        await flow.run("t")
        assert factory.built[0].calls[0]["abort_signal"] is explicit

    @pytest.mark.asyncio
    async def test_ambient_ctx_halt_used_when_no_explicit(self) -> None:
        """Without ``Loop(halt=)``, ``ctx.halt`` (from ``Flow.with_halt``) is used."""
        ambient = asyncio.Event()
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).with_halt(ambient).create().call(Loop(ROLE_A))
        await flow.run("t")
        assert factory.built[0].calls[0]["abort_signal"] is ambient

    @pytest.mark.asyncio
    async def test_explicit_wins_over_ambient(self) -> None:
        """Both wired → explicit wins."""
        explicit = asyncio.Event()
        ambient = asyncio.Event()
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).with_halt(ambient).create().call(Loop(ROLE_A, halt=explicit))
        await flow.run("t")
        assert factory.built[0].calls[0]["abort_signal"] is explicit


# -----------------------------------------------------------------------------
# Lifecycle hooks
# -----------------------------------------------------------------------------


class TestLifecycleHooks:
    """Hook firing conditions + ordering."""

    @pytest.mark.asyncio
    async def test_start_and_complete_fire_on_normal_run(self) -> None:
        """Non-resume, non-paused: on_start → complete → on_cost → on_complete."""
        events: list[str] = []
        factory = _CompleteFactory()

        def on_start(ctx: Context) -> None:
            events.append("start")

        def on_complete(result: Any, ctx: Context) -> None:
            events.append("complete")

        def on_cost(result: Any, ctx: Context) -> None:
            events.append("cost")

        loop = Loop(ROLE_A, on_start=on_start, on_complete=on_complete, on_cost=on_cost)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert events == ["start", "cost", "complete"]

    @pytest.mark.asyncio
    async def test_on_executor_ready_fires_before_complete(self) -> None:
        """on_executor_ready runs after ctx.saia resolves, before saia.complete."""
        events: list[str] = []
        factory = _CompleteFactory()

        def on_ready(saia: Any, ctx: Context) -> None:
            events.append(f"ready:{type(saia).__name__}")

        def on_start(ctx: Context) -> None:
            events.append("start")

        loop = Loop(ROLE_A, on_executor_ready=on_ready, on_start=on_start)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        # on_executor_ready fires before on_start
        assert events == ["ready:_CompleteSAIA", "start"]

    @pytest.mark.asyncio
    async def test_iteration_hook_bridges_to_saia_per_turn(self) -> None:
        """SAIA's per-turn callback reaches ``on_iteration`` with ctx bound."""
        turns: list[tuple[int, str]] = []
        factory = _CompleteFactory(iterations=3)

        async def on_iter(i: int, response: Any, ctx: Context) -> None:
            turns.append((i, response.text))
            assert ctx.role is ROLE_A

        loop = Loop(ROLE_A, on_iteration=on_iter)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert turns == [(0, "turn-0"), (1, "turn-1"), (2, "turn-2")]

    @pytest.mark.asyncio
    async def test_paused_skips_on_complete_but_runs_on_cost(self) -> None:
        """Paused result: on_cost still fires, on_complete does not."""
        events: list[str] = []
        factory = _CompleteFactory(result=_StubResult(paused=True))

        def on_complete(result: Any, ctx: Context) -> None:
            events.append("complete")

        def on_cost(result: Any, ctx: Context) -> None:
            events.append("cost")

        loop = Loop(ROLE_A, on_complete=on_complete, on_cost=on_cost)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert events == ["cost"]

    @pytest.mark.asyncio
    async def test_async_hook_is_awaited(self) -> None:
        """A coroutine-returning hook is awaited before Loop proceeds."""
        events: list[str] = []
        factory = _CompleteFactory()

        async def on_start(ctx: Context) -> None:
            await asyncio.sleep(0)
            events.append("start")

        loop = Loop(ROLE_A, on_start=on_start)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert events == ["start"]


class _RaisingSAIA:
    """SAIA stub whose ``.complete`` raises a configured exception."""

    def __init__(self, role: Role, *, exc: BaseException) -> None:
        self.role = role
        self._exc = exc

    async def complete(self, task: str, **_: Any) -> Any:
        raise self._exc


class _RaisingFactory:
    """SAIAFactory that hands out ``_RaisingSAIA`` instances."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def build(self, role: Role) -> _RaisingSAIA:
        return _RaisingSAIA(role, exc=self._exc)


class TestPausedAndCompleteReturn:
    """on_paused / on_complete fire exclusively; non-None returns override."""

    @pytest.mark.asyncio
    async def test_paused_result_fires_on_paused_not_on_complete(self) -> None:
        """Paused result → on_paused fires, on_complete skipped."""
        events: list[str] = []
        factory = _CompleteFactory(result=_StubResult(paused=True, reason="tool"))

        def on_complete(result: Any, ctx: Context) -> None:
            events.append("complete")

        def on_paused(result: Any, ctx: Context) -> None:
            events.append(f"paused:{result.reason}")

        loop = Loop(ROLE_A, on_complete=on_complete, on_paused=on_paused)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert events == ["paused:tool"]

    @pytest.mark.asyncio
    async def test_on_complete_non_none_return_replaces_result(self) -> None:
        """Non-None from on_complete becomes Loop's return value."""
        factory = _CompleteFactory()

        def on_complete(result: Any, ctx: Context) -> dict[str, Any]:
            return {"wrapped": result}

        loop = Loop(ROLE_A, on_complete=on_complete)
        flow = make_ff(saia_f=factory).create().call(loop)
        got = await flow.run("t")
        assert isinstance(got, dict)
        assert isinstance(got["wrapped"], _StubResult)

    @pytest.mark.asyncio
    async def test_on_complete_none_return_preserves_raw_result(self) -> None:
        """Existing hooks returning None keep the raw SAIA result (bc-compat)."""
        factory = _CompleteFactory()

        def on_complete(result: Any, ctx: Context) -> None:
            return None

        loop = Loop(ROLE_A, on_complete=on_complete)
        flow = make_ff(saia_f=factory).create().call(loop)
        got = await flow.run("t")
        assert isinstance(got, _StubResult)

    @pytest.mark.asyncio
    async def test_on_paused_non_none_return_replaces_result(self) -> None:
        """Non-None from on_paused becomes Loop's return value."""
        factory = _CompleteFactory(result=_StubResult(paused=True, reason="halt"))

        def on_paused(result: Any, ctx: Context) -> str:
            return f"paused-token:{result.reason}"

        loop = Loop(ROLE_A, on_paused=on_paused)
        flow = make_ff(saia_f=factory).create().call(loop)
        got = await flow.run("t")
        assert got == "paused-token:halt"


class TestExceptionalPaths:
    """on_cancelled / on_failed / on_finally lifecycle."""

    @pytest.mark.asyncio
    async def test_on_cancelled_fires_then_reraises(self) -> None:
        """CancelledError → on_cancelled fires, on_finally fires, cancellation propagates."""
        events: list[str] = []
        factory = _RaisingFactory(asyncio.CancelledError())

        def on_cancelled(ctx: Context) -> None:
            events.append("cancelled")

        def on_failed(exc: Exception, ctx: Context) -> None:
            events.append("failed")

        def on_finally(ctx: Context) -> None:
            events.append("finally")

        loop = Loop(
            ROLE_A,
            on_cancelled=on_cancelled,
            on_failed=on_failed,
            on_finally=on_finally,
        )
        flow = make_ff(saia_f=factory).create().call(loop)
        with pytest.raises(asyncio.CancelledError):
            await flow.run("t")
        assert events == ["cancelled", "finally"]

    @pytest.mark.asyncio
    async def test_on_failed_fires_then_reraises(self) -> None:
        """Non-cancellation Exception → on_failed fires with exc, on_finally, re-raise."""
        events: list[Any] = []
        boom = RuntimeError("boom")
        factory = _RaisingFactory(boom)

        def on_cancelled(ctx: Context) -> None:
            events.append("cancelled")

        def on_failed(exc: Exception, ctx: Context) -> None:
            events.append(("failed", exc))

        def on_finally(ctx: Context) -> None:
            events.append("finally")

        loop = Loop(
            ROLE_A,
            on_cancelled=on_cancelled,
            on_failed=on_failed,
            on_finally=on_finally,
        )
        flow = make_ff(saia_f=factory).create().call(loop)
        with pytest.raises(RuntimeError, match="boom"):
            await flow.run("t")
        assert events == [("failed", boom), "finally"]

    @pytest.mark.asyncio
    async def test_on_finally_fires_on_normal_completion(self) -> None:
        """Normal (non-paused) run: on_finally fires last, after on_complete."""
        events: list[str] = []
        factory = _CompleteFactory()

        def on_complete(result: Any, ctx: Context) -> None:
            events.append("complete")

        def on_finally(ctx: Context) -> None:
            events.append("finally")

        loop = Loop(ROLE_A, on_complete=on_complete, on_finally=on_finally)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert events == ["complete", "finally"]

    @pytest.mark.asyncio
    async def test_on_finally_fires_on_paused(self) -> None:
        """Paused run: on_finally fires last, after on_paused."""
        events: list[str] = []
        factory = _CompleteFactory(result=_StubResult(paused=True))

        def on_paused(result: Any, ctx: Context) -> None:
            events.append("paused")

        def on_finally(ctx: Context) -> None:
            events.append("finally")

        loop = Loop(ROLE_A, on_paused=on_paused, on_finally=on_finally)
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert events == ["paused", "finally"]

    @pytest.mark.asyncio
    async def test_cancelled_does_not_delete_checkpoint(self) -> None:
        """Cancellation preserves the checkpoint (resume path)."""
        store = _RecordingStore()
        factory = _RaisingFactory(asyncio.CancelledError())
        loop = Loop(ROLE_A, checkpointer=store)
        flow = make_ff(saia_f=factory).create()
        flow.register(loop, name="loop")
        with pytest.raises(asyncio.CancelledError):
            await flow.dispatch("loop", "t", scope_id="s1", run_id=1)
        assert store.deletes == []

    @pytest.mark.asyncio
    async def test_failed_does_not_delete_checkpoint(self) -> None:
        """Failure preserves the checkpoint (resume path)."""
        store = _RecordingStore()
        factory = _RaisingFactory(RuntimeError("boom"))
        loop = Loop(ROLE_A, checkpointer=store)
        flow = make_ff(saia_f=factory).create()
        flow.register(loop, name="loop")
        with pytest.raises(RuntimeError):
            await flow.dispatch("loop", "t", scope_id="s1", run_id=1)
        assert store.deletes == []

    @pytest.mark.asyncio
    async def test_on_failed_does_not_swallow_cancellation(self) -> None:
        """CancelledError takes on_cancelled path, NOT on_failed."""
        called: list[str] = []
        factory = _RaisingFactory(asyncio.CancelledError())

        def on_failed(exc: Exception, ctx: Context) -> None:
            called.append("failed")

        loop = Loop(ROLE_A, on_failed=on_failed)
        flow = make_ff(saia_f=factory).create().call(loop)
        with pytest.raises(asyncio.CancelledError):
            await flow.run("t")
        assert called == []


# -----------------------------------------------------------------------------
# Checkpointer seam
# -----------------------------------------------------------------------------


class TestCheckpointer:
    """Load-at-start decides start/resume path; delete-on-non-paused only."""

    @pytest.mark.asyncio
    async def test_no_checkpoint_takes_start_path(self) -> None:
        """``load_checkpoint`` → None → on_start fires (not on_resume)."""
        events: list[str] = []
        store = _RecordingStore(preload=None)

        def on_start(ctx: Context) -> None:
            events.append("start")

        def on_resume(state: Any, ctx: Context) -> None:
            events.append("resume")

        loop = Loop(ROLE_A, checkpointer=store, on_start=on_start, on_resume=on_resume)
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).create()
        flow.register(loop, name="loop")
        await flow.dispatch("loop", "t", scope_id="s1", run_id=1)
        assert events == ["start"]
        assert store.loads == [("s1", 1)]

    @pytest.mark.asyncio
    async def test_present_checkpoint_takes_resume_path(self) -> None:
        """``load_checkpoint`` → dict → on_resume fires (not on_start), resume=True."""
        events: list[Any] = []
        preload = {"turn": 3, "history": ["a"]}
        store = _RecordingStore(preload=preload)

        def on_start(ctx: Context) -> None:
            events.append("start")

        def on_resume(state: Any, ctx: Context) -> None:
            events.append(("resume", state))

        loop = Loop(ROLE_A, checkpointer=store, on_start=on_start, on_resume=on_resume)
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).create()
        flow.register(loop, name="loop")
        await flow.dispatch("loop", "t", scope_id="s1", run_id=2)
        assert events == [("resume", preload)]
        assert factory.built[0].calls[0]["resume"] is True

    @pytest.mark.asyncio
    async def test_delete_on_non_paused(self) -> None:
        """Successful (non-paused) result → delete_checkpoint called."""
        store = _RecordingStore()
        loop = Loop(ROLE_A, checkpointer=store)
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).create()
        flow.register(loop, name="loop")
        await flow.dispatch("loop", "t", scope_id="s9", run_id=7)
        assert store.deletes == [("s9", 7)]

    @pytest.mark.asyncio
    async def test_no_delete_on_paused(self) -> None:
        """Paused result → delete_checkpoint NOT called (checkpoint stays)."""
        store = _RecordingStore()
        loop = Loop(ROLE_A, checkpointer=store)
        factory = _CompleteFactory(result=_StubResult(paused=True))
        flow = make_ff(saia_f=factory).create()
        flow.register(loop, name="loop")
        await flow.dispatch("loop", "t", scope_id="s9", run_id=7)
        assert store.deletes == []

    @pytest.mark.asyncio
    async def test_no_scope_id_bypasses_checkpointer_entirely(self) -> None:
        """``scope_id=None`` → checkpointer is not consulted regardless of wiring."""
        store = _RecordingStore(preload={"anything": True})
        loop = Loop(ROLE_A, checkpointer=store)
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")  # no scope_id → no load, no delete
        assert store.loads == []
        assert store.deletes == []


# -----------------------------------------------------------------------------
# LoopFactory
# -----------------------------------------------------------------------------


class TestLoopFactory:
    """Default inheritance, per-create overrides, with_* derivations."""

    def test_create_inherits_defaults(self) -> None:
        """``create(role)`` yields a Loop carrying the factory's halt + checkpointer."""
        halt = asyncio.Event()
        store = _RecordingStore()
        lf = LoopFactory(make_test_logger(), checkpointer=store, halt=halt)
        loop = lf.create(ROLE_A)
        assert loop._halt is halt
        assert loop._checkpointer is store
        assert loop.role is ROLE_A

    def test_per_create_halt_overrides_factory_default(self) -> None:
        """``create(halt=Y)`` wins over factory-wide default."""
        default = asyncio.Event()
        override = asyncio.Event()
        lf = LoopFactory(make_test_logger(), halt=default)
        loop = lf.create(ROLE_A, halt=override)
        assert loop._halt is override

    def test_with_halt_derivation_preserves_other_slots(self) -> None:
        """``with_halt`` swaps only halt; saia_f + checkpointer + lg carry over."""
        store = _RecordingStore()
        first = LoopFactory(make_test_logger(), checkpointer=store)
        new_event = asyncio.Event()
        second = first.with_halt(new_event)
        assert second is not first
        assert second.halt is new_event
        assert second.checkpointer is store
        assert second._lg is first._lg

    def test_with_checkpointer_derivation_preserves_other_slots(self) -> None:
        """``with_checkpointer`` swaps only checkpointer; halt + lg carry over."""
        halt = asyncio.Event()
        first = LoopFactory(make_test_logger(), halt=halt)
        new_store = _RecordingStore()
        second = first.with_checkpointer(new_store)
        assert second.checkpointer is new_store
        assert second.halt is halt

    def test_with_saia_f_preserves_halt_and_checkpointer(self) -> None:
        """``with_saia_f`` swaps only saia_f; halt + checkpointer carry over."""
        halt = asyncio.Event()
        store = _RecordingStore()
        first = LoopFactory(make_test_logger(), checkpointer=store, halt=halt)

        class _F:
            def build(self, role: Role) -> Any:
                return _CompleteSAIA(role)

        replacement = _F()
        second = first.with_saia_f(replacement)
        assert second.saia_f is replacement
        assert first.saia_f is None
        assert second.halt is halt
        assert second.checkpointer is store

    @pytest.mark.asyncio
    async def test_factory_halt_reaches_saia_abort_signal_end_to_end(self) -> None:
        """``LoopFactory(..).with_halt(e)`` → Loops built by it hand ``e`` to saia."""
        halt = asyncio.Event()
        lf = LoopFactory(make_test_logger()).with_halt(halt)
        loop = lf.create(ROLE_A)
        factory = _CompleteFactory()
        flow = make_ff(saia_f=factory).create().call(loop)
        await flow.run("t")
        assert factory.built[0].calls[0]["abort_signal"] is halt


# -----------------------------------------------------------------------------
# Protocol conformance
# -----------------------------------------------------------------------------


class TestCheckpointStoreProtocol:
    """``_RecordingStore`` satisfies the CheckpointStore Protocol structurally."""

    def test_recording_store_matches_protocol(self) -> None:
        """``_RecordingStore`` satisfies CheckpointStore structurally."""
        store = _RecordingStore()
        # Protocol without @runtime_checkable — verify method presence at runtime.
        assert callable(store.save_checkpoint)
        assert callable(store.load_checkpoint)
        assert callable(store.delete_checkpoint)
        # Static check: assignment to Protocol type verifies structural conformance.
        _: CheckpointStore = store
        Loop(ROLE_A, checkpointer=store)


# -----------------------------------------------------------------------------
# Cross-check: Loop as verb in a chain of verbs
# -----------------------------------------------------------------------------


@verb(role=ROLE_A)
async def _extract_task(ctx: Context, payload: dict[str, Any]) -> str:
    """Pull ``payload["task"]`` out for the Loop that follows."""
    return payload["task"]


class TestChainedUse:
    """Loop composes naturally with @verb functions in a Flow chain."""

    @pytest.mark.asyncio
    async def test_verb_then_loop_pipeline(self) -> None:
        """``.call(verb).then(loop)`` pipes the verb's return into Loop's task."""
        factory = _CompleteFactory()
        loop = Loop(ROLE_A)
        flow = make_ff(saia_f=factory).create().call(_extract_task).then(loop)
        await flow.run({"task": "extract-me"})
        assert factory.built[0].calls[0]["task"] == "extract-me"
