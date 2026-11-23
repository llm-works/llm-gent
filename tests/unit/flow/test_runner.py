# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""FlowRunner and FlowRun: a runner starts a run of a flow; the run is awaited, held, halted."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from appinfra.cas import MemoryStore

from llm_gent.flow import (
    COST,
    HALTED,
    Context,
    Flow,
    FlowRun,
    FlowRunner,
    Fresh,
    Restart,
    Resume,
    verb,
)
from llm_gent.flow._node_id import iter_flows

from .conftest import make_ff, make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


@verb
async def double(ctx: Context[Any], x: int) -> int:
    return 2 * x


class TestStart:
    async def test_awaiting_the_run_gives_the_result(self) -> None:
        run = FlowRunner(LG).start(make_ff().create("d").call(double), 21)
        assert isinstance(run, FlowRun)
        assert await run == 42
        assert run.done

    async def test_runs_report_to_the_runner_logger(self) -> None:
        run = FlowRunner(LG).start(make_ff().create("d").call(double), 1)
        await run
        assert run.lg is LG

    async def test_one_flow_runs_many_times_at_once(self) -> None:
        flow = make_ff().create("d").call(double)
        runner = FlowRunner(LG)
        runs = [runner.start(flow, i) for i in range(5)]
        assert await asyncio.gather(*runs) == [0, 2, 4, 6, 8]

    async def test_a_flow_keyword_reaches_the_first_node(self) -> None:
        @verb
        async def consume(ctx: Context[Any], *, flow: str) -> str:
            return flow

        graph = make_ff().create("c").call(consume)
        assert await FlowRunner(LG).start(graph, flow="input") == "input"

    def test_needs_a_running_event_loop(self) -> None:
        with pytest.raises(RuntimeError, match="no running event loop"):
            FlowRunner(LG).start(make_ff().create("d").call(double), 1)


class TestChecksRaiseAtStart:
    """A misconfigured run raises from ``start()`` itself, before any task runs."""

    async def test_start_must_be_a_start(self) -> None:
        flow = make_ff().create("d").call(double)
        with pytest.raises(TypeError, match=r"start= takes Fresh\(\), Resume\(at\)"):
            FlowRunner(LG).start(flow, 1, start="latest")  # type: ignore[arg-type]

    @pytest.mark.parametrize("start", [Resume(), Restart("mid")])
    async def test_a_start_from_a_commit_needs_a_history(self, start: Any) -> None:
        flow = make_ff(checkpoint_store=MemoryStore(LG)).create("d").call(double)
        with pytest.raises(RuntimeError, match="but no history"):
            FlowRunner(LG).start(flow, 1, start=start)

    async def test_an_empty_flow_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="no nodes to run"):
            FlowRunner(LG).start(make_ff().create("empty"))


class TestStartTypes:
    @pytest.mark.parametrize("at", ["latest", "a" * 64, "mid"])
    def test_at_takes_latest_a_hash_or_a_checkpoint_name(self, at: str) -> None:
        assert Resume(at).at == at
        assert Restart(at).at == at

    @pytest.mark.parametrize("at", ["", "complete", "off", True])
    def test_anything_else_is_rejected(self, at: Any) -> None:
        with pytest.raises(ValueError, match=r"Resume\(at=\) takes 'latest'"):
            Resume(at)
        with pytest.raises(ValueError, match=r"Restart\(at=\) takes 'latest'"):
            Restart(at)

    def test_the_default_is_latest_and_fresh_has_no_commit(self) -> None:
        assert Resume() == Resume("latest")
        assert Restart() == Restart("latest")
        assert Fresh() == Fresh()


class TestHalt:
    async def test_halting_a_run_started_without_a_halt_stops_it_at_the_next_step(
        self,
    ) -> None:
        entered, gate = asyncio.Event(), asyncio.Event()
        ran: list[str] = []

        @verb
        async def first(ctx: Context[Any]) -> None:
            entered.set()
            await gate.wait()
            ran.append("first")

        @verb
        async def second(ctx: Context[Any]) -> None:
            ran.append("second")

        run = FlowRunner(LG).start(make_ff().create("h").call(first).call(second))
        await entered.wait()
        run.halt()
        gate.set()
        assert await run is HALTED
        assert ran == ["first"]

    async def test_halt_sets_the_event_the_run_was_given(self) -> None:
        halt = asyncio.Event()
        run = FlowRunner(LG).start(make_ff().create("d").call(double), 1, halt=halt)
        run.halt()
        assert halt.is_set()
        await run


class TestCancel:
    async def test_cancelling_the_awaiting_task_cancels_the_run(self) -> None:
        started = asyncio.Event()

        @verb
        async def forever(ctx: Context[Any]) -> None:
            started.set()
            await asyncio.Event().wait()

        run = FlowRunner(LG).start(make_ff().create("c").call(forever))
        waiter = asyncio.create_task(_await(run))
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert run.done
        assert run.task is not None and run.task.cancelled()


async def _await(run: FlowRun) -> Any:
    return await run


def _tree() -> tuple[Flow, list[Flow]]:
    """A flow with a called subflow, branch arms, an iterate body and a map body."""
    ff = make_ff()
    called = ff.create("called").call(double)
    flow = (
        ff.create("top")
        .call(called)
        .branch(when=lambda _r, _c: True, then=lambda b: b.call(double))
        .iterate(lambda b: b.call(double), max_iters=1)
        .map(lambda b: b.call(double), items=lambda _r, _c: [1])
    )
    return flow, iter_flows(flow)


# Each builder method with arguments it would accept on an open flow.
_CHANGES: dict[str, Any] = {
    "register": lambda f: f.register(double, "d"),
    "call": lambda f: f.call(double),
    "then": lambda f: f.then(double),
    "conclude": lambda f: f.conclude(double),
    "rescue": lambda f: f.rescue(lambda *_: None),
    "after": lambda f: f.after(lambda *_: None),
    "branch": lambda f: f.branch(when=lambda _r, _c: True, then=lambda b: b.call(double)),
    "iterate": lambda f: f.iterate(lambda b: b.call(double), max_iters=1),
    "map": lambda f: f.map(lambda b: b.call(double), items=lambda _r, _c: [1]),
    "guard": lambda f: f.guard(lambda *_: True),
    "on_error": lambda f: f.on_error(lambda *_: None),
    "on_item_complete": lambda f: f.on_item_complete(lambda *_: None),
    "with_shortcut": lambda f: f.with_shortcut("cut"),
    "with_budget": lambda f: f.with_budget(1.0),
    "with_resource": lambda f: f.with_resource(COST),
    "with_checkpointer": lambda f: f.with_checkpointer(),
    "with_checkpoint_policy": lambda f: f.with_checkpoint_policy(on_iterate=True),
    "with_hooks": lambda f: f.with_hooks(),
}


class TestFrozenOnceStarted:
    """A started flow's runs share its structure: from its first start on, it no longer changes."""

    async def test_starting_freezes_the_flow_and_every_flow_under_it(self) -> None:
        flow, flows = _tree()
        assert len(flows) == 5
        assert not any(f.frozen for f in flows)
        await FlowRunner(LG).start(flow, 1)
        assert all(f.frozen for f in flows)

    @pytest.mark.parametrize("method", sorted(_CHANGES))
    async def test_every_change_to_a_started_flow_raises(self, method: str) -> None:
        flow = (
            make_ff()
            .create("started")
            .call(double)
            .map(lambda b: b.call(double), items=lambda _r, _c: [1])
        )
        await FlowRunner(LG).start(flow, 1)
        with pytest.raises(RuntimeError, match=rf"'started' has been started .*\.{method}\(\)"):
            _CHANGES[method](flow)

    async def test_a_started_flow_runs_again(self) -> None:
        flow = make_ff().create("d").call(double)
        runner = FlowRunner(LG)
        assert await runner.start(flow, 1) == 2
        assert await runner.start(flow, 2) == 4

    async def test_a_run_rejected_at_start_leaves_the_flow_open(self) -> None:
        flow = make_ff().create("d").call(double)
        with pytest.raises(RuntimeError, match="but no history"):
            FlowRunner(LG).start(flow, 1, start=Resume())
        assert not flow.frozen
        flow.call(double)

    async def test_an_open_flow_can_call_a_started_one(self) -> None:
        started = make_ff().create("inner").call(double)
        await FlowRunner(LG).start(started, 1)
        outer = make_ff().create("outer").call(started).then(double)
        assert await FlowRunner(LG).start(outer, 1) == 4
