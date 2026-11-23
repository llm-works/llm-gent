# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Concurrent flow runs: throughput and memory per run.

N runs of one small flow start at once on a scripted model that takes time
to answer (:class:`~llm_gent.core.llm.testing.DelayedScriptedBackend`), so
all N are in flight together. The flow: ``plan`` (one model call), then
``iterate`` over ``work`` (1-3 ``Loop`` calls of two model calls each, on a
conversation the run carries in its state) and ``check`` (one call), for a
pass count drawn from the run's seed. Every SAIA and the ``Loop`` are shared
by all runs, unless a flow is built per run.

Each test prints one JSON line; run ``make test.perf.v`` to see them.

- Throughput: wall time, runs and model calls per second, event-loop lag.
  Printed only: a time threshold would only measure the machine.
- Memory: Python allocations per run in flight (``tracemalloc``), as the
  slope between two sizes so the costs every run shares cancel out. In its
  own test because tracing slows the run; bounded at ``max_bytes_per_run``.
- Both check every run: it finished its own passes, its conversation holds
  only its own tasks, and the model saw exactly the calls the runs counted.

:data:`SMALL` (the default, ~5 s) is a regression check that fits ``make
check``; ``LLM_GENT_FLOW_BENCH=full`` runs :data:`FULL` (~1 min).

One event loop is the ceiling of a process: measured at 10,000 and 100,000
runs, a shared flow handles ~2,200-2,800 model calls/s at ~50 KB RSS per
run, linear in runs. More loops on threads of one process add no throughput
under the GIL; more processes do.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import random
import statistics
import time
import tracemalloc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from appinfra.cas import MemoryStore
from appinfra.log import Logger
from llm_infer.client import ChatRequest, ChatResponse, LLMClient, SAIAAdapter
from llm_infer.schemas.openai import FinishReason, FunctionCall, ToolCall
from llm_kelt.conversation import ConversationFactory
from llm_saia import SAIA, ToolDef

from llm_gent.core import Role
from llm_gent.core.llm.testing import DelayedScriptedBackend, uniform_delay
from llm_gent.flow import Context, Factory, Flow, Loop, StateDataclass, TypeStateFactory, verb
from tests.helpers import make_test_logger


pytestmark = pytest.mark.performance


@dataclass(frozen=True)
class Scale:
    """How big a run of these tests is, and what it must show at that size."""

    runs: int  # runs started at once for throughput
    latency: tuple[float, float]  # seconds per model call, drawn uniformly
    memory_runs: tuple[int, int]  # the two sizes the memory slope is taken between
    max_bytes_per_run: int  # ceiling on Python allocations per run in flight


SMALL = Scale(runs=150, latency=(0.001, 0.01), memory_runs=(20, 50), max_bytes_per_run=100_000)
FULL = Scale(runs=1_500, latency=(0.05, 0.5), memory_runs=(200, 500), max_bytes_per_run=100_000)
SCALE = FULL if os.environ.get("LLM_GENT_FLOW_BENCH") == "full" else SMALL
SEED = 7

ROLE = Role(name="worker")
MAX_PASSES = 5
MODEL_CALLS_PER_LOOP = 2
"""A ``Loop`` dispatch: the model calls ``lookup``, then ``submit``."""

ANSWER = " ".join(f"word{i}" for i in range(40))
"""Every answer: 40 words, so a conversation grows by a realistic amount per call."""

LAG_TICK = 0.01
"""Seconds the lag sampler sleeps; lag is how much later than this it wakes."""


@dataclass
class RunState(StateDataclass):
    """One run's state: its identity, its progress and its conversation."""

    run: int
    seed: int
    passes: int
    done: int = 0
    calls: int = 0
    work_calls: int = 0
    conversation: dict[str, Any] = field(default_factory=dict)


LOOKUP = ToolDef(
    name="lookup",
    description="Look up the material for the task.",
    parameters={"type": "object", "properties": {"term": {"type": "string"}}},
)
SUBMIT = ToolDef(
    name="submit",
    description="Submit the result. Call this last.",
    parameters={"type": "object", "properties": {"result": {"type": "string"}}},
)


@dataclass
class Note:
    """What ``plan`` and ``check`` ask the model for."""

    text: str


async def execute(name: str, arguments: dict[str, Any]) -> str:
    """The tools' executor: stateless, shared by every run."""
    return ANSWER


class Model:
    """The scripted model; counts the calls it sees.

    With tools offered, a new task gets ``lookup`` and a tool result gets
    ``submit``: two calls per ``Loop`` dispatch. A request for structured output
    (``plan``, ``check``) gets a :class:`Note` at once.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if not request.tools or (request.extra or {}).get("response_format"):
            note = json.dumps({"text": ANSWER})
            return ChatResponse(content=note, finish_reason=FinishReason.STOP)
        tool = "submit" if request.messages[-1].get("role") == "tool" else "lookup"
        call = ToolCall(
            id=f"call-{self.calls}",
            function=FunctionCall(name=tool, arguments=json.dumps({"result": ANSWER})),
        )
        return ChatResponse(content="", tool_calls=[call], finish_reason=FinishReason.TOOL_CALLS)


class CountingStore(MemoryStore):
    """MemoryStore that counts the objects and bytes the runs write."""

    def __init__(self, lg: Logger) -> None:
        super().__init__(lg)
        self.objects = 0
        self.bytes = 0

    def write_object(self, repo: str, object_id: str, object_type: Any, data: bytes) -> None:
        self.objects += 1
        self.bytes += len(data)
        super().write_object(repo, object_id, object_type, data)


class _SAIAFactory:
    """SAIAFactory: every role gets a SAIA over the one shared client."""

    def __init__(self, lg: Logger, client: LLMClient) -> None:
        self._lg = lg
        self._backend = SAIAAdapter(client)

    def build(self, role: Role) -> SAIA:
        return (
            SAIA.builder()
            .backend(self._backend)
            .logger(self._lg)
            .tools([LOOKUP, SUBMIT], execute)
            .terminal_tool("submit", require_confirmation=False)
            .build()
        )


Step = Callable[[Context[RunState]], Awaitable[None]]


def steps(conversations: ConversationFactory) -> tuple[Step, Step, Step]:
    """The flow's three steps; ``work`` continues the conversation the run carries."""
    loop = Loop(ROLE, conversation_factory=conversations)

    @verb(role=ROLE)
    async def plan(ctx: Context[RunState]) -> None:
        await ctx.saia.complete_structured(f"plan run {ctx.data.run} .", Note)
        ctx.data.calls += 1

    @verb(role=ROLE)
    async def work(ctx: Context[RunState]) -> None:
        data = ctx.data
        conv = conversations.create_from_state(data.conversation) if data.conversation else None
        conv = conv or conversations.create()
        for i in range(_work_calls(data)):
            await loop(ctx, f"run {data.run} . pass {data.done} call {i}", conversation=conv)
            data.calls += MODEL_CALLS_PER_LOOP
            data.work_calls += 1
        data.conversation = conv.to_dict()

    @verb(role=ROLE)
    async def check(ctx: Context[RunState]) -> None:
        prompt = f"check run {ctx.data.run} . pass {ctx.data.done}"
        await ctx.saia.complete_structured(prompt, Note)
        ctx.data.calls += 1
        ctx.data.done += 1

    return plan, work, check


def _work_calls(data: RunState) -> int:
    """1-3 calls for this pass, drawn from the run's seed."""
    return random.Random(f"{data.seed}:{data.run}:{data.done}").randint(1, 3)


def build_flow(ff: Factory[Flow], conversations: ConversationFactory, checkpoint: bool) -> Flow:
    plan, work, check = steps(conversations)
    flow = (
        ff.create("bench")
        .call(plan)
        .iterate(
            lambda body: body.call(work).call(check),
            until=lambda _result, ctx: ctx.data.done >= ctx.data.passes,
            max_iters=MAX_PASSES,
        )
    )
    if checkpoint:
        flow = flow.with_checkpointer().with_checkpoint_policy(on_iterate=True)
    return flow


@dataclass
class Bench:
    """One configuration: what every run shares, and how each run starts."""

    name: str
    model: Model
    ff: Factory[Flow]
    conversations: ConversationFactory
    flow_per_run: bool
    store: CountingStore | None
    shared: Flow | None = None

    def flow(self) -> Flow:
        if self.flow_per_run:
            return build_flow(self.ff, self.conversations, self.store is not None)
        if self.shared is None:
            self.shared = build_flow(self.ff, self.conversations, self.store is not None)
        return self.shared

    async def run(self, state: RunState) -> None:
        if self.store is None:
            await self.flow().run(state=state)
            return
        history = f"run-{state.run}"
        await self.flow().run(state=state, history=history, retention="gc_on_success")


def make_bench(lg: Logger, variant: str) -> Bench:
    """``shared``, ``checkpoint`` (a commit per pass in a MemoryStore) or ``flow_per_run``."""
    model = Model()
    delay = uniform_delay(*SCALE.latency, seed=SEED)
    client = LLMClient(lg, DelayedScriptedBackend(lg, delay, model, record=False))
    store = CountingStore(lg) if variant == "checkpoint" else None
    ff: Factory[Flow] = Factory(
        lg,
        saia_factory=_SAIAFactory(lg, client),
        state_factory=TypeStateFactory(RunState),
        checkpoint_store=store,
    )
    conversations = ConversationFactory(lg)
    return Bench(variant, model, ff, conversations, variant == "flow_per_run", store)


def new_states(runs: int) -> list[RunState]:
    return [
        RunState(run=i, seed=SEED, passes=random.Random(f"{SEED}:{i}").randint(1, MAX_PASSES))
        for i in range(runs)
    ]


async def sample_lag(stop: asyncio.Event, lags: list[float]) -> None:
    """Record how late a ``LAG_TICK`` sleep wakes, until ``stop`` is set."""
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        start = loop.time()
        await asyncio.sleep(LAG_TICK)
        lags.append(loop.time() - start - LAG_TICK)


def assert_runs(bench: Bench, states: list[RunState]) -> None:
    """Each run finished its own passes on its own conversation; the model saw every call."""
    assert [s.run for s in states if s.done != s.passes] == []
    assert [s.run for s in states if not _own_conversation(s, bench.conversations)] == []
    assert sum(s.calls for s in states) == bench.model.calls


def _own_conversation(state: RunState, conversations: ConversationFactory) -> bool:
    """One task per work call, every task from this run."""
    messages = conversations.create_from_state(state.conversation).as_messages()
    mine = f"run {state.run} ."
    tasks = [m.content for m in messages if m.role == "user"]
    return len(tasks) == state.work_calls and all(mine in str(t) for t in tasks)


def _store_figures(bench: Bench) -> dict[str, Any]:
    if bench.store is None:
        return {}
    return {"store_objects": bench.store.objects, "store_mib": round(bench.store.bytes / 2**20, 1)}


VARIANTS = ["shared", "checkpoint", "flow_per_run"]


@pytest.mark.parametrize("variant", VARIANTS)
async def test_throughput(variant: str) -> None:
    bench = make_bench(make_test_logger(), variant)
    states = new_states(SCALE.runs)
    lags: list[float] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_lag(stop, lags))
    start = time.perf_counter()
    await asyncio.gather(*(bench.run(s) for s in states))
    wall = time.perf_counter() - start
    stop.set()
    await sampler
    assert_runs(bench, states)
    lag_ms = sorted(x * 1000 for x in lags) or [0.0]
    figures = {
        "test": "throughput",
        "variant": variant,
        "runs": SCALE.runs,
        "latency_s": SCALE.latency,
        "wall_s": round(wall, 2),
        "runs_per_s": round(SCALE.runs / wall, 1),
        "calls_per_s": round(bench.model.calls / wall),
        "lag_median_ms": round(statistics.median(lag_ms), 2),
        "lag_p99_ms": round(lag_ms[int(0.99 * (len(lag_ms) - 1))], 2),
        "lag_max_ms": round(lag_ms[-1], 2),
    }
    print(json.dumps(figures | _store_figures(bench)))


async def traced_peak(variant: str, runs: int) -> int:
    """Peak Python allocations while ``runs`` runs are in flight, over the start."""
    bench = make_bench(make_test_logger(), variant)
    gc.collect()
    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        states = new_states(runs)
        await asyncio.gather(*(bench.run(s) for s in states))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert_runs(bench, states)
    return peak - base


@pytest.mark.parametrize("variant", VARIANTS)
async def test_memory_per_run(variant: str) -> None:
    low, high = SCALE.memory_runs
    per_run = (await traced_peak(variant, high) - await traced_peak(variant, low)) / (high - low)
    figures = {"test": "memory", "variant": variant, "runs": [low, high]}
    print(json.dumps(figures | {"bytes_per_run": round(per_run)}))
    assert 0 < per_run < SCALE.max_bytes_per_run
