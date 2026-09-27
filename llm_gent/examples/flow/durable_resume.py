#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run: --smoke

"""Durable pause / resume across process restarts with real inference.

Verifies Flow's checkpoint + mid-SAIA-turn pause + resume story
end-to-end. Structurally exercises every save site: iterate
boundary, halt-observation mid-turn, ``paused_turn`` trace_ref on
the CAS commit, :class:`JsonFileCheckpointStore` persistence, and
:meth:`Flow.run(resume="replay")` hydration on a subsequent process.

Backend modes:

- **Real, OpenAI-compatible** (``--base-url URL``, e.g. a local
  server): ``llm_infer.client.Factory(lg).openai()``. Model is
  ``--model`` or the first one the endpoint lists.
- **Real, Anthropic** (default, requires ``ANTHROPIC_API_KEY``):
  ``llm_infer.client.Factory(lg).anthropic()``.
- Either real :class:`~llm_infer.client.LLMClient` is wrapped as
  a ``saia.Backend`` by :class:`SAIAAdapter`.
- **Smoke** (``--smoke``): :class:`_FakeBackend` returns scripted
  tool-call responses and cooperates with ``abort_signal``. No
  network; CI-safe.

The mechanism the demo verifies is the same in both modes: the
first tool call sets a shared :class:`asyncio.Event` and the
subsequent SAIA iteration's LLM call catches it via
``abort_signal``, returns ``TaskResult(paused=True)``, and Flow
captures the conversation onto the CAS halt commit for a later
``resume="replay"`` to re-arm.

Layout
------

- :class:`Digest` — the :class:`~llm_gent.flow.StateDataclass`
  the framework serializes at every iterate boundary.
- :func:`_make_tool_executor` — closure over the halt event; the
  first successful ``lookup_reference`` call flips it.
- :class:`_FakeBackend` — scripted ``saia.Backend`` for
  ``--smoke``; also cooperates with ``abort_signal``.
- :func:`_build_real_backend` — OpenAI-compatible vs Anthropic
  client, per ``--base-url``.
- :class:`_SAIAFactory` — gent :class:`~llm_gent.flow.SAIAFactory`
  binding a role to a SAIA built from the shared backend +
  per-run tool executor.
- :func:`_make_summarize` — builds the role-bound SAIA-backed
  verb run inside ``.iterate(until=<queue empty>)``; each
  completed iteration pops one topic and appends its summary, a
  paused one leaves state untouched for resume.
- :func:`_invoke` — one invocation: always ``run(resume="replay")``,
  halt armed only when that run will start fresh
  (:func:`_resume_pending`).
- :func:`main` — real mode runs one :func:`_invoke` per process
  against a fixed on-disk store; ``--smoke`` runs both phases in
  one process against a temp store and fails on a broken
  round-trip.

Running
-------

Real inference — two separate processes::

    python -m llm_gent.examples.flow.durable_resume \
        --base-url http://localhost:18300   # halts mid-turn
    python -m llm_gent.examples.flow.durable_resume \
        --base-url http://localhost:18300   # resumes, finishes

Omit ``--base-url`` to use Anthropic (``ANTHROPIC_API_KEY`` in
the environment; ``--model`` optional).

A third invocation starts a fresh cycle; ``--reset`` wipes the
store first.

Smoke (no network, both phases in one invocation)::

    python -m llm_gent.examples.flow.durable_resume --smoke

See ``durable_resume.md`` next to this file for a walk through a real
run and the on-disk store it leaves.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import asyncio
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from appinfra.log import Logger, quick_console_logger
from llm_infer.client import Factory as LLMInferFactory
from llm_infer.client.saia import SAIAAdapter
from llm_kelt.conversation import ConversationFactory as KeltConversationFactory
from llm_saia import SAIA, ChatResponse, Message, ToolCall, ToolDef
from llm_saia.core.backend import Backend
from llm_saia.core.errors import PauseRequested

from llm_gent.flow import (
    Context,
    Flow,
    FlowFactory,
    History,
    Loop,
    Role,
    StateDataclass,
    TypeStateFactory,
    verb,
)
from llm_gent.flow.stores import JsonFileCheckpointStore


TOPICS: tuple[str, ...] = ("content-addressed storage", "async cancellation")
"""Terms the flow summarizes — two iterations exercise a boundary between them."""

STORE_DIR = Path.home() / ".cache" / "llm-gent-durable-resume"
"""On-disk store path — stable across process invocations so run 2 finds run 1's commit."""

CLIENT_FLOW_ID = "durable-resume-demo"
"""Client flow id naming this example's history — the resume path reads back commits under it."""

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
"""Cheap, tool-capable Anthropic model. Override with ``--model``."""


@dataclass
class Digest(StateDataclass):
    """Typed state — queue of pending topics + accumulated summaries.

    ``pending`` shrinks as the flow progresses; ``summaries``
    grows. On resume both round-trip through :meth:`from_dict` so
    the second process invocation sees exactly the mid-run state
    the halt commit captured.
    """

    pending: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)


SEARCH_TOOL = ToolDef(
    name="lookup_reference",
    description=(
        "Return a short reference blurb for a technical term. Call this "
        "once with the term you want to summarize before writing the summary."
    ),
    parameters={
        "type": "object",
        "properties": {"term": {"type": "string", "description": "The term to look up."}},
        "required": ["term"],
    },
)

DONE_TOOL = ToolDef(
    name="submit_summary",
    description="Submit the final one-sentence summary. Call this last.",
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "One-sentence summary."}},
        "required": ["summary"],
    },
)


def _make_tool_executor(
    halt_event: asyncio.Event, arm_halt: bool
) -> Callable[[str, dict[str, Any]], Awaitable[str]]:
    """Return a SAIA tool executor closed over ``halt_event``.

    ``arm_halt=True`` (the fresh invocation): the FIRST successful
    ``lookup_reference`` returns its blurb AND flips
    ``halt_event``. SAIA's next iteration's ``Backend.chat`` sees
    the event via ``abort_signal`` and raises :class:`PauseRequested`
    — SAIA returns ``TaskResult(paused=True)``, Loop stashes the
    conversation onto ``env.pending_paused_turns``, and the halt-
    observation site writes a CAS commit whose ``trace_ref`` slot
    holds the paused-turn payload.

    ``arm_halt=False`` (resume invocation): the tool call is
    already in the persisted conversation, so this executor is
    typically not reinvoked. If SAIA does call again (model asks
    for another tool round), no halt fires and the turn completes.
    """
    fired = False

    async def execute(name: str, args: dict[str, Any]) -> str:
        nonlocal fired
        if name == "lookup_reference":
            term = args.get("term", "<missing>")
            print(f"  tool: lookup_reference({term!r})")
            result = f"'{term}' is a foundational concept in modern systems software."
            if arm_halt and not fired:
                fired = True
                halt_event.set()
            return result
        if name == "submit_summary":
            return "acknowledged"
        return f"unknown tool: {name}"

    return execute


class _FakeBackend(Backend):
    """Scripted :class:`saia.Backend` for ``--smoke``: no network.

    Two-iteration script matching what the model would produce:

    1. Last message is a user prompt → return a tool_call requesting
       ``lookup_reference(term=<derived-from-task>)``.
    2. Last message is a tool_result → return a tool_call for the
       terminal ``submit_summary`` with a canned summary.

    Cooperates with ``abort_signal`` — checks it at entry and raises
    :class:`PauseRequested`, matching the real
    :class:`SAIAAdapter._chat_with_abort` contract.

    :attr:`lookups` counts ``lookup_reference`` calls issued — the
    smoke check uses it to tell a resumed turn from a restarted one.
    """

    def __init__(self) -> None:
        self._call = 0
        self.lookups = 0

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
        if abort_signal is not None and abort_signal.is_set():
            raise PauseRequested()
        self._call += 1
        last = messages[-1] if messages else None
        if last is not None and last.role == "tool":
            summary = "A concise concept from systems research."
            return self._tool_call("submit_summary", {"summary": summary})
        term = self._extract_term(last.content if last else "")
        self.lookups += 1
        return self._tool_call("lookup_reference", {"term": term})

    def _tool_call(self, name: str, arguments: dict[str, Any]) -> ChatResponse:
        """Build a single-tool-call response, as a tool-using model would return."""
        return ChatResponse(
            content="",
            tool_calls=[ToolCall(id=f"fake-{self._call}", name=name, arguments=arguments)],
            finish_reason="tool_use",
            input_tokens=0,
            output_tokens=0,
            model="fake",
        )

    @staticmethod
    def _extract_term(text: str) -> str:
        marker = "term: "
        idx = text.lower().find(marker)
        return text[idx + len(marker) :].strip() if idx >= 0 else "unknown"


def _build_real_backend(lg: Logger, base_url: str | None, model: str | None) -> Backend:
    """Wrap a real :class:`~llm_infer.client.LLMClient` as a ``saia.Backend``.

    ``base_url`` set → OpenAI-compatible endpoint (e.g. a local
    server); otherwise Anthropic, keyed by ``ANTHROPIC_API_KEY``.
    """
    factory = LLMInferFactory(lg)
    if base_url is not None:
        # llm_infer appends endpoint paths (/models, /chat/completions) to
        # base_url verbatim, so a bare server root would 404.
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        model = model or _first_served_model(factory, base_url)
        print(f"  model: {model} @ {base_url}")
        client = factory.openai(base_url=base_url, default_model=model)
    else:
        client = factory.anthropic(default_model=model or ANTHROPIC_MODEL)
    return SAIAAdapter(client)


def _first_served_model(factory: LLMInferFactory, base_url: str) -> str:
    """Return the first model the endpoint lists on ``/v1/models``."""
    with factory.openai(base_url=base_url) as probe:
        models = probe.backend.list_models()
    if not models:
        raise RuntimeError(f"{base_url} lists no models; pass --model")
    return models[0]


SUMMARIZE_ROLE = Role(
    name="summarizer",
    backend="runtime",
    model="runtime",
    style=(
        "You are a terse technical writer. For each term, call "
        "`lookup_reference` exactly once, read the returned blurb, then "
        "call `submit_summary` with a one-sentence definition. "
        "Do not call any tool more than once."
    ),
)
"""Role identity + system prompt. ``_SAIAFactory`` reads only ``style``;
the backend and model are chosen at runtime by ``--base-url`` / ``--model``."""


class _SAIAFactory:
    """Gent :class:`~llm_gent.flow.SAIAFactory` — one backend, per-Loop executor.

    A single :class:`_SAIAFactory` is captured on the
    :class:`FlowFactory`; :meth:`build` fires once per role during
    Flow assembly. The backend is shared; each build gets a fresh
    tool executor closure so the halt-arm state (see
    :func:`_make_tool_executor`) is per-Loop, not global.
    """

    def __init__(self, lg: Logger, backend: Backend, halt: asyncio.Event, arm_halt: bool) -> None:
        self._lg = lg
        self._backend = backend
        self._halt = halt
        self._arm_halt = arm_halt

    def build(self, role: Role) -> SAIA:
        executor = _make_tool_executor(self._halt, self._arm_halt)
        builder = (
            SAIA.builder()
            .backend(self._backend)
            .logger(self._lg)
            .max_iterations(6)
            .tools([SEARCH_TOOL, DONE_TOOL], executor)
            # Default confirmation mode nudges "call it again to confirm",
            # which contradicts the role's call-each-tool-once instruction.
            .terminal_tool("submit_summary", require_confirmation=False)
        )
        if role.style:
            builder = builder.system(role.style)
        return builder.build()


def _make_summarize(loop: Loop) -> Callable[[Context[Digest]], Awaitable[Digest]]:
    """Return the role-bound ``summarize`` verb, closed over ``loop``.

    Role-bound so the flow's :class:`SAIAFactory` populates
    ``ctx.saia`` for the :class:`Loop` to dispatch through.
    """

    @verb(role=SUMMARIZE_ROLE)
    async def summarize(ctx: Context[Digest]) -> Digest:
        """Run SAIA against ``pending[0]``; on completion pop it and append the summary.

        The :class:`Loop` bridges ``ctx.halt`` into SAIA's
        ``abort_signal``. On a fresh invocation the tool executor
        sets it after ``lookup_reference`` returns; SAIA's next
        :meth:`Backend.chat` aborts and the Loop returns a paused
        result. On resume the Loop re-arms the saved conversation
        (tool result already in place) and SAIA completes the
        follow-up call.

        Returns ``ctx.data`` so :meth:`Flow.run` hands the caller
        the live (possibly resume-hydrated) state.
        """
        if ctx.data.pending:
            term = ctx.data.pending[0]
            result = await loop(ctx, task=f"Summarize the following. term: {term}")
            _record(ctx.data, term, result)
        return ctx.data

    return summarize


def _record(data: Digest, term: str, result: Any) -> None:
    """Drain ``term`` into ``data`` on a completed turn; leave ``data`` untouched on a paused one.

    Paused: the halt commit snapshots state after the verb returns,
    and resume re-dispatches with ``pending[0]`` still the halted
    term — matching the task the saved conversation carries.
    Mutating here would checkpoint a half-applied state.
    """
    if getattr(result, "paused", False):
        print(f"  paused mid-turn on {term!r}")
        return
    summary = _extract_summary(result)
    data.pending.pop(0)
    data.summaries.append(summary)
    print(f"  summarized {term!r} -> {summary!r}")


def _extract_summary(result: Any) -> str:
    """Pull the terminal tool's summary field out of a SAIA :class:`TaskResult`.

    ``terminal_data`` holds the parsed args of the terminal tool
    call (``submit_summary``). Falls back to ``.output`` when the
    model finished without calling the terminal tool.
    """
    terminal = getattr(result, "terminal_data", None)
    if isinstance(terminal, dict) and "summary" in terminal:
        return str(terminal["summary"])
    return str(getattr(result, "output", "") or "")


def _build_flow(lg: Logger, ff: FlowFactory, halt: asyncio.Event) -> Flow:
    """Assemble the demo flow: iterate over topics, summarize each.

    A single :meth:`Flow.iterate` drains the ``pending`` queue
    with a body that ``.call``s the ``summarize`` verb, which
    dispatches through a :class:`Loop` bound to
    :data:`SUMMARIZE_ROLE`. The Loop's
    :class:`~llm_kelt.conversation.ConversationFactory` is what
    lets the halt-observation site round-trip the paused SAIA
    conversation through the CAS commit.

    Termination is state-driven (``until`` on an empty queue).
    ``max_iters`` is only a safety bound, and it counts
    cumulatively across resumes — the paused pass consumes one
    iteration without draining a topic, hence ``+ 1``.
    """
    conv_factory = KeltConversationFactory(lg)
    summarize = _make_summarize(Loop(SUMMARIZE_ROLE, conversation_factory=conv_factory))
    flow = ff.create(
        "durable-resume-demo",
        state=Digest(pending=list(TOPICS)),
        halt=halt,
        client_flow_id=CLIENT_FLOW_ID,
    )
    flow.iterate(
        lambda body: body.call(summarize),
        until=lambda _result, ctx: not ctx.data.pending,
        max_iters=len(TOPICS) + 1,
    )
    return flow


async def _resume_pending(history: History) -> bool:
    """True when ``run(resume="replay")`` will resume rather than start fresh.

    Same rule as :meth:`Resume.hydrate`: resume from the head unless the
    history is empty or complete (the head is the final-state commit the
    default ``retain`` policy writes on clean exit). This flow writes no
    ``ok`` iterate commits, so a pending resume here is always a halt.
    """
    return await history.head() is not None and not await history.is_complete()


async def _paused_turn_saved(history: History) -> bool:
    """True when the head carries a ``paused_turn`` trace ref (the paused conversation)."""
    head = await history.head()
    return head is not None and any(r.kind == "paused_turn" for r in head.meta.trace_ref)


async def _invoke(lg: Logger, store_dir: Path, backend: Backend, mode: str) -> tuple[Digest, bool]:
    """One process-level invocation; return ``(final state, halted)``.

    Always runs with ``resume="replay"`` and lets the framework pick
    the path: empty or complete history → fresh run, any other
    head → resume. The halt is armed only on a fresh run so the
    resumed turn completes.
    """
    store = JsonFileCheckpointStore(lg, store_dir)
    history = History(store, CLIENT_FLOW_ID)
    resuming = await _resume_pending(history)
    halt = asyncio.Event()
    ff = FlowFactory(
        lg,
        saia_factory=_SAIAFactory(lg, backend, halt, arm_halt=not resuming),
        state_factory=TypeStateFactory(Digest),
        checkpointer=store,
    )
    print(f"--- Run ({'resume' if resuming else 'fresh'}, {mode}) ---")
    print(f"  store: {store_dir}")
    final: Digest = await _build_flow(lg, ff, halt).run(resume="replay")
    halted = await _resume_pending(history)
    await _report(history, store_dir, final, halted)
    return final, halted


async def _report(history: History, store_dir: Path, final: Digest, halted: bool) -> None:
    """Print the post-run state and the ref files the store holds."""
    refs = sorted(str(p.relative_to(store_dir)) for p in store_dir.rglob("*.json"))
    print(f"  pending: {final.pending}")
    print(f"  summaries: {final.summaries}")
    print(f"  ref files: {refs}")
    if halted:
        saved = (
            "paused_turn saved"
            if await _paused_turn_saved(history)
            else "NO paused_turn on halt commit"
        )
        print(f"  halted mid-turn ({saved}) — invoke again to resume")
    else:
        print("  complete — final state committed and tagged; next invocation starts fresh")


async def _run_smoke(lg: Logger) -> int:
    """Both phases in one process against a temp store; non-zero exit on a broken round-trip.

    Each phase builds a fresh store, factory, backend and flow, so
    the only thing carried from phase 1 to phase 2 is what landed on
    disk.
    """
    store_dir = Path(tempfile.mkdtemp(prefix="gent-example-durable-resume-"))
    resumed_backend = _FakeBackend()
    try:
        first, halted = await _invoke(lg, store_dir, _FakeBackend(), "smoke")
        turn_saved = await _paused_turn_saved(
            History(JsonFileCheckpointStore(lg, store_dir), CLIENT_FLOW_ID)
        )
        print()
        final, still_halted = await _invoke(lg, store_dir, resumed_backend, "smoke")
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)
    # One lookup in phase 2 = topic 2 only; the halted topic-1 turn resumed past
    # its tool call instead of restarting from the task.
    checks = {
        "halted with state untouched": halted and first.pending == list(TOPICS),
        "halt commit carries paused_turn": turn_saved,
        "resume continued the paused turn": resumed_backend.lookups == len(TOPICS) - 1,
        "full drain": not still_halted
        and not final.pending
        and len(final.summaries) == len(TOPICS)
        and all(final.summaries),
    }
    for name, passed in checks.items():
        if not passed:
            print(f"SMOKE FAILED: {name}", file=sys.stderr)
    return 0 if all(checks.values()) else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fake backend (no network); runs halt + resume in one process on a temp store.",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete the on-disk store before running."
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible server (e.g. http://localhost:18300; /v1 appended if missing). "
        "Omitted → Anthropic via ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--model",
        help="Model id. Default: first model the --base-url endpoint lists, "
        f"else {ANTHROPIC_MODEL}.",
    )
    return parser.parse_args(argv)


async def main() -> int:
    args = _parse_args(sys.argv[1:])
    lg = quick_console_logger("durable-resume", config={"level": "warning"})
    if args.smoke:
        return await _run_smoke(lg)
    if not args.base_url and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Pass --base-url, set ANTHROPIC_API_KEY, or use --smoke.", file=sys.stderr)
        return 1
    if args.reset:
        shutil.rmtree(STORE_DIR, ignore_errors=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    backend = _build_real_backend(lg, args.base_url, args.model)
    await _invoke(lg, STORE_DIR, backend, "real")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
