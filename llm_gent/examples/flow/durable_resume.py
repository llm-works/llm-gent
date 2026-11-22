#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run: --smoke

"""Durable pause / resume across process restarts with real inference.

Verifies Flow's checkpoint + mid-SAIA-turn pause + resume story
end-to-end. Structurally exercises every save site: iterate
boundary, halt-observation mid-turn, the paused turn in the CAS
commit's snapshot, :class:`appinfra.cas.FileStore` persistence, and
:meth:`Flow.run(resume="latest")` checkout on a subsequent process.

Backend modes:

- **Real, OpenAI-compatible** (``--base-url URL``, e.g. a local
  server): ``llm_infer.client.Factory(lg).openai()``. Model is
  ``--model`` or the first one the endpoint lists.
- **Real, Anthropic** (default, requires ``ANTHROPIC_API_KEY``):
  ``llm_infer.client.Factory(lg).anthropic()``.
- Either real :class:`~llm_infer.client.LLMClient` is wrapped as
  a ``saia.Backend`` by :class:`SAIAAdapter`.
- **Smoke** (``--smoke``): llm-infer's scripted client, wrapped by
  :class:`SAIAAdapter` like a real one: it calls ``lookup_reference``
  on a turn's first call and ``submit_summary`` after. No network;
  CI-safe.

The mechanism the demo verifies is the same in both modes: the
first tool call sets a shared :class:`asyncio.Event` and the
subsequent SAIA iteration's LLM call catches it via
``abort_signal``, returns ``TaskResult(paused=True)``, and Flow
captures the conversation onto the CAS halt commit for a later
``resume="latest"`` to re-arm.

Layout
------

- :class:`Digest` — the :class:`~llm_gent.flow.StateDataclass`
  the framework serializes at every iterate boundary.
- :func:`_make_tool_executor` — closure over the halt event; the
  first successful ``lookup_reference`` call flips it.
- :func:`~llm_gent.examples._clients.real_client` — OpenAI-compatible
  vs Anthropic client, per ``--base-url``.
- :class:`_SAIAFactory` — gent :class:`~llm_gent.flow.SAIAFactory`
  binding a role to a SAIA built from the shared backend +
  per-run tool executor.
- :func:`_make_summarize` — builds the role-bound SAIA-backed
  verb run inside ``.iterate(until=<queue empty>)``; each
  completed iteration pops one topic and appends its summary, a
  paused one leaves state untouched for resume.
- :func:`_invoke` — one invocation: ``run(resume="latest")`` when the
  last run halted (:func:`_resume_pending`), otherwise a fresh run with
  the halt armed.
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

from appinfra.cas import FileStore
from appinfra.log import Logger, quick_console_logger
from llm_infer.client import Factory as LLMInferFactory
from llm_infer.client import LLMClient, ScriptedBackend
from llm_infer.client.saia import SAIAAdapter
from llm_kelt.conversation import ConversationFactory as KeltConversationFactory
from llm_saia import SAIA, ToolDef
from llm_saia.core.backend import Backend

from llm_gent.examples._clients import ANTHROPIC_MODEL, real_client
from llm_gent.flow import (
    HALTED,
    Context,
    Flow,
    Factory,
    History,
    Loop,
    Role,
    StateDataclass,
    TypeStateFactory,
    verb,
)


TOPICS: tuple[str, ...] = ("content-addressed storage", "async cancellation")
"""Terms the flow summarizes — two iterations exercise a boundary between them."""

STORE_DIR = Path.home() / ".cache" / "llm-gent-durable-resume"
"""On-disk store path — stable across process invocations so run 2 finds run 1's commit."""

CLIENT_FLOW_ID = "durable-resume-demo"
"""Client flow id naming this example's history — the resume path reads back commits under it."""


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
    — SAIA returns ``TaskResult(paused=True)``, Loop keeps the
    task and conversation in the run's snapshots at its call's
    path, and once the run has stopped it writes a CAS commit whose
    snapshot holds that paused turn.

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


def _fresh_turns(client: LLMClient) -> int:
    """Requests the scripted ``client`` got with no tool call yet in their conversation.

    Each is a turn started from its task; a resumed turn's requests
    already carry the tool call made before the halt.
    """
    backend = client.backend
    assert isinstance(backend, ScriptedBackend)
    return sum(
        1
        for request in backend.requests
        if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in request.messages)
    )


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
    flow :class:`~llm_gent.flow.Factory`; :meth:`build` fires once per role during
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
    Applying the step's work here (popping the term, appending a
    summary) would checkpoint a half-applied step; recording the
    turn's own bookkeeping would not.
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


def _build_flow(lg: Logger, ff: Factory[Flow]) -> Flow:
    """Assemble the demo flow: iterate over topics, summarize each.

    A single :meth:`Flow.iterate` drains the ``pending`` queue
    with a body that ``.call``s the ``summarize`` verb, which
    dispatches through a :class:`Loop` bound to
    :data:`SUMMARIZE_ROLE`. The Loop's
    :class:`~llm_kelt.conversation.ConversationFactory` is what
    lets the paused SAIA conversation round-trip through the halt
    commit.

    Termination is state-driven (``until`` on an empty queue).
    ``max_iters`` is only a safety bound, counted across resumes: the
    halt checkpoint keeps the iterate in the paused pass, so resume
    finishes that pass instead of starting another.
    """
    conv_factory = KeltConversationFactory(lg)
    summarize = _make_summarize(Loop(SUMMARIZE_ROLE, conversation_factory=conv_factory))
    flow = ff.create("durable-resume-demo")
    flow.iterate(
        lambda body: body.call(summarize),
        until=lambda _result, ctx: not ctx.data.pending,
        max_iters=len(TOPICS),
    )
    return flow


async def _resume_pending(history: History) -> bool:
    """True when the last run halted, so the next run resumes it rather than starting fresh.

    This flow writes no ``ok`` iterate commits: its history's head is
    either a halt commit (a run to resume) or the final-state commit the
    default ``retain`` policy writes on clean exit (a finished run).
    """
    head = await history.head()
    return head is not None and head.meta.outcome == "halted"


async def _paused_turn_saved(history: History) -> bool:
    """True when the head's snapshot holds a Loop's paused turn (the paused conversation)."""
    head = await history.head()
    if head is None:
        return False
    snapshot = await history.snapshot(head)
    return any("turn" in cursor for cursor in snapshot.cursors.values())


async def _invoke(lg: Logger, store_dir: Path, backend: Backend, mode: str) -> tuple[Digest, bool]:
    """One process-level invocation; return ``(final state, halted)``.

    Resumes with ``resume="latest"`` when the last run halted, and starts
    fresh (``resume="off"``) on an empty or finished history. The halt is
    armed only on a fresh run so the resumed turn completes.
    """
    store = FileStore(lg, store_dir)
    history = History(store, CLIENT_FLOW_ID)
    resuming = await _resume_pending(history)
    halt = asyncio.Event()
    ff = Factory(
        lg,
        saia_factory=_SAIAFactory(lg, backend, halt, arm_halt=not resuming),
        state_factory=TypeStateFactory(Digest),
        checkpoint_store=store,
    )
    print(f"--- Run ({'resume' if resuming else 'fresh'}, {mode}) ---")
    print(f"  store: {store_dir}")
    # A resumed run's state comes from the checkpoint; the given one starts a fresh run.
    result = await _build_flow(lg, ff).run(
        state=Digest(pending=list(TOPICS)),
        halt=halt,
        history=CLIENT_FLOW_ID,
        resume="latest" if resuming else "off",
    )
    # A halted run returns HALTED; its state is in the halt checkpoint.
    halted = result is HALTED
    final = await _halted_state(history) if halted else result
    await _report(history, store_dir, final, halted)
    return final, halted


async def _halted_state(history: History) -> Digest:
    """The state the halt checkpoint at the history's head holds."""
    head = await history.head()
    assert head is not None
    return Digest.from_dict((await history.snapshot(head)).root)


async def _report(history: History, store_dir: Path, final: Digest, halted: bool) -> None:
    """Print the post-run state and the ref files the store holds."""
    refs = sorted(str(p.relative_to(store_dir)) for p in store_dir.glob("*/refs/*"))
    print(f"  pending: {final.pending}")
    print(f"  summaries: {final.summaries}")
    print(f"  ref files: {refs}")
    if halted:
        saved = (
            "paused turn saved"
            if await _paused_turn_saved(history)
            else "NO paused turn in the halt commit"
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
    resumed = LLMInferFactory(lg).scripted()
    try:
        fresh = SAIAAdapter(LLMInferFactory(lg).scripted())
        first, halted = await _invoke(lg, store_dir, fresh, "smoke")
        turn_saved = await _paused_turn_saved(History(FileStore(lg, store_dir), CLIENT_FLOW_ID))
        print()
        final, still_halted = await _invoke(lg, store_dir, SAIAAdapter(resumed), "smoke")
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)
    # One fresh turn in phase 2 = topic 2 only; the halted topic-1 turn resumed
    # past its tool call instead of restarting from the task.
    checks = {
        "halted with state untouched": halted and first.pending == list(TOPICS),
        "halt commit holds the paused turn": turn_saved,
        "resume continued the paused turn": _fresh_turns(resumed) == len(TOPICS) - 1,
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
        help="Scripted client (no network); runs halt + resume in one process on a temp store.",
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
    backend = SAIAAdapter(real_client(lg, args.base_url, args.model))
    await _invoke(lg, STORE_DIR, backend, "real")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
