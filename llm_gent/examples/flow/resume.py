#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""Halt-and-resume demo on a counter loop.

Exercises the Flow-level checkpointer end-to-end without touching an LLM:

1. **Run 1** — a verb increments a typed-state counter each iteration and
   the next step sets ``ctx.halt`` when ``count == HALT_AFTER`` to request
   a cooperative halt. The framework writes a halt checkpoint at that step,
   preserves it on halt exit, and returns.
2. **Run 2** — a fresh :class:`~llm_gent.flow.Flow` (new halt event, same
   checkpointer + client_flow_id) is called with ``resume="latest"``. The
   framework checks out the latest checkpoint, reconstructs ``state.data``
   via :meth:`Counter.from_dict`, and continues where run 1 stopped: the
   next pass runs, then any remaining passes.

``max_iters=5`` is the cumulative bound across resumes — run 1 does 3
iterations, run 2 does 2, total 5. The history is kept: run 1 ends it with
a halt commit, run 2 with the final-state commit tagged ``complete`` (the
store's default ``retain`` retention; ``gc_on_success`` deletes the
history on a natural completion instead).

Run standalone::

    python -m llm_gent.examples.flow.resume
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
import shutil
import tempfile
from dataclasses import dataclass, field

from appinfra.log import quick_console_logger

from llm_gent.flow import (
    HALTED,
    Context,
    Flow,
    Factory,
    History,
    StateDataclass,
    TypeStateFactory,
    verb,
)
from llm_gent.flow.stores import JsonFileCheckpointStore


HALT_AFTER = 3
"""Iteration count at which run 1 signals halt from within the verb.

The check is ``count == HALT_AFTER`` (equality, not ``>=``) so run 2 —
which starts at ``count = HALT_AFTER`` after resume — never re-triggers.
"""

MAX_ITERS = 5
"""Cumulative iterate bound across resumes. Run 1 stops at HALT_AFTER; run 2 runs the rest."""


@dataclass
class Counter(StateDataclass):
    """Typed state payload — a running count and its per-iteration trace.

    Inherits :class:`~llm_gent.flow.StateDataclass` for ``to_dict`` /
    ``from_dict`` — flat dataclass, no override needed. Bound as
    ``state_factory=TypeStateFactory(Counter)`` on the :class:`~llm_gent.flow.Factory` so
    the framework calls :meth:`from_dict` on ``run(resume="latest")`` to
    reconstruct an instance from the checkpoint payload.
    """

    count: int = 0
    log: list[int] = field(default_factory=list)


@verb
async def tick(ctx: Context[Counter]) -> int:
    """Increment the counter.

    Pure-Python verb (``@verb`` bare form) — no :class:`Role`, no
    ``ctx.saia`` access. The framework's signature-aware dispatch
    drops the iterate chain's previous value rather than requiring a
    placeholder ``_prev`` parameter. Mutates ``ctx.data`` in
    place (typed as :class:`Counter` via the :class:`Context`
    parameterization) and returns the new count.
    """
    ctx.data.count += 1
    ctx.data.log.append(ctx.data.count)
    print(f"  tick: count={ctx.data.count} log={ctx.data.log}")
    return ctx.data.count


@verb
async def stop_at_limit(ctx: Context[Counter], count: int) -> int:
    """Simulate a crash via ``ctx.halt`` once the count reaches ``HALT_AFTER``.

    A separate step from :func:`tick` so the halt fires after the count
    mutation completes. The cursor moves to the next pass once this step
    returns, so resume continues from there.
    """
    if count == HALT_AFTER and ctx.halt is not None:
        print(f"  halt fired at count={HALT_AFTER}")
        ctx.halt.set()
    return count


def _build_flow(
    ff: Factory[Flow],
    store: JsonFileCheckpointStore,
    client_flow_id: str,
    *,
    halt: bool,
) -> Flow:
    """Assemble a Flow wired to the shared store, with a halt event when ``halt``.

    Each call returns a fresh :class:`~llm_gent.flow.Flow` pointing at the
    same checkpoint history. Only run 1 gets a halt event: run 2 resumes at
    the step that halted, and without an event that step passes through.
    The halt and checkpointer bindings ride on :meth:`Factory.create`
    kwargs so this reads as a single construction step rather than a chain
    of ``.with_*`` setters.
    """
    flow = ff.create(
        "resume-demo",
        state=Counter(),
        halt=asyncio.Event() if halt else None,
        checkpointer=(store, client_flow_id),
    )
    flow.iterate(lambda body: body.call(tick).call(stop_at_limit), max_iters=MAX_ITERS)
    return flow


async def _describe_head(history: History) -> str:
    """The head commit's outcome and the count its state holds."""
    head = await history.head()
    if head is None:
        return "none"
    count = (await history.snapshot(head)).root["count"]
    done = "complete" if await history.is_complete() else "resumable"
    return f"{head.meta.outcome} at count={count} ({done})"


async def main() -> int:
    """Run the demo end-to-end, printing state progression."""
    lg = quick_console_logger("resume-example", config={"level": "warning"})
    tmp_root = Path(tempfile.mkdtemp(prefix="gent-example-resume-"))
    try:
        store = JsonFileCheckpointStore(lg, tmp_root)
        client_flow_id = "resume-demo"
        ff = Factory(lg, state_factory=TypeStateFactory(Counter))

        print(f"--- Run 1: fresh start, halts at count={HALT_AFTER} ---")
        flow1 = _build_flow(ff, store, client_flow_id, halt=True)
        result1 = await flow1.run()
        history = History(store, client_flow_id)
        assert result1 is HALTED
        print(f"run 1 returned: {result1} (its state is in the halt checkpoint)")
        print(f"history head: {await _describe_head(history)}")

        print(f"\n--- Run 2: resume=latest (cumulative max_iters={MAX_ITERS}) ---")
        flow2 = _build_flow(ff, store, client_flow_id, halt=False)
        result2 = await flow2.run(resume="latest")
        print(f"run 2 returned: count={result2}")
        print(f"history head: {await _describe_head(history)}")

        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
