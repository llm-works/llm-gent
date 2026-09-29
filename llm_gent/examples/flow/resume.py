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
   halted step runs again, then the remaining passes.

``max_iters=5`` is the cumulative bound across resumes — run 1 does 3
iterations, run 2 does 2, total 5. On a natural completion the checkpoint
is deleted; on halt / cancel / exception it is preserved so a subsequent
resume can pick up.

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

from llm_gent.flow import Context, Flow, FlowFactory, StateDataclass, TypeStateFactory, verb
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
    ``state_factory=TypeStateFactory(Counter)`` on the :class:`~llm_gent.flow.FlowFactory` so
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

    A separate step from :func:`tick` because the step during which the
    halt arrives runs again on resume: this one only passes its input
    through, so running it twice changes nothing.
    """
    if count == HALT_AFTER and ctx.halt is not None:
        print(f"  halt fired at count={HALT_AFTER}")
        ctx.halt.set()
    return count


def _build_flow(
    ff: FlowFactory,
    store: JsonFileCheckpointStore,
    client_flow_id: str,
    *,
    halt: bool,
) -> Flow:
    """Assemble a Flow wired to the shared store, with a halt event when ``halt``.

    Each call returns a fresh :class:`~llm_gent.flow.Flow` pointing at the
    same checkpoint history. Only run 1 gets a halt event: run 2 resumes at
    the step that halted, and without an event that step passes through.
    The halt and checkpointer bindings ride on :meth:`FlowFactory.create`
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


async def main() -> int:
    """Run the demo end-to-end, printing state progression."""
    lg = quick_console_logger("resume-example", config={"level": "warning"})
    tmp_root = Path(tempfile.mkdtemp(prefix="gent-example-resume-"))
    try:
        store = JsonFileCheckpointStore(lg, tmp_root)
        client_flow_id = "resume-demo"
        ff = FlowFactory(lg, state_factory=TypeStateFactory(Counter))

        print(f"--- Run 1: fresh start, halts at count={HALT_AFTER} ---")
        flow1 = _build_flow(ff, store, client_flow_id, halt=True)
        result1 = await flow1.run()
        print(f"run 1 returned: count={result1}")
        print(f"checkpoint on disk: {sorted(p.name for p in tmp_root.rglob('*.json'))}")

        print(f"\n--- Run 2: resume=latest (cumulative max_iters={MAX_ITERS}) ---")
        flow2 = _build_flow(ff, store, client_flow_id, halt=False)
        result2 = await flow2.run(resume="latest")
        print(f"run 2 returned: count={result2}")
        print(f"checkpoint on disk: {sorted(p.name for p in tmp_root.rglob('*.json'))}")
        print("(empty after run 2 because natural completion deletes the checkpoint)")

        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
