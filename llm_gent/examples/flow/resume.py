#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""Halt-and-resume demo on a counter loop.

Exercises the Flow-level checkpointer end-to-end without touching an LLM:

1. **Run 1** — a verb increments a typed-state counter each iteration and
   sets ``ctx.halt`` when ``count == HALT_AFTER`` to request a cooperative halt.
   The framework saves at every iterate boundary, preserves the checkpoint
   on halt exit, and returns.
2. **Run 2** — a fresh :class:`~llm_gent.flow.Flow` (new halt event, same
   checkpointer + client_flow_id) is called with ``resume=True``. The
   framework loads the latest checkpoint, reconstructs ``state.data`` via
   :meth:`Counter.from_dict`, restores the iteration counter, and runs
   the remaining passes.

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

from llm_gent.flow import Context, Flow, FlowFactory, StateDataclass, verb
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
    ``state_type=Counter`` on the :class:`~llm_gent.flow.FlowFactory` so
    the framework calls :meth:`from_dict` on ``run(resume=True)`` to
    reconstruct an instance from the checkpoint payload.
    """

    count: int = 0
    log: list[int] = field(default_factory=list)


@verb
async def tick(ctx: Context[Counter]) -> int:
    """Increment the counter and simulate a crash via ``ctx.halt``.

    Pure-Python verb (``@verb`` bare form) — no :class:`Role`, no
    ``ctx.saia`` access. The framework's signature-aware dispatch
    drops the iterate chain's previous value rather than requiring a
    placeholder ``_prev`` parameter. Mutates ``ctx.data`` in
    place (typed as :class:`Counter` via the :class:`Context`
    parameterization) and returns the new count so the
    :meth:`Flow.iterate` loop threads it as the next iteration's
    input.
    """
    ctx.data.count += 1
    ctx.data.log.append(ctx.data.count)
    print(f"  tick: count={ctx.data.count} log={ctx.data.log}")
    if ctx.data.count == HALT_AFTER and ctx.halt is not None:
        print(f"  halt fired at count={HALT_AFTER}")
        ctx.halt.set()
    return ctx.data.count


def _build_flow(
    ff: FlowFactory,
    store: JsonFileCheckpointStore,
    client_flow_id: str,
) -> Flow:
    """Assemble a Flow wired to the shared store with its own halt event.

    Each call returns a fresh :class:`~llm_gent.flow.Flow`, so the two
    demo runs can operate on independent halt events while pointing at
    the same checkpoint trajectory. The halt and checkpointer bindings
    ride on :meth:`FlowFactory.create` kwargs so this reads as a single
    construction step rather than a chain of ``.with_*`` setters.
    """
    flow = ff.create(
        "resume-demo",
        state=Counter(),
        halt=asyncio.Event(),
        checkpointer=(store, client_flow_id),
    )
    flow.iterate(lambda body: body.call(tick), max_iters=MAX_ITERS)
    return flow


async def main() -> int:
    """Run the demo end-to-end, printing state progression."""
    lg = quick_console_logger("resume-example", config={"level": "warning"})
    tmp_root = Path(tempfile.mkdtemp(prefix="gent-example-resume-"))
    try:
        store = JsonFileCheckpointStore(lg, tmp_root)
        client_flow_id = "resume-demo"
        ff = FlowFactory(lg, state_type=Counter)

        print(f"--- Run 1: fresh start, halts at count={HALT_AFTER} ---")
        flow1 = _build_flow(ff, store, client_flow_id)
        result1 = await flow1.run()
        print(f"run 1 returned: count={result1}")
        print(f"checkpoint on disk: {sorted(p.name for p in tmp_root.rglob('*.json'))}")

        print(f"\n--- Run 2: resume=True (cumulative max_iters={MAX_ITERS}) ---")
        flow2 = _build_flow(ff, store, client_flow_id)
        result2 = await flow2.run(resume=True)
        print(f"run 2 returned: count={result2}")
        print(f"checkpoint on disk: {sorted(p.name for p in tmp_root.rglob('*.json'))}")
        print("(empty after run 2 because natural completion deletes the checkpoint)")

        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
