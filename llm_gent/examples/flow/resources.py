#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""A resource of the app's own, with a fluent method of its own: ``flow.with_stats(...)``.

``Stats`` tallies what a run's steps record, and rolls each tally up to the
``Stats`` it came from. It implements the :class:`~llm_gent.flow.Resource`
protocol: ``snapshot()`` / ``restore(data)`` put its tallies in the run's
checkpoints and bring them back on resume, and the optional ``child()``
gives each run a ``Stats`` of its own.

``MyFlow`` adds ``with_stats`` with :func:`~llm_gent.flow.resource_method`,
and :class:`~llm_gent.flow.Factory` builds it (``flow_class=``), so the
method is typed — and the subflows a map body builds are ``MyFlow`` too.

1. **Run 1** — a map scores three documents; each item runs with its own
   child ``Stats`` (``with_stats(per_item=True)``). Once every item has
   been fetched, the halt stops the run before any is scored.
2. **Run 2** — a fresh flow and fresh ``Stats`` resume it: the top-level
   tallies and each item's come back, and the items finish.

Run standalone::

    python -m llm_gent.examples.flow.resources
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
from collections import Counter
from typing import Any

from appinfra.log import quick_console_logger

from llm_gent.flow import (
    HALTED,
    Context,
    Flow,
    Factory,
    Interrupted,
    Resource,
    ResourceKey,
    resource_method,
    verb,
)
from llm_gent.flow.stores import InMemoryCheckpointStore


DOCS = ["alpha", "beta", "gamma"]


class Stats(Resource):
    """Tallies by name; a child's tallies roll up to its parent."""

    def __init__(self, parent: Stats | None = None) -> None:
        self.tallies: Counter[str] = Counter()
        self.parent = parent

    def record(self, name: str, n: int = 1) -> None:
        """Add ``n`` to ``name`` here and in every parent."""
        self.tallies[name] += n
        if self.parent is not None:
            self.parent.record(name, n)

    def snapshot(self) -> dict[str, Any]:
        """The tallies, for the run's checkpoints."""
        return {"tallies": dict(self.tallies)}

    def restore(self, data: dict[str, Any]) -> None:
        """Set the tallies back from a checkpoint."""
        self.tallies = Counter(data["tallies"])

    def child(self, per_item: bool = True) -> Stats:
        """A ``Stats`` for one run, reporting to this one."""
        return Stats(self)


STATS = ResourceKey[Stats]("stats")


class MyFlow(Flow):
    """A flow with ``with_stats(...)``: ``with_resource(STATS, ...)``, typed."""

    with_stats = resource_method(STATS)


def build(ff: Factory[MyFlow], stats: Stats, halt: asyncio.Event, arm: bool) -> MyFlow:
    """Fetch every document, then score each; armed, halt once all are fetched."""
    fetched: set[str] = set()

    @verb
    async def fetch(ctx: Context[Any], doc: str) -> str:
        ctx.resource(STATS).record("fetched")
        fetched.add(doc)
        if arm and len(fetched) == len(DOCS):
            halt.set()
        return doc

    @verb
    async def score(ctx: Context[Any], doc: str) -> int:
        await asyncio.sleep(0)  # lets a set halt reach the item
        if ctx.halt is not None and ctx.halt.is_set():
            raise Interrupted()
        ctx.resource(STATS).record("scored")
        return len(doc)

    def per_document(body: MyFlow) -> None:
        body.with_stats(per_item=True).call(fetch).then(score)

    return (
        ff.create("score-docs", client_flow_id="docs")
        .with_checkpointer()
        .with_halt(halt)
        .with_stats(stats)
        .map(per_document, items=lambda _prev, _ctx: DOCS)
    )


async def main() -> int:
    """Run 1 halts mid-map; run 2 resumes it."""
    lg = quick_console_logger("resources-example", config={"level": "warning"})
    store = InMemoryCheckpointStore()
    ff = Factory(lg, flow_class=MyFlow, checkpoint_store=store)

    print("--- Run 1: halts once every document is fetched ---")
    stats1 = Stats()
    assert await build(ff, stats1, asyncio.Event(), arm=True).run() is HALTED
    print(f"halted with tallies {dict(stats1.tallies)}")

    print("\n--- Run 2: fresh flow and Stats, resume='latest' ---")
    stats2 = Stats()
    scores = await build(ff, stats2, asyncio.Event(), arm=False).run(resume="latest")
    print(f"scores {scores}, tallies {dict(stats2.tallies)}")
    assert stats2.tallies == Counter(fetched=len(DOCS), scored=len(DOCS))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
