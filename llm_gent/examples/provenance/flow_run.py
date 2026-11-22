#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run: --smoke

"""Version control of a Flow run, with an approval gate that pauses it.

A three-step flow (``write`` → ``review`` → ``publish``) runs with two hooks
attached (:meth:`Flow.with_hooks`):

- :class:`~llm_gent.provenance.VersionControl` commits a record of every step,
  every model call the ``write`` step's Loop makes, every checkpoint and every
  run, next to the run's checkpoints in the same repository.
- ``EditorApproval`` is governance, not provenance: its ``before_step`` keeps
  ``publish`` from running until an editor approved it. Without approval it
  returns :class:`~llm_gent.flow.Pause`, which halts the run at a checkpoint.
  With approval it allows the step and records the approval with
  ``vc.commit(..., within=event.ctx)``, inside the ``publish`` step.

The first run halts before ``publish``. The second resumes from the halt
checkpoint with the approval in hand; only ``publish`` runs again, and its
``before_step`` asks again: permission is decided at resume, never restored
from a record. The run's record names the checkpoint it resumed from.

Backend modes:

- **Real, OpenAI-compatible** (``--base-url URL``) or **Anthropic**
  (default, requires ``ANTHROPIC_API_KEY``); ``--model`` optional.
- **Smoke** (``--smoke``): llm-infer's scripted client, no network; CI-safe.

Running::

    python -m llm_gent.examples.provenance.flow_run --base-url http://localhost:18300
    python -m llm_gent.examples.provenance.flow_run --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

from appinfra.cas import MemoryStore, Store
from appinfra.log import Logger, quick_console_logger
from llm_infer.client.saia import SAIAAdapter
from llm_kelt.conversation import ConversationFactory as KeltConversationFactory
from llm_saia import SAIA
from llm_saia.core.backend import Backend

from llm_gent.examples._clients import example_client, parse_client_args
from llm_gent.examples.provenance._shared import print_records, summary_of, writer_saia
from llm_gent.flow import HALTED, Context, Factory, Flow, Loop, Pause, Role, verb
from llm_gent.flow.hooks import Hook, StepEvent
from llm_gent.provenance import Actor, Record, VersionControl


TOPIC = "content-addressed storage"
REPO = "publish-summary"
NOTES = "objects named by hash; immutable history; free deduplication"

WRITER = Role(name="writer", backend="runtime", model="runtime")
"""The role the ``write`` step's Loop runs as; the backend is chosen by the command line."""


class EditorApproval(Hook):
    """``publish`` runs only once an editor approved it; the approval is recorded.

    Deciding is governance; the record of the decision is provenance.
    """

    def __init__(self, vc: VersionControl, approved_by: str | None) -> None:
        """Gate ``publish`` on ``approved_by``; record approvals with ``vc``."""
        self._vc = vc
        self._approved_by = approved_by

    def before_step(self, event: StepEvent) -> bool | Pause:
        """Pause ``publish`` without an approval; with one, record it and allow."""
        if event.where.step_name != "publish":
            return True
        if self._approved_by is None:
            return Pause("publish needs an editor's approval")
        self._vc.commit(
            "approval",
            inputs={"text": event.args[0]},
            actor=Actor(role="editor"),
            facets={"approved_by": self._approved_by, "policy": "publish-v1"},
            within=event.ctx,
        )
        return True


class _WriterFactory:
    """The flow's SAIA factory: the writer's SAIA on one backend."""

    def __init__(self, lg: Logger, backend: Backend) -> None:
        self._lg = lg
        self._backend = backend

    def build(self, role: Role) -> SAIA:
        """The writer's SAIA, whatever the role."""
        return writer_saia(self._lg, self._backend, NOTES)


def _make_write(loop: Loop) -> Callable[[Context[Any], str], Awaitable[str]]:
    """The ``write`` verb: a writer task through ``loop``."""

    @verb(role=WRITER)
    async def write(ctx: Context[Any], topic: str) -> str:
        return summary_of(await loop(ctx, task=f"Write a one-sentence summary of {topic}."))

    return write


@verb
async def review(ctx: Context[Any], text: str) -> str:
    """Reject an empty or overlong summary."""
    if not text or len(text.split()) > 60:
        raise ValueError(f"summary fails review: {text!r}")
    return text


@verb
async def publish(ctx: Context[Any], text: str) -> str:
    """Publish the summary."""
    return f"published: {text}"


def _flow(lg: Logger, store: Store, backend: Backend, hooks: tuple[Hook, ...]) -> Flow:
    """The flow, checkpointing to ``REPO`` in ``store``; run it with a halt, for a gate's pause."""
    loop = Loop(WRITER, conversation_factory=KeltConversationFactory(lg))
    ff = Factory(lg, saia_factory=_WriterFactory(lg, backend))
    return (
        ff.create("publish-summary", state={})
        .with_checkpoint_store(store, REPO)
        .call(_make_write(loop), name="write")
        .then(review, name="review")
        .then(publish, name="publish")
        .with_hooks(*hooks)
    )


async def _run(lg: Logger, backend: Backend) -> tuple[Any, Any, VersionControl, MemoryStore]:
    """Run until the gate pauses, then resume with the approval."""
    store = MemoryStore(lg)
    vc = VersionControl(lg, store.repo(REPO))
    flow = _flow(lg, store, backend, (EditorApproval(vc, None), vc))
    first = await flow.run(TOPIC, halt=asyncio.Event())
    print(f"--- run 1: {'halted, waiting for approval' if first is HALTED else first}")
    gate = EditorApproval(vc, approved_by="editor-1")
    second = await _flow(lg, store, backend, (gate, vc)).run(halt=asyncio.Event(), resume="latest")
    print(f"--- run 2 (resumed, approved): {second}")
    return first, second, vc, store


def _smoke_checks(
    first: Any, second: Any, store: MemoryStore, records: list[Record]
) -> dict[str, bool]:
    """What ``--smoke`` verifies."""
    by_kind = [(r.kind, (r.meta.get("where") or {}).get("step_name")) for r in records]
    runs = [r for r in records if r.kind == "run"]
    checkpoints = [r for r in records if r.kind == "checkpoint"]
    approval = next(r for r in records if r.kind == "approval")
    publish_ok = next(
        r
        for r in records
        if r.kind == "step"
        and r.meta["where"]["step_name"] == "publish"
        and r.meta["outcome"] == "ok"
    )
    return {
        "run 1 halted, run 2 published": first is HALTED and str(second).startswith("published"),
        "the gate paused publish": ("step", "publish") in by_kind
        and any(r.meta.get("outcome") == "paused" for r in records),
        "only publish ran again": [s for k, s in by_kind if k == "step"].count("write") == 1,
        "the approval is inside the publish step": approval.meta["parent"]
        == publish_ok.meta["activity"],
        "run 2 names the halt checkpoint": runs[1].meta["resumed_from"]
        == checkpoints[0].meta["checkpoint"],
        "records sit next to the checkpoints": set(store.repo(REPO).refs())
        == {"HEAD", "tags/complete", "provenance"},
    }


async def main() -> int:
    args = parse_client_args(__doc__.splitlines()[0], sys.argv[1:])
    lg = quick_console_logger("flow-run", config={"level": "warning"})
    if not args.smoke and not args.base_url and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Pass --base-url, set ANTHROPIC_API_KEY, or use --smoke.", file=sys.stderr)
        return 1
    client = example_client(lg, smoke=args.smoke, base_url=args.base_url, model=args.model)
    first, second, vc, store = await _run(lg, SAIAAdapter(client))
    records = print_records(vc, store, REPO)
    if not args.smoke:
        return 0
    failed = [name for name, ok in _smoke_checks(first, second, store, records).items() if not ok]
    for name in failed:
        print(f"SMOKE FAILED: {name}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
