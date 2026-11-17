#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run: --smoke

"""Version control of model calls made without Flow.

An agent researches a topic in two calls, and commits both by hand to an
:class:`appinfra.cas` repository with :class:`~llm_gent.provenance.VersionControl`:

1. a plain llm-infer call, committed with ``act.llm_call(messages, response)``.
   Nothing here needs llm-infer: any message list and any response value
   with a JSON form (a raw HTTP call's JSON included) commits the same way;
2. a SAIA task, committed through its ``on_iteration`` callback
   (``act.llm_calls(conversation)``): one commit per backend call.

Both are inside one activity (``vc.activity("research")``), which commits
itself last, with the topic in and the summary out. The example then prints
the records and what the repository holds: each call's context is a vector of
its messages, so the messages two calls share are stored once.

Backend modes:

- **Real, OpenAI-compatible** (``--base-url URL``) or **Anthropic**
  (default, requires ``ANTHROPIC_API_KEY``); ``--model`` optional.
- **Smoke** (``--smoke``): llm-infer's scripted client, no network; CI-safe.

Running::

    python -m llm_gent.examples.provenance.explicit_calls --base-url http://localhost:18300
    python -m llm_gent.examples.provenance.explicit_calls --smoke
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
import os

from appinfra.cas import MemoryStore, Vector
from appinfra.log import Logger, quick_console_logger
from llm_infer.client import LLMClient
from llm_infer.client.saia import SAIAAdapter
from llm_saia import ListConversation

from llm_gent.examples._clients import example_client, parse_client_args
from llm_gent.examples.provenance._shared import print_records, summary_of, writer_saia
from llm_gent.provenance import Activity, Record, VersionControl


TOPIC = "content-addressed storage"
REPO = "research-agent"


async def research(lg: Logger, vc: VersionControl, client: LLMClient, topic: str) -> str:
    """Research ``topic``: one plain call, then a SAIA task, all inside one activity."""
    with vc.activity("research", inputs={"topic": topic}) as act:
        ideas = await _key_ideas(act, client, topic)
        summary = await _summarize(act, lg, client, topic, ideas)
        act.outputs["summary"] = summary
    return summary


async def _key_ideas(act: Activity, client: LLMClient, topic: str) -> str:
    """A plain call, committed by hand: the messages sent and the response."""
    messages = [{"role": "user", "content": f"Name three key ideas of {topic}, one line each."}]
    response = await client.chat_async(messages)
    act.llm_call(messages, response)
    return response.content


async def _summarize(act: Activity, lg: Logger, client: LLMClient, topic: str, notes: str) -> str:
    """A SAIA task on the same client: each call committed through ``on_iteration``."""
    conversation = ListConversation()
    result = await writer_saia(lg, SAIAAdapter(client), notes).complete(
        f"Write a one-sentence summary of {topic}.",
        conversation=conversation,
        on_iteration=act.llm_calls(conversation, role="writer"),
    )
    return summary_of(result)


def _smoke_checks(vc: VersionControl, records: list[Record]) -> dict[str, bool]:
    """What ``--smoke`` verifies."""
    root, calls = records[-1], records[:-1]
    first = vc.load(calls[0].outputs["response"])
    saia_contexts = [list(Vector(vc.repo, c.inputs["context"].id)) for c in calls[1:]]
    return {
        "three calls, then the activity": [r.kind for r in records]
        == ["llm_call", "llm_call", "llm_call", "research"],
        "every call is inside the activity": all(
            c.meta["parent"] == root.meta["activity"] for c in calls
        ),
        "the activity ended ok": root.meta["outcome"] == "ok",
        "a response is stored without its request": "request" not in first,
        "SAIA's calls share their messages": saia_contexts[1][: len(saia_contexts[0])]
        == saia_contexts[0],
    }


async def _run(lg: Logger, client: LLMClient) -> tuple[VersionControl, list[Record]]:
    """Research :data:`TOPIC` with version control on a fresh in-memory store."""
    store = MemoryStore(lg)
    vc = VersionControl(lg, store.repo(REPO))
    summary = await research(lg, vc, client, TOPIC)
    print(f"  summary: {summary}")
    return vc, print_records(vc, store, REPO)


async def main() -> int:
    args = parse_client_args(__doc__.splitlines()[0], sys.argv[1:])
    lg = quick_console_logger("explicit-calls", config={"level": "warning"})
    if not args.smoke and not args.base_url and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Pass --base-url, set ANTHROPIC_API_KEY, or use --smoke.", file=sys.stderr)
        return 1
    client = example_client(lg, smoke=args.smoke, base_url=args.base_url, model=args.model)
    vc, records = await _run(lg, client)
    if not args.smoke:
        return 0
    failed = [name for name, ok in _smoke_checks(vc, records).items() if not ok]
    for name in failed:
        print(f"SMOKE FAILED: {name}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
