#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""Multi-topic research agent — .map(state=, merge=) + state.root() + synthesizer.

Shape:

1. **decompose** — a :func:`planner` archetype verb reads
   ``ctx.data.question``, calls :meth:`complete_structured(prompt,
   TopicList)`, and returns the list of topic names. Also pins
   ``ctx.data.topics`` so a resumed run can inspect what was planned.

2. **.map(body, state=, merge=, state_type=)** — fan out one sub-flow
   per topic. Each item's body sees its own projected
   :class:`TopicState` on ``ctx.state``, so concurrent per-topic
   scratch spaces (snippets, working summary) don't collide.

   - ``state=lambda _parent: TopicState()`` — the projection runs once
     per item but only receives ``parent_state``; the item (topic name)
     is threaded to the body via signature-aware dispatch and the body
     writes it into ``ctx.data.topic`` on entry. That is the correct
     idiom: the projection creates the shape, the body stamps identity.

   - ``merge=merge_topic`` — after each item's body returns, fold its
     :class:`TopicSummary` into ``parent.topic_summaries`` keyed by the
     topic name the body stamped.

   - ``state_type=TopicState`` — makes the framework hydrate child
     state as :class:`TopicState` on any future checkpoint resume;
     harmless without a checkpointer.

3. **body(gather → summarize)** — each per-topic sub-flow runs two
   role-bound verbs against the projected :class:`TopicState`. The
   researcher reads the outer question via ``ctx.state.root().data`` —
   the canonical way for a projected sub-flow to reach run-wide state
   that its projected slice does not carry. The typed-cast helper
   :func:`_outer` narrows ``.root().data`` (typed :data:`Any`) back to
   :class:`ResearchState` at the call site.

4. **synthesize** — a :func:`synthesizer` archetype verb reads
   ``ctx.data.topic_summaries`` (populated by merge) and emits the
   final :class:`Report`. The map's own return value (a list of
   :class:`TopicSummary`) is ignored; the merged view is the load-
   bearing one.

Not covered here: **checkpointer across sub-flow boundaries** — worth
its own example (interrupt mid-map, resume; requires a two-run harness
like :mod:`~llm_gent.examples.flow.resume`). Left for a follow-up.

Run standalone::

    python -m llm_gent.examples.flow.subflow_research
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
from dataclasses import dataclass, field
from typing import Any

from appinfra.log import quick_console_logger
from pydantic import BaseModel, ConfigDict, Field

from llm_gent.examples.flow._infra import StructuredSAIA, StructuredStubSAIAFactory
from llm_gent.flow import (
    Context,
    FlowFactory,
    Role,
    StateDataclass,
    planner,
    synthesizer,
    verb,
)


# ── schemas ──────────────────────────────────────────────────────────


class TopicList(BaseModel):
    """Decomposer output — the list of topics the researcher will investigate."""

    model_config = ConfigDict(extra="forbid")

    topics: list[str] = Field(min_length=2, max_length=4)


class Snippet(BaseModel):
    """One evidence fragment gathered for a topic."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source: str


class SnippetList(BaseModel):
    """Researcher output — the snippets gathered for one topic."""

    model_config = ConfigDict(extra="forbid")

    items: list[Snippet]


class TopicSummary(BaseModel):
    """Per-topic condensed view — key points plus a bottom-line verdict."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    key_points: list[str]
    verdict: str


class Report(BaseModel):
    """Synthesizer output — the composed answer to the outer question."""

    model_config = ConfigDict(extra="forbid")

    question: str
    verdict: str
    sections: dict[str, str]


# ── roles ────────────────────────────────────────────────────────────


DECOMPOSER = Role(name="decomposer", backend="stub", model="decomposer-model")
"""Splits the outer question into a list of topics."""

RESEARCHER = Role(name="researcher", backend="stub", model="researcher-model")
"""Gathers per-topic evidence snippets (bound per-item under the projected state)."""

SUMMARIZER = Role(name="summarizer", backend="stub", model="summarizer-model")
"""Condenses gathered snippets into a :class:`TopicSummary` per topic."""

SYNTHESIZER = Role(name="synthesizer", backend="stub", model="synthesizer-model")
"""Composes the final :class:`Report` from the merged per-topic summaries."""


# ── states ───────────────────────────────────────────────────────────


@dataclass
class ResearchState(StateDataclass):
    """Outer, run-wide state.

    :attr:`topic_summaries` is filled by :func:`merge_topic` as each map
    item completes; :attr:`final_report` is set by :func:`synthesize`.
    """

    question: str
    topics: list[str] = field(default_factory=list)
    topic_summaries: dict[str, TopicSummary] = field(default_factory=dict)
    final_report: Report | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "topics": list(self.topics),
            "topic_summaries": {
                k: v.model_dump(mode="json") for k, v in self.topic_summaries.items()
            },
            "final_report": self.final_report.model_dump(mode="json")
            if self.final_report
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchState:
        report_data = data.get("final_report")
        return cls(
            question=data["question"],
            topics=list(data.get("topics", [])),
            topic_summaries={
                k: TopicSummary.model_validate(v)
                for k, v in (data.get("topic_summaries") or {}).items()
            },
            final_report=Report.model_validate(report_data) if report_data else None,
        )


@dataclass
class TopicState(StateDataclass):
    """Per-item projected scratch space for one topic's sub-flow.

    :attr:`topic` starts empty because :data:`StateProject` receives
    only ``parent_state`` — the item (topic name) is threaded through
    the body's positional arg, not the projection. :func:`gather`
    stamps :attr:`topic` on entry so :func:`merge_topic` knows which
    key to fold under.
    """

    topic: str = ""
    snippets: list[Snippet] = field(default_factory=list)
    summary: TopicSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "snippets": [s.model_dump(mode="json") for s in self.snippets],
            "summary": self.summary.model_dump(mode="json") if self.summary else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicState:
        summary_data = data.get("summary")
        return cls(
            topic=data.get("topic", ""),
            snippets=[Snippet.model_validate(s) for s in data.get("snippets", [])],
            summary=TopicSummary.model_validate(summary_data) if summary_data else None,
        )


# ── outer-state cast helper ─────────────────────────────────────────


def _outer(ctx: Context[TopicState]) -> ResearchState:
    """Return the run-wide :class:`ResearchState` from a projected sub-flow verb.

    ``ctx.state.root().data`` is typed :data:`Any` because a child's
    payload type has no static relationship to its ancestors'. Verbs
    that need to read the outer state pass through this helper to get a
    narrowed handle without repeating the ``isinstance`` cast at every
    call site — the file-scoped helper pattern the framework docs
    recommend for narrowing :attr:`Context.saia` and :attr:`state.root`
    alike.
    """
    root = ctx.state.root().data
    assert isinstance(root, ResearchState), f"unexpected root type {type(root).__name__}"
    return root


# ── verbs ────────────────────────────────────────────────────────────


@planner(DECOMPOSER)
async def decompose(ctx: Context[ResearchState]) -> list[str]:
    """Decompose the outer question into a list of topics.

    Returns the list so :meth:`.map` (with no explicit ``items=``)
    picks it up as its item iterable, and pins it into state so a
    resumed run can inspect what was planned.
    """
    saia: StructuredSAIA = ctx.saia
    prompt = f"Decompose this research question into 2-4 focused topics: {ctx.data.question!r}"
    result = await saia.complete_structured(prompt, TopicList)
    ctx.data.topics = list(result.value.topics)
    print(f"[decompose] topics: {ctx.data.topics}")
    return ctx.data.topics


@verb(role=RESEARCHER)
async def gather(ctx: Context[TopicState], topic: str) -> list[Snippet]:
    """Gather evidence snippets for one topic under its projected state.

    Stamps ``ctx.data.topic`` from the map's item on entry so
    :func:`merge_topic` can find it after this scope closes. Reads the
    outer question via :func:`_outer` — a projected sub-flow reaches
    run-wide state through ``ctx.state.root()``, not its projected
    slice.
    """
    ctx.data.topic = topic
    question = _outer(ctx).question
    saia: StructuredSAIA = ctx.saia
    prompt = (
        f"Research question: {question!r}.\n"
        f"For the topic {topic!r}, produce 3 short evidence snippets, each with a source."
    )
    result = await saia.complete_structured(prompt, SnippetList)
    ctx.data.snippets = list(result.value.items)
    print(f"[gather/{topic}] {len(ctx.data.snippets)} snippets")
    return ctx.data.snippets


@verb(role=SUMMARIZER)
async def summarize(ctx: Context[TopicState], snippets: list[Snippet]) -> TopicSummary:
    """Condense the topic's snippets into a :class:`TopicSummary`.

    Pins the summary into ``ctx.data.summary`` so
    :func:`merge_topic` can fold it into the outer state; also returns
    it as the body's final result, which lands as the map's per-item
    output in list form.
    """
    saia: StructuredSAIA = ctx.saia
    joined = "\n".join(f"- {s.text} ({s.source})" for s in snippets)
    prompt = (
        f"Summarize these snippets on {ctx.data.topic!r} into a "
        f"TopicSummary with key_points and a verdict:\n{joined}"
    )
    result = await saia.complete_structured(prompt, TopicSummary)
    ctx.data.summary = result.value
    print(f"[summarize/{ctx.data.topic}] verdict={result.value.verdict!r}")
    return result.value


@synthesizer(SYNTHESIZER)
async def synthesize(ctx: Context[ResearchState], _prev: Any) -> Report:
    """Compose the final :class:`Report` from the merged per-topic summaries.

    Reads ``ctx.data.topic_summaries`` (populated by
    :func:`merge_topic`), not ``_prev`` — the map's list is redundant
    with the merged dict, and the dict preserves the topic-name keys.
    """
    saia: StructuredSAIA = ctx.saia
    joined = "\n\n".join(
        f"## {name}\nverdict: {s.verdict}\nkey_points:\n"
        + "\n".join(f"- {kp}" for kp in s.key_points)
        for name, s in ctx.data.topic_summaries.items()
    )
    prompt = (
        f"Compose a final Report answering {ctx.data.question!r}. "
        f"Provide an overall verdict and one paragraph per topic.\n\n{joined}"
    )
    result = await saia.complete_structured(prompt, Report)
    ctx.data.final_report = result.value
    return result.value


# ── merge fn ─────────────────────────────────────────────────────────


def merge_topic(parent: ResearchState, child: TopicState) -> None:
    """Fold one topic's summary into ``parent.topic_summaries``.

    Runs once per map item, after that item's body returns
    successfully. ``child.topic`` was stamped by :func:`gather` on
    entry; sync callback so concurrent merges are preemption-safe.
    """
    if child.summary is not None:
        parent.topic_summaries[child.topic] = child.summary


# ── demo scripts ─────────────────────────────────────────────────────


_DEMO_QUESTION = "What are the trade-offs of Python vs Rust for backend web services?"

_DEMO_TOPICS = TopicList(topics=["performance", "developer experience", "ecosystem"])

_DEMO_SNIPPETS: dict[str, SnippetList] = {
    "performance": SnippetList(
        items=[
            Snippet(
                text="Rust services routinely handle 10x the RPS of Python at similar CPU cost.",
                source="benchmark-a",
            ),
            Snippet(
                text="Python's GIL caps single-process concurrency; async helps I/O but not CPU.",
                source="python-docs",
            ),
            Snippet(
                text="Rust's zero-cost abstractions keep latency tails tight under load.",
                source="benchmark-b",
            ),
        ]
    ),
    "developer experience": SnippetList(
        items=[
            Snippet(
                text="Python's REPL and dynamic typing shorten the write-test loop.",
                source="survey-2025",
            ),
            Snippet(
                text="Rust's borrow checker frontloads correctness work at compile time.",
                source="rust-book",
            ),
            Snippet(
                text="Framework maturity (FastAPI, Django) means less scaffolding in Python.",
                source="framework-review",
            ),
        ]
    ),
    "ecosystem": SnippetList(
        items=[
            Snippet(
                text="Python's PyPI has an order of magnitude more web libraries than crates.io.",
                source="registry-stats",
            ),
            Snippet(
                text="Rust's ecosystem is younger but rapidly maturing (axum, tokio, sqlx).",
                source="ecosystem-report",
            ),
            Snippet(
                text="Interop with C/ML tooling is broadly better in Python.",
                source="ml-integration",
            ),
        ]
    ),
}

_DEMO_TOPIC_SUMMARIES: dict[str, TopicSummary] = {
    "performance": TopicSummary(
        topic="performance",
        key_points=[
            "Rust ~10x RPS at similar CPU",
            "Python GIL caps CPU concurrency",
            "Rust tail latency stays flat under load",
        ],
        verdict="Rust wins on raw throughput and tail latency.",
    ),
    "developer experience": TopicSummary(
        topic="developer experience",
        key_points=[
            "Python's write-test loop is tighter",
            "Rust frontloads correctness at compile time",
            "Python frameworks are more mature",
        ],
        verdict="Python is faster to iterate; Rust catches more bugs earlier.",
    ),
    "ecosystem": TopicSummary(
        topic="ecosystem",
        key_points=[
            "PyPI dwarfs crates.io for web libs",
            "Rust web stack (axum/tokio/sqlx) is maturing fast",
            "Python has broader ML/C interop",
        ],
        verdict="Python's ecosystem is deeper today; Rust's is closing the gap.",
    ),
}

_DEMO_REPORT = Report(
    question=_DEMO_QUESTION,
    verdict=(
        "Rust wins on runtime performance; Python wins on iteration speed and ecosystem breadth. "
        "Pick Rust for latency-critical services, Python for services that ship features fast."
    ),
    sections={
        "performance": "Rust delivers substantially higher throughput and tighter tail latency.",
        "developer experience": "Python iterates faster; Rust's compiler catches bugs earlier.",
        "ecosystem": "Python's web + ML ecosystem is broader today; Rust's is maturing rapidly.",
    },
)


def _demo_scripts() -> dict[str, list[tuple[type, Any]]]:
    """Assemble ``{role_name: [(schema, value), ...]}`` for one end-to-end run."""
    return {
        "decomposer": [(TopicList, _DEMO_TOPICS)],
        "researcher": [(SnippetList, _DEMO_SNIPPETS[t]) for t in _DEMO_TOPICS.topics],
        "summarizer": [(TopicSummary, _DEMO_TOPIC_SUMMARIES[t]) for t in _DEMO_TOPICS.topics],
        "synthesizer": [(Report, _DEMO_REPORT)],
    }


# ── main ─────────────────────────────────────────────────────────────


def _print_report(state: ResearchState) -> None:
    """Render the final report + per-topic breakdown."""
    print("\n=== research complete ===")
    r = state.final_report
    assert r is not None, "synthesize should have set final_report"
    print(f"question: {r.question}")
    print(f"verdict:  {r.verdict}\n")
    for topic, section in r.sections.items():
        print(f"[{topic}] {section}")


async def main() -> int:
    """Run the multi-topic research flow on the canned question."""
    lg = quick_console_logger("subflow-research-example", config={"level": "warning"})
    saia_f = StructuredStubSAIAFactory(_demo_scripts())
    ff = FlowFactory(lg, saia_f=saia_f, state_type=ResearchState)

    flow = ff.create("subflow-research", state=ResearchState(question=_DEMO_QUESTION))
    (
        flow.call(decompose)
        .map(
            lambda body: body.call(gather).call(summarize),
            state=lambda _parent: TopicState(),
            merge=merge_topic,
            state_type=TopicState,
        )
        .call(synthesize)
    )

    print(f"Question: {_DEMO_QUESTION}\n")
    await flow.run()
    _print_report(flow.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
