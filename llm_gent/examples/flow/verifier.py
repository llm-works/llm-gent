#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""Multi-model consensus verifier.

Shape:

1. **Answer** — the primary LLM answers a user query. The answer lands in
   ``ctx.state.data.answer``.
2. **Verify loop** (bounded to ``MAX_ROUNDS``, exits on consensus):

   a. Primary self-reviews its current answer.
   b. Verifier LLM (a different model) reviews the same answer
      independently.
   c. Judge LLM decides whether the two reviews semantically agree.
   d. If reviews disagree, primary corrects its answer given both
      reviews; loop repeats.

Roles map to three distinct saia instances (:data:`PRIMARY`,
:data:`VERIFIER`, :data:`JUDGE`), each built by
:class:`~._infra.StubSAIAFactory` with a scripted response queue so
the example is deterministic and runs without a backend. Swap the stub
factory for a real :class:`~llm_gent.flow.SAIAFactory` to run against
production models.

Run standalone::

    python -m llm_gent.examples.flow.verifier
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
from dataclasses import dataclass

from appinfra.log import quick_console_logger

from llm_gent.examples.flow._infra import ExampleSAIA, StubSAIAFactory
from llm_gent.flow import Context, FlowFactory, Role, StateDataclass, verb


MAX_ROUNDS = 5
"""Iterate bound — the loop exits at consensus or after this many rounds."""

PRIMARY = Role(name="primary", backend="stub", model="primary-model")
"""Answers the query, self-reviews, corrects on disagreement."""

VERIFIER = Role(name="verifier", backend="stub", model="verifier-model")
"""Independent second-pass reviewer of the primary's answer."""

JUDGE = Role(name="judge", backend="stub", model="judge-model")
"""Decides whether the two reviews semantically agree."""


@dataclass
class VerifierState(StateDataclass):
    """Typed state for the verifier flow.

    Inherits :class:`~llm_gent.flow.StateDataclass` for ``to_dict`` /
    ``from_dict`` — flat dataclass, no override needed. Bound as
    ``state_type=VerifierState`` so the flow can be checkpointed and
    resumed alongside the mechanics :mod:`resume` demonstrates. Not
    exercised in this example's ``main`` (no checkpointer wired), but
    the shape carries the resume-ready surface.
    """

    query: str
    answer: str = ""
    self_review: str = ""
    external_review: str = ""
    reviews_agree: bool = False
    round: int = 0


@verb(role=PRIMARY)
async def answer_query(ctx: Context[VerifierState]) -> str:
    """Primary produces the initial answer.

    Reads the query from ``ctx.state.data.query`` — the same value the
    :class:`VerifierState` was constructed with — so ``.run()`` does
    not need to also pass it as a positional argument.
    """
    saia: ExampleSAIA = ctx.saia
    prompt = f"Answer this question concisely:\n{ctx.state.data.query}"
    ctx.state.data.answer = saia.answer(prompt)
    print(f"[primary/answer] {ctx.state.data.answer}")
    return ctx.state.data.answer


@verb(role=PRIMARY)
async def self_review(ctx: Context[VerifierState]) -> str:
    """Primary reviews its own current answer."""
    ctx.state.data.round += 1
    print(f"\n--- round {ctx.state.data.round} ---")
    saia: ExampleSAIA = ctx.saia
    prompt = (
        f"Review this answer for correctness. Query: {ctx.state.data.query!r}. "
        f"Answer: {ctx.state.data.answer!r}."
    )
    ctx.state.data.self_review = saia.answer(prompt)
    print(f"[primary/self-review] {ctx.state.data.self_review}")
    return ctx.state.data.self_review


@verb(role=VERIFIER)
async def external_review(ctx: Context[VerifierState]) -> str:
    """Verifier (a different model) reviews the same answer independently."""
    saia: ExampleSAIA = ctx.saia
    prompt = (
        f"Independently review this answer. Query: {ctx.state.data.query!r}. "
        f"Answer: {ctx.state.data.answer!r}."
    )
    ctx.state.data.external_review = saia.answer(prompt)
    print(f"[verifier/external-review] {ctx.state.data.external_review}")
    return ctx.state.data.external_review


@verb(role=JUDGE)
async def judge(ctx: Context[VerifierState]) -> bool:
    """Judge decides whether the two reviews semantically agree.

    Expects the scripted judge response to start with ``AGREE`` or
    ``DISAGREE`` (case-insensitive). Real deployments would use
    :meth:`llm_saia.SAIA.complete_structured` with a bool schema; the
    ``.answer(prompt) -> str`` surface stays deliberately simple.
    """
    saia: ExampleSAIA = ctx.saia
    prompt = (
        f"Do these two reviews semantically agree? "
        f"Review A: {ctx.state.data.self_review!r}. "
        f"Review B: {ctx.state.data.external_review!r}. "
        f"Answer with AGREE or DISAGREE plus a one-sentence rationale."
    )
    verdict = saia.answer(prompt)
    ctx.state.data.reviews_agree = verdict.strip().upper().startswith("AGREE")
    print(f"[judge] {verdict} (agree={ctx.state.data.reviews_agree})")
    return ctx.state.data.reviews_agree


@verb(role=PRIMARY)
async def maybe_correct(ctx: Context[VerifierState]) -> str:
    """Primary corrects its answer iff reviewers disagreed; else passes through.

    Short-circuits on consensus so a converged round does not consume
    another scripted response from the primary's queue.
    """
    if ctx.state.data.reviews_agree:
        return ctx.state.data.answer
    saia: ExampleSAIA = ctx.saia
    prompt = (
        f"Reviewers disagreed about your answer. "
        f"Your review: {ctx.state.data.self_review!r}. "
        f"Independent review: {ctx.state.data.external_review!r}. "
        f"Emit a corrected answer to: {ctx.state.data.query!r}."
    )
    ctx.state.data.answer = saia.answer(prompt)
    print(f"[primary/correct] {ctx.state.data.answer}")
    return ctx.state.data.answer


def _demo_scripts() -> dict[str, list[str]]:
    """Scripted responses that produce a 2-round convergence for the demo.

    Round 1 reviews disagree → primary corrects. Round 2 reviews agree
    → judge signals consensus, ``.iterate`` exits via ``until=``.
    """
    return {
        "primary": [
            # answer_query
            "The capital of France is Paris.",
            # round 1: self_review
            "My answer 'Paris' is accurate — Paris is the capital.",
            # round 1: maybe_correct (reviewers disagreed)
            "Correcting: The capital of France is Paris, verified against encyclopedic sources.",
            # round 2: self_review
            "My corrected answer stands — Paris is the capital.",
        ],
        "verifier": [
            # round 1: external_review — hedges, disagreeing with primary's confidence
            "The answer says 'Paris' but the source is not cited; needs verification.",
            # round 2: external_review — agrees with corrected answer
            "The corrected answer states Paris with an explicit source note; "
            "matches known geography.",
        ],
        "judge": [
            # round 1
            "DISAGREE — the primary claims accuracy while the verifier flags missing sourcing.",
            # round 2
            "AGREE — both reviews now affirm the 'Paris' answer.",
        ],
    }


async def main() -> int:
    """Run the verifier flow on a canned query."""
    lg = quick_console_logger("verifier-example", config={"level": "warning"})
    saia_f = StubSAIAFactory(_demo_scripts())
    ff = FlowFactory(lg, saia_f=saia_f, state_type=VerifierState)

    query = "What is the capital of France?"
    flow = ff.create("verifier", state=VerifierState(query=query))
    flow.call(answer_query).iterate(
        lambda body: body.call(self_review).call(external_review).call(judge).call(maybe_correct),
        max_iters=MAX_ROUNDS,
        until=lambda _result, ctx: ctx.state.data.reviews_agree,
    )

    print(f"Query: {query}\n")
    final = await flow.run()
    print("\n=== converged ===")
    print(f"final answer: {final}")
    print(f"rounds run:   {len(saia_f.instance('judge').call_log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
