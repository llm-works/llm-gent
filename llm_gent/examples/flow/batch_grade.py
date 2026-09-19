#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""Batch grader — .map(strict=False) + Panel majority-vote + .guard + .rescue.

Shape:

1. **load_submissions** — a single verb that returns the list of
   :class:`Submission` from ``ctx.data.submissions`` so ``.map`` can
   pick it up as the item iterable.

2. **.map(grade_one, strict=False, aggregate=summarize)** — fan out one
   grade per submission concurrently. Per-item failures are wrapped as
   :class:`~llm_gent.flow.Failure` instead of propagating, so a single
   bad submission does not kill the batch.

3. **.guard(is_well_formed)** — attached to the map; a falsy verdict
   short-circuits the item before ``grade_one`` runs, filling the slot
   with :class:`~llm_gent.flow.Skipped`. Empty answers are skipped.

4. **grade_one** — a pure-Python verb (no LLM of its own) that fans out
   a :class:`~llm_gent.flow.Panel` of three grader verbs, each bound to
   a distinct :class:`Role` / model, and majority-votes on the score.
   Each grader verb calls
   :meth:`llm_saia.SAIA.complete_structured(prompt, Grade)` — no string
   parsing.

5. **assert_batch_quality** — a downstream verb that raises
   :class:`BatchQualityError` if the summary reports any failed items.
   Its ``.rescue(rescue_summary)`` catches that and marks the summary
   degraded. This split (permissive aggregate + strict downstream
   check) keeps the map's rescue slot free for map-internal failures
   and gives the quality gate its own node with clean rescue
   semantics — a rescue policy receives the failing node's *input*,
   which for ``assert_batch_quality`` is exactly the summary we want
   to salvage.

The canned batch is designed to exercise every primitive on one run:

* item 1 — valid answer, three graders agree on 4 → majority-vote 4.
* item 2 — valid answer, graders split 4/5/4 → majority-vote 4.
* item 3 — empty answer → guard returns falsy → :class:`Skipped`.
* item 4 — valid answer, one grader raises → Panel's gather propagates
  the exception → strict=False wraps it as :class:`Failure`.

``assert_batch_quality`` then raises because ``n_failed >= 1`` →
rescue lands the degraded summary.

Run standalone::

    python -m llm_gent.examples.flow.batch_grade
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any

from appinfra.log import quick_console_logger
from pydantic import BaseModel, ConfigDict, Field

from llm_gent.examples.flow._infra import StructuredSAIA, StructuredStubSAIAFactory
from llm_gent.flow import (
    Context,
    Failure,
    FlowFactory,
    Panel,
    Role,
    Skipped,
    StateDataclass,
    verb,
)
from llm_gent.flow.panel import majority


# ── schemas + roles ─────────────────────────────────────────────────


class Grade(BaseModel):
    """One grader's verdict on one submission."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=5)
    rationale: str


class BatchQualityError(RuntimeError):
    """Raised by :func:`assert_batch_quality` when any item failed.

    A deliberately strict gate downstream of the map so the example's
    :func:`rescue_summary` node has a real path to fire on the canned
    run.
    """


STRICT_GRADER = Role(name="strict-grader", backend="stub", model="strict-model")
"""Terse, hard-line grader. Weighs correctness above all else."""

LENIENT_GRADER = Role(name="lenient-grader", backend="stub", model="lenient-model")
"""Awards partial credit generously."""

HOLISTIC_GRADER = Role(name="holistic-grader", backend="stub", model="holistic-model")
"""Balances correctness, clarity, and completeness."""


# ── state ───────────────────────────────────────────────────────────


@dataclass
class Submission:
    """One student's answer to a question in the batch."""

    student_id: str
    question: str
    answer: str


@dataclass
class BatchGradingState(StateDataclass):
    """Batch input plus the final aggregate summary.

    Overrides :meth:`to_dict` / :meth:`from_dict` because
    :attr:`submissions` is a list of plain dataclasses and
    :attr:`final_summary` may hold Pydantic-serialized results; a
    ``dataclasses.asdict`` round-trip is not enough on its own.
    """

    submissions: list[Submission]
    final_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _serialize_result(r: Grade | Failure | Skipped) -> dict[str, Any]:
            if isinstance(r, Grade):
                return {"_t": "Grade", **r.model_dump()}
            if isinstance(r, Failure):
                return {
                    "_t": "Failure",
                    "exc_type": type(r.exception).__name__,
                    "exc_msg": str(r.exception),
                    "item": asdict(r.item) if hasattr(r.item, "__dataclass_fields__") else r.item,
                }
            return {
                "_t": "Skipped",
                "item": asdict(r.item) if hasattr(r.item, "__dataclass_fields__") else r.item,
            }

        summary = dict(self.final_summary)
        if "grades" in summary:
            summary["grades"] = [_serialize_result(r) for r in summary["grades"]]
        return {
            "submissions": [asdict(s) for s in self.submissions],
            "final_summary": summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchGradingState:
        def _deserialize_result(d: dict[str, Any]) -> Grade | Failure | Skipped:
            t = d.get("_t")
            if t == "Grade":
                return Grade(score=d["score"], rationale=d["rationale"])
            if t == "Failure":
                return Failure(
                    exception=RuntimeError(f"{d['exc_type']}: {d['exc_msg']}"),
                    item=Submission(**d["item"]) if isinstance(d["item"], dict) else d["item"],
                )
            return Skipped(
                item=Submission(**d["item"]) if isinstance(d["item"], dict) else d["item"],
            )

        raw_summary = data.get("final_summary", {})
        summary: dict[str, Any] = dict(raw_summary)
        if "grades" in summary:
            summary["grades"] = [_deserialize_result(r) for r in summary["grades"]]
        return cls(
            submissions=[Submission(**s) for s in data["submissions"]],
            final_summary=summary,
        )


# ── grader verbs (three roles, three SAIAs) ─────────────────────────


async def _grade(ctx: Context[BatchGradingState], submission: Submission, style: str) -> Grade:
    """Shared body for the three grader verbs.

    Each grader passes its own ``style`` label so the scripted stub can
    tell the prompts apart while the code stays DRY. Real graders would
    swap the prompt template rather than this ``style`` marker.
    """
    saia: StructuredSAIA = ctx.saia
    prompt = (
        f"[{style}] Grade this answer on a 0-5 scale.\n"
        f"Question: {submission.question!r}\n"
        f"Answer:   {submission.answer!r}\n"
        f"Return {{score, rationale}}."
    )
    result = await saia.complete_structured(prompt, Grade)
    return result.value


@verb(role=STRICT_GRADER)
async def grade_strict(ctx: Context[BatchGradingState], submission: Submission) -> Grade:
    """Strict grader — pins on correctness."""
    return await _grade(ctx, submission, "strict")


@verb(role=LENIENT_GRADER)
async def grade_lenient(ctx: Context[BatchGradingState], submission: Submission) -> Grade:
    """Lenient grader — generous with partial credit."""
    return await _grade(ctx, submission, "lenient")


@verb(role=HOLISTIC_GRADER)
async def grade_holistic(ctx: Context[BatchGradingState], submission: Submission) -> Grade:
    """Holistic grader — balances correctness, clarity, completeness."""
    return await _grade(ctx, submission, "holistic")


# ── aggregate ───────────────────────────────────────────────────────


def _score_majority(grades: list[Grade]) -> Grade:
    """Return the :class:`Grade` whose score wins a majority vote across graders.

    Ties break by first-occurrence order (mirrors
    :func:`llm_gent.flow.panel.majority`). The returned rationale is the
    first grade whose score matches the winning value, so downstream
    consumers see a real, model-authored explanation rather than a
    synthesized composite.
    """
    winning_score = majority([g.score for g in grades])
    for g in grades:
        if g.score == winning_score:
            return g
    raise AssertionError("unreachable: winning score always came from the list")


_grader_panel = Panel(
    [grade_strict, grade_lenient, grade_holistic],
    aggregate=_score_majority,
)
"""Three-model panel that majority-votes on the score for one submission."""


# ── per-item body + guard ───────────────────────────────────────────


@verb(role=None)
async def grade_one(ctx: Context[BatchGradingState], submission: Submission) -> Grade:
    """Fan out the grader panel for a single submission and majority-vote."""
    grade = await _grader_panel.run(ctx, submission)
    # Panel.run is typed as ``-> Any`` because the aggregate is caller-defined;
    # narrow to Grade here so downstream consumers see the concrete type.
    assert isinstance(grade, Grade)
    print(f"[grade_one/{submission.student_id}] score={grade.score} rationale={grade.rationale!r}")
    return grade


def is_well_formed(item: Submission, _ctx: Context[BatchGradingState]) -> bool:
    """Guard predicate: skip submissions with empty or whitespace-only answers.

    ``.guard`` runs after per-item state projection and before the body,
    so a skipped item never consumes any grader's scripted response.
    """
    return bool(item.answer.strip())


# ── outer verbs + aggregate ─────────────────────────────────────────


@verb(role=None)
async def load_submissions(ctx: Context[BatchGradingState]) -> list[Submission]:
    """Yield the batch as the map's item iterable.

    Real callers would fetch from a queue / DB here. Kept as a verb (not
    a plain lambda) so the flow's first node is stateful enough to
    checkpoint if a caller wires a store.
    """
    return list(ctx.data.submissions)


def summarize(results: list[Grade | Failure | Skipped]) -> dict[str, Any]:
    """Partition the map's results into an ok / failed / skipped summary.

    Always returns cleanly — never raises. The downstream
    :func:`assert_batch_quality` verb owns the strict-quality decision;
    keeping the aggregate permissive means the summary is always
    available for either normal or degraded reporting.
    """
    ok = [r for r in results if isinstance(r, Grade)]
    failed = [r for r in results if isinstance(r, Failure)]
    skipped = [r for r in results if isinstance(r, Skipped)]
    return {
        "grades": results,
        "n_ok": len(ok),
        "n_failed": len(failed),
        "n_skipped": len(skipped),
        "mean_score": (sum(g.score for g in ok) / len(ok)) if ok else None,
    }


@verb(role=None)
async def assert_batch_quality(
    _ctx: Context[BatchGradingState],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Raise :class:`BatchQualityError` if any submission failed to grade.

    Split from :func:`summarize` so its ``.rescue`` receives the *summary*
    as ``pending_input`` — a rescue policy fires with the failing node's
    input, which for a chain step is the previous node's return value.
    """
    if summary["n_failed"] > 0:
        raise BatchQualityError(
            f"{summary['n_failed']} of {len(summary['grades'])} submissions failed to grade"
        )
    return summary


async def rescue_summary(
    exc: BaseException,
    summary: dict[str, Any] | None,
    _ctx: Context[BatchGradingState],
) -> dict[str, Any]:
    """Salvage the summary when :func:`assert_batch_quality` refuses to certify.

    ``summary`` is the failing node's input — the map's aggregated
    output. Preserves per-item results, adds a degraded flag and the
    reason so the caller sees exactly why the gate tripped.

    Rescue policy signature: ``(exc, pending_input, ctx) -> fallback``.
    """
    base = summary if summary is not None else {"grades": []}
    print(f"[assert_batch_quality/rescue] degraded: {type(exc).__name__}: {exc}")
    return {**base, "degraded": True, "reason": str(exc)}


@verb(role=None)
async def pin_summary(
    ctx: Context[BatchGradingState],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Persist the summary into state so a resumed run can inspect it."""
    ctx.data.final_summary = summary
    return summary


# ── demo scripts ────────────────────────────────────────────────────


_BOOM = RuntimeError("scripted grader outage")


def _demo_scripts() -> dict[str, list[tuple[type, Any]]]:
    """Scripted ``(schema, value)`` pairs for the four-submission demo.

    Each grader is called once per non-skipped submission, in student-id
    order (``s1``, ``s2``, ``s4``). ``s3`` is skipped by the guard and
    consumes no responses.

    Split votes on ``s2``: strict says 5, lenient and holistic say 4 →
    majority 4. On ``s4`` the strict grader raises so the whole panel
    (asyncio.gather) fails and the item is wrapped as :class:`Failure`.
    """
    return {
        "strict-grader": [
            (Grade, Grade(score=4, rationale="Correct and concise.")),
            (Grade, Grade(score=5, rationale="Precise; identifies the capital plainly.")),
            (Grade, _BOOM),
        ],
        "lenient-grader": [
            (Grade, Grade(score=4, rationale="Correct answer.")),
            (Grade, Grade(score=4, rationale="Answers the question, minor phrasing quibble.")),
            (Grade, Grade(score=5, rationale="Correct — full credit.")),
        ],
        "holistic-grader": [
            (Grade, Grade(score=4, rationale="Correct, clear, complete for a one-line answer.")),
            (Grade, Grade(score=4, rationale="Correct; sentence structure a bit inverted.")),
            (Grade, Grade(score=4, rationale="Correct answer, adequately expressed.")),
        ],
    }


def _demo_submissions() -> list[Submission]:
    """Canonical four-submission batch that hits every primitive."""
    q = "What is the capital of France?"
    return [
        Submission(student_id="s1", question=q, answer="The capital of France is Paris."),
        Submission(student_id="s2", question=q, answer="Paris, France's capital city."),
        Submission(student_id="s3", question=q, answer="   "),  # guard → Skipped
        Submission(student_id="s4", question=q, answer="Paris."),  # strict raises
    ]


# ── main ────────────────────────────────────────────────────────────


def _print_summary(summary: dict[str, Any]) -> None:
    """Render the final summary + per-item breakdown."""
    print("\n=== batch grading complete ===")
    print(f"ok:       {summary['n_ok']}")
    print(f"skipped:  {summary['n_skipped']}")
    print(f"failed:   {summary['n_failed']}")
    print(f"mean:     {summary['mean_score']}")
    if summary.get("degraded"):
        print(f"degraded: {summary['reason']}")
    markers = {Grade: "ok", Failure: "FAIL", Skipped: "SKIP"}
    for i, r in enumerate(summary["grades"]):
        if isinstance(r, Grade):
            detail = f"score={r.score} — {r.rationale}"
        elif isinstance(r, Failure):
            detail = f"{type(r.exception).__name__}: {r.exception}"
        else:
            detail = "skipped by guard"
        print(f"  [{i}] {markers[type(r)]}: {detail}")


async def main() -> int:
    """Grade the demo batch end-to-end."""
    lg = quick_console_logger("batch-grade-example", config={"level": "warning"})
    saia_f = StructuredStubSAIAFactory(_demo_scripts())
    ff = FlowFactory(lg, saia_f=saia_f, state_type=BatchGradingState)

    submissions = _demo_submissions()
    flow = ff.create("batch-grade", state=BatchGradingState(submissions=submissions))

    # Panel dispatches inner verbs via ctx.flow.dispatch(name), which
    # requires them to be registered on the flow that carries the
    # runtime. The map/aggregate verbs are wired positionally by
    # .call() / .map() and do not need explicit registration.
    flow.register(grade_strict)
    flow.register(grade_lenient)
    flow.register(grade_holistic)

    (
        flow.call(load_submissions)
        .map(grade_one, aggregate=summarize, strict=False)
        .guard(is_well_formed)
        .call(assert_batch_quality)
        .rescue(rescue_summary)
        .call(pin_summary)
    )

    print(f"Batch: {len(submissions)} submissions\n")
    summary = await flow.run()
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
