#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""Structured-output agent — classify + triage a bug report.

Demonstrates the production-shaped LLM surface: verbs call
:meth:`llm_saia.SAIA.complete_structured(prompt, schema) -> VerbResult[T]`
so the LLM returns a validated Pydantic model, not a string the verb
hand-parses. Two verbs run in sequence:

1. **classify** — reads ``ctx.data.report``, returns a
   :class:`Classification` (category, severity, summary).
2. **triage** — receives the classification as the previous node's
   result via signature-aware dispatch, returns a :class:`TriageResponse`
   (owner_team, first_reply, needs_escalation) and pins it into
   ``ctx.data.final_triage``. The triage return is the flow's final value.

:class:`BugReportState` inherits :class:`~llm_gent.flow.StateDataclass`
with no override — the mixin recurses into the ``TriageResponse |
None`` Pydantic field automatically, so the state class stays a
straight dataclass declaration.

The example uses :class:`~._infra.StructuredStubSAIAFactory`, which
scripts ``(schema, value)`` pairs and validates that the verb asks for
the same schema it was scripted with — drift surfaces at the call site
instead of silently on the wire.

Run standalone::

    python -m llm_gent.examples.flow.structured_agent
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from appinfra.log import quick_console_logger
from pydantic import BaseModel, ConfigDict

from llm_gent.examples.flow._infra import StructuredSAIA, StructuredStubSAIAFactory
from llm_gent.flow import Context, FlowFactory, Role, StateDataclass, verb


# ── schemas ──────────────────────────────────────────────────────────


class Category(StrEnum):
    """Bug-report top-level category — the classifier's primary axis."""

    CRASH = "crash"
    DATA_LOSS = "data_loss"
    PERFORMANCE = "performance"
    UI = "ui"
    OTHER = "other"


class Severity(StrEnum):
    """Coarse severity — drives escalation in :func:`triage`."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Classification(BaseModel):
    """Structured classifier output — what :func:`classify` returns."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    severity: Severity
    summary: str


class TriageResponse(BaseModel):
    """Structured triage output — the flow's final return."""

    model_config = ConfigDict(extra="forbid")

    owner_team: str
    first_reply: str
    needs_escalation: bool


# ── roles + state ────────────────────────────────────────────────────


CLASSIFIER = Role(name="classifier", backend="stub", model="classifier-model")
"""Reads the raw bug report; emits a :class:`Classification`."""

TRIAGER = Role(name="triager", backend="stub", model="triager-model")
"""Consumes the classification; emits a :class:`TriageResponse`."""


@dataclass
class BugReportState(StateDataclass):
    """Raw input + final triage — the two values that persist across resume.

    The intermediate :class:`Classification` produced by :func:`classify`
    is threaded to :func:`triage` via the previous node's return value,
    so it never needs a state slot. :class:`StateDataclass` auto-recurses
    into the ``TriageResponse | None`` field, so no ``to_dict`` /
    ``from_dict`` override is needed.
    """

    report: str
    final_triage: TriageResponse | None = None


# ── verbs ────────────────────────────────────────────────────────────


@verb(role=CLASSIFIER)
async def classify(ctx: Context[BugReportState]) -> Classification:
    """Classify the raw report into (category, severity, summary).

    Returns the validated :class:`Classification`; the next verb
    receives it via signature-aware dispatch and never round-trips
    through state.
    """
    saia: StructuredSAIA = ctx.saia
    prompt = f"Classify this bug report:\n{ctx.data.report}"
    result = await saia.complete_structured(prompt, Classification)
    c = result.value
    print(f"[classifier] {c.category.value} / {c.severity.value} — {c.summary}")
    return c


@verb(role=TRIAGER)
async def triage(
    ctx: Context[BugReportState],
    classification: Classification,
) -> TriageResponse:
    """Turn the classification into an actionable triage response.

    ``classification`` arrives as the previous node's return value —
    signature-aware dispatch binds it because :func:`triage` declares
    the extra positional. A verb without that parameter would silently
    drop the prev-result.
    """
    saia: StructuredSAIA = ctx.saia
    prompt = (
        f"Given this classification, produce a triage response as JSON:\n"
        f"{classification.model_dump_json()}"
    )
    result = await saia.complete_structured(prompt, TriageResponse)
    ctx.data.final_triage = result.value
    print(f"[triager] owner={result.value.owner_team} escalate={result.value.needs_escalation}")
    print(f"[triager] reply: {result.value.first_reply}")
    return result.value


# ── demo scripts ─────────────────────────────────────────────────────


def _demo_scripts() -> dict[str, list[tuple[type, Any]]]:
    """Scripted ``(schema, value)`` pairs for one classify+triage pass."""
    return {
        "classifier": [
            (
                Classification,
                Classification(
                    category=Category.CRASH,
                    severity=Severity.HIGH,
                    summary="App crashes on export when the workbook has >10 sheets.",
                ),
            ),
        ],
        "triager": [
            (
                TriageResponse,
                TriageResponse(
                    owner_team="exports-team",
                    first_reply=(
                        "Thanks for the report — reproduced locally. "
                        "Assigning to the exports team; expect an update within 24h."
                    ),
                    needs_escalation=True,
                ),
            ),
        ],
    }


async def main() -> int:
    """Run the classify → triage flow on a canned bug report."""
    lg = quick_console_logger("structured-agent-example", config={"level": "warning"})
    saia_f = StructuredStubSAIAFactory(_demo_scripts())
    ff = FlowFactory(lg, saia_f=saia_f, state_type=BugReportState)

    report = (
        "Excel export crashes the app when I try to export a workbook "
        "with more than 10 sheets. Reproduces every time."
    )
    flow = ff.create("structured-agent", state=BugReportState(report=report))
    flow.call(classify).call(triage)

    print(f"Report: {report}\n")
    final = await flow.run()
    print("\n=== triage complete ===")
    print(f"owner:       {final.owner_team}")
    print(f"escalate:    {final.needs_escalation}")
    print(f"first reply: {final.first_reply!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
