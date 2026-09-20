#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

# ci-run:

"""Plan-and-execute agent — planner + extractor + .branch + .iterate.

Shape of the canonical ReAct-flavored agent:

1. **plan** — a :func:`planner` archetype verb calls
   :meth:`~llm_saia.SAIA.complete_structured(prompt, Plan)` to emit a
   :class:`Plan` of *freeform* step intents (natural-language strings
   like ``"search Tokyo population"``). The plan lives on
   ``ctx.data.plan`` and is fixed for the run — this example demos the
   one-shot plan-then-execute shape; a re-planning variant would move
   ``plan`` inside the loop.

2. **iterate(body, until, max_iters)** — the execution loop. Each
   iteration:

   a. **next_step** — an :func:`extractor` archetype verb calls
      :meth:`complete_structured(prompt, ToolCall)` to parse the current
      step intent (``plan.steps[step_index]``) into a typed
      :class:`ToolCall`. This is the archetype's canonical shape:
      freeform evidence in, structured record out.

   b. **branch** — routes on ``ToolCall.tool``. :meth:`.branch` is
      binary, so 3-way routing nests one branch in the ``else_`` arm of
      another. A future ``.match(cases)`` primitive would flatten this
      shape; today, nested :meth:`.branch` is the idiom.

   c. **tool verb** — one of :func:`search_tool`, :func:`calc_tool`,
      :func:`finish_tool`. All three are pure-Python
      (:func:`~llm_gent.flow.verb` with no role) — no LLM call.
      :func:`finish_tool` sets ``ctx.data.final_answer``, which is what
      the :func:`is_done` predicate checks.

   d. **record_result** — appends the :class:`StepResult` to
      ``ctx.data.step_results`` and increments ``step_index``.

3. **until** — :func:`is_done` fires when ``final_answer`` is set OR
   ``step_index`` has walked past the plan's last step. ``max_iters=10``
   is a hard bound in case both signals somehow fail to trip.

State serialization: :class:`Plan`, :class:`ToolCall`, and
:class:`StepResult` are :class:`~pydantic.BaseModel` subclasses;
:class:`~llm_gent.flow.StateDataclass` auto-recurses into ``T | None``
and ``list[T]`` of BaseModels, so :class:`PlanExecuteState` needs no
``to_dict`` / ``from_dict`` override.

Run standalone::

    python -m llm_gent.examples.flow.plan_execute
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Callable
from typing import Any

from appinfra.log import quick_console_logger
from pydantic import BaseModel, ConfigDict, Field

from llm_gent.examples.flow._infra import StructuredSAIA, StructuredStubSAIAFactory
from llm_gent.flow import (
    Context,
    FlowFactory,
    Role,
    StateDataclass,
    TypeStateFactory,
    extractor,
    planner,
    verb,
)


# ── schemas ──────────────────────────────────────────────────────────


class ToolName(StrEnum):
    """The tools the planner can request and the executor knows how to run."""

    SEARCH = "search"
    CALCULATE = "calculate"
    FINISH = "finish"


class ToolCall(BaseModel):
    """A typed tool invocation produced by the extractor from a step intent.

    ``args`` is intentionally loose (``dict[str, str]``): each tool
    reads the keys it knows about. A real system would use per-tool
    schemas and a discriminated union; the loose shape keeps this
    example focused on flow composition.
    """

    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    args: dict[str, str]


class Plan(BaseModel):
    """The planner's output — freeform step intents plus a rationale.

    Steps are natural-language strings, not pre-parsed :class:`ToolCall`
    records. This is what makes the extractor archetype earn its place
    in the loop: it converts free text to structure per iteration.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    steps: list[str] = Field(min_length=1)


class StepResult(BaseModel):
    """One step's outcome — echoes the invocation and carries the output."""

    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    args: dict[str, str]
    output: str


# ── roles + state ────────────────────────────────────────────────────


PLANNER = Role(name="planner", backend="stub", model="planner-model")
"""Emits the freeform :class:`Plan` from ``ctx.data.question``."""

EXTRACTOR = Role(name="extractor", backend="stub", model="extractor-model")
"""Parses one step intent into a typed :class:`ToolCall` per iteration."""


@dataclass
class PlanExecuteState(StateDataclass):
    """Question in, plan + step trace out.

    :class:`StateDataclass` auto-recurses into ``Plan | None`` and
    ``list[StepResult]`` (both Pydantic-backed) so no serializer
    override is needed.
    """

    question: str
    plan: Plan | None = None
    step_index: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    final_answer: str | None = None


# ── verbs ────────────────────────────────────────────────────────────


@planner(PLANNER)
async def plan(ctx: Context[PlanExecuteState]) -> Plan:
    """Emit a freeform plan for the question.

    Returns the whole :class:`Plan` up front (one-shot planning); the
    loop below walks its ``steps`` list. Also pins the plan into state
    so :func:`next_step` can read the current intent without threading
    the plan through every dispatch.
    """
    saia: StructuredSAIA = ctx.saia
    prompt = f"Plan how to answer this question step by step:\n{ctx.data.question}"
    result = await saia.complete_structured(prompt, Plan)
    ctx.data.plan = result.value
    print(f"[planner] reasoning: {result.value.reasoning}")
    for i, step in enumerate(result.value.steps):
        print(f"[planner]   step[{i}]: {step}")
    return result.value


@extractor(EXTRACTOR)
async def next_step(ctx: Context[PlanExecuteState]) -> ToolCall:
    """Parse the current step's freeform intent into a typed :class:`ToolCall`.

    Reads ``plan.steps[step_index]``; the extractor's LLM decides which
    tool and which args best realize that intent. Emitting a
    :class:`ToolCall` directly (rather than a string the branch would
    have to re-parse) is what makes ``.branch(when=step.tool == ...)``
    readable.
    """
    assert ctx.data.plan is not None, "planner must run before next_step"
    intent = ctx.data.plan.steps[ctx.data.step_index]
    saia: StructuredSAIA = ctx.saia
    prompt = (
        f"The overall question is: {ctx.data.question!r}.\n"
        f"Parse this step intent into a tool call: {intent!r}"
    )
    result = await saia.complete_structured(prompt, ToolCall)
    print(
        f"[extractor] step[{ctx.data.step_index}] → {result.value.tool.value}({result.value.args})"
    )
    return result.value


@verb
async def search_tool(_ctx: Context[PlanExecuteState], step: ToolCall) -> StepResult:
    """Canned search — returns a hardcoded snippet for the demo query.

    A production tool would call a real search API here; the flow shape
    is identical.
    """
    query = step.args.get("query", "")
    output = f"Search results for {query!r}: Tokyo has a metropolitan population of 14,000,000."
    print(f"[search] {output}")
    return StepResult(tool=ToolName.SEARCH, args=step.args, output=output)


@verb
async def calc_tool(_ctx: Context[PlanExecuteState], step: ToolCall) -> StepResult:
    """Evaluate an integer arithmetic expression from ``step.args['expression']``.

    Restricted to ``0-9`` / ``+-*/`` / whitespace / parens so
    :func:`eval` can be used safely on the trusted, planner-authored
    input for the demo. A production tool would use a proper expression
    parser.
    """
    expr = step.args.get("expression", "")
    allowed = set("0123456789+-*/(). ")
    if not expr or set(expr) - allowed:
        output = f"Cannot evaluate expression {expr!r}: unsupported characters."
    elif "**" in expr:
        output = f"Cannot evaluate expression {expr!r}: exponentiation not allowed."
    else:
        try:
            output = str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
        except (SyntaxError, ZeroDivisionError, ValueError, OverflowError) as e:
            output = f"Evaluation of {expr!r} failed: {type(e).__name__}: {e}"
    print(f"[calc] {expr} = {output}")
    return StepResult(tool=ToolName.CALCULATE, args=step.args, output=output)


@verb
async def finish_tool(ctx: Context[PlanExecuteState], step: ToolCall) -> StepResult:
    """Terminal step — pin ``args['answer']`` into ``ctx.data.final_answer``.

    Setting :attr:`final_answer` here is what makes :func:`is_done`
    fire and the outer :meth:`.iterate` exit before it walks off the
    plan's end.
    """
    answer = step.args.get("answer", "")
    ctx.data.final_answer = answer
    print(f"[finish] final answer: {answer!r}")
    return StepResult(tool=ToolName.FINISH, args=step.args, output=answer)


@verb
async def record_result(
    ctx: Context[PlanExecuteState],
    step_result: StepResult,
) -> StepResult:
    """Append the step's result and advance the plan cursor.

    Kept separate from the tool verbs so per-tool code stays focused on
    its own action; the bookkeeping is one place, uniform across tools.
    """
    ctx.data.step_results.append(step_result)
    ctx.data.step_index += 1
    return step_result


# ── predicates ───────────────────────────────────────────────────────


def _tool_is(name: ToolName) -> Callable[[ToolCall, Context[PlanExecuteState]], bool]:
    """Return a :meth:`.branch` predicate that matches ``step.tool == name``."""

    def when(step: ToolCall, _ctx: Context[PlanExecuteState]) -> bool:
        return step.tool == name

    return when


def is_done(_result: Any, ctx: Context[PlanExecuteState]) -> bool:
    """Iteration stops when the plan is exhausted or ``finish`` has fired.

    Either signal alone would suffice for a well-behaved planner /
    extractor pair; both are checked so a plan that ends without an
    explicit ``finish`` step also terminates cleanly.
    """
    assert ctx.data.plan is not None, "planner must run before is_done"
    return ctx.data.final_answer is not None or ctx.data.step_index >= len(ctx.data.plan.steps)


# ── demo scripts ─────────────────────────────────────────────────────


_DEMO_PLAN = Plan(
    reasoning=(
        "The question asks for twice the population of Tokyo. First look up "
        "Tokyo's population, then multiply by two, then report the doubled value."
    ),
    steps=[
        "search Tokyo population",
        "multiply the found population by two",
        "finish with the doubled population as the answer",
    ],
)
"""Canned plan the stub planner returns for the demo question."""


_DEMO_TOOL_CALLS = [
    ToolCall(tool=ToolName.SEARCH, args={"query": "Tokyo population"}),
    ToolCall(tool=ToolName.CALCULATE, args={"expression": "14000000 * 2"}),
    ToolCall(
        tool=ToolName.FINISH,
        args={"answer": "Twice Tokyo's population is 28,000,000."},
    ),
]
"""One :class:`ToolCall` per iteration — extractor's scripted responses."""


def _demo_scripts() -> dict[str, list[tuple[type, Any]]]:
    """Assemble the ``{role_name: [(schema, value), ...]}`` script.

    The calculator tool actually evaluates its expression, so the demo
    is not purely scripted — ``28000000`` is computed at runtime from
    the search's canned snippet.
    """
    return {
        "planner": [(Plan, _DEMO_PLAN)],
        "extractor": [(ToolCall, tc) for tc in _DEMO_TOOL_CALLS],
    }


# ── main ─────────────────────────────────────────────────────────────


def _build_body(body: Any) -> Any:
    """Assemble the per-iteration body: extract → route by tool → record.

    The 3-way tool dispatch is a chain of nested :meth:`.branch` calls
    because :meth:`.branch` is binary; the outer arm matches SEARCH,
    the else-arm matches CALCULATE, and the else-else falls through to
    FINISH.
    """
    return (
        body.call(next_step)
        .branch(
            when=_tool_is(ToolName.SEARCH),
            then=search_tool,
            else_=lambda f: f.branch(
                when=_tool_is(ToolName.CALCULATE),
                then=calc_tool,
                else_=finish_tool,
            ),
        )
        .call(record_result)
    )


def _print_result(state: PlanExecuteState) -> None:
    """Render the final answer and the per-step trace."""
    print("\n=== plan-execute complete ===")
    print(f"final answer: {state.final_answer}")
    print(f"steps taken:  {state.step_index}")
    for i, r in enumerate(state.step_results):
        print(f"  [{i}] {r.tool.value}({r.args}) → {r.output}")


async def main() -> int:
    """Run the plan-and-execute flow on the canned question."""
    lg = quick_console_logger("plan-execute-example", config={"level": "warning"})
    saia_factory = StructuredStubSAIAFactory(_demo_scripts())
    ff = FlowFactory(
        lg, saia_factory=saia_factory, state_factory=TypeStateFactory(PlanExecuteState)
    )

    question = "What is twice the population of Tokyo?"
    flow = ff.create("plan-execute", state=PlanExecuteState(question=question))
    flow.call(plan).iterate(_build_body, until=is_done, max_iters=10)

    print(f"Question: {question}\n")
    await flow.run()
    _print_result(flow.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
