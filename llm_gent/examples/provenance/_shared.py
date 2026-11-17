# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""What the provenance examples share: the writer's SAIA and the report."""

from __future__ import annotations

from typing import Any

from appinfra.cas import Store, Vector
from appinfra.log import Logger
from llm_saia import SAIA, ToolDef
from llm_saia.core.backend import Backend

from llm_gent.provenance import Record, VersionControl


NOTES_TOOL = ToolDef(
    name="read_notes",
    description="Return the notes gathered on the topic. Call it once before writing.",
    parameters={"type": "object", "properties": {}},
)
SUBMIT_TOOL = ToolDef(
    name="submit_summary",
    description="Submit the one-sentence summary. Call this last.",
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "One sentence."}},
        "required": ["summary"],
    },
)
WRITER_STYLE = (
    "You are a terse technical writer. Call `read_notes` once, then call "
    "`submit_summary` with a one-sentence summary. Call each tool once."
)


def writer_saia(lg: Logger, backend: Backend, notes: str) -> SAIA:
    """A SAIA on ``backend`` whose tools hand the model ``notes`` and take its summary."""

    async def execute(name: str, _args: dict[str, Any]) -> str:
        return notes if name == "read_notes" else "acknowledged"

    return (
        SAIA.builder()
        .backend(backend)
        .logger(lg)
        .system(WRITER_STYLE)
        .max_iterations(4)
        .tools([NOTES_TOOL, SUBMIT_TOOL], execute)
        .terminal_tool("submit_summary", require_confirmation=False)
        .build()
    )


def summary_of(result: Any) -> str:
    """The summary a writer task submitted, else its final output."""
    terminal = getattr(result, "terminal_data", None)
    if isinstance(terminal, dict) and "summary" in terminal:
        return str(terminal["summary"])
    return str(getattr(result, "output", "") or "")


def print_records(vc: VersionControl, store: Store, repo: str) -> list[Record]:
    """Print the records oldest first, nested by parent, and what the repository holds.

    Returns the records, oldest first.
    """
    records = list(reversed(list(vc.log())))
    depth: dict[str | None, int] = {}
    for record in reversed(records):  # a parent commits after the records inside it
        depth[record.meta["activity"]] = depth.get(record.meta["parent"], -1) + 1
    for record in records:
        print(f"  {'  ' * depth[record.meta['activity']]}{_line(record)}")
    contexts = [r.inputs["context"] for r in records if "context" in r.inputs]
    messages = [m for c in contexts for m in Vector(vc.repo, c.id)]
    print(
        f"  {len(records)} records, {len(store.list_objects(repo))} objects in the repository; "
        f"{len(messages)} messages sent across {len(contexts)} model calls, "
        f"{len(set(messages))} stored"
    )
    return records


def _line(record: Record) -> str:
    """One record: kind, step name, actor, outcome, entity names."""
    meta = record.meta
    step = (meta.get("where") or {}).get("step_name")
    actor = {k: v for k, v in meta["actor"].items() if v}
    parts = [record.kind + (f" {step}" if step else "")]
    if actor:
        parts.append(f"actor={actor}")
    if meta.get("outcome"):
        parts.append(f"outcome={meta['outcome']}")
    if record.inputs or record.outputs:
        parts.append(f"in={sorted(record.inputs)} out={sorted(record.outputs)}")
    if meta.get("facets"):
        parts.append(f"facets={meta['facets']}")
    return "  ".join(parts)
