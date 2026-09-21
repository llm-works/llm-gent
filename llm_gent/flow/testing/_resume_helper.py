# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Subprocess entry point for :func:`llm_gent.flow.testing.checkpoint.resume_in_subprocess`.

Invocation contract: the parent process runs ``python -m
llm_gent.flow.testing._resume_helper`` and writes a single JSON object to
stdin describing which store class to reconstruct and which flow builder
to call. This process performs the resume and prints the flow's final
result to stdout as JSON — one line, so the parent can parse it back.

Runs under whatever Python interpreter the parent launched (``sys.executable``);
inherits environment variables (matters for :class:`PgCheckpointStore` and
its ``APPINFRA_TEST_PG_URL``).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from typing import Any

from appinfra.log import quick_console_logger


def _resolve(module: str, attr: str) -> Any:
    """Import ``module`` and return its ``attr``."""
    return getattr(importlib.import_module(module), attr)


async def _run(payload: dict[str, Any]) -> Any:
    """Reconstruct store + Flow from the payload and drive ``run(resume=True)``."""
    lg = quick_console_logger("resume-helper", config={"level": "error"})
    store_factory = _resolve(payload["store_module"], payload["store_factory"])
    store = store_factory(lg, **payload["store_kwargs"])
    flow_builder = _resolve(payload["flow_module"], payload["flow_builder"])
    flow = flow_builder(
        lg,
        store=store,
        trajectory_id=payload["trajectory_id"],
        **payload["flow_builder_kwargs"],
    )
    return await flow.run(resume=True)


def main() -> None:
    """Read JSON payload from stdin, run the resume, print JSON result to stdout."""
    payload = json.loads(sys.stdin.read())
    result = asyncio.run(_run(payload))
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
