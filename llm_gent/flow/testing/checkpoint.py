# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Test harness for Flow-level checkpoint/resume determinism.

Ships:

- :class:`CanonicalCounter` — a small :class:`StateDataclass` payload.
- :func:`build_canonical_flow` — a deterministic multi-stage Flow
  (``+1 → ×2 → -3``, iterated) whose composition is stable across
  interrupted, resumed, and uninterrupted runs.
- :func:`resume_in_subprocess` — cross-process boundary helper: spawns a
  fresh Python process that loads a checkpoint via a caller-specified
  store class, resumes the caller-specified Flow builder, and returns the
  final state as a dict. The subprocess entry point is
  :mod:`llm_gent.flow.testing._resume_helper`.

Both primitives support the same testing recipe — run baseline,
interrupt-and-save, resume, assert final states equal — either in-process
(new :class:`Flow` instance for the resume) or cross-process (subprocess
for the resume). Consumers can substitute their own Flow builder module
(the second positional to :func:`resume_in_subprocess`) when their state
comparator needs domain knowledge.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from appinfra.log import Logger

from ..checkpoint import CheckpointStore
from ..context import Context
from ..factory import FlowFactory
from ..flow import Flow
from ..state import StateDataclass, TypeStateFactory
from ..verb import verb


# -----------------------------------------------------------------------------
# Canonical multi-stage state
# -----------------------------------------------------------------------------


@dataclass
class CanonicalCounter(StateDataclass):
    """Deterministic multi-stage flow payload.

    Three fields:

    - ``n``: running accumulator mutated by the three arithmetic verbs.
    - ``log``: every intermediate value pushed by each verb — the primary
      determinism observable (same halt point ⇒ same log tail on resume).
    - ``iterations_completed``: bumped by the last verb of the body;
      drives the halt-after-N mechanism.
    """

    n: int = 0
    log: list[int] = field(default_factory=list)
    iterations_completed: int = 0


# -----------------------------------------------------------------------------
# Verbs — module-level so the subprocess helper can import them cheaply.
# -----------------------------------------------------------------------------


@verb
async def _add_one(ctx: Context[CanonicalCounter], _prev: Any = None) -> int:
    """Stage 1 of the canonical body: ``n += 1``, push to ``log``."""
    ctx.state.data.n += 1
    ctx.state.data.log.append(ctx.state.data.n)
    return ctx.state.data.n


@verb
async def _times_two(ctx: Context[CanonicalCounter], _prev: Any = None) -> int:
    """Stage 2 of the canonical body: ``n *= 2``, push to ``log``."""
    ctx.state.data.n *= 2
    ctx.state.data.log.append(ctx.state.data.n)
    return ctx.state.data.n


@verb
async def _minus_three(ctx: Context[CanonicalCounter], _prev: Any = None) -> int:
    """Stage 3 of the canonical body: ``n -= 3``, push to ``log``, bump ``iterations_completed``."""
    ctx.state.data.n -= 3
    ctx.state.data.log.append(ctx.state.data.n)
    ctx.state.data.iterations_completed += 1
    return ctx.state.data.n


@verb
async def _snapshot(ctx: Context[CanonicalCounter], _prev: Any = None) -> dict[str, Any]:
    """Tail verb — returns state.data as a JSON-serializable dict.

    Enables :meth:`Flow.run` to yield the final state directly (needed
    by :func:`resume_in_subprocess` — the subprocess prints the run's
    return to stdout as JSON).
    """
    return ctx.state.data.to_dict()


def _make_halt_check(halt: asyncio.Event | None, halt_after_iteration: int | None) -> Any:
    """Return a body-tail verb that fires ``halt`` once ``iterations_completed`` hits the threshold.

    Both arguments are captured in the closure so a resume flow (called
    with ``halt=None``) yields a no-op halt-check — the composition graph
    stays identical between interrupt and resume, but only the interrupt
    run's copy of the verb can trip halt.
    """

    @verb
    async def _halt_check(ctx: Context[CanonicalCounter], _prev: Any = None) -> Any:
        """Fire the captured halt event once the iteration threshold is met; passthrough otherwise."""
        if (
            halt is not None
            and halt_after_iteration is not None
            and ctx.state.data.iterations_completed >= halt_after_iteration
        ):
            halt.set()
        return _prev

    return _halt_check


# -----------------------------------------------------------------------------
# Canonical flow builder
# -----------------------------------------------------------------------------


def build_canonical_flow(
    lg: Logger,
    *,
    state: CanonicalCounter | None = None,
    max_iters: int = 5,
    halt: asyncio.Event | None = None,
    halt_after_iteration: int | None = None,
    store: CheckpointStore | None = None,
    trajectory_id: str = "canonical-multi-stage",
) -> Flow:
    """Build the canonical multi-stage Flow.

    Composition::

        .iterate(add_one → times_two → minus_three → halt_check, max_iters)
        .call(snapshot)

    Deterministic — no clock, no randomness, no I/O. The tail
    ``snapshot`` returns ``state.data.to_dict()`` so :meth:`Flow.run`
    yields the final state directly (uniform observable across
    baseline, interrupt, and resume paths).

    Args:
        lg: Logger.
        state: Initial payload. Defaults to a fresh ``CanonicalCounter()``.
        max_iters: Iterate cap (cumulative across resumes — the resume
            of an interrupt saved at iteration N runs ``max_iters - N``
            further passes).
        halt: Optional ambient halt event. When paired with
            ``halt_after_iteration``, iterate exits between passes once
            the threshold is reached and the checkpoint is preserved
            (halt-set exit skips the delete-on-success path).
        halt_after_iteration: Iteration count at which to fire ``halt``.
            Requires ``halt``.
        store: Optional :class:`CheckpointStore`. Wires
            ``.with_checkpointer(store, trajectory_id)`` when present.
        trajectory_id: Checkpoint trajectory identifier.

    Raises:
        ValueError: When exactly one of ``halt`` / ``halt_after_iteration``
            is provided (they must be paired or both absent).
    """
    if (halt is None) != (halt_after_iteration is None):
        raise ValueError("halt and halt_after_iteration must be provided together (or neither)")

    ff = FlowFactory(lg, state_factory=TypeStateFactory(CanonicalCounter))
    flow = ff.create(state=state if state is not None else CanonicalCounter())
    if store is not None:
        flow.with_checkpointer(store, trajectory_id)
    if halt is not None:
        flow.with_halt(halt)

    halt_check = _make_halt_check(halt, halt_after_iteration)

    def _body(f: Flow) -> None:
        f.call(_add_one).then(_times_two).then(_minus_three).then(halt_check)

    return flow.iterate(_body, max_iters=max_iters).call(_snapshot)


# -----------------------------------------------------------------------------
# Same-process determinism assertion
# -----------------------------------------------------------------------------


async def assert_resume_determinism(
    lg: Logger,
    store: CheckpointStore,
    *,
    halt_after_iteration: int = 2,
    max_iters: int = 5,
    trajectory_id: str = "determinism-check",
) -> dict[str, Any]:
    """Run baseline, interrupt, resume, assert equality, return final state.

    The load-bearing invariant: a Flow resumed from a mid-run checkpoint
    reaches the same final state as an uninterrupted run. This helper
    wraps the three-step pattern (baseline / interrupt / resume) and
    asserts equality, returning the final state dict on success.

    Uses :func:`build_canonical_flow` internally — deterministic,
    no I/O, stable composition across interrupt and resume.

    Args:
        lg: Logger.
        store: Checkpoint store for the interrupt and resume runs.
        halt_after_iteration: Iteration at which to fire halt (default 2).
        max_iters: Total iterations for the flow (default 5).
        trajectory_id: Checkpoint trajectory identifier.

    Returns:
        The final state dict (``CanonicalCounter.to_dict()``).

    Raises:
        AssertionError: If the resumed state differs from baseline.
    """
    baseline = await build_canonical_flow(lg, max_iters=max_iters).run()

    halt = asyncio.Event()
    await build_canonical_flow(
        lg,
        max_iters=max_iters,
        halt=halt,
        halt_after_iteration=halt_after_iteration,
        store=store,
        trajectory_id=trajectory_id,
    ).run()

    resumed = await build_canonical_flow(
        lg,
        max_iters=max_iters,
        store=store,
        trajectory_id=trajectory_id,
    ).run(resume=True)

    assert resumed == baseline, (
        f"resumed state differs from baseline:\n  baseline={baseline}\n  resumed={resumed}"
    )
    return dict(resumed)


# -----------------------------------------------------------------------------
# Cross-process resume helper
# -----------------------------------------------------------------------------


def pg_checkpoint_store_from_config(lg: Logger, *, url: str, schema: str) -> Any:
    """Subprocess-friendly :class:`PgCheckpointStore` factory.

    :class:`PgCheckpointStore` takes a live :class:`~appinfra.db.pg.PG`
    handle — not JSON-serializable, so it can't cross the subprocess
    boundary directly. This factory takes the connection URL and schema
    name as strings (both trivially serializable), builds the PG handle
    inside the subprocess, and returns a store bound to it. Pair with
    :func:`resume_in_subprocess` by passing this function's dotted path
    as ``store_factory``.
    """
    from appinfra.db.pg import PG

    from ..stores.postgres import PgCheckpointStore

    pg = PG(lg, {"url": url}, schema=schema)
    return PgCheckpointStore(lg, pg)


def resume_in_subprocess(
    *,
    store_module: str,
    store_factory: str,
    store_kwargs: dict[str, Any],
    flow_module: str = "llm_gent.flow.testing.checkpoint",
    flow_builder: str = "build_canonical_flow",
    flow_builder_kwargs: dict[str, Any] | None = None,
    trajectory_id: str = "canonical-multi-stage",
    subprocess_timeout: float = 30.0,
) -> dict[str, Any]:
    """Spawn a fresh Python process, resume the Flow from checkpoint, return final state.

    The subprocess entry point is :mod:`llm_gent.flow.testing._resume_helper`.
    It reconstructs the store by resolving ``store_module.store_factory``
    (a class or a function) and calling it as
    ``store_factory(lg, **store_kwargs)``, resolves and calls the flow
    builder with ``store=<instance>, trajectory_id=<>, **flow_builder_kwargs``,
    invokes ``await flow.run(resume=True)``, and prints the return value
    as JSON to stdout. The parent parses and returns it.

    Every value in ``store_kwargs`` must be JSON-serializable (paths as
    strings, connection URLs and schema names as strings). Stores whose
    constructors take live handles use a factory function that builds
    the handle from strings — :func:`pg_checkpoint_store_from_config` is
    the canonical example (PG handle from URL + schema).

    Args:
        store_module: Dotted module path exposing the store factory
            (e.g. ``"llm_gent.flow.stores.json_file"`` for the class
            directly, or ``"llm_gent.flow.testing.checkpoint"`` for a
            wrapper function like :func:`pg_checkpoint_store_from_config`).
        store_factory: Class or function name inside that module. Must
            be callable as ``store_factory(lg, **store_kwargs) -> CheckpointStore``.
        store_kwargs: JSON-serializable kwargs for the store factory.
        flow_module: Dotted module path exposing the flow builder.
        flow_builder: Builder function name inside that module. Must
            accept ``(lg, *, store, trajectory_id, **kwargs) -> Flow``.
        flow_builder_kwargs: Additional kwargs passed to the flow builder.
        trajectory_id: Checkpoint trajectory identifier — must match the
            id used by the interrupt run that wrote the checkpoint.
        subprocess_timeout: Wall-clock cap on the subprocess in seconds.

    Returns:
        The final state dict — whatever the flow's tail node returned.

    Raises:
        subprocess.CalledProcessError: The subprocess exited non-zero.
            The stderr is included in the exception.
        subprocess.TimeoutExpired: The subprocess ran past ``subprocess_timeout``.
        json.JSONDecodeError: The subprocess's stdout was not valid JSON.
    """
    payload = {
        "store_module": store_module,
        "store_factory": store_factory,
        "store_kwargs": store_kwargs,
        "flow_module": flow_module,
        "flow_builder": flow_builder,
        "flow_builder_kwargs": flow_builder_kwargs or {},
        "trajectory_id": trajectory_id,
    }
    result = subprocess.run(
        [sys.executable, "-m", "llm_gent.flow.testing._resume_helper"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=subprocess_timeout,
        check=True,
    )
    parsed = json.loads(result.stdout)
    if not isinstance(parsed, dict):
        raise TypeError(
            f"resume_in_subprocess requires the flow to return a dict; got {type(parsed).__name__}"
        )
    return dict(parsed)
