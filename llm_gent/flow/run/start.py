# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""How a run starts: fresh, or from a commit of its history.

``at`` names the commit: ``"latest"`` (the newest commit with state), a
commit hash as ``ctx.checkpoint()`` returns it, or a checkpoint name as
``ctx.checkpoint(name)`` took it. Resuming and restarting need the run's
``history``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..checkpoint import checkpoint_tag, is_commit_hash


@dataclass(frozen=True)
class Fresh:
    """Run from the ``state`` given; with a history, the run's commits append to it."""


@dataclass(frozen=True)
class Resume:
    """Continue where the run was at the commit ``at``.

    The commit's root state replaces ``state``, every scope comes back,
    and every chain, iterate and branch that was running continues at its
    saved step, pass and arm, so only the step running at the checkpoint
    runs again; a Loop that paused mid-turn resumes that turn. Payloads
    are rebuilt via ``state_factory`` when the flow has one, else used as
    plain dicts. With ``"latest"`` on an empty history the run proceeds
    with ``state`` as given; a corrupt history raises
    :class:`~llm_gent.flow.HistoryCorrupt`. Once the run commits, the
    commits after ``at`` leave the history's line and stay resumable by
    hash until a prune deletes them; a run that fails before its first
    commit leaves ``HEAD`` where it was.

    Halt, signals' events, cost tracker, SAIA, traits and logger are never
    serialized: the resumed run uses the ones it is given now. ``max_iters``
    is cumulative across resumes (the pass counter is restored); a deadline
    starts fresh.

    Raises:
        ValueError: ``at`` is not ``"latest"``, a commit hash or a valid
            checkpoint name.
    """

    at: str = "latest"

    def __post_init__(self) -> None:
        _check_at(self.at, "Resume")


@dataclass(frozen=True)
class Restart:
    """Start over from the data of the commit ``at``.

    The run takes the commit's root state in place of ``state`` and the
    accounting of the resources the flow declares (its cost tracker's
    spend); nothing else. Every chain starts at its first step, no scope,
    paused turn or set signal comes back, and per-run resources start
    fresh, so a flow whose structure changed since the commit was written
    runs normally where :class:`Resume` would fail on a saved step the
    flow no longer has. The run's commits are parented on the commit. A
    commit older than the head brings its resources back as they were
    then: spend recorded after it is not counted. With ``"latest"`` on an
    empty history the run proceeds with ``state`` as given.

    Raises:
        ValueError: ``at`` is not ``"latest"``, a commit hash or a valid
            checkpoint name.
    """

    at: str = "latest"

    def __post_init__(self) -> None:
        _check_at(self.at, "Restart")


Start = Fresh | Resume | Restart
"""How a run starts (:class:`Fresh`, :class:`Resume` or :class:`Restart`)."""

FRESH = Fresh()


def _check_at(at: object, what: str) -> None:
    """Raise ``ValueError`` unless ``at`` is ``"latest"``, a commit hash or a checkpoint name."""
    if at == "latest" or is_commit_hash(at):
        return
    try:
        checkpoint_tag(at)  # type: ignore[arg-type]
    except ValueError as e:
        raise ValueError(
            f"{what}(at=) takes 'latest', a commit hash or a checkpoint name; got {at!r} ({e})"
        ) from None
