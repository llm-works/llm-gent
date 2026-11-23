# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shortcuts at run time: on a signal, a flow fast-forwards to its end with what it has.

A run's signals are named events the app sets, given to the run with
``signals=``. :meth:`Flow.with_shortcut` makes a flow F a region that one
of them cuts short. While the signal is set, F and every flow under it
fast-forward (:func:`is_fast_forward`): a chain starts no new step and ends
with its last result, an iterate starts no new pass and returns its carried
value, a map starts no new item (those are ``Skipped``), and a Loop turn in
flight is aborted, its call returning the result SAIA paused it with. What
is already running is neither stopped nor run again: it finishes with what
it has. The step after F then runs as usual. A signal set while the run is
outside every region it cuts short does nothing until such a region starts;
the region then ends at once.

A region that drains (``with_shortcut(signal, drain=True)``) stops
admitting work the same way, but a map item that had started is left out
of it: the item, its chain and everything under it run to their end as
if the region were not cut (:func:`leave_drains`), and its steps see the
drain (:func:`is_draining`). The map item is the unit of work; a chain
step or an iterate pass in flight is not one.

A cut is not a halt: it writes no checkpoint and runs nothing again.
Which signals are set is part of where the run is, though: every
checkpoint the run takes records them (:func:`run_signals`), and resume
sets them again before the first step, so a run halted while it
fast-forwards goes on fast-forwarding, and one halted while it drains goes
on draining: the items the map's cursor holds as running are started.

A region may not contain another region on the same signal
(:func:`check_shortcuts`); regions on different signals may nest.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .._node_id import _child_flows, iter_flows
from .._shortcut import Shortcut, _check_name
from ..flow import Flow
from ..state.snapshot import SIGNALS


if TYPE_CHECKING:
    from ._run import Run
    from .scope import _Frame


def check_signals(signals: Mapping[str, asyncio.Event] | None) -> dict[str, asyncio.Event]:
    """The run's signals as ``signals=`` gives them, checked.

    Raises:
        ValueError: A name is not a non-empty str.
        TypeError: ``signals`` is not a mapping, or an event is not an
            :class:`asyncio.Event`.
    """
    if signals is None:
        return {}
    if not isinstance(signals, Mapping):
        raise TypeError(f"start(signals=) takes a mapping; got {type(signals).__name__}")
    for name, event in signals.items():
        _check_name(name, "a signal's name")
        if not isinstance(event, asyncio.Event):
            raise TypeError(f"signal {name!r} must be an asyncio.Event; got {type(event).__name__}")
    return dict(signals)


def check_shortcuts(root: Flow, signals: Mapping[str, asyncio.Event]) -> None:
    """Raise unless every region in ``root``'s run names one of ``signals`` and holds no other on it.

    Raises:
        RuntimeError: A shortcut's signal is not one of the run's
            ``signals``, or a flow under a region declares a shortcut on
            that region's signal.
    """
    for flow in iter_flows(root):
        if flow._shortcut is not None:
            _check_one(flow, signals)


def _check_one(flow: Flow, signals: Mapping[str, asyncio.Event]) -> None:
    """Raise unless ``flow``'s shortcut is on one of ``signals`` and holds no other on it."""
    signal = flow._shortcut.signal  # type: ignore[union-attr]
    if signal not in signals:
        raise RuntimeError(
            f"Flow {_label(flow)} has with_shortcut({signal!r}) but the run has no such "
            f"signal: pass start(flow, signals={{{signal!r}: event}})"
        )
    inner = _inner_region(flow, signal)
    if inner is not None:
        raise RuntimeError(
            f"Flow {_label(inner)} has with_shortcut({signal!r}) inside Flow {_label(flow)}, "
            f"which has it too: the outer one already cuts everything under it short"
        )


def _inner_region(flow: Flow, signal: str) -> Flow | None:
    """A flow under ``flow`` declaring a shortcut on ``signal``; ``None`` when there is none."""
    seen: set[int] = set()
    stack: list[Flow] = [child for node in flow._nodes for _, child in _child_flows(node)]
    while stack:
        child = stack.pop()
        if id(child) in seen:
            continue
        seen.add(id(child))
        if child._shortcut is not None and child._shortcut.signal == signal:
            return child
        stack.extend(grandchild for node in child._nodes for _, grandchild in _child_flows(node))
    return None


def _label(flow: Flow) -> str:
    """``flow``'s name for messages."""
    return repr(flow._name or "<anonymous>")


class _SetSignals:
    """Cursor at the root: the names of the run's signals that are set."""

    def __init__(self, signals: Mapping[str, asyncio.Event]) -> None:
        self.signals = signals

    def cursor(self) -> dict[str, Any]:
        """``{"signals": [names]}``; nothing while none is set."""
        names = sorted(name for name, event in self.signals.items() if event.is_set())
        return {SIGNALS: names} if names else {}


@contextlib.contextmanager
def run_signals(run: Run) -> Iterator[None]:
    """Keep ``run``'s set signals in its checkpoints while it goes.

    First sets the signals the checked-out snapshot recorded, so the run
    continues as it would have. The record is dropped when the run
    finishes — a later session starts with none set — and kept when the
    halt stops the run, for its halt checkpoint.
    """
    scopes = run.scopes
    _, recorded = scopes.take_cursor((), SIGNALS)
    for name in recorded or ():
        event = run.signals.get(name)
        if event is not None:
            event.set()
    runner = _SetSignals(run.signals)
    scopes.open_cursor((), runner)
    yield
    scopes.close_cursor((), runner)


@dataclass(frozen=True)
class ShortcutRun:
    """One run of a region: the signal that cuts it short, and how.

    A boundary reads ``signal`` (:func:`is_fast_forward`); a Loop turn
    aborts on it while the turn is in the region (:func:`turn_abort`).

    ``drain`` is the region's mode. ``left`` marks the region as one a
    started map item under it has left (:func:`leave_drains`): it no
    longer cuts that item short.
    """

    signal: asyncio.Event
    drain: bool
    left: bool


def is_fast_forward(frame: _Frame) -> bool:
    """True when the flow running under ``frame`` is in a region whose signal is set."""
    return any(s.signal.is_set() for s in frame.scope.shortcuts if not s.left)


def is_draining(frame: _Frame) -> bool:
    """True when ``frame`` runs in a started map item of a drain region whose signal is set."""
    return any(s.signal.is_set() for s in frame.scope.shortcuts if s.left)


@contextlib.asynccontextmanager
async def turn_abort(frame: _Frame) -> AsyncIterator[asyncio.Event | None]:
    """What a Loop turn under ``frame`` aborts on, for the duration of the turn.

    The run's halt, or the signal of any region the turn is in and has not
    left (:func:`leave_drains`). Built per turn from ``frame``, so a region a
    started map item left never reaches a turn in it, however the regions
    nest. One source is used as is; ``None`` when there is none.
    """
    sources = [s.signal for s in frame.scope.shortcuts if not s.left]
    if frame.halt is not None:
        sources.append(frame.halt)
    if len(sources) <= 1:
        yield sources[0] if sources else None
        return
    abort = asyncio.Event()
    tasks = [task for source in sources if (task := follow(source, abort)) is not None]
    try:
        yield abort
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


def leave_drains(shortcuts: tuple[ShortcutRun, ...]) -> tuple[ShortcutRun, ...]:
    """The regions a map item that starts runs under: every drain region is left behind.

    A started item runs to its end however its drain regions are cut;
    fast-forward regions still cut it short.
    """
    return tuple(replace(s, left=True) if s.drain else s for s in shortcuts)


def follow(source: asyncio.Event, target: asyncio.Event) -> asyncio.Task[None] | None:
    """Set ``target`` once ``source`` is set; the task doing it, ``None`` when already set."""
    if source.is_set():
        target.set()
        return None

    async def follow() -> None:
        await source.wait()
        target.set()

    return asyncio.create_task(follow())


def region_run(
    shortcut: Shortcut | None, signals: Mapping[str, asyncio.Event]
) -> ShortcutRun | None:
    """The region one run of a flow declaring ``shortcut`` makes; ``None`` without one.

    ``signals`` are the run's.
    """
    if shortcut is None:
        return None
    return ShortcutRun(signals[shortcut.signal], drain=shortcut.drain, left=False)
