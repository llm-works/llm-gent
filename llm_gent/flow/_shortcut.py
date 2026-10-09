# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shortcuts: on a signal, a flow fast-forwards to its end with what it has.

A run's signals are named events the app sets, declared on its top-level
flow with :meth:`Flow.with_signal`. :meth:`Flow.with_shortcut` makes a
flow F a region that one of them cuts short. While the signal is set, F
and every flow under it fast-forward (:func:`is_fast_forward`): a chain
starts no new step and ends with its last result, an iterate starts no
new pass and returns its carried value, a map starts no new item (those
are ``Skipped``), and a Loop
turn in flight is aborted, its call returning the result SAIA paused it
with. What is already running is neither stopped nor run again: it
finishes with what it has. The step after F then runs as usual. A signal
set while the run is outside every region it cuts short does nothing
until such a region starts; the region then ends at once.

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
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .state.snapshot import SIGNALS


if TYPE_CHECKING:
    from .flow import Flow
    from .nodes import _RunEnv


@dataclass(frozen=True)
class Shortcut:
    """What :meth:`Flow.with_shortcut` declares: the signal that cuts the flow short, and how."""

    signal: str
    drain: bool
    """Started map items run to their end instead of fast-forwarding."""


def check_signal(name: object, event: object) -> tuple[str, asyncio.Event]:
    """``(name, event)`` when :meth:`Flow.with_signal` accepts them.

    Raises:
        ValueError: ``name`` is not a non-empty str.
        TypeError: ``event`` is not an :class:`asyncio.Event`.
    """
    _check_name(name, ".with_signal(name)")
    if not isinstance(event, asyncio.Event):
        raise TypeError(f"with_signal takes an asyncio.Event; got {type(event).__name__}")
    return name, event  # type: ignore[return-value]


def check_shortcut(signal: object, drain: object) -> Shortcut:
    """The :class:`Shortcut` :meth:`Flow.with_shortcut` declares, when its arguments are valid.

    Raises:
        ValueError: ``signal`` is not a non-empty str.
        TypeError: ``drain`` is not a bool.
    """
    _check_name(signal, ".with_shortcut(signal)")
    if not isinstance(drain, bool):
        raise TypeError(f"with_shortcut(drain=) takes a bool; got {type(drain).__name__}")
    return Shortcut(signal, drain)  # type: ignore[arg-type]


def _check_name(name: object, what: str) -> None:
    """Raise ``ValueError`` naming ``what`` unless ``name`` is a non-empty str."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} must be a non-empty str; got {name!r}")


def check_shortcuts(root: Flow) -> None:
    """Raise unless signals are on ``root`` alone, every region names one, and no region holds another on the same signal.

    Raises:
        RuntimeError: A nested flow declares a signal; a shortcut's signal
            is not one of ``root``'s; or a flow under a region declares a
            shortcut on that region's signal.
    """
    from ._node_id import iter_flows

    for flow in iter_flows(root):
        if flow is not root and flow._signals:
            raise RuntimeError(
                f"Flow {_label(flow)} declares a signal but runs inside {_label(root)}: "
                f"a run's signals are declared on its top-level flow with with_signal()"
            )
        if flow._shortcut is not None:
            _check_one(flow, root)


def _check_one(flow: Flow, root: Flow) -> None:
    """Raise unless ``flow``'s shortcut is on one of ``root``'s signals and holds no other on it."""
    signal = flow._shortcut.signal  # type: ignore[union-attr]
    if signal not in root._signals:
        raise RuntimeError(
            f"Flow {_label(flow)} has with_shortcut({signal!r}) but the run declares no such "
            f"signal: call with_signal({signal!r}, event) on {_label(root)}"
        )
    inner = _inner_region(flow, signal)
    if inner is not None:
        raise RuntimeError(
            f"Flow {_label(inner)} has with_shortcut({signal!r}) inside Flow {_label(flow)}, "
            f"which has it too: the outer one already cuts everything under it short"
        )


def _inner_region(flow: Flow, signal: str) -> Flow | None:
    """A flow under ``flow`` declaring a shortcut on ``signal``; ``None`` when there is none."""
    from ._node_id import _child_flows

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

    def __init__(self, signals: dict[str, asyncio.Event]) -> None:
        self.signals = signals

    def cursor(self) -> dict[str, Any]:
        """``{"signals": [names]}``; nothing while none is set."""
        names = sorted(name for name, event in self.signals.items() if event.is_set())
        return {SIGNALS: names} if names else {}


@contextlib.contextmanager
def run_signals(flow: Flow) -> Iterator[None]:
    """Keep ``flow``'s set signals in the run's checkpoints while the run goes.

    First sets the signals the checked-out snapshot recorded, so the run
    continues as it would have. The record is dropped when the run
    finishes — a later session starts with none set — and kept when the
    halt stops the run, for its halt checkpoint.
    """
    scopes = flow._scopes
    _, recorded = scopes.take_cursor((), SIGNALS)
    for name in recorded or ():
        event = flow._signals.get(name)
        if event is not None:
            event.set()
    runner = _SetSignals(flow._signals)
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


def is_fast_forward(env: _RunEnv) -> bool:
    """True when the flow running under ``env`` is in a region whose signal is set."""
    return any(s.signal.is_set() for s in env.shortcuts if not s.left)


def is_draining(env: _RunEnv) -> bool:
    """True when ``env`` runs in a started map item of a drain region whose signal is set."""
    return any(s.signal.is_set() for s in env.shortcuts if s.left)


@contextlib.asynccontextmanager
async def turn_abort(env: _RunEnv) -> AsyncIterator[asyncio.Event | None]:
    """What a Loop turn under ``env`` aborts on, for the duration of the turn.

    The run's halt, or the signal of any region the turn is in and has not
    left (:func:`leave_drains`). Built per turn from ``env``, so a region a
    started map item left never reaches a turn in it, however the regions
    nest. One source is used as is; ``None`` when there is none.
    """
    sources = [s.signal for s in env.shortcuts if not s.left]
    if env.halt is not None:
        sources.append(env.halt)
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


def region_run(flow: Flow, signals: dict[str, asyncio.Event]) -> ShortcutRun | None:
    """The region one run of ``flow`` makes; ``None`` when it declares no shortcut.

    ``signals`` are the run's.
    """
    shortcut = flow._shortcut
    if shortcut is None:
        return None
    return ShortcutRun(signals[shortcut.signal], drain=shortcut.drain, left=False)
