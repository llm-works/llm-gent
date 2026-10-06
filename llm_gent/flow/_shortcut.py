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

A cut is not a halt: it writes no checkpoint and runs nothing again.
Which signals are set is part of where the run is, though: every
checkpoint the run takes records them (:func:`run_signals`), and resume
sets them again before the first step, so a run halted while it
fast-forwards goes on fast-forwarding.

A region may not contain another region on the same signal
(:func:`check_shortcuts`); regions on different signals may nest.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .state.snapshot import SIGNALS


if TYPE_CHECKING:
    from .flow import Flow
    from .nodes import _RunEnv


@dataclass(frozen=True)
class Shortcut:
    """What :meth:`Flow.with_shortcut` declares: the signal that cuts the flow short."""

    signal: str


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


def check_shortcut(signal: object) -> Shortcut:
    """The :class:`Shortcut` :meth:`Flow.with_shortcut` declares, when ``signal`` is valid.

    Raises:
        ValueError: ``signal`` is not a non-empty str.
    """
    _check_name(signal, ".with_shortcut(signal)")
    return Shortcut(signal)  # type: ignore[arg-type]


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


@dataclass
class ShortcutRun:
    """One run of a region: its signal, and the event a Loop turn in it aborts on.

    ``stop`` is set by ``signal`` and by what a turn around the region
    aborts on — the run's halt or the enclosing region's ``stop``
    (:func:`run_shortcut`). SAIA observes it as ``abort_signal``; a
    boundary reads ``signal`` itself (:func:`is_fast_forward`).
    """

    signal: asyncio.Event
    stop: asyncio.Event


def is_fast_forward(env: _RunEnv) -> bool:
    """True when the flow running under ``env`` is in a region whose signal is set."""
    return any(shortcut.signal.is_set() for shortcut in env.shortcuts)


def abort_event(env: _RunEnv) -> asyncio.Event | None:
    """What a Loop turn under ``env`` aborts on: the innermost region's stop, else the run's halt."""
    return env.shortcuts[-1].stop if env.shortcuts else env.halt


def follow(
    enclosing: asyncio.Event | None, stop: asyncio.Event | None
) -> asyncio.Task[None] | None:
    """Set ``stop`` once ``enclosing`` is set; the task doing it, ``None`` when nothing to do."""
    if enclosing is None or stop is None:
        return None
    if enclosing.is_set():
        stop.set()
        return None

    async def follow() -> None:
        await enclosing.wait()
        stop.set()

    return asyncio.create_task(follow())


@contextlib.asynccontextmanager
async def run_shortcut(
    flow: Flow, parent: asyncio.Event | None, signals: dict[str, asyncio.Event]
) -> AsyncIterator[ShortcutRun | None]:
    """The region one run of ``flow`` makes; ``None`` when it declares no shortcut.

    ``parent`` is what a Loop turn around it aborts on (:func:`abort_event`);
    ``signals`` are the run's.
    """
    shortcut = flow._shortcut
    if shortcut is None:
        yield None
        return
    signal = signals[shortcut.signal]
    run = ShortcutRun(signal, asyncio.Event())
    tasks = [follow(parent, run.stop), follow(signal, run.stop)]
    try:
        yield run
    finally:
        for task in tasks:
            if task is not None:
                task.cancel()
        for task in tasks:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
