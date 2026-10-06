# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shortcuts: on a signal, a flow continues at a later step with what it has.

A run's signals are named events the app sets, declared on its top-level
flow with :meth:`Flow.with_signal`. :meth:`Flow.with_shortcut` declares,
on any flow F, a shortcut on one of them. When the signal is set, F stops
the way the run's halt stops it — every part under F at its next
boundary, a Loop turn paused and held. Unlike a halt, the run does not
end there: F continues at once from where it stopped, in shortcut mode,
and ends at its step ``to`` (or its end) with what the stopped work
produced. A halt is a pause; a shortcut is a jump.

Which signals are set is part of where the run is: every checkpoint the
run takes records them (:func:`run_signals`), and resume sets them again
before the first step. A flow that starts while its signal is set — a
later step, the next iterate pass, a map item — starts in shortcut mode.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .nodes import step_name
from .state.snapshot import CHAIN, SIGNALS, ScopePath, ScopeRegistry


if TYPE_CHECKING:
    from .flow import Flow


SHORTCUT = "shortcut"
"""Field of a chain's cursor: the chain is in shortcut mode, on its way to its landing step."""


@dataclass(frozen=True)
class Shortcut:
    """What :meth:`Flow.with_shortcut` declares: the signal and the step it lands on.

    ``signal`` names one of the run's signals; ``to`` is the ``name=`` of a
    step of the declaring flow's chain, ``None`` its end.
    """

    signal: str
    to: str | None


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


def check_shortcut(signal: object, to: object) -> Shortcut:
    """The :class:`Shortcut` :meth:`Flow.with_shortcut` declares, when its arguments are valid.

    Raises:
        ValueError: ``signal`` is not a non-empty str, or ``to`` is neither
            ``None`` nor one.
    """
    _check_name(signal, ".with_shortcut(signal)")
    if to is not None:
        _check_name(to, ".with_shortcut(to=)")
    return Shortcut(signal, to)  # type: ignore[arg-type]


def _check_name(name: object, what: str) -> None:
    """Raise ``ValueError`` naming ``what`` unless ``name`` is a non-empty str."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} must be a non-empty str; got {name!r}")


def check_shortcuts(root: Flow) -> None:
    """Raise unless signals are on ``root`` alone and every shortcut is on one and lands somewhere.

    Checked at run start: the fluent API lets ``with_shortcut(to=...)``
    come before the step it names is appended.

    Raises:
        RuntimeError: A nested flow declares a signal; a shortcut's signal
            is not one of ``root``'s; or its ``to`` names no step of the
            declaring flow's chain, or more than one.
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
    """Raise unless ``flow``'s shortcut is on one of ``root``'s signals and lands on one step."""
    shortcut = flow._shortcut
    assert shortcut is not None
    if shortcut.signal not in root._signals:
        raise RuntimeError(
            f"Flow {_label(flow)} has with_shortcut({shortcut.signal!r}) but the run declares "
            f"no such signal: call with_signal({shortcut.signal!r}, event) on {_label(root)}"
        )
    if shortcut.to is None:
        return
    count = sum(1 for node in flow._nodes if step_name(node) == shortcut.to)
    if count != 1:
        found = "no step" if count == 0 else f"{count} steps"
        raise RuntimeError(
            f"Flow {_label(flow)} has with_shortcut(to={shortcut.to!r}) but {found} of its "
            f"chain is named {shortcut.to!r}: name exactly one step with name="
        )


def _label(flow: Flow) -> str:
    """``flow``'s name for messages."""
    return repr(flow._name or "<anonymous>")


def target_index(flow: Flow) -> int | None:
    """Chain index of the step ``flow``'s shortcut lands on; ``None`` for its end."""
    shortcut = flow._shortcut
    assert shortcut is not None
    if shortcut.to is None:
        return None
    return next(i for i, node in enumerate(flow._nodes) if step_name(node) == shortcut.to)


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
    """One run of a flow that declares a shortcut.

    ``stop`` is the halt the run observes: set by ``parent`` (the halt it
    would observe without a shortcut) and by the signal ``event`` while
    the chain has not reached ``to_index`` (``None``: the flow's end).
    ``active`` is shortcut mode, recorded in the chain's cursor; it ends
    when the chain lands, and ``landed`` stays set from then on: the
    shortcut is used up.
    """

    event: asyncio.Event
    to_index: int | None
    stop: asyncio.Event
    parent: asyncio.Event | None
    active: bool = False
    landed: bool = False
    continued: bool = field(default=False, repr=False)

    def take_over(self, run_halted: bool) -> bool:
        """After the run stopped: True when it continues now, in shortcut mode.

        A stop with the signal set puts the run in shortcut mode, recorded
        in its chain's cursor — also when the enclosing halt (``parent``,
        or the run's: ``run_halted``) is set too: the run then stops with
        everything else, and resume continues the shortcut. Only a stop by
        the shortcut alone continues at once, and only once.
        """
        if self.landed or not self.event.is_set():
            return False
        self.active = True
        halted = run_halted or (self.parent is not None and self.parent.is_set())
        if halted or self.continued:
            return False
        self.continued = True
        self.stop.clear()
        return True

    @property
    def pending(self) -> bool:
        """True when the signal is set and the run has neither taken it over nor landed.

        Boundaries check this directly (:func:`~._halt_observer.is_halt_signaled`):
        :attr:`stop` follows the signal one loop tick later, and a step
        that sets the signal without awaiting reaches its boundary first.
        """
        return self.event.is_set() and not self.active and not self.landed

    async def _watch(self) -> None:
        """Stop the run once the signal is set, unless it has landed or taken it over by then.

        A boundary can see the signal first and the run take it over
        (:meth:`take_over` clears :attr:`stop`) before this task runs;
        setting :attr:`stop` then would stop the continuation.
        """
        await self.event.wait()
        if not self.landed and not self.active:
            self.stop.set()


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
    flow: Flow,
    path: ScopePath,
    scopes: ScopeRegistry,
    halt: asyncio.Event | None,
    signals: dict[str, asyncio.Event],
) -> AsyncIterator[ShortcutRun | None]:
    """The shortcut of one run of ``flow`` at ``path``; ``None`` when it declares none.

    ``signals`` are the run's. A run whose checked-out chain cursor was
    in shortcut mode, or that starts with its signal set, starts in
    shortcut mode.
    """
    shortcut = flow._shortcut
    if shortcut is None:
        yield None
        return
    event = signals[shortcut.signal]
    saved = scopes.peek_cursor(path, CHAIN)
    resumed = isinstance(saved, dict) and bool(saved.get(SHORTCUT))
    run = ShortcutRun(event, target_index(flow), asyncio.Event(), halt)
    run.active = resumed or event.is_set()
    tasks = [follow(halt, run.stop)]
    if not run.active:
        tasks.append(asyncio.create_task(run._watch()))
    try:
        yield run
    finally:
        for task in tasks:
            if task is not None:
                task.cancel()
