# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One Flow's chain of composition steps at a single walk level.

Extracted from :class:`Flow` in :mod:`.flow`. A :class:`Chain` walks a
Flow's ordered node list, threading each step's return into the next,
and keeps its cursor — the step it is at and that step's input — where
a checkpoint reads it. On ``resume="latest"`` the chain continues at the
cursor the checkpoint saved. After every step it observes the halt and,
when set, stops: with the cursor on that step when the halt interrupted
it, past it when the step completed. The stopped chain stays registered
at its position for the run's halt checkpoint.

Constructed once per :meth:`Flow._run_as_subflow` entry:
``Chain(flow, env)`` eagerly computes ``self.ids`` from
``env.chain_context`` and the flow's nodes. Caller then invokes
``await chain.walk(args, kwargs)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._executor import _build_ctx, _execute_node, _running, _step_inputs
from ._halt_observer import is_halt_signaled, note_halt
from ._node_id import _compute_node_ids
from ._shortcut import SHORTCUT
from .nodes import Interrupted
from .state.snapshot import CHAIN, TURN, path_str


if TYPE_CHECKING:
    from .flow import Flow
    from .nodes import _RunEnv


Inputs = tuple[tuple[Any, ...], dict[str, Any]]


@dataclass(frozen=True)
class _Saved:
    """Where a checkout continues a chain: the step, its input, and whether it had started.

    ``pending`` marks a step the chain stopped before (it moved past a
    completed one); ``prev`` is that completed step's result.
    """

    index: int
    inputs: Inputs
    pending: bool
    prev: Any


class Chain:
    """A Flow's chain of composition steps at one execution level.

    Eagerly resolves node ids in ``__init__`` and holds the flow +
    env for the walk. The chain is single-use — a fresh instance is
    created for each subflow-run entry.
    """

    def __init__(self, flow: Flow, env: _RunEnv) -> None:
        self.flow = flow
        self.env = env
        self.ids: tuple[str, ...] = _compute_node_ids(env.chain_context, flow._nodes)
        # The cursor: the step running and the input it gets. Kept here, not
        # in _walk_steps' locals, so a checkpoint reads it. ``pending``: the
        # chain stopped past a completed step, before this one started;
        # ``prev`` is that completed step's result.
        self.index: int | None = None
        self.step_args: tuple[Any, ...] = ()
        self.step_kwargs: dict[str, Any] = {}
        self.pending = False
        self.prev: Any = None

    def cursor(self) -> dict[str, Any]:
        """The step this chain is at (its node id), that step's input, and its mode.

        ``resume="latest"`` continues the chain at this step with this
        input (:meth:`_saved_step`). A step the chain stopped before also
        carries ``pending`` and the result it follows (``prev``); a chain
        in shortcut mode carries ``shortcut``.
        """
        if self.index is None:
            return {}
        step: dict[str, Any] = {
            "step": self.ids[self.index],
            "args": list(self.step_args),
            "kwargs": dict(self.step_kwargs),
        }
        if self.pending:
            step["pending"], step["prev"] = True, self.prev
        if self.env.shortcut is not None and self.env.shortcut.active:
            step[SHORTCUT] = True
        return {CHAIN: step}

    async def walk(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Execute chain steps in order, threading returns; return the last result.

        Starts at the first step, or at the step a checkout saved for this
        chain. The chain's cursor is in every snapshot taken while it walks.
        """
        saved = self._saved_step()
        with _running(self.env, self.env.path, self):
            return await self._walk_steps(saved, args, kwargs)

    def _saved_step(self) -> _Saved | None:
        """Where a checkout continues this chain; ``None`` when not resuming.

        Steps before it completed before the checkpoint and do not run
        again; the step itself runs with the input it had.

        Raises:
            RuntimeError: The saved step is no longer in this chain.
        """
        found, step = self.env.scopes.take_cursor(self.env.path, CHAIN)
        if not found:
            return None
        if step["step"] not in self.ids:
            where = path_str((*self.env.path, CHAIN))
            label = self.flow._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r}: the checkpoint's cursor at {where!r} is at step "
                f"{step['step']!r}, which this chain no longer has"
            )
        index = self.ids.index(step["step"])
        inputs = (tuple(step["args"]), step["kwargs"])
        return _Saved(index, inputs, bool(step.get("pending")), step.get("prev"))

    async def _walk_steps(
        self, saved: _Saved | None, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Execute chain steps from the first (or the saved) one onward, threading returns.

        A saved step gets the input it had when the checkpoint was taken.
        The cursor moves to each step, with the step's input (after
        ``project``), before the step runs; after it returns,
        :meth:`_halted_after` observes the halt. In shortcut mode only the
        step that was interrupted runs before the landing step
        (:meth:`_shortcut_jump`).

        Raises:
            Interrupted: The halt stopped the chain before its last step
                completed.
        """
        index = saved.index if saved is not None else 0
        rerun = saved.index if saved is not None and not saved.pending else None
        last = (True, saved.prev) if saved is not None and saved.pending else (False, None)
        while index < len(self.flow._nodes):
            jump = self._shortcut_jump(index, rerun, last, args, kwargs)
            if jump is not None:
                target, inputs = jump
                if target is None:
                    return _passed_through(last, args, kwargs)
                index = target
            elif saved is not None and index == saved.index:
                inputs = saved.inputs
            else:
                inputs = _step_inputs(index, self.flow._nodes[index], last[1], args, kwargs)
            last = (True, await self._step(index, inputs, args, kwargs))
            index += 1
        return last[1]

    async def _step(
        self, index: int, inputs: Inputs, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Run step ``index`` with ``inputs``; its result, unless the halt stops the chain.

        Raises:
            Interrupted: The halt stopped the chain at or after this step.
        """
        node, node_id = self.flow._nodes[index], self.ids[index]
        self.index, (self.step_args, self.step_kwargs), self.pending = index, inputs, False
        result, interrupted = await self._run_step(node, node_id, *inputs)
        if self._halted_after(index, result, interrupted, args, kwargs):
            # The chain stops before its end: it is interrupted, so the step
            # running it (a .call, an iterate pass, a map item) is too. What
            # the halt stopped under this step stays registered where it
            # stopped, for the run's halt checkpoint.
            raise Interrupted()
        # Past this step: a paused turn left under it was paused by a halt
        # that is not the run's, and belongs to no later snapshot.
        self.env.scopes.close_under(self.env.owner_path(node_id))
        return result

    def _shortcut_jump(
        self,
        index: int,
        rerun: int | None,
        last: tuple[bool, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[int | None, Inputs] | None:
        """Where a shortcut takes the chain before step ``index``; ``None`` to run it.

        Reaching the landing step uses the shortcut up. Before it, in
        shortcut mode, only the step that was interrupted (``rerun``)
        runs; any other jumps to the landing step with the last completed
        result as its input (the chain's input when none completed), or
        ends the chain (landing step ``None``).
        """
        shortcut = self.env.shortcut
        if shortcut is None or shortcut.landed:
            return None
        target = shortcut.to_index
        if target is not None and index >= target:
            shortcut.active, shortcut.landed = False, True
            return None
        if not shortcut.active or index == rerun:
            return None
        shortcut.active, shortcut.landed = False, True
        if target is None:
            return None, (args, kwargs)
        has_result, prev = last
        if not has_result:
            return target, (args, kwargs)
        return target, _step_inputs(target, self.flow._nodes[target], prev, args, kwargs)

    async def _run_step(
        self, node: Any, node_id: str, node_args: tuple[Any, ...], node_kwargs: dict[str, Any]
    ) -> tuple[Any, bool]:
        """Run one step; return its result and whether the halt interrupted it.

        Interrupted means the step stopped before finishing its work: it
        raised :class:`Interrupted` — itself, or a chain, iterate or map it
        runs that stopped early — or a Loop call in it left a paused SAIA
        turn.

        Raises:
            RuntimeError: The step raised :class:`Interrupted` while no halt
                was set.
        """
        ctx = _build_ctx(node.target, self.env, node_id)
        try:
            result = await _execute_node(node, ctx, self.env, node_args, node_kwargs, node_id)
        except Interrupted:
            if not is_halt_signaled(self.env):
                label = self.flow._name or "<anonymous>"
                raise RuntimeError(
                    f"Flow {label!r}: step {node_id!r} raised Interrupted while no halt is set"
                ) from None
            return None, True
        return result, self.env.scopes.holds_under(self.env.owner_path(node_id), TURN)

    def _halted_after(
        self,
        index: int,
        result: Any,
        interrupted: bool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        """After a step: True when the halt stops the chain here, with its cursor set.

        An interrupted step keeps the cursor: a checkout runs it again with
        the input it had. A completed step moves the cursor to the next step
        and its input, so a checkout continues there. After a completed last
        step the chain does not stop: it returns, and the enclosing
        structure decides where the run continues (an iterate's next pass,
        the parent chain's next step, or the end of the run).

        The chain does not write the halt checkpoint: :meth:`Flow.run` writes
        it once everything has stopped, from the positions left registered.
        """
        if not is_halt_signaled(self.env):
            return False
        at = index
        if not interrupted:
            if index + 1 == len(self.flow._nodes):
                return False
            at = index + 1
            self._move_to(at, result, args, kwargs)
        note_halt(self.env, 0, self.ids[at])
        return True

    def _move_to(
        self, index: int, prev_result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Move the cursor to step ``index`` (not started) with its input after ``prev_result``."""
        node = self.flow._nodes[index]
        self.index, self.pending, self.prev = index, True, prev_result
        self.step_args, self.step_kwargs = _step_inputs(index, node, prev_result, args, kwargs)


def _passed_through(last: tuple[bool, Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """What a chain a shortcut ended returns: its last completed result, else its input.

    The input passes through when it is one positional value; otherwise
    nothing does (``None``).
    """
    has_result, prev = last
    if has_result:
        return prev
    return args[0] if len(args) == 1 and not kwargs else None
