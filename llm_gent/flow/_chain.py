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

from typing import TYPE_CHECKING, Any

from ._executor import _build_ctx, _execute_node, _running, _step_inputs
from ._halt_observer import is_halt_signaled, note_halt
from ._node_id import _compute_node_ids
from .nodes import UNSET, Interrupted
from .state.snapshot import CHAIN, TURN, path_str


if TYPE_CHECKING:
    from .flow import Flow
    from .nodes import _RunEnv


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
        # in _walk_steps' locals, so a checkpoint reads it.
        self.index: int | None = None
        self.step_args: tuple[Any, ...] = ()
        self.step_kwargs: dict[str, Any] = {}

    def cursor(self) -> dict[str, Any]:
        """The step this chain is at (its node id) and that step's input.

        ``resume="latest"`` continues the chain at this step with this
        input (:meth:`_saved_step`).
        """
        if self.index is None:
            return {}
        step = {
            "step": self.ids[self.index],
            "args": list(self.step_args),
            "kwargs": dict(self.step_kwargs),
        }
        return {CHAIN: step}

    async def walk(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Execute chain steps in order, threading returns; return the last result.

        Starts at the first step, or at the step a checkout saved for this
        chain. The chain's cursor is in every snapshot taken while it walks.
        """
        start_index, first_input = 0, None
        saved = self._saved_step()
        if saved is not None:
            start_index, first_input = saved
        with _running(self.env, self.env.path, self):
            return await self._walk_steps(start_index, args, kwargs, first_input)

    def _saved_step(self) -> tuple[int, tuple[tuple[Any, ...], dict[str, Any]]] | None:
        """The step and input a checkout continues this chain at; ``None`` when not resuming.

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
        return self.ids.index(step["step"]), (tuple(step["args"]), step["kwargs"])

    async def _walk_steps(
        self,
        start_index: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        first_input: tuple[tuple[Any, ...], dict[str, Any]] | None,
    ) -> Any:
        """Execute chain steps from ``start_index`` onward, threading returns.

        With ``first_input`` (a checkout) the step at ``start_index`` gets
        the input it had when the checkpoint was taken. The cursor moves
        to each step, with the step's input (after ``project``), before the
        step runs; after it returns, :meth:`_halted_after` observes the halt.

        Raises:
            Interrupted: The halt stopped the chain before its last step
                completed.
        """
        result: Any = UNSET
        for index in range(start_index, len(self.flow._nodes)):
            node = self.flow._nodes[index]
            node_id = self.ids[index]
            node_args: tuple[Any, ...]
            node_kwargs: dict[str, Any]
            if index == start_index and first_input is not None:
                node_args, node_kwargs = first_input
            else:
                node_args, node_kwargs = _step_inputs(index, node, result, args, kwargs)
            self.index, self.step_args, self.step_kwargs = index, node_args, node_kwargs
            result, interrupted = await self._run_step(node, node_id, node_args, node_kwargs)
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
        """Move the cursor to step ``index`` with the input it gets after ``prev_result``."""
        node = self.flow._nodes[index]
        self.index = index
        self.step_args, self.step_kwargs = _step_inputs(index, node, prev_result, args, kwargs)
