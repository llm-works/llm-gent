# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""One Flow's chain of composition steps at a single walk level.

Extracted from :class:`Flow` in :mod:`.flow`. A :class:`Chain` owns
everything the executor needs to walk a Flow's ordered node list
correctly under a potentially-in-progress resume replay: the
per-node id array, replay reachability checks, resume start-index
selection, the async step-by-step walk that threads returns
node-to-node, and halt observation between steps + at the trailing
edge.

Constructed once per :meth:`Flow._run_as_subflow` entry:
``Chain(flow, env)`` eagerly computes ``self.ids`` from
``env.chain_context`` and the flow's nodes. Caller then invokes
``await chain.walk(args, kwargs)`` — the single public entry
resolves reachability + start_index internally and drives the walk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._executor import _build_ctx, _execute_node, _step_inputs
from ._halt_observer import HaltSaveObserver, is_halt_signaled
from ._node_id import _compute_node_id
from .nodes import UNSET, _Iterate, _Node, _RunEnv


if TYPE_CHECKING:
    from .flow import Flow


class Chain:
    """A Flow's chain of composition steps at one execution level.

    Eagerly resolves node ids in ``__init__`` and holds the flow +
    env for the walk. The chain is single-use — a fresh instance is
    created for each subflow-run entry.
    """

    def __init__(self, flow: Flow, env: _RunEnv) -> None:
        self.flow = flow
        self.env = env
        self.ids: tuple[str, ...] = tuple(
            _compute_node_id(env.chain_context, n, i) for i, n in enumerate(flow._nodes)
        )

    async def walk(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Execute chain steps in order, threading returns; return the last result.

        First asserts the resume replay (if any) can reach a step at
        this level, then selects the start index (0 on a fresh run,
        the on-path index on resume), then walks. Halt observation
        between steps and after the trailing edge is internal.
        """
        self._assert_replay_reachable()
        start_index = self._resume_start_index()
        return await self._walk_steps(start_index, args, kwargs)

    def _assert_replay_reachable(self) -> None:
        """Fail-fast: raise if the replay's remaining head is unreachable at this level.

        If a resume replay is threaded and its ``remaining_path[0]``
        does not match any of this level's ``ids``, the composition
        graph has changed since the checkpoint was written and there
        is no descent from this level that could reach the save-point.
        Raising here prevents unrelated chain steps and iterates from
        running fresh under a doomed replay — the whole point of the
        head-pop redesign. No-op when there is no replay, the path is
        empty, or the leaf has already been consumed.
        """
        replay = self.env.replay
        if replay is None or not replay.remaining_path:
            return
        if self.env.runtime._replay_consumed:
            return
        head = replay.remaining_path[0]
        if head in self.ids:
            return
        label = self.flow._name or "<anonymous>"
        ids_repr = ", ".join(self.ids) if self.ids else "<empty>"
        full_repr = " → ".join(replay.full_path) if replay.full_path else "<empty>"
        raise RuntimeError(
            f"Flow {label!r}: resume path head {head!r} not found in this level's "
            f"chain step ids [{ids_repr}] — the composition graph has "
            f"structurally changed since the checkpoint was written. "
            f"Saved path (root→leaf): {full_repr}"
        )

    def _resume_start_index(self) -> int:
        """Return the chain index to start execution from on resume.

        ``0`` on a fresh run (no replay, empty path, or already-
        consumed leaf). On resume, returns the index of the chain
        step whose id matches ``env.replay.remaining_path[0]`` —
        that step is on the save-point ancestor chain (or IS the
        save-point leaf when ``remaining_path`` has shrunk to one
        entry); every prior chain step completed BEFORE the
        checkpoint was written and re-running them would clobber
        hydrated state (their verbs typically write to
        ``ctx.state.data`` on the way through).

        :meth:`_assert_replay_reachable` guarantees the head is in
        ``self.ids`` before this fires, so the loop is a lookup, not
        a search.

        Contract on the on-path node when ``start_index > 0``: its
        predecessor didn't re-run, so there is no return value to
        thread in. At the top-level chain it receives the input stored
        on the halt commit (``replay.step_input``) when that input
        survived a JSON round trip; otherwise it runs with no
        ``prev_result``, as does an on-path step of a nested chain.
        Steps that depend on their input in those cases must either be
        at chain index 0 or read from state.
        """
        replay = self.env.replay
        if replay is None or not replay.remaining_path:
            return 0
        if self.env.runtime._replay_consumed:
            return 0
        head = replay.remaining_path[0]
        for i, cid in enumerate(self.ids):
            if cid == head:
                # Single-element remaining path AND target is a plain-verb
                # or Branch/Map/subflow chain step means the halt-observation
                # site saved here — mark consumed so ``_assert_replay_consumed``
                # does not fire. Iterate has its own consumption in
                # ``_resolve_iterate_resume``; Branch/Map/subflow do not
                # (after pop the path is empty, nothing inside will consume it).
                if len(replay.remaining_path) == 1 and not isinstance(
                    self.flow._nodes[i].target, _Iterate
                ):
                    self.env.runtime._replay_consumed = True
                return i
        return 0

    async def _walk_steps(
        self,
        start_index: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute chain steps from ``start_index`` onward, threading returns.

        ``start_index > 0`` on resume: predecessors already completed
        before the checkpoint was written; the on-path step at
        ``start_index`` runs with the input stored on the halt commit,
        or none (see :meth:`_resume_start_index` for the contract).

        Halt observation: between chain steps (never at the very
        first iteration of this walk, so resume runs at least the
        halted step), if ``env.halt`` is set, stamp a halted commit
        and break. On a subsequent ``run(resume="replay")``, that ref
        resolves to this commit and the walk restarts at the step it
        anchors at.
        """
        result: Any = UNSET
        for index in range(start_index, len(self.flow._nodes)):
            node = self.flow._nodes[index]
            node_args, node_kwargs = self._inputs_for(
                index, start_index, node, result, args, kwargs
            )
            if await self._observe_halt_between(index, start_index):
                break
            node_id = self.ids[index]
            ctx = _build_ctx(node.target, self.env, node_id)
            result = await _execute_node(node, ctx, self.env, node_args, node_kwargs, node_id)
        else:
            # for-else: chain exhausted without a between-steps halt-save. If halt
            # cut the LAST step short, save there — no next step exists to save at.
            await self._observe_halt_trailing()
        return result

    def _inputs_for(
        self,
        index: int,
        start_index: int,
        node: _Node,
        result: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return step ``index``'s (args, kwargs); record a top-level step's input.

        The resumed step (``index == start_index > 0``) gets the input
        its halt commit carried, or none. Top-level steps past index 0
        record their input so a halt commit landing on them can store
        it; index 0 re-receives the run's own inputs on resume.
        """
        if index == start_index and start_index > 0:
            node_args: tuple[Any, ...] = self._restored_input(index)
            node_kwargs: dict[str, Any] = {}
        else:
            node_args, node_kwargs = _step_inputs(index, node, result, args, kwargs)
        if index > 0 and node_args and self.env.runtime is self.flow:
            self.env.step_inputs.record(self.ids[index], node_args[0])
        return node_args, node_kwargs

    def _restored_input(self, index: int) -> tuple[Any, ...]:
        """The input the replayed halt commit stored for step ``index``, as args; else ``()``."""
        replay = self.env.replay
        step_input = replay.step_input if replay is not None else None
        if step_input is None or step_input[0] != self.ids[index]:
            return ()
        return (step_input[1],)

    async def _observe_halt_between(self, index: int, start_index: int) -> bool:
        """Save a halt commit between chain steps when appropriate; return True if saved.

        Only fires at the top-level chain (``env.runtime is self.flow``)
        with a checkpointer bound and past the first step of this walk
        (so resume runs at least the halted step). Nested body chains
        let halt propagate to iterate boundaries where iteration state
        is consistent.

        When the just-completed step was cut short by the halt (it owns
        an ``env.cut_short`` entry: a paused Loop, a map that skipped
        items, a verb that called :meth:`Context.mark_cut_short`), the
        halt commit anchors per :meth:`_cut_short_anchor` so resume
        re-runs it. Otherwise saves at the not-yet-run step (the
        normal chain-halt case).
        """
        env = self.env
        if (
            index <= start_index
            or env.runtime is not self.flow
            or env.checkpoint_ctx is None
            or not is_halt_signaled(env)
        ):
            return False
        just_completed = index - 1
        anchor = (
            self._cut_short_anchor(just_completed)
            if env.cut_short.owns(self.ids[just_completed])
            else index
        )
        return await HaltSaveObserver.save_if_signaled(env, 0, self.ids[anchor], env.state)

    async def _observe_halt_trailing(self) -> None:
        """Save a halt commit after the LAST chain step when halt cut it short.

        :meth:`_observe_halt_between` only fires between steps. When
        halt was signaled during the final step's dispatch, no next
        step exists to save at and the walker just returns — losing
        the work halt cut short on resume. This mirror observes halt
        at the trailing edge and, when the last step was cut short,
        saves per :meth:`_cut_short_anchor` so resume re-runs it.

        No-op when halt is not set, the last step was not cut short,
        or the run is nested / has no checkpointer bound. A run whose
        halt arrived after all its work completed writes no halt commit
        here; it finishes as a clean exit.
        """
        env = self.env
        if (
            env.runtime is not self.flow
            or env.checkpoint_ctx is None
            or not is_halt_signaled(env)
            or not self.ids
        ):
            return
        last = len(self.ids) - 1
        if not env.cut_short.owns(self.ids[last]):
            return
        anchor = self._cut_short_anchor(last)
        await HaltSaveObserver.save_if_signaled(env, 0, self.ids[anchor], env.state)

    def _cut_short_anchor(self, index: int) -> int:
        """Chain index a halt commit anchors at when step ``index`` was cut short.

        The step itself when resume can hand it what it needs: at index
        0 (the run's own inputs), when it owns a paused turn (whose
        envelope carries the Loop's task), or when its input is storable
        with the commit. Otherwise the step before it, which re-runs to
        produce that input again.
        """
        step_id = self.ids[index]
        env = self.env
        if (
            index == 0
            or env.pending_paused_turns.owns(step_id)
            or env.step_inputs.storable(step_id)
        ):
            return index
        return index - 1
