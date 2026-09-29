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

from ._executor import _build_ctx, _execute_node, _running, _step_inputs
from ._halt_observer import HaltSaveObserver, is_halt_signaled
from ._node_id import _compute_node_ids
from .nodes import UNSET, _Iterate
from .state.snapshot import CHAIN, path_str


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
        # The cursor: the step running (or about to run) and the input it
        # gets. Kept here, not in _walk_steps' locals, so a checkpoint reads it.
        self.index: int | None = None
        self.step_args: tuple[Any, ...] = ()
        self.step_kwargs: dict[str, Any] = {}

    def cursor(self) -> dict[str, Any]:
        """The step this chain is at (its node id) and that step's input.

        ``resume="latest"`` continues the chain at this step with this
        input (:meth:`_saved_step`). ``resume="replay"`` does not read it:
        it starts the step with no input (:meth:`_resume_start_index`).
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

        First asserts the resume replay (if any) can reach a step at
        this level, then selects the start index (0 on a fresh run,
        the on-path index on resume), then walks. Halt observation
        between steps and after the trailing edge is internal. The
        chain's cursor is in every snapshot taken while it walks.
        """
        self._assert_replay_reachable()
        start_index = self._resume_start_index()
        first_input = None
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

        Contract on the on-path node when ``start_index > 0``: it
        runs with no ``prev_result`` — its predecessor didn't re-
        run, so there is no return value to thread in. Iterate
        bodies that depend on the outer chain's return value on
        resume-first-iteration must either be at chain index 0 or
        read from state.
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
        first_input: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
    ) -> Any:
        """Execute chain steps from ``start_index`` onward, threading returns.

        On resume, predecessors already completed before the checkpoint
        was written. With ``first_input`` (``resume="latest"``) the step
        at ``start_index`` gets the input it had; under replay
        (``start_index > 0``) it runs with no ``prev_result`` (see
        :meth:`_resume_start_index` for that contract).

        Halt observation: between chain steps (never at the very
        first iteration of this walk, so resume runs at least the
        halted step), if ``env.halt`` is set, stamp a halted commit
        at the not-yet-run step's position and break. On a subsequent
        ``run(resume="replay")``, that ref resolves to this commit and
        the walk restarts at the halted step.

        The cursor moves to a step, with the step's input (after
        ``project``), before the halt check in front of it, so a halt
        commit there records the step that runs next.
        """
        result: Any = UNSET
        for index in range(start_index, len(self.flow._nodes)):
            node = self.flow._nodes[index]
            node_id = self.ids[index]
            node_args: tuple[Any, ...]
            node_kwargs: dict[str, Any]
            if index == start_index and first_input is not None:
                node_args, node_kwargs = first_input
            elif index == start_index and start_index > 0:
                node_args, node_kwargs = (), {}
            else:
                node_args, node_kwargs = _step_inputs(index, node, result, args, kwargs)
            previous = (self.index, self.step_args, self.step_kwargs)
            self.index, self.step_args, self.step_kwargs = index, node_args, node_kwargs
            if await self._observe_halt_between(index, start_index, previous):
                break
            ctx = _build_ctx(node.target, self.env, node_id)
            result = await _execute_node(node, ctx, self.env, node_args, node_kwargs, node_id)
        else:
            # for-else: chain exhausted without a between-steps halt-save. A halt
            # set during the LAST step saves at that step — no next step exists.
            await self._observe_halt_trailing()
        return result

    async def _observe_halt_between(
        self,
        index: int,
        start_index: int,
        previous: tuple[int | None, tuple[Any, ...], dict[str, Any]],
    ) -> bool:
        """Save a halt commit between chain steps when appropriate; return True if saved.

        Only fires at the top-level chain (``env.runtime is self.flow``)
        with a checkpointer bound and past the first step of this walk
        (so resume runs at least the halted step). Nested body chains
        let halt propagate to iterate boundaries where iteration state
        is consistent.

        When the just-completed step paused SAIA mid-turn (a Loop
        deposited bytes on ``env.pending_paused_turns``), lands
        the halt commit at THAT step's node so resume re-dispatches it
        — its Loop's ``__call__`` then picks up the paused_turn entry
        and hands SAIA ``resume=True`` with the rebuilt conversation.
        Otherwise saves at the not-yet-run step (the normal chain-halt
        case). The cursor follows: back to the just-completed step and
        its ``previous`` input in the first case, at the next step in
        the second.
        """
        env = self.env
        if (
            index <= start_index
            or env.runtime is not self.flow
            or env.checkpoint_ctx is None
            or not is_halt_signaled(env)
        ):
            return False
        just_completed = self.ids[index - 1]
        halt_node_id = self.ids[index]
        if _just_completed_owns_paused_turn(env, just_completed):
            halt_node_id = just_completed
            self.index, self.step_args, self.step_kwargs = previous
        return await HaltSaveObserver.save_if_signaled(env, 0, halt_node_id, env.state)

    async def _observe_halt_trailing(self) -> None:
        """Save a halt commit at the LAST chain step when halt was signaled during it.

        :meth:`_observe_halt_between` only fires between steps. When
        halt was signaled during the final step's dispatch, no next
        step exists to save at. Without a commit here the head would
        be an older save point, and a later run would start from
        stale state.

        The commit carries the run's final state, and the history is
        not marked complete: the framework cannot tell whether the
        last step finished its work or returned early because of the
        halt. ``resume="restart"`` continues from that state;
        ``resume="replay"`` re-runs the last step (and a Loop in it
        picks up its paused_turn entry).

        No-op when halt is not set, a halt commit was already written
        (the save kernel latches), or the run is nested / has no
        checkpointer bound.
        """
        env = self.env
        if (
            env.runtime is not self.flow
            or env.checkpoint_ctx is None
            or not is_halt_signaled(env)
            or not self.ids
        ):
            return
        await HaltSaveObserver.save_if_signaled(env, 0, self.ids[-1], env.state)


def _just_completed_owns_paused_turn(env: _RunEnv, node_id: str) -> bool:
    """True when ``node_id`` is a pending Loop's own id or an ancestor of one.

    Direct match covers the ``.call(loop_verb)`` case (Loop's
    ``ctx._node_id`` IS the chain step's id). Ancestry match covers
    nested Loops — Loop paused inside an iterate body inside the
    chain step, where the pending entry's key is the Loop's
    descendant id computed under the chain step's descent context.
    Either match means resume should re-dispatch the chain step so
    the Loop's ``__call__`` picks up the paused_turn entry.
    """
    return env.pending_paused_turns.owns(node_id)
