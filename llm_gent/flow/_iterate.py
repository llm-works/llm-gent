# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Iterate-node dispatcher: bounds + halt observation + resume replay.

Extracted from :mod:`._executor`. An :class:`IterateRunner` drives
one ``.iterate`` node: projects the body's child :class:`State`,
resolves the resume replay (fast-forward the iteration counter and
restore the checkpointed child scope when this iterate is the
save-point leaf), then loops the body under
``max_iters`` / ``deadline`` / halt bounds, threading each pass's
result to the next and merging the child state back into the parent
when the loop exits.

The primitives it composes with — replay pop / scope-consumption
helpers, ``_project_state`` / ``_merge_state`` / ``_check_until``
callbacks, ``_save_scope_commit`` at policy-driven boundaries, and
:meth:`HaltSaveObserver.save_if_signaled` for the between-iterations
halt check — all live in :mod:`._executor` and :mod:`._halt_observer`.
The runner is single-use — a fresh instance per iterate node
execution — so mutating ``self.env`` when the replay-resolution step
returns a rebound env is safe.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any

from ._executor import (
    _check_until,
    _consume_scope_data,
    _enter_scope,
    _live_scope,
    _merge_state,
    _pop_replay_for,
    _restore_scope_state,
    _running,
    _save_scope_commit,
)
from ._halt_observer import HaltSaveObserver
from ._node_id import _descend_context
from .nodes import UNSET
from .state import State
from .state.snapshot import CARRY, PASS, ScopePath


if TYPE_CHECKING:
    from .nodes import _Iterate, _RunEnv


class IterateRunner:
    """Drive one ``.iterate`` node's body under bounds + halt + resume.

    Constructed with the node config (``it``), the current run env,
    and the runtime ``node_id`` for this iterate. Caller invokes
    :meth:`run` with the incoming ``node_args`` and receives the last
    body-pass result. Post-check loop semantics — the body always
    runs at least once, then ``until`` (if set) is evaluated.
    """

    def __init__(self, it: _Iterate, env: _RunEnv, node_id: str) -> None:
        self.it = it
        # env may be rebound once by _resolve_child_scope when the replay
        # advances during scope-data consumption. All subsequent methods
        # read from self.env so the update propagates.
        self.env = env
        self.node_id = node_id
        # The cursor: the pass the loop is in, and the value it carries into
        # that pass. Kept here, not in _loop's locals, so a checkpoint reads them.
        self.iteration = 0
        self.carry: Any = None

    def cursor(self) -> dict[str, Any]:
        """The pass this iterate is in (0-based) and the value carried into it."""
        return {PASS: self.iteration, CARRY: self.carry}

    async def run(self, node_args: tuple[Any, ...]) -> Any:
        """Drive the loop; return the last body result.

        Save-at-iterate-boundary: under the ``on_iterate`` policy, the
        framework commits a snapshot of the run after each successful
        pass. The snapshot holds every live scope, including the child
        scope ``state=`` projects, and the pass counter.

        ``resume="latest"``: the iterate continues in the saved pass with
        the saved carried value (:meth:`_take_saved_position`); completed
        passes do not run again, and ``max_iters`` is a bound across runs.

        Restart: the child scope is restored from the snapshot, but the
        counter starts at 0 and every pass runs again; state-driven bodies
        make the repeated passes skip their completed work.

        Replay: when the resume replay's ``remaining_path`` has been
        head-popped down to a single entry equal to this iterate's
        runtime ``node_id`` (this iterate IS the save-point leaf), the
        counter starts at the saved iteration instead of 0 — so
        ``max_iters`` becomes a cumulative bound across resumes, not
        per-run. At most one iterate per run consumes the replay;
        :attr:`Flow._replay_consumed` flips on the first match so
        re-entrant dispatches of the same node (an inner iterate spun
        up by an outer loop) do not re-apply the fast-forward.
        ``deadline`` is not restored — the wall clock resets each run.
        """
        self.iteration, restored_child, restored_carry = self._resume_iteration()
        if restored_carry is not UNSET:
            self.carry = restored_carry
        else:
            self.carry = node_args[0] if node_args else None
        path = self.env.owner_path(self.node_id)
        self._take_saved_position(path)
        env, child_state = await self._resolve_child_scope(restored_child)
        self.env = env
        with _live_scope(env, path, self.it.state_fn, child_state), _running(env, path, self):
            result = await self._loop(path, child_state)
            await _merge_state(self.it.merge_fn, self.env.state, child_state)
        return result

    def _take_saved_position(self, path: ScopePath) -> None:
        """Continue at the pass and carried value a checkout saved at ``path``, when there is one."""
        found, saved_pass = self.env.scopes.take_cursor(path, PASS)
        if found:
            self.iteration = saved_pass
        found, saved_carry = self.env.scopes.take_cursor(path, CARRY)
        if found:
            self.carry = saved_carry

    async def _loop(self, path: ScopePath, child_state: State[Any]) -> Any:
        """Run passes until a bound, the halt or ``until`` stops them; return the last result.

        ``self.iteration`` and ``self.carry`` advance together after each
        pass, with no await in between, so a checkpoint always sees a pass
        number and the value carried into that pass.
        """
        started = time.monotonic()
        while True:
            if self.it.max_iters is not None and self.iteration >= self.it.max_iters:
                break
            if self.it.deadline is not None and time.monotonic() - started >= self.it.deadline:
                break
            if await HaltSaveObserver.save_if_signaled(
                self.env, self.iteration, self.node_id, child_state
            ):
                break
            pass_path = (*path, "p", str(self.iteration))
            result = await self._dispatch_body(child_state, self.carry, pass_path)
            self.carry, self.iteration = result, self.iteration + 1
            if self.env.policy.on_iterate:
                await _save_scope_commit(self.env, self.iteration, self.node_id, child_state, "ok")
            if await _check_until(self.it.until, result, child_state, self.env, self.node_id):
                break
        return self.carry

    def _resume_iteration(self) -> tuple[int, Any, Any]:
        """Return the starting iteration, restored child state, and carried value.

        ``(0, None, UNSET)`` for a fresh run. On resume, when ``env.replay``
        is set and ``remaining_path == (node_id,)`` — the head-pop
        path has shrunk to a single entry equal to this iterate's
        ``node_id`` — this iterate IS the save-point leaf: returns
        the saved iteration, ``child_state_data`` (if present), and the
        ``carry`` value from the cursor, then flips
        :attr:`Flow._replay_consumed` on the top-level runtime
        so the fast-forward fires exactly once. Any longer remaining
        path means this iterate is an ancestor of the leaf (its body
        descent will head-pop and thread the tail); a non-matching
        head, empty path, or already-consumed replay all yield
        ``(0, None, UNSET)`` and the iterate runs from scratch.
        """
        replay = self.env.replay
        if replay is None or not replay.remaining_path:
            return 0, None, UNSET
        if self.env.runtime._replay_consumed:
            return 0, None, UNSET
        if replay.remaining_path[0] != self.node_id or len(replay.remaining_path) != 1:
            return 0, None, UNSET
        self.env.runtime._replay_consumed = True
        return replay.iteration, replay.child_state_data, replay.carry

    async def _resolve_child_scope(self, restored_child: Any) -> tuple[_RunEnv, State[Any]]:
        """Build the iterate body's child :class:`State` for this run.

        Three paths, tried in order:

        1. ``restored_child`` is not ``None`` — this iterate is the
           leaf, use the checkpointed leaf-scope data (via
           ``_ResumeReplay.child_state_data``).
        2. This iterate is a middle scope on the replay path with a
           scoping ``state_fn`` — consume the next intermediate scope
           payload from the replay and return an env with the popped
           replay so the body descent sees the aligned tail.
        3. Fresh projection via ``state_fn`` (or passthrough when
           ``state_fn`` is ``None``).
        """
        it = self.it
        env = self.env
        effective_factory = it.state_factory if it.state_factory is not None else env.state._factory
        if restored_child is not None:
            restored_data = (
                effective_factory.restore(restored_child)
                if effective_factory is not None
                else restored_child
            )
            return env, State(data=restored_data, _parent=env.state, _factory=effective_factory)
        on_path = (
            env.replay is not None
            and env.replay.remaining_path
            and env.replay.remaining_path[0] == self.node_id
            and not env.runtime._replay_consumed
        )
        if on_path:
            raw, updated_replay = _consume_scope_data(env.replay, it.state_fn)
            if raw is not UNSET:
                env = dataclasses.replace(env, replay=updated_replay)
                return env, _restore_scope_state(env.state, raw, effective_factory)
        path = env.owner_path(self.node_id)
        child_state, _ = await _enter_scope(env, path, it.state_fn, it.state_factory, None)
        return env, child_state

    async def _dispatch_body(
        self, child_state: State[Any], prev_result: Any, pass_path: ScopePath
    ) -> Any:
        """Run one pass of the body under the current env.

        Descends into the body with
        ``chain_context = _descend_context(node_id, "body")`` and
        ``ancestor_chain`` extended by ``node_id`` — the same context
        every pass, so the body's chain steps have iteration-
        invariant IDs. The pass number enters the snapshot path instead
        (``pass_path``), so scopes opened in different passes have
        different paths.
        """
        env = self.env
        return await self.it.body._run_as_subflow(
            prev_result,
            state=child_state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_budget=env.budget,
            parent_checkpoint_ctx=env.checkpoint_ctx,
            parent_chain_context=_descend_context(self.node_id, "body"),
            parent_ancestor_chain=env.ancestor_chain + (self.node_id,),
            parent_replay=_pop_replay_for(env, self.node_id),
            parent_extra=env.extra,
            parent_policy=env.policy,
            parent_path=pass_path,
        )
