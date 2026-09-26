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
    _merge_state,
    _pop_replay_for,
    _project_state,
    _restore_scope_state,
    _save_scope_commit,
)
from ._halt_observer import HaltSaveObserver
from ._node_id import _descend_context
from .nodes import UNSET
from .state import State


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

    async def run(self, node_args: tuple[Any, ...]) -> Any:
        """Drive the loop; return the last body result.

        Save-at-iterate-boundary: when the runtime carries a
        checkpointer + ``client_flow_id``, the framework builds a
        content-addressed commit after each successful iteration with
        the parent-scope payload (``env.state``, which persists across
        iterations). When ``state=`` projects a child scope, only the
        parent state is checkpointed — progress in the child state is
        lost on resume. To preserve iteration progress, accumulate
        results in the parent state or use ``until=`` with state-driven
        termination.

        Resume: when the resume replay's ``remaining_path`` has been
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
        iteration, restored_child = self._resume_iteration()
        env, child_state = await self._resolve_child_scope(restored_child)
        self.env = env
        result: Any = node_args[0] if node_args else None
        started = time.monotonic()
        while True:
            if self.it.max_iters is not None and iteration >= self.it.max_iters:
                break
            if self.it.deadline is not None and time.monotonic() - started >= self.it.deadline:
                break
            if await HaltSaveObserver.save_if_signaled(
                self.env, iteration, self.node_id, child_state
            ):
                break
            result = await self._dispatch_body(child_state, result)
            iteration += 1
            if self.env.policy.on_iterate:
                await _save_scope_commit(self.env, iteration, self.node_id, child_state, "ok")
            if await _check_until(self.it.until, result, child_state, self.env, self.node_id):
                break
        await _merge_state(self.it.merge_fn, self.env.state, child_state)
        return result

    def _resume_iteration(self) -> tuple[int, Any]:
        """Return the starting iteration count and restored child state.

        ``(0, None)`` for a fresh run. On resume, when ``env.replay``
        is set and ``remaining_path == (node_id,)`` — the head-pop
        path has shrunk to a single entry equal to this iterate's
        ``node_id`` — this iterate IS the save-point leaf: returns
        the saved iteration and ``child_state_data`` (if present) and
        flips :attr:`Flow._replay_consumed` on the top-level runtime
        so the fast-forward fires exactly once. Any longer remaining
        path means this iterate is an ancestor of the leaf (its body
        descent will head-pop and thread the tail); a non-matching
        head, empty path, or already-consumed replay all yield
        ``(0, None)`` and the iterate runs from scratch.
        """
        replay = self.env.replay
        if replay is None or not replay.remaining_path:
            return 0, None
        if self.env.runtime._replay_consumed:
            return 0, None
        if replay.remaining_path[0] != self.node_id or len(replay.remaining_path) != 1:
            return 0, None
        self.env.runtime._replay_consumed = True
        return replay.iteration, replay.child_state_data

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
        return env, await _project_state(it.state_fn, env.state, it.state_factory)

    async def _dispatch_body(self, child_state: State[Any], prev_result: Any) -> Any:
        """Run one pass of the body under the current env.

        Descends into the body with
        ``chain_context = _descend_context(node_id, "body")`` and
        ``ancestor_chain`` extended by ``node_id`` — the same context
        every pass, so the body's chain steps have iteration-
        invariant IDs (the runtime pass counter is stored alongside
        the path, not baked into node identity).
        """
        env = self.env
        return await self.it.body._run_as_subflow(
            prev_result,
            state=child_state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_budget=env.budget,
            parent_checkpointer=env.checkpointer,
            parent_client_flow_id=env.client_flow_id,
            parent_chain_context=_descend_context(self.node_id, "body"),
            parent_ancestor_chain=env.ancestor_chain + (self.node_id,),
            parent_replay=_pop_replay_for(env, self.node_id),
            parent_extra=env.extra,
            parent_policy=env.policy,
        )
