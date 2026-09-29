# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Iterate-node dispatcher: bounds, halt observation and the iterate's cursor.

Extracted from :mod:`._executor`. An :class:`IterateRunner` drives one
``.iterate`` node: enters the body's child :class:`State` (restored from
a checkout, or projected), then loops the body under ``max_iters`` /
``deadline`` / halt bounds, threading each pass's result to the next and
merging the child state back into the parent when the loop exits.

Its cursor — the pass it is in and the value carried into that pass —
is state on the runner, in every snapshot taken while it runs. On
``resume="latest"`` the iterate continues at the saved cursor. The
runner is single-use: a fresh instance per iterate node execution.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ._executor import (
    _check_until,
    _enter_scope,
    _live_scope,
    _merge_state,
    _running,
    _save_scope_commit,
)
from ._halt_observer import HaltSaveObserver, is_halt_signaled, saves_run_halt
from ._node_id import _descend_context
from .nodes import Interrupted
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
        pass, holding every live scope and cursor.

        ``resume="latest"``: the iterate continues in the saved pass with
        the saved carried value (:meth:`_take_saved_position`); completed
        passes do not run again, and ``max_iters`` is a bound across runs.
        ``deadline`` is not restored — the wall clock resets each run.
        """
        self.carry = node_args[0] if node_args else None
        path = self.env.owner_path(self.node_id)
        self._take_saved_position(path)
        env = self.env
        child_state = await _enter_scope(env, path, self.it.state_fn, self.it.state_factory)
        with _live_scope(env, path, self.it.state_fn, child_state), _running(env, path, self):
            result = await self._loop(path, child_state)
            await _merge_state(self.it.merge_fn, env.state, child_state)
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
        number and the value carried into that pass. A halt that stops the
        loop before its bounds do interrupts it: a body interrupted inside
        a pass raises :class:`Interrupted` through the loop, which neither
        advances nor writes a policy commit after it.

        Raises:
            Interrupted: The halt stopped the loop before its bounds or
                ``until`` did.
        """
        started = time.monotonic()
        while True:
            if self.it.max_iters is not None and self.iteration >= self.it.max_iters:
                break
            if self.it.deadline is not None and time.monotonic() - started >= self.it.deadline:
                break
            if await self._halted_before_pass(child_state):
                raise Interrupted()
            pass_path = (*path, "p", str(self.iteration))
            result = await self._dispatch_body(child_state, self.carry, pass_path)
            if self.env.runtime._halt_saved:
                raise Interrupted()  # the run halted elsewhere (a sibling map item)
            self.carry, self.iteration = result, self.iteration + 1
            if self.env.policy.on_iterate:
                await _save_scope_commit(self.env, self.iteration, self.node_id, child_state, "ok")
            if await _check_until(self.it.until, result, child_state, self.env, self.node_id):
                break
        return self.carry

    async def _halted_before_pass(self, child_state: State[Any]) -> bool:
        """True when a halt stops the loop before the next pass; the run's halt is saved.

        Reached when the halt was set outside any step — before the loop
        started, or during the last step of the previous pass, which
        completed — so the cursor is at the pass that has not run yet. A
        subflow's own halt stops the loop without a save.
        """
        if not is_halt_signaled(self.env):
            return False
        if saves_run_halt(self.env):
            await HaltSaveObserver.save_if_signaled(
                self.env, self.iteration, self.node_id, child_state
            )
        return True

    async def _dispatch_body(
        self, child_state: State[Any], prev_result: Any, pass_path: ScopePath
    ) -> Any:
        """Run one pass of the body under the current env.

        Descends into the body with
        ``chain_context = _descend_context(node_id, "body")`` and
        ``ancestor_chain`` extended by ``node_id`` — the same context
        every pass, so the body's chain steps have iteration-
        invariant IDs. The pass number enters the snapshot path instead
        (``pass_path``), so scopes and cursors in different passes have
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
            parent_extra=env.extra,
            parent_policy=env.policy,
            parent_path=pass_path,
        )
