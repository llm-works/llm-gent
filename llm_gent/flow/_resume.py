# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resume protocol: read-side hydration + write-side framework commits.

Read side — :class:`Resume` is constructed per resuming ``run()`` with
the flow being resumed, and reads the history through the public
:class:`~llm_gent.flow.history.History` API:

- :meth:`Resume.replay` (``resume="replay"``) returns the
  ``(State, _ResumeReplay | None)`` pair the executor threads through the
  walk: the last save point's root scope hydrates the top-level
  :class:`State`, every non-root scope payload rides on
  ``_ResumeReplay.intermediate_scope_data`` so descent sites
  (``_consume_scope_data``) restore their own scope in order.
- :meth:`Resume.restart` (``resume="restart"``) returns only the root
  :class:`State` of the newest commit with usable state; the run starts
  at the first node.

Write side — :func:`apply_clean_exit_retention` and
:func:`commit_completion` commit the final state on clean exit and move
the ``complete`` tag to it; :func:`commit_failure` commits the state at
the moment a run raised. :func:`assert_replay_consumed` is the
belt-and-suspenders check called after a replay run to fail-fast when
the save-point iterate was never found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._recorder import RunRecorder
from .checkpoint import COMPLETE_TAG
from .history import History
from .state import State, restore_state_data, serialize_state_data
from .state.cas import Commit, Tree, canonical_json
from .state.record import RecordError


if TYPE_CHECKING:
    from .flow import Flow
    from .nodes import _ResumeReplay


class Resume:
    """Hydrate the initial state (and replay plan) for a resuming ``run()``.

    Constructed with a flow that has a checkpointer. Both entry points take
    the fallback :class:`State` (the wrapped ``run(state=...)`` payload)
    and return it unchanged when there is nothing to resume, so the caller
    falls through to a fresh run — whose commits still append to the same
    history. Both raise :class:`~llm_gent.flow.history.HistoryCorrupt` on a
    corrupt history rather than silently starting over from ``fallback``.
    """

    def __init__(self, flow: Flow) -> None:
        ctx = flow._checkpoint_ctx
        assert ctx is not None
        self.flow = flow
        self._ctx = ctx
        self._history = History(ctx.store, ctx.client_flow_id)

    async def replay(self, fallback: State[Any]) -> tuple[State[Any], _ResumeReplay | None]:
        """Replay: rebuild the last save point's scope tree and replay context.

        1. :meth:`History.replay_point` picks the resume commit: the head, or the
           newest commit before a run of ``$failed`` commits. A final-state
           commit there is a finished run: fresh run (even if the
           ``complete`` tag write after it never landed).
        2. :meth:`History.scopes` walks the commit's tree to one JSON
           payload per scope, root → leaf via the zero-padded ``scope_id``.
           A commit written by a flow of a different structure is refused
           (:meth:`_assert_same_structure`).
        3. The commit's ``paused_turn`` trace refs load as resume entries
           for the Loops that paused.
        4. The root scope's payload rehydrates the top-level
           :class:`State`; every non-root scope rides on the
           :class:`_ResumeReplay` for its descent site to restore.
        5. ``node_path`` splits on ``"/"`` back into the ancestor chain the
           executor's head-pop replay expects. blake2b hex has no slashes,
           so the round-trip is exact.

        Returns ``(fallback, None)`` when there is nothing to replay.
        """
        commit = await self._history.replay_point()
        if commit is None:
            return fallback, None
        scope_data = await self._history.scopes(commit)
        self._assert_same_structure(commit)
        await self.flow._resume_paused_turns.load_from_commit(self._ctx, commit)
        return self._split_scopes(commit, scope_data)

    def _assert_same_structure(self, commit: Commit) -> None:
        """Refuse to replay a commit written by a flow with a different structure.

        Replay starts at the saved step's index in the current chain and
        skips every step before it. Node ids do not encode chain position,
        so an edited chain can still contain the saved step while placing
        a new or already-run step before or after it; replaying would skip
        the new step or run the other one twice. ``flow_root_hash`` covers
        step order, so equal hashes guarantee the chain that wrote the
        commit. ``resume="restart"`` continues from the saved state instead.
        """
        saved = commit.meta.flow_root_hash
        current = self.flow.root_hash()
        if saved == current:
            return
        saved_path = commit.meta.node_path.replace("/", " → ")
        raise RuntimeError(
            f"cannot replay: the composition graph has structurally changed since the "
            f"checkpoint was written (flow_root_hash {saved!r} != {current!r}). "
            f'Saved path (root→leaf): {saved_path}. Use resume="restart" to continue '
            f"from the saved state."
        )

    async def restart(self, fallback: State[Any]) -> State[Any]:
        """Restart: the root state of the newest commit that has usable state.

        Walks back from the head past ``$failed`` commits (the state at a
        failure may be half-updated) and commits with an empty tree (state
        that could not be serialized). The run then starts at the first
        node: child scopes are not restored and no replay context is built,
        so iterate counters start at zero. Paused turns are not offered —
        a step's node id does not identify a map item across runs.

        Skipping is logged: the restored state may predate the head by
        whole runs (a stateless ``$end`` sends the walk into the previous
        run). Returns ``fallback`` when the history is empty or holds no
        commit with usable state (warning in the latter case).
        """
        skipped: list[str] = []
        async for commit in self._history.commits():
            scope_data = [] if History.is_failed(commit) else await self._history.scopes(commit)
            if scope_data:
                if skipped:
                    self._warn_restart_skipped(skipped, commit)
                return self._root_state(scope_data[0])
            skipped.append(commit.meta.node_path)
        if skipped:
            self.flow._lg.warning(
                "no commit with usable state to restart from; starting from state=",
                extra={"client_flow_id": self._history.client_flow_id, "skipped": skipped},
            )
        return fallback

    def _warn_restart_skipped(self, skipped: list[str], restored: Commit) -> None:
        """Log the ``$failed`` / stateless commits restart walked past, and where it landed."""
        self.flow._lg.warning(
            "restart skipped commits without usable state",
            extra={
                "client_flow_id": self._history.client_flow_id,
                "skipped": skipped,
                "restored_commit": restored.content_hash,
                "restored_node_path": restored.meta.node_path,
            },
        )

    def _root_state(self, root_raw: Any) -> State[Any]:
        """Top-level :class:`State` from a stored root payload (factory-restored when bound)."""
        factory = self.flow._state_factory
        data = (
            root_raw
            if factory is None or root_raw is None
            else restore_state_data(factory, root_raw)
        )
        return State(data=data, _factory=factory)

    def _split_scopes(
        self, commit: Commit, scope_data: list[Any]
    ) -> tuple[State[Any], _ResumeReplay | None]:
        """Split root / non-root scope payloads; return (State, replay).

        Root scope hydrates the top-level :class:`State`. Every
        non-root scope rides on ``intermediate_scope_data`` in
        order; each scope-creating descent (``.call(state=)``,
        ``.iterate(state=)``, ``.map(state=)``) consumes the next
        entry at its own descent site via
        ``_consume_scope_data``. A leaf iterate without a
        ``state_fn`` creates no scope of its own and simply reuses
        the parent scope with the fast-forwarded iteration count.
        """
        from .nodes import _ResumeReplay

        root_raw = scope_data[0] if scope_data else None
        intermediate_raw = tuple(scope_data[1:])
        path_tuple = tuple(commit.meta.node_path.split("/")) if commit.meta.node_path else ()
        return (
            self._root_state(root_raw),
            _ResumeReplay(
                remaining_path=path_tuple,
                full_path=path_tuple,
                iteration=commit.meta.iteration,
                child_state_data=None,
                intermediate_scope_data=intermediate_raw,
            ),
        )


async def apply_clean_exit_retention(flow: Flow, final_state: State[Any]) -> None:
    """Apply the store's retention policy on the clean-exit path.

    Halt-triggered exits preserve the history regardless of policy:
    the halt commit (the chain writes one at the last step when halt
    was set during it) is the head and carries the run's final state.
    On a clean exit: ``gc_on_success`` prunes; ``retain`` keeps the
    record and commits ``final_state`` tagged ``complete`` so a
    subsequent ``run(resume="replay")`` doesn't replay the last save point
    and re-execute chain steps after it.
    """
    if (
        flow._checkpoint_ctx is None
        or flow._halt_saved
        or (flow._halt_event is not None and flow._halt_event.is_set())
    ):
        return
    ctx = flow._checkpoint_ctx
    if ctx.retention == "gc_on_success":
        await ctx.gc_history()
    else:
        await commit_completion(flow, final_state)


async def commit_completion(flow: Flow, final_state: State[Any]) -> Commit:
    """Commit the run's final state at ``$end`` and move the ``complete`` tag to it.

    The commit holds the top-level state, so the history's head always
    carries the state the last run ended with — even when no save point
    fired during the run. It is attributed to the framework
    (:data:`~llm_gent.flow.checkpoint.COMPLETION_PRODUCER`), not to a node.

    A final state that cannot be serialized (live handles, non-JSON
    values) must not fail a run whose work is done: the commit is then
    written with an empty tree, which still marks the history complete
    but carries no state.
    """
    ctx = flow._checkpoint_ctx
    assert ctx is not None
    tree = await _put_root_tree(flow, final_state, None)
    commit = await ctx.save_completion_commit(tree)
    await ctx.put_tag(COMPLETE_TAG, commit.content_hash)
    return commit


async def commit_failure(flow: Flow, failed_state: State[Any]) -> None:
    """Commit the root state at ``$failed`` after the run raised.

    Called from :meth:`Flow.run`'s exception path before the original
    exception is re-raised. Any error while writing is logged at warning
    level and swallowed, so it never replaces the exception that ended the
    run. A no-op without a checkpointer.

    The commit carries the run's execution record and open scopes, so the
    work that completed before the failure is on record. When an open
    scope cannot be serialized (a verb put a live handle into it), the
    commit is written with the root state alone: a record without the
    scopes its entries' effects live in would be inconsistent, while a
    commit without a record is merely less complete.
    """
    ctx = flow._checkpoint_ctx
    if ctx is None:
        return
    try:
        tree = await _failure_tree(flow, failed_state)
        await ctx.save_failure_commit(tree)
    except Exception as e:
        flow._lg.warning("failure commit could not be written", extra={"exception": e})


async def _failure_tree(flow: Flow, state: State[Any]) -> Tree:
    """The failure commit's tree: with the run's record and scopes, else the root alone."""
    try:
        return await _put_root_tree(flow, state, flow._recorder)
    except RecordError as e:
        flow._lg.warning(
            "failure commit carries no execution record: an open scope cannot be serialized",
            extra={"exception": e},
        )
        return await _put_root_tree(flow, state, None)


async def _put_root_tree(flow: Flow, state: State[Any], run: RunRecorder | None) -> Tree:
    """Put ``state``'s commit tree (with ``run``'s record and scopes), or an empty tree.

    The tree is empty when the root state cannot be serialized.
    """
    ctx = flow._checkpoint_ctx
    assert ctx is not None
    if _serializable(flow, state):
        return await ctx.put_state_tree(state, run)
    tree = Tree.from_entries([])
    await ctx.put_tree(tree)
    return tree


def _serializable(flow: Flow, state: State[Any]) -> bool:
    """True when ``state`` can be checkpointed; warn and return False otherwise."""
    try:
        canonical_json(serialize_state_data(state.data))
    except (TypeError, ValueError) as e:
        flow._lg.warning(
            "run state is not serializable; framework commit carries no state",
            extra={"exception": e},
        )
        return False
    return True


def assert_replay_consumed(flow: Flow, replay: _ResumeReplay | None) -> None:
    """Raise if a resume request never found its save-point iterate.

    Belt-and-suspenders check called after :meth:`Flow._run_as_subflow`
    returns. :meth:`Chain._assert_replay_reachable` will typically have
    raised earlier — as soon as a Flow entry sees a path head that no
    chain step at that level provides. This post-run raise catches the
    residual cases where the entry-level pre-scan matched something but
    no iterate ever consumed the tail (structurally impossible under
    normal composition, but the check costs nothing and keeps the
    invariant explicit).

    The raise includes the full saved path (root → leaf) so ops triage
    can correlate the ancestor chain with the current composition tree
    and locate the layer where the graph diverged.
    """
    if replay is None or flow._replay_consumed:
        return
    target_id = replay.remaining_path[-1] if replay.remaining_path else "<empty>"
    label = flow._name or "<anonymous>"
    path_repr = " → ".join(replay.full_path) if replay.full_path else "<empty>"
    raise RuntimeError(
        f"Flow {label!r}: resume checkpoint's save-point iterate "
        f"id {target_id!r} was not found in the composition graph "
        f"during the run — the graph has structurally changed "
        f"since the checkpoint was written. "
        f"Saved path (root→leaf): {path_repr}"
    )
