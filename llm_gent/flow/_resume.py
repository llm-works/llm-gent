# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resume protocol: checkout on the read side, framework commits on the write side.

Read side — :class:`Resume` is constructed per resuming ``run()`` with the
flow being resumed, and reads the history through the public
:class:`~llm_gent.flow.history.History` API. :meth:`Resume.checkout`
(``resume="latest"``) checks out the snapshot of the newest commit with
usable state: its root scope hydrates the top-level :class:`State`; each
child scope goes back to the block that owns it, and each cursor to the
chain, iterate or branch that registered it, when the run reaches their
paths; paused turns go back to the Loops that paused.

Write side — :func:`apply_clean_exit_retention` and
:func:`commit_completion` commit the final state on clean exit and move
the ``complete`` tag to it; :func:`commit_failure` commits the state at
the moment a run raised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .checkpoint import COMPLETE_TAG
from .history import History
from .state import State, restore_state_data, serialize_state_data
from .state.cas import Commit, Tree, canonical_json
from .state.snapshot import Snapshot, path_str


if TYPE_CHECKING:
    from .flow import Flow


class Resume:
    """Check out the snapshot a resuming ``run()`` continues from.

    Constructed with a flow that has a checkpointer. :meth:`checkout` takes
    the fallback :class:`State` (the wrapped ``run(state=...)`` payload) and
    returns it unchanged when there is nothing to resume, so the caller
    falls through to a fresh run — whose commits still append to the same
    history. It raises :class:`~llm_gent.flow.history.HistoryCorrupt` on a
    corrupt history rather than silently starting over from ``fallback``.
    """

    def __init__(self, flow: Flow) -> None:
        ctx = flow._checkpoint_ctx
        assert ctx is not None
        self.flow = flow
        self._ctx = ctx
        self._history = History(ctx.store, ctx.client_flow_id)

    async def checkout(self, fallback: State[Any]) -> tuple[State[Any], Snapshot | None]:
        """Check out the snapshot of the newest commit that has usable state.

        Walks back from the head past ``$failed`` commits (the state at a
        failure may be half-updated) and commits with an empty tree (state
        that could not be serialized). Returns the snapshot's root as the
        top-level :class:`State` together with the snapshot itself, whose
        scopes and cursors the run takes as it reaches their paths. The
        commit's paused turns load as resume entries for the Loops that
        paused: the halted step reruns, so its Loop finds its entry.

        Skipping is logged: the restored state may predate the head by
        whole runs (a stateless ``$end`` sends the walk into the previous
        run). Returns ``(fallback, None)`` when the history is empty or
        holds no commit with usable state (warning in the latter case).
        """
        skipped: list[str] = []
        async for commit in self._history.commits():
            if not History.is_failed(commit):
                snapshot = await self._history.snapshot(commit)
                if snapshot.has_state:
                    if skipped:
                        self._warn_skipped(skipped, commit)
                    await self.flow._resume_paused_turns.load_from_commit(self._ctx, commit)
                    return self._root_state(snapshot.root), snapshot
            skipped.append(commit.meta.node_path)
        if skipped:
            self.flow._lg.warning(
                "no commit with usable state to resume from; starting from state=",
                extra={"client_flow_id": self._history.client_flow_id, "skipped": skipped},
            )
        return fallback, None

    def _warn_skipped(self, skipped: list[str], restored: Commit) -> None:
        """Log the ``$failed`` / stateless commits the checkout walked past, and where it landed."""
        self.flow._lg.warning(
            "resume skipped commits without usable state",
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


async def apply_clean_exit_retention(flow: Flow, final_state: State[Any]) -> None:
    """Apply the store's retention policy on the clean-exit path.

    Halt-triggered exits preserve the history regardless of policy: the
    halt commit is the head. On a clean exit: ``gc_on_success`` prunes;
    ``retain`` keeps the record and commits ``final_state`` tagged
    ``complete``, so a later ``run(resume="latest")`` continues from the
    finished run's final state.

    A resumed run that finished without reaching some of its saved scopes
    or cursors drops them first (:func:`_drop_unreached_scopes`), so the
    final commit holds the root scope alone.
    """
    if (
        flow._checkpoint_ctx is None
        or flow._halt_saved
        or (flow._halt_event is not None and flow._halt_event.is_set())
    ):
        return
    ctx = flow._checkpoint_ctx
    _drop_unreached_scopes(flow, ctx.client_flow_id)
    if ctx.retention == "gc_on_success":
        await ctx.gc_history()
    else:
        await commit_completion(flow, final_state)


def _drop_unreached_scopes(flow: Flow, client_flow_id: str) -> None:
    """Forget saved scopes and cursors a finished run never reached, with a warning naming them.

    The run executed every step it was going to, so a saved entry nothing
    took belongs to no step of this flow: its step was removed, or the run
    took another branch arm or fewer map items or iterate passes. Kept, it
    would ride along in every later snapshot and could be handed to a step
    that reuses its node id.
    """
    dropped = flow._scopes.drop_saved()
    if dropped:
        flow._lg.warning(
            "resumed run finished without reaching saved entries; dropped them",
            extra={"client_flow_id": client_flow_id, "paths": [path_str(p) for p in dropped]},
        )


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
    tree = await _put_root_tree(flow, final_state)
    commit = await ctx.save_completion_commit(tree)
    await ctx.put_tag(COMPLETE_TAG, commit.content_hash)
    return commit


async def commit_failure(flow: Flow, failed_state: State[Any]) -> None:
    """Commit the root state at ``$failed`` after the run raised.

    Called from :meth:`Flow.run`'s exception path before the original
    exception is re-raised. Any error while writing is logged at warning
    level and swallowed, so it never replaces the exception that ended the
    run. A no-op without a checkpointer.
    """
    ctx = flow._checkpoint_ctx
    if ctx is None:
        return
    try:
        tree = await _put_root_tree(flow, failed_state)
        await ctx.save_failure_commit(tree)
    except Exception as e:
        flow._lg.warning("failure commit could not be written", extra={"exception": e})


async def _put_root_tree(flow: Flow, state: State[Any]) -> Tree:
    """Put the run's snapshot, or an empty tree when its root ``state`` cannot be serialized.

    At the end of a run every child scope has closed, so the snapshot
    holds the root scope alone.
    """
    ctx = flow._checkpoint_ctx
    assert ctx is not None
    if _serializable(flow, state):
        return await ctx.put_snapshot(flow._scopes)
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
