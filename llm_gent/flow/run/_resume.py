# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resume protocol: checkout on the read side, framework commits on the write side.

Read side — :class:`Checkout` is constructed per run that starts from its
history, and reads the history through the public
:class:`~llm_gent.flow.history.History` API. :meth:`Checkout.checkout`
(``Resume("latest")``) checks out the snapshot of the newest commit with
usable state, :meth:`Checkout.checkout_at` (``Resume(<hash>)`` or
``Resume(<name>)``) the snapshot of one commit or named checkpoint: its
root scope hydrates the top-level
:class:`State`; each
child scope goes back to the block that owns it, and each cursor to the
chain, iterate, branch or Loop call that registered it, when the run
reaches their paths. :meth:`Checkout.restart` (``Restart(...)``)
checks out a commit the same way for a run that starts again from its
first step, taking the commit's root state and the top-level flow's
resources alone.

Write side — :func:`apply_clean_exit_retention` and
:func:`commit_completion` commit the final state on clean exit and move
the ``complete`` tag to it; :func:`commit_halt` commits a halted run's
position once everything has stopped. A run that raises writes nothing:
its history's head is its last save, where resume continues.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.cas import Tree, canonical_json

from ..checkpoint import COMPLETE_TAG, is_commit_hash
from ..history import History
from ..state import State, restore_state_data, serialize_state_data
from ..state.cas import FlowCommit
from ..state.snapshot import RESOURCES, Snapshot, path_str
from ._hook_points import checkpoint_written


if TYPE_CHECKING:
    from ._run import Run


class Checkout:
    """Check out the snapshot a run that starts from its history continues from.

    Constructed with a run that has a history writer. :meth:`checkout` takes
    a builder of the fallback :class:`State` (the wrapped ``state=``
    payload, else a fresh state) and calls it only when there is nothing to
    resume, so the caller falls through to a fresh run — whose commits still
    append to the same history. A history with a commit to check out never
    builds the fallback. It raises
    :class:`~llm_gent.flow.history.HistoryCorrupt` on a corrupt history
    rather than silently starting over from ``fallback``.
    """

    def __init__(self, run: Run) -> None:
        ctx = run.checkpoint_ctx
        assert ctx is not None
        self.flow = run.flow
        self._ctx = ctx
        self._history = History(ctx.store, ctx.client_flow_id)
        self.checked_out: str | None = None
        """The commit whose snapshot the run starts from, once one is checked out."""

    async def checkout(
        self, fallback: Callable[[], State[Any]]
    ) -> tuple[State[Any], Snapshot | None]:
        """Check out the snapshot of the newest commit that has state.

        Starts at the head — after a run that raised, its last save — and
        walks back past commits with an empty tree (state that could not be
        serialized). Returns the snapshot's root as the top-level
        :class:`State` together with the snapshot itself, whose scopes and
        cursors the run takes as it reaches their paths.

        Skipping is logged: the restored state may predate the head by
        whole runs (a stateless ``$end`` sends the walk into the previous
        run). Returns ``(fallback(), None)`` when the history is empty or
        holds no commit with state (warning in the latter case).

        Captures where ``HEAD`` points at the start of the walk: a writer
        that commits between checkout and this run's first commit raises
        :class:`~llm_gent.flow.ConcurrentWriteError` instead of being
        absorbed silently.
        """
        skipped: list[str] = []
        head: str | None = None
        async for commit in self._history.commits():
            if head is None:
                head = commit.id  # first in the walk is HEAD
            snapshot = await self._history.snapshot(commit)
            if snapshot.has_state:
                if skipped:
                    self._warn_skipped(skipped, commit)
                # Pass expect_head so a concurrent writer is detected at first commit.
                if head is not None:
                    await self._ctx.continue_from(head, expect_head=head)
                self.checked_out = commit.id
                return self._root_state(snapshot.root), snapshot
            skipped.append(commit.meta.node_path)
        if skipped:
            self.flow._lg.warning(
                "no commit with usable state to resume from; starting from state=",
                extra={"client_flow_id": self._history.client_flow_id, "skipped": skipped},
            )
        return fallback(), None

    async def checkout_at(self, target: str) -> tuple[State[Any], Snapshot]:
        """Check out the commit ``target`` names; the run's commits are parented on it.

        ``target`` is a commit hash of the history or a checkpoint name.
        The run continues from that commit's snapshot as ``latest`` does
        from the newest one. Its first commit moves ``HEAD`` from where it
        points now (compare-and-set), and the commits written after
        ``target`` leave the history's line: ``latest`` no longer sees
        them. They stay resumable by hash until a prune of the repository
        deletes them. A run that fails before its first commit leaves
        ``HEAD`` where it was.

        Raises:
            ValueError: The history holds no commit ``target`` (a hash) or
                no checkpoint named ``target`` (the error lists the names it
                has), ``target`` cannot name a checkpoint, or the commit
                holds no state.
        """
        commit = await self._resolve(target)
        snapshot = await self._history.snapshot(commit)
        if not snapshot.has_state:
            raise ValueError(
                f"history {self._history.client_flow_id!r}: commit {commit.id} "
                "holds no state to resume from"
            )
        root = self._root_state(snapshot.root)
        await self._ctx.continue_from(commit.id)
        self.checked_out = commit.id
        return root, snapshot

    async def restart(
        self, target: str, fallback: Callable[[], State[Any]]
    ) -> tuple[State[Any], Snapshot | None]:
        """Check out ``target`` to start over from: its root state and top-level resources alone.

        ``target`` is ``"latest"`` (as :meth:`checkout`, including its
        fallback on an empty history), a commit hash or a checkpoint name
        (as :meth:`checkout_at`); the run's commits are parented on the
        commit. Of its snapshot the run takes the root state and the
        accounting of the resources the top-level flow declares; positions,
        child scopes, paused turns, set signals and the top-level run's
        per-run resources stay behind, so every chain starts at its first
        step whatever the flow's structure is now.

        Raises:
            ValueError: As :meth:`checkout_at`, for a hash or a name.
        """
        if target == "latest":
            root, snapshot = await self.checkout(fallback)
        else:
            root, snapshot = await self.checkout_at(target)
        return root, None if snapshot is None else _restart_snapshot(snapshot)

    async def _resolve(self, target: str) -> FlowCommit:
        """The commit ``target`` names: a commit hash of this history, else a checkpoint name."""
        history = self._history
        if is_commit_hash(target):
            commit = await history.commit(target)
            if commit is None:
                raise ValueError(f"history {history.client_flow_id!r} has no commit {target}")
            return commit
        commit = await history.checkpoint(target)
        if commit is None:
            names = await history.checkpoint_names()
            raise ValueError(
                f"history {history.client_flow_id!r} has no checkpoint named {target!r}; "
                f"checkpoints: {names}"
            )
        return commit

    def _warn_skipped(self, skipped: list[str], restored: FlowCommit) -> None:
        """Log the stateless commits the checkout walked past, and where it landed."""
        self.flow._lg.warning(
            "resume skipped commits without state",
            extra={
                "client_flow_id": self._history.client_flow_id,
                "skipped": skipped,
                "restored_commit": restored.id,
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


def _restart_snapshot(snapshot: Snapshot) -> Snapshot:
    """What a restart takes of ``snapshot``: its root state and the top-level declared resources.

    The top-level flow's declared resources are the root path's
    :data:`~llm_gent.flow.state.snapshot.RESOURCES` cursor entry; every
    other cursor entry and every child scope is a position of the run that
    wrote the snapshot.
    """
    root_cursors = snapshot.cursors.get(path_str(()), {})
    kept = {RESOURCES: root_cursors[RESOURCES]} if RESOURCES in root_cursors else {}
    return Snapshot(
        has_state=snapshot.has_state,
        root=snapshot.root,
        cursors={path_str(()): kept} if kept else {},
    )


async def apply_clean_exit_retention(run: Run, final_state: State[Any]) -> None:
    """Apply the history's retention policy on the clean-exit path.

    Only a run that finished every step gets here: a halted run ends in its
    halt commit (:func:`commit_halt`), kept regardless of policy. A run is
    clean even when the halt was set during its last step — the step
    completed, so there is nothing left to resume. On a clean exit:
    ``gc_on_success`` deletes the history; ``retain`` keeps the record and commits
    ``final_state`` tagged ``complete``, so a later ``Resume("latest")``
    continues from the finished run's final state.

    A resumed run that finished without reaching some of its saved scopes
    or cursors drops them first (:func:`_drop_unreached_scopes`), so the
    final commit holds the root scope and the top-level flow's resources
    alone.
    """
    ctx = run.checkpoint_ctx
    if ctx is None:
        return
    _drop_unreached_scopes(run, ctx.client_flow_id)
    if ctx.retention == "gc_on_success":
        await ctx.delete_history()
    else:
        await _run_checkpoint_written(run, await commit_completion(run, final_state))


def _drop_unreached_scopes(run: Run, client_flow_id: str) -> None:
    """Forget saved scopes and cursors a finished run never reached, with a warning naming them.

    The run executed every step it was going to, so a saved entry nothing
    took belongs to no step of this flow: its step was removed, or the run
    took another branch arm or fewer map items or iterate passes. Kept, it
    would ride along in every later snapshot and could be handed to a step
    that reuses its node id.
    """
    dropped = run.scopes.drop_saved()
    if dropped:
        run.lg.warning(
            "resumed run finished without reaching saved entries; dropped them",
            extra={"client_flow_id": client_flow_id, "paths": [path_str(p) for p in dropped]},
        )


async def commit_completion(run: Run, final_state: State[Any]) -> FlowCommit:
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
    ctx = run.checkpoint_ctx
    assert ctx is not None
    tree = _put_root_tree(run, final_state)
    commit = await ctx.save_completion_commit(tree)
    await ctx.put_tag(COMPLETE_TAG, commit.id)
    return commit


async def commit_halt(run: Run) -> None:
    """Commit the run's halt checkpoint, once everything the halt stopped has stopped.

    Every structure the halt stopped — at any depth, in every map item —
    left its scope and cursor registered where it stopped, so the snapshot
    holds the whole run's position: resume continues each part there. The
    commit's metadata records where the halt was first observed. A no-op
    without a checkpointer.
    """
    ctx = run.checkpoint_ctx
    if ctx is None:
        return
    at = run.halt_at
    assert at is not None, "a halted run reached its end without observing the halt"
    try:
        commit = await ctx.save_scope_commit(
            at.ancestor_chain, at.iteration, at.node_id, run.scopes, at.state, "halted"
        )
    except TypeError:
        # Serialization failed: no checkpoint exists, don't claim one does.
        raise
    except Exception as e:
        # Store/IO errors propagate: the caller expects the halt checkpoint to exist.
        run.lg.warning("halt commit could not be written", extra={"exception": e})
        raise
    await _run_checkpoint_written(run, commit)


async def _run_checkpoint_written(run: Run, commit: FlowCommit) -> None:
    """Call the top-level hooks' ``after_checkpoint`` for one of the run's own commits."""
    hooks = None if run.hooks is None else run.hooks.hooks
    await checkpoint_written(hooks, None, run.info, commit)


def _put_root_tree(run: Run, state: State[Any]) -> Tree:
    """Put the run's snapshot, or an empty tree when its root ``state`` cannot be serialized.

    At the end of a run every child scope has closed, so the snapshot
    holds the root scope, and the top-level flow's resources — its cost
    tracker among them — when it has any
    (:func:`~llm_gent.flow.resource._runtime.run_resources`).
    """
    ctx = run.checkpoint_ctx
    assert ctx is not None
    if _serializable(run, state):
        return ctx.put_snapshot(run.scopes)
    tree = Tree()
    ctx.put(tree)
    return tree


def _serializable(run: Run, state: State[Any]) -> bool:
    """True when ``state`` can be checkpointed; warn and return False otherwise."""
    try:
        canonical_json(serialize_state_data(state.data))
    except (TypeError, ValueError) as e:
        run.lg.warning(
            "run state is not serializable; framework commit carries no state",
            extra={"exception": e},
        )
        return False
    return True
