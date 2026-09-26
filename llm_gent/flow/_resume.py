# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resume hydration: load the latest halt commit, rebuild state + replay plan.

Extracted from :class:`Flow`. A :class:`Resume` is constructed per
``run(resume=True)`` call with the flow being resumed; its public
:meth:`hydrate` returns the ``(State, _ResumeReplay | None)`` pair the
executor threads through the walk.

The flow provides the checkpointer + ``client_flow_id`` + state
factory; :class:`Resume` uses them to (a) fetch the latest commit
under the trajectory ref, (b) walk its tree to collect one JSON
payload per scope, (c) hydrate the top-level :class:`State` from
the root scope, and (d) hand every non-root scope payload to
``_ResumeReplay.intermediate_scope_data`` so descent sites
(``_consume_scope_data``) can restore their own scope in order.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .checkpoint import maybe_await
from .state import State
from .state.cas import Commit, Tree


if TYPE_CHECKING:
    from .flow import Flow
    from .nodes import _ResumeReplay


class Resume:
    """Hydrate initial state + replay plan for a ``run(resume=True)`` call.

    Constructed with the flow being resumed. Caller invokes
    :meth:`hydrate` with the fallback :class:`State` (the wrapped
    ``run(state=...)`` payload) and receives the hydrated state +
    replay tuple. On a fresh run (no commit yet, or the trajectory
    already stamped a ``$complete`` marker), returns ``(fallback,
    None)`` so the caller falls through to a normal fresh run.
    """

    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    async def hydrate(self, fallback: State[Any]) -> tuple[State[Any], _ResumeReplay | None]:
        """Load the latest commit and reconstruct state + replay context.

        Sequence:

        1. :meth:`CheckpointStore.resolve_ref` under
           :attr:`Flow._client_flow_id` returns the latest commit
           hash across every ``node_path``, or ``None`` (fresh run
           — no prior checkpoint).
        2. :meth:`CheckpointStore.get_object` fetches the commit
           bytes; :meth:`Commit.from_bytes` re-derives
           :class:`CommitMeta` + :class:`ProducedBy` +
           :class:`TraceRef`.
        3. The commit's ``root_tree_hash`` fetches the
           :class:`Tree`; each :class:`TreeEntry` fetches its Blob.
           Scope order is root → leaf via the zero-padded
           ``scope_id`` (canonical sort).
        4. The root scope's payload rehydrates the top-level
           :class:`State` (via ``state_factory.restore`` when
           bound, passthrough otherwise). The leaf scope's payload
           rides on the :class:`_ResumeReplay` as
           ``child_state_data`` for the save-point iterate to
           restore its own scope.
        5. ``node_path`` splits on ``"/"`` back into the ancestor
           chain the executor's head-pop replay expects. blake2b
           hex has no slashes, so the round-trip is exact.

        Returns ``(fallback, None)`` when no commit exists yet or
        the trajectory has a completion marker.
        """
        flow = self.flow
        assert flow._checkpointer is not None
        if flow._client_flow_id is None:
            label = flow._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} was run with resume=True but has no "
                f"client_flow_id — call .with_checkpointer(store, client_flow_id) first"
            )
        loaded = await self._load_latest_commit_scopes()
        if loaded is None:
            return fallback, None
        commit, scope_data = loaded
        # A completion marker (stamped on clean exit under "retain") means
        # the trajectory finished successfully — do not replay.
        if commit.meta.node_path == "$complete":
            return fallback, None
        await flow._resume_saia_turns.load_from_commit(
            flow._checkpointer, flow._client_flow_id, commit
        )
        return self._split_scopes(commit, scope_data)

    async def _load_latest_commit_scopes(self) -> tuple[Commit, list[Any]] | None:
        """Resolve the latest commit and walk its tree.

        Returns ``(Commit, scope_data)`` where ``scope_data`` is
        root → leaf JSON payloads (one per :class:`Tree` entry).
        Returns ``None`` when the ref, commit, tree, or any blob is
        missing — the caller treats each miss as "no resumable
        checkpoint" and falls through to a fresh run.
        """
        flow = self.flow
        assert flow._checkpointer is not None
        assert flow._client_flow_id is not None
        commit_hash = await maybe_await(flow._checkpointer.resolve_ref(flow._client_flow_id))
        if commit_hash is None:
            return None
        commit_bytes = await maybe_await(
            flow._checkpointer.get_object(flow._client_flow_id, "commit", commit_hash)
        )
        if commit_bytes is None:
            return None
        commit = Commit.from_bytes(commit_bytes)
        tree_bytes = await maybe_await(
            flow._checkpointer.get_object(flow._client_flow_id, "tree", commit.root_tree_hash)
        )
        if tree_bytes is None:
            return None
        tree = Tree.from_bytes(tree_bytes)
        scope_data: list[Any] = []
        for entry in tree.entries:
            blob = await maybe_await(
                flow._checkpointer.get_object(flow._client_flow_id, "blob", entry.child_hash)
            )
            if blob is None:
                return None
            scope_data.append(json.loads(blob.decode("utf-8")))
        return commit, scope_data

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

        flow = self.flow
        root_raw = scope_data[0] if scope_data else None
        hydrated_root = (
            root_raw
            if flow._state_factory is None or root_raw is None
            else flow._state_factory.restore(root_raw if isinstance(root_raw, dict) else {})
        )
        intermediate_raw = tuple(scope_data[1:])
        path_tuple = tuple(commit.meta.node_path.split("/")) if commit.meta.node_path else ()
        return (
            State(data=hydrated_root, _factory=flow._state_factory),
            _ResumeReplay(
                remaining_path=path_tuple,
                full_path=path_tuple,
                iteration=commit.meta.iteration,
                child_state_data=None,
                intermediate_scope_data=intermediate_raw,
            ),
        )
