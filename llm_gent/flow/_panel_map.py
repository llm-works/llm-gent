# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""A :class:`~llm_gent.flow.Panel` inside a run: a map over its verbs, in the run's snapshots.

The Panel run is the ``k``-th Panel the calling step starts, at
``<step>/panel/<k>`` (:meth:`ScopeRegistry.next_panel`); verb ``i`` is map
item ``i``, run as a one-step body flow at ``<step>/panel/<k>/i/<i>``. The
map's cursor holds the finished verbs and their results: a rerun of the
step finds the same Panel run at the same path, finished verbs do not run
again, and a running one continues at its position (a paused Loop turn
resumes mid-turn).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ._map import MapItemRunner, MapRunner
from ._node_id import _descend_context
from .nodes import _Map


if TYPE_CHECKING:
    from .flow import Flow
    from .nodes import _RunEnv
    from .panel import AggregateFn
    from .state.snapshot import ScopePath


async def run_panel(
    env: _RunEnv,
    node_id: str,
    verbs: list[Any],
    aggregate: AggregateFn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run ``verbs`` as a map under the step ``node_id`` of ``env``'s flow; aggregate the results.

    Raises:
        Interrupted: The halt stopped a verb before it finished.
    """
    from .flow import Flow

    path = env.scopes.next_panel(env.owner_path(node_id))
    bodies = [Flow(env.runtime._lg).call(v) for v in verbs]
    # body is a placeholder—_PanelItemRunner._dispatch_body selects the correct body per item.
    mp = _Map(body=bodies[0], items=None, aggregate=aggregate, strict=True)
    runner = _PanelRunner(mp, env, node_id, path, bodies, args, kwargs)
    return await runner.run((list(range(len(verbs))),))


class _PanelRunner(MapRunner):
    """A map whose item ``i`` runs ``bodies[i]`` with the Panel's arguments."""

    def __init__(
        self,
        mp: _Map,
        env: _RunEnv,
        node_id: str,
        path: ScopePath,
        bodies: list[Flow],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        super().__init__(mp, env, node_id, path)
        self.bodies = bodies
        self.args = args
        self.kwargs = kwargs

    def _item_runner(self, item: Any, index: int, merge_lock: asyncio.Lock) -> MapItemRunner:
        return _PanelItemRunner(self, item, index, merge_lock)


class _PanelItemRunner(MapItemRunner):
    """Runs one Panel verb: its body flow, with the Panel's arguments, at the item's path."""

    owner: _PanelRunner

    async def _dispatch_body(self, child_state: Any) -> Any:
        """Run verb ``item_index``'s body; its chain context is unique to this Panel run."""
        env = self.env
        owner = self.owner
        context_key = f"{'/'.join(owner.path[-2:])}:{self.item_index}"
        return await owner.bodies[self.item_index]._run_as_subflow(
            *owner.args,
            state=child_state,
            runtime=env.runtime,
            parent_halt=env.halt,
            parent_budget=env.budget,
            parent_checkpoint_ctx=env.checkpoint_ctx,
            parent_chain_context=_descend_context(self.node_id, context_key),
            parent_ancestor_chain=env.ancestor_chain + (self.node_id,),
            parent_extra=env.extra,
            parent_policy=env.policy,
            parent_path=self.path,
            **owner.kwargs,
        )
