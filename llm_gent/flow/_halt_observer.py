# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Halt observation seam: detection primitive + save-site kernel.

Three halt-observation sites in the executor persist a halt commit
when the ambient halt event fires: the iterate boundary check
(:func:`_run_iterate`), the chain between-step walker
(:meth:`Flow._observe_chain_halt`), and the chain trailing edge
(:meth:`Flow._observe_final_chain_halt`). Two additional sites in
map-item dispatch (strict + non-strict) observe halt to short-
circuit the item as :class:`Skipped` without persisting; they use
:func:`is_halt_signaled` only.

:class:`HaltSaveObserver` owns the shared kernel of the save sites:
halt detection + delegation to :func:`_save_halt_checkpoint`. Each
save site keeps its own walker-side context (which ``node_id``,
which ``iteration``, which state, and site-specific guards like
top-level runtime identity and checkpointer bindings) and hands it
to :meth:`HaltSaveObserver.save_if_signaled`. Because every save-
site routes through this one call, the set of halt-save sites is
the set of :meth:`save_if_signaled` callers — the coverage matrix
is grep-discoverable and adding a new site means adding a call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .nodes import _RunEnv
    from .state import State


def is_halt_signaled(env: _RunEnv) -> bool:
    """True when the run's ambient halt event is set.

    Shared detection primitive for every halt-observation site,
    save and skip alike.
    """
    return env.halt is not None and env.halt.is_set()


class HaltSaveObserver:
    """Halt-save kernel: single entrypoint for persisting a halt commit.

    The three save-site callers (iterate boundary, chain between-
    step, chain tail) each hold their own walker-side context — the
    ``node_id`` to save at, the iteration counter, the child state,
    plus site-specific pre-conditions like top-level runtime
    identity and node-id selection between the just-completed and
    not-yet-run chain steps. They resolve those first and then
    call :meth:`save_if_signaled` with the resolved values. This
    class holds the shared detection + save kernel that used to be
    inlined at each site.
    """

    @staticmethod
    async def save_if_signaled(
        env: _RunEnv,
        iteration: int,
        node_id: str,
        state: State[Any],
    ) -> bool:
        """Save a halt commit at ``node_id`` when halt is signaled.

        Returns ``True`` when the save fired (caller should
        ``break``/``return`` its walker); ``False`` otherwise.
        Idempotency comes from the ``_halt_saved`` latch on
        ``env.runtime`` — a redundant call after a first save is
        a no-op inside :func:`_save_halt_checkpoint`.
        """
        if not is_halt_signaled(env):
            return False
        # Local import to avoid an import cycle: _halt_observer is
        # imported by _executor.py which owns _save_halt_checkpoint.
        from ._executor import _save_halt_checkpoint

        await _save_halt_checkpoint(env, iteration, node_id, state)
        return True
