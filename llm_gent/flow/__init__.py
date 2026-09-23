# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent flow framework — role-based verb dispatch on top of saia.

Public surface:

- :class:`Role` — pure config for a persona (name, backend, model, sampling)
- :class:`SAIAFactory` — protocol that turns a Role into a saia instance
- :func:`verb` — decorator marking an async function as a role-bound verb
- :class:`Context` — runtime environment injected into every verb
- :class:`State` — scope-aware wrapper around the user-owned payload on
  ``ctx.state`` (``.data`` reaches the payload; ``.root()`` walks to the
  outermost scope)
- :class:`StateData` — serialization contract for ``state.data`` payloads
  that need to round-trip through a checkpoint (``to_dict`` + ``from_dict``)
- :class:`StateDataclass` — opt-in mixin satisfying :class:`StateData` for
  flat dataclasses via :func:`dataclasses.asdict` and ``cls(**data)``
- :class:`StateFactory` — protocol the framework calls at checkpoint
  restore to reconstruct ``ctx.state.data``; captures runtime handles
  (Logger, storage) at construction and threads them into the restored
  state
- :class:`TypeStateFactory` — :class:`StateFactory` adapter for stateless
  state types (wraps a :class:`StateData` class so the framework's restore
  call routes through ``state_type.from_dict``)
- :class:`Flow` — verb registry + role-routed dispatch + fluent composition
- :class:`FlowFactory` — app-scoped :class:`Flow` builder (captures ``lg``
  and one :class:`SAIAFactory`); preferred entry point at the application
  boundary
- :class:`Loop` — Flow-body primitive wrapping one ``saia.complete()``
  invocation with lifecycle hooks + halt bridging + optional checkpointer
- :class:`LoopFactory` — app-scoped :class:`Loop` builder (mirrors
  :class:`FlowFactory` for ``with_halt``); pair on the same halt event to
  thread it across a mixed Loop-and-Flow tree
- :class:`CheckpointStore` — Flow-level pause/resume Protocol; persists
  composition-graph state at ``.iterate`` boundaries
- :class:`LoopCheckpointStore` — 3-method Protocol :class:`Loop` drives
  for SAIA-turn pause/resume (distinct layer from the Flow-level Protocol)
- :class:`Failure` — sentinel returned for a failed item in ``Flow.map(strict=False)``
- :class:`Skipped` — sentinel returned for an item gated out by
  ``Flow.guard`` on a ``Flow.map`` node
- :data:`UNSET` — "no value here" sentinel (distinct from ``None``), used by
  :meth:`Flow.run`'s ``state=`` default and by rescue callbacks'
  ``pending_input`` positional
- :class:`Panel` — fan-out N verbs in parallel + aggregate their results
- Archetype decorators: :func:`planner`, :func:`extractor`, :func:`grader`,
  :func:`synthesizer` — semantic tags for the standard agent shape

Aggregation helpers exposed via :mod:`llm_gent.flow.panel`: ``majority``,
``unanimous``, ``mean``, ``weighted``.

State (facts, KG, RAG, conversation) is not provided here — consumers
compose those from ``llm_kelt`` (default) or their own implementations, and
mount them via the existing trait system.
"""

from .archetypes import extractor, grader, planner, synthesizer
from .checkpoint import CheckpointPolicy, CheckpointStore
from .context import Context
from .factory import FlowFactory, SAIAFactory
from .flow import Flow
from .loop import Loop, LoopCheckpointStore, LoopFactory
from .nodes import UNSET, Failure, Skipped, Unset
from .panel import Panel
from .role import Role
from .state import State, StateData, StateDataclass, StateFactory, TypeStateFactory
from .verb import verb


__all__ = [
    "UNSET",
    "CheckpointPolicy",
    "CheckpointStore",
    "Context",
    "Failure",
    "Flow",
    "FlowFactory",
    "Loop",
    "LoopCheckpointStore",
    "LoopFactory",
    "Panel",
    "Role",
    "SAIAFactory",
    "Skipped",
    "State",
    "StateData",
    "StateDataclass",
    "StateFactory",
    "TypeStateFactory",
    "Unset",
    "extractor",
    "grader",
    "planner",
    "synthesizer",
    "verb",
]
