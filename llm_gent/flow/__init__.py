# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent flow framework — role-based verb dispatch on top of saia.

Public surface:

- :class:`Role` — pure config for a persona (name, backend, model, sampling)
- :class:`SAIAFactory` — protocol that turns a Role into a saia instance
- :func:`verb` — decorator marking an async function as a role-bound verb
- :data:`VerbCallable` — the type of a verb: an async callable taking
  ``(ctx, *args, **kwargs)``
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
- :class:`Factory` — app-scoped :class:`Flow` builder (captures ``lg``
  and one :class:`SAIAFactory`); preferred entry point at the application
  boundary
- :class:`Resource` — protocol of a run-scoped object checkpointed with
  the run (``snapshot()`` / ``restore(data)``, optional ``child(...)`` for
  a child per run); :class:`ResourceKey` — its typed handle, attached with
  :meth:`Flow.with_resource` and read with ``ctx.resource(key)``;
  :func:`resource_method` — a fluent name of the app's own for one
  (``flow.with_stats(...)``); :data:`COST` — the cost tracker's key
  (``with_cost_tracker`` / ``with_budget`` / ``ctx.cost`` are sugar over it)
- :class:`Loop` — Flow-body primitive wrapping one ``saia.complete()``
  invocation with lifecycle hooks + halt bridging; CAS-native
  pause/resume via the framework halt-save site
- :class:`LoopFactory` — app-scoped :class:`Loop` builder (mirrors the
  flow :class:`Factory` for ``with_halt``); pair on the same halt event to
  thread it across a mixed Loop-and-Flow tree
- :class:`CheckpointStore` — Flow-level pause/resume Protocol; persists
  snapshots of a run — every scope and running structure's position,
  paused SAIA turns included — at save points and when the halt stops it
- :class:`History` — read API over one checkpointed history: head, last
  completed run, commit chain, per-commit state and flow structure;
  :class:`HistoryCorrupt` when a referenced object is missing from the store
- :class:`Structure` — a flow's composition tree by step identity;
  :class:`StructureDiff` — how two structures differ (``Structure.diff``)
- :func:`collect_unreachable` — delete the objects of a history no ref
  reaches (left by ``resume=<name>`` or a process that died mid-commit)
- :class:`Failure` — sentinel returned for a failed item in ``Flow.map(strict=False)``;
  :class:`RestoredError` — its exception when the failure was restored from a checkpoint
- :class:`Skipped` — sentinel returned for an item gated out by
  ``Flow.guard`` on a ``Flow.map`` node
- :class:`Interrupted` — raised by a step that stops because of the halt
  before finishing its work, so it runs again on resume
- :data:`HALTED` — what ``Flow.run`` returns when the halt stopped the run
- :data:`UNSET` — "no value here" sentinel (distinct from ``None``), used by
  :meth:`Flow.run`'s ``state=`` default and by rescue callbacks'
  ``pending_input`` positional
- Archetype decorators: :func:`planner`, :func:`extractor`, :func:`grader`,
  :func:`synthesizer` — semantic tags for the standard agent shape

Aggregators for ``Flow.map(aggregate=...)``, mostly an ensemble's votes
(``.map([judge_a, judge_b], aggregate=majority)``), from
:mod:`llm_gent.flow.aggregate`: :func:`majority`, :func:`unanimous`,
:func:`mean`, :func:`weighted`.

State (facts, KG, RAG, conversation) is not provided here — consumers
compose those from ``llm_kelt`` (default) or their own implementations, and
mount them via the existing trait system.
"""

from .aggregate import majority, mean, unanimous, weighted
from .archetypes import extractor, grader, planner, synthesizer
from .checkpoint import CheckpointPolicy, CheckpointStore, ConcurrentWriteError, ResumeMode
from .context import Context
from .factory import Factory, SAIAFactory
from .flow import Flow
from .gc import collect_unreachable
from .history import History, HistoryCorrupt
from .loop import Loop, LoopFactory
from .nodes import HALTED, UNSET, Failure, Halted, Interrupted, RestoredError, Skipped, Unset
from .resource import COST, Resource, ResourceKey, resource_method
from .role import Role
from .state import State, StateData, StateDataclass, StateFactory, TypeStateFactory
from .structure import StepPath, Structure, StructureDiff, path_label
from .verb import VerbCallable, verb


__all__ = [
    "COST",
    "HALTED",
    "UNSET",
    "CheckpointPolicy",
    "CheckpointStore",
    "ConcurrentWriteError",
    "Context",
    "Factory",
    "Failure",
    "Flow",
    "Halted",
    "History",
    "HistoryCorrupt",
    "Interrupted",
    "Loop",
    "LoopFactory",
    "Resource",
    "ResourceKey",
    "RestoredError",
    "ResumeMode",
    "Role",
    "SAIAFactory",
    "Skipped",
    "State",
    "StateData",
    "StateDataclass",
    "StateFactory",
    "StepPath",
    "Structure",
    "StructureDiff",
    "TypeStateFactory",
    "Unset",
    "VerbCallable",
    "collect_unreachable",
    "extractor",
    "grader",
    "majority",
    "mean",
    "path_label",
    "planner",
    "resource_method",
    "synthesizer",
    "unanimous",
    "verb",
    "weighted",
]
