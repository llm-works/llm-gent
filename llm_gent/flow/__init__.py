# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Agent flow framework — role-based verb dispatch on top of saia.

Public surface (what a run may use and how roles become SAIA instances,
:class:`~llm_gent.core.Environment`, :class:`~llm_gent.core.Role` and
:class:`~llm_gent.core.SAIAFactory`, are :mod:`llm_gent.core`'s):

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
- :class:`FlowRunner` — executes flows: ``start(flow, ...)`` returns a
  :class:`FlowRun` to await, hold or halt; :class:`Fresh`, :class:`Resume`
  and :class:`Restart` (a :data:`Start`) say whether a run starts from its
  state or from a commit of its history; its ``env=``
  (:class:`~llm_gent.core.Environment`) is what the runs may use
- :class:`Factory` — app-scoped :class:`Flow` builder (captures ``lg``
  and the shared declarations); preferred entry point at the application
  boundary
- :class:`Resource` — protocol of a run-scoped object checkpointed with
  the run (``snapshot()`` / ``restore(data)``, optional ``child(...)`` for
  a child per run); :class:`ResourceKey` — its typed handle, attached with
  :meth:`Flow.with_resource` and read with ``ctx.resource(key)``;
  :func:`resource_method` — a fluent name of the app's own for one
  (``flow.with_stats(...)``); :data:`COST` — the cost tracker's key (the
  run's is ``FlowRunner.start(cost=)``; ``with_budget`` / ``ctx.cost`` are
  sugar over it)
- :class:`Loop` — Flow-body primitive wrapping one ``saia.complete()``
  invocation with lifecycle hooks + halt bridging; CAS-native
  pause/resume via the framework halt-save site
- :class:`LoopFactory` — app-scoped :class:`Loop` builder; its
  ``with_halt`` gives its Loops a halt of their own, over the run's
- Checkpoints — a runner built with ``store=`` (:class:`FlowRunner`)
  keeps each run's history, ``start(flow, history=...)``, in
  one repository of an :class:`appinfra.cas.Store`
  (``MemoryStore``, ``FileStore``, :class:`~llm_gent.flow.stores.PgStore`):
  snapshots of the run — every scope and running structure's position,
  paused SAIA turns included — at save points and when the halt stops it
- :class:`History` — read API over one checkpointed history: head, last
  completed run, commit chain, per-commit state and flow structure, as
  :class:`FlowCommit`; :class:`HistoryCorrupt` when a referenced object is
  missing from the store
- :class:`Structure` — a flow's composition tree by step identity;
  :class:`StructureDiff` — how two structures differ (``Structure.diff``)
- :class:`Failure` — sentinel returned for a failed item in ``Flow.map(strict=False)``;
  :class:`RestoredError` — its exception when the failure was restored from a checkpoint
- :class:`Skipped` — sentinel returned for an item gated out by
  ``Flow.guard`` on a ``Flow.map`` node
- :class:`Interrupted` — raised by a step that stops because of the halt
  before finishing its work, so it runs again on resume
- :data:`HALTED` — what awaiting the run ``FlowRunner.start`` returns gives when the halt
  stopped it
- :data:`UNSET` — "no value here" sentinel (distinct from ``None``), used by
  :meth:`FlowRunner.start`'s ``state=`` default and by rescue callbacks'
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
from .checkpoint import CheckpointPolicy, ConcurrentWriteError, Retention
from .context import Context
from .factory import Factory
from .flow import Flow
from .history import History, HistoryCorrupt
from .hooks import Hook, HookDenied, Pause
from .loop import Loop, LoopFactory
from .nodes import HALTED, UNSET, Failure, Halted, Interrupted, RestoredError, Skipped, Unset
from .resource import COST, Resource, ResourceKey, resource_method
from .run._run import Run as FlowRun
from .run.runner import Runner as FlowRunner
from .run.start import Fresh, Restart, Resume, Start
from .state import State, StateData, StateDataclass, StateFactory, TypeStateFactory
from .state.cas import CommitMeta, FlowCommit
from .structure import StepPath, Structure, StructureDiff, path_label
from .verb import VerbCallable, verb


__all__ = [
    "COST",
    "HALTED",
    "UNSET",
    "CheckpointPolicy",
    "CommitMeta",
    "ConcurrentWriteError",
    "Context",
    "Factory",
    "Failure",
    "Flow",
    "FlowCommit",
    "FlowRun",
    "FlowRunner",
    "Fresh",
    "Halted",
    "History",
    "HistoryCorrupt",
    "Hook",
    "HookDenied",
    "Interrupted",
    "Loop",
    "LoopFactory",
    "Pause",
    "Resource",
    "ResourceKey",
    "RestoredError",
    "Restart",
    "Resume",
    "Retention",
    "Skipped",
    "Start",
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
