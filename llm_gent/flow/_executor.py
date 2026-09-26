# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow execution helpers — walk the composition graph, dispatch each node.

Internal to :mod:`llm_gent.flow`: the coroutines that :meth:`Flow.run` calls
into to execute each node in order (verbs, subflows, and the
branch/iterate/map control-flow primitives), together with the scoped-state
projection/merge plumbing.

Depends on :mod:`.flow` for the :class:`Flow` class itself, which appears in
three ``isinstance`` checks used to distinguish subflow nodes from verb
nodes and control-flow primitives (:func:`_build_ctx`,
:func:`_invoke_target`, :func:`_target_label`). Each uses a localized late
import (``from .flow import Flow`` inside the function body) to break the
circular dependency between the executor and the class it operates on.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .checkpoint import maybe_await
from .context import Context
from .nodes import (
    UNSET,
    StateMerge,
    StateProject,
    UntilFn,
    _Branch,
    _Iterate,
    _Map,
    _Node,
    _ResumeReplay,
    _RunEnv,
)
from .state import State, StateFactory
from .state.cas import (
    Blob,
    Commit,
    CommitMeta,
    CommitOutcome,
    ProducedBy,
    TraceRef,
    Tree,
    TreeEntry,
    canonical_json,
)


if TYPE_CHECKING:
    from .flow import Flow


def _build_ctx(target: Any, env: _RunEnv, node_id: str | None = None) -> Context[Any]:
    """Build the Context passed to the node's verb (and to its hooks).

    Verb nodes get a role-bound ctx; ``ctx.saia`` resolves lazily on first
    read via the flow's SAIAFactory. Subflow nodes and control-flow nodes
    (branch/iterate/map) get an ambient ctx with ``role=None`` — those nodes
    have no single role, so ``ctx.saia`` returns ``None`` (each inner verb
    builds its own role-bound ctx as it runs).

    ``node_id`` is the composition-graph id of the node the ctx is being
    built for; threaded onto ``ctx._node_id`` so :meth:`Context.checkpoint`
    can address the currently-executing position. ``None`` for hook ctxs
    (until predicates, on_error, on_item_complete) where no verb is
    running under a single node id.
    """
    from .flow import Flow

    traits = env.runtime._traits
    if isinstance(target, Flow | _Branch | _Iterate | _Map):
        return Context(
            role=None,
            state=env.state,
            flow=env.runtime,
            traits=traits,
            halt=env.halt,
            budget=env.budget,
            extra=env.extra,
            _env=env,
            _node_id=node_id,
        )
    return Context(
        role=target.role,
        state=env.state,
        flow=env.runtime,
        traits=traits,
        halt=env.halt,
        budget=env.budget,
        extra=env.extra,
        _env=env,
        _node_id=node_id,
    )


async def _execute_node(
    node: _Node,
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Invoke a node's target with cancellation-safe rescue + optional after hook.

    ``asyncio.CancelledError`` propagates unchanged past any rescue policy.
    Other exceptions surface unless a rescue is attached; the rescue's
    return value (awaited if awaitable) becomes the node's result.
    """
    target_name = _target_label(node.target)
    env.lg.debug("executing node", extra={"target": target_name})
    try:
        result = await _invoke_target(node, ctx, env, node_args, node_kwargs, node_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if node.rescue is None:
            raise
        env.lg.warning("rescue policy invoked", extra={"target": target_name, "exception": exc})
        pending_input = node_args[0] if node_args else UNSET
        fallback = node.rescue(exc, pending_input, ctx)
        if inspect.isawaitable(fallback):
            fallback = await fallback
        result = fallback
    if node.after is not None:
        env.lg.debug("running after hook", extra={"target": target_name})
        hook_result = node.after(result, ctx)
        if inspect.isawaitable(hook_result):
            await hook_result
    return result


async def _invoke_target(
    node: _Node,
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Dispatch a node's target: verb, subflow, or control-flow primitive."""
    from .flow import Flow

    target = node.target
    if isinstance(target, Flow):
        return await _run_subflow(
            target,
            env,
            node.state_fn,
            node.merge_fn,
            node.state_factory,
            node_args,
            node_kwargs,
            node_id,
        )
    if isinstance(target, _Branch):
        return await _run_branch(target, ctx, env, node_args, node_id)
    if isinstance(target, _Iterate):
        from ._iterate import IterateRunner

        return await IterateRunner(target, env, node_id).run(node_args)
    if isinstance(target, _Map):
        from ._map import MapRunner

        return await MapRunner(target, env, node_id).run(node_args)
    passed_args, passed_kwargs = _filter_verb_args(target, node_args, node_kwargs)
    return await target(ctx, *passed_args, **passed_kwargs)


@dataclasses.dataclass(frozen=True)
class _VerbArity:
    """Cached signature shape of a verb callable, minus the ``ctx`` slot.

    ``var_positional`` — verb declares ``*args``. ``var_keyword`` — verb
    declares ``**kwargs``. ``positional_slots`` — count of fixed positional
    parameters after ``ctx`` (POSITIONAL_ONLY + POSITIONAL_OR_KEYWORD).
    ``keyword_names`` — names of parameters reachable by keyword
    (POSITIONAL_OR_KEYWORD + KEYWORD_ONLY). ``introspection_failed`` —
    the callable's signature could not be inspected (C callables, some
    partials); the caller then forwards args verbatim so behavior matches
    pre-filter dispatch.
    """

    var_positional: bool
    var_keyword: bool
    positional_slots: int
    keyword_names: frozenset[str]
    introspection_failed: bool = False


_VERB_ARITY_ATTR = "_llm_gent_verb_arity"
"""Attribute name used to memoize :class:`_VerbArity` on the verb itself.

Attaching to the callable ties the cache lifetime to the target — a
locally-defined verb GC'd at the end of a test cannot leak into a later
test that happens to allocate a different signature at the same id
slot. Falls back to a fresh analysis when ``setattr`` is rejected
(``__slots__``, some C-level callables, class instances forbidding
attribute assignment).
"""


def _verb_arity(target: Any) -> _VerbArity:
    """Return the arity of ``target`` (skipping ``ctx``); memoize on the target itself."""
    cached = getattr(target, _VERB_ARITY_ATTR, None)
    if isinstance(cached, _VerbArity):
        return cached
    computed = _compute_verb_arity(target)
    with contextlib.suppress(AttributeError, TypeError):
        setattr(target, _VERB_ARITY_ATTR, computed)
    return computed


def _compute_verb_arity(target: Any) -> _VerbArity:
    """Inspect ``target``'s signature and derive its :class:`_VerbArity`."""
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return _VerbArity(
            var_positional=True,
            var_keyword=True,
            positional_slots=0,
            keyword_names=frozenset(),
            introspection_failed=True,
        )
    params = list(sig.parameters.values())
    # Skip ctx if it's a fixed positional (not *args).
    fixed_pos = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if params and params[0].kind in fixed_pos:
        params = params[1:]
    return _VerbArity(
        var_positional=any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params),
        var_keyword=any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params),
        positional_slots=sum(1 for p in params if p.kind in fixed_pos),
        keyword_names=frozenset(
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ),
    )


def _filter_verb_args(
    target: Any,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Trim ``node_args``/``node_kwargs`` to what ``target``'s signature accepts.

    Verbs may declare ``(ctx)`` only and still sit at any chain position —
    the previous node's result is dropped rather than raising, so
    pure-Python verbs don't need a placeholder ``_prev`` parameter. Verbs
    declaring ``*args`` / ``**kwargs`` see the full inputs unchanged.
    Introspection failures fall through with all args forwarded so C-level
    callables and exotic partials keep working.
    """
    arity = _verb_arity(target)
    if arity.introspection_failed:
        return node_args, node_kwargs
    passed_args = node_args if arity.var_positional else node_args[: arity.positional_slots]
    passed_kwargs = (
        node_kwargs
        if arity.var_keyword
        else {k: v for k, v in node_kwargs.items() if k in arity.keyword_names}
    )
    return passed_args, passed_kwargs


async def _run_subflow(
    body: Flow,
    env: _RunEnv,
    state_fn: StateProject | None,
    merge_fn: StateMerge | None,
    state_factory: StateFactory[Any] | None,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    node_id: str,
) -> Any:
    """Run a subflow node, honoring optional scoped-state projection/merge.

    Threads the composition-tree identity into the child Flow:
    ``ancestor_chain`` gains ``node_id`` (this call's ``_Node`` is now
    an ancestor of everything inside), and ``chain_context`` becomes
    ``_descend_context(node_id, "call")`` — the hash the child's
    chain-step IDs are computed against.
    """
    from ._node_id import _descend_context

    child_replay = _pop_replay_for(env, node_id)
    raw, child_replay = _consume_scope_data(child_replay, state_fn)
    if raw is UNSET:
        child_state = await _project_state(state_fn, env.state, state_factory)
    else:
        effective_factory = state_factory if state_factory is not None else env.state._factory
        child_state = _restore_scope_state(env.state, raw, effective_factory)
    result = await body._run_as_subflow(
        *node_args,
        state=child_state,
        runtime=env.runtime,
        parent_halt=env.halt,
        parent_budget=env.budget,
        parent_checkpointer=env.checkpointer,
        parent_client_flow_id=env.client_flow_id,
        parent_chain_context=_descend_context(node_id, "call"),
        parent_ancestor_chain=env.ancestor_chain + (node_id,),
        parent_replay=child_replay,
        parent_extra=env.extra,
        parent_policy=env.policy,
        **node_kwargs,
    )
    await _merge_state(merge_fn, env.state, child_state)
    return result


def _consume_scope_data(
    replay: _ResumeReplay | None,
    state_fn: StateProject | None,
) -> tuple[Any, _ResumeReplay | None]:
    """Head-pop the next intermediate-scope payload from the replay when applicable.

    Returns ``(raw_scope_data, updated_replay)``. Returns
    ``(UNSET, replay)`` — signaling "no restored data, project fresh" —
    when any of:

    - ``replay`` is ``None`` (off the replay path).
    - ``state_fn`` is ``None`` (no scope is being created at this
      descent, so nothing to consume).
    - ``replay.intermediate_scope_data`` is empty (all middle scopes
      already consumed, or the checkpointed stack had no middle scopes).

    On a hit, the returned replay carries the tail so the child scope's
    intermediate list stays aligned with its own remaining descents.
    """
    if replay is None or state_fn is None or not replay.intermediate_scope_data:
        return UNSET, replay
    raw = replay.intermediate_scope_data[0]
    updated = dataclasses.replace(
        replay, intermediate_scope_data=replay.intermediate_scope_data[1:]
    )
    return raw, updated


def _restore_scope_state(
    parent: State[Any],
    raw: Any,
    factory: StateFactory[Any] | None,
) -> State[Any]:
    """Wrap a restored raw scope payload as a child :class:`State`.

    Companion to :func:`_project_state` — same shape as the fresh
    projection but uses ``factory.restore(raw)`` (or a passthrough when
    ``factory is None``) instead of running ``state_fn(parent.data)``.
    """
    child_payload = factory.restore(raw) if factory is not None else raw
    return State(data=child_payload, _parent=parent, _factory=factory)


async def _project_state(
    state_fn: StateProject | None,
    parent: State[Any],
    factory: StateFactory[Any] | None = None,
) -> State[Any]:
    """Build the child :class:`State` for a scoped block; pass-through when unset.

    With no projection, the subflow sees the parent's :class:`State` object
    directly — same reference, shared payload, ``is_root`` echoes the parent,
    and ``_factory`` is inherited.

    With a projection, ``state_fn(parent.data)`` produces the child payload,
    which the framework wraps as ``State(data=child_payload, _parent=parent,
    _factory=...)`` so the child's :meth:`State.root` still walks back to
    the outermost scope. The factory attached is ``factory`` if provided,
    else inherited from ``parent._factory``.
    """
    if state_fn is None:
        return parent
    child_payload = state_fn(parent.data)
    if inspect.isawaitable(child_payload):
        child_payload = await child_payload
    effective_factory = factory if factory is not None else parent._factory
    return State(data=child_payload, _parent=parent, _factory=effective_factory)


async def _merge_state(merge_fn: StateMerge | None, parent: State[Any], child: State[Any]) -> None:
    """Fold the child payload back into the parent's; no-op when unset.

    The merge callback receives the two payloads (``parent.data``,
    ``child.data``); the :class:`State` wrappers are unwrapped for the user
    so signatures match the pre-unification shape.
    """
    if merge_fn is None:
        return
    result = merge_fn(parent.data, child.data)
    if inspect.isawaitable(result):
        await result


async def _check_until(
    until_fn: UntilFn | None,
    result: Any,
    iterate_state: State[Any],
    env: _RunEnv,
    node_id: str,
) -> bool:
    """Evaluate the iterate node's until predicate with the last body result and scoped state."""
    if until_fn is None:
        return False
    ctx = Context(
        role=None,
        state=iterate_state,
        flow=env.runtime,
        traits=env.runtime._traits,
        halt=env.halt,
        budget=env.budget,
        extra=env.extra,
        _env=env,
        _node_id=node_id,
    )
    verdict = until_fn(result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    return bool(verdict)


async def _run_branch(
    br: _Branch,
    ctx: Context[Any],
    env: _RunEnv,
    node_args: tuple[Any, ...],
    node_id: str,
) -> Any:
    """Evaluate the predicate and dispatch the chosen subflow.

    Passes the branch input (``node_args[0]``, or ``None`` when the branch
    is the chain's head with no positional) as the sole positional to the
    chosen subflow. Falsy predicate with no ``else_`` returns the input
    unchanged. Both bodies share the parent's active state.

    The chosen arm's ``chain_context`` is
    ``_descend_context(node_id, "then"|"else")`` so the two arms are at
    identity-distinct positions in the composition tree even when they
    share a target Flow.
    """
    from ._node_id import _descend_context

    prev_result = node_args[0] if node_args else None
    verdict = br.when(prev_result, ctx)
    if inspect.isawaitable(verdict):
        verdict = await verdict
    chosen = br.then_flow if verdict else br.else_flow
    if chosen is None:
        _assert_replay_allows_skip(
            env,
            node_id,
            "branch predicate returned falsy with no else_ arm, but resume "
            "path expected descent through this branch — predicate behavior "
            "has changed since checkpoint.",
        )
        return prev_result
    return await chosen._run_as_subflow(
        prev_result,
        state=env.state,
        runtime=env.runtime,
        parent_halt=env.halt,
        parent_budget=env.budget,
        parent_checkpointer=env.checkpointer,
        parent_client_flow_id=env.client_flow_id,
        parent_chain_context=_descend_context(node_id, "then" if verdict else "else"),
        parent_ancestor_chain=env.ancestor_chain + (node_id,),
        parent_replay=_pop_replay_for(env, node_id),
        parent_extra=env.extra,
        parent_policy=env.policy,
    )


def _pop_replay_for(env: _RunEnv, node_id: str) -> _ResumeReplay | None:
    """Return the replay to thread through a descent under ``node_id``.

    Head-pop at descent site: when ``env.replay.remaining_path[0]``
    equals ``node_id``, the descent is on the saved ancestor chain —
    pop the head and thread the tail to the child Flow. Otherwise the
    descent is off-path (its subtree cannot contain the leaf) or the
    replay has already been consumed at the leaf, and no replay is
    threaded. Called by every descent helper (``_run_subflow``,
    ``_run_branch``, ``IterateRunner._dispatch_body``, ``_dispatch_map_body``).
    ``full_path`` is preserved verbatim across the pop so downstream
    triage messages can show the whole saved ancestor chain.
    """
    replay = env.replay
    if replay is None or not replay.remaining_path:
        return None
    if env.runtime._replay_consumed:
        return None
    if replay.remaining_path[0] != node_id:
        return None
    return dataclasses.replace(replay, remaining_path=replay.remaining_path[1:])


def _assert_replay_allows_skip(env: _RunEnv, node_id: str, reason: str) -> None:
    """Fail-fast when a skipped descent was on the replay's saved path.

    Called when a descent site decides not to descend (e.g., branch with
    falsy predicate and no ``else_`` arm). If the replay's
    ``remaining_path[0]`` equals ``node_id``, the checkpoint expected
    descent through this node — skipping it means the composition
    graph's runtime behavior has changed since the checkpoint was
    written. Raising here preserves the head-pop invariant: no chain
    step after a doomed skip runs before the error.
    """
    replay = env.replay
    if replay is None or not replay.remaining_path:
        return
    if env.runtime._replay_consumed:
        return
    if replay.remaining_path[0] != node_id:
        return
    path_repr = " → ".join(replay.full_path) if replay.full_path else "<empty>"
    raise RuntimeError(f"{reason} Saved path: {path_repr}")


async def _save_halt_checkpoint(
    env: _RunEnv,
    iteration: int,
    node_id: str,
    current_state: State[Any],
) -> None:
    """Persist an ``outcome="halted"`` commit at a halt observation point.

    Called from the executor's halt-observation sites — the between-
    iterations check in :meth:`IterateRunner.run` and the between-chain-
    steps check in :meth:`Chain._walk_steps` — so a ``run(resume=True)``
    after a halted process restart resolves to this commit and re-
    enters at the halted position.

    At most one halt-save fires per run: :attr:`Flow._halt_saved` on
    the top-level runtime latches after the first save so a halt fires
    through an inner iterate's boundary check + the outer chain-walk's
    between-steps check does not double-save (the inner save is the
    finer-grained resume anchor).

    ``iteration`` is the iterate's current counter at halt time (``0``
    for a chain-only halt with no enclosing iterate). ``node_id`` is
    the node the halt was observed against — the iterate's node id for
    the iterate case; the not-yet-run chain step's id for the chain
    case. State is captured verbatim; verbs are expected to be
    idempotent-in-effects to survive re-run on resume, the same
    contract that already governs iterate re-run-iteration-N.
    """
    if env.runtime._halt_saved:
        return
    env.runtime._halt_saved = True
    saved = False
    trace_ref: tuple[TraceRef, ...] = ()
    stashed_ids: tuple[str, ...] = ()
    try:
        if env.checkpointer is not None and env.client_flow_id is not None:
            trace_ref, stashed_ids = await env.runtime._pending_saia_turns.stash_to_store(
                env.checkpointer, env.client_flow_id
            )
        await _save_scope_commit(env, iteration, node_id, current_state, "halted", trace_ref)
        saved = True
    except Exception as e:
        env.lg.warning(
            "halt-save failed; un-latching for retry at next observation",
            extra={"exception": e},
        )
        raise
    finally:
        if not saved:
            # Both Exception and asyncio.CancelledError land here; the flag must
            # un-latch either way so a later halt-observation site can retry.
            env.runtime._halt_saved = False
    # Drop only the entries we stashed. Late arrivals from concurrent .map
    # items that landed after the snapshot stay on the runtime dict.
    for stashed_id in stashed_ids:
        env.runtime._pending_saia_turns.remove(stashed_id)


async def _save_scope_commit(
    env: _RunEnv,
    iteration: int,
    node_id: str,
    current_state: State[Any],
    outcome: CommitOutcome,
    trace_ref: tuple[TraceRef, ...] = (),
) -> None:
    """Persist a content-addressed commit at ``node_id`` under ``env``.

    Walks the scope stack from root to ``current_state``. For each scope:
    serialize its ``data`` via the state-data contract (dict passthrough
    or ``StateData.to_dict``) to canonical JSON bytes and
    :meth:`put_object` a Blob keyed by content hash. Bundle every scope's
    blob hash into a Tree (one :class:`TreeEntry` per scope, ordered by
    depth via a two-digit ``scope_id``). Wrap the Tree in a Commit whose
    :class:`CommitMeta` pins ``(client_flow_id, node_path, iteration)``
    and the provenance triple (``produced_by``, ``trace_ref``,
    ``outcome``). Finally put_ref points this boundary at the commit
    hash.

    ``node_path`` is the ``"/"``-joined ancestor chain (from run root to
    this node, inclusive). blake2b hex has no ``"/"``, so split
    round-trips on resume.

    ``outcome`` records why the commit fired — ``"ok"`` for a successful
    iterate boundary, ``"halted"`` when the halt-observation site
    triggered the save.

    No-op when the runtime has no checkpointer / client_flow_id bound.
    """
    if env.checkpointer is None or env.client_flow_id is None:
        return
    scopes = _collect_scope_stack(current_state)
    entries = await _put_scope_blobs(env, scopes)
    tree = Tree.from_entries(entries)
    await maybe_await(
        env.checkpointer.put_object(env.client_flow_id, "tree", tree.content_hash, tree.to_bytes())
    )
    node_path = "/".join(env.ancestor_chain + (node_id,))
    meta = _build_commit_meta(env, node_path, iteration, node_id, outcome, trace_ref)
    commit = Commit.build(root_tree_hash=tree.content_hash, parent_hashes=(), meta=meta)
    await maybe_await(
        env.checkpointer.put_object(
            env.client_flow_id, "commit", commit.content_hash, commit.to_bytes()
        )
    )
    await maybe_await(
        env.checkpointer.put_ref(env.client_flow_id, node_path, iteration, commit.content_hash)
    )


def _collect_scope_stack(current: State[Any]) -> list[State[Any]]:
    """Return the ``State`` chain from run-root down to ``current``."""
    scopes: list[State[Any]] = []
    node: State[Any] | None = current
    while node is not None:
        scopes.append(node)
        node = node._parent
    scopes.reverse()
    return scopes


async def _put_scope_blobs(env: _RunEnv, scopes: list[State[Any]]) -> list[TreeEntry]:
    """Serialize each scope's data to a blob, put_object it, return tree entries.

    Uses a zero-padded two-digit index as :attr:`TreeEntry.scope_id` so
    canonical sort ordering matches root→leaf depth ordering.
    """
    entries: list[TreeEntry] = []
    for depth, scope in enumerate(scopes):
        blob_bytes = canonical_json(_serialize_state_data(scope.data))
        blob = Blob.from_bytes(blob_bytes)
        assert env.checkpointer is not None
        assert env.client_flow_id is not None
        await maybe_await(
            env.checkpointer.put_object(env.client_flow_id, "blob", blob.content_hash, blob.payload)
        )
        entries.append(
            TreeEntry(scope_id=f"{depth:02d}", kind="blob", child_hash=blob.content_hash)
        )
    return entries


def _build_commit_meta(
    env: _RunEnv,
    node_path: str,
    iteration: int,
    node_id: str,
    outcome: CommitOutcome,
    trace_ref: tuple[TraceRef, ...] = (),
) -> CommitMeta:
    """Assemble :class:`CommitMeta` for one scope-commit save.

    ``produced_by`` records the node's ``node_id`` — verb-level
    attribution (``verb_name`` / ``role`` / ``result_hash``) lands with
    the SAIA-verb-wrapper wiring. ``trace_ref`` carries cross-system
    pointers stamped by the caller — halt-save passes one
    ``TraceRef(kind="saia_turn", id=f"{node_id}:{blob_hash}")`` per
    Loop that published paused-conversation bytes, empty tuple
    otherwise.

    ``outcome`` is set by the caller: ``"ok"`` at an iterate boundary,
    ``"halted"`` at a halt-observation save.

    ``flow_root_id`` is the run's ``client_flow_id`` — a stable
    per-run identifier — until the framework computes a proper
    composition-tree root hash.
    """
    from llm_gent import __version__

    return CommitMeta(
        client_flow_id=env.client_flow_id or "",
        node_path=node_path,
        iteration=iteration,
        produced_by=ProducedBy(node_id=node_id, verb_name=None, role=None, result_hash=None),
        trace_ref=trace_ref,
        outcome=outcome,
        flow_root_id=env.client_flow_id or "",
        timestamp_iso=datetime.now(UTC).isoformat(),
        framework_version=__version__,
    )


def _serialize_state_data(data: Any) -> Any:
    """Return a JSON-compatible view of ``data`` for the checkpoint.

    Plain dicts pass through as-is (the framework does not deep-copy — the
    store implementation owns durability). Objects satisfying
    :class:`StateData` are converted via ``to_dict()``. ``None`` also
    passes through (a payload that never carried structured data).
    Anything else raises :class:`TypeError` at the save site with a
    pointer to the contract.
    """
    if data is None or isinstance(data, dict):
        return data
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(
        f"cannot checkpoint state.data of type {type(data).__name__} — "
        f"payload must be a plain dict or satisfy StateData (to_dict/from_dict)"
    )


def _target_label(target: Any) -> str:
    """Return a human-readable label for a node target."""
    from .flow import Flow

    if isinstance(target, Flow):
        return f"Flow({target.name!r})" if target.name else "Flow(<anonymous>)"
    if isinstance(target, _Branch):
        return "branch"
    if isinstance(target, _Iterate):
        return "iterate"
    if isinstance(target, _Map):
        return "map"
    return getattr(target, "__name__", type(target).__name__)


def _step_inputs(
    index: int,
    node: _Node,
    prev_result: Any,
    run_args: tuple[Any, ...],
    run_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return the (args, kwargs) to feed the node's target.

    First node: forward ``run()`` inputs verbatim. Later nodes: pass the
    previous result as the single positional (through ``project`` if set).
    Later nodes never inherit ``run()`` kwargs — those are input to the
    chain's head only.
    """
    if index == 0:
        return run_args, run_kwargs
    projected = node.project(prev_result) if node.project is not None else prev_result
    return (projected,), {}
