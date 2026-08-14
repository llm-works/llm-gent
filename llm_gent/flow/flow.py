"""Flow — verb registry, role-routed dispatch, and fluent composition graph.

A :class:`Flow` plays two roles that share one object:

1. **Runtime / registry.** Holds a :class:`SAIAFactory` (for turning roles
   into saia instances, cached per-role), a shared user-owned ``state``
   object, and an optional verb-by-name registry used by :meth:`dispatch`
   and by :class:`Panel`.

2. **Composition graph.** A sequence of nodes built up via the fluent
   methods :meth:`call` / :meth:`then` / :meth:`rescue` / :meth:`after` and
   executed by :meth:`run`. A node's target is either a verb (any callable
   carrying a ``.role``) or another :class:`Flow` — subflows are recursively
   run against the same runtime, so saia caching is shared across the whole
   tree.

Both roles are optional. A top-level flow that only serves as a verb
registry never needs to call the fluent methods; a subflow that only exists
to structure composition never needs a factory of its own — it borrows from
the runtime it is executed under.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger

from .context import Context
from .factory import SAIAFactory
from .role import Role


RescuePolicy = Callable[[BaseException, Context], Any]
"""Failure hook: ``(exception, ctx) -> fallback``. May be async."""

AfterHook = Callable[[Any, Context], Any]
"""Success hook: ``(result, ctx) -> None`` (return value ignored). May be async."""

ProjectFn = Callable[[Any], Any]
"""Data-flow projection: transforms the previous node's result into the next input."""


_UNSET: Any = object()
"""Sentinel used to distinguish "state kwarg omitted" from "state kwarg = None"."""


@dataclass
class _Node:
    """One step in a Flow composition chain."""

    target: Any
    project: ProjectFn | None = None
    rescue: RescuePolicy | None = None
    after: AfterHook | None = None


class Flow:
    """Verb registry + role-routed dispatch + fluent composition graph.

    A ``Flow`` used as the top-level runtime is constructed with a factory.
    Subflows used only for composition can be constructed without one — at
    :meth:`run` time they borrow the factory (and saia cache) of the flow
    that invoked them.
    """

    def __init__(
        self,
        lg: Logger,
        name: str = "",
        *,
        factory: SAIAFactory | None = None,
        state: Any = None,
    ) -> None:
        """Initialize a flow.

        Args:
            lg: Logger instance for tracing execution.
            name: Optional identifier — used in error messages and traces.
                Also lets a flow serve as a named node inside a parent chain.
            factory: Builds role-bound saia instances. Required for a
                top-level runtime. Optional for a subflow — the runtime it
                runs under supplies one.
            state: User-owned shared state object. Verbs read and (typically)
                mutate it in place. Opaque to the flow; may be overridden per
                :meth:`run` invocation.
        """
        self._lg = lg
        self._name = name
        self._factory = factory
        self._state = state
        self._verbs: dict[str, Any] = {}
        self._saia_by_role: dict[Role, Any] = {}
        self._nodes: list[_Node] = []

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The flow's identifier (empty string when unnamed)."""
        return self._name

    @property
    def state(self) -> Any:
        """The flow's default shared state object (user-owned)."""
        return self._state

    # -------------------------------------------------------------------------
    # Registration (unchanged from PR 1-3)
    # -------------------------------------------------------------------------

    def register(self, verb: Any, name: str | None = None) -> None:
        """Register a verb under a name (default: the verb's ``__name__``).

        The verb must carry a ``role`` attribute of type :class:`Role`.
        """
        if not callable(verb):
            raise TypeError(f"verb must be callable; got {type(verb).__name__}")
        if not hasattr(verb, "role"):
            raise TypeError(
                f"verb must carry a .role attribute; got {type(verb).__name__} without one"
            )
        if not isinstance(verb.role, Role):
            raise TypeError(f"verb.role must be a Role instance; got {type(verb.role).__name__}")
        resolved_name = name or getattr(verb, "__name__", None)
        if not resolved_name:
            raise TypeError("verb has no __name__ and no explicit name was provided")
        if resolved_name in self._verbs:
            raise ValueError(f"verb {resolved_name!r} already registered")
        verb._registered_name = resolved_name
        self._verbs[resolved_name] = verb

    def registered(self, name: str) -> bool:
        """Return True if a verb is registered under ``name``."""
        return name in self._verbs

    # -------------------------------------------------------------------------
    # Dispatch (unchanged from PR 1-3)
    # -------------------------------------------------------------------------

    async def dispatch(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a registered verb by name, awaiting its result.

        The verb receives a fresh :class:`Context` as its first argument,
        followed by ``*args`` / ``**kwargs`` from the caller.
        """
        if name not in self._verbs:
            raise KeyError(f"no verb registered under name {name!r}")
        verb = self._verbs[name]
        self._lg.debug("dispatching verb", extra={"verb": name, "role": verb.role.name})
        saia = self._saia_for(verb.role)
        ctx = Context(saia=saia, role=verb.role, state=self._state, flow=self)
        return await verb(ctx, *args, **kwargs)

    # -------------------------------------------------------------------------
    # Fluent composition
    # -------------------------------------------------------------------------

    def call(
        self,
        target: Any,
        *,
        project: ProjectFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
    ) -> Flow:
        """Append a node to the composition chain.

        ``target`` is a verb (any async callable carrying a ``.role``) or
        another :class:`Flow`. The first appended node receives the args
        passed to :meth:`run`; each subsequent node receives the previous
        node's result (optionally reshaped by ``project``).

        Hooks may be attached inline (kwargs) or via chained
        :meth:`rescue` / :meth:`after` calls — the two forms are equivalent.

        Returns ``self`` for chaining.
        """
        _validate_target(target)
        self._nodes.append(_Node(target=target, project=project, rescue=rescue, after=after))
        return self

    def then(
        self,
        target: Any,
        *,
        project: ProjectFn | None = None,
        rescue: RescuePolicy | None = None,
        after: AfterHook | None = None,
    ) -> Flow:
        """Append a chained node — semantic alias for :meth:`call`.

        Provided for readability: ``.call(a).then(b).then(c)`` reads as a
        pipeline. Positionally identical to :meth:`call` — data flow is
        determined by position in the chain, not by which method was used.
        """
        return self.call(target, project=project, rescue=rescue, after=after)

    def rescue(self, policy: RescuePolicy) -> Flow:
        """Attach a failure policy to the most recently appended node.

        The policy runs when the node raises anything other than
        :class:`asyncio.CancelledError` (cancellation is never rescued).
        Signature: ``(exception, ctx) -> fallback`` — may be async.
        """
        if not self._nodes:
            raise RuntimeError(".rescue() requires a preceding .call()/.then() step")
        self._nodes[-1].rescue = policy
        return self

    def after(self, hook: AfterHook) -> Flow:
        """Attach a success hook to the most recently appended node.

        The hook runs when the node returns without raising, after any
        ``rescue`` fallback has resolved. Signature: ``(result, ctx) -> None``
        — may be async. Return value ignored (side effects only).
        """
        if not self._nodes:
            raise RuntimeError(".after() requires a preceding .call()/.then() step")
        self._nodes[-1].after = hook
        return self

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def run(
        self,
        *args: Any,
        state: Any = _UNSET,
        global_state: Any = None,  # noqa: ARG002 — accepted; wired in PR 6
        _runtime: Flow | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the composition graph.

        The first node receives ``*args`` / ``**kwargs``. Each subsequent
        node receives one positional — the previous node's result, optionally
        transformed by its ``project`` callable.

        Cancellation is honored end-to-end: :class:`asyncio.CancelledError`
        raised by any node (verb or subflow) propagates out of :meth:`run`
        unchanged, past any ``rescue`` policy.

        Args:
            *args: Positional inputs to the first node.
            state: If provided, overrides ``self.state`` for this run and is
                exposed as ``ctx.state`` to every verb. Omitted → the flow's
                default state is used.
            global_state: Reserved. Accepted so callers can start passing it;
                actual wiring lands in PR 6 with the two-channel state model.
            _runtime: Internal — used when this flow runs as a subflow to
                share the outer flow's factory and saia cache. Not part of
                the public surface.
            **kwargs: Keyword inputs to the first node.

        Raises:
            RuntimeError: The flow has no nodes, or is running as a
                top-level runtime without a factory.
        """
        if not self._nodes:
            raise RuntimeError(f"Flow {self._name!r} has no nodes to run")
        runtime = _runtime if _runtime is not None else self
        if runtime._factory is None:
            label = runtime._name or "<anonymous>"
            raise RuntimeError(
                f"Flow {label!r} has no SAIAFactory — provide factory=... "
                "when constructing the top-level Flow"
            )
        active_state = self._state if state is _UNSET else state
        lg = runtime._lg
        label = self._name or "<anonymous>"
        is_subflow = _runtime is not None

        lg.debug(
            "starting flow run",
            extra={"flow": label, "nodes": len(self._nodes), "subflow": is_subflow},
        )
        result: Any = _UNSET
        for index, node in enumerate(self._nodes):
            node_args, node_kwargs = _step_inputs(index, node, result, args, kwargs)
            ctx = _build_ctx(node.target, runtime, active_state)
            result = await _execute_node(
                node, ctx, runtime, active_state, node_args, node_kwargs, lg
            )
        lg.debug("completed flow run", extra={"flow": label, "subflow": is_subflow})
        return result

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _saia_for(self, role: Role) -> Any:
        """Return a cached saia for ``role``, building it on first request."""
        cached = self._saia_by_role.get(role)
        if cached is not None:
            return cached
        if self._factory is None:
            label = self._name or "<anonymous>"
            raise RuntimeError(f"Flow {label!r} has no SAIAFactory to build saia for {role.name!r}")
        built = self._factory.build(role)
        self._saia_by_role[role] = built
        return built


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _validate_target(target: Any) -> None:
    """Reject anything that isn't a verb (callable with a Role) or a Flow."""
    if isinstance(target, Flow):
        return
    if not callable(target):
        raise TypeError(
            f"target must be a verb (callable with .role) or a Flow; got {type(target).__name__}"
        )
    if not hasattr(target, "role"):
        raise TypeError(f"verb target must carry a .role attribute; got {type(target).__name__}")
    if not isinstance(target.role, Role):
        raise TypeError(
            f"verb target .role must be a Role instance; got {type(target.role).__name__}"
        )


def _build_ctx(target: Any, runtime: Flow, state: Any) -> Context:
    """Build the Context passed to the node's verb (and to its hooks).

    Verb nodes get a role-bound ctx (role + saia populated). Subflow nodes
    get an ambient ctx — role and saia are ``None`` because the subflow-node
    itself has no single role; the subflow's inner verbs each build their
    own role-bound ctx as they run.
    """
    if isinstance(target, Flow):
        return Context(saia=None, role=None, state=state, flow=runtime)
    saia = runtime._saia_for(target.role)
    return Context(saia=saia, role=target.role, state=state, flow=runtime)


async def _execute_node(
    node: _Node,
    ctx: Context,
    runtime: Flow,
    active_state: Any,
    node_args: tuple[Any, ...],
    node_kwargs: dict[str, Any],
    lg: Logger,
) -> Any:
    """Invoke a node's target with cancellation-safe rescue + optional after hook.

    ``asyncio.CancelledError`` propagates unchanged past any rescue policy.
    Other exceptions surface unless a rescue is attached; the rescue's
    return value (awaited if awaitable) becomes the node's result.
    """
    target_name = _target_label(node.target)
    lg.debug("executing node", extra={"target": target_name})
    try:
        if isinstance(node.target, Flow):
            result = await node.target.run(
                *node_args, state=active_state, _runtime=runtime, **node_kwargs
            )
        else:
            result = await node.target(ctx, *node_args, **node_kwargs)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if node.rescue is None:
            raise
        lg.warning("rescue policy invoked", extra={"target": target_name, "exception": exc})
        fallback = node.rescue(exc, ctx)
        if inspect.isawaitable(fallback):
            fallback = await fallback
        result = fallback
    if node.after is not None:
        lg.debug("running after hook", extra={"target": target_name})
        hook_result = node.after(result, ctx)
        if inspect.isawaitable(hook_result):
            await hook_result
    return result


def _target_label(target: Any) -> str:
    """Return a human-readable label for a node target."""
    if isinstance(target, Flow):
        return f"Flow({target.name!r})" if target.name else "Flow(<anonymous>)"
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
