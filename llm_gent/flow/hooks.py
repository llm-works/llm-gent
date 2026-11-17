# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Hooks — user code a Flow calls at fixed points of a run.

A hook is an object with any of the methods named in :data:`HookName`;
:class:`Hook` is a convenient base whose methods do nothing. Flow calls
each method with one typed event: :class:`RunEvent`, :class:`StepEvent`,
:class:`LLMCallEvent` or :class:`CheckpointEvent`. All of them carry the
fields of :class:`HookEvent` — where in the run the event happened, which
invocation it belongs to, who acted, the ids that join it to SAIA's trace
and llm-infer's callbacks, the verb context, and :attr:`HookEvent.facets`,
a dict every hook on the event shares.

``before_*`` methods may return a verdict: ``True`` or ``None`` allows,
``False`` denies, a ``str`` denies with that reason, and :class:`Pause`
pauses the run. The first hook that does not allow decides; the hooks
after it are not called. A ``before_*`` hook that raises denies (fails
closed). ``after_*`` methods observe; one that raises is logged and the
run continues (fails open). Methods may be ``def`` or ``async def``.

:class:`Hooks` holds the hooks attached to one part of a run, outermost
first: ``before_*`` methods run outermost first, ``after_*`` methods
innermost first, as nested context managers would.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from appinfra.cas import Blob, canonical_json
from appinfra.log import Logger

from .checkpoint import maybe_await
from .context import Context
from .state.cas import CommitOutcome
from .state.codec import encode


if TYPE_CHECKING:
    from .flow import Flow


HookName = Literal[
    "before_run",
    "after_run",
    "before_step",
    "after_step",
    "after_llm_call",
    "after_checkpoint",
]
"""The methods a hook can have: one per point of a run Flow calls hooks at."""

BeforeName = Literal["before_run", "before_step"]
"""The hook methods whose verdict can stop what comes next."""

AfterName = Literal["after_run", "after_step", "after_llm_call", "after_checkpoint"]
"""The hook methods that observe."""

HOOK_NAMES: tuple[HookName, ...] = (
    "before_run",
    "after_run",
    "before_step",
    "after_step",
    "after_llm_call",
    "after_checkpoint",
)

StepKind = Literal["verb", "loop", "subflow", "branch", "iterate", "map"]
"""What a chain step runs."""

RunOutcome = Literal["ok", "halted", "failed", "denied"]
"""How a run ended: finished, stopped by its halt, raised, or refused by ``before_run``."""

StepOutcome = Literal["ok", "interrupted", "failed", "denied", "paused"]
"""How a step ended: finished, stopped by the halt, raised, or stopped by ``before_step``."""


@dataclass(frozen=True)
class Pause:
    """A ``before_*`` verdict: stop the run here; resuming reaches the same hook again."""

    reason: str = ""


Verdict = bool | str | Pause | None
"""A ``before_*`` result: ``True`` / ``None`` allow, ``False`` / a str deny, ``Pause`` pauses."""


@dataclass(frozen=True)
class Decision:
    """The outcome of a ``before_*`` call: allow, or which hook denied or paused and why."""

    verdict: Literal["allow", "deny", "pause"]
    reason: str = ""
    hook: str | None = None

    @property
    def allowed(self) -> bool:
        """True when every hook allowed."""
        return self.verdict == "allow"


ALLOW = Decision("allow")


class HookDenied(Exception):
    """A ``before_*`` hook denied what was about to run."""

    def __init__(self, event: HookName, decision: Decision) -> None:
        super().__init__(f"{event} denied by {decision.hook}: {decision.reason}")
        self.event = event
        self.decision = decision


# --- events ---


@dataclass(frozen=True)
class RunInfo:
    """The run an event belongs to: one :meth:`Flow.run` call."""

    run_id: str
    """Unique per :meth:`Flow.run` call."""
    flow_name: str | None
    structure_hash: str
    """The top-level flow's structure hash (``flow_root_hash`` on its checkpoints)."""
    repo: str | None
    """The run's repository name (``client_flow_id``), or ``None`` without a store."""
    resumed_from: str | None = None
    """The commit the run checked out (``resume=`` / ``restart=``), or ``None``."""


@dataclass(frozen=True)
class Where:
    """Where in the run an event happened. Fields are ``None`` / empty at run level."""

    flow: Flow | None = None
    """The flow whose chain holds the step (the top-level flow for run events)."""
    node_id: str | None = None
    """The step's structural id: the same for every run of the same flow."""
    node_path: tuple[str, ...] = ()
    """Node ids from the run's root down to the step, inclusive."""
    scope_path: str = ""
    """Snapshot path of the step's position, iterate pass and map item included."""
    kind: StepKind | None = None
    step_name: str | None = None


@dataclass(frozen=True)
class Actor:
    """Who acted: the verb, its role, and the model for LLM calls."""

    verb: str | None = None
    role: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class TraceIds:
    """Ids shared with SAIA's trace and llm-infer's callbacks, when the event has them."""

    trace_id: str | None = None
    call_id: str | None = None
    llm_request_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class HookEvent:
    """Fields every event carries."""

    name: HookName
    ts: datetime
    run: RunInfo
    where: Where = field(default_factory=Where)
    activity_id: str
    """Unique per invocation: each run, step execution or LLM call has its own."""
    parent_activity_id: str | None = None
    """The invocation this one runs inside: the enclosing step, or the run."""
    actor: Actor = field(default_factory=Actor)
    trace: TraceIds = field(default_factory=TraceIds)
    ctx: Context[Any] | None = None
    """The context of the step's verb; ``None`` for run events."""
    facets: dict[str, Any] = field(default_factory=dict)
    """Annotations shared by every hook on this event, in calling order."""
    _ids: dict[int, tuple[Any, str]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def content_id(self, value: Any) -> str:
        """Content id of ``value``: the id of the blob of its JSON form, computed once per value.

        Uses the checkpoint codec's JSON form, so a value a checkpoint can hold
        has an id here.

        Raises:
            TypeError: ``value`` has no JSON form.
        """
        cached = self._ids.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
        object_id = Blob(canonical_json(encode(value, f"{self.name} event"))).id
        # The value is kept with its id, so its id() cannot be reused meanwhile.
        self._ids[id(value)] = (value, object_id)
        return object_id


@dataclass(frozen=True, kw_only=True)
class RunEvent(HookEvent):
    """``before_run`` / ``after_run``."""

    outcome: RunOutcome | None = None
    """How the run ended; ``None`` on ``before_run``."""
    error: BaseException | None = None


@dataclass(frozen=True, kw_only=True)
class StepEvent(HookEvent):
    """``before_step`` / ``after_step``: one execution of a chain step."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: BaseException | None = None
    outcome: StepOutcome | None = None
    """How the step ended; ``None`` on ``before_step``."""
    decision: Decision | None = None
    """On ``after_step`` of a step a ``before_step`` hook stopped: that decision."""


@dataclass(frozen=True, kw_only=True)
class LLMCallEvent(HookEvent):
    """``after_llm_call``: one model call of a Loop turn."""

    conversation: Any = None
    """The conversation as sent: the context the model saw, before its response."""
    response: Any = None


@dataclass(frozen=True, kw_only=True)
class CheckpointEvent(HookEvent):
    """``after_checkpoint``: a checkpoint commit was written."""

    commit_id: str
    outcome: CommitOutcome


# --- hooks ---


class Hook:
    """Base for hooks: every method does nothing. Override the ones needed.

    Subclassing is optional; any object with some of these methods is a
    hook. A method left as inherited from this class is never called.
    """

    def before_run(self, event: RunEvent) -> Verdict | Awaitable[Verdict]:
        """Before the run starts; a verdict other than allow stops it before any step."""
        return None

    def after_run(self, event: RunEvent) -> None | Awaitable[None]:
        """After the run ended, however it ended."""

    def before_step(self, event: StepEvent) -> Verdict | Awaitable[Verdict]:
        """Before a chain step runs; a verdict other than allow keeps it from running."""
        return None

    def after_step(self, event: StepEvent) -> None | Awaitable[None]:
        """After a chain step ended, however it ended (including denied or paused)."""

    def after_llm_call(self, event: LLMCallEvent) -> None | Awaitable[None]:
        """After a model call of a Loop turn returned."""

    def after_checkpoint(self, event: CheckpointEvent) -> None | Awaitable[None]:
        """After a checkpoint commit was written."""

    def child(self, where: Where) -> Hook:
        """The hook to use inside the subflow or map item at ``where``; this one by default."""
        return self


HookT = TypeVar("HookT")


def hook_label(hook: object) -> str:
    """How a hook is named in logs and decisions: its class name."""
    return type(hook).__name__


def _method(hook: object, name: HookName) -> Any:
    """The bound method ``name`` of ``hook``, or ``None`` when absent or :class:`Hook`'s no-op."""
    found = getattr(hook, name, None)
    if found is None or getattr(found, "__func__", None) is getattr(Hook, name):
        return None
    return found


def _child(hook: object, where: Where) -> object:
    """``hook.child(where)`` when the hook has a ``child`` method, else the hook itself."""
    child = getattr(hook, "child", None)
    return child(where) if callable(child) else hook


def _decision(verdict: Any, hook: str) -> Decision:
    """Turn a ``before_*`` return value into a :class:`Decision`."""
    if verdict is True or verdict is None:
        return ALLOW
    if verdict is False:
        return Decision("deny", f"{hook} returned False", hook)
    if isinstance(verdict, Pause):
        return Decision("pause", verdict.reason, hook)
    if isinstance(verdict, str):
        return Decision("deny", verdict, hook)
    raise TypeError(f"{hook} returned {verdict!r}; expected True, None, False, a str or Pause")


class Hooks:
    """The hooks attached to one part of a run, outermost first, with each method resolved once."""

    def __init__(self, lg: Logger, hooks: Iterable[object] = ()) -> None:
        self._lg = lg
        self.hooks: tuple[object, ...] = tuple(hooks)
        self._methods: dict[HookName, tuple[tuple[str, Any], ...]] = {
            name: tuple(
                (hook_label(h), m) for h in self.hooks if (m := _method(h, name)) is not None
            )
            for name in HOOK_NAMES
        }

    def __bool__(self) -> bool:
        """True when any hook is attached."""
        return bool(self.hooks)

    def has(self, name: HookName) -> bool:
        """True when some hook implements ``name``: lets a caller skip building the event."""
        return bool(self._methods[name])

    def extend(self, hooks: Iterable[object], *, replace: bool = False) -> Hooks:
        """These hooks with ``hooks`` attached inside them; only ``hooks`` with ``replace``."""
        inner = tuple(hooks)
        return Hooks(self._lg, inner if replace else (*self.hooks, *inner))

    def child(self, where: Where) -> Hooks:
        """The hooks for the subflow or map item at ``where``: each hook's :meth:`Hook.child`."""
        return Hooks(self._lg, (_child(h, where) for h in self.hooks))

    def find(self, cls: type[HookT]) -> HookT | None:
        """The innermost attached hook that is a ``cls``, or ``None``."""
        for hook in reversed(self.hooks):
            if isinstance(hook, cls):
                return hook
        return None

    async def before(self, name: BeforeName, event: HookEvent) -> Decision:
        """Call ``name`` outermost first; the first verdict other than allow decides.

        A hook that raises denies, with the exception as its reason.
        """
        for label, method in self._methods[name]:
            try:
                decision = _decision(await maybe_await(method(event)), label)
            except Exception as e:
                self._lg.warning(
                    "hook failed", extra={"hook": label, "event": name, "exception": e}
                )
                decision = Decision("deny", f"{label} raised {type(e).__name__}: {e}", label)
            if not decision.allowed:
                return decision
        return ALLOW

    async def after(self, name: AfterName, event: HookEvent) -> None:
        """Call ``name`` innermost first; a hook that raises is logged and the rest still run."""
        for label, method in reversed(self._methods[name]):
            try:
                await maybe_await(method(event))
            except Exception as e:
                self._lg.warning(
                    "hook failed", extra={"hook": label, "event": name, "exception": e}
                )
