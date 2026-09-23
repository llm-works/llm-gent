# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for :class:`Context` — ``ctx.lg`` and generic parameterization."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llm_gent.flow import Context, Flow, Panel, State, verb
from llm_gent.flow.stores import JsonFileCheckpointStore

from .conftest import ROLE_A, StubFactory, StubSAIA, make_test_logger


class TestCtxLg:
    """``ctx.lg`` delegates to the dispatching flow's logger."""

    @pytest.mark.asyncio
    async def test_ctx_lg_is_flow_lg(self) -> None:
        """A module-level verb reads the ambient logger via ``ctx.lg``."""
        lg = make_test_logger()
        flow = Flow(lg=lg, saia_factory=StubFactory())

        @verb(role=ROLE_A)
        async def read_lg(ctx: Context) -> object:
            """Return the logger reached via ``ctx.lg``."""
            return ctx.lg

        flow.register(read_lg)
        assert (await flow.dispatch("read_lg")) is lg

    @pytest.mark.asyncio
    async def test_ctx_lg_survives_dispatch(self) -> None:
        """Repeated dispatches surface the same logger reference."""
        lg = make_test_logger()
        flow = Flow(lg=lg, saia_factory=StubFactory())

        @verb(role=ROLE_A)
        async def read_lg(ctx: Context) -> object:
            """Return the logger for identity checks across dispatches."""
            return ctx.lg

        flow.register(read_lg)
        first = await flow.dispatch("read_lg")
        second = await flow.dispatch("read_lg")
        assert first is second is lg


@dataclass
class _Payload:
    """Sample dataclass payload for generic-parameterization tests."""

    turn: int = 0
    findings: list[str] | None = None


class TestContextGeneric:
    """``Context[T]`` and ``State[T]`` accept parameterization at runtime."""

    def test_state_generic_construction(self) -> None:
        """``State[_Payload]`` constructs with a matching payload."""
        payload = _Payload(turn=3, findings=["a", "b"])
        state: State[_Payload] = State(data=payload)
        assert state.data is payload
        assert state.data.turn == 3

    def test_state_unparameterized_still_works(self) -> None:
        """Bare ``State`` remains valid — the payload types as :data:`Any`."""
        state = State(data={"any": 1})
        assert state.data == {"any": 1}

    @pytest.mark.asyncio
    async def test_context_generic_dispatch(self) -> None:
        """A verb annotated as ``Context[_Payload]`` runs without error."""
        payload = _Payload(turn=0)
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory(), state=payload)

        @verb(role=ROLE_A)
        async def bump(ctx: Context[_Payload]) -> int:
            """Access the typed payload and mutate its ``turn`` field."""
            ctx.state.data.turn += 1
            return ctx.state.data.turn

        flow.register(bump)
        assert await flow.dispatch("bump") == 1
        assert payload.turn == 1


class TestCtxData:
    """``ctx.data`` is a shortcut alias for ``ctx.state.data`` typed as :data:`T`."""

    @pytest.mark.asyncio
    async def test_ctx_data_returns_payload(self) -> None:
        """``ctx.data`` returns the same object as ``ctx.state.data``."""
        payload = _Payload(turn=7)
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory(), state=payload)

        @verb(role=ROLE_A)
        async def probe(ctx: Context[_Payload]) -> tuple[object, object]:
            """Return (state.data, data) for identity comparison."""
            return ctx.state.data, ctx.data

        flow.register(probe)
        via_state, via_alias = await flow.dispatch("probe")
        assert via_state is via_alias is payload

    @pytest.mark.asyncio
    async def test_ctx_data_mutation_is_visible(self) -> None:
        """Mutating through ``ctx.data`` is visible on ``ctx.state.data``."""
        payload = _Payload(turn=0)
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory(), state=payload)

        @verb(role=ROLE_A)
        async def bump(ctx: Context[_Payload]) -> int:
            """Mutate via the shortcut; the underlying payload advances."""
            ctx.data.turn += 1
            return ctx.state.data.turn

        flow.register(bump)
        assert await flow.dispatch("bump") == 1
        assert payload.turn == 1

    def test_ctx_data_reaches_scope_not_root(self) -> None:
        """``ctx.data`` is the local scope's payload — not the root's.

        Pins the documented behavior: the shortcut aliases
        ``ctx.state.data``, not ``ctx.state.root().data``. A subflow's
        verb sees the child scope's payload; run-wide state stays behind
        the explicit ``ctx.state.root().data`` traversal.
        """
        root_payload = {"root": True}
        child_payload = _Payload(turn=99)
        root_state: State[dict[str, bool]] = State(data=root_payload)
        child_state: State[_Payload] = State(data=child_payload, _parent=root_state)
        ctx: Context[_Payload] = Context(role=None, state=child_state, flow=None)
        assert ctx.data is child_payload
        assert ctx.state.root().data is root_payload


class TestCtxExtra:
    """``ctx.extra`` surfaces caller-supplied ``Flow.run(extra=)`` at the verb."""

    @pytest.mark.asyncio
    async def test_ctx_extra_reaches_verb(self) -> None:
        """A verb reads ``ctx.extra`` values supplied at ``Flow.run(extra=)``."""
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        sentinel = object()

        @verb
        async def probe(ctx: Context, n: int) -> tuple[object, int]:
            """Return the extra sentinel + input for identity assertion."""
            return ctx.extra["handle"], n

        flow.call(probe)
        got, n = await flow.run(4, extra={"handle": sentinel})
        assert got is sentinel
        assert n == 4

    @pytest.mark.asyncio
    async def test_ctx_extra_defaults_to_empty(self) -> None:
        """Omitting ``extra=`` yields an empty ``ctx.extra`` dict."""
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())

        @verb
        async def probe(ctx: Context) -> dict[str, Any]:
            """Return ``ctx.extra`` for shape assertion."""
            return ctx.extra

        flow.call(probe)
        got = await flow.run()
        assert got == {}

    @pytest.mark.asyncio
    async def test_ctx_extra_identity_preserved(self) -> None:
        """The exact dict handed to ``Flow.run(extra=)`` reaches the verb."""
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        supplied: dict[str, Any] = {"k": 1}

        @verb
        async def probe(ctx: Context) -> dict[str, Any]:
            """Return ``ctx.extra`` for identity assertion."""
            return ctx.extra

        flow.call(probe)
        got = await flow.run(extra=supplied)
        assert got is supplied


class TestCtxExtraPropagation:
    """``ctx.extra`` reaches every dispatch site via env.extra threading."""

    @pytest.mark.asyncio
    async def test_extra_reaches_subflow_via_call(self) -> None:
        """A ``.call(subflow)`` descent surfaces the same ``ctx.extra`` at the inner verb."""
        sentinel = object()

        @verb
        async def inner(ctx: Context) -> object:
            """Return the extra handle for identity assertion."""
            return ctx.extra["h"]

        subflow = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        subflow.call(inner)

        outer = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        outer.call(subflow)
        got = await outer.run(extra={"h": sentinel})
        assert got is sentinel

    @pytest.mark.asyncio
    async def test_extra_reaches_branch_arm(self) -> None:
        """A ``.branch()`` descent into the ``then`` arm surfaces ``ctx.extra``."""
        sentinel = object()

        @verb
        async def arm(ctx: Context, _prev: object) -> object:
            """Return the extra handle for identity assertion."""
            return ctx.extra["h"]

        then_flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        then_flow.call(arm)

        outer = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        outer.branch(when=lambda _p, _c: True, then=then_flow)
        got = await outer.run(0, extra={"h": sentinel})
        assert got is sentinel

    @pytest.mark.asyncio
    async def test_extra_reaches_iterate_body(self) -> None:
        """A ``.iterate()`` body dispatch surfaces ``ctx.extra`` on every pass."""
        seen: list[object] = []
        sentinel = object()

        @verb
        async def body_step(ctx: Context, prev: int) -> int:
            """Capture ``ctx.extra["h"]`` and increment the counter."""
            seen.append(ctx.extra["h"])
            return prev + 1

        body = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        body.call(body_step)

        outer = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        outer.iterate(body, max_iters=3)
        await outer.run(0, extra={"h": sentinel})
        assert seen == [sentinel, sentinel, sentinel]

    @pytest.mark.asyncio
    async def test_extra_reaches_map_item(self) -> None:
        """A ``.map()`` per-item descent surfaces ``ctx.extra`` at the item verb."""
        sentinel = object()

        @verb
        async def per_item(ctx: Context, item: int) -> tuple[int, object]:
            """Pair the item with the extra handle for identity assertion."""
            return item, ctx.extra["h"]

        body = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        body.call(per_item)

        outer = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        outer.map(body, items=lambda _p, _c: [1, 2, 3])
        got = await outer.run(extra={"h": sentinel})
        assert got == [(1, sentinel), (2, sentinel), (3, sentinel)]

    @pytest.mark.asyncio
    async def test_extra_forwarded_through_panel(self) -> None:
        """``Panel`` forwards ``ctx.extra`` to each inner verb's dispatch."""
        sentinel = object()

        @verb
        async def pane_a(ctx: Context) -> object:
            """Return the extra handle observed inside pane a."""
            return ctx.extra["h"]

        @verb
        async def pane_b(ctx: Context) -> object:
            """Return the extra handle observed inside pane b."""
            return ctx.extra["h"]

        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())
        flow.register(pane_a)
        flow.register(pane_b)
        panel = Panel([pane_a, pane_b], aggregate=list)

        @verb
        async def outer(ctx: Context) -> list[object]:
            """Run the panel and return its aggregate."""
            return await panel.run(ctx)

        flow.register(outer)
        flow.call(outer)
        got = await flow.run(extra={"h": sentinel})
        assert got == [sentinel, sentinel]


class TestCtxExtraResume:
    """``ctx.extra`` is not persisted and is re-supplied by the caller on resume."""

    @pytest.mark.asyncio
    async def test_extra_not_checkpointed_and_re_supplied_on_resume(self, tmp_path: Path) -> None:
        """Non-picklable extra survives halt+resume; verb sees the fresh dict post-resume.

        Places a :class:`threading.Lock` (non-picklable) in ``extra`` and
        halts partway through an ``.iterate`` body. If ``extra`` had been
        serialized into a checkpoint blob, the run would raise. Resume
        supplies a new Lock; the body verb observes the fresh instance,
        proving the caller's re-supplied dict wins on the resume path.
        """
        store = JsonFileCheckpointStore(make_test_logger(), tmp_path / "cp")
        seen: list[object] = []
        halt = asyncio.Event()

        @verb
        async def body_step(ctx: Context[dict[str, int]], _prev: Any = None) -> int:
            """Capture the extra handle; halt after two iterations."""
            seen.append(ctx.extra["lock"])
            ctx.state.data["counter"] += 1
            if ctx.state.data["counter"] >= 2:
                halt.set()
            return ctx.state.data["counter"]

        def build(halt_event: asyncio.Event | None) -> Flow:
            """Build the outer flow bound to the shared store + optional halt."""
            body = Flow(lg=make_test_logger(), saia_factory=StubFactory())
            body.call(body_step)
            outer = Flow(
                lg=make_test_logger(),
                saia_factory=StubFactory(),
                state={"counter": 0},
            )
            outer.iterate(body, max_iters=5)
            outer = outer.with_checkpointer(store, "extra-resume-1")
            return outer.with_halt(halt_event) if halt_event is not None else outer

        pre_lock = threading.Lock()
        await build(halt).run(extra={"lock": pre_lock})
        # Pre-halt: 2 iterations, both observed the pre-halt lock.
        assert seen == [pre_lock, pre_lock]

        # Resume with a fresh, distinct lock — must survive replay + reach
        # the post-halt iterations by identity.
        resume_lock = threading.Lock()
        assert resume_lock is not pre_lock
        await build(halt_event=None).run(resume=True, extra={"lock": resume_lock})
        # Every post-resume iteration observed the resume lock, never the
        # pre-halt one (which would prove extra leaked through a blob).
        assert len(seen) == 5, f"expected 5 iterations total, got {len(seen)}"
        assert all(x is resume_lock for x in seen[2:])
        assert pre_lock not in seen[2:]


class TestPureVerbCtx:
    """A pure-Python verb (``@verb`` without a role) runs with ``ctx.saia is None``."""

    @pytest.mark.asyncio
    async def test_pure_verb_dispatch_has_no_saia(self) -> None:
        """Dispatch of a role-less verb builds a Context whose ``saia`` is ``None``."""
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())

        @verb
        async def tick(ctx: Context) -> object:
            """Return ``ctx.saia`` for identity inspection."""
            return ctx.saia

        flow.register(tick)
        assert (await flow.dispatch("tick")) is None

    @pytest.mark.asyncio
    async def test_pure_verb_in_chain(self) -> None:
        """A pure-Python verb runs as a chain step (no role, no ``ctx.saia`` access)."""
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())

        @verb
        async def head(ctx: Context, n: int) -> int:
            """Return the input verbatim; no LLM."""
            return n + 1

        flow.call(head)
        assert await flow.run(4) == 5


class TestSaiaAs:
    """``ctx.saia_as(cls)`` is a typing helper — runtime returns ``ctx.saia``."""

    @pytest.mark.asyncio
    async def test_saia_as_returns_saia(self) -> None:
        """The helper returns the same object as ``ctx.saia`` when a role is bound."""
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())

        @verb(role=ROLE_A)
        async def probe(ctx: Context) -> tuple[object, object]:
            """Return (saia, saia_as) for identity comparison."""
            return ctx.saia, ctx.saia_as(StubSAIA)

        flow.register(probe)
        s, s_as = await flow.dispatch("probe")
        assert s is s_as
        assert isinstance(s, StubSAIA)

    @pytest.mark.asyncio
    async def test_saia_as_on_roleless_returns_none(self) -> None:
        """Without a role, ``ctx.saia_as(cls)`` returns ``None`` (no saia to bind)."""
        flow = Flow(lg=make_test_logger(), saia_factory=StubFactory())

        @verb
        async def probe(ctx: Context) -> object:
            """Return the typed saia — expected ``None`` when role is unbound."""
            return ctx.saia_as(StubSAIA)

        flow.register(probe)
        assert (await flow.dispatch("probe")) is None
