# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for :class:`Context` — ``ctx.lg`` and generic parameterization."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_gent.flow import Context, Flow, State, verb

from .conftest import ROLE_A, StubFactory, StubSAIA, make_test_logger


class TestCtxLg:
    """``ctx.lg`` delegates to the dispatching flow's logger."""

    @pytest.mark.asyncio
    async def test_ctx_lg_is_flow_lg(self) -> None:
        """A module-level verb reads the ambient logger via ``ctx.lg``."""
        lg = make_test_logger()
        flow = Flow(lg=lg, saia_f=StubFactory())

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
        flow = Flow(lg=lg, saia_f=StubFactory())

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
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory(), state=payload)

        @verb(role=ROLE_A)
        async def bump(ctx: Context[_Payload]) -> int:
            """Access the typed payload and mutate its ``turn`` field."""
            ctx.state.data.turn += 1
            return ctx.state.data.turn

        flow.register(bump)
        assert await flow.dispatch("bump") == 1
        assert payload.turn == 1


class TestPureVerbCtx:
    """A pure-Python verb (``@verb`` without a role) runs with ``ctx.saia is None``."""

    @pytest.mark.asyncio
    async def test_pure_verb_dispatch_has_no_saia(self) -> None:
        """Dispatch of a role-less verb builds a Context whose ``saia`` is ``None``."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        @verb
        async def tick(ctx: Context) -> object:
            """Return ``ctx.saia`` for identity inspection."""
            return ctx.saia

        flow.register(tick)
        assert (await flow.dispatch("tick")) is None

    @pytest.mark.asyncio
    async def test_pure_verb_in_chain(self) -> None:
        """A pure-Python verb runs as a chain step (no role, no ``ctx.saia`` access)."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

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
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

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
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        @verb
        async def probe(ctx: Context) -> object:
            """Return the typed saia — expected ``None`` when role is unbound."""
            return ctx.saia_as(StubSAIA)

        flow.register(probe)
        assert (await flow.dispatch("probe")) is None
