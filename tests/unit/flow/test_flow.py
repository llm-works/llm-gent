# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for the Flow class — register + dispatch + saia caching."""

from __future__ import annotations

from typing import Any

import pytest

from llm_gent.flow import UNSET, Context, Flow, Role, Unset, verb

from .conftest import ROLE_A, ROLE_B, StubFactory, StubSAIA, make_test_logger


class TestUnsetSentinel:
    """UNSET is the public "no value" sentinel used by rescue policies."""

    def test_unset_is_singleton(self) -> None:
        """Constructing Unset() returns the module-level UNSET (identity)."""
        assert Unset() is UNSET
        assert Unset() is Unset()

    def test_unset_isinstance(self) -> None:
        """isinstance(UNSET, Unset) holds — enables type-check style guards."""
        assert isinstance(UNSET, Unset)

    def test_unset_repr(self) -> None:
        """Repr surfaces the sentinel's name for readable log output."""
        assert repr(UNSET) == "UNSET"


class TestRegistration:
    """Flow.register accepts anything with a .role attribute + a name."""

    def test_register_decorated_function(self) -> None:
        """A @verb-decorated function registers under its __name__."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        @verb(role=ROLE_A)
        async def do_thing(ctx: Context) -> None:
            """Do nothing."""

        flow.register(do_thing)
        assert flow.registered("do_thing") is True

    def test_register_with_explicit_name(self) -> None:
        """An explicit name overrides the function's __name__."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        @verb(role=ROLE_A)
        async def do_thing(ctx: Context) -> None:
            """Do nothing."""

        flow.register(do_thing, name="aliased")
        assert flow.registered("aliased") is True
        assert flow.registered("do_thing") is False

    def test_register_rejects_missing_role_attr(self) -> None:
        """Registration rejects callables without a .role attribute."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        async def not_a_verb(ctx: Context) -> None:
            """No role attribute attached."""

        with pytest.raises(TypeError, match="role"):
            flow.register(not_a_verb)

    def test_register_rejects_non_role_role_attr(self) -> None:
        """Registration rejects a .role attribute that isn't a Role."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        async def bogus(ctx: Context) -> None:
            """Has a role attr but it's the wrong type."""

        bogus.role = "just a string"  # type: ignore[attr-defined]

        with pytest.raises(TypeError, match="Role"):
            flow.register(bogus)

    def test_register_class_instance(self) -> None:
        """A class instance with .role and async __call__ registers cleanly."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        class MyVerb:
            """Test class-based verb."""

            role = ROLE_A
            __name__ = "my_verb"

            async def __call__(self, ctx: Context) -> str:
                """Return a marker."""
                return "class-verb"

        flow.register(MyVerb())
        assert flow.registered("my_verb") is True


class TestDispatch:
    """Flow.dispatch injects a Context, routes to the role's saia, awaits result."""

    @pytest.mark.asyncio
    async def test_dispatch_invokes_verb_with_context(self) -> None:
        """The verb receives a Context whose saia is bound to its role."""
        saia = StubFactory()
        flow = Flow(lg=make_test_logger(), saia_f=saia)

        @verb(role=ROLE_A)
        async def check(ctx: Context, value: int) -> tuple[Any, int]:
            """Return (saia, value) for inspection."""
            return ctx.saia, value

        flow.register(check)
        saia, value = await flow.dispatch("check", 42)

        assert isinstance(saia, StubSAIA)
        assert saia.role is ROLE_A
        assert value == 42

    @pytest.mark.asyncio
    async def test_dispatch_passes_state_through(self) -> None:
        """ctx.state exposes whatever the flow was constructed with."""
        user_state = {"counter": 0}
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory(), state=user_state)

        @verb(role=ROLE_A)
        async def bump(ctx: Context) -> int:
            """Increment the shared counter."""
            ctx.state.data["counter"] += 1
            return ctx.state.data["counter"]

        flow.register(bump)
        assert await flow.dispatch("bump") == 1
        assert await flow.dispatch("bump") == 2
        assert user_state["counter"] == 2

    @pytest.mark.asyncio
    async def test_dispatch_reports_ctx_role(self) -> None:
        """ctx.role reflects the role attached to the dispatched verb."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())

        @verb(role=ROLE_B)
        async def report(ctx: Context) -> Role:
            """Return the ctx role."""
            return ctx.role

        flow.register(report)
        assert (await flow.dispatch("report")) is ROLE_B

    @pytest.mark.asyncio
    async def test_dispatch_unknown_name_raises(self) -> None:
        """Dispatching an unregistered name raises KeyError with the name."""
        flow = Flow(lg=make_test_logger(), saia_f=StubFactory())
        with pytest.raises(KeyError, match="no verb"):
            await flow.dispatch("missing")


class TestSAIACaching:
    """The SAIAFactory is invoked at most once per Role name."""

    @pytest.mark.asyncio
    async def test_saia_built_once_per_role(self) -> None:
        """Repeated dispatch of the same role reuses the cached saia."""
        saia = StubFactory()
        flow = Flow(lg=make_test_logger(), saia_f=saia)

        @verb(role=ROLE_A)
        async def a1(ctx: Context) -> Any:
            """Return ctx.saia."""
            return ctx.saia

        @verb(role=ROLE_A)
        async def a2(ctx: Context) -> Any:
            """Return ctx.saia."""
            return ctx.saia

        flow.register(a1)
        flow.register(a2)

        s1 = await flow.dispatch("a1")
        s2 = await flow.dispatch("a2")
        s3 = await flow.dispatch("a1")

        assert s1 is s2 is s3
        assert saia.built_for == [ROLE_A]

    @pytest.mark.asyncio
    async def test_saia_built_per_distinct_role(self) -> None:
        """Distinct roles get distinct saia instances."""
        saia = StubFactory()
        flow = Flow(lg=make_test_logger(), saia_f=saia)

        @verb(role=ROLE_A)
        async def alpha(ctx: Context) -> Any:
            """Return ctx.saia."""
            return ctx.saia

        @verb(role=ROLE_B)
        async def beta(ctx: Context) -> Any:
            """Return ctx.saia."""
            return ctx.saia

        flow.register(alpha)
        flow.register(beta)

        sa = await flow.dispatch("alpha")
        sb = await flow.dispatch("beta")

        assert sa is not sb
        assert saia.built_for == [ROLE_A, ROLE_B]
