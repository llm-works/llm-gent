"""Tests for Panel + aggregation helpers."""

from __future__ import annotations

import pytest

from llm_gent.flow import Context, Panel, Role, verb
from llm_gent.flow.panel import majority, mean, unanimous, weighted

from .conftest import ROLE_A, ROLE_B, make_ff


class TestAggregators:
    """Aggregation helpers exposed by the panel module."""

    def test_majority_returns_most_common(self) -> None:
        """majority picks the most frequently occurring value."""
        assert majority(["a", "b", "a", "c", "a"]) == "a"

    def test_majority_tie_first_seen_wins(self) -> None:
        """On a tie, the first-inserted value wins (Counter semantics)."""
        assert majority(["a", "b", "a", "b"]) == "a"

    def test_majority_empty_raises(self) -> None:
        """majority on an empty list is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            majority([])

    def test_unanimous_all_agree(self) -> None:
        """unanimous returns the shared value when every vote matches."""
        assert unanimous(["yes", "yes", "yes"]) == "yes"

    def test_unanimous_disagree_returns_none(self) -> None:
        """unanimous returns None when any vote diverges."""
        assert unanimous(["yes", "no", "yes"]) is None

    def test_unanimous_empty_returns_none(self) -> None:
        """unanimous on an empty list returns None (no value to agree on)."""
        assert unanimous([]) is None

    def test_mean(self) -> None:
        """mean returns the arithmetic mean of the votes."""
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty(self) -> None:
        """mean on an empty list returns 0.0 (avoids ZeroDivisionError)."""
        assert mean([]) == 0.0

    def test_weighted_normal(self) -> None:
        """weighted returns a properly weight-normalized average."""
        assert weighted([(1.0, 0.5), (3.0, 0.5)]) == pytest.approx(2.0)

    def test_weighted_uneven_weights(self) -> None:
        """weighted respects unequal weights."""
        assert weighted([(1.0, 0.25), (5.0, 0.75)]) == pytest.approx(4.0)

    def test_weighted_zero_weight_returns_zero(self) -> None:
        """weighted with total weight of 0 returns 0.0 (safe on empty / all-zero)."""
        assert weighted([]) == 0.0
        assert weighted([(1.0, 0.0), (5.0, 0.0)]) == 0.0


class TestPanel:
    """Panel fans out via ctx.flow and aggregates results."""

    def test_empty_verbs_raises(self) -> None:
        """Constructing a Panel with no verbs is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            Panel(verbs=[], aggregate=majority)

    @pytest.mark.asyncio
    async def test_panel_fans_out_and_sums(self) -> None:
        """Each verb runs in parallel and its result feeds the aggregate."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def add_one(ctx: Context, x: int) -> int:
            """Return x + 1."""
            return x + 1

        @verb(role=ROLE_A)
        async def add_two(ctx: Context, x: int) -> int:
            """Return x + 2."""
            return x + 2

        flow.register(add_one)
        flow.register(add_two)

        panel = Panel([add_one, add_two], aggregate=sum)

        @verb(role=ROLE_A)
        async def outer(ctx: Context, x: int) -> int:
            """Run the panel and return its aggregate."""
            return await panel.run(ctx, x)

        flow.register(outer)
        result = await flow.dispatch("outer", 10)
        # add_one(10)=11, add_two(10)=12, sum=23
        assert result == 23

    @pytest.mark.asyncio
    async def test_panel_with_custom_registered_names(self) -> None:
        """Panel dispatches correctly when verbs are registered with custom names."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def impl_a(ctx: Context, x: int) -> int:
            """Return x * 2."""
            return x * 2

        @verb(role=ROLE_A)
        async def impl_b(ctx: Context, x: int) -> int:
            """Return x * 3."""
            return x * 3

        # Register with custom names different from __name__
        flow.register(impl_a, name="custom_doubler")
        flow.register(impl_b, name="custom_tripler")

        panel = Panel([impl_a, impl_b], aggregate=sum)

        @verb(role=ROLE_A)
        async def outer(ctx: Context, x: int) -> int:
            """Run the panel."""
            return await panel.run(ctx, x)

        flow.register(outer)
        result = await flow.dispatch("outer", 5)
        # impl_a(5)=10, impl_b(5)=15, sum=25
        assert result == 25

    @pytest.mark.asyncio
    async def test_panel_with_majority(self) -> None:
        """A 3-judge panel returning majority verdict works end-to-end."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def yes_a(ctx: Context) -> str:
            """Vote yes."""
            return "yes"

        @verb(role=ROLE_A)
        async def yes_b(ctx: Context) -> str:
            """Vote yes."""
            return "yes"

        @verb(role=ROLE_A)
        async def no_one(ctx: Context) -> str:
            """Vote no."""
            return "no"

        for v in (yes_a, yes_b, no_one):
            flow.register(v)

        panel = Panel([yes_a, yes_b, no_one], aggregate=majority)

        @verb(role=ROLE_A)
        async def outer(ctx: Context) -> str:
            """Run the panel."""
            return await panel.run(ctx)

        flow.register(outer)
        assert await flow.dispatch("outer") == "yes"

    @pytest.mark.asyncio
    async def test_panel_routes_per_verb_role(self) -> None:
        """Each inner verb receives a saia bound to its own role, not the caller's."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def see_a(ctx: Context) -> Role:
            """Return the ctx role."""
            return ctx.role

        @verb(role=ROLE_B)
        async def see_b(ctx: Context) -> Role:
            """Return the ctx role."""
            return ctx.role

        flow.register(see_a)
        flow.register(see_b)

        panel = Panel([see_a, see_b], aggregate=list)

        @verb(role=ROLE_A)
        async def outer(ctx: Context) -> list[Role]:
            """Run the panel and return the two roles each verb saw."""
            return await panel.run(ctx)

        flow.register(outer)
        roles = await flow.dispatch("outer")
        assert set(roles) == {ROLE_A, ROLE_B}
