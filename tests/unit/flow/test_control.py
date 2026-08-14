"""Tests for the branch/loop/map control-flow primitives on Flow."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import Context, Failure, Flow, verb

from .conftest import ROLE_A, StubFactory, make_test_logger


# -----------------------------------------------------------------------------
# Helpers reused by many tests
# -----------------------------------------------------------------------------


@verb(role=ROLE_A)
async def _identity(ctx: Context, x: Any) -> Any:
    """Return the input unchanged."""
    return x


@verb(role=ROLE_A)
async def _double(ctx: Context, x: int) -> int:
    """Return ``x * 2``."""
    return x * 2


@verb(role=ROLE_A)
async def _plus_one(ctx: Context, x: int) -> int:
    """Return ``x + 1``."""
    return x + 1


# -----------------------------------------------------------------------------
# .branch
# -----------------------------------------------------------------------------


class TestBranchBuild:
    """.branch build-time validation."""

    def test_rejects_non_flow_non_callable_then(self) -> None:
        """A ``then`` that is neither Flow nor callable is a build-time TypeError."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        with pytest.raises(TypeError, match="branch.then"):
            flow.branch(when=lambda _p, _c: True, then=42)

    def test_rejects_non_flow_non_callable_else(self) -> None:
        """A non-Flow non-callable ``else_`` fails at build time too."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        with pytest.raises(TypeError, match="branch.else"):
            flow.branch(
                when=lambda _p, _c: True,
                then=lambda f: f.call(_identity),
                else_="not-a-flow",
            )


class TestBranchExecution:
    """.branch dispatches to the correct arm and threads results."""

    @pytest.mark.asyncio
    async def test_true_runs_then_arm(self) -> None:
        """Truthy predicate → the ``then`` subflow runs."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).branch(
            when=lambda prev, _ctx: prev > 0,
            then=lambda f: f.call(_double),
            else_=lambda f: f.call(_plus_one),
        )
        assert await flow.run(5) == 10

    @pytest.mark.asyncio
    async def test_false_runs_else_arm(self) -> None:
        """Falsy predicate → the ``else_`` subflow runs."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).branch(
            when=lambda prev, _ctx: prev > 0,
            then=lambda f: f.call(_double),
            else_=lambda f: f.call(_plus_one),
        )
        assert await flow.run(-3) == -2  # (-3)+1

    @pytest.mark.asyncio
    async def test_false_without_else_passes_prev_through(self) -> None:
        """No ``else_`` and falsy predicate → prev result flows unchanged."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).branch(
            when=lambda _p, _c: False,
            then=lambda f: f.call(_double),
        )
        assert await flow.run(7) == 7

    @pytest.mark.asyncio
    async def test_when_receives_prev_and_ctx(self) -> None:
        """Predicate sees the previous result and an ambient (state-carrying) ctx."""
        seen: dict[str, Any] = {}

        def when(prev: Any, ctx: Context) -> bool:
            """Record inputs and choose the true arm."""
            seen["prev"] = prev
            seen["state"] = ctx.state
            seen["role"] = ctx.role
            return True

        flow = Flow(make_test_logger(), factory=StubFactory(), state={"k": "v"})
        flow.call(_identity).branch(when=when, then=lambda f: f.call(_double))
        await flow.run(4)
        assert seen == {"prev": 4, "state": {"k": "v"}, "role": None}

    @pytest.mark.asyncio
    async def test_async_when_is_awaited(self) -> None:
        """An async predicate is awaited before dispatching."""

        async def when(prev: int, _ctx: Context) -> bool:
            """Async predicate."""
            await asyncio.sleep(0)
            return prev > 0

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).branch(when=when, then=lambda f: f.call(_double))
        assert await flow.run(3) == 6

    @pytest.mark.asyncio
    async def test_named_flow_as_then(self) -> None:
        """A pre-built Flow works as ``then`` interchangeably with a callback."""
        then_flow = Flow(make_test_logger(), "double").call(_double)
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).branch(when=lambda _p, _c: True, then=then_flow)
        assert await flow.run(6) == 12

    @pytest.mark.asyncio
    async def test_rescue_catches_arm_exception(self) -> None:
        """A rescue on the branch node fires when the chosen arm raises."""

        @verb(role=ROLE_A)
        async def boom(ctx: Context, _: Any) -> None:
            """Always raise."""
            raise ValueError("arm-failed")

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).branch(
            when=lambda _p, _c: True,
            then=lambda f: f.call(boom),
            rescue=lambda exc, _c: f"rescued:{exc}",
        )
        assert await flow.run(0) == "rescued:arm-failed"

    @pytest.mark.asyncio
    async def test_cancellation_propagates_through_arm(self) -> None:
        """CancelledError inside the chosen arm surfaces past the branch's rescue."""

        @verb(role=ROLE_A)
        async def cancel_me(ctx: Context, _: Any) -> None:
            """Raise CancelledError from inside the arm."""
            raise asyncio.CancelledError

        rescued = False

        def rescue(exc: BaseException, _ctx: Context) -> str:
            """Should never fire."""
            nonlocal rescued
            rescued = True
            return "should-not-see"

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).branch(
            when=lambda _p, _c: True,
            then=lambda f: f.call(cancel_me),
            rescue=rescue,
        )
        with pytest.raises(asyncio.CancelledError):
            await flow.run(0)
        assert rescued is False


# -----------------------------------------------------------------------------
# .loop
# -----------------------------------------------------------------------------


class TestLoopBuild:
    """.loop build-time validation."""

    def test_requires_until_or_max_iters(self) -> None:
        """Without ``until`` and without ``max_iters`` termination is not guaranteed."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        with pytest.raises(ValueError, match="until= or max_iters"):
            flow.loop(lambda f: f.call(_double))

    def test_max_iters_must_be_positive(self) -> None:
        """``max_iters`` of 0 or negative is a build error."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        with pytest.raises(ValueError, match="max_iters"):
            flow.loop(lambda f: f.call(_double), max_iters=0)

    def test_deadline_must_be_positive(self) -> None:
        """``deadline`` of 0 or negative is a build error."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        with pytest.raises(ValueError, match="deadline"):
            flow.loop(lambda f: f.call(_double), max_iters=3, deadline=0)


class TestLoopExecution:
    """.loop iterates the body under bounds and threads results."""

    @pytest.mark.asyncio
    async def test_max_iters_bounds_iteration(self) -> None:
        """With only ``max_iters``, the loop runs exactly N iterations."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(lambda f: f.call(_double), max_iters=3)
        assert await flow.run(1) == 8  # 1→2→4→8

    @pytest.mark.asyncio
    async def test_result_is_last_iteration(self) -> None:
        """The loop's output is the last iteration's return value."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(lambda f: f.call(_plus_one), max_iters=5)
        assert await flow.run(0) == 5

    @pytest.mark.asyncio
    async def test_until_stops_iteration_post_check(self) -> None:
        """``until`` fires AFTER an iteration completes; body runs at least once."""
        counter = {"n": 0}

        @verb(role=ROLE_A)
        async def tick(ctx: Context, _: Any) -> int:
            """Bump the shared counter and return it."""
            counter["n"] += 1
            return counter["n"]

        def until(ctx: Context) -> bool:
            """Stop when counter has reached 3."""
            return counter["n"] >= 3

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(lambda f: f.call(tick), until=until, max_iters=10)
        assert await flow.run(None) == 3
        assert counter["n"] == 3

    @pytest.mark.asyncio
    async def test_until_never_true_hits_max_iters(self) -> None:
        """``max_iters`` is a hard cap even if ``until`` never fires."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(
            lambda f: f.call(_plus_one),
            until=lambda _c: False,
            max_iters=4,
        )
        assert await flow.run(0) == 4

    @pytest.mark.asyncio
    async def test_until_sees_state_mutation(self) -> None:
        """``until`` reads ``ctx.state`` — verbs mutating state can drive termination."""

        @verb(role=ROLE_A)
        async def push(ctx: Context, item: int) -> int:
            """Append to the state list, return the input."""
            ctx.state["items"].append(item)
            return item + 1

        def until(ctx: Context) -> bool:
            """Stop after 3 items."""
            return len(ctx.state["items"]) >= 3

        flow = Flow(make_test_logger(), factory=StubFactory(), state={"items": []})
        flow.call(_identity).loop(lambda f: f.call(push), until=until, max_iters=10)
        state: dict[str, Any] = {"items": []}
        await flow.run(0, state=state)
        assert state["items"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_async_until_awaited(self) -> None:
        """An async ``until`` is awaited before continuing."""
        counter = {"n": 0}

        @verb(role=ROLE_A)
        async def tick(ctx: Context, _: Any) -> int:
            """Bump the counter."""
            counter["n"] += 1
            return counter["n"]

        async def until(_ctx: Context) -> bool:
            """Async predicate."""
            await asyncio.sleep(0)
            return counter["n"] >= 2

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(lambda f: f.call(tick), until=until, max_iters=10)
        assert await flow.run(None) == 2

    @pytest.mark.asyncio
    async def test_named_flow_as_body(self) -> None:
        """A pre-built Flow works as the loop body."""
        body = Flow(make_test_logger(), "step").call(_plus_one)
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(body, max_iters=3)
        assert await flow.run(10) == 13

    @pytest.mark.asyncio
    async def test_deadline_stops_between_iterations(self) -> None:
        """A running body is not interrupted, but no new iteration starts past deadline."""

        @verb(role=ROLE_A)
        async def slow(ctx: Context, x: int) -> int:
            """Sleep so the deadline is exceeded within a couple iterations."""
            await asyncio.sleep(0.05)
            return x + 1

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(lambda f: f.call(slow), max_iters=100, deadline=0.06)
        result = await flow.run(0)
        # ~1-2 iterations should fit in 60ms; must not run all 100.
        assert 1 <= result <= 5

    @pytest.mark.asyncio
    async def test_cancellation_in_body_propagates(self) -> None:
        """CancelledError from a loop iteration surfaces past the loop's rescue."""

        @verb(role=ROLE_A)
        async def cancel_me(ctx: Context, _: Any) -> None:
            """Cancel from inside the body."""
            raise asyncio.CancelledError

        rescued = False

        def rescue(exc: BaseException, _c: Context) -> str:
            """Should never fire."""
            nonlocal rescued
            rescued = True
            return "swallowed"

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).loop(
            lambda f: f.call(cancel_me),
            max_iters=3,
            rescue=rescue,
        )
        with pytest.raises(asyncio.CancelledError):
            await flow.run(None)
        assert rescued is False


# -----------------------------------------------------------------------------
# .map
# -----------------------------------------------------------------------------


class TestMapExecution:
    """.map fans out the body over items, preserving order."""

    @pytest.mark.asyncio
    async def test_uses_prev_result_as_items_by_default(self) -> None:
        """Omitted ``items=`` treats the previous result as the iterable."""

        @verb(role=ROLE_A)
        async def make_list(ctx: Context) -> list[int]:
            """Produce a list of ints."""
            return [1, 2, 3]

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(make_list).map(lambda f: f.call(_double))
        assert await flow.run() == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_items_callable_receives_prev_and_ctx(self) -> None:
        """``items=`` callable reshapes the previous result into the iterable."""
        seen: dict[str, Any] = {}

        def items(prev: Any, ctx: Context) -> list[int]:
            """Record inputs and produce the item list."""
            seen["prev"] = prev
            seen["state"] = ctx.state
            return list(range(prev))

        flow = Flow(make_test_logger(), factory=StubFactory(), state={"tag": "s"})
        flow.call(_identity).map(lambda f: f.call(_double), items=items)
        assert await flow.run(3) == [0, 2, 4]
        assert seen == {"prev": 3, "state": {"tag": "s"}}

    @pytest.mark.asyncio
    async def test_preserves_order_across_concurrency(self) -> None:
        """Results are ordered by input position, not completion order."""

        @verb(role=ROLE_A)
        async def delayed(ctx: Context, x: int) -> int:
            """Later items sleep less so they finish first."""
            await asyncio.sleep(0.02 - x * 0.005)
            return x

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(lambda f: f.call(delayed))
        assert await flow.run([0, 1, 2, 3]) == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_aggregate_reduces_results(self) -> None:
        """``aggregate=sum`` returns the reduced value instead of the list."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(lambda f: f.call(_double), aggregate=sum)
        assert await flow.run([1, 2, 3, 4]) == 20  # sum of doubles

    @pytest.mark.asyncio
    async def test_strict_true_propagates_first_exception(self) -> None:
        """A failing item raises out of .run when strict=True."""

        @verb(role=ROLE_A)
        async def maybe_fail(ctx: Context, x: int) -> int:
            """Raise for x==2."""
            if x == 2:
                raise ValueError(f"bad:{x}")
            return x

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(lambda f: f.call(maybe_fail))
        with pytest.raises(ValueError, match="bad:2"):
            await flow.run([1, 2, 3])

    @pytest.mark.asyncio
    async def test_strict_false_wraps_failures_in_failure_sentinel(self) -> None:
        """strict=False → each failure becomes a Failure(exception, item) in the list."""

        @verb(role=ROLE_A)
        async def maybe_fail(ctx: Context, x: int) -> int:
            """Raise for x==2."""
            if x == 2:
                raise ValueError(f"bad:{x}")
            return x * 10

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(lambda f: f.call(maybe_fail), strict=False)
        results = await flow.run([1, 2, 3])
        assert results[0] == 10
        assert isinstance(results[1], Failure)
        assert isinstance(results[1].exception, ValueError)
        assert results[1].item == 2
        assert results[2] == 30

    @pytest.mark.asyncio
    async def test_strict_false_cancellation_still_propagates(self) -> None:
        """CancelledError from an item is never wrapped in Failure."""

        @verb(role=ROLE_A)
        async def cancel_at_two(ctx: Context, x: int) -> int:
            """Cancel from item x==2."""
            if x == 2:
                raise asyncio.CancelledError
            return x

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(lambda f: f.call(cancel_at_two), strict=False)
        with pytest.raises(asyncio.CancelledError):
            await flow.run([1, 2, 3])

    @pytest.mark.asyncio
    async def test_non_iterable_items_raises(self) -> None:
        """A non-iterable resolved-items value is a clear TypeError."""
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(lambda f: f.call(_double))
        with pytest.raises(TypeError, match="iterable"):
            await flow.run(42)

    @pytest.mark.asyncio
    async def test_named_flow_as_body(self) -> None:
        """A pre-built Flow works as the map body."""
        body = Flow(make_test_logger(), "double").call(_double)
        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(body)
        assert await flow.run([2, 5]) == [4, 10]

    @pytest.mark.asyncio
    async def test_body_receives_item_as_single_positional(self) -> None:
        """Each item is passed to the body's first node as one positional arg."""

        @verb(role=ROLE_A)
        async def wrap(ctx: Context, x: int) -> dict[str, int]:
            """Return a small dict per item."""
            return {"x": x}

        flow = Flow(make_test_logger(), factory=StubFactory())
        flow.call(_identity).map(lambda f: f.call(wrap))
        assert await flow.run([7, 8]) == [{"x": 7}, {"x": 8}]


# -----------------------------------------------------------------------------
# Buildable materialization
# -----------------------------------------------------------------------------


class TestBuildable:
    """Named Flow and inline callback are interchangeable everywhere Buildable is accepted."""

    @pytest.mark.asyncio
    async def test_callback_and_flow_produce_same_result(self) -> None:
        """A branch built two ways runs identically."""
        factory = StubFactory()

        named = Flow(make_test_logger(), "then").call(_double)
        flow_named = Flow(make_test_logger(), factory=factory)
        flow_named.call(_identity).branch(when=lambda _p, _c: True, then=named)

        flow_cb = Flow(make_test_logger(), factory=factory)
        flow_cb.call(_identity).branch(when=lambda _p, _c: True, then=lambda f: f.call(_double))

        assert await flow_named.run(4) == await flow_cb.run(4) == 8
