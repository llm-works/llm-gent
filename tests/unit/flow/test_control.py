"""Tests for the branch/loop/map control-flow primitives on Flow."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import Context, Failure, Flow, Skipped, verb

from .conftest import ROLE_A, StubFactory, make_ff


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
        flow = make_ff().create()
        with pytest.raises(TypeError, match="branch.then"):
            flow.branch(when=lambda _p, _c: True, then=42)

    def test_rejects_non_flow_non_callable_else(self) -> None:
        """A non-Flow non-callable ``else_`` fails at build time too."""
        flow = make_ff().create()
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
        flow = make_ff().create()
        flow.call(_identity).branch(
            when=lambda prev, _ctx: prev > 0,
            then=lambda f: f.call(_double),
            else_=lambda f: f.call(_plus_one),
        )
        assert await flow.run(5) == 10

    @pytest.mark.asyncio
    async def test_false_runs_else_arm(self) -> None:
        """Falsy predicate → the ``else_`` subflow runs."""
        flow = make_ff().create()
        flow.call(_identity).branch(
            when=lambda prev, _ctx: prev > 0,
            then=lambda f: f.call(_double),
            else_=lambda f: f.call(_plus_one),
        )
        assert await flow.run(-3) == -2  # (-3)+1

    @pytest.mark.asyncio
    async def test_false_without_else_passes_prev_through(self) -> None:
        """No ``else_`` and falsy predicate → prev result flows unchanged."""
        flow = make_ff().create()
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
            seen["state"] = ctx.state.data
            seen["role"] = ctx.role
            return True

        flow = make_ff().create(state={"k": "v"})
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

        flow = make_ff().create()
        flow.call(_identity).branch(when=when, then=lambda f: f.call(_double))
        assert await flow.run(3) == 6

    @pytest.mark.asyncio
    async def test_named_flow_as_then(self) -> None:
        """A pre-built Flow works as ``then`` interchangeably with a callback."""
        then_flow = make_ff().create("double").call(_double)
        flow = make_ff().create()
        flow.call(_identity).branch(when=lambda _p, _c: True, then=then_flow)
        assert await flow.run(6) == 12

    @pytest.mark.asyncio
    async def test_bare_verb_as_then_and_else(self) -> None:
        """A bare ``@verb`` is auto-wrapped in either arm — same path as .map."""
        flow = make_ff().create()
        flow.call(_identity).branch(
            when=lambda prev, _c: prev > 0,
            then=_double,
            else_=_plus_one,
        )
        assert await flow.run(3) == 6
        flow2 = make_ff().create()
        flow2.call(_identity).branch(
            when=lambda prev, _c: prev > 0,
            then=_double,
            else_=_plus_one,
        )
        assert await flow2.run(-1) == 0

    @pytest.mark.asyncio
    async def test_rescue_catches_arm_exception(self) -> None:
        """A rescue on the branch node fires when the chosen arm raises."""

        @verb(role=ROLE_A)
        async def boom(ctx: Context, _: Any) -> None:
            """Always raise."""
            raise ValueError("arm-failed")

        flow = make_ff().create()
        flow.call(_identity).branch(
            when=lambda _p, _c: True,
            then=lambda f: f.call(boom),
            rescue=lambda exc, _prev, _c: f"rescued:{exc}",
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

        def rescue(exc: BaseException, _prev: Any, _ctx: Context) -> str:
            """Should never fire."""
            nonlocal rescued
            rescued = True
            return "should-not-see"

        flow = make_ff().create()
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
        flow = make_ff().create()
        with pytest.raises(ValueError, match="until= or max_iters"):
            flow.loop(lambda f: f.call(_double))

    def test_max_iters_must_be_positive(self) -> None:
        """``max_iters`` of 0 or negative is a build error."""
        flow = make_ff().create()
        with pytest.raises(ValueError, match="max_iters"):
            flow.loop(lambda f: f.call(_double), max_iters=0)

    def test_deadline_must_be_positive(self) -> None:
        """``deadline`` of 0 or negative is a build error."""
        flow = make_ff().create()
        with pytest.raises(ValueError, match="deadline"):
            flow.loop(lambda f: f.call(_double), max_iters=3, deadline=0)


class TestLoopExecution:
    """.loop iterates the body under bounds and threads results."""

    @pytest.mark.asyncio
    async def test_max_iters_bounds_iteration(self) -> None:
        """With only ``max_iters``, the loop runs exactly N iterations."""
        flow = make_ff().create()
        flow.call(_identity).loop(lambda f: f.call(_double), max_iters=3)
        assert await flow.run(1) == 8  # 1→2→4→8

    @pytest.mark.asyncio
    async def test_result_is_last_iteration(self) -> None:
        """The loop's output is the last iteration's return value."""
        flow = make_ff().create()
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

        flow = make_ff().create()
        flow.call(_identity).loop(lambda f: f.call(tick), until=until, max_iters=10)
        assert await flow.run(None) == 3
        assert counter["n"] == 3

    @pytest.mark.asyncio
    async def test_until_never_true_hits_max_iters(self) -> None:
        """``max_iters`` is a hard cap even if ``until`` never fires."""
        flow = make_ff().create()
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
            ctx.state.data["items"].append(item)
            return item + 1

        def until(ctx: Context) -> bool:
            """Stop after 3 items."""
            return len(ctx.state.data["items"]) >= 3

        flow = make_ff().create(state={"items": []})
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

        flow = make_ff().create()
        flow.call(_identity).loop(lambda f: f.call(tick), until=until, max_iters=10)
        assert await flow.run(None) == 2

    @pytest.mark.asyncio
    async def test_bare_verb_as_body(self) -> None:
        """A bare ``@verb`` is auto-wrapped as the loop body."""
        flow = make_ff().create()
        flow.call(_identity).loop(_plus_one, max_iters=4)
        assert await flow.run(0) == 4

    @pytest.mark.asyncio
    async def test_named_flow_as_body(self) -> None:
        """A pre-built Flow works as the loop body."""
        body = make_ff().create("step").call(_plus_one)
        flow = make_ff().create()
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

        flow = make_ff().create()
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

        def rescue(exc: BaseException, _prev: Any, _c: Context) -> str:
            """Should never fire."""
            nonlocal rescued
            rescued = True
            return "swallowed"

        flow = make_ff().create()
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

        flow = make_ff().create()
        flow.call(make_list).map(lambda f: f.call(_double))
        assert await flow.run() == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_items_callable_receives_prev_and_ctx(self) -> None:
        """``items=`` callable reshapes the previous result into the iterable."""
        seen: dict[str, Any] = {}

        def items(prev: Any, ctx: Context) -> list[int]:
            """Record inputs and produce the item list."""
            seen["prev"] = prev
            seen["state"] = ctx.state.data
            return list(range(prev))

        flow = make_ff().create(state={"tag": "s"})
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

        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(delayed))
        assert await flow.run([0, 1, 2, 3]) == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_aggregate_reduces_results(self) -> None:
        """``aggregate=sum`` returns the reduced value instead of the list."""
        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(_double), aggregate=sum)
        assert await flow.run([1, 2, 3, 4]) == 20  # sum of doubles

    @pytest.mark.asyncio
    async def test_async_aggregate_is_awaited(self) -> None:
        """An async aggregate callback is awaited before the map returns."""

        async def total(results: list[int]) -> int:
            await asyncio.sleep(0)
            return sum(results)

        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(_double), aggregate=total)
        assert await flow.run([1, 2, 3]) == 12

    @pytest.mark.asyncio
    async def test_async_items_is_awaited(self) -> None:
        """An async items callback is awaited before fan-out."""

        async def items(prev: int, _ctx: Context) -> list[int]:
            await asyncio.sleep(0)
            return list(range(prev))

        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(_double), items=items)
        assert await flow.run(3) == [0, 2, 4]

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_list(self) -> None:
        """An empty item source produces an empty result list."""
        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(_double))
        assert await flow.run([]) == []

    @pytest.mark.asyncio
    async def test_strict_true_propagates_first_exception(self) -> None:
        """A failing item raises out of .run when strict=True."""

        @verb(role=ROLE_A)
        async def maybe_fail(ctx: Context, x: int) -> int:
            """Raise for x==2."""
            if x == 2:
                raise ValueError(f"bad:{x}")
            return x

        flow = make_ff().create()
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

        flow = make_ff().create()
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

        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(cancel_at_two), strict=False)
        with pytest.raises(asyncio.CancelledError):
            await flow.run([1, 2, 3])

    @pytest.mark.asyncio
    async def test_non_iterable_items_raises(self) -> None:
        """A non-iterable resolved-items value is a clear TypeError."""
        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(_double))
        with pytest.raises(TypeError, match="iterable"):
            await flow.run(42)

    @pytest.mark.asyncio
    async def test_named_flow_as_body(self) -> None:
        """A pre-built Flow works as the map body."""
        body = make_ff().create("double").call(_double)
        flow = make_ff().create()
        flow.call(_identity).map(body)
        assert await flow.run([2, 5]) == [4, 10]

    @pytest.mark.asyncio
    async def test_body_receives_item_as_single_positional(self) -> None:
        """Each item is passed to the body's first node as one positional arg."""

        @verb(role=ROLE_A)
        async def wrap(ctx: Context, x: int) -> dict[str, int]:
            """Return a small dict per item."""
            return {"x": x}

        flow = make_ff().create()
        flow.call(_identity).map(lambda f: f.call(wrap))
        assert await flow.run([7, 8]) == [{"x": 7}, {"x": 8}]

    @pytest.mark.asyncio
    async def test_module_verb_as_body(self) -> None:
        """A bare module-level ``@verb`` is auto-wrapped as the map body."""
        flow = make_ff().create()
        flow.call(_identity).map(_double)
        assert await flow.run([1, 2, 3]) == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_bound_instance_method_verb_as_body(self) -> None:
        """A bound ``@verb`` instance method is auto-wrapped; ``self`` stays bound."""

        class Doubler:
            def __init__(self, offset: int) -> None:
                """Capture a per-instance offset added to each item."""
                self.offset = offset

            @verb(role=ROLE_A)
            async def scale(self, ctx: Context, x: int) -> int:
                """Return ``x * 2 + self.offset``, proving ``self`` is bound."""
                assert ctx.role is ROLE_A
                return x * 2 + self.offset

        instance = Doubler(offset=100)
        flow = make_ff().create()
        flow.call(_identity).map(instance.scale)
        assert await flow.run([1, 2, 3]) == [102, 104, 106]

    @pytest.mark.asyncio
    async def test_non_verb_callable_still_treated_as_builder(self) -> None:
        """A callable without ``.role`` (e.g. a plain classmethod) hits the
        builder-callback path — the verb-shape check must not steal it."""
        witnessed: list[Flow] = []

        def build_body(f: Flow) -> None:
            """Standard builder callback that mutates the fresh flow in place."""
            witnessed.append(f)
            f.call(_double)

        flow = make_ff().create()
        flow.call(_identity).map(build_body)
        assert await flow.run([1, 2]) == [2, 4]
        assert len(witnessed) == 1  # invoked once at materialize time


# -----------------------------------------------------------------------------
# .map — .guard() chained method
# -----------------------------------------------------------------------------


class TestMapGuard:
    """.guard() attaches a per-item skip predicate to the preceding map node."""

    @pytest.mark.asyncio
    async def test_falsy_verdict_yields_skipped_in_position(self) -> None:
        """A False verdict skips the body and lands ``Skipped(item)`` in that slot."""
        flow = make_ff().create()
        flow.call(_identity).map(_double).guard(lambda item, _ctx: item != 2)
        results = await flow.run([1, 2, 3])
        assert results[0] == 2
        assert isinstance(results[1], Skipped)
        assert results[1].item == 2
        assert results[2] == 6

    @pytest.mark.asyncio
    async def test_truthy_verdict_runs_body(self) -> None:
        """A truthy verdict runs the body unchanged."""
        flow = make_ff().create()
        flow.call(_identity).map(_double).guard(lambda _item, _ctx: True)
        assert await flow.run([1, 2, 3]) == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_guard_reads_projected_state(self) -> None:
        """Guard runs after per-item state projection so it can read ``ctx.state``."""
        seen: list[dict[str, int]] = []

        def project(_parent: Any) -> dict[str, int]:
            """Give each item its own child state tagged with a stub budget."""
            return {"budget": 100}

        def guard(item: int, ctx: Context) -> bool:
            """Read the projected budget and drop items over it."""
            seen.append(dict(ctx.state.data))
            return item <= ctx.state.data["budget"]

        flow = make_ff().create()
        flow.call(_identity).map(_double, state=project).guard(guard)
        results = await flow.run([50, 150, 25])
        assert results[0] == 100
        assert isinstance(results[1], Skipped)
        assert results[2] == 50
        assert all(s == {"budget": 100} for s in seen)

    @pytest.mark.asyncio
    async def test_async_guard_awaited(self) -> None:
        """An async guard is awaited before verdict is applied."""

        async def guard(item: int, _ctx: Context) -> bool:
            """Skip evens asynchronously."""
            await asyncio.sleep(0)
            return item % 2 == 1

        flow = make_ff().create()
        flow.call(_identity).map(_double).guard(guard)
        results = await flow.run([1, 2, 3])
        assert results[0] == 2
        assert isinstance(results[1], Skipped)
        assert results[2] == 6

    @pytest.mark.asyncio
    async def test_skip_does_not_fire_merge(self) -> None:
        """A skipped item's projected child state must not merge back."""
        merged: list[int] = []
        items = [1, 2, 3]
        item_iter = iter(items)

        def project(_parent: Any) -> dict[str, int]:
            """Tag each child state with its item identity (relies on projection order)."""
            return {"item": next(item_iter)}

        def merge(parent: dict[str, list[int]], child: dict[str, int]) -> None:
            """Record which child payloads made it back."""
            merged.append(child["item"])
            parent["seen"].append(child["item"])

        flow = make_ff().create(state={"seen": []})
        flow.call(_identity).map(
            _double,
            state=project,
            merge=merge,
        ).guard(lambda item, _ctx: item != 2)
        parent: dict[str, list[int]] = {"seen": []}
        await flow.run(items, state=parent)
        assert sorted(merged) == [1, 3]  # only survivors merged, skipped item excluded
        assert sorted(parent["seen"]) == [1, 3]

    @pytest.mark.asyncio
    async def test_guard_exception_becomes_failure_in_non_strict(self) -> None:
        """A guard that raises is wrapped as Failure in non-strict mode."""

        def exploding_guard(item: int, _ctx: Context) -> bool:
            """Raise for item 2, pass others."""
            if item == 2:
                raise ValueError("guard-boom")
            return True

        flow = make_ff().create()
        flow.call(_identity).map(_double, strict=False).guard(exploding_guard)
        results = await flow.run([1, 2, 3])
        assert results[0] == 2
        assert isinstance(results[1], Failure)
        assert isinstance(results[1].exception, ValueError)
        assert results[2] == 6


# -----------------------------------------------------------------------------
# .map — .on_error() chained method
# -----------------------------------------------------------------------------


class TestMapOnError:
    """.on_error() fires side-effect narration under both strict modes."""

    @pytest.mark.asyncio
    async def test_strict_true_fires_before_propagate(self) -> None:
        """In strict mode, on_error runs before the exception escapes."""
        observed: list[tuple[type, int]] = []

        @verb(role=ROLE_A)
        async def blow(_ctx: Context, x: int) -> int:
            """Raise for x == 2."""
            if x == 2:
                raise ValueError(f"bad:{x}")
            return x

        def on_err(exc: BaseException, item: int, _ctx: Context) -> None:
            """Record the exception and item."""
            observed.append((type(exc), item))

        flow = make_ff().create()
        flow.call(_identity).map(blow).on_error(on_err)
        with pytest.raises(ValueError, match="bad:2"):
            await flow.run([1, 2, 3])
        assert (ValueError, 2) in observed

    @pytest.mark.asyncio
    async def test_strict_false_fires_and_produces_failure(self) -> None:
        """In non-strict mode, on_error runs and the item still becomes a Failure."""
        observed: list[int] = []

        @verb(role=ROLE_A)
        async def blow(_ctx: Context, x: int) -> int:
            """Raise for even items."""
            if x % 2 == 0:
                raise ValueError(f"bad:{x}")
            return x

        async def on_err(_exc: BaseException, item: int, _ctx: Context) -> None:
            """Record the failing item asynchronously."""
            await asyncio.sleep(0)
            observed.append(item)

        flow = make_ff().create()
        flow.call(_identity).map(blow, strict=False).on_error(on_err)
        results = await flow.run([1, 2, 3, 4])
        assert results[0] == 1
        assert isinstance(results[1], Failure)
        assert results[2] == 3
        assert isinstance(results[3], Failure)
        assert sorted(observed) == [2, 4]

    @pytest.mark.asyncio
    async def test_on_error_exception_is_swallowed(self) -> None:
        """A hook that raises must not mask the original per-item exception."""

        @verb(role=ROLE_A)
        async def blow(_ctx: Context, _x: int) -> int:
            """Always raise."""
            raise ValueError("original")

        def on_err(_exc: BaseException, _item: int, _ctx: Context) -> None:
            """Raise a different exception from inside the hook."""
            raise RuntimeError("hook-blew-up")

        flow = make_ff().create()
        flow.call(_identity).map(blow).on_error(on_err)
        with pytest.raises(ValueError, match="original"):
            await flow.run([1])


# -----------------------------------------------------------------------------
# .map — chained-method attachment rules
# -----------------------------------------------------------------------------


class TestMapChainedApi:
    """.guard() and .on_error() are guarded against wrong-node attachment."""

    def test_guard_without_preceding_map_raises(self) -> None:
        """Chain with no nodes rejects .guard() with a clear TypeError."""
        flow = make_ff().create()
        with pytest.raises(TypeError, match=r"\.guard\(\) requires a preceding .map"):
            flow.guard(lambda _i, _c: True)

    def test_on_error_on_non_map_node_raises(self) -> None:
        """.on_error() after a plain .call node rejects with a clear TypeError."""
        flow = make_ff().create()
        flow.call(_identity)
        with pytest.raises(TypeError, match=r"\.on_error\(\) applies to map nodes"):
            flow.on_error(lambda _e, _i, _c: None)

    def test_double_guard_rejected(self) -> None:
        """Calling .guard() twice on the same map node is a TypeError."""
        flow = make_ff().create()
        flow.call(_identity).map(_double).guard(lambda _i, _c: True)
        with pytest.raises(TypeError, match="already set"):
            flow.guard(lambda _i, _c: True)

    def test_halt_without_preceding_map_raises(self) -> None:
        """Chain with no nodes rejects .halt() with a clear TypeError."""
        flow = make_ff().create()
        with pytest.raises(TypeError, match=r"\.halt\(\) requires a preceding .map"):
            flow.halt(asyncio.Event())

    def test_halt_on_non_map_node_raises(self) -> None:
        """.halt() after a plain .call node rejects with a clear TypeError."""
        flow = make_ff().create()
        flow.call(_identity)
        with pytest.raises(TypeError, match=r"\.halt\(\) applies to map nodes"):
            flow.halt(asyncio.Event())

    def test_double_halt_rejected(self) -> None:
        """Calling .halt() twice on the same map node is a TypeError."""
        flow = make_ff().create()
        flow.call(_identity).map(_double).halt(asyncio.Event())
        with pytest.raises(TypeError, match="already set"):
            flow.halt(asyncio.Event())


# -----------------------------------------------------------------------------
# .map — max_concurrency= throttle
# -----------------------------------------------------------------------------


class TestMapMaxConcurrency:
    """.map(max_concurrency=N) caps in-flight per-item runners."""

    def test_zero_rejected_at_build_time(self) -> None:
        """max_concurrency=0 is a ValueError at build time."""
        flow = make_ff().create()
        with pytest.raises(ValueError, match=r"\.map\(max_concurrency=\) must be an int >= 1"):
            flow.call(_identity).map(_double, max_concurrency=0)

    def test_negative_rejected_at_build_time(self) -> None:
        """max_concurrency=-1 is a ValueError at build time."""
        flow = make_ff().create()
        with pytest.raises(ValueError, match=r"\.map\(max_concurrency=\) must be an int >= 1"):
            flow.call(_identity).map(_double, max_concurrency=-1)

    def test_float_rejected_at_build_time(self) -> None:
        """max_concurrency=1.5 is a ValueError — must be int, not float."""
        flow = make_ff().create()
        with pytest.raises(ValueError, match=r"\.map\(max_concurrency=\) must be an int >= 1"):
            flow.call(_identity).map(_double, max_concurrency=1.5)

    def test_bool_rejected_at_build_time(self) -> None:
        """max_concurrency=True is a ValueError — bool is a subclass of int but not allowed."""
        flow = make_ff().create()
        with pytest.raises(ValueError, match=r"\.map\(max_concurrency=\) must be an int >= 1"):
            flow.call(_identity).map(_double, max_concurrency=True)

    @pytest.mark.asyncio
    async def test_none_default_leaves_unthrottled(self) -> None:
        """Without max_concurrency= the default behavior is unbounded gather."""
        flow = make_ff().create()
        flow.call(_identity).map(_double)
        assert await flow.run([1, 2, 3]) == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_caps_concurrent_in_flight_items(self) -> None:
        """At most max_concurrency runners execute their body at once."""
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        @verb(role=ROLE_A)
        async def slow(_ctx: Context, x: int) -> int:
            """Track concurrency by bumping a shared counter across a sleep."""
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return x

        flow = make_ff().create()
        flow.call(_identity).map(slow, max_concurrency=2)
        results = await flow.run([1, 2, 3, 4, 5])
        assert results == [1, 2, 3, 4, 5]
        assert peak <= 2
        assert peak >= 1  # at least one item ran

    @pytest.mark.asyncio
    async def test_max_concurrency_of_one_is_serial(self) -> None:
        """max_concurrency=1 forces sequential execution."""
        order: list[int] = []

        @verb(role=ROLE_A)
        async def track(_ctx: Context, x: int) -> int:
            """Record the enter/exit interleaving as a start/stop pair."""
            order.append(x)
            await asyncio.sleep(0)
            order.append(-x)
            return x

        flow = make_ff().create()
        flow.call(_identity).map(track, max_concurrency=1)
        await flow.run([1, 2, 3])
        # Serial: every start is followed by its matching stop before the next start.
        assert order == [1, -1, 2, -2, 3, -3]


# -----------------------------------------------------------------------------
# .map — .halt() chained method
# -----------------------------------------------------------------------------


class TestMapHalt:
    """.halt(event) short-circuits queued items to Skipped once the event fires."""

    @pytest.mark.asyncio
    async def test_pre_set_halt_skips_every_item(self) -> None:
        """A halt event that fires before .run() skips every item without running the body."""
        halt = asyncio.Event()
        halt.set()
        ran: list[int] = []

        @verb(role=ROLE_A)
        async def track(_ctx: Context, x: int) -> int:
            """Record any item that reaches the body (should never happen here)."""
            ran.append(x)
            return x

        flow = make_ff().create()
        flow.call(_identity).map(track).halt(halt)
        results = await flow.run([1, 2, 3])
        assert ran == []
        assert all(isinstance(r, Skipped) for r in results)
        assert [r.item for r in results] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_halt_mid_wave_skips_remaining_queue(self) -> None:
        """A halt set by an in-flight item drains the remaining queue as Skipped."""
        halt = asyncio.Event()
        ran: list[int] = []

        @verb(role=ROLE_A)
        async def track(_ctx: Context, x: int) -> int:
            """Set halt after processing item 1 to prove queued items short-circuit."""
            ran.append(x)
            if x == 1:
                halt.set()
            return x * 10

        flow = make_ff().create()
        flow.call(_identity).map(track, max_concurrency=1).halt(halt)
        results = await flow.run([1, 2, 3, 4])
        assert ran == [1]
        assert results[0] == 10
        assert all(isinstance(r, Skipped) for r in results[1:])
        assert [r.item for r in results[1:]] == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_halt_does_not_fire_merge(self) -> None:
        """A halted item's projected child state must not merge back."""
        merged: list[int] = []
        halt = asyncio.Event()
        halt.set()

        def project(_parent: Any) -> dict[str, int]:
            """Give every item a fresh child state — should never merge because halt is set."""
            return {"payload": 1}

        def merge(_parent: Any, child: dict[str, int]) -> None:
            """Record any child payload that leaks past the halt."""
            merged.append(child["payload"])

        flow = make_ff().create(state={})
        flow.call(_identity).map(_double, state=project, merge=merge).halt(halt)
        await flow.run([1, 2, 3])
        assert merged == []

    @pytest.mark.asyncio
    async def test_halt_unset_leaves_all_items_processed(self) -> None:
        """A halt event that never fires is a no-op — every item runs to completion."""
        halt = asyncio.Event()  # never set
        flow = make_ff().create()
        flow.call(_identity).map(_double).halt(halt)
        assert await flow.run([1, 2, 3]) == [2, 4, 6]


# -----------------------------------------------------------------------------
# .map — state-projection failure handling (CodeRabbit follow-up)
# -----------------------------------------------------------------------------


class TestMapProjectionErrors:
    """State-projection exceptions are handled symmetrically with guard/body errors."""

    @pytest.mark.asyncio
    async def test_projection_exception_wraps_as_failure_in_non_strict(self) -> None:
        """A state= projection that raises becomes a Failure(item) under strict=False."""

        def blow(_parent: Any) -> dict[str, int]:
            """Always raise from the projection callback."""
            raise RuntimeError("proj-boom")

        flow = make_ff().create()
        flow.call(_identity).map(_double, state=blow, strict=False)
        results = await flow.run([1, 2, 3])
        assert len(results) == 3
        for i, item in enumerate([1, 2, 3]):
            assert isinstance(results[i], Failure)
            assert isinstance(results[i].exception, RuntimeError)
            assert results[i].item == item

    @pytest.mark.asyncio
    async def test_projection_exception_fires_on_error_in_non_strict(self) -> None:
        """on_error narrates projection failures under strict=False."""
        observed: list[tuple[type, int]] = []
        observed_states: list[Any] = []

        def blow(_parent: Any) -> dict[str, int]:
            """Raise from projection to trigger the on_error narration path."""
            raise RuntimeError("proj-boom")

        def on_err(exc: BaseException, item: int, ctx: Context) -> None:
            """Record (exception type, item) so the test can assert coverage."""
            observed.append((type(exc), item))
            observed_states.append(ctx.state.data)

        flow = make_ff().create(state={"parent": True})
        flow.call(_identity).map(_double, state=blow, strict=False).on_error(on_err)
        await flow.run([1, 2])
        assert sorted(observed) == [(RuntimeError, 1), (RuntimeError, 2)]
        # Projection failed → on_error sees parent state, not (non-existent) child
        assert all(s == {"parent": True} for s in observed_states)

    @pytest.mark.asyncio
    async def test_projection_exception_propagates_in_strict(self) -> None:
        """A state= projection that raises still propagates when strict=True."""

        def blow(_parent: Any) -> dict[str, int]:
            """Raise from projection to check the strict-mode propagation path."""
            raise RuntimeError("proj-boom")

        flow = make_ff().create()
        flow.call(_identity).map(_double, state=blow)
        with pytest.raises(RuntimeError, match="proj-boom"):
            await flow.run([1])


# -----------------------------------------------------------------------------
# Buildable materialization
# -----------------------------------------------------------------------------


class TestBuildable:
    """Named Flow and inline callback are interchangeable everywhere Buildable is accepted."""

    @pytest.mark.asyncio
    async def test_callback_and_flow_produce_same_result(self) -> None:
        """A branch built two ways runs identically."""
        saia = StubFactory()

        named = make_ff().create("then").call(_double)
        flow_named = make_ff(saia_f=saia).create()
        flow_named.call(_identity).branch(when=lambda _p, _c: True, then=named)

        flow_cb = make_ff(saia_f=saia).create()
        flow_cb.call(_identity).branch(when=lambda _p, _c: True, then=lambda f: f.call(_double))

        assert await flow_named.run(4) == await flow_cb.run(4) == 8
