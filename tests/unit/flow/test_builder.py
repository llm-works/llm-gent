"""Tests for the fluent Flow builder — .call/.then/.rescue/.after/.run."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.flow import UNSET, Context, Flow, Unset, verb

from .conftest import ROLE_A, ROLE_B, StubFactory, StubSAIA, make_ff, make_test_logger


# -----------------------------------------------------------------------------
# Fluent construction — surface & validation
# -----------------------------------------------------------------------------


class TestConstruction:
    """Flow construction via FlowFactory.create and direct Flow()."""

    def test_named_flow(self) -> None:
        """The ``name`` argument to create() becomes the flow's identifier."""
        flow = make_ff().create("scoring")
        assert flow.name == "scoring"

    def test_default_name_is_empty(self) -> None:
        """Name is optional and defaults to the empty string."""
        assert make_ff().create().name == ""

    def test_subflow_needs_no_saia(self) -> None:
        """A subflow (no saia) constructs cleanly — saia is optional."""
        Flow(make_test_logger(), "sub")  # no saia, no error


class TestBuilderValidation:
    """Fluent methods reject bad targets and misuse."""

    def test_call_rejects_non_verb_non_flow(self) -> None:
        """A plain callable without .role is neither a verb nor a Flow."""
        flow = make_ff().create()

        async def plain(ctx: Context) -> None:
            """No .role attached."""

        with pytest.raises(TypeError, match=".role"):
            flow.call(plain)

    def test_call_rejects_non_callable(self) -> None:
        """A non-callable, non-Flow value is rejected outright."""
        flow = make_ff().create()
        with pytest.raises(TypeError, match="verb .* or a Flow"):
            flow.call(42)

    def test_call_rejects_non_role_role_attr(self) -> None:
        """A .role attribute that isn't a Role is rejected."""
        flow = make_ff().create()

        async def bogus(ctx: Context) -> None:
            """.role is a string, not a Role."""

        bogus.role = "not a role"  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="Role instance"):
            flow.call(bogus)

    def test_call_rejects_verb_declaring_state_kwarg(self) -> None:
        """A verb with a ``state=`` parameter collides with ``Flow.run(state=...)``."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def bad(ctx: Context, *, state: dict[str, Any]) -> None:
            """Verb tries to bind ``state`` — would never receive it via .run()."""

        with pytest.raises(TypeError, match="reserved parameter 'state'"):
            flow.call(bad)

    def test_call_rejects_verb_state_positional(self) -> None:
        """Positional-or-keyword ``state`` collides too, not just keyword-only."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def bad(ctx: Context, state: dict[str, Any]) -> None:
            """``state`` bindable by name is also rejected."""

        with pytest.raises(TypeError, match="reserved parameter 'state'"):
            flow.call(bad)

    def test_call_accepts_verb_with_kwargs_absorbing_state(self) -> None:
        """``**kwargs``-only verbs are fine — .run(state=...) is stripped first."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def ok(ctx: Context, **kwargs: Any) -> None:
            """``**kwargs`` cannot collide — the reserved name never reaches it."""

        flow.call(ok)  # no error

    def test_rescue_before_any_call_raises(self) -> None:
        """.rescue requires a preceding node to attach to."""
        flow = make_ff().create()
        with pytest.raises(RuntimeError, match="preceding"):
            flow.rescue(lambda _exc, _prev, _ctx: None)

    def test_after_before_any_call_raises(self) -> None:
        """.after requires a preceding node to attach to."""
        flow = make_ff().create()
        with pytest.raises(RuntimeError, match="preceding"):
            flow.after(lambda _result, _ctx: None)


# -----------------------------------------------------------------------------
# Execution — data flow through the chain
# -----------------------------------------------------------------------------


class TestExecutionShape:
    """Flow.run walks the chain, threading data node-to-node."""

    @pytest.mark.asyncio
    async def test_single_node_receives_run_args(self) -> None:
        """First node gets *args, **kwargs verbatim from .run()."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def echo(ctx: Context, x: int, *, tag: str) -> tuple[int, str]:
            """Return the inputs."""
            return x, tag

        flow.call(echo)
        assert await flow.run(7, tag="hi") == (7, "hi")

    @pytest.mark.asyncio
    async def test_chained_positional_dataflow(self) -> None:
        """Later nodes receive the previous result as their single positional."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def double(ctx: Context, x: int) -> int:
            """Double the input."""
            return x * 2

        @verb(role=ROLE_A)
        async def inc(ctx: Context, x: int) -> int:
            """Increment the input."""
            return x + 1

        flow.call(double).then(inc)
        assert await flow.run(3) == 7  # (3*2)+1

    @pytest.mark.asyncio
    async def test_project_transforms_between_nodes(self) -> None:
        """A project callable reshapes the previous result into the next input."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def make_dict(ctx: Context, x: int) -> dict[str, int]:
            """Return a small dict."""
            return {"n": x, "extra": 99}

        @verb(role=ROLE_A)
        async def times_ten(ctx: Context, n: int) -> int:
            """Multiply by ten."""
            return n * 10

        flow.call(make_dict).then(times_ten, project=lambda d: d["n"])
        assert await flow.run(4) == 40

    @pytest.mark.asyncio
    async def test_later_nodes_do_not_inherit_run_kwargs(self) -> None:
        """Run kwargs feed only the first node; later nodes see just prev result."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def head(ctx: Context, x: int, *, tag: str) -> str:
            """Combine positional and kwarg into a string."""
            return f"{tag}={x}"

        @verb(role=ROLE_A)
        async def tail(ctx: Context, s: str) -> int:
            """Return the length of the string."""
            return len(s)

        flow.call(head).then(tail)
        assert await flow.run(42, tag="v") == len("v=42")

    @pytest.mark.asyncio
    async def test_verb_ctx_has_role_and_saia(self) -> None:
        """Each verb receives a Context bound to its own role's saia."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def check_a(ctx: Context, _: Any = None) -> tuple[Any, Any]:
            """Return (role, saia) for inspection."""
            return ctx.role, ctx.saia

        @verb(role=ROLE_B)
        async def check_b(ctx: Context, _: Any) -> tuple[Any, Any]:
            """Return (role, saia) for inspection."""
            return ctx.role, ctx.saia

        flow.call(check_a).then(check_b)
        role_b, saia_b = await flow.run(None)
        assert role_b is ROLE_B
        assert isinstance(saia_b, StubSAIA)
        assert saia_b.role is ROLE_B


# -----------------------------------------------------------------------------
# Hooks — rescue and after
# -----------------------------------------------------------------------------


class TestRescue:
    """.rescue converts exceptions to fallback values, chain continues."""

    @pytest.mark.asyncio
    async def test_rescue_returns_fallback_and_chain_continues(self) -> None:
        """Rescue's return value becomes the node's result; next node sees it."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def fail(ctx: Context, _: Any = None) -> int:
            """Always raise."""
            raise ValueError("boom")

        @verb(role=ROLE_A)
        async def add_one(ctx: Context, x: int) -> int:
            """Increment."""
            return x + 1

        flow.call(fail).rescue(lambda _exc, _prev, _ctx: 100).then(add_one)
        assert await flow.run(None) == 101

    @pytest.mark.asyncio
    async def test_rescue_receives_exception_and_ctx(self) -> None:
        """Rescue callback sees the raised exception and the node's ctx."""
        seen: dict[str, Any] = {}

        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def fail(ctx: Context) -> None:
            """Raise a specific error."""
            raise RuntimeError("marker")

        def policy(exc: BaseException, _prev: Any, ctx: Context) -> str:
            """Record what was seen and return a fallback."""
            seen["exc"] = exc
            seen["role"] = ctx.role
            return "fell-back"

        flow.call(fail, rescue=policy)
        assert await flow.run() == "fell-back"
        assert isinstance(seen["exc"], RuntimeError)
        assert seen["role"] is ROLE_A

    @pytest.mark.asyncio
    async def test_rescue_can_be_async(self) -> None:
        """An async rescue is awaited before its return becomes the result."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def fail(ctx: Context) -> None:
            """Always raise."""
            raise ValueError

        async def policy(exc: BaseException, _prev: Any, ctx: Context) -> str:
            """Async fallback."""
            await asyncio.sleep(0)
            return "async-fallback"

        flow.call(fail).rescue(policy)
        assert await flow.run() == "async-fallback"

    @pytest.mark.asyncio
    async def test_rescue_does_not_catch_cancelled(self) -> None:
        """asyncio.CancelledError propagates past any rescue policy."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def cancel_me(ctx: Context) -> None:
            """Raise cancellation from inside the verb."""
            raise asyncio.CancelledError

        rescue_calls = 0

        def policy(exc: BaseException, _prev: Any, ctx: Context) -> str:
            """Should never fire."""
            nonlocal rescue_calls
            rescue_calls += 1
            return "should-not-see-this"

        flow.call(cancel_me).rescue(policy)
        with pytest.raises(asyncio.CancelledError):
            await flow.run()
        assert rescue_calls == 0

    @pytest.mark.asyncio
    async def test_no_rescue_lets_exception_propagate(self) -> None:
        """Without .rescue, the original exception bubbles out of .run()."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def fail(ctx: Context) -> None:
            """Raise."""
            raise RuntimeError("propagates")

        flow.call(fail)
        with pytest.raises(RuntimeError, match="propagates"):
            await flow.run()

    @pytest.mark.asyncio
    async def test_rescue_pending_input_unset_for_first_node_no_positional(self) -> None:
        """First node + no ``.run()`` positional → rescue's pending_input is UNSET."""
        seen: dict[str, Any] = {}

        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def fail(ctx: Context) -> None:
            """Raise."""
            raise RuntimeError("x")

        def policy(_exc: BaseException, prev: Any, _ctx: Context) -> str:
            """Record the pending_input for assertion."""
            seen["prev"] = prev
            return "fb"

        flow.call(fail, rescue=policy)
        assert await flow.run() == "fb"
        assert seen["prev"] is UNSET
        assert isinstance(seen["prev"], Unset)

    @pytest.mark.asyncio
    async def test_rescue_pending_input_is_run_positional_on_first_node(self) -> None:
        """First node + ``.run(x)`` → rescue sees ``x`` as pending_input."""
        seen: dict[str, Any] = {}

        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def fail(_ctx: Context, _x: Any) -> None:
            """Raise."""
            raise RuntimeError("x")

        def policy(_exc: BaseException, prev: Any, _ctx: Context) -> str:
            """Record the pending_input for assertion."""
            seen["prev"] = prev
            return "fb"

        flow.call(fail, rescue=policy)
        assert await flow.run("hello") == "fb"
        assert seen["prev"] == "hello"

    @pytest.mark.asyncio
    async def test_rescue_pending_input_is_prior_result_on_chained_node(self) -> None:
        """A rescue on a ``.then`` node sees the prior node's return."""
        seen: dict[str, Any] = {}

        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def produce(_ctx: Context) -> int:
            """Emit the value the failing node would have received."""
            return 42

        @verb(role=ROLE_A)
        async def fail(_ctx: Context, _prev: int) -> None:
            """Raise so rescue fires."""
            raise RuntimeError("boom")

        def policy(_exc: BaseException, prev: Any, _ctx: Context) -> Any:
            """Return the received pending_input verbatim as the fallback."""
            seen["prev"] = prev
            return prev

        flow.call(produce).then(fail, rescue=policy)
        assert await flow.run() == 42
        assert seen["prev"] == 42

    @pytest.mark.asyncio
    async def test_rescue_pending_input_reflects_project(self) -> None:
        """When ``project=`` reshapes the input, rescue sees the post-project value."""
        seen: dict[str, Any] = {}

        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def produce(_ctx: Context) -> int:
            """Return the pre-project value."""
            return 10

        @verb(role=ROLE_A)
        async def fail(_ctx: Context, _prev: Any) -> None:
            """Raise so rescue fires."""
            raise RuntimeError("boom")

        def policy(_exc: BaseException, prev: Any, _ctx: Context) -> Any:
            """Record the pending_input; return it as the fallback."""
            seen["prev"] = prev
            return prev

        flow.call(produce).then(fail, project=lambda x: x * 3, rescue=policy)
        assert await flow.run() == 30
        assert seen["prev"] == 30


class TestAfter:
    """.after is a side-effect hook fired after successful (or rescued) results."""

    @pytest.mark.asyncio
    async def test_after_receives_result_and_ctx(self) -> None:
        """After hook sees the node's final result and its ctx."""
        seen: dict[str, Any] = {}

        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def make(ctx: Context) -> int:
            """Produce a value."""
            return 5

        flow.call(make).after(lambda result, ctx: seen.update(result=result, role=ctx.role))  # noqa: ARG005
        assert await flow.run() == 5
        assert seen == {"result": 5, "role": ROLE_A}

    @pytest.mark.asyncio
    async def test_after_does_not_transform_result(self) -> None:
        """Return value of .after is ignored — the node's result is unchanged."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def make(ctx: Context) -> int:
            """Produce 3."""
            return 3

        flow.call(make).after(lambda _result, _ctx: 9999)
        assert await flow.run() == 3

    @pytest.mark.asyncio
    async def test_after_fires_after_rescue(self) -> None:
        """When rescue fires, .after sees the rescued (fallback) value."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def fail(ctx: Context) -> None:
            """Always raise."""
            raise ValueError

        seen: list[Any] = []
        flow.call(fail).rescue(lambda _e, _prev, _c: "rescued").after(lambda r, _c: seen.append(r))
        assert await flow.run() == "rescued"
        assert seen == ["rescued"]

    @pytest.mark.asyncio
    async def test_after_can_be_async(self) -> None:
        """An async after hook is awaited before the next node runs."""
        flow = make_ff().create()

        @verb(role=ROLE_A)
        async def one(ctx: Context) -> int:
            """Return 1."""
            return 1

        @verb(role=ROLE_A)
        async def two(ctx: Context, x: int) -> int:
            """Return input + 1."""
            return x + 1

        order: list[str] = []

        async def hook(result: Any, ctx: Context) -> None:
            """Mark ordering."""
            await asyncio.sleep(0)
            order.append(f"after:{result}")

        flow.call(one).after(hook).then(two)
        assert await flow.run() == 2
        assert order == ["after:1"]


class TestChainedVsKwargsHooks:
    """The two hook-attachment styles produce equivalent flows."""

    @pytest.mark.asyncio
    async def test_kwargs_rescue_equals_chained(self) -> None:
        """.call(x, rescue=r) matches .call(x).rescue(r)."""
        saia = StubFactory()

        @verb(role=ROLE_A)
        async def fail(ctx: Context) -> None:
            """Raise."""
            raise ValueError

        chained = make_ff(saia_f=saia).create().call(fail).rescue(lambda _e, _prev, _c: "R")
        kwargs = make_ff(saia_f=saia).create().call(fail, rescue=lambda _e, _prev, _c: "R")

        assert await chained.run() == await kwargs.run() == "R"

    @pytest.mark.asyncio
    async def test_kwargs_after_equals_chained(self) -> None:
        """.call(x, after=h) matches .call(x).after(h)."""
        saia = StubFactory()
        seen: list[Any] = []

        @verb(role=ROLE_A)
        async def make(ctx: Context) -> int:
            """Return 42."""
            return 42

        chained = (
            make_ff(saia_f=saia).create().call(make).after(lambda r, _c: seen.append(("c", r)))
        )
        kwargs = make_ff(saia_f=saia).create().call(make, after=lambda r, _c: seen.append(("k", r)))

        await chained.run()
        await kwargs.run()
        assert seen == [("c", 42), ("k", 42)]


# -----------------------------------------------------------------------------
# Subflow recursion
# -----------------------------------------------------------------------------


class TestSubflow:
    """A Flow used as a node is executed recursively; runtime is shared."""

    @pytest.mark.asyncio
    async def test_subflow_runs_as_node(self) -> None:
        """A subflow's output flows into the next node like a verb's would."""
        saia = StubFactory()

        @verb(role=ROLE_A)
        async def add_one(ctx: Context, x: int) -> int:
            """+1."""
            return x + 1

        @verb(role=ROLE_A)
        async def times_two(ctx: Context, x: int) -> int:
            """*2."""
            return x * 2

        @verb(role=ROLE_A)
        async def stringify(ctx: Context, x: int) -> str:
            """Format."""
            return f"n={x}"

        sub = make_ff().create("inner").call(add_one).then(times_two)  # no saia
        main = make_ff(saia_f=saia).create("outer").call(sub).then(stringify)

        assert await main.run(5) == "n=12"  # ((5+1)*2)

    @pytest.mark.asyncio
    async def test_subflow_shares_runtime_saia_cache(self) -> None:
        """Verbs across parent and subflow with same role share one saia."""
        saia = StubFactory()

        @verb(role=ROLE_A)
        async def outer_verb(ctx: Context) -> Any:
            """Return the saia."""
            return ctx.saia

        @verb(role=ROLE_A)
        async def inner_verb(ctx: Context, prev: Any) -> Any:
            """Return outer_saia identity check."""
            return ctx.saia is prev

        sub = make_ff().create("inner").call(inner_verb)
        main = make_ff(saia_f=saia).create().call(outer_verb).then(sub)

        assert await main.run() is True
        assert saia.built_for == [ROLE_A]

    @pytest.mark.asyncio
    async def test_subflow_hooks_receive_ambient_ctx(self) -> None:
        """Rescue/after on a subflow node get a ctx with role=None, saia_f=None."""
        saia = StubFactory()

        @verb(role=ROLE_A)
        async def boom(ctx: Context, _: Any = None) -> None:
            """Raise from inside the subflow."""
            raise ValueError("inside")

        sub = make_ff().create("inner").call(boom)
        seen: dict[str, Any] = {}

        def policy(exc: BaseException, _prev: Any, ctx: Context) -> str:
            """Record ambient ctx fields."""
            seen["role"] = ctx.role
            seen["saia"] = ctx.saia
            seen["flow"] = ctx.flow
            return "handled"

        main = make_ff(saia_f=saia).create("outer").call(sub, rescue=policy)
        assert await main.run(None) == "handled"
        assert seen["role"] is None
        assert seen["saia"] is None
        assert seen["flow"] is main

    @pytest.mark.asyncio
    async def test_subflow_state_overrides_propagate(self) -> None:
        """The active state passed by run() reaches subflow verbs' ctx.state."""
        saia = StubFactory()

        @verb(role=ROLE_A)
        async def read_state(ctx: Context, _: Any = None) -> Any:
            """Return the payload wrapped in ctx.state."""
            return ctx.state.data

        sub = make_ff().create("inner").call(read_state)
        main = make_ff(saia_f=saia).create(state={"default": True}).call(sub)

        assert await main.run(None) == {"default": True}
        assert await main.run(None, state={"overridden": True}) == {"overridden": True}


# -----------------------------------------------------------------------------
# Runtime errors
# -----------------------------------------------------------------------------


class TestRuntimeErrors:
    """Preflight errors caught at .run() time."""

    @pytest.mark.asyncio
    async def test_empty_flow_raises(self) -> None:
        """.run() on a flow with no nodes is a programming error."""
        flow = make_ff().create("empty")
        with pytest.raises(RuntimeError, match="no nodes"):
            await flow.run()

    @pytest.mark.asyncio
    async def test_top_level_without_saia_raises(self) -> None:
        """A top-level flow with no SAIAFactory can't build a saia to run verbs.

        Under the ``nodes are verbs, always`` principle, every verb needs a
        role-bound saia; a Flow that dispatches verbs must therefore carry a
        factory. Enforced at run time so the collision surfaces immediately
        rather than silently binding ``ctx.saia=None`` and either
        AttributeError-ing deep in the verb body or (worse) succeeding when
        the verb happens not to reach for saia that run.
        """

        @verb(role=ROLE_A)
        async def do(ctx: Context) -> int:
            """No-op."""
            return 0

        flow = Flow(make_test_logger(), "naked").call(do)
        with pytest.raises(RuntimeError, match="SAIAFactory"):
            await flow.run()


# -----------------------------------------------------------------------------
# Cancellation propagation across the chain
# -----------------------------------------------------------------------------


class TestCancellation:
    """asyncio.CancelledError propagates end-to-end without being trapped."""

    @pytest.mark.asyncio
    async def test_cancellation_in_subflow_propagates_through_parent(self) -> None:
        """Cancelling inside a subflow verb surfaces past parent rescue too."""
        saia = StubFactory()

        @verb(role=ROLE_A)
        async def inner_cancel(ctx: Context, _: Any = None) -> None:
            """Cancel from deep inside."""
            raise asyncio.CancelledError

        parent_rescued = False

        def parent_rescue(exc: BaseException, _prev: Any, ctx: Context) -> str:
            """Should never fire."""
            nonlocal parent_rescued
            parent_rescued = True
            return "swallowed"

        sub = make_ff().create("inner").call(inner_cancel)
        main = make_ff(saia_f=saia).create("outer").call(sub, rescue=parent_rescue)

        with pytest.raises(asyncio.CancelledError):
            await main.run(None)
        assert parent_rescued is False

    @pytest.mark.asyncio
    async def test_task_cancellation_reaches_running_verb(self) -> None:
        """Cancelling the task running .run() cancels the awaited verb."""
        saia = StubFactory()
        started = asyncio.Event()

        @verb(role=ROLE_A)
        async def slow(ctx: Context) -> None:
            """Signal, then wait forever."""
            started.set()
            await asyncio.sleep(3600)

        flow = make_ff(saia_f=saia).create().call(slow)
        task = asyncio.create_task(flow.run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
