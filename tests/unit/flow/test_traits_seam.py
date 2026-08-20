"""Tests for the Trait ↔ Flow seam.

Flow, FlowFactory, and Context each carry an optional ``traits`` registry.
Verbs reach mounted platform capabilities via ``ctx.traits``. When no
registry is supplied, ``ctx.traits is None`` and behavior is unchanged
from before the seam existed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.core.traits import Registry
from llm_gent.core.traits.base import BaseTrait
from llm_gent.flow import Flow, FlowFactory, verb

from .conftest import ROLE_A, ROLE_B, StubFactory, make_test_logger


class _MemoryStub(BaseTrait):
    """Minimal trait carrying an in-memory list — stands in for kelt-shaped traits."""

    def __init__(self, agent: Any) -> None:
        """Store the stub agent ref; init an empty write log."""
        super().__init__(agent)
        self.writes: list[str] = []

    def record(self, item: str) -> None:
        """Append ``item`` to the write log."""
        self.writes.append(item)


class _StorageStub(BaseTrait):
    """Second trait type — proves multi-trait lookup by type keying."""

    def __init__(self, agent: Any) -> None:
        """Store the stub agent ref."""
        super().__init__(agent)


def _fresh_registry(*traits: BaseTrait) -> Registry:
    """Build a Registry pre-populated with ``traits``."""
    reg = Registry(make_test_logger())
    for t in traits:
        reg.register(t)
    return reg


class TestContextField:
    """Context carries ``traits`` as an optional field with default ``None``."""

    def test_default_is_none(self) -> None:
        """A Flow constructed without traits produces ctx.traits == None."""
        captured: dict[str, Any] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Any) -> None:
            captured["traits"] = ctx.traits

        flow = Flow(make_test_logger(), saia_f=StubFactory()).call(peek)
        asyncio.run(flow.run())
        assert captured["traits"] is None

    def test_verb_reads_registered_trait(self) -> None:
        """Verb reads a trait via ctx.traits.get(TraitType)."""
        memory = _MemoryStub(agent=object())
        registry = _fresh_registry(memory)

        @verb(role=ROLE_A)
        async def write_fact(ctx: Any, item: str) -> str:
            trait = ctx.traits.get(_MemoryStub)
            trait.record(item)
            return item

        flow = Flow(make_test_logger(), saia_f=StubFactory(), traits=registry).call(write_fact)
        asyncio.run(flow.run("first"))
        asyncio.run(flow.run("second"))
        assert memory.writes == ["first", "second"]

    def test_verb_reads_via_require(self) -> None:
        """require() returns the same instance and raises on missing traits."""
        memory = _MemoryStub(agent=object())
        registry = _fresh_registry(memory)

        @verb(role=ROLE_A)
        async def read(ctx: Any) -> _MemoryStub:
            return ctx.traits.require(_MemoryStub)

        flow = Flow(make_test_logger(), saia_f=StubFactory(), traits=registry).call(read)
        result = asyncio.run(flow.run())
        assert result is memory

    def test_multiple_traits_keyed_by_type(self) -> None:
        """Registry lookup is type-keyed; two distinct trait types coexist."""
        mem = _MemoryStub(agent=object())
        stg = _StorageStub(agent=object())
        registry = _fresh_registry(mem, stg)

        @verb(role=ROLE_A)
        async def peek(ctx: Any) -> tuple[Any, Any]:
            return ctx.traits.get(_MemoryStub), ctx.traits.get(_StorageStub)

        flow = Flow(make_test_logger(), saia_f=StubFactory(), traits=registry).call(peek)
        m, s = asyncio.run(flow.run())
        assert m is mem
        assert s is stg


class TestFlowConstructor:
    """Flow.__init__ accepts a ``traits=`` kwarg; ``flow.traits`` property exposes it."""

    def test_traits_kwarg_stored(self) -> None:
        """The registry passed at construction is retrievable via the property."""
        registry = _fresh_registry()
        flow = Flow(make_test_logger(), saia_f=StubFactory(), traits=registry)
        assert flow.traits is registry

    def test_traits_default_none(self) -> None:
        """Without the kwarg the property returns None."""
        flow = Flow(make_test_logger(), saia_f=StubFactory())
        assert flow.traits is None


class TestFlowFactoryPropagation:
    """FlowFactory captures ``traits`` once; every ``.create()`` inherits it."""

    def test_factory_traits_reaches_flow(self) -> None:
        """A FlowFactory-built flow surfaces the captured registry."""
        registry = _fresh_registry()
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), traits=registry)
        assert ff.create().traits is registry

    def test_factory_traits_reaches_verb(self) -> None:
        """A verb dispatched through a FlowFactory-built flow reads ctx.traits."""
        memory = _MemoryStub(agent=object())
        registry = _fresh_registry(memory)
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), traits=registry)

        captured: dict[str, Any] = {}

        @verb(role=ROLE_A)
        async def peek(ctx: Any) -> None:
            captured["trait"] = ctx.traits.get(_MemoryStub)

        flow = ff.create("grade").call(peek)
        asyncio.run(flow.run())
        assert captured["trait"] is memory

    def test_factory_default_traits_none(self) -> None:
        """Without traits= on the factory, built flows carry None."""
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory())
        assert ff.create().traits is None


class TestFactoryDerivers:
    """with_saia_f and with_traits preserve unrelated fields."""

    def test_with_saia_f_preserves_traits(self) -> None:
        """Swapping the SAIA factory keeps the trait registry intact."""
        registry = _fresh_registry()
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), traits=registry)
        derived = ff.with_saia_f(StubFactory())
        assert derived.create().traits is registry

    def test_with_traits_returns_new_factory(self) -> None:
        """with_traits() derives a new factory (immutable-style swap)."""
        first, second = _fresh_registry(), _fresh_registry()
        ff = FlowFactory(make_test_logger(), saia_f=StubFactory(), traits=first)
        derived = ff.with_traits(second)
        assert derived is not ff
        assert derived.create().traits is second
        # Original untouched.
        assert ff.create().traits is first

    def test_with_traits_preserves_saia_f_and_state(self) -> None:
        """with_traits() carries saia_f, state, and lg forward."""
        sf = StubFactory()
        state = {"scope": "shared"}
        ff = FlowFactory(make_test_logger(), saia_f=sf, state=state, traits=_fresh_registry())
        derived = ff.with_traits(_fresh_registry())
        built = derived.create()
        assert built._saia_f is sf
        assert built.state is state


class TestSubflowInheritance:
    """Subflows borrow the outer runtime's traits (matches saia_f behavior)."""

    def test_lambda_subflow_sees_parent_traits(self) -> None:
        """A lambda-form (Buildable) subflow reads the parent's registered traits.

        Uses :meth:`Flow.branch` because ``.then`` takes a verb/Flow target
        directly, not a Buildable — the lambda-materialization path lives on
        the branch/loop/map primitives.
        """
        memory = _MemoryStub(agent=object())
        registry = _fresh_registry(memory)

        @verb(role=ROLE_A)
        async def outer(ctx: Any) -> bool:
            return True

        @verb(role=ROLE_B)
        async def inner(ctx: Any, prev: Any) -> str:
            ctx.traits.require(_MemoryStub).record("inner")
            return "done"

        flow = (
            Flow(make_test_logger(), saia_f=StubFactory(), traits=registry)
            .call(outer)
            .branch(when=lambda prev, _ctx: prev, then=lambda f: f.call(inner))
        )
        asyncio.run(flow.run())
        assert memory.writes == ["inner"]

    def test_flow_as_node_subflow_sees_parent_traits(self) -> None:
        """A Flow-as-node subflow reads the outer runtime's traits, not its own."""
        memory = _MemoryStub(agent=object())
        parent_registry = _fresh_registry(memory)

        @verb(role=ROLE_A)
        async def inner(ctx: Any) -> Any:
            return ctx.traits.get(_MemoryStub)

        # Child constructed with NO traits — should still see the parent's registry
        # when run as a subflow under the parent runtime.
        child = Flow(make_test_logger(), name="child").call(inner)
        parent = Flow(make_test_logger(), saia_f=StubFactory(), traits=parent_registry).call(child)
        result = asyncio.run(parent.run())
        assert result is memory


class TestBackwardCompatibility:
    """Existing code paths that never touch traits remain unchanged."""

    def test_run_without_traits_matches_prior_behavior(self) -> None:
        """A flow built the pre-seam way runs identically."""

        @verb(role=ROLE_A)
        async def echo(ctx: Any, value: str) -> str:
            assert ctx.traits is None
            return value

        flow = Flow(make_test_logger(), saia_f=StubFactory()).call(echo)
        assert asyncio.run(flow.run("hello")) == "hello"

    def test_missing_trait_via_require_raises(self) -> None:
        """require() on an unregistered trait type raises TraitNotFoundError."""
        registry = _fresh_registry()

        @verb(role=ROLE_A)
        async def demand(ctx: Any) -> None:
            ctx.traits.require(_MemoryStub)

        flow = Flow(make_test_logger(), saia_f=StubFactory(), traits=registry).call(demand)
        with pytest.raises(Exception, match="_MemoryStub"):
            asyncio.run(flow.run())
