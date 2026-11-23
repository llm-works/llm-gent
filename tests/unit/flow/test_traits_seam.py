# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The run's environment: what every step of a run sees as ``ctx.traits`` and ``ctx.saia``.

The environment is given to the runner, not to the flow: every flow of a
run, nested ones included, sees the same one, and runs sharing an
environment share its SAIA instances, one per role.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_gent.core import Environment
from llm_gent.core.errors import TraitNotFoundError
from llm_gent.core.traits import Registry
from llm_gent.core.traits.base import BaseTrait
from llm_gent.flow import Factory, Flow, FlowRunner, verb

from .conftest import ROLE_A, ROLE_B, StubFactory, make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


class _MemoryStub(BaseTrait):
    """Minimal trait carrying an in-memory list — stands in for kelt-shaped traits."""

    def __init__(self, agent: Any) -> None:
        super().__init__(agent)
        self.writes: list[str] = []

    def record(self, item: str) -> None:
        self.writes.append(item)


class _StorageStub(BaseTrait):
    """Second trait type — proves multi-trait lookup by type keying."""


def _registry(*traits: BaseTrait) -> Registry:
    reg = Registry(LG)
    for t in traits:
        reg.register(t)
    return reg


def _flow() -> Flow:
    return Factory(LG).create()


class TestTraits:
    """``ctx.traits`` is the run's environment's registry."""

    async def test_none_without_a_registry(self) -> None:
        @verb(role=ROLE_A)
        async def peek(ctx: Any) -> Any:
            return ctx.traits

        assert await FlowRunner(LG).start(_flow().call(peek)) is None

    async def test_a_verb_reads_and_uses_a_registered_trait(self) -> None:
        memory = _MemoryStub(agent=object())
        runner = FlowRunner(LG, env=Environment(traits=_registry(memory)))

        @verb(role=ROLE_A)
        async def write(ctx: Any, item: str) -> str:
            ctx.traits.get(_MemoryStub).record(item)
            return item

        flow = _flow().call(write)
        await runner.start(flow, "first")
        await runner.start(flow, "second")
        assert memory.writes == ["first", "second"]

    async def test_traits_are_keyed_by_type(self) -> None:
        mem, stg = _MemoryStub(agent=object()), _StorageStub(agent=object())
        runner = FlowRunner(LG, env=Environment(traits=_registry(mem, stg)))

        @verb(role=ROLE_A)
        async def peek(ctx: Any) -> tuple[Any, Any]:
            return ctx.traits.get(_MemoryStub), ctx.traits.require(_StorageStub)

        assert await runner.start(_flow().call(peek)) == (mem, stg)

    async def test_require_on_a_missing_trait_raises(self) -> None:
        runner = FlowRunner(LG, env=Environment(traits=_registry()))

        @verb(role=ROLE_A)
        async def demand(ctx: Any) -> None:
            ctx.traits.require(_MemoryStub)

        with pytest.raises(TraitNotFoundError, match="_MemoryStub"):
            await runner.start(_flow().call(demand))


class TestNestedFlows:
    """Every flow of a run, nested ones included, sees the run's environment."""

    async def test_a_branch_arm_sees_it(self) -> None:
        memory = _MemoryStub(agent=object())
        runner = FlowRunner(LG, env=Environment(traits=_registry(memory)))

        @verb(role=ROLE_A)
        async def outer(ctx: Any) -> bool:
            return True

        @verb(role=ROLE_B)
        async def inner(ctx: Any, prev: Any) -> str:
            ctx.traits.require(_MemoryStub).record("inner")
            return "done"

        flow = _flow().call(outer).branch(when=lambda p, _c: p, then=lambda f: f.call(inner))
        await runner.start(flow)
        assert memory.writes == ["inner"]

    async def test_a_called_flow_sees_it(self) -> None:
        memory = _MemoryStub(agent=object())
        runner = FlowRunner(LG, env=Environment(traits=_registry(memory)))

        @verb(role=ROLE_A)
        async def inner(ctx: Any) -> Any:
            return ctx.traits.get(_MemoryStub)

        child = Flow(LG, name="child").call(inner)
        assert await runner.start(_flow().call(child)) is memory


class TestSAIA:
    """``ctx.saia`` comes from the environment, one per role, shared by the runs that share it."""

    async def test_runs_sharing_an_environment_share_a_role_s_saia(self) -> None:
        factory = StubFactory()
        runner = FlowRunner(LG, env=Environment(saia_factory=factory))

        @verb(role=ROLE_A)
        async def get(ctx: Any) -> Any:
            return ctx.saia

        flow = _flow().call(get)
        first, second = await runner.start(flow), await runner.start(_flow().call(get))
        assert first is second
        assert factory.built_for == [ROLE_A]

    async def test_another_environment_builds_its_own(self) -> None:
        factory = StubFactory()

        @verb(role=ROLE_A)
        async def get(ctx: Any) -> Any:
            return ctx.saia

        flow = _flow().call(get)
        a = await FlowRunner(LG, env=Environment(saia_factory=factory)).start(flow)
        b = await FlowRunner(LG, env=Environment(saia_factory=factory)).start(flow)
        assert a is not b
        assert factory.built_for == [ROLE_A, ROLE_A]

    async def test_a_role_bound_verb_without_a_saia_factory_raises_at_ctx_saia(self) -> None:
        @verb(role=ROLE_A)
        async def get(ctx: Any) -> Any:
            return ctx.saia

        with pytest.raises(RuntimeError, match="no SAIAFactory .* role 'a'"):
            await FlowRunner(LG).start(_flow().call(get))

    def test_the_runner_exposes_its_environment(self) -> None:
        env = Environment()
        assert FlowRunner(LG, env=env).env is env
        assert isinstance(FlowRunner(LG).env, Environment)
