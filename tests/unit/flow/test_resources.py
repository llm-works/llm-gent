# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Resources — ``with_resource``, ``ctx.resource``, scoping, checkpoints, fluent names.

The test resource is a counter that rolls its counts up to its parent,
the way a cost tracker rolls up spend: a per-run child counts its run, the
resource it came from counts everything below it. "A fresh process" here
is a new Flow and new counters over the same store.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gent.core.cost import CostTracker, PricingConfig
from llm_gent.flow import (
    COST,
    HALTED,
    Context,
    Factory,
    Flow,
    History,
    Interrupted,
    Resource,
    ResourceKey,
    resource_method,
    verb,
)
from llm_gent.flow.state.snapshot import RESOURCES
from llm_gent.flow.stores import InMemoryCheckpointStore

from .conftest import make_test_logger


pytestmark = [pytest.mark.unit]

NAME = "resources"


class Counter(Resource):
    """A count that rolls up to its parent; ``child(**args)`` records its args."""

    def __init__(self, parent: Counter | None = None, **args: Any) -> None:
        self.count = 0
        self.parent = parent
        self.args = args
        self.restored: list[dict[str, Any]] = []

    def add(self, n: int = 1) -> None:
        self.count += n
        if self.parent is not None:
            self.parent.add(n)

    def snapshot(self) -> dict[str, Any]:
        return {"count": self.count}

    def restore(self, data: dict[str, Any]) -> None:
        self.restored.append(data)
        self.count = data["count"]

    def child(self, **args: Any) -> Counter:
        return Counter(self, **args)


class Plain(Resource):
    """A resource without ``child()``."""

    def snapshot(self) -> dict[str, Any]:
        return {}

    def restore(self, data: dict[str, Any]) -> None:
        pass


class Structural:
    """Implements the protocol without subclassing it."""

    def __init__(self) -> None:
        self.restored: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {"n": 1}

    def restore(self, data: dict[str, Any]) -> None:
        self.restored.append(data)


COUNTER = ResourceKey[Counter]("counter")


def _ff() -> Factory[Flow]:
    return Factory(make_test_logger())


@verb
async def tick(ctx: Context[Any], x: Any) -> Any:
    ctx.resource(COUNTER).add()
    return x


class TestDeclared:
    async def test_a_verb_reads_it_typed_by_its_key(self) -> None:
        counter = Counter()
        seen: list[Counter] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.resource(COUNTER))
            return x

        assert await _ff().create().with_resource(COUNTER, counter).call(step).run(1) == 1
        assert seen == [counter]

    async def test_flows_below_inherit_it(self) -> None:
        counter = Counter()
        inner = _ff().create().call(tick)
        body = _ff().create().call(inner)
        flow = (
            _ff()
            .create()
            .with_resource(COUNTER, counter)
            .call(body)
            .map(lambda b: b.call(tick), items=lambda _p, _c: [1, 2])
            .iterate(lambda b: b.call(tick), max_iters=2)
        )
        await flow.run(0)
        assert counter.count == 1 + 2 + 2

    async def test_a_nested_flow_s_own_replaces_it_below(self) -> None:
        outer, inner = Counter(), Counter()
        sub = _ff().create().with_resource(COUNTER, inner).call(tick)
        flow = _ff().create().with_resource(COUNTER, outer).call(tick).then(sub).then(tick)
        await flow.run(0)
        assert (outer.count, inner.count) == (2, 1)

    async def test_none_in_scope_raises_naming_the_key(self) -> None:
        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            ctx.resource(COUNTER)
            return x

        with pytest.raises(KeyError, match="'counter'"):
            await _ff().create().call(step).run(1)

    async def test_a_default_when_none_is_in_scope(self) -> None:
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.resource(COUNTER, None))
            return x

        await _ff().create().call(step).run(1)
        assert seen == [None]

    async def test_dispatch_hands_the_flow_s_resources_to_the_verb(self) -> None:
        counter = Counter()
        flow = _ff().create().with_resource(COUNTER, counter)
        flow.register(tick)
        await flow.dispatch("tick", 0)
        assert counter.count == 1


class TestPerRunChild:
    async def test_each_item_runs_with_its_own_child_of_the_enclosing_one(self) -> None:
        root = Counter()
        seen: list[Counter] = []

        @verb
        async def step(ctx: Context[Any], n: int) -> int:
            counter = ctx.resource(COUNTER)
            seen.append(counter)
            counter.add(n)
            return n

        flow = (
            _ff()
            .create()
            .with_resource(COUNTER, root)
            .map(lambda b: b.with_resource(COUNTER, per="item").call(step), items=lambda *_: [1, 2])
        )
        assert await flow.run() == [1, 2]
        assert seen[0] is not seen[1]
        assert all(c.parent is root and c.args == {"per": "item"} for c in seen)
        assert sorted(c.count for c in seen) == [1, 2]
        assert root.count == 3

    async def test_a_value_and_child_arguments_give_each_run_a_child_of_its_own(self) -> None:
        outer, own = Counter(), Counter()
        seen: list[Counter] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.resource(COUNTER))
            ctx.resource(COUNTER).add()
            return x

        sub = _ff().create().with_resource(COUNTER, own, per="call").call(step)
        flow = _ff().create().with_resource(COUNTER, outer).call(sub).then(sub)
        await flow.run(1)
        assert seen[0] is not seen[1] and all(c.parent is own for c in seen)
        assert (outer.count, own.count) == (0, 2)

    async def test_calls_for_one_key_combine(self) -> None:
        own = Counter()
        seen: list[Counter] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.resource(COUNTER))
            return x

        sub = _ff().create().with_resource(COUNTER, own).with_resource(COUNTER, per=1).call(step)
        await _ff().create().call(sub).run(0)
        assert seen[0].parent is own and seen[0].args == {"per": 1}


class TestAcrossResume:
    @staticmethod
    def _flow(store: Any, counter: Counter, halt: asyncio.Event, arm: bool) -> Any:
        """``a`` counts 1, ``b`` counts 2 and (armed) sets the halt, ``c`` counts 4."""

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            ctx.resource(COUNTER).add(1)
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            ctx.resource(COUNTER).add(2)
            if arm:
                halt.set()
            return x

        @verb
        async def c(ctx: Context[Any], x: int) -> int:
            if ctx.halt is not None and ctx.halt.is_set():
                raise Interrupted()
            ctx.resource(COUNTER).add(4)
            return x

        return (
            _ff()
            .create(state={})
            .with_checkpoint_store(store, NAME)
            .with_checkpointer()
            .with_halt(halt)
            .with_resource(COUNTER, counter)
            .call(a)
            .then(b)
            .then(c)
        )

    async def test_the_count_comes_back_after_a_halt(self) -> None:
        store = InMemoryCheckpointStore()
        assert await self._flow(store, Counter(), asyncio.Event(), True).run(1) is HALTED

        fresh = Counter()
        assert await self._flow(store, fresh, asyncio.Event(), False).run(resume="latest") == 1
        assert fresh.restored == [{"count": 3}]
        assert fresh.count == 3 + 4

    async def test_a_finished_run_s_count_carries_into_the_next_run(self) -> None:
        store = InMemoryCheckpointStore()
        await self._flow(store, Counter(), asyncio.Event(), False).run(1)
        history = History(store, NAME)
        head = await history.head()
        assert head is not None and await history.is_complete()
        snapshot = await history.snapshot(head)
        assert snapshot.cursors[""][RESOURCES] == {"counter": {"count": 7}}

        fresh = Counter()
        await self._flow(store, fresh, asyncio.Event(), False).run(1, resume="latest")
        assert fresh.count == 7 + 7

    async def test_one_every_flow_inherits_is_kept_once_at_the_top(self) -> None:
        """A Factory.with_resource puts the same object on every flow: one position."""
        store = InMemoryCheckpointStore()
        halt = asyncio.Event()

        def build(counter: Counter, arm: bool) -> Any:
            ff = _ff().with_resource(COUNTER, counter)

            @verb
            async def b(ctx: Context[Any], x: int) -> int:
                ctx.resource(COUNTER).add(2)
                if arm:
                    halt.set()
                    raise Interrupted()
                return x

            inner = ff.create().call(b)
            return (
                ff.create(state={})
                .with_checkpoint_store(store, NAME)
                .with_halt(halt)
                .call(tick)
                .then(inner)
            )

        assert await build(Counter(), arm=True).run(1) is HALTED
        history = History(store, NAME)
        snapshot = await history.snapshot(await history.head())
        assert [p for p, c in snapshot.cursors.items() if RESOURCES in c] == [""]

        fresh = Counter()
        assert await build(fresh, arm=False).run(1, resume="latest") == 1
        assert fresh.count == 3 + 2  # 3 restored, b again: 2

    async def test_items_halted_mid_way_resume_with_their_children_s_counts(self) -> None:
        store = InMemoryCheckpointStore()
        halt = asyncio.Event()

        def build(root: Counter, arm: bool, before: dict[int, int]) -> Any:
            started: set[int] = set()

            @verb
            async def first(ctx: Context[Any], n: int) -> int:
                started.add(n)
                ctx.resource(COUNTER).add(n)
                if arm and len(started) == 2:
                    halt.set()
                return n

            @verb
            async def second(ctx: Context[Any], n: int) -> int:
                await asyncio.sleep(0)  # lets a set run halt reach the item's own halt
                if ctx.halt is not None and ctx.halt.is_set():
                    raise Interrupted()
                before[n] = ctx.resource(COUNTER).count
                ctx.resource(COUNTER).add(10)
                return n

            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
                .with_halt(halt)
                .with_resource(COUNTER, root)
                .map(
                    lambda b: b.with_resource(COUNTER, per="item").call(first).then(second),
                    items=lambda *_: [1, 2],
                )
            )

        root1 = Counter()
        assert await build(root1, arm=True, before={}).run() is HALTED
        assert root1.count == 3

        root2, before = Counter(), {}
        halt.clear()
        assert await build(root2, arm=False, before=before).run(resume="latest") == [1, 2]
        assert before == {1: 1, 2: 2}  # each item's child came back with its own count
        # 3 restored at the root; the items' restored counts do not roll up again.
        assert root2.count == 3 + 20

    async def test_a_saved_resource_nothing_declares_any_more_is_dropped(self) -> None:
        """The run continues; a resource under another name is not handed the old one's data."""
        store = InMemoryCheckpointStore()
        other = ResourceKey[Counter]("other")

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            return x

        def build(key: ResourceKey[Counter], counter: Counter) -> Any:
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
                .with_resource(key, counter)
                .call(step)
            )

        await build(COUNTER, Counter()).run(1)
        fresh = Counter()
        assert await build(other, fresh).run(1, resume="latest") == 1
        assert fresh.restored == []


class TestAcrossShortcut:
    async def test_a_per_run_child_carries_through_the_shortcut(self) -> None:
        """The continuation runs with the same child: counts before and after the cut add up once."""
        root = Counter()
        cut = asyncio.Event()
        seen: list[Counter] = []

        @verb
        async def a(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.resource(COUNTER))
            ctx.resource(COUNTER).add(1)
            cut.set()
            await asyncio.sleep(0)  # the stop follows the signal
            return x

        @verb
        async def b(ctx: Context[Any], x: int) -> int:
            ctx.resource(COUNTER).add(100)
            return x

        @verb
        async def c(ctx: Context[Any], x: int) -> int:
            seen.append(ctx.resource(COUNTER))
            ctx.resource(COUNTER).add(2)
            return x

        sub = (
            _ff()
            .create()
            .with_resource(COUNTER, per="call")
            .with_shortcut("cut", to="c")
            .call(a)
            .then(b)
            .then(c, name="c")
        )
        flow = _ff().create().with_signal("cut", cut).with_resource(COUNTER, root).call(sub)
        assert await flow.run(1) == 1
        assert seen[0] is seen[1] and seen[0].count == 3
        assert root.count == 3

    @pytest.mark.parametrize("map_cut", [False, True], ids=["items-alone", "map-too"])
    async def test_items_continue_with_their_children_s_counts(self, map_cut: bool) -> None:
        """Alone, each item continues in place; with the map cut too, from restaged entries."""
        root = Counter()
        cut = asyncio.Event()
        gate = asyncio.Event()
        at_extract: dict[int, int] = {}

        @verb
        async def query(ctx: Context[Any], n: int) -> int:
            ctx.resource(COUNTER).add(1)
            if n == 1:
                cut.set()
                await asyncio.sleep(0)  # the stop follows the signal
                gate.set()
            else:
                await gate.wait()
            return n

        @verb
        async def explore(ctx: Context[Any], n: int) -> int:
            ctx.resource(COUNTER).add(100)
            return n

        @verb
        async def extract(ctx: Context[Any], n: int) -> int:
            at_extract[n] = ctx.resource(COUNTER).count
            ctx.resource(COUNTER).add(2)
            return n

        def item(b: Any) -> None:
            b.with_resource(COUNTER, per="item").call(query).then(explore)
            b.then(extract, name="extract").with_shortcut("cut", to="extract")

        waves = _ff().create().map(item, items=lambda *_: [0, 1], max_concurrency=2)
        if map_cut:
            waves.with_shortcut("cut")
        flow = _ff().create().with_signal("cut", cut).with_resource(COUNTER, root).call(waves)
        assert await flow.run() == [0, 1]
        assert at_extract == {0: 1, 1: 1}
        assert root.count == 2 * (1 + 2)


class TestProtocol:
    async def test_a_class_with_the_methods_and_no_base_is_a_resource(self) -> None:
        """Structural: checkpointed and restored like a subclass of the protocol."""
        store = InMemoryCheckpointStore()
        key = ResourceKey[Structural]("structural")

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            return x

        def build(resource: Structural) -> Any:
            return (
                _ff()
                .create(state={})
                .with_checkpoint_store(store, NAME)
                .with_checkpointer()
                .with_resource(key, resource)
                .call(step)
            )

        await build(Structural()).run(0)
        fresh = Structural()
        await build(fresh).run(0, resume="latest")
        assert fresh.restored == [{"n": 1}]

    def test_a_subclass_that_leaves_a_method_out_cannot_be_made(self) -> None:
        class Half(Resource):
            def snapshot(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError, match="restore"):
            Half()  # type: ignore[abstract]


class TestCostIsAResource:
    """The cost API is sugar over the COST resource."""

    async def test_ctx_cost_is_the_cost_resource_and_a_budget_its_child(self) -> None:
        root = CostTracker(make_test_logger(), PricingConfig())
        seen: list[tuple[Any, Any]] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            seen.append((ctx.cost, ctx.resource(COST)))
            return x

        sub = _ff().create().with_budget(2.0).call(step)
        await _ff().create().with_cost_tracker(root).call(step).then(sub).run(1)
        (top, top_key), (child, child_key) = seen
        assert top is root and top_key is root
        assert child is child_key and child.parent is root and child.budget == 2.0

    async def test_dispatch_cost_replaces_or_removes_the_tracker(self) -> None:
        own = CostTracker(make_test_logger(), PricingConfig())
        other = CostTracker(make_test_logger(), PricingConfig())
        seen: list[Any] = []

        @verb
        async def step(ctx: Context[Any]) -> None:
            seen.append(ctx.cost)

        flow = _ff().create().with_cost_tracker(own)
        flow.register(step)
        await flow.dispatch("step")
        await flow.dispatch("step", cost=other)
        await flow.dispatch("step", cost=None)
        assert seen == [own, other, None]


class TestValidation:
    def test_with_resource_takes_a_key(self) -> None:
        with pytest.raises(TypeError, match="ResourceKey"):
            _ff().create().with_resource("counter", Counter())  # type: ignore[arg-type]

    def test_with_resource_needs_a_value_or_child_arguments(self) -> None:
        with pytest.raises(ValueError, match="needs a resource"):
            _ff().create().with_resource(COUNTER)

    def test_a_value_implements_snapshot_and_restore(self) -> None:
        with pytest.raises(TypeError, match="snapshot"):
            _ff().create().with_resource(COUNTER, object())

    def test_child_arguments_need_a_child(self) -> None:
        with pytest.raises(TypeError, match="no child"):
            _ff().create().with_resource(COUNTER, Plain(), per=1)

    @pytest.mark.parametrize("bad", ["", 1])
    def test_a_key_has_a_name(self, bad: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            ResourceKey(bad)

    async def test_two_keys_with_one_name_raise_at_run_start(self) -> None:
        twin = ResourceKey[Counter]("counter")
        sub = _ff().create().with_resource(twin, Counter()).call(tick)
        flow = _ff().create().with_resource(COUNTER, Counter()).call(sub)
        with pytest.raises(ValueError, match="'counter'"):
            await flow.run(0)

    async def test_a_per_run_child_with_none_above_raises_at_run_start(self) -> None:
        ran: list[int] = []

        @verb
        async def step(ctx: Context[Any], x: int) -> int:
            ran.append(x)
            return x

        flow = (
            _ff()
            .create()
            .call(step)
            .map(lambda b: b.with_resource(COUNTER, per=1).call(step), items=lambda *_: [1])
        )
        with pytest.raises(RuntimeError, match="none is in scope"):
            await flow.run(0)
        assert ran == []

    async def test_a_per_run_child_of_a_resource_without_child_raises(self) -> None:
        plain = ResourceKey[Plain]("plain")
        sub = _ff().create().with_resource(plain, per=1).call(tick)
        flow = _ff().create().with_resource(plain, Plain()).call(sub)
        with pytest.raises(TypeError, match="no child"):
            await flow.run(0)


class TestFluentNames:
    async def test_a_factory_s_resource_method_reaches_its_flows_and_their_bodies(self) -> None:
        root = Counter()
        seen: list[Counter] = []

        @verb
        async def step(ctx: Context[Any], n: int) -> int:
            seen.append(ctx.resource(COUNTER))
            return n

        ff = _ff().with_resource_method("with_counter", COUNTER)
        flow: Any = ff.create()
        flow.with_counter(root).map(
            lambda b: b.with_counter(per="item").call(step), items=lambda *_: [1, 2]
        ).iterate(lambda b: b.with_counter(per="pass").call(step), max_iters=1)
        assert await flow.run() == [1, 2]
        assert [c.args for c in seen] == [{"per": "item"}, {"per": "item"}, {"per": "pass"}]
        assert all(c.parent is root for c in seen)

    def test_flow_and_other_factories_are_unchanged(self) -> None:
        ff = _ff()
        ff.with_resource_method("with_counter", COUNTER).create()
        assert not hasattr(Flow, "with_counter")
        assert not hasattr(ff.create(), "with_counter")

    def test_a_name_flow_has_raises(self) -> None:
        with pytest.raises(ValueError, match="already an attribute"):
            _ff().with_resource_method("map", COUNTER)

    def test_a_name_that_is_not_an_identifier_raises(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            _ff().with_resource_method("with counter", COUNTER)

    def test_one_name_for_two_keys_raises(self) -> None:
        ff = _ff().with_resource_method("with_counter", COUNTER)
        assert ff.with_resource_method("with_counter", COUNTER) is not ff  # same key: fine
        with pytest.raises(ValueError, match="already maps"):
            ff.with_resource_method("with_counter", ResourceKey[Counter]("counter2"))

    async def test_a_subclass_s_resource_method_with_flow_class(self) -> None:
        class MyFlow(Flow):
            with_counter = resource_method(COUNTER)

        root = Counter()
        bodies: list[Flow] = []

        def body(b: MyFlow) -> None:
            bodies.append(b)
            b.with_counter(per="item").call(tick)

        ff = Factory(make_test_logger(), flow_class=MyFlow)
        flow = ff.create().with_counter(root).map(body, items=lambda *_: [1, 2])
        assert isinstance(flow, MyFlow)
        assert await flow.run() == [1, 2]
        assert all(type(b) is MyFlow for b in bodies)
        assert root.count == 2

    def test_the_factory_s_other_methods_keep_its_class_and_methods(self) -> None:
        class MyFlow(Flow):
            pass

        ff = (
            Factory(make_test_logger(), flow_class=MyFlow)
            .with_resource_method("with_counter", COUNTER)
            .with_halt(asyncio.Event())
        )
        flow = ff.create()
        assert isinstance(flow, MyFlow) and hasattr(flow, "with_counter")
