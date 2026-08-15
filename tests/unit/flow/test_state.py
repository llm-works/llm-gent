"""Two-channel state — scoped ``ctx.state`` + run-wide ``ctx.global_state``."""

from __future__ import annotations

import asyncio

import pytest

from llm_gent.flow import Failure, Flow, verb

from .conftest import ROLE_A, ROLE_B, StubFactory, make_test_logger


# -----------------------------------------------------------------------------
# ctx.global_state — defaults, propagation, subflow inheritance
# -----------------------------------------------------------------------------


class TestGlobalStateDefault:
    """``ctx.global_state`` defaults to an empty dict at the top-level."""

    async def test_default_is_empty_dict(self) -> None:
        """Top-level Flow.run with no global_state kwarg → verbs see {}."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.global_state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run()

        assert seen == [{}]

    async def test_default_isolated_between_runs(self) -> None:
        """Successive top-level runs receive independent default containers."""
        captured: list[dict] = []

        @verb(role=ROLE_A)
        async def touch(ctx) -> None:
            ctx.global_state["mark"] = id(ctx.global_state)
            captured.append(ctx.global_state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(touch)

        await flow.run()
        await flow.run()

        assert len(captured) == 2
        assert captured[0] is not captured[1]

    async def test_explicit_dict_overrides_default(self) -> None:
        """Callers may pass any dict — the framework wires it verbatim."""
        supplied = {"budget": 100}
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.global_state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run(global_state=supplied)

        assert seen[0] is supplied

    async def test_explicit_non_dict_overrides_default(self) -> None:
        """The default is dict-shaped, but callers can pass any object."""

        class Bag:
            budget = 42

        bag = Bag()
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.global_state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run(global_state=bag)

        assert seen[0] is bag

    async def test_explicit_none_honored(self) -> None:
        """Passing ``None`` explicitly is honored (caller opts out of the default)."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.global_state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run(global_state=None)

        assert seen == [None]

    async def test_dispatch_provides_empty_dict(self) -> None:
        """Panel/dispatch entrypoints build a fresh empty-dict global_state."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.global_state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory())
        flow.register(check)
        await flow.dispatch("check")

        assert seen == [{}]


class TestGlobalStateInheritance:
    """Subflows inherit the outermost ``global_state`` container by identity."""

    async def test_subflow_inherits_container(self) -> None:
        """The same object surfaces on ``ctx.global_state`` inside a subflow."""
        captured: list[object] = []

        @verb(role=ROLE_A)
        async def top_verb(ctx) -> None:
            captured.append(("top", ctx.global_state))

        @verb(role=ROLE_B)
        async def inner_verb(ctx, _prev) -> None:
            captured.append(("inner", ctx.global_state))

        lg = make_test_logger()
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory()).call(top_verb).call(inner)

        outer = {"budget": 500}
        await top.run(global_state=outer)

        assert captured[0] == ("top", outer)
        assert captured[1] == ("inner", outer)
        assert captured[0][1] is captured[1][1] is outer

    async def test_branch_body_inherits(self) -> None:
        """Branch bodies see the outermost ``global_state``."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def hit(ctx, _prev) -> str:
            seen.append(ctx.global_state)
            return "ok"

        lg = make_test_logger()
        top = Flow(lg, "top", factory=StubFactory()).branch(
            when=lambda _prev, _ctx: True, then=lambda f: f.call(hit)
        )

        outer = {"key": "value"}
        await top.run("input", global_state=outer)

        assert seen == [outer]

    async def test_loop_body_inherits(self) -> None:
        """Loop iterations see the outermost ``global_state``."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def hit(ctx, _prev) -> str:
            seen.append(ctx.global_state)
            return "ok"

        lg = make_test_logger()
        top = Flow(lg, "top", factory=StubFactory()).loop(lambda f: f.call(hit), max_iters=3)

        outer = {"key": "value"}
        await top.run("input", global_state=outer)

        assert len(seen) == 3
        assert all(s is outer for s in seen)

    async def test_map_body_inherits(self) -> None:
        """Every map item sees the outermost ``global_state``."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def hit(ctx, item) -> str:
            seen.append(ctx.global_state)
            return item

        lg = make_test_logger()
        top = Flow(lg, "top", factory=StubFactory()).map(lambda f: f.call(hit))

        outer = {"key": "value"}
        await top.run([1, 2, 3], global_state=outer)

        assert len(seen) == 3
        assert all(s is outer for s in seen)


# -----------------------------------------------------------------------------
# ctx.state — shared-by-default, opt-in projection
# -----------------------------------------------------------------------------


class TestScopedStateSharing:
    """``ctx.state`` shares the parent reference into a subflow by default."""

    async def test_subflow_mutations_visible_in_parent(self) -> None:
        """No projection → parent and subflow reference the same object."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state["from_inner"] = True
            return "ok"

        lg = make_test_logger()
        parent_state = {"from_inner": False, "root": "yes"}
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).call(inner)

        await top.run()

        assert parent_state["from_inner"] is True

    async def test_loop_default_shares_state(self) -> None:
        """Loop body without ``state=`` mutates the parent's state directly."""

        @verb(role=ROLE_A)
        async def bump(ctx, prev) -> int:
            ctx.state["count"] = ctx.state.get("count", 0) + 1
            return prev

        lg = make_test_logger()
        parent_state: dict = {}
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).loop(
            lambda f: f.call(bump), max_iters=4
        )

        await top.run("seed")

        assert parent_state == {"count": 4}


class TestScopedStateProjection:
    """``state=`` on ``.call``/``.loop``/``.map`` isolates the child from the parent."""

    async def test_call_state_isolates_subflow(self) -> None:
        """A projected child state does not leak into the parent."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state["only_child"] = True
            return "ok"

        lg = make_test_logger()
        parent_state = {"parent_key": "kept"}
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).call(
            inner, state=lambda _parent: {}
        )

        await top.run()

        assert parent_state == {"parent_key": "kept"}

    async def test_call_state_and_merge_folds_back(self) -> None:
        """``merge=`` runs after the subflow returns successfully."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state["scratch"] = 7
            return "ok"

        def merge(parent: dict, child: dict) -> None:
            parent["fold"] = child["scratch"]

        lg = make_test_logger()
        parent_state: dict = {}
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).call(
            inner, state=lambda _parent: {}, merge=merge
        )

        await top.run()

        assert parent_state == {"fold": 7}

    async def test_call_merge_skipped_on_failure(self) -> None:
        """A subflow that raises → merge does not fire; the parent stays clean."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state["scratch"] = 7
            raise RuntimeError("boom")

        merged: list[bool] = []

        def merge(parent: dict, child: dict) -> None:
            merged.append(True)
            parent["fold"] = child["scratch"]

        lg = make_test_logger()
        parent_state: dict = {}
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).call(
            inner, state=lambda _parent: {}, merge=merge
        )

        with pytest.raises(RuntimeError, match="boom"):
            await top.run()

        assert merged == []
        assert parent_state == {}

    async def test_call_projection_receives_parent_state(self) -> None:
        """``state(parent)`` sees the parent's active state at build time."""
        received: list[object] = []

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            return "ok"

        def project(parent: dict) -> dict:
            received.append(parent)
            return {"copied_from": parent.get("root")}

        lg = make_test_logger()
        parent_state = {"root": "R"}
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).call(inner, state=project)

        await top.run()

        assert received == [parent_state]

    async def test_loop_state_projected_once_persists_across_iters(self) -> None:
        """``.loop(state=)`` projects once — iterations share the child state."""
        snapshots: list[dict] = []

        @verb(role=ROLE_A)
        async def bump(ctx, prev) -> int:
            ctx.state["count"] = ctx.state.get("count", 0) + 1
            snapshots.append(dict(ctx.state))
            return prev

        def merge(parent: dict, child: dict) -> None:
            parent["final_count"] = child["count"]

        lg = make_test_logger()
        parent_state: dict = {"unrelated": True}
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).loop(
            lambda f: f.call(bump),
            max_iters=3,
            state=lambda _parent: {},
            merge=merge,
        )

        await top.run("seed")

        assert snapshots == [{"count": 1}, {"count": 2}, {"count": 3}]
        assert parent_state == {"unrelated": True, "final_count": 3}

    async def test_loop_merge_skipped_when_body_raises(self) -> None:
        """A loop body that raises past ``rescue`` → merge does not fire."""
        merged: list[bool] = []

        @verb(role=ROLE_A)
        async def bump(ctx, prev) -> int:
            ctx.state["count"] = ctx.state.get("count", 0) + 1
            if ctx.state["count"] == 2:
                raise RuntimeError("boom")
            return prev

        def merge(_parent: dict, _child: dict) -> None:
            merged.append(True)

        lg = make_test_logger()
        parent_state: dict = {}
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).loop(
            lambda f: f.call(bump),
            max_iters=5,
            state=lambda _parent: {},
            merge=merge,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await top.run("seed")

        assert merged == []

    async def test_map_state_projected_per_item(self) -> None:
        """``.map(state=)`` builds an isolated state per item — no cross-contamination."""
        seen: list[tuple[int, dict]] = []

        @verb(role=ROLE_A)
        async def record(ctx, item) -> int:
            ctx.state["item"] = item
            await asyncio.sleep(0)
            seen.append((item, dict(ctx.state)))
            return item

        lg = make_test_logger()
        parent_state = {"parent_key": "kept"}
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).map(
            lambda f: f.call(record),
            state=lambda _parent: {},
        )

        await top.run([1, 2, 3])

        seen.sort(key=lambda p: p[0])
        assert seen == [(1, {"item": 1}), (2, {"item": 2}), (3, {"item": 3})]
        assert parent_state == {"parent_key": "kept"}

    async def test_map_merge_runs_only_for_successful_items(self) -> None:
        """``strict=False`` map: failed items' merges are skipped."""

        @verb(role=ROLE_A)
        async def maybe_fail(ctx, item) -> int:
            ctx.state["item"] = item
            if item == 2:
                raise RuntimeError(f"bad {item}")
            return item

        def merge(parent: dict, child: dict) -> None:
            parent.setdefault("merged", []).append(child["item"])

        lg = make_test_logger()
        parent_state: dict = {}
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).map(
            lambda f: f.call(maybe_fail),
            state=lambda _parent: {},
            merge=merge,
            strict=False,
        )

        results = await top.run([1, 2, 3])

        assert results[0] == 1
        assert isinstance(results[1], Failure)
        assert results[2] == 3
        assert sorted(parent_state["merged"]) == [1, 3]


class TestStateProjectionAsync:
    """Async state/merge callables are awaited transparently."""

    async def test_async_state_and_merge(self) -> None:
        """Both projection and merge accept coroutines."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state["hit"] = True
            return "ok"

        async def async_state(_parent: dict) -> dict:
            await asyncio.sleep(0)
            return {"from_async": True}

        async def async_merge(parent: dict, child: dict) -> None:
            await asyncio.sleep(0)
            parent["carried"] = child["hit"]

        lg = make_test_logger()
        parent_state: dict = {}
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).call(
            inner, state=async_state, merge=async_merge
        )

        await top.run()

        assert parent_state == {"carried": True}


# -----------------------------------------------------------------------------
# Build-time validation
# -----------------------------------------------------------------------------


class TestStateKwargValidation:
    """State/merge kwargs are validated at build time, not run time."""

    def test_call_merge_without_state_rejected(self) -> None:
        """``merge=`` on ``.call`` without ``state=`` fails eagerly."""

        @verb(role=ROLE_A)
        async def _v(ctx) -> None: ...

        lg = make_test_logger()
        inner = Flow(lg, "inner").call(_v)
        flow = Flow(lg, "top", factory=StubFactory())

        with pytest.raises(ValueError, match="requires state="):
            flow.call(inner, merge=lambda p, c: None)  # noqa: ARG005

    def test_loop_merge_without_state_rejected(self) -> None:
        """``.loop(merge=...)`` without ``state=`` fails eagerly."""

        @verb(role=ROLE_A)
        async def _v(ctx, _prev) -> None: ...

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory())

        with pytest.raises(ValueError, match="requires state="):
            flow.loop(
                lambda f: f.call(_v),
                max_iters=1,
                merge=lambda p, c: None,  # noqa: ARG005
            )

    def test_map_merge_without_state_rejected(self) -> None:
        """``.map(merge=...)`` without ``state=`` fails eagerly."""

        @verb(role=ROLE_A)
        async def _v(ctx, _item) -> None: ...

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory())

        with pytest.raises(ValueError, match="requires state="):
            flow.map(
                lambda f: f.call(_v),
                merge=lambda p, c: None,  # noqa: ARG005
            )

    def test_call_state_on_verb_target_rejected(self) -> None:
        """``state=`` on a verb target has no meaning — fail eagerly."""

        @verb(role=ROLE_A)
        async def leaf(ctx) -> None: ...

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory())

        with pytest.raises(TypeError, match="only valid when target is a Flow"):
            flow.call(leaf, state=lambda _p: {})

    def test_call_merge_on_verb_target_rejected(self) -> None:
        """``merge=`` on a verb target has no meaning — fail eagerly."""

        @verb(role=ROLE_A)
        async def leaf(ctx) -> None: ...

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory())

        with pytest.raises(TypeError, match="only valid when target is a Flow"):
            flow.call(leaf, merge=lambda p, c: None)  # noqa: ARG005
