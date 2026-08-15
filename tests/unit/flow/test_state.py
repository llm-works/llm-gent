"""Unified ``ctx.state`` — :class:`State` wrapper, scoped projection, root navigation."""

from __future__ import annotations

import asyncio

import pytest

from llm_gent.flow import Failure, Flow, State, verb

from .conftest import ROLE_A, ROLE_B, StubFactory, make_test_logger


# -----------------------------------------------------------------------------
# ctx.state wrapper — top-level payload defaults and overrides
# -----------------------------------------------------------------------------


class TestStateWrapper:
    """``ctx.state`` is always a :class:`State`; the payload is on ``.data``."""

    async def test_default_payload_is_empty_dict(self) -> None:
        """Top-level Flow.run with no state kwarg → ctx.state.data == {}."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run()

        assert len(seen) == 1
        assert isinstance(seen[0], State)
        assert seen[0].data == {}
        assert seen[0].is_root is True

    async def test_default_isolated_between_runs(self) -> None:
        """Successive top-level runs receive independent default payloads."""
        captured: list[dict] = []

        @verb(role=ROLE_A)
        async def touch(ctx) -> None:
            ctx.state.data["mark"] = id(ctx.state.data)
            captured.append(ctx.state.data)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(touch)

        await flow.run()
        await flow.run()

        assert len(captured) == 2
        assert captured[0] is not captured[1]

    async def test_explicit_dict_overrides_default(self) -> None:
        """Callers may pass any dict — the framework wraps it verbatim."""
        supplied = {"budget": 100}
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.state.data)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run(state=supplied)

        assert seen[0] is supplied

    async def test_explicit_non_dict_overrides_default(self) -> None:
        """The default is dict-shaped, but callers can pass any object as payload."""

        class Bag:
            budget = 42

        bag = Bag()
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.state.data)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run(state=bag)

        assert seen[0] is bag

    async def test_explicit_none_honored(self) -> None:
        """Passing ``state=None`` explicitly is honored — payload is None."""
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.state.data)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run(state=None)

        assert seen == [None]

    async def test_prewrapped_state_passed_through(self) -> None:
        """A caller-constructed :class:`State` is used verbatim (not re-wrapped)."""
        outer = State(data={"pre": "wrapped"})
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run(state=outer)

        assert seen[0] is outer

    async def test_dispatch_wraps_construction_state(self) -> None:
        """``Flow.dispatch`` also wraps its state as a top-level :class:`State`."""
        user_state = {"counter": 0}
        seen: list[object] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            seen.append(ctx.state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory(), state=user_state)
        flow.register(check)
        await flow.dispatch("check")

        assert isinstance(seen[0], State)
        assert seen[0].data is user_state
        assert seen[0].is_root is True


# -----------------------------------------------------------------------------
# ctx.state.root() — verbs reach run-wide payload from any depth
# -----------------------------------------------------------------------------


class TestStateNavigation:
    """Subflows can reach the outermost :class:`State` via :meth:`State.root`."""

    async def test_subflow_reaches_root_from_projection(self) -> None:
        """A projected subflow's ``ctx.state.root().data`` is the outermost payload."""
        captured: list[tuple[str, dict]] = []

        @verb(role=ROLE_A)
        async def top_verb(ctx) -> None:
            captured.append(("top", ctx.state.root().data))

        @verb(role=ROLE_B)
        async def inner_verb(ctx, _prev) -> None:
            captured.append(("inner", ctx.state.root().data))

        lg = make_test_logger()
        inner = Flow(lg, "inner").call(inner_verb)
        top = (
            Flow(lg, "top", factory=StubFactory())
            .call(top_verb)
            .call(inner, state=lambda _p: {"scoped": True})
        )

        outer = {"budget": 500}
        await top.run(state=outer)

        assert captured[0] == ("top", outer)
        assert captured[1] == ("inner", outer)
        assert captured[0][1] is captured[1][1] is outer

    async def test_subflow_without_projection_shares_root(self) -> None:
        """Without a state= projection, the subflow's ``ctx.state`` IS the parent's."""
        captured: list[State] = []

        @verb(role=ROLE_A)
        async def top_verb(ctx) -> None:
            captured.append(ctx.state)

        @verb(role=ROLE_B)
        async def inner_verb(ctx, _prev) -> None:
            captured.append(ctx.state)

        lg = make_test_logger()
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory()).call(top_verb).call(inner)

        outer = {"budget": 500}
        await top.run(state=outer)

        assert captured[0] is captured[1]
        assert captured[0].is_root is True

    async def test_branch_body_reaches_root(self) -> None:
        """Branch bodies see the outermost payload via ``.root().data``."""
        seen: list[dict] = []

        @verb(role=ROLE_A)
        async def hit(ctx, _prev) -> str:
            seen.append(ctx.state.root().data)
            return "ok"

        lg = make_test_logger()
        top = Flow(lg, "top", factory=StubFactory()).branch(
            when=lambda _prev, _ctx: True, then=lambda f: f.call(hit)
        )

        outer = {"key": "value"}
        await top.run("input", state=outer)

        assert seen == [outer]

    async def test_loop_iterations_reach_root(self) -> None:
        """Loop iterations under state= projection still see the outermost via .root()."""
        seen: list[dict] = []

        @verb(role=ROLE_A)
        async def hit(ctx, _prev) -> str:
            seen.append(ctx.state.root().data)
            return "ok"

        lg = make_test_logger()
        top = Flow(lg, "top", factory=StubFactory()).loop(
            lambda f: f.call(hit), max_iters=3, state=lambda _p: {}
        )

        outer = {"key": "value"}
        await top.run("input", state=outer)

        assert len(seen) == 3
        assert all(s is outer for s in seen)

    async def test_map_items_reach_root(self) -> None:
        """Every map item sees the outermost payload via .root()."""
        seen: list[dict] = []

        @verb(role=ROLE_A)
        async def hit(ctx, item) -> int:
            seen.append(ctx.state.root().data)
            return item

        lg = make_test_logger()
        top = Flow(lg, "top", factory=StubFactory()).map(lambda f: f.call(hit), state=lambda _p: {})

        outer = {"key": "value"}
        await top.run([1, 2, 3], state=outer)

        assert len(seen) == 3
        assert all(s is outer for s in seen)

    async def test_root_of_root_is_self(self) -> None:
        """At the outermost scope, ``root()`` returns the same object."""
        captured: list[State] = []

        @verb(role=ROLE_A)
        async def check(ctx) -> None:
            captured.append(ctx.state)

        lg = make_test_logger()
        flow = Flow(lg, "top", factory=StubFactory()).call(check)
        await flow.run()

        s = captured[0]
        assert s.root() is s
        assert s.is_root is True


# -----------------------------------------------------------------------------
# Scoped state — shared-by-default, opt-in projection
# -----------------------------------------------------------------------------


class TestScopedStateSharing:
    """Without ``state=``, the subflow shares the parent's :class:`State` by reference."""

    async def test_subflow_mutations_visible_in_parent(self) -> None:
        """No projection → parent and subflow reference the same payload."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state.data["from_inner"] = True
            return "ok"

        lg = make_test_logger()
        parent_state = {"from_inner": False, "root": "yes"}
        inner = Flow(lg, "inner").call(inner_verb)
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).call(inner)

        await top.run()

        assert parent_state["from_inner"] is True

    async def test_loop_default_shares_state(self) -> None:
        """Loop body without ``state=`` mutates the parent's payload directly."""

        @verb(role=ROLE_A)
        async def bump(ctx, prev) -> int:
            ctx.state.data["count"] = ctx.state.data.get("count", 0) + 1
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
        """A projected child payload does not leak into the parent."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state.data["only_child"] = True
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
            ctx.state.data["scratch"] = 7
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
            ctx.state.data["scratch"] = 7
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

    async def test_call_projection_receives_parent_payload(self) -> None:
        """``state(parent_payload)`` sees the parent's active payload (unwrapped)."""
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
        """``.loop(state=)`` projects once — iterations share the child payload."""
        snapshots: list[dict] = []

        @verb(role=ROLE_A)
        async def bump(ctx, prev) -> int:
            ctx.state.data["count"] = ctx.state.data.get("count", 0) + 1
            snapshots.append(dict(ctx.state.data))
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
            ctx.state.data["count"] = ctx.state.data.get("count", 0) + 1
            if ctx.state.data["count"] == 2:
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

    async def test_loop_until_sees_projected_child_state(self) -> None:
        """``until`` predicate sees the projected child payload, not the parent."""
        counts: list[int] = []

        @verb(role=ROLE_A)
        async def bump(ctx, prev) -> int:
            ctx.state.data["count"] = ctx.state.data.get("count", 0) + 1
            counts.append(ctx.state.data["count"])
            return prev

        lg = make_test_logger()
        parent_state: dict = {"count": 999}
        top = Flow(lg, "top", factory=StubFactory(), state=parent_state).loop(
            lambda f: f.call(bump),
            until=lambda ctx: ctx.state.data.get("count", 0) >= 3,
            state=lambda _parent: {},
        )

        await top.run("seed")

        assert counts == [1, 2, 3]
        assert parent_state == {"count": 999}

    async def test_map_state_projected_per_item(self) -> None:
        """``.map(state=)`` builds an isolated payload per item — no cross-contamination."""
        seen: list[tuple[int, dict]] = []

        @verb(role=ROLE_A)
        async def record(ctx, item) -> int:
            ctx.state.data["item"] = item
            await asyncio.sleep(0)
            seen.append((item, dict(ctx.state.data)))
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
            ctx.state.data["item"] = item
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
        """Both projection and merge accept coroutines returning payloads."""

        @verb(role=ROLE_A)
        async def inner_verb(ctx) -> str:
            ctx.state.data["hit"] = True
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
