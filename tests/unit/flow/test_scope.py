# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The scope fold: how each field of a :class:`Scope` derives as the walk enters flows.

``scope(node) = root ⊕ decl(f₁) ⊕ … ⊕ decl(fₙ)``. The fold tests enumerate
every combination of declarations over a few levels: the result is the
field's rule applied along the path, at any depth.
"""

from __future__ import annotations

import asyncio
import itertools
from functools import reduce
from typing import Any

import pytest

from llm_gent.core import Environment
from llm_gent.flow import State
from llm_gent.flow._node_id import _descend_context
from llm_gent.flow._shortcut import Shortcut
from llm_gent.flow.checkpoint import CheckpointPolicy
from llm_gent.flow.decl import Decl
from llm_gent.flow.hooks import Hook, Hooks, Where
from llm_gent.flow.nodes import Checkpointer
from llm_gent.flow.run import Position, Scope
from llm_gent.flow.run._run import Run
from llm_gent.flow.run._shortcut import ShortcutRun
from llm_gent.flow.run.scope import _Frame

from .conftest import make_ff, make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()
WHERE = Where()
SIGNALS = {"a": asyncio.Event(), "b": asyncio.Event()}


def _enter(scope: Scope, decl: Decl) -> Scope:
    """One step of the fold: the scope of a flow declaring ``decl``, entered at ``scope``."""
    return scope.child(decl, resources=scope.resources, where=WHERE, signals=SIGNALS, lg=LG)


def _fold(decls: tuple[Decl, ...]) -> Scope:
    return reduce(_enter, decls, Scope(state=State(data={})))


class TestOverride:
    """Checkpointer and checkpoint policy: the innermost declaration wins, unset inherits."""

    _CHECKPOINTERS = (None, Checkpointer("outer"), Checkpointer("inner"))
    _POLICIES = (None, CheckpointPolicy(on_iterate=True), CheckpointPolicy())

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_checkpointer_is_the_innermost_declared(self, depth: int) -> None:
        for path in itertools.product(self._CHECKPOINTERS, repeat=depth):
            scope = _fold(tuple(Decl(checkpointer=c) for c in path))
            declared = [c for c in path if c is not None]
            assert scope.checkpointer is (declared[-1] if declared else None), path

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_policy_is_the_innermost_declared_else_the_default(self, depth: int) -> None:
        for path in itertools.product(self._POLICIES, repeat=depth):
            scope = _fold(tuple(Decl(policy=p) for p in path))
            declared = [p for p in path if p is not None]
            if declared:
                assert scope.policy is declared[-1], path
            else:
                assert scope.policy == CheckpointPolicy(), path


class TestAccumulate:
    """Shortcut regions: each flow's region adds to the enclosing ones, outermost first."""

    _SHORTCUTS = (None, Shortcut("a", drain=False), Shortcut("b", drain=True))

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_regions_are_the_declared_ones_in_path_order(self, depth: int) -> None:
        for path in itertools.product(self._SHORTCUTS, repeat=depth):
            scope = _fold(tuple(Decl(shortcut=s) for s in path))
            expected = tuple(
                ShortcutRun(SIGNALS[s.signal], drain=s.drain, left=False)
                for s in path
                if s is not None
            )
            assert scope.shortcuts == expected, path


class _Tagged(Hook):
    """A hook whose child records the place it was made for."""

    def __init__(self, name: str, places: tuple[Where, ...] = ()) -> None:
        self.name = name
        self.places = places

    def child(self, where: Where) -> Hook:
        return _Tagged(self.name, (*self.places, where))


def _names(hooks: Hooks | None) -> list[str] | None:
    return None if hooks is None else [h.name for h in hooks.hooks]  # type: ignore[attr-defined]


class TestDeriveHooks:
    """Hooks: the enclosing hooks' children at the flow's place, then the flow's own."""

    def test_no_hooks_anywhere_is_none(self) -> None:
        assert _fold((Decl(), Decl())).hooks is None

    def test_inner_hooks_extend_the_inherited_children(self) -> None:
        scope = _fold((Decl(hooks=(_Tagged("outer"),)), Decl(hooks=(_Tagged("inner"),))))
        assert _names(scope.hooks) == ["outer", "inner"]

    def test_inherited_hooks_are_children_made_for_each_flow_entered(self) -> None:
        scope = _fold((Decl(hooks=(_Tagged("outer"),)), Decl(), Decl()))
        outer = scope.hooks.hooks[0]  # type: ignore[union-attr]
        assert outer.places == (WHERE, WHERE)  # type: ignore[attr-defined]

    def test_replace_drops_the_inherited_hooks(self) -> None:
        scope = _fold(
            (Decl(hooks=(_Tagged("outer"),)), Decl(hooks=(_Tagged("inner"),), hooks_replace=True))
        )
        assert _names(scope.hooks) == ["inner"]

    def test_replace_without_hooks_of_its_own_leaves_none(self) -> None:
        scope = _fold((Decl(hooks=(_Tagged("outer"),)), Decl(hooks_replace=True)))
        assert scope.hooks is None

    def test_hooks_of_a_flow_under_none_are_its_own(self) -> None:
        scope = _fold((Decl(), Decl(hooks=(_Tagged("inner"),))))
        assert _names(scope.hooks) == ["inner"]


class TestChild:
    """What ``Scope.child`` takes as given rather than deriving."""

    def test_state_carries_over(self) -> None:
        state = State(data={"k": 1})
        scope = Scope(state=state)
        assert _enter(scope, Decl()).state is state

    def test_resources_are_the_ones_given(self) -> None:
        given: dict[Any, Any] = {}
        scope = Scope(state=State(data={})).child(
            Decl(), resources=given, where=WHERE, signals=SIGNALS, lg=LG
        )
        assert scope.resources is given


class TestPosition:
    """A node moves the position one level down; the walk computes it, nobody gives it."""

    def test_descend_extends_the_ancestor_chain_and_hashes_the_boundary(self) -> None:
        root = Position(activity_id="run-1")
        child = root.descend("n1", "call", ("n", "n1"))
        grandchild = child.descend("n2", "body", ("n", "n1", "n", "n2"))
        assert grandchild.ancestor_chain == ("n1", "n2")
        assert grandchild.chain_context == _descend_context("n2", "body")
        assert grandchild.path == ("n", "n1", "n", "n2")
        assert grandchild.activity_id == "run-1"


def _frame() -> _Frame:
    flow = make_ff().create("top")
    run = Run(lg=LG, env=Environment(), flow=flow, signals={}, checkpoint_ctx=None)
    return _Frame(run=run, lg=LG, scope=Scope(state=State(data={"root": True})))


class TestNodeStep:
    """The node's part of a descent: position, state, and the shortcuts a map item leaves."""

    def test_descend_hands_the_node_state_at_the_new_position(self) -> None:
        frame = _frame()
        state = State(data={"child": True})
        child = frame.descend("n1", "call", frame.owner_path("n1"), state)
        assert child.scope.state is state
        assert child.position.path == ("n", "n1")
        assert child.position.ancestor_chain == ("n1",)
        assert child.run is frame.run

    def test_descend_keeps_the_shortcuts_unless_the_node_changes_them(self) -> None:
        region = ShortcutRun(SIGNALS["a"], drain=True, left=False)
        frame = _frame()
        frame = frame.descend("n0", "call", (), frame.scope.state, shortcuts=(region,))
        kept = frame.descend("n1", "call", (), frame.scope.state)
        left = frame.descend("n1", "map:k", (), frame.scope.state, shortcuts=())
        assert kept.scope.shortcuts == (region,)
        assert left.scope.shortcuts == ()

    def test_at_activity_moves_only_the_activity(self) -> None:
        frame = _frame()
        moved = frame.at_activity("step-7")
        assert moved.position.activity_id == "step-7"
        assert moved.position.path == frame.position.path
        assert moved.scope is frame.scope
