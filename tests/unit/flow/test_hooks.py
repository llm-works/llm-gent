# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :mod:`llm_gent.flow.hooks`: verdicts, order, failure policy, attachment."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.cas import Blob, canonical_json
from appinfra.log import Logger

from llm_gent.flow.hooks import (
    ALLOW,
    Decision,
    Hook,
    Hooks,
    Pause,
    RunInfo,
    StepEvent,
    Where,
)


def _event(**fields: Any) -> StepEvent:
    """A ``before_step`` event at no particular place."""
    return StepEvent(
        name="before_step",
        ts=datetime(2026, 1, 1, tzinfo=UTC),
        run=RunInfo(run_id="r", flow_name=None, structure_hash="h", repo=None),
        activity_id="a",
        **fields,
    )


def _lg() -> MagicMock:
    return MagicMock(spec=Logger)


class _Verdict(Hook):
    """Returns a fixed verdict from ``before_step`` and records that it was called."""

    def __init__(self, verdict: Any, calls: list[str] | None = None, name: str = "") -> None:
        self.verdict, self.calls, self.name = verdict, calls, name

    def before_step(self, event: StepEvent) -> Any:
        if self.calls is not None:
            self.calls.append(self.name)
        return self.verdict


class _Records(Hook):
    """Appends its name to ``calls`` on ``before_step`` and ``after_step``."""

    def __init__(self, name: str, calls: list[str]) -> None:
        self.name, self.calls = name, calls

    def before_step(self, event: StepEvent) -> None:
        self.calls.append(f"before {self.name}")

    def after_step(self, event: StepEvent) -> None:
        self.calls.append(f"after {self.name}")


class TestVerdicts:
    @pytest.mark.parametrize("verdict", [True, None])
    async def test_true_and_none_allow(self, verdict: Any) -> None:
        assert await Hooks(_lg(), [_Verdict(verdict)]).before("before_step", _event()) == ALLOW

    async def test_false_denies_naming_the_hook(self) -> None:
        decision = await Hooks(_lg(), [_Verdict(False)]).before("before_step", _event())
        assert decision == Decision("deny", "_Verdict returned False", "_Verdict")

    async def test_a_string_denies_with_that_reason(self) -> None:
        decision = await Hooks(_lg(), [_Verdict("needs approval")]).before("before_step", _event())
        assert decision == Decision("deny", "needs approval", "_Verdict")

    async def test_pause_pauses(self) -> None:
        decision = await Hooks(_lg(), [_Verdict(Pause("ask a human"))]).before(
            "before_step", _event()
        )
        assert decision == Decision("pause", "ask a human", "_Verdict")

    async def test_another_return_value_denies(self) -> None:
        """A verdict of the wrong type fails closed rather than allowing."""
        decision = await Hooks(_lg(), [_Verdict(42)]).before("before_step", _event())
        assert decision.verdict == "deny"
        assert "42" in decision.reason

    async def test_the_first_verdict_other_than_allow_decides(self) -> None:
        calls: list[str] = []
        hooks = Hooks(
            _lg(),
            [_Verdict(True, calls, "a"), _Verdict("no", calls, "b"), _Verdict(True, calls, "c")],
        )
        decision = await hooks.before("before_step", _event())
        assert decision.reason == "no"
        assert calls == ["a", "b"]

    async def test_no_hooks_allow(self) -> None:
        assert await Hooks(_lg()).before("before_step", _event()) == ALLOW


class TestOrder:
    async def test_before_runs_outermost_first_and_after_innermost_first(self) -> None:
        calls: list[str] = []
        hooks = Hooks(_lg(), [_Records("outer", calls), _Records("inner", calls)])
        await hooks.before("before_step", _event())
        await hooks.after("after_step", _event())
        assert calls == ["before outer", "before inner", "after inner", "after outer"]

    async def test_facets_reach_later_hooks(self) -> None:
        class _Tags(Hook):
            def before_step(self, event: StepEvent) -> None:
                event.facets["tag"] = "x"

        seen: list[Any] = []

        class _Reads(Hook):
            def before_step(self, event: StepEvent) -> None:
                seen.append(event.facets.get("tag"))

        await Hooks(_lg(), [_Tags(), _Reads()]).before("before_step", _event())
        assert seen == ["x"]


class TestFailures:
    async def test_a_before_hook_that_raises_denies(self) -> None:
        class _Raises(Hook):
            def before_step(self, event: StepEvent) -> None:
                raise ValueError("boom")

        lg = _lg()
        decision = await Hooks(lg, [_Raises()]).before("before_step", _event())
        assert decision == Decision("deny", "_Raises raised ValueError: boom", "_Raises")
        lg.warning.assert_called_once()

    async def test_an_after_hook_that_raises_is_logged_and_the_rest_run(self) -> None:
        calls: list[str] = []

        class _Raises(Hook):
            def after_step(self, event: StepEvent) -> None:
                raise ValueError("boom")

        lg = _lg()
        await Hooks(lg, [_Records("outer", calls), _Raises()]).after("after_step", _event())
        assert calls == ["after outer"]
        lg.warning.assert_called_once()


class TestMethods:
    async def test_async_methods_are_awaited(self) -> None:
        class _Async(Hook):
            async def before_step(self, event: StepEvent) -> str:
                return "async no"

        decision = await Hooks(_lg(), [_Async()]).before("before_step", _event())
        assert decision.reason == "async no"

    def test_methods_left_from_hook_are_not_resolved(self) -> None:
        hooks = Hooks(_lg(), [_Records("r", [])])
        assert hooks.has("before_step") and hooks.has("after_step")
        assert not hooks.has("after_llm_call")

    async def test_any_object_with_the_methods_is_a_hook(self) -> None:
        class _Plain:
            def before_step(self, event: StepEvent) -> str:
                return "plain no"

        decision = await Hooks(_lg(), [_Plain()]).before("before_step", _event())
        assert decision.reason == "plain no"

    async def test_a_method_set_on_the_instance_is_resolved(self) -> None:
        hook = SimpleNamespace(before_step=lambda _event: "instance no")
        decision = await Hooks(_lg(), [hook]).before("before_step", _event())
        assert decision.reason == "instance no"

    async def test_an_instance_method_replaces_the_inherited_no_op(self) -> None:
        hook = Hook()
        hook.before_step = lambda _event: "replaced no"  # type: ignore[method-assign]
        decision = await Hooks(_lg(), [hook]).before("before_step", _event())
        assert decision.reason == "replaced no"


class TestAttachment:
    def test_extend_adds_inside(self) -> None:
        a, b = _Records("a", []), _Records("b", [])
        assert Hooks(_lg(), [a]).extend([b]).hooks == (a, b)

    def test_extend_with_replace_drops_the_inherited(self) -> None:
        a, b = _Records("a", []), _Records("b", [])
        assert Hooks(_lg(), [a]).extend([b], replace=True).hooks == (b,)

    def test_child_asks_each_hook(self) -> None:
        replacement = _Records("child", [])

        class _PerScope(Hook):
            def child(self, where: Where) -> Hook:
                return replacement

        shared = _Records("shared", [])
        children = Hooks(_lg(), [shared, _PerScope()]).child(Where(node_id="n"))
        assert children.hooks == (shared, replacement)

    def test_find_returns_the_innermost_of_a_type(self) -> None:
        outer, inner = _Records("outer", []), _Records("inner", [])
        hooks = Hooks(_lg(), [outer, _Verdict(True), inner])
        assert hooks.find(_Records) is inner
        assert hooks.find(Pause) is None


class TestContentId:
    def test_is_the_id_of_the_blob_of_the_json_form(self) -> None:
        assert _event().content_id({"a": 1}) == Blob(canonical_json({"a": 1})).id

    def test_equal_values_share_an_id(self) -> None:
        event = _event()
        assert event.content_id([1, 2]) == event.content_id([1, 2])

    def test_a_tuple_and_a_list_differ(self) -> None:
        event = _event()
        assert event.content_id((1, 2)) != event.content_id([1, 2])

    def test_a_value_without_a_json_form_raises(self) -> None:
        with pytest.raises(TypeError):
            _event().content_id(object())
