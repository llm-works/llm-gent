# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.provenance.VersionControl`: records, edges, sharing a repo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from appinfra.cas import MemoryStore, RefConflictError, Vector

from llm_gent.flow import Context, Factory, Flow, Loop, Role, verb
from llm_gent.flow.checkpoint import HEAD_REF
from llm_gent.flow.hooks import Hook, StepEvent
from llm_gent.provenance import REF, Messages, Record, VersionControl
from tests.helpers import make_test_logger, memory_store


pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

REPO = "run"


@verb
async def add_one(ctx: Context[Any], x: int) -> int:
    return x + 1


@verb
async def double(ctx: Context[Any], x: int) -> int:
    return x * 2


def _flow(**kw: Any) -> Flow:
    return Factory(make_test_logger(), **kw).create(state={})


def _vc(store: MemoryStore) -> VersionControl:
    return VersionControl(make_test_logger(), store.repo(REPO))


def _oldest_first(vc: VersionControl) -> list[Record]:
    return list(reversed(list(vc.log())))


class TestSteps:
    async def test_every_step_and_the_run_are_records_in_order(self) -> None:
        vc = _vc(memory_store())
        await _flow().call(add_one, name="a").then(double, name="b").with_hooks(vc).run(1)
        records = _oldest_first(vc)
        assert [(r.kind, r.meta.get("where", {}).get("step_name")) for r in records] == [
            ("step", "a"),
            ("step", "b"),
            ("run", None),
        ]
        assert [r.meta["outcome"] for r in records] == ["ok", "ok", "ok"]

    async def test_a_steps_input_is_the_blob_of_the_previous_steps_result(self) -> None:
        vc = _vc(memory_store())
        await _flow().call(add_one, name="a").then(double, name="b").with_hooks(vc).run(1)
        a, b, _ = _oldest_first(vc)
        assert b.inputs["0"].id == a.outputs["result"].id
        assert vc.load(a.inputs["0"]) == 1
        assert vc.load(b.outputs["result"]) == 4

    async def test_records_name_their_activity_and_its_parent(self) -> None:
        vc = _vc(memory_store())
        await _flow().call(add_one).with_hooks(vc).run(1)
        step, run = _oldest_first(vc)
        assert step.meta["parent"] == run.meta["activity"] == run.meta["run"]
        assert step.meta["run"] == run.meta["run"]

    async def test_each_record_is_parented_on_the_one_before(self) -> None:
        vc = _vc(memory_store())
        await _flow().call(add_one).then(double).with_hooks(vc).run(1)
        records = _oldest_first(vc)
        assert records[0].parents == ()
        assert [r.parents for r in records[1:]] == [(r.id,) for r in records[:-1]]

    async def test_a_denied_step_records_the_decision(self) -> None:
        class Deny(Hook):
            def before_step(self, event: StepEvent) -> str:
                return "no"

        vc = _vc(memory_store())
        flow = _flow().call(add_one, rescue=lambda _e, _p, _c: 0).with_hooks(Deny(), vc)
        assert await flow.run(1) == 0
        step = _oldest_first(vc)[0]
        assert step.meta["outcome"] == "denied"
        assert step.meta["decision"] == {"verdict": "deny", "reason": "no", "hook": "Deny"}

    async def test_a_value_with_no_json_form_is_left_out_and_named(self) -> None:
        @verb
        async def opaque(ctx: Context[Any], x: int) -> object:
            return object()

        vc = _vc(memory_store())
        await _flow().call(opaque).with_hooks(vc).run(1)
        step = _oldest_first(vc)[0]
        assert step.meta["omitted"] == ["outputs/result"]
        assert "0" in step.inputs

    async def test_parallel_map_items_form_one_line(self) -> None:
        vc = _vc(memory_store())
        body = _flow().call(add_one, name="item")
        await _flow().map(body, items=lambda _p, _c: list(range(8))).with_hooks(vc).run(0)
        records = list(vc.log())
        items = [r for r in records if r.meta.get("where", {}).get("step_name") == "item"]
        assert len(items) == 8
        assert all(len(r.parents) == 1 for r in records[:-1])


@dataclass
class _Response:
    content: str
    model: str = "stub"
    call_id: str = ""
    raw: Any = None


class _SAIA:
    def __init__(self, role: Role) -> None:
        self.role = role

    async def complete(
        self, task: str, *, on_iteration: Any = None, conversation: Any = None, **_: Any
    ) -> Any:
        for i in range(3):
            response = _Response(f"turn-{i}", raw=object())
            await on_iteration(i, response)
            conversation.append({"role": "assistant", "content": response.content})
        return None


class _Factory:
    def build(self, role: Role) -> _SAIA:
        return _SAIA(role)


class _Conversations:
    def create(self) -> list[Any]:
        return [{"role": "user", "content": "go"}]


class TestModelCalls:
    async def test_each_call_records_its_context_as_a_vector_and_its_response(self) -> None:
        vc = _vc(memory_store())
        loop = Loop(Role(name="worker"), conversation_factory=_Conversations())  # type: ignore[arg-type]
        await _flow(saia_factory=_Factory()).call(loop).with_hooks(vc).run("go")
        calls = [r for r in _oldest_first(vc) if r.kind == "llm_call"]
        assert [len(vc.load(c.inputs["context"])) for c in calls] == [1, 2, 3]
        assert vc.load(calls[2].inputs["context"])[1] == {"role": "assistant", "content": "turn-0"}
        response = vc.load(calls[0].outputs["response"])
        assert response["content"] == "turn-0" and "raw" not in response
        assert calls[0].meta["actor"]["role"] == "worker"

    async def test_calls_share_the_messages_they_have_in_common(self) -> None:
        store = memory_store()
        vc = _vc(store)
        loop = Loop(Role(name="worker"), conversation_factory=_Conversations())  # type: ignore[arg-type]
        await _flow(saia_factory=_Factory()).call(loop).with_hooks(vc).run("go")
        calls = [r for r in _oldest_first(vc) if r.kind == "llm_call"]
        contexts = [list(Vector(vc.repo, c.inputs["context"].id)) for c in calls]
        # Each context extends the one before: its messages are the same blobs.
        assert contexts[1][:1] == contexts[0] and contexts[2][:2] == contexts[1]
        # 1 + 2 + 3 messages across three contexts, 3 distinct: one blob each.
        assert len({m for context in contexts for m in context}) == 3


class TestCheckpointsAndRepo:
    async def test_checkpoints_are_recorded_next_to_the_history(self) -> None:
        store = memory_store()
        vc = _vc(store)

        @verb
        async def save(ctx: Context[Any], x: int) -> int:
            await ctx.checkpoint()
            return x

        flow = (
            _flow().with_checkpoint_store(store, REPO).with_checkpointer().call(save).with_hooks(vc)
        )
        await flow.run(1)
        checkpoints = [r for r in _oldest_first(vc) if r.kind == "checkpoint"]
        head = store.repo(REPO).get_ref(HEAD_REF)
        assert checkpoints[-1].meta["checkpoint"] == head  # the final-state commit
        assert set(store.repo(REPO).refs()) == {HEAD_REF, "tags/complete", REF}

    async def test_gc_on_success_keeps_the_records(self) -> None:
        store = memory_store()
        vc = _vc(store)
        flow = (
            _flow()
            .with_checkpoint_store(store, REPO, retention="gc_on_success")
            .call(add_one)
            .with_hooks(vc)
        )
        await flow.run(1)
        assert set(store.repo(REPO).refs()) == {REF}
        assert [r.kind for r in _oldest_first(vc)] == ["step", "run"]

    async def test_a_verb_records_its_own_activity(self) -> None:
        vc = _vc(memory_store())

        @verb
        async def choose(ctx: Context[Any], x: int) -> int:
            found = ctx.hook(VersionControl)
            assert found is not None
            found.record(
                ctx,
                "chose",
                inputs={"options": [1, 2]},
                outputs={"choice": 2, "log": Messages(["a", "b"])},
                facets={"why": "bigger"},
            )
            return 2

        await _flow().call(choose).with_hooks(vc).run(1)
        chose, step, _ = _oldest_first(vc)
        assert chose.kind == "chose"
        assert chose.meta["parent"] == step.meta["activity"]
        assert chose.meta["facets"] == {"why": "bigger"}
        assert vc.load(chose.outputs["log"]) == ["a", "b"]
        assert chose.outputs["log"].type == "vector"

    async def test_a_ref_moved_by_another_writer_is_committed_on(self) -> None:
        vc = _vc(memory_store())
        real = vc.repo.commit
        calls = {"n": 0}

        def flaky(ref: str, tree: str, payload: bytes = b"") -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RefConflictError(REPO, ref, None)
            return real(ref, tree, payload)

        vc.repo.commit = flaky  # type: ignore[method-assign]
        await _flow().call(add_one).with_hooks(vc).run(1)
        assert [r.kind for r in _oldest_first(vc)] == ["step", "run"]
