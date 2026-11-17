# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Unit tests for :class:`llm_gent.provenance.VersionControl`: records, edges, sharing a repo."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from appinfra.cas import MemoryStore, RefConflictError, Tree, Vector, canonical_json
from llm_saia import ListConversation, Message
from pydantic import BaseModel

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

    async def test_an_input_the_step_mutates_is_recorded_as_it_was_received(self) -> None:
        @verb
        async def start(ctx: Context[Any], x: int) -> list[int]:
            return [x]

        @verb
        async def extend(ctx: Context[Any], items: list[int]) -> int:
            items.append(2)
            return len(items)

        vc = _vc(memory_store())
        await _flow().call(start, name="a").then(extend, name="b").with_hooks(vc).run(1)
        a, b, _ = _oldest_first(vc)
        assert vc.load(b.inputs["0"]) == [1]
        assert b.inputs["0"].id == a.outputs["result"].id
        assert vc._received == {}

    async def test_a_keyword_named_like_a_positional_index_is_left_out_and_named(self) -> None:
        @verb
        async def take(ctx: Context[Any], x: int, **kw: int) -> int:
            return x

        vc = _vc(memory_store())
        await _flow().call(take).with_hooks(vc).run(1, **{"0": 2, "y": 3})
        step = _oldest_first(vc)[0]
        assert vc.load(step.inputs["0"]) == 1
        assert vc.load(step.inputs["y"]) == 3
        assert step.meta["omitted"] == ["inputs/0 (keyword)"]

    async def test_a_step_a_later_hook_denies_keeps_its_encoded_inputs(self) -> None:
        class Deny(Hook):
            def before_step(self, event: StepEvent) -> str:
                return "no"

        vc = _vc(memory_store())
        flow = _flow().call(add_one, rescue=lambda _e, _p, _c: 0).with_hooks(vc, Deny())
        assert await flow.run(5) == 0
        step = _oldest_first(vc)[0]
        assert step.meta["outcome"] == "denied"
        assert vc.load(step.inputs["0"]) == 5
        assert vc._received == {}

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


class _Unreadable:
    """A conversation whose ``to_dict()`` fails."""

    def __init__(self) -> None:
        self.items: list[Any] = []

    def append(self, message: Any) -> None:
        self.items.append(message)

    def to_dict(self) -> dict[str, Any]:
        raise RuntimeError("conversation unreadable")


class _UnreadableConversations:
    def create(self) -> _Unreadable:
        return _Unreadable()


class TestModelCalls:
    async def test_a_context_that_cannot_be_read_is_named_and_the_call_kept(self) -> None:
        vc = _vc(memory_store())
        loop = Loop(Role(name="worker"), conversation_factory=_UnreadableConversations())  # type: ignore[arg-type]
        await _flow(saia_factory=_Factory()).call(loop).with_hooks(vc).run("go")
        calls = [r for r in _oldest_first(vc) if r.kind == "llm_call"]
        assert len(calls) == 3
        assert calls[0].meta["omitted"] == ["inputs/context"]
        assert vc.load(calls[0].outputs["response"])["content"] == "turn-0"

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

    async def test_a_verb_commits_a_record_of_its_own_inside_its_step(self) -> None:
        vc = _vc(memory_store())

        @verb
        async def choose(ctx: Context[Any], x: int) -> int:
            found = ctx.hook(VersionControl)
            assert found is not None
            found.commit(
                "chose",
                inputs={"options": [1, 2]},
                outputs={"choice": 2, "log": Messages(["a", "b"])},
                facets={"why": "bigger"},
                within=ctx,
            )
            return 2

        await _flow().call(choose).with_hooks(vc).run(1)
        chose, step, run = _oldest_first(vc)
        assert chose.kind == "chose"
        assert chose.meta["parent"] == step.meta["activity"]
        assert chose.meta["run"] == run.meta["run"]
        assert chose.meta["facets"] == {"why": "bigger"}
        assert vc.load(chose.outputs["log"]) == ["a", "b"]
        assert chose.outputs["log"].type == "vector"

    async def test_a_verb_opens_an_activity_inside_its_step(self) -> None:
        vc = _vc(memory_store())

        @verb
        async def work(ctx: Context[Any], x: int) -> int:
            found = ctx.hook(VersionControl)
            assert found is not None
            with found.activity("plan", within=ctx) as act:
                act.commit("note", outputs={"text": "hi"})
            return x

        await _flow().call(work).with_hooks(vc).run(1)
        note, plan, step, _ = _oldest_first(vc)
        assert note.meta["parent"] == plan.meta["activity"]
        assert plan.meta["parent"] == step.meta["activity"]
        assert plan.meta["run"] == step.meta["run"]

    async def test_a_cyclic_value_is_left_out_and_its_siblings_kept(self) -> None:
        vc = _vc(memory_store())
        cyclic: list[Any] = []
        cyclic.append(cyclic)
        vc.commit("kept", inputs={"loop": cyclic, "ok": 1}, facets={"loop": cyclic})
        (kept,) = _oldest_first(vc)
        assert kept.meta["omitted"] == ["inputs/loop"]
        assert vc.load(kept.inputs["ok"]) == 1
        assert kept.meta["facets"] == {"loop": "[[...]]"}

    async def test_a_value_whose_encoding_fails_is_left_out_and_its_siblings_kept(self) -> None:
        class Model(BaseModel):
            value: Any

        vc = _vc(memory_store())
        broken = Model(value=object())  # model_dump raises PydanticSerializationError
        vc.commit("kept", inputs={"model": broken, "ok": 1}, facets={"model": broken})
        (kept,) = _oldest_first(vc)
        assert kept.meta["omitted"] == ["inputs/model"]
        assert vc.load(kept.inputs["ok"]) == 1
        assert kept.meta["facets"] == {"model": repr(broken)}

    async def test_a_ref_moved_by_another_writer_is_committed_on(self) -> None:
        vc = _vc(memory_store())
        real = vc.repo.commit
        competing: list[str] = []

        def racing(ref: str, tree: str, payload: bytes = b"") -> str:
            if not competing:  # another writer commits first; this attempt loses the race
                empty = vc.repo.put(Tree(()))
                competing.append(real(ref, empty, canonical_json({"kind": "other"})))
                raise RefConflictError(REPO, ref, None)
            return real(ref, tree, payload)

        vc.repo.commit = racing  # type: ignore[method-assign]
        await _flow().call(add_one).with_hooks(vc).run(1)
        other, step, run = _oldest_first(vc)
        assert (other.id, other.kind) == (competing[0], "other")
        assert step.parents == (other.id,)
        assert (step.kind, run.kind) == ("step", "run")


@dataclass
class _InferRequest:
    messages: list[Any]
    id: str = "req-1"


@dataclass
class _InferResponse:
    """Shaped like llm-infer's ChatResponse: a back-reference to its request, and ``raw``."""

    content: str
    model: str = "m-1"
    raw: Any = None
    request: _InferRequest | None = None


class TestExplicit:
    """Records made without Flow: commit, activity, llm_call, llm_calls."""

    async def test_a_record_on_its_own_has_no_parent_and_no_run(self) -> None:
        vc = _vc(memory_store())
        vc.commit("note", inputs={"q": "why"}, outputs={"a": 42}, facets={"tag": "x"})
        (note,) = _oldest_first(vc)
        assert note.kind == "note"
        assert (note.meta["parent"], note.meta["run"]) == (None, None)
        assert vc.load(note.outputs["a"]) == 42
        assert note.meta["facets"] == {"tag": "x"}

    async def test_an_activity_commits_itself_after_the_records_inside_it(self) -> None:
        vc = _vc(memory_store())
        with vc.activity("summarize", inputs={"doc": "text"}) as act:
            act.commit("note", outputs={"n": 1})
            with act.activity("critique") as sub:
                sub.commit("note", outputs={"n": 2})
            act.outputs["summary"] = "short"
            act.facets["words"] = 1
        first, second, critique, summarize = _oldest_first(vc)
        assert summarize.meta["activity"] == act.id == summarize.meta["run"]
        assert summarize.meta["parent"] is None and summarize.meta["outcome"] == "ok"
        assert first.meta["parent"] == act.id == critique.meta["parent"]
        assert second.meta["parent"] == critique.meta["activity"] == sub.id
        assert {r.meta["run"] for r in (first, second, critique)} == {act.id}
        assert vc.load(summarize.inputs["doc"]) == "text"
        assert vc.load(summarize.outputs["summary"]) == "short"
        assert summarize.meta["facets"] == {"words": 1}

    async def test_an_activity_records_its_inputs_as_it_received_them(self) -> None:
        vc = _vc(memory_store())
        items = [1]
        with vc.activity("extend", inputs={"items": items}):
            items.append(2)
        (extend,) = _oldest_first(vc)
        assert vc.load(extend.inputs["items"]) == [1]

    async def test_an_activity_an_exception_escapes_is_failed(self) -> None:
        vc = _vc(memory_store())
        with pytest.raises(ValueError, match="boom"), vc.activity("work"):
            raise ValueError("boom")
        (work,) = _oldest_first(vc)
        assert work.meta["outcome"] == "failed"
        assert work.meta["error"] == "ValueError: boom"

    async def test_a_cancelled_activity_is_interrupted(self) -> None:
        vc = _vc(memory_store())
        with pytest.raises(asyncio.CancelledError), vc.activity("work"):
            raise asyncio.CancelledError
        assert _oldest_first(vc)[0].meta["outcome"] == "interrupted"

    async def test_a_failed_activity_record_does_not_replace_the_activitys_error(self) -> None:
        vc = _vc(memory_store())

        def unavailable(ref: str, tree: str, payload: bytes = b"") -> str:
            raise OSError("store unavailable")

        vc.repo.commit = unavailable  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="boom"), vc.activity("work"):
            raise ValueError("boom")
        with pytest.raises(OSError, match="store unavailable"), vc.activity("work"):
            pass

    async def test_a_plain_http_call_is_a_message_vector_and_its_json(self) -> None:
        vc = _vc(memory_store())
        messages = [{"role": "user", "content": "hi"}]
        with vc.activity("chat") as act:
            act.llm_call(messages, {"model": "m-2", "choices": [{"text": "hello"}]})
        call, _ = _oldest_first(vc)
        assert call.kind == "llm_call" and call.inputs["context"].type == "vector"
        assert vc.load(call.inputs["context"]) == messages
        assert vc.load(call.outputs["response"])["choices"] == [{"text": "hello"}]
        assert call.meta["actor"]["model"] == "m-2"
        assert call.meta["trace"] is None

    async def test_a_response_is_stored_without_its_request_and_raw(self) -> None:
        vc = _vc(memory_store())
        messages = [{"role": "user", "content": "hi"}]
        response = _InferResponse("hello", raw={"x": 1}, request=_InferRequest(messages))
        vc.llm_call(messages, response)
        (call,) = _oldest_first(vc)
        assert vc.load(call.outputs["response"]) == {"content": "hello", "model": "m-1"}
        assert call.meta["trace"]["llm_request_id"] == "req-1"

    async def test_a_context_that_cannot_be_read_is_named_and_the_call_kept(self) -> None:
        vc = _vc(memory_store())
        vc.llm_call(_Unreadable(), _Response("done"))
        (call,) = _oldest_first(vc)
        assert call.meta["omitted"] == ["inputs/context"]
        assert vc.load(call.outputs["response"])["content"] == "done"

    async def test_a_saia_list_conversation_is_read_through_as_messages(self) -> None:
        vc = _vc(memory_store())
        conversation = ListConversation()
        conversation.append(Message(role="user", content="go"))
        vc.llm_call(conversation, _Response("done"))
        (call,) = _oldest_first(vc)
        assert call.inputs["context"].type == "vector"
        # A typed value is stored in the checkpoint codec's form, naming its type.
        (message,) = vc.load(call.inputs["context"])
        assert message["$type"].endswith(":Message")
        assert message["$data"] == {"role": "user", "content": "go"}

    async def test_saia_iterations_are_calls_that_share_their_messages(self) -> None:
        vc = _vc(memory_store())
        conversation: list[Any] = [{"role": "user", "content": "go"}]
        with vc.activity("task") as act:
            on_iteration = act.llm_calls(conversation, role="writer")
            for i in range(3):
                response = _Response(f"turn-{i}", call_id=f"c-{i}")
                await on_iteration(i, response)
                conversation.append({"role": "assistant", "content": response.content})
        calls = [r for r in _oldest_first(vc) if r.kind == "llm_call"]
        assert [c.meta["facets"]["iteration"] for c in calls] == [0, 1, 2]
        assert [c.meta["trace"]["call_id"] for c in calls] == ["c-0", "c-1", "c-2"]
        assert {c.meta["actor"]["role"] for c in calls} == {"writer"}
        contexts = [list(Vector(vc.repo, c.inputs["context"].id)) for c in calls]
        assert contexts[2][:2] == contexts[1] and contexts[1][:1] == contexts[0]
