# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""An agent's sessions: ids the agent issues, one conversation each, saved and reloaded."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from appinfra import DotDict
from llm_kelt.conversation import Conversation, SessionStorage, StoredSession
from llm_kelt.core.errors import NotFoundError
from llm_saia import TaskResult

from llm_gent.agents.default import Agent as DefaultAgent
from llm_gent.agents.default.factory import _session_storage
from llm_gent.core.agent import Identity, SessionConfig, Sessions
from llm_gent.core.errors import ConfigError, SessionNotFoundError
from llm_gent.core.traits.builtin.memory import MemoryTrait
from llm_gent.core.traits.builtin.saia import SAIATrait
from tests.helpers import create_mock_trait, make_platform


pytestmark = pytest.mark.unit


class _DictStorage(SessionStorage):
    """kelt's session storage contract over a dict: save, load, list, delete."""

    def __init__(self) -> None:
        self.saved: dict[str, list[dict]] = {}
        self.loads: list[str] = []

    def save(self, session_id: str, conversation: Conversation, extra: dict | None = None) -> None:
        self.saved[session_id] = conversation.messages_as_dicts()

    def load(self, session_id: str) -> StoredSession:
        self.loads.append(session_id)
        if session_id not in self.saved:
            raise NotFoundError(f"Session not found: {session_id}")
        return StoredSession(session_id=session_id, messages=self.saved[session_id])

    def list(self, limit: int = 100) -> list:
        return []

    def delete(self, session_id: str) -> bool:
        return self.saved.pop(session_id, None) is not None


def _sessions(storage=None, identity=Identity("jokester"), **config) -> Sessions:
    return Sessions(MagicMock(), identity, SessionConfig(**config), storage)


class TestSessions:
    def test_open_issues_a_fresh_uuid(self):
        sessions = _sessions()

        first, second = sessions.open(), sessions.open()

        assert str(uuid.UUID(first.id)) == first.id
        assert first.id != second.id

    def test_an_open_session_is_found_again(self):
        sessions = _sessions()
        session = sessions.open()

        assert sessions.get(session.id) is session

    def test_an_id_never_issued_is_refused(self):
        with pytest.raises(SessionNotFoundError):
            _sessions().get(str(uuid.uuid4()))

    @pytest.mark.parametrize("session_id", ["made-up", "", "3F2B6C1E-9D4A-4B7E-8C21-5A0F7E9D2B13"])
    def test_anything_but_a_canonical_uuid_is_refused(self, session_id):
        with pytest.raises(SessionNotFoundError):
            _sessions(_DictStorage()).get(session_id)

    def test_a_saved_session_reloads_in_a_new_process(self):
        storage = _DictStorage()
        first = _sessions(storage)
        session = first.open()
        session.conversation.add("hello")
        first.save(session)

        reloaded = _sessions(storage).get(session.id)

        assert [m.content for m in reloaded.conversation.messages] == ["hello"]
        assert list(storage.saved) == [f"jokester/jokester/{session.id}"]

    def test_an_id_is_valid_only_at_the_agent_that_issued_it(self):
        storage = _DictStorage()
        mine = _sessions(storage)
        session = mine.open()
        mine.save(session)

        with pytest.raises(SessionNotFoundError):
            _sessions(storage, identity=Identity("other", context_key="jokester")).get(session.id)

    def test_without_storage_a_session_is_gone_after_a_restart(self):
        session = _sessions().open()

        with pytest.raises(SessionNotFoundError):
            _sessions().get(session.id)

    def test_beyond_max_live_an_idle_session_leaves_memory_and_reloads(self):
        storage = _DictStorage()
        sessions = _sessions(storage, max_live=1)
        first = sessions.open()
        sessions.save(first)
        sessions.open()

        reloaded = sessions.get(first.id)

        assert reloaded is not first
        assert storage.loads == [f"jokester/jokester/{first.id}"]

    def test_a_session_in_an_ask_stays_in_memory(self):
        storage = _DictStorage()
        sessions = _sessions(storage, max_live=1)
        busy = sessions.open()

        with busy.lock:
            sessions.open()
            assert sessions.get(busy.id) is busy
        assert storage.loads == []

    def test_without_storage_no_session_leaves_memory(self):
        sessions = _sessions(max_live=1)
        first = sessions.open()
        sessions.open()

        assert sessions.get(first.id) is first

    def test_an_unknown_compactor_is_a_config_error(self):
        with pytest.raises(ConfigError, match="compactor"):
            _sessions(compactor="summarizing")

    def test_a_key_longer_than_kelt_takes_is_a_config_error(self):
        with pytest.raises(ConfigError, match="too long"):
            _sessions(identity=Identity("a" * 120, context_key="b" * 120))


@pytest.fixture
def saia():
    """A SAIA whose every task completes with its prompt echoed back."""
    saia = MagicMock()
    saia.compose.side_effect = lambda context, task: f"{context}|{task}"
    saia.complete = AsyncMock(
        side_effect=lambda prompt: TaskResult(
            completed=True, output=f"answer to {prompt}", iterations=1, history=[]
        )
    )
    return saia


def _agent(saia, config=None, storage=None) -> DefaultAgent:
    agent = DefaultAgent(
        lg=MagicMock(),
        config=DotDict({"identity": {"name": "jokester"}, **(config or {})}),
        session_storage=storage,
    )
    agent._traits.register(create_mock_trait(SAIATrait, _saia=saia))
    return agent


class TestAgentAsk:
    def test_an_ask_without_a_session_opens_one(self, saia):
        result = _agent(saia).ask("hello")

        assert result.success is True
        assert str(uuid.UUID(result.session)) == result.session

    def test_an_ask_in_a_session_sees_the_conversation_so_far(self, saia):
        agent = _agent(saia)
        first = agent.ask("my name is Ada")

        second = agent.ask("what is my name?", session=first.session)

        assert second.session == first.session
        context = saia.compose.call_args[0][0]
        assert "User: my name is Ada" in context
        assert f"Assistant: {first.content}" in context

    def test_sessions_keep_their_conversations_apart(self, saia):
        agent = _agent(saia)
        agent.ask("my name is Ada")

        agent.ask("what is my name?")

        assert "Ada" not in saia.compose.call_args[0][0]

    def test_an_unknown_session_is_refused(self, saia):
        with pytest.raises(SessionNotFoundError):
            _agent(saia).ask("hello", session=str(uuid.uuid4()))

    def test_each_ask_saves_its_session(self, saia):
        storage = _DictStorage()

        result = _agent(saia, storage=storage).ask("hello")

        saved = storage.saved[f"jokester/jokester/{result.session}"]
        assert [m["content"] for m in saved] == ["hello", result.content]

    def test_an_ask_holds_its_session_for_its_whole_run(self, saia):
        """Asks in one session run one at a time: each holds the session's lock while it runs."""
        agent = _agent(saia)
        session = agent._sessions.get(agent.ask("first").session)
        held_during_the_call: list[bool] = []

        async def complete(prompt):
            held_during_the_call.append(session.lock.locked())
            return TaskResult(completed=True, output="done", iterations=1, history=[])

        saia.complete = AsyncMock(side_effect=complete)
        agent.ask("second", session=session.id)

        assert held_during_the_call == [True]
        assert not session.lock.locked()

    def test_run_once_runs_in_no_session(self, saia):
        agent = _agent(saia, config={"default_prompt": "tell a joke"})

        assert agent.run_once().session == ""

    def test_the_conversation_section_sets_the_limits(self, saia):
        agent = _agent(saia, config={"conversation": {"max_tokens": 8000, "max_live": 5}})

        assert agent._sessions._config == SessionConfig(max_tokens=8000, max_live=5)

    def test_an_unknown_conversation_setting_is_a_config_error(self, saia):
        with pytest.raises(ConfigError, match="conversation"):
            _agent(saia, config={"conversation": {"window": 3}})


class TestSessionStorage:
    def test_a_platform_without_memory_keeps_sessions_in_memory(self):
        assert _session_storage(make_platform(MagicMock())) is None

    def test_sessions_persist_in_the_memory_schema(self):
        database = MagicMock()
        with patch("llm_gent.core.traits.builtin.memory.KeltClient"):
            memory = MemoryTrait(
                MagicMock(),
                DotDict({"schema": {"name": "mem"}}),
                database=database,
                chat_client=MagicMock(),
            )
        platform = make_platform(MagicMock(), memory_config={"db": {}}, memory=memory)

        storage = _session_storage(platform)

        assert storage is not None
        database.scoped.assert_called_once_with("mem")
