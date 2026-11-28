# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""An agent's sessions: ids the agent issues, one conversation each, saved and reloaded."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from appinfra import DotDict
from llm_kelt.conversation import Conversation, SessionStorage, StoredSession
from llm_kelt.core.errors import NotFoundError
from llm_saia import TaskResult

from llm_gent.agents.default import Agent as DefaultAgent
from llm_gent.agents.default import ChatAgent, Factory
from llm_gent.agents.default.factory import _session_storage
from llm_gent.core.agent import AskStep, Chat, Identity, Session, SessionConfig, Sessions
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


def _use(sessions: Sessions, session_id: str | None = None) -> Session:
    """The session one ask would use, released again."""
    with sessions.use(session_id) as session:
        return session


class TestSessions:
    def test_no_id_opens_a_session_under_a_fresh_uuid(self):
        sessions = _sessions()

        first, second = _use(sessions), _use(sessions)

        assert str(uuid.UUID(first.id)) == first.id
        assert first.id != second.id

    def test_an_open_session_is_found_again(self):
        sessions = _sessions()
        session = _use(sessions)

        assert _use(sessions, session.id) is session

    def test_a_session_in_use_holds_its_lock(self):
        sessions = _sessions()

        with sessions.use(None) as session:
            assert session.lock.locked()
        assert not session.lock.locked()

    def test_an_id_never_issued_is_refused(self):
        with pytest.raises(SessionNotFoundError):
            _use(_sessions(), str(uuid.uuid4()))

    @pytest.mark.parametrize("session_id", ["made-up", "", "3F2B6C1E-9D4A-4B7E-8C21-5A0F7E9D2B13"])
    def test_anything_but_a_canonical_uuid_is_refused(self, session_id):
        with pytest.raises(SessionNotFoundError):
            _use(_sessions(_DictStorage()), session_id)

    def test_a_saved_session_reloads_in_a_new_process(self):
        storage = _DictStorage()
        first = _sessions(storage)
        with first.use(None) as session:
            session.conversation.add("hello")
            first.save(session)

        reloaded = _use(_sessions(storage), session.id)

        assert [m.content for m in reloaded.conversation.messages] == ["hello"]
        assert list(storage.saved) == [f"jokester/jokester/{session.id}"]

    def test_an_id_is_valid_only_at_the_agent_that_issued_it(self):
        storage = _DictStorage()
        mine = _sessions(storage)
        with mine.use(None) as session:
            mine.save(session)

        with pytest.raises(SessionNotFoundError):
            _use(_sessions(storage, identity=Identity("other", context_key="jokester")), session.id)

    def test_without_storage_a_session_is_gone_after_a_restart(self):
        session = _use(_sessions())

        with pytest.raises(SessionNotFoundError):
            _use(_sessions(), session.id)

    def test_beyond_max_live_an_idle_session_leaves_memory_and_reloads(self):
        storage = _DictStorage()
        sessions = _sessions(storage, max_live=1)
        with sessions.use(None) as first:
            sessions.save(first)
        _use(sessions)

        reloaded = _use(sessions, first.id)

        assert reloaded is not first
        assert storage.loads == [f"jokester/jokester/{first.id}"]

    def test_a_session_in_use_stays_in_memory(self):
        """Another ask opening a session cannot push out one in use: no second copy of it."""
        storage = _DictStorage()
        sessions = _sessions(storage, max_live=1)

        with sessions.use(None) as busy:
            _use(sessions)
            assert sessions._live.get(busy.id) is busy
        assert storage.loads == []

    def test_a_session_is_in_use_before_its_lock_is_taken(self):
        """Counted at lookup, so an ask waiting for the lock keeps its session in memory."""
        sessions = _sessions(_DictStorage(), max_live=1)
        waiting = sessions._acquire(None)

        _use(sessions)

        assert not waiting.lock.locked()
        assert sessions._live.get(waiting.id) is waiting

    def test_without_storage_no_session_leaves_memory(self):
        sessions = _sessions(max_live=1)
        first = _use(sessions)
        _use(sessions)

        assert _use(sessions, first.id) is first

    def test_an_unknown_compactor_is_a_config_error(self):
        with pytest.raises(ConfigError, match="compactor"):
            _sessions(compactor="summarizing")

    def test_a_slash_in_the_name_is_a_config_error(self):
        """Else ``a/b`` + ``c`` and ``a`` + ``b/c`` would share one key prefix."""
        with pytest.raises(ConfigError, match="'/'"):
            _sessions(identity=Identity("b/c", context_key="a"))

    def test_a_hierarchical_context_key_is_accepted(self):
        storage = _DictStorage()
        sessions = _sessions(storage, identity=Identity("c", context_key="a/b"))
        with sessions.use(None) as session:
            sessions.save(session)

        assert list(storage.saved) == [f"a/b/c/{session.id}"]

    def test_a_key_longer_than_kelt_takes_is_a_config_error(self):
        with pytest.raises(ConfigError, match="too long"):
            _sessions(identity=Identity("a" * 120, context_key="b" * 120))


@pytest.fixture
def saia():
    """A SAIA whose every task completes with its prompt echoed back."""
    saia = MagicMock()
    saia.compose.side_effect = lambda context, task: f"{context}|{task}"
    saia.complete = AsyncMock(
        side_effect=lambda prompt, **_: TaskResult(
            completed=True, output=f"answer to {prompt}", iterations=1, history=[]
        )
    )
    return saia


def _agent(saia, config=None, storage=None) -> ChatAgent:
    lg = MagicMock()
    agent = ChatAgent(
        lg=lg,
        platform=make_platform(lg),
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
        session = agent._sessions._live[agent.ask("first").session]
        held_during_the_call: list[bool] = []

        async def complete(prompt, **_):
            held_during_the_call.append(session.lock.locked())
            return TaskResult(completed=True, output="done", iterations=1, history=[])

        saia.complete = AsyncMock(side_effect=complete)
        agent.ask("second", session=session.id)

        assert held_during_the_call == [True]
        assert not session.lock.locked()

    def test_run_once_runs_in_no_session(self, saia):
        agent = _agent(saia, config={"default_prompt": "tell a joke"})

        assert agent.run_once().session == ""

    def test_the_chat_section_sets_the_limits(self, saia):
        agent = _agent(saia, config={"chat": {"max_tokens": 8000, "max_live": 5}})

        assert agent._sessions._config == SessionConfig(max_tokens=8000, max_live=5)

    def test_an_unknown_chat_setting_is_a_config_error(self, saia):
        with pytest.raises(ConfigError, match="chat"):
            _agent(saia, config={"chat": {"window": 3}})

    def test_each_model_call_is_a_step(self, saia):
        response = SimpleNamespace(
            input_tokens=3, output_tokens=4, tool_calls=[SimpleNamespace(name="complete_task")]
        )

        async def complete(prompt, on_iteration):
            await on_iteration(1, response)
            return TaskResult(completed=True, output="done", iterations=1, history=[])

        saia.complete = AsyncMock(side_effect=complete)
        steps: list[AskStep] = []

        _agent(saia).ask("hello", on_step=steps.append)

        assert steps == [
            AskStep("model_call", {"iteration": 1, "tools": ["complete_task"], "tokens": 7})
        ]


class TestChatOptIn:
    def test_an_agent_without_a_chat_section_has_no_chat(self):
        agent = Factory(make_platform(MagicMock())).create({"identity": {"name": "a"}})

        assert not isinstance(agent, Chat)

    def test_a_chat_section_makes_a_chat_agent(self):
        agent = Factory(make_platform(MagicMock())).create(
            {"identity": {"name": "a"}, "chat": {"max_tokens": 8000}}
        )

        assert isinstance(agent, ChatAgent)
        assert agent._sessions._config.max_tokens == 8000

    def test_a_factorys_own_chat_agent_class_is_kept(self):
        class Reviewer(ChatAgent):
            pass

        class ReviewerFactory(Factory):
            agent_class = Reviewer

        agent = ReviewerFactory(make_platform(MagicMock())).create(
            {"identity": {"name": "a"}, "chat": {}}
        )

        assert type(agent) is Reviewer

    def test_a_factorys_own_chat_agent_class_has_chat_without_a_chat_section(self):
        class Reviewer(ChatAgent):
            pass

        class ReviewerFactory(Factory):
            agent_class = Reviewer

        agent = ReviewerFactory(make_platform(MagicMock())).create({"identity": {"name": "a"}})

        assert type(agent) is Reviewer

    def test_a_chat_section_on_a_factorys_agent_class_without_chat_is_a_config_error(self):
        class Plain(DefaultAgent):
            pass

        class PlainFactory(Factory):
            agent_class = Plain

        with pytest.raises(ConfigError, match="Plain has no chat"):
            PlainFactory(make_platform(MagicMock())).create({"identity": {"name": "a"}, "chat": {}})


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
