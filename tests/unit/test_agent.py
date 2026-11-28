# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for Agent base class and agents.default.Agent."""

from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from appinfra import DotDict

from llm_gent.core.agent import Agent, ExecutionResult, Identity, RunnableAgent
from llm_gent.core.agent.helpers import _substitute_variables
from llm_gent.core.traits import BaseTrait
from tests.helpers import create_mock_trait, make_platform


pytestmark = pytest.mark.unit


class _ResourceTrait(BaseTrait):
    """A trait on one injected resource, closed on stop iff the trait owns it."""

    def __init__(self, agent: Any, resource: Any, *, owns_resource: bool = False) -> None:
        super().__init__(agent)
        self.resource = resource
        self._owns_resource = owns_resource

    def with_resource(self, resource: Any) -> Self:
        """A detached trait on ``resource``, which it does not own."""
        return type(self)(self.agent, resource)

    def on_stop(self) -> None:
        if self._owns_resource:
            self.resource.close()


def _recording_trait(name: str, events: list[str], *, fails: bool = False) -> type[BaseTrait]:
    """A trait class of its own that records its start and stop in ``events``."""

    def on_start(self: BaseTrait) -> None:
        events.append(f"{name}.start")
        if fails:
            raise RuntimeError(f"{name} down")

    def on_stop(self: BaseTrait) -> None:
        events.append(f"{name}.stop")

    return type(name, (BaseTrait,), {"on_start": on_start, "on_stop": on_stop})


class TestAgentBaseClass:
    """Tests for the slim Agent container."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    def test_agent_requires_identity(self, mock_logger):
        """Agent construction fails without identity.name."""
        from llm_gent.core.errors import ConfigError

        with pytest.raises(ConfigError, match="identity.name is required"):
            Agent(lg=mock_logger, platform=make_platform(mock_logger))

    @pytest.mark.parametrize(
        ("config", "match"),
        [
            ({"identity": {"name": "a"}, "llm": "anthropic"}, "'llm' key"),
            ({"identity": {"name": "a"}, "kelt": {"schema": {"name": "custom"}}}, "kelt.schema"),
        ],
    )
    def test_an_agent_built_directly_rejects_platform_settings(self, mock_logger, config, match):
        from llm_gent.core.errors import ConfigError

        with pytest.raises(ConfigError, match=match):
            Agent(lg=mock_logger, platform=make_platform(mock_logger), config=DotDict(config))

    def test_agent_container_can_be_instantiated(self, mock_logger):
        """Plain Agent is directly instantiable with just identity."""
        config = DotDict(identity={"name": "test"})
        agent = Agent(lg=mock_logger, platform=make_platform(mock_logger), config=config)
        assert agent.name == "test"

    def test_runnable_agent_requires_execution_methods(self, mock_logger):
        """RunnableAgent subclass without abstracts cannot instantiate."""

        class IncompleteRunnable(RunnableAgent):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRunnable(lg=mock_logger, platform=make_platform(mock_logger))

    def test_runnable_agent_subclass_can_be_instantiated(self, mock_logger):
        """RunnableAgent subclass with all abstracts is instantiable."""

        class CompleteAgent(RunnableAgent):
            def run_once(self) -> ExecutionResult:
                return ExecutionResult(success=True, content="done")

            def ask(self, question: str) -> str:
                return "answer"

            def record_feedback(self, message: str) -> None:
                pass

            def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
                return []

        config = DotDict(identity={"name": "test"})
        agent = CompleteAgent(lg=mock_logger, platform=make_platform(mock_logger), config=config)
        assert agent.name == "test"


class TestAgentLifecycle:
    """Tests for Agent lifecycle."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        """Create a test agent using agents.default.Agent."""
        from llm_gent.agents.default import Agent as DefaultAgent

        return DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test-agent"}, default_prompt="You are a test agent."),
        )

    def test_init(self, agent):
        assert agent.name == "test-agent"

    def test_start_calls_trait_on_start(self, agent):
        mock_trait = MagicMock()
        agent._traits.register(mock_trait)

        agent.start()

        mock_trait.on_start.assert_called_once()

    def test_stop_calls_trait_on_stop(self, agent):
        mock_trait = MagicMock()
        agent._traits.register(mock_trait)
        agent._started = True

        agent.stop()

        mock_trait.on_stop.assert_called_once()

    def test_a_failed_start_stops_the_failed_one_then_the_started_in_reverse(self, agent):
        events: list[str] = []
        for name, fails in (("A", False), ("B", False), ("C", True)):
            agent._traits.register(_recording_trait(name, events, fails=fails)(agent))

        with pytest.raises(RuntimeError, match="C down"):
            agent.start()

        assert events == ["A.start", "B.start", "C.start", "C.stop", "B.stop", "A.stop"]
        assert not agent._started

    def test_a_trait_added_to_a_started_agent_that_fails_to_start_is_stopped(self, agent):
        events: list[str] = []
        agent.start()
        failing = _recording_trait("A", events, fails=True)

        with pytest.raises(RuntimeError, match="A down"):
            agent.add_trait(failing(agent))

        assert events == ["A.start", "A.stop"]
        assert not agent.has(failing)


class TestAgentTraits:
    """Tests for Agent trait management."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        """Create a test agent."""
        from llm_gent.agents.default import Agent as DefaultAgent

        return DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )

    def test_add_trait(self, agent):
        mock_trait = MagicMock()

        agent.add_trait(mock_trait)

        # Trait is registered in agent's registry
        assert agent.has(type(mock_trait))

    def test_add_duplicate_trait_raises(self, agent):
        from llm_gent.core.errors import DuplicateTraitError

        agent.add_trait(_ResourceTrait(agent, MagicMock()))

        with pytest.raises(DuplicateTraitError, match="already registered"):
            agent.add_trait(_ResourceTrait(agent, MagicMock()))

    def test_get_trait(self, agent):
        mock_trait = MagicMock()
        agent.add_trait(mock_trait)

        result = agent.get(type(mock_trait))

        assert result is mock_trait

    def test_get_trait_not_found(self, agent):
        result = agent.get(MagicMock)

        assert result is None

    def test_require_trait_found(self, agent):
        mock_trait = MagicMock()
        agent.add_trait(mock_trait)

        result = agent.require(type(mock_trait))

        assert result is mock_trait

    def test_require_trait_not_found_raises(self, agent):
        from llm_gent.core.errors import TraitNotFoundError

        with pytest.raises(TraitNotFoundError, match="required but not attached"):
            agent.require(MagicMock)


class _Binding:
    """One agent's binding of _PlatformTrait: the agent's data, started and stopped by the agent."""

    def __init__(self, trait: "_PlatformTrait", owner: str) -> None:
        self.trait = trait
        self.owner = owner
        self.started = 0
        self.stopped = 0

    def on_start(self) -> None:
        self.started += 1

    def on_stop(self) -> None:
        self.stopped += 1


class _PlatformTrait:
    """A scoped platform trait: shared resources only; each agent binds it."""

    trait_name = "thing"

    def __init__(self) -> None:
        self.on_start = MagicMock()
        self.on_stop = MagicMock()

    def bind(self, agent: Agent) -> _Binding:
        return _Binding(self, agent.name)


class _SharedTrait:
    """A platform trait that is not scoped: every agent uses the platform's instance."""

    trait_name = "shared"

    def __init__(self) -> None:
        self.on_start = MagicMock()
        self.on_stop = MagicMock()


class TestAgentLookup:
    """agent.get / has / require: the agent's own traits, its bindings and the platform's traits."""

    @pytest.fixture
    def platform(self):
        from llm_gent.core.traits.registry import Registry
        from tests.helpers import make_platform

        lg = MagicMock()
        traits = Registry(lg)
        traits.register(_PlatformTrait())
        traits.register(_SharedTrait())
        return make_platform(lg, traits=traits)

    def _agent(self, platform, name: str = "a") -> Agent:
        return Agent(lg=MagicMock(), platform=platform, config=DotDict(identity={"name": name}))

    def test_the_agents_own_trait(self, platform):
        agent = self._agent(platform)
        trait = MagicMock()
        agent.add_trait(trait)

        assert agent.require(type(trait)) is trait
        assert agent.get(type(trait)) is trait
        assert agent.has(type(trait))

    def test_use_holds_the_agents_binding_of_the_platform_trait(self, platform):
        agent = self._agent(platform, name="jokester")

        agent.use(_PlatformTrait)

        binding = agent.require(_PlatformTrait)
        assert isinstance(binding, _Binding)
        assert binding.owner == "jokester"
        assert binding.trait is platform.traits.require(_PlatformTrait)
        assert agent.get(_PlatformTrait) is binding
        assert agent.has(_PlatformTrait)

    def test_every_agent_holds_its_own_binding(self, platform):
        first, second = self._agent(platform, name="a"), self._agent(platform, name="b")
        first.use(_PlatformTrait)
        second.use(_PlatformTrait)

        assert first.require(_PlatformTrait) is not second.require(_PlatformTrait)

    def test_the_platform_trait_keeps_nothing_per_agent(self, platform):
        trait = platform.traits.require(_PlatformTrait)
        before = dict(vars(trait))

        for i in range(5):
            self._agent(platform, name=f"agent-{i}").use(_PlatformTrait)

        assert vars(trait) == before

    @pytest.mark.parametrize("started", [False, True])
    def test_a_second_use_raises_and_keeps_the_first_binding(self, platform, started):
        from llm_gent.core.errors import DuplicateTraitError

        agent = self._agent(platform)
        agent.use(_PlatformTrait)
        first = agent.require(_PlatformTrait)
        if started:
            agent.start()

        with (
            patch.object(_PlatformTrait, "bind") as bind,
            pytest.raises(DuplicateTraitError, match="_PlatformTrait"),
        ):
            agent.use(_PlatformTrait)

        bind.assert_not_called()
        assert agent.require(_PlatformTrait) is first
        assert first.stopped == 0

    def test_a_binding_that_fails_to_start_is_stopped_and_dropped(self, platform):
        agent = self._agent(platform)
        agent.start()
        binding = _Binding(platform.traits.require(_PlatformTrait), agent.name)
        binding.on_start = MagicMock(side_effect=RuntimeError("bind down"))  # type: ignore[method-assign]

        with (
            patch.object(_PlatformTrait, "bind", return_value=binding),
            pytest.raises(RuntimeError, match="bind down"),
        ):
            agent.use(_PlatformTrait)

        assert binding.stopped == 1
        assert not agent.has(_PlatformTrait)

    def test_the_agent_stops_in_reverse_of_its_start(self, platform):
        events: list[str] = []
        agent = self._agent(platform)
        agent.use(_PlatformTrait)
        binding = agent.require(_PlatformTrait)
        binding.on_start = lambda: events.append("binding.start")  # type: ignore[method-assign]
        binding.on_stop = lambda: events.append("binding.stop")  # type: ignore[method-assign]
        for name in ("A", "B", "C"):
            agent.add_trait(_recording_trait(name, events)(agent))

        agent.start()
        agent.stop()

        started = ["binding.start", "A.start", "B.start", "C.start"]
        assert events == [*started, "C.stop", "B.stop", "A.stop", "binding.stop"]

    def test_the_agent_starts_and_stops_its_bindings(self, platform):
        agent = self._agent(platform)
        agent.use(_PlatformTrait)
        binding = agent.require(_PlatformTrait)

        agent.start()
        agent.stop()

        assert (binding.started, binding.stopped) == (1, 1)

    def test_a_binding_made_while_started_is_started(self, platform):
        agent = self._agent(platform)
        agent.start()

        agent.use(_PlatformTrait)

        assert agent.require(_PlatformTrait).started == 1

    def test_the_agent_never_starts_or_stops_the_platform_trait(self, platform):
        agent = self._agent(platform)
        agent.use(_PlatformTrait)

        agent.start()
        agent.stop()

        trait = platform.traits.require(_PlatformTrait)
        trait.on_start.assert_not_called()
        trait.on_stop.assert_not_called()

    def test_a_scoped_trait_the_agent_does_not_use_is_not_found(self, platform):
        from llm_gent.core.errors import TraitNotFoundError

        agent = self._agent(platform)

        assert agent.get(_PlatformTrait) is None
        assert not agent.has(_PlatformTrait)
        with pytest.raises(
            TraitNotFoundError, match=r"_PlatformTrait is scoped per agent: agent\.use"
        ):
            agent.require(_PlatformTrait)

    def test_a_trait_that_is_not_scoped_is_the_platforms_instance(self, platform):
        agent = self._agent(platform)
        shared = platform.traits.require(_SharedTrait)

        assert agent.require(_SharedTrait) is shared
        assert agent.get(_SharedTrait) is shared
        assert agent.has(_SharedTrait)
        assert agent.require(_SharedTrait) is self._agent(platform, name="b").require(_SharedTrait)

    def test_a_trait_that_is_not_scoped_is_not_used(self, platform):
        agent = self._agent(platform)

        with pytest.raises(TypeError, match="_SharedTrait is not scoped"):
            agent.use(_SharedTrait)

    def test_the_agent_never_starts_or_stops_a_trait_that_is_not_scoped(self, platform):
        agent = self._agent(platform)
        agent.require(_SharedTrait)

        agent.start()
        agent.stop()

        shared = platform.traits.require(_SharedTrait)
        shared.on_start.assert_not_called()
        shared.on_stop.assert_not_called()

    def test_the_agents_own_trait_wins_over_the_platforms(self, platform):
        agent = self._agent(platform)
        own = _SharedTrait()
        agent.add_trait(own)

        assert agent.require(_SharedTrait) is own

    def test_using_a_trait_the_platform_lacks_raises(self, platform):
        from llm_gent.core.errors import TraitNotFoundError

        agent = self._agent(platform)

        with pytest.raises(TraitNotFoundError, match="MagicMock"):
            agent.use(MagicMock)

    def test_the_trait_suffixed_lookups_are_gone(self, platform):
        agent = self._agent(platform)

        assert not hasattr(agent, "get_trait")
        assert not hasattr(agent, "has_trait")
        assert not hasattr(agent, "require_trait")


class TestAgentReplaceTrait:
    """Tests for Agent.replace_trait()."""

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        from llm_gent.agents.default import Agent as DefaultAgent

        return DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )

    def _make_trait(self, agent, **kw):
        return _ResourceTrait(agent, MagicMock(), **kw)

    def test_replace_swaps_registry_entry(self, agent):

        original = self._make_trait(agent)
        agent.add_trait(original)

        replacement = self._make_trait(agent)
        agent.replace_trait(replacement)

        assert agent.require(_ResourceTrait) is replacement

    def test_replace_when_no_existing_trait(self, agent):

        trait = self._make_trait(agent)
        agent.replace_trait(trait)

        assert agent.require(_ResourceTrait) is trait

    def test_replace_calls_on_start_when_agent_started(self, agent):
        original = self._make_trait(agent)
        agent.add_trait(original)
        agent._started = True

        replacement = self._make_trait(agent)
        with patch.object(replacement, "on_start") as mock_start:
            agent.replace_trait(replacement)

        mock_start.assert_called_once()

    def test_replace_does_not_call_on_start_when_agent_not_started(self, agent):
        original = self._make_trait(agent)
        agent.add_trait(original)

        replacement = self._make_trait(agent)
        with patch.object(replacement, "on_start") as mock_start:
            agent.replace_trait(replacement)

        mock_start.assert_not_called()

    def test_replace_auto_stops_old_trait_when_started(self, agent):
        """Registry membership implies agent-managed lifecycle."""
        original = self._make_trait(agent)
        agent.add_trait(original)
        agent._started = True

        replacement = self._make_trait(agent)
        with patch.object(original, "on_stop") as mock_stop:
            agent.replace_trait(replacement)

        mock_stop.assert_called_once()

    def test_replace_does_not_stop_old_when_agent_not_started(self, agent):
        """Only started agents own the lifecycle — unstarted swap is registry-only."""
        original = self._make_trait(agent)
        agent.add_trait(original)

        replacement = self._make_trait(agent)
        with patch.object(original, "on_stop") as mock_stop:
            agent.replace_trait(replacement)

        mock_stop.assert_not_called()

    def test_replace_starts_new_before_stopping_old(self, agent):
        """Order matters: a failing new.on_start must not have already closed old."""
        original = self._make_trait(agent)
        agent.add_trait(original)
        agent._started = True

        events: list[str] = []
        replacement = self._make_trait(agent)
        with (
            patch.object(original, "on_stop", side_effect=lambda: events.append("old.on_stop")),
            patch.object(
                replacement, "on_start", side_effect=lambda: events.append("new.on_start")
            ),
        ):
            agent.replace_trait(replacement)

        assert events == ["new.on_start", "old.on_stop"]

    def test_replace_rolls_back_on_new_on_start_failure(self, agent):
        """On failure the old trait is restored and NOT stopped."""

        original = self._make_trait(agent)
        agent.add_trait(original)
        agent._started = True

        replacement = self._make_trait(agent)
        with (
            patch.object(replacement, "on_start", side_effect=RuntimeError("boom")),
            patch.object(original, "on_stop") as mock_stop,
            pytest.raises(RuntimeError, match="boom"),
        ):
            agent.replace_trait(replacement)

        assert agent.require(_ResourceTrait) is original
        mock_stop.assert_not_called()

    def test_replace_swallows_old_on_stop_exception(self, agent, mock_logger):
        """A failing old.on_stop is logged but does not propagate."""
        original = self._make_trait(agent)
        agent.add_trait(original)
        agent._started = True

        replacement = self._make_trait(agent)
        with patch.object(original, "on_stop", side_effect=RuntimeError("cleanup failed")):
            agent.replace_trait(replacement)  # should not raise

        mock_logger.warning.assert_called_once()

    def test_with_resource_and_replace_end_to_end(self, agent):
        """with_<resource> + replace_trait: registered old trait's on_stop fires,
        closing its owned resource; the injected new resource is untouched."""
        old_resource = MagicMock()
        original = _ResourceTrait(agent, old_resource, owns_resource=True)
        agent.add_trait(original)
        agent._started = True

        new_resource = MagicMock()
        replacement = agent.require(_ResourceTrait).with_resource(new_resource)
        agent.replace_trait(replacement)

        assert agent.require(_ResourceTrait) is replacement
        # Old trait owned its resource — replace_trait's auto-stop closed it.
        old_resource.close.assert_called_once()
        # New trait was built with owns_resource=False; its resource is external.
        new_resource.close.assert_not_called()


class TestAgentExecution:
    """Tests for Agent execution methods."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_saia_trait(self):
        """Create mock SAIATrait with mock SAIA."""
        from llm_saia import TaskResult

        from llm_gent.core.traits.builtin.saia import SAIATrait

        # Create mock saia that returns TaskResult
        mock_saia = MagicMock()
        mock_saia.complete = AsyncMock(
            return_value=TaskResult(completed=True, output="Test output", iterations=1, history=[])
        )

        # Create trait with correct type for registry lookup
        # Use _saia (the backing field) instead of saia (the property)
        return create_mock_trait(SAIATrait, _saia=mock_saia)

    @pytest.fixture
    def agent_with_saia(self, mock_logger, mock_saia_trait):
        """Create agent with SAIATrait attached."""
        from llm_gent.agents.default import ChatAgent
        from llm_gent.agents.default.factory import Factory
        from tests.helpers import make_platform

        agent = ChatAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt="Test task"),
            session_storage=None,
        )
        agent._traits.register(mock_saia_trait)

        # Register event handlers using factory's method (handlers now live in factory)
        factory = Factory(make_platform(mock_logger))
        factory._configure_event_handlers(agent, {})  # Empty config = use defaults

        return agent

    def test_run_once_without_saia_trait_fails(self, mock_logger):
        from llm_gent.agents.default import Agent as DefaultAgent

        agent = DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt="Test task"),
        )

        result = agent.run_once()

        assert result.success is False
        assert "SAIATrait not attached" in result.content

    def test_run_once_without_default_prompt_fails(self, mock_logger, mock_saia_trait):
        from llm_gent.agents.default import Agent as DefaultAgent

        agent = DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )
        agent._traits.register(mock_saia_trait)

        result = agent.run_once()

        assert result.success is False
        assert "No default prompt" in result.content

    def test_run_once_increments_cycle_count(self, agent_with_saia):
        assert agent_with_saia.cycle_count == 0

        agent_with_saia.run_once()
        assert agent_with_saia.cycle_count == 1

        agent_with_saia.run_once()
        assert agent_with_saia.cycle_count == 2

    def test_run_once_returns_result(self, agent_with_saia, mock_saia_trait):
        from llm_saia import TaskResult

        mock_saia_trait.saia.complete = AsyncMock(
            return_value=TaskResult(
                completed=True, output="Task completed", iterations=3, history=[]
            )
        )

        result = agent_with_saia.run_once()

        assert result.success is True
        assert result.content == "Task completed"
        assert result.iterations == 3

    def test_ask_returns_response(self, agent_with_saia, mock_saia_trait):
        from llm_saia import TaskResult

        mock_saia_trait.saia.complete = AsyncMock(
            return_value=TaskResult(
                completed=True, output="The answer is 42", iterations=1, history=[]
            )
        )

        result = agent_with_saia.ask("What is the answer?")

        assert result.content == "The answer is 42"
        assert result.session

    def test_ask_without_saia_trait_returns_error(self, mock_logger):
        from llm_gent.agents.default import ChatAgent

        agent = ChatAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
            session_storage=None,
        )

        result = agent.ask("Question?")

        assert "SAIATrait not attached" in result.content

    def test_run_once_incomplete_when_saia_not_completed(self, agent_with_saia, mock_saia_trait):
        """Agent reports failure when SAIA returns completed=False (TEXT_ONLY scenario).

        This tests the case where the LLM responds with text but doesn't call
        the terminal tool (complete_task). SAIA sets completed=False in this case.
        """
        from llm_saia import TaskResult

        mock_saia_trait.saia.complete = AsyncMock(
            return_value=TaskResult(
                completed=False,  # LLM didn't call terminal tool
                output="I tried but couldn't complete the task",
                iterations=5,  # Hit max iterations
                history=[],
            )
        )

        result = agent_with_saia.run_once()

        assert result.success is False
        assert result.content == "I tried but couldn't complete the task"
        assert result.iterations == 5


class TestAgentRecentResults:
    """Tests for Agent.get_recent_results() method."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        """Create a test agent."""
        from llm_gent.agents.default import Agent as DefaultAgent

        return DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )

    def test_get_recent_results_empty(self, agent):
        assert agent.get_recent_results() == []

    def test_get_recent_results_respects_limit(self, agent):
        # Add some results manually
        for i in range(5):
            agent._recent_results.append(
                ExecutionResult(success=True, content=f"Result {i}", iterations=1)
            )

        results = agent.get_recent_results(limit=3)
        assert len(results) == 3
        assert results[0].content == "Result 2"
        assert results[2].content == "Result 4"


class TestAgentRecordFeedback:
    """Tests for Agent.record_feedback() method."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    def test_record_feedback_is_noop(self, mock_logger):
        """Default agent's record_feedback is a no-op."""
        from llm_gent.agents.default import Agent as DefaultAgent

        agent = DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )

        # Should not raise
        agent.record_feedback("Some feedback message")


class TestSubstituteVariables:
    """Tests for variable substitution in templates."""

    def test_substitute_from_dict(self):
        """Variables are substituted from provided dict."""
        text = "Hello {{NAME}}, welcome to {{PLACE}}!"
        variables = {"NAME": "Alice", "PLACE": "Wonderland"}

        result = _substitute_variables(text, variables)

        assert result == "Hello Alice, welcome to Wonderland!"

    def test_substitute_multiple_same_variable(self):
        """Same variable can appear multiple times."""
        text = "{{VAR}} and {{VAR}} again"
        variables = {"VAR": "value"}

        result = _substitute_variables(text, variables)

        assert result == "value and value again"

    def test_substitute_from_env(self, monkeypatch):
        """Falls back to environment variables if not in dict."""
        monkeypatch.setenv("TEST_ENV_VAR", "from_env")
        text = "Value: {{TEST_ENV_VAR}}"

        result = _substitute_variables(text, {})

        assert result == "Value: from_env"

    def test_substitute_dict_overrides_env(self, monkeypatch):
        """Dict values take precedence over environment variables."""
        monkeypatch.setenv("MY_VAR", "from_env")
        text = "Value: {{MY_VAR}}"
        variables = {"MY_VAR": "from_dict"}

        result = _substitute_variables(text, variables)

        assert result == "Value: from_dict"

    def test_substitute_missing_raises(self):
        """Missing variable raises ValueError."""
        text = "Hello {{MISSING_VAR}}"

        with pytest.raises(ValueError, match="Variable {{MISSING_VAR}} not found"):
            _substitute_variables(text, {})

    def test_substitute_no_variables(self):
        """Text without variables is returned unchanged."""
        text = "No variables here"

        result = _substitute_variables(text, {})

        assert result == "No variables here"

    def test_substitute_preserves_partial_braces(self):
        """Partial brace patterns are not substituted."""
        text = "Single {BRACE} and {{VALID}}"
        variables = {"VALID": "replaced"}

        result = _substitute_variables(text, variables)

        assert result == "Single {BRACE} and replaced"


class TestFactorySystemPrompt:
    """Tests for Factory._build_system_prompt() method."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def factory(self, mock_logger):
        """Create Factory with minimal config."""
        return self._create_factory(mock_logger)

    def _create_factory(self, mock_logger):
        """Create Factory instance."""
        from llm_gent.agents.default import Factory
        from tests.helpers import make_platform

        return Factory(platform=make_platform(mock_logger))

    def test_system_prompt_with_identity_only(self, mock_logger):
        """System prompt contains identity when only identity is configured."""
        from llm_gent.agents.default import Agent as DefaultAgent
        from llm_gent.core.traits.builtin.directive import DirectiveTrait

        factory = self._create_factory(mock_logger)
        agent = DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )
        agent.add_trait(DirectiveTrait(agent, "You are a helpful assistant."))

        system_prompt = factory._build_system_prompt(agent)

        assert system_prompt == "You are a helpful assistant."

    def test_system_prompt_with_method_only(self, mock_logger):
        """System prompt contains method when only method is configured."""
        from llm_gent.agents.default import Agent as DefaultAgent
        from llm_gent.core.traits.builtin.directive import MethodTrait

        factory = self._create_factory(mock_logger)
        agent = DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )
        agent.add_trait(MethodTrait(agent, "- Step 1\n- Step 2"))

        system_prompt = factory._build_system_prompt(agent)

        assert system_prompt == "## Method\n- Step 1\n- Step 2"

    def test_system_prompt_with_identity_and_method(self, mock_logger):
        """System prompt combines identity and method."""
        from llm_gent.agents.default import Agent as DefaultAgent
        from llm_gent.core.traits.builtin.directive import DirectiveTrait, MethodTrait

        factory = self._create_factory(mock_logger)
        agent = DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )
        agent.add_trait(DirectiveTrait(agent, "You are a helpful assistant."))
        agent.add_trait(MethodTrait(agent, "- Step 1\n- Step 2"))

        system_prompt = factory._build_system_prompt(agent)

        assert system_prompt == "You are a helpful assistant.\n\n## Method\n- Step 1\n- Step 2"

    def test_system_prompt_empty_when_no_traits(self, mock_logger):
        """System prompt is None when no identity or method traits."""
        from llm_gent.agents.default import Agent as DefaultAgent

        factory = self._create_factory(mock_logger)
        agent = DefaultAgent(
            lg=mock_logger,
            platform=make_platform(mock_logger),
            config=DotDict(identity={"name": "test"}, default_prompt=""),
        )

        system_prompt = factory._build_system_prompt(agent)

        assert system_prompt is None

    def test_factory_create_passes_system_prompt_to_saia(self, mock_logger):
        """Factory.create() passes system prompt to SAIATrait."""
        from llm_gent.core.traits.builtin.saia import SAIATrait

        factory = self._create_factory(mock_logger)
        config = {
            "identity": {"name": "test-agent"},
            "directive": "You are a code analyst.",
            "method": "- Analyze carefully",
            "task": {"description": "Analyze the code."},
            "traits": {"required": ["directive", "method"]},
        }

        agent = factory.create(config)
        saia_trait = agent.get(SAIATrait)

        assert saia_trait is not None
        expected = "You are a code analyst.\n\n## Method\n- Analyze carefully"
        assert saia_trait.config.system_prompt == expected


class TestIdentity:
    """Tests for Identity construction ergonomics."""

    def test_name_only_defaults_context_key_to_name(self):
        identity = Identity("jokester")

        assert identity.name == "jokester"
        assert identity.context_key == "jokester"

    def test_explicit_context_key_preserved(self):
        identity = Identity("joke-master", context_key="jokester")

        assert identity.name == "joke-master"
        assert identity.context_key == "jokester"

    def test_empty_context_key_falls_back_to_name(self):
        identity = Identity(name="foo", context_key="")

        assert identity.context_key == "foo"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Identity("")

    def test_from_name_still_works(self):
        identity = Identity.from_name("jokester")

        assert identity.name == "jokester"
        assert identity.context_key == "jokester"

    def test_frozen_dataclass(self):
        import dataclasses

        identity = Identity("foo")

        with pytest.raises(dataclasses.FrozenInstanceError):
            identity.name = "bar"  # type: ignore[misc]
