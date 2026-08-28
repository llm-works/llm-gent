# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for Agent base class and agents.default.Agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from appinfra import DotDict

from llm_gent.core.agent import Agent, ExecutionResult
from llm_gent.core.agent.helpers import _substitute_variables
from tests.helpers import create_mock_trait


pytestmark = pytest.mark.unit


class TestAgentBaseClass:
    """Tests for the abstract Agent base class."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    def test_agent_is_abstract(self, mock_logger):
        """Agent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Agent(lg=mock_logger)

    def test_agent_subclass_must_implement_abstract_methods(self, mock_logger):
        """Subclass must implement all abstract methods."""

        class IncompleteAgent(Agent):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAgent(lg=mock_logger)

    def test_agent_subclass_can_be_instantiated(self, mock_logger):
        """Complete subclass can be instantiated."""

        class CompleteAgent(Agent):
            def start(self) -> None:
                self._start_traits()
                self._started = True

            def stop(self) -> None:
                self._stop_traits()
                self._started = False

            def run_once(self) -> ExecutionResult:
                return ExecutionResult(success=True, content="done")

            def ask(self, question: str) -> str:
                return "answer"

            def record_feedback(self, message: str) -> None:
                pass

            def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
                return []

        config = DotDict(identity={"name": "test"})
        agent = CompleteAgent(lg=mock_logger, config=config)
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
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
        )

    def test_add_trait(self, agent):
        mock_trait = MagicMock()

        agent.add_trait(mock_trait)

        # Trait is registered in agent's registry
        assert agent.has_trait(type(mock_trait))

    def test_add_duplicate_trait_raises(self, agent):
        from llm_gent.core.errors import DuplicateTraitError
        from llm_gent.core.traits.builtin.llm import LLMTrait

        trait1 = LLMTrait(agent, {})
        agent.add_trait(trait1)

        with pytest.raises(DuplicateTraitError, match="already registered"):
            trait2 = LLMTrait(agent, {})
            agent.add_trait(trait2)

    def test_get_trait(self, agent):
        mock_trait = MagicMock()
        agent.add_trait(mock_trait)

        result = agent.get_trait(type(mock_trait))

        assert result is mock_trait

    def test_get_trait_not_found(self, agent):
        result = agent.get_trait(MagicMock)

        assert result is None

    def test_require_trait_found(self, agent):
        mock_trait = MagicMock()
        agent.add_trait(mock_trait)

        result = agent.require_trait(type(mock_trait))

        assert result is mock_trait

    def test_require_trait_not_found_raises(self, agent):
        from llm_gent.core.errors import TraitNotFoundError

        with pytest.raises(TraitNotFoundError, match="required but not attached"):
            agent.require_trait(MagicMock)


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
        trait = create_mock_trait(SAIATrait, _saia=mock_saia)
        return trait

    @pytest.fixture
    def agent_with_saia(self, mock_logger, mock_saia_trait):
        """Create agent with SAIATrait attached."""
        from llm_gent.agents.default import Agent as DefaultAgent
        from llm_gent.agents.default.factory import Factory
        from llm_gent.core.platform import PlatformContext

        agent = DefaultAgent(
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="Test task")
        )
        agent._traits.register(mock_saia_trait)

        # Register event handlers using factory's method (handlers now live in factory)
        # Create a minimal mock platform just for the logger
        mock_platform = MagicMock(spec=PlatformContext)
        mock_platform.logger = mock_logger
        factory = Factory(mock_platform)
        factory._configure_event_handlers(agent, {})  # Empty config = use defaults

        return agent

    def test_run_once_without_saia_trait_fails(self, mock_logger):
        from llm_gent.agents.default import Agent as DefaultAgent

        agent = DefaultAgent(
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="Test task")
        )

        result = agent.run_once()

        assert result.success is False
        assert "SAIATrait not attached" in result.content

    def test_run_once_without_default_prompt_fails(self, mock_logger, mock_saia_trait):
        from llm_gent.agents.default import Agent as DefaultAgent

        agent = DefaultAgent(
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
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

        assert result == "The answer is 42"

    def test_ask_without_saia_trait_returns_error(self, mock_logger):
        from llm_gent.agents.default import Agent as DefaultAgent

        agent = DefaultAgent(
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
        )

        result = agent.ask("Question?")

        assert "SAIATrait not attached" in result

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
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
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
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
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
        from unittest.mock import MagicMock

        from llm_gent.agents.default import Factory
        from llm_gent.core.traits.factory import Factory as TraitFactory

        # Create mock platform context
        mock_platform = MagicMock()
        mock_platform.logger = mock_logger
        mock_platform.llm_config.return_value = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                    "model": "test",
                }
            },
        }
        # Create a real trait factory for the mock platform
        mock_platform.trait_factory = TraitFactory(mock_platform)
        return Factory(platform=mock_platform)

    def test_system_prompt_with_identity_only(self, mock_logger):
        """System prompt contains identity when only identity is configured."""
        from llm_gent.agents.default import Agent as DefaultAgent
        from llm_gent.core.traits.builtin.directive import DirectiveTrait

        factory = self._create_factory(mock_logger)
        agent = DefaultAgent(
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
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
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
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
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
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
            lg=mock_logger, config=DotDict(identity={"name": "test"}, default_prompt="")
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
        saia_trait = agent.get_trait(SAIATrait)

        assert saia_trait is not None
        expected = "You are a code analyst.\n\n## Method\n- Analyze carefully"
        assert saia_trait.config.system_prompt == expected
