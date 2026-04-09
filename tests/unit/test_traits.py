"""Tests for traits."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llm_gent import (
    BaseTrait,
    Directive,
    DirectiveTrait,
    MethodTrait,
    StructuredOutputError,
)
from llm_gent.core.llm.types import Message
from llm_gent.core.traits.builtin.llm import LLMTrait


pytestmark = pytest.mark.unit


class TestBaseTrait:
    """Tests for BaseTrait."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        return MagicMock()

    def test_init(self, mock_agent):
        """BaseTrait requires agent in constructor."""
        trait = BaseTrait(mock_agent)

        assert trait._agent == mock_agent

    def test_agent_property_returns_agent(self, mock_agent):
        """Agent property returns the agent passed in constructor."""
        trait = BaseTrait(mock_agent)

        assert trait.agent == mock_agent

    def test_lifecycle_methods_exist(self, mock_agent):
        """Lifecycle methods can be called - no-op by default."""
        trait = BaseTrait(mock_agent)

        # Should not raise - these are no-op by default
        trait.on_start()
        trait.on_stop()


class TestCustomTrait:
    """Tests for custom traits using BaseTrait."""

    def test_custom_trait_inherits_behavior(self):
        """Custom traits can extend BaseTrait."""

        class MyTrait(BaseTrait):
            def __init__(self, agent, value: str) -> None:
                super().__init__(agent)
                self.value = value

            def get_value(self) -> str:
                return f"{self.agent.name}: {self.value}"

        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        trait = MyTrait(mock_agent, "test")

        assert trait.get_value() == "test-agent: test"


class TestDirective:
    """Tests for Directive."""

    def test_identity_with_prompt(self):
        identity = Directive(prompt="You are a code reviewer.")

        assert identity.prompt == "You are a code reviewer."
        assert identity.extensions == {}

    def test_identity_with_extensions(self):
        identity = Directive(
            prompt="You are a code reviewer.",
            extensions={"custom_field": "value"},
        )

        assert identity.prompt == "You are a code reviewer."
        assert identity.extensions == {"custom_field": "value"}


class TestDirectiveTrait:
    """Tests for DirectiveTrait."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        return MagicMock()

    def test_init_with_identity_object(self, mock_agent):
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(mock_agent, identity)

        assert trait.directive == identity

    def test_init_with_string(self, mock_agent):
        """DirectiveTrait can be initialized with a string."""
        trait = DirectiveTrait(mock_agent, "You are a code reviewer.")

        assert trait.directive.prompt == "You are a code reviewer."

    def test_agent_assigned(self, mock_agent):
        """Trait receives agent on construction."""
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(mock_agent, identity)

        assert trait.agent == mock_agent

    def test_build_prompt(self, mock_agent):
        identity = Directive(prompt="You are a code reviewer. Be critical.")
        trait = DirectiveTrait(mock_agent, identity)

        result = trait.build_prompt("Base system prompt.")

        # Directive is prepended
        assert result.startswith("You are a code reviewer.")
        assert "Base system prompt." in result


class TestMethodTrait:
    """Tests for MethodTrait."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        return MagicMock()

    def test_init(self, mock_agent):
        trait = MethodTrait(mock_agent, "- Step 1\n- Step 2")

        assert trait.method == "- Step 1\n- Step 2"

    def test_build_prompt(self, mock_agent):
        trait = MethodTrait(mock_agent, "- Step 1\n- Step 2")

        result = trait.build_prompt("Base prompt.")

        assert "Base prompt." in result
        assert "## Method" in result
        assert "- Step 1\n- Step 2" in result

    def test_update(self, mock_agent):
        trait = MethodTrait(mock_agent, "Original method")

        trait.update("Updated method")

        assert trait.method == "Updated method"


class TestAgentTraits:
    """Tests for Agent trait support."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock Logger."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_logger):
        """Create a test agent."""
        from appinfra import DotDict

        from llm_gent.agents.default import Agent as DefaultAgent

        config = DotDict(identity={"name": "test"}, default_prompt="")
        return DefaultAgent(lg=mock_logger, config=config)

    def test_add_trait(self, agent):
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(agent, identity)
        agent.add_trait(trait)

        assert agent.has_trait(DirectiveTrait)
        assert agent.get_trait(DirectiveTrait) == trait

    def test_add_trait_attaches(self, agent):
        identity = Directive(prompt="Test identity")
        trait = DirectiveTrait(agent, identity)
        agent.add_trait(trait)

        assert trait.agent == agent

    def test_add_duplicate_trait_raises(self, agent):
        from llm_gent.core.errors import DuplicateTraitError

        identity = Directive(prompt="Test identity")
        agent.add_trait(DirectiveTrait(agent, identity))

        with pytest.raises(DuplicateTraitError, match="already registered"):
            agent.add_trait(DirectiveTrait(agent, identity))

    def test_get_trait_returns_none_if_not_added(self, agent):
        assert agent.get_trait(DirectiveTrait) is None

    def test_has_trait_returns_false_if_not_added(self, agent):
        assert not agent.has_trait(DirectiveTrait)


class TestLLMTraitStructuredOutput:
    """Tests for LLMTrait structured output support."""

    class Answer(BaseModel):
        """Test schema for structured output."""

        answer: str
        confidence: float

    @pytest.fixture
    def trait(self):
        """Create LLMTrait with mocked router."""
        trait = LLMTrait(MagicMock(), {})
        trait._router = MagicMock()
        return trait

    def test_structured_output_basic(self, trait):
        """Test successful structured output parsing."""
        # Mock response with valid JSON
        mock_response = MagicMock()
        mock_response.content = '{"answer": "42", "confidence": 0.95}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [Message(role="user", content="What is the meaning of life?")]
        result = trait.complete(messages, output_schema=self.Answer)

        assert result.parsed is not None
        assert isinstance(result.parsed, self.Answer)
        assert result.parsed.answer == "42"
        assert result.parsed.confidence == 0.95

    def test_structured_output_injects_schema_prompt(self, trait):
        """Test that schema prompt is injected into system message."""
        mock_response = MagicMock()
        mock_response.content = '{"answer": "test", "confidence": 1.0}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Question?"),
        ]
        trait.complete(messages, output_schema=self.Answer)

        # Check the messages passed to chat
        call_args = trait._router.chat.call_args
        sent_messages = call_args.kwargs["messages"]

        # Schema prompt should be appended to system message
        assert "You are helpful." in sent_messages[0]["content"]
        assert "json" in sent_messages[0]["content"].lower()
        assert "answer" in sent_messages[0]["content"]

    def test_structured_output_creates_system_message_if_none(self, trait):
        """Test that schema prompt creates system message if none exists."""
        mock_response = MagicMock()
        mock_response.content = '{"answer": "test", "confidence": 1.0}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [Message(role="user", content="Question?")]
        trait.complete(messages, output_schema=self.Answer)

        call_args = trait._router.chat.call_args
        sent_messages = call_args.kwargs["messages"]

        # First message should be a system message with schema
        assert sent_messages[0]["role"] == "system"
        assert "json" in sent_messages[0]["content"].lower()

    def test_structured_output_enables_json_mode(self, trait):
        """Test that JSON mode is enabled via extra_body."""
        mock_response = MagicMock()
        mock_response.content = '{"answer": "test", "confidence": 1.0}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [Message(role="user", content="Question?")]
        trait.complete(messages, output_schema=self.Answer)

        call_args = trait._router.chat.call_args
        extra_body = call_args.kwargs.get("extra_body")

        assert extra_body is not None
        assert extra_body == {"response_format": {"type": "json_object"}}

    def test_structured_output_invalid_json_raises(self, trait):
        """Test that invalid JSON raises StructuredOutputError."""
        mock_response = MagicMock()
        mock_response.content = "not valid json {"
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [Message(role="user", content="Question?")]

        with pytest.raises(StructuredOutputError, match="Invalid JSON"):
            trait.complete(messages, output_schema=self.Answer)

    def test_structured_output_schema_mismatch_raises(self, trait):
        """Test that schema mismatch raises StructuredOutputError."""
        mock_response = MagicMock()
        # Missing required 'confidence' field
        mock_response.content = '{"answer": "test"}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [Message(role="user", content="Question?")]

        with pytest.raises(StructuredOutputError, match="doesn't match schema"):
            trait.complete(messages, output_schema=self.Answer)

    def test_structured_output_wrong_type_raises(self, trait):
        """Test that wrong field type raises StructuredOutputError."""
        mock_response = MagicMock()
        # confidence should be float, not string
        mock_response.content = '{"answer": "test", "confidence": "high"}'
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [Message(role="user", content="Question?")]

        with pytest.raises(StructuredOutputError, match="doesn't match schema"):
            trait.complete(messages, output_schema=self.Answer)

    def test_no_schema_parsed_is_none(self, trait):
        """Test backward compatibility - no schema means parsed is None."""
        mock_response = MagicMock()
        mock_response.content = "Just plain text response"
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50)
        mock_response.tool_calls = None
        mock_response.adapter = None
        trait._router.chat.return_value = mock_response

        messages = [Message(role="user", content="Question?")]
        result = trait.complete(messages)

        assert result.parsed is None
        assert result.content == "Just plain text response"

    def test_tools_and_schema_raises(self, trait):
        """Test that using both tools and output_schema raises ValueError."""
        messages = [Message(role="user", content="Question?")]
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]

        with pytest.raises(ValueError, match="Cannot use both tools and output_schema"):
            trait.complete(messages, tools=tools, output_schema=self.Answer)


class TestResolveLLMDefaults:
    """Tests for _resolve_llm_defaults function."""

    def test_single_backend_config(self):
        from appinfra import DotDict

        from llm_gent.core.traits.builtin.llm import _resolve_llm_defaults

        config = DotDict({"model": "qwen2.5", "temperature": 0.3, "adapter": "lora1"})
        result = _resolve_llm_defaults(config)

        assert result["model"] == "qwen2.5"
        assert result["temperature"] == 0.3
        assert result["adapter"] == "lora1"

    def test_single_backend_defaults(self):
        from appinfra import DotDict

        from llm_gent.core.traits.builtin.llm import _resolve_llm_defaults

        config = DotDict({})
        result = _resolve_llm_defaults(config)

        assert result["model"] == "default"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] is None

    def test_multi_backend_with_default(self):
        from appinfra import DotDict

        from llm_gent.core.traits.builtin.llm import _resolve_llm_defaults

        config = DotDict(
            {
                "default": "cloud",
                "backends": {
                    "local": {"model": "qwen2.5", "temperature": 0.3},
                    "cloud": {"model": "claude-3", "temperature": 0.5},
                },
            }
        )
        result = _resolve_llm_defaults(config)

        assert result["model"] == "claude-3"
        assert result["temperature"] == 0.5

    def test_multi_backend_no_default_picks_first(self):
        from appinfra import DotDict

        from llm_gent.core.traits.builtin.llm import _resolve_llm_defaults

        config = DotDict(
            {
                "backends": {
                    "local": {"model": "qwen2.5"},
                    "cloud": {"model": "claude-3"},
                },
            }
        )
        result = _resolve_llm_defaults(config)
        # Should pick first backend
        assert result["model"] in ("qwen2.5", "claude-3")


class TestLLMTraitLifecycle:
    """Tests for on_start/on_stop."""

    def test_on_start_creates_router(self):
        from unittest.mock import patch

        from appinfra import DotDict

        config = DotDict({"default": "test", "backends": {"test": {"type": "openai_compatible"}}})
        trait = LLMTrait(MagicMock(), config)

        with patch("llm_gent.core.traits.builtin.llm.LLMClientFactory") as mock_factory:
            mock_client = MagicMock()
            mock_factory.return_value.from_config.return_value = mock_client
            trait.on_start()

            assert trait._router is mock_client

    def test_on_stop_closes_router(self):
        trait = LLMTrait(MagicMock(), {})
        trait._router = MagicMock()
        trait.on_stop()

        trait._router is None  # noqa: B015 — checking side effect happened
        # Actually check it was set to None
        assert trait._router is None

    def test_on_stop_safe_when_no_router(self):
        trait = LLMTrait(MagicMock(), {})
        trait.on_stop()  # Should not raise


class TestLLMTraitRouterProperty:
    """Tests for router property."""

    def test_raises_when_not_started(self):
        trait = LLMTrait(MagicMock(), {})
        with pytest.raises(RuntimeError, match="not started"):
            _ = trait.router

    def test_returns_client_when_started(self):
        trait = LLMTrait(MagicMock(), {})
        mock_client = MagicMock()
        trait._router = mock_client
        assert trait.router is mock_client


class TestLLMTraitAdapterProperty:
    """Tests for adapter property."""

    def test_adapter_none_by_default(self):
        trait = LLMTrait(MagicMock(), {})
        assert trait.adapter is None

    def test_adapter_from_defaults(self):
        trait = LLMTrait(MagicMock(), {})
        trait._defaults = {"adapter": "lora-v1"}
        assert trait.adapter == "lora-v1"


class TestMessagesToDicts:
    """Tests for _messages_to_dicts."""

    def test_basic_messages(self):
        trait = LLMTrait(MagicMock(), {})
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        result = trait._messages_to_dicts(messages)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}

    def test_message_with_tool_calls(self):
        trait = LLMTrait(MagicMock(), {})
        messages = [Message(role="assistant", content="", tool_calls=[{"id": "tc1"}])]
        result = trait._messages_to_dicts(messages)
        assert result[0]["tool_calls"] == [{"id": "tc1"}]

    def test_message_with_tool_call_id(self):
        trait = LLMTrait(MagicMock(), {})
        messages = [Message(role="tool", content="result", tool_call_id="tc1")]
        result = trait._messages_to_dicts(messages)
        assert result[0]["tool_call_id"] == "tc1"

    def test_omits_none_fields(self):
        trait = LLMTrait(MagicMock(), {})
        messages = [Message(role="user", content="Hi")]
        result = trait._messages_to_dicts(messages)
        assert "tool_calls" not in result[0]
        assert "tool_call_id" not in result[0]


class TestExtractTokens:
    """Tests for _extract_tokens."""

    def test_no_usage(self):
        trait = LLMTrait(MagicMock(), {})
        response = MagicMock()
        response.usage = None
        assert trait._extract_tokens(response) == 0

    def test_total_tokens(self):
        trait = LLMTrait(MagicMock(), {})
        response = MagicMock()
        response.usage.total_tokens = 150
        assert trait._extract_tokens(response) == 150

    def test_fallback_prompt_plus_completion(self):
        trait = LLMTrait(MagicMock(), {})
        response = MagicMock()
        response.usage.total_tokens = 0
        response.usage.prompt_tokens = 80
        response.usage.completion_tokens = 40
        assert trait._extract_tokens(response) == 120

    def test_fallback_with_none_components(self):
        trait = LLMTrait(MagicMock(), {})
        response = MagicMock()
        response.usage.total_tokens = 0
        response.usage.prompt_tokens = None
        response.usage.completion_tokens = 50
        assert trait._extract_tokens(response) == 50


class TestCheckAdapterFallback:
    """Tests for _check_adapter_fallback."""

    def test_no_adapter_noop(self):
        trait = LLMTrait(MagicMock(), {})
        response = MagicMock()
        response.adapter = None
        trait._check_adapter_fallback(response)  # Should not raise

    def test_no_fallback_noop(self):
        trait = LLMTrait(MagicMock(), {})
        response = MagicMock()
        response.adapter.fallback = False
        trait._check_adapter_fallback(response)  # Should not raise

    def test_first_fallback_warns(self):
        agent = MagicMock()
        trait = LLMTrait(agent, {})
        trait._last_adapter_fallback_warning = 0.0

        response = MagicMock()
        response.adapter.fallback = True

        trait._check_adapter_fallback(response)
        agent.lg.warning.assert_called_once()

    def test_subsequent_fallback_debugs(self):
        import time

        agent = MagicMock()
        trait = LLMTrait(agent, {})
        trait._last_adapter_fallback_warning = time.monotonic()  # Just warned

        response = MagicMock()
        response.adapter.fallback = True

        trait._check_adapter_fallback(response)
        agent.lg.debug.assert_called_once()
        agent.lg.warning.assert_not_called()


class TestLLMTraitBackendAdapter:
    """Tests for LLMTraitBackend wrapper."""

    def test_complete_delegates(self):
        from llm_gent.core.traits.builtin.llm import LLMTraitBackend

        mock_trait = MagicMock()
        mock_trait.complete.return_value = MagicMock()

        backend = LLMTraitBackend(mock_trait)
        messages = [Message(role="user", content="Hi")]
        backend.complete(messages, model="test")

        mock_trait.complete.assert_called_once_with(
            messages=messages,
            model="test",
            temperature=None,
            max_tokens=None,
            tools=None,
            adapter=None,
        )

    def test_load_adapter_raises(self):
        from llm_gent.core.traits.builtin.llm import LLMTraitBackend

        backend = LLMTraitBackend(MagicMock())
        with pytest.raises(NotImplementedError):
            backend.load_adapter("path")

    def test_unload_adapter_raises(self):
        from llm_gent.core.traits.builtin.llm import LLMTraitBackend

        backend = LLMTraitBackend(MagicMock())
        with pytest.raises(NotImplementedError):
            backend.unload_adapter()
