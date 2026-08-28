# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for agent infrastructure: platform, helpers, dispatcher, caller, factory."""

from unittest.mock import MagicMock

import pytest
from appinfra import DotDict

from llm_gent.core.agent.factory import Factory
from llm_gent.core.agent.helpers import _substitute_in_dict
from llm_gent.core.dispatcher import Dispatcher
from llm_gent.core.llm.caller import LLMCaller
from llm_gent.core.platform import PlatformContext


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# PlatformContext
# ---------------------------------------------------------------------------


class TestPlatformContext:
    def test_init(self):
        lg = MagicMock()
        ctx = PlatformContext(lg, config={"llm": {"model": "test"}})

        assert ctx.logger is lg
        assert ctx.config == {"llm": {"model": "test"}}
        assert ctx.trait_factory is not None
        assert ctx.tool_factory is not None

    def test_from_config(self):
        lg = MagicMock()
        ctx = PlatformContext.from_config(
            lg, llm_config={"model": "test"}, learn_config={"db": "x"}
        )

        assert ctx.llm_config() == {"model": "test"}
        assert ctx.learn_config() == {"db": "x"}

    def test_llm_config_default(self):
        ctx = PlatformContext(MagicMock(), config={})
        assert ctx.llm_config() == {}

    def test_learn_config_none(self):
        ctx = PlatformContext(MagicMock(), config={})
        assert ctx.learn_config() is None

    def test_cleanup(self):
        lg = MagicMock()
        ctx = PlatformContext(lg, config={})
        ctx.cleanup()  # Should not raise


# ---------------------------------------------------------------------------
# helpers._substitute_in_dict
# ---------------------------------------------------------------------------


class TestSubstituteInDict:
    def test_string(self):
        result = _substitute_in_dict("Hello {{NAME}}", {"NAME": "World"})
        assert result == "Hello World"

    def test_dict(self):
        data = {"greeting": "Hello {{NAME}}"}
        result = _substitute_in_dict(data, {"NAME": "World"})
        assert result == {"greeting": "Hello World"}

    def test_list(self):
        data = ["Hello {{NAME}}", "Bye {{NAME}}"]
        result = _substitute_in_dict(data, {"NAME": "World"})
        assert result == ["Hello World", "Bye World"]

    def test_passthrough(self):
        assert _substitute_in_dict(42, {}) == 42
        assert _substitute_in_dict(None, {}) is None
        assert _substitute_in_dict(True, {}) is True

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_XYZ", "from_env")
        result = _substitute_in_dict("{{TEST_VAR_XYZ}}", {})
        assert result == "from_env"

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _substitute_in_dict("{{NONEXISTENT_VAR_ABC}}", {})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    @pytest.mark.asyncio
    async def test_register_and_trigger(self):
        d = Dispatcher()

        async def handler(x: int) -> int:
            return x * 2

        d.on("double", handler)
        result = await d.trigger("double", x=5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_unknown_event_raises(self):
        d = Dispatcher()
        with pytest.raises(ValueError, match="No handler registered"):
            await d.trigger("unknown")

    def test_has_handler(self):
        d = Dispatcher()

        async def handler() -> None:
            pass

        assert d.has_handler("test") is False
        d.on("test", handler)
        assert d.has_handler("test") is True


# ---------------------------------------------------------------------------
# LLMCaller
# ---------------------------------------------------------------------------


class TestLLMCaller:
    def test_init(self):
        lg = MagicMock()
        router = MagicMock()
        caller = LLMCaller(lg, router)
        assert caller.router is router
        assert caller.dry_run is False

    def test_dry_run_mode(self):
        caller = LLMCaller(MagicMock(), MagicMock(), dry_run=True)
        assert caller.dry_run is True

        result = caller.chat(messages=[{"role": "user", "content": "Hi"}])
        assert result.dry_run is True
        assert result.content == "[dry-run: no response]"
        assert result.raw_response is None
        # Router should NOT be called
        caller.router.chat.assert_not_called()

    def test_normal_call(self):
        lg = MagicMock()
        router = MagicMock()
        response = MagicMock()
        response.content = "Hello!"
        response.model = "test-model"
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        response.usage.total_tokens = 30
        response.tool_calls = None
        response.adapter = None
        router.chat.return_value = response

        caller = LLMCaller(lg, router)
        result = caller.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            temperature=0.5,
        )

        assert result.content == "Hello!"
        assert result.model == "test-model"
        assert result.dry_run is False
        assert result.usage["total_tokens"] == 30
        router.chat.assert_called_once()

    def test_backend_and_extra_body_kwargs(self):
        router = MagicMock()
        response = MagicMock()
        response.content = "ok"
        response.model = "m"
        response.usage = None
        response.tool_calls = None
        response.adapter = None
        router.chat.return_value = response

        caller = LLMCaller(MagicMock(), router)
        caller.chat(
            messages=[{"role": "user", "content": "Hi"}],
            backend="cloud",
            extra_body={"foo": "bar"},
        )

        call_kwargs = router.chat.call_args
        assert call_kwargs.kwargs["backend"] == "cloud"
        assert call_kwargs.kwargs["extra_body"] == {"foo": "bar"}

    def test_to_result_with_tool_calls(self):
        router = MagicMock()
        tc = MagicMock()
        tc.function.name = "search"
        tc.function.arguments = '{"q": "test"}'
        response = MagicMock()
        response.content = ""
        response.model = "m"
        response.usage = None
        response.tool_calls = [tc]
        response.adapter = None
        router.chat.return_value = response

        caller = LLMCaller(MagicMock(), router)
        result = caller.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.tool_calls is not None
        assert result.tool_calls[0]["name"] == "search"

    def test_to_result_none_content(self):
        router = MagicMock()
        response = MagicMock()
        response.content = None
        response.model = None
        response.usage = None
        response.tool_calls = None
        response.adapter = None
        router.chat.return_value = response

        caller = LLMCaller(MagicMock(), router)
        result = caller.chat(messages=[{"role": "user", "content": "Hi"}])
        assert result.content == ""
        assert result.model == "unknown"

    def test_close(self):
        router = MagicMock()
        caller = LLMCaller(MagicMock(), router)
        caller.close()
        router.close.assert_called_once()


# ---------------------------------------------------------------------------
# AgentFactory (core)
# ---------------------------------------------------------------------------


class TestCoreAgentFactory:
    def test_agent_class_not_set_raises(self):
        platform = PlatformContext(MagicMock(), config={"llm": {}})

        class BadFactory(Factory):
            agent_class = None

        f = BadFactory(platform=platform)
        with pytest.raises(Exception, match="agent_class"):
            f.create(DotDict({"identity": {"name": "test"}}))

    def test_variable_substitution(self):
        config = {"directive": "Hello {{NAME}}", "method": "Step {{NUM}}"}
        result = _substitute_in_dict(config, {"NAME": "World", "NUM": "1"})
        assert result["directive"] == "Hello World"
        assert result["method"] == "Step 1"
