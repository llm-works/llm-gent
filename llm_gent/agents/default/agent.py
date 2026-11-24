# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Default agent implementation.

A concrete Agent that uses SAIA for execution. Provides run_once() and the
other methods expected by the runtime runner; :class:`ChatAgent` also answers
questions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from appinfra import DotDict
from appinfra.log import Logger
from appinfra.time import time
from llm_kelt.conversation import Role, SessionStorage
from llm_saia.core.backend import ChatResponse

from ...core.agent import AskStep, Chat, OnAskStep
from ...core.agent import RunnableAgent as BaseAgent
from ...core.agent.session import Session, SessionConfig, Sessions
from ...core.dispatcher import Dispatcher
from ...core.errors import ConfigError
from ...core.runnable import ExecutionResult
from ...core.traits.builtin.memory import MemoryTrait
from ...core.traits.builtin.saia import SAIATrait


@dataclass
class _ConclusionSummary:
    """Summary of agent execution findings."""

    summary: str


class Agent(BaseAgent):
    """Default agent - uses SAIA for execution.

    Provides the interface expected by the runtime runner:
    - run_once(): Execute default task
    - record_feedback(): Feedback handling (no-op for non-learning)
    - cycle_count: Execution counter
    - get_recent_results(): Recent execution history

    Example:
        # Plain dicts are accepted and converted to DotDict internally
        agent = Agent(
            lg,
            config={
                "identity": {"name": "explorer"},
                "default_prompt": "Analyze the codebase",
            },
        )
        agent.add_trait(SAIATrait(agent, backend=backend))
        agent.add_trait(ToolsTrait(agent))
        agent.start()

        result = agent.run_once()
    """

    def __init__(self, lg: Logger, config: DotDict) -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
            config: Agent configuration with keys:
                - identity.name: Agent name (required)
                - default_prompt: Default prompt for run_once() execution (default: "")
        """
        super().__init__(lg, config)
        self._default_prompt = config.get("default_prompt", "")
        self._recent_results: list[ExecutionResult] = []
        self._max_recent = 100
        # Event dispatcher for declarative behavior
        self._dispatcher = Dispatcher()

    def start(self) -> None:
        """Start the agent and all attached traits."""
        if self._started:
            return
        self._start_traits()
        self._register_event_handlers()
        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop the agent and all attached traits."""
        if not self._started:
            return
        self._stop_traits()
        self._started = False
        self._lg.info("agent stopped", extra={"agent": self.name})

    def _register_event_handlers(self) -> None:
        """Register event handlers.

        For default agents, handlers are registered by the factory.
        This method exists as a hook for subclasses to customize behavior.
        """

    async def handle_task(
        self,
        task: str,
        recall_strategy: str = "chronological",
        recall_limit: int = 5,
        session: Session | None = None,
        on_iteration: OnIteration | None = None,
    ) -> ExecutionResult:
        """Handle task execution with configurable memory recall.

        This is the core event handler used by schedule/question events.

        Args:
            task: The task or question to execute.
            recall_strategy: "chronological" or "semantic" recall.
            recall_limit: Maximum number of past solutions to recall.
            session: The session whose conversation is the context. ``None``:
                no conversation; the context is recalled from memory.
            on_iteration: Called by SAIA after each model call.

        Returns:
            ExecutionResult with outcome.
        """
        saia_trait = self.get_trait(SAIATrait)
        memory_trait = self.get_trait(MemoryTrait)

        if saia_trait is None:
            return ExecutionResult(success=False, content="SAIATrait not attached")

        # Get context (the session's conversation if it has one, otherwise recall)
        context = self._get_context(memory_trait, task, recall_strategy, recall_limit, session)

        # Execute task
        prompt = saia_trait.saia.compose(context, task)
        saia_result = await saia_trait.saia.complete(prompt, on_iteration=on_iteration)
        result = saia_trait.to_execution_result(saia_result)

        # Persist successful outcomes
        if result.success and memory_trait is not None:
            await self._persist_outcome(memory_trait, saia_trait, task, result)

        return result

    def _recall_context(
        self,
        memory_trait: MemoryTrait | None,
        task: str,
        recall_strategy: str,
        recall_limit: int,
    ) -> str:
        """Recall past solutions and format as context string."""
        from ...core.memory import (
            format_solutions_context,
            recall_chronological,
            recall_semantic,
        )

        if memory_trait is None:
            return ""

        if recall_strategy == "semantic":
            past = recall_semantic(
                memory_trait, query=task, limit=recall_limit, agent_name=self.name
            )
        else:  # chronological
            past = recall_chronological(memory_trait, self.name, limit=recall_limit)

        return format_solutions_context(past)

    def _get_context(
        self,
        memory_trait: MemoryTrait | None,
        task: str,
        recall_strategy: str,
        recall_limit: int,
        session: Session | None,
    ) -> str:
        """Get context for task execution.

        The session's conversation when it has one; otherwise solutions
        recalled from MemoryTrait.
        """
        if session is not None:
            conversation = _format_conversation(session)
            if conversation:
                return conversation
        return self._recall_context(memory_trait, task, recall_strategy, recall_limit)

    async def _persist_outcome(
        self,
        memory_trait: MemoryTrait,
        saia_trait: SAIATrait,
        task: str,
        result: ExecutionResult,
    ) -> None:
        """Persist execution outcome to learning database."""
        if not result.content.strip():
            return

        try:
            summary = await self._summarize_outcome(saia_trait, result.content)
            if summary:
                self._record_solution(memory_trait, task, result, summary)
        except Exception as e:
            self._lg.warning(
                "failed to persist solution", extra={"agent": self.name, "exception": e}
            )

    async def _summarize_outcome(self, saia_trait: SAIATrait, content: str) -> str:
        """Summarize execution outcome using SAIA extract."""
        summary_result = await saia_trait.saia.extract(
            content,
            _ConclusionSummary,
            instructions=(
                "Summarize the key findings and conclusions from this agent "
                "execution. Be concise — focus on what was discovered, not "
                "the steps taken. Output only factual findings."
            ),
        )
        return summary_result.value.summary.strip()

    def _record_solution(
        self,
        memory_trait: MemoryTrait,
        task: str,
        result: ExecutionResult,
        summary: str,
    ) -> None:
        """Record solution to learning database."""
        memory_trait.record_solution(self.name, task, result, summary)
        self._lg.debug(
            "solution persisted",
            extra={"agent": self.name, "tokens": result.tokens_used},
        )

    def run_once(self) -> ExecutionResult:
        """Execute one cycle using the default prompt.

        Returns:
            ExecutionResult with success status and content.
        """
        if not self._default_prompt:
            return ExecutionResult(success=False, content="No default prompt configured")

        start_t = time.start()
        result = asyncio.run(self._run_once_pipeline())
        result.latency_ms = int(time.since(start_t))
        self._cycle_count += 1
        self._store_result(result)

        self._lg.info(
            "execution completed",
            extra={
                "after": result.latency_ms,
                "agent": self.name,
                "success": result.success,
                "iters": result.iterations,
                "tokens": result.tokens_used,
                "output": result.content,
            },
        )

        return result

    async def _run_once_pipeline(self) -> ExecutionResult:
        """Run execution pipeline.

        Executes the agent's task using the event dispatcher system.
        """
        return await self._run_with_dispatcher()

    async def _run_with_dispatcher(self) -> ExecutionResult:
        """Run execution with chronological recall (for repeated tasks)."""
        return await self.handle_task(
            task=self._default_prompt,
            recall_strategy="chronological",
            recall_limit=5,
        )

    def record_feedback(self, message: str) -> None:
        """Record feedback (no-op for non-learning agent)."""
        self._lg.debug("feedback received (not stored)", extra={"agent": self.name})

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of recent ExecutionResult objects.
        """
        return self._recent_results[-limit:]

    def _store_result(self, result: ExecutionResult) -> None:
        """Store result in recent history."""
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent :]


class ChatAgent(Agent, Chat):
    """Default agent that also answers questions, each in a session.

    Every ask runs in a session: one conversation, under an id the agent
    issues. Asking without a session opens one; the result's ``session``
    continues it. Asks in one session run one at a time; asks in different
    sessions run side by side. Each model call answering the question is a
    ``model_call`` step: its iteration, the tools it called, the tokens it spent.

    Example:
        agent = ChatAgent(lg, config={"identity": {"name": "explorer"}}, session_storage=None)
        first = agent.ask("What files are in src/?")
        agent.ask("Which of them are tests?", session=first.session)
    """

    def __init__(
        self, lg: Logger, config: DotDict, *, session_storage: SessionStorage | None
    ) -> None:
        """Initialize agent.

        Args:
            lg: Logger instance.
            config: :class:`Agent`'s keys, plus ``chat``: how sessions keep
                their conversations (:class:`~llm_gent.core.agent.SessionConfig`).
            session_storage: Where sessions persist; ``None`` keeps them in memory only.
        """
        super().__init__(lg, config)
        self._sessions = Sessions(lg, self.identity, _session_config(config), session_storage)

    def ask(
        self, question: str, session: str | None = None, *, on_step: OnAskStep | None = None
    ) -> ExecutionResult:
        """Answer ``question`` in a session; see :meth:`Chat.ask`."""
        on_iteration = None if on_step is None else _model_calls(on_step)
        with self._sessions.use(session) as current:
            result = asyncio.run(
                self.handle_task(question, "semantic", 5, current, on_iteration=on_iteration)
            )
            current.conversation.add(question)
            current.conversation.add(result.content, Role.ASSISTANT)
            self._sessions.save(current)
        result.session = current.id
        self._store_result(result)
        return result


OnIteration = Callable[[int, ChatResponse], Awaitable[None]]
"""SAIA's callback after each model call: the iteration and the model's response."""


def _model_calls(on_step: OnAskStep) -> OnIteration:
    """A SAIA callback reporting each model call to ``on_step`` as a ``model_call`` step."""

    async def on_iteration(iteration: int, response: ChatResponse) -> None:
        tokens = response.input_tokens + response.output_tokens
        tools = [call.name for call in response.tool_calls]
        on_step(AskStep("model_call", {"iteration": iteration, "tools": tools, "tokens": tokens}))

    return on_iteration


def _session_config(config: DotDict) -> SessionConfig:
    """The sessions' settings from the config's ``chat`` section."""
    declared = dict(config.get("chat") or {})
    try:
        return SessionConfig(**declared)
    except TypeError as e:
        raise ConfigError(f"invalid chat section: {e}") from e


def _format_conversation(session: Session) -> str:
    """The session's conversation as context text; empty when it has none.

    System messages are left out: the system prompt is SAIA's.
    """
    parts = [
        f"{message.role.capitalize()}: {message.content}"
        for message in session.conversation.messages
        if message.role != "system"
    ]
    return "\n\n".join(parts)
