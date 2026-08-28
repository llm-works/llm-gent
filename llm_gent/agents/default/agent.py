# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Default agent implementation.

A concrete Agent that uses SAIA for execution. Provides run_once(), ask(),
and other methods expected by the runtime runner.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from appinfra import DotDict
from appinfra.log import Logger
from appinfra.time import time

from ...core.agent import Agent as BaseAgent
from ...core.dispatcher import Dispatcher
from ...core.runnable import ExecutionResult
from ...core.traits.builtin.learn import LearnTrait
from ...core.traits.builtin.saia import SAIATrait


if TYPE_CHECKING:
    from ...core.traits.builtin.conversation import ConversationTrait


@dataclass
class _ConclusionSummary:
    """Summary of agent execution findings."""

    summary: str


class Agent(BaseAgent):
    """Default agent - uses SAIA for execution.

    Provides the interface expected by the runtime runner:
    - run_once(): Execute default task
    - ask(): Interactive Q&A
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
        response = agent.ask("What files are in src/?")
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
        pass

    async def handle_task(
        self,
        task: str,
        recall_strategy: str = "chronological",
        recall_limit: int = 5,
    ) -> ExecutionResult:
        """Handle task execution with configurable memory recall.

        This is the core event handler used by schedule/question events.

        Args:
            task: The task or question to execute.
            recall_strategy: "chronological" or "semantic" recall.
            recall_limit: Maximum number of past solutions to recall.

        Returns:
            ExecutionResult with outcome.
        """
        saia_trait = self.get_trait(SAIATrait)
        learn_trait = self.get_trait(LearnTrait)

        if saia_trait is None:
            return ExecutionResult(success=False, content="SAIATrait not attached")

        # Get context (conversation if available, otherwise recall from learning)
        context = self._get_context(learn_trait, task, recall_strategy, recall_limit)

        # Execute task
        prompt = saia_trait.saia.compose(context, task)
        saia_result = await saia_trait.saia.complete(prompt)
        result = saia_trait.to_execution_result(saia_result)

        # Add turn to conversation if trait present
        self._record_conversation_turn(task, result)

        # Persist successful outcomes
        if result.success and learn_trait is not None:
            await self._persist_outcome(learn_trait, saia_trait, task, result)

        return result

    def _recall_context(
        self,
        learn_trait: LearnTrait | None,
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

        if learn_trait is None:
            return ""

        if recall_strategy == "semantic":
            past = recall_semantic(
                learn_trait, query=task, limit=recall_limit, agent_name=self.name
            )
        else:  # chronological
            past = recall_chronological(learn_trait, self.name, limit=recall_limit)

        return format_solutions_context(past)

    def _get_context(
        self,
        learn_trait: LearnTrait | None,
        task: str,
        recall_strategy: str,
        recall_limit: int,
    ) -> str:
        """Get context for task execution.

        Uses conversation history if ConversationTrait is present,
        otherwise falls back to solution recall from LearnTrait.
        """
        from ...core.traits.builtin.conversation import ConversationTrait

        conv_trait = self.get_trait(ConversationTrait)

        # Prefer conversation context if available and non-empty
        if conv_trait is not None:
            conv_context = self._format_conversation_context(conv_trait)
            if conv_context:
                return conv_context

        # Fall back to recall if no conversation context
        return self._recall_context(learn_trait, task, recall_strategy, recall_limit)

    def _format_conversation_context(self, conv_trait: ConversationTrait) -> str:
        """Format conversation history as context string."""
        messages = conv_trait.get_context()
        if not messages:
            return ""

        # Format messages as text (skip system message, it's in SAIA config)
        parts = []
        for msg in messages:
            if msg.role == "system":
                continue
            role = msg.role.capitalize()
            content = msg.content
            parts.append(f"{role}: {content}")

        return "\n\n".join(parts)

    def _record_conversation_turn(self, task: str, result: ExecutionResult) -> None:
        """Record conversation turn if ConversationTrait is present."""
        from ...core.traits.builtin.conversation import ConversationTrait

        conv_trait = self.get_trait(ConversationTrait)
        if conv_trait is not None:
            conv_trait.add_turn(task, result.content)

    async def _persist_outcome(
        self,
        learn_trait: LearnTrait,
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
                self._record_solution(learn_trait, task, result, summary)
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
        learn_trait: LearnTrait,
        task: str,
        result: ExecutionResult,
        summary: str,
    ) -> None:
        """Record solution to learning database."""
        learn_trait.record_solution(self.name, task, result, summary)
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

    def ask(self, question: str) -> str:
        """Ask the agent a question.

        Args:
            question: The question to ask.

        Returns:
            Response string from the agent.
        """
        result = asyncio.run(self._ask_async(question))
        self._store_result(result)
        return result.content

    async def _ask_async(self, question: str) -> ExecutionResult:
        """Async implementation of ask() with semantic recall (for varied questions)."""
        return await self.handle_task(
            task=question,
            recall_strategy="semantic",
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
