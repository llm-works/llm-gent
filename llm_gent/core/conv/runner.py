"""Conversation-based task execution runner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from llm_kelt.conversation import Compactor, Conversation, Message, Role

from ..llm.types import Message as LLMMessage
from ..task import Task, TaskResult


if TYPE_CHECKING:
    from llm_saia import SAIA

    from llm_gent.core.traits.builtin.llm import LLMTrait


# Type alias for identity prompt builder
IdentityBuilder = Callable[[], str]
"""Function that builds the identity/system prompt."""


def _to_llm_messages(messages: list[Message]) -> list[LLMMessage]:
    """Convert kelt conversation Messages to LLM backend Messages."""
    from dataclasses import asdict

    result = []
    for m in messages:
        tool_calls = [asdict(tc) for tc in m.tool_calls] if m.tool_calls else None
        role: Any = m.role  # Role enum → Literal coercion
        result.append(
            LLMMessage(
                role=role,
                content=m.content,
                tool_calls=tool_calls,
                tool_call_id=m.tool_call_id,
            )
        )
    return result


# Conversation context limits
_CONTEXT_MESSAGE_LIMIT = 5  # Last N messages for context
_SYSTEM_CONTENT_TRUNCATE = 500  # Max chars for system context
_ASSISTANT_CONTENT_TRUNCATE = 200  # Max chars for assistant context


@dataclass
class ConversationRunner:
    """Runs conversation-based task execution with optional SAIA integration.

    Manages the conversation lifecycle:
    - Prepares conversation (compaction, initialization/continuation)
    - Runs task execution via SAIA (if available) or simple LLM completion
    - Builds results with optional structured output extraction

    Example:
        runner = ConversationRunner(
            lg=lg,
            conversation=conversation,
            compactor=compactor,
            llm_trait=llm_trait,
            saia=saia,  # Optional - enables tool use
            default_task=Task(name="default", description="..."),
            identity_builder=lambda: agent._build_identity_prompt(),
        )
        result = runner.run(task=task, is_first_run=True)
    """

    lg: Logger
    conversation: Conversation
    compactor: Compactor
    llm_trait: LLMTrait | None
    saia: SAIA | None
    default_task: Task
    identity_builder: IdentityBuilder

    _agent_name: str = field(default="agent", repr=False)

    def run(self, task: Task | None, is_first_run: bool) -> TaskResult:
        """Run one conversation cycle.

        Args:
            task: Task to execute (uses default prompt if None).
            is_first_run: Whether this is the first run (initializes vs continues).

        Returns:
            TaskResult from this execution cycle.
        """
        self._prepare_conversation(is_first_run, task)
        return self._run_conversation_loop(task)

    # -------------------------------------------------------------------------
    # Conversation Preparation
    # -------------------------------------------------------------------------

    def _prepare_conversation(self, is_first_run: bool, task: Task | None) -> None:
        """Prepare conversation for this run."""
        if self.conversation.needs_compaction():
            self._compact_conversation()

        if is_first_run:
            self._initialize_conversation(task)
        else:
            self._continue_conversation(task)

    def _initialize_conversation(self, task: Task | None) -> None:
        """Initialize conversation with system prompt and task for first run."""
        # Build system prompt using identity builder
        system_prompt = self.identity_builder()
        self.conversation.add(system_prompt, Role.SYSTEM)

        # Use task description or default task description
        effective_task = task or self.default_task
        self.conversation.add(effective_task.description)

    def _continue_conversation(self, task: Task | None) -> None:
        """Add continuation prompt for subsequent runs."""
        if task:
            self.conversation.add(task.description)
        else:
            continue_prompt = (
                "Continue your exploration and analysis. Build on what you've learned so far. "
                "Focus on areas you haven't fully explored yet, or dive deeper into interesting "
                "findings. Use your tools to gather more information."
            )
            self.conversation.add(continue_prompt)

    def _compact_conversation(self) -> None:
        """Compact conversation when approaching token limit."""
        before_tokens = self.conversation.token_count
        self.compactor.compact(self.conversation)
        after_tokens = self.conversation.token_count

        self.lg.info(
            "conversation compacted",
            extra={
                "agent": self._agent_name,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "reduction": before_tokens - after_tokens,
            },
        )

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def _run_conversation_loop(self, task: Task | None) -> TaskResult:
        """Run task execution with current conversation state."""
        if self.saia is not None:
            return self._run_with_saia(task)
        return self._run_simple_completion(task)

    def _run_simple_completion(self, task: Task | None) -> TaskResult:
        """Run simple completion without tools (requires llm_trait)."""
        if self.llm_trait is None:
            raise RuntimeError("Either SAIA or LLMTrait required for execution")

        result = self.llm_trait.complete(messages=_to_llm_messages(self.conversation.messages))
        self.conversation.add(result.content, Role.ASSISTANT)

        # Handle structured output extraction if requested
        parsed, extra_tokens = None, 0
        if task is not None and task.output_schema is not None:
            extraction = self._try_extract_structured_simple(result, task.output_schema)
            if isinstance(extraction, TaskResult):
                return extraction
            parsed, extra_tokens = extraction

        return TaskResult(
            success=True,
            content=result.content,
            parsed=parsed,
            tool_calls=[],
            iterations=1,
            tokens_used=result.tokens_used + extra_tokens,
        )

    def _run_with_saia(self, task: Task | None) -> TaskResult:
        """Run task execution with SAIA."""
        assert self.saia is not None

        effective_task = task or self.default_task

        try:
            # Build task description from conversation context
            messages = self.conversation.messages
            task_context = self._build_task_from_conversation(messages, effective_task)

            # Run SAIA complete - handle sync/async boundary
            saia_result = self._run_saia_complete(task_context)

            # Update conversation with result
            self.conversation.add(saia_result.output, Role.ASSISTANT)

            # Build result
            return self._build_saia_result(effective_task, saia_result)

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            self.lg.warning(
                "SAIA execution failed",
                extra={"agent": self._agent_name, "exception": e},
            )
            return TaskResult(success=False, content="", error=str(e))

    def _run_saia_complete(self, task_context: str) -> Any:
        """Run SAIA complete, handling sync/async boundary."""
        assert self.saia is not None
        return self._run_async(self.saia.complete(task_context))

    def _build_task_from_conversation(self, messages: list[Message], task: Task) -> str:
        """Build task description incorporating conversation context."""
        # Extract system prompt and recent messages for context
        context_parts = []

        for msg in messages[-_CONTEXT_MESSAGE_LIMIT:]:
            if msg.role == "system":
                context_parts.append(f"Context: {msg.content[:_SYSTEM_CONTENT_TRUNCATE]}")
            elif msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                truncated = msg.content[:_ASSISTANT_CONTENT_TRUNCATE]
                suffix = "..." if len(msg.content) > _ASSISTANT_CONTENT_TRUNCATE else ""
                context_parts.append(f"Previous response: {truncated}{suffix}")

        context = "\n".join(context_parts)
        return f"{context}\n\nCurrent task: {task.description}"

    def _build_saia_result(self, task: Task, saia_result: Any) -> TaskResult:
        """Build TaskResult from SAIA execution result."""
        parsed = None

        # Handle structured output extraction if requested
        if task.output_schema is not None and saia_result.completed:
            parsed = self._try_extract_with_saia(saia_result.output, task.output_schema)

        # Extract completion info from terminal data
        completion = self._extract_completion(saia_result.terminal_data)

        return TaskResult(
            success=saia_result.completed,
            content=saia_result.output,
            parsed=parsed,
            completion=completion,
            tool_calls=[],  # SAIA tracks tool calls differently
            iterations=saia_result.iterations,
            tokens_used=0,  # SAIA tracks tokens differently
        )

    def _try_extract_with_saia(self, content: str, output_schema: type) -> Any:
        """Try to extract structured output using SAIA extract verb.

        Handles sync/async boundary the same way as _run_saia_complete().
        """
        if self.saia is None:
            return None

        try:
            return self._run_async(self.saia.extract(content, output_schema))
        except Exception as e:
            self.lg.warning(
                "SAIA extraction failed",
                extra={"agent": self._agent_name, "exception": e},
            )
            return None

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine, handling sync/async boundary.

        Detects whether we're already in an async context and handles
        accordingly to avoid nested asyncio.run() errors.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context - run in thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            # Not in async context - safe to use asyncio.run
            return asyncio.run(coro)

    def _extract_completion(self, terminal_data: dict[str, Any] | None) -> Any:
        """Extract TaskCompletion from terminal_data if valid."""
        from llm_gent.core.task import TaskCompletion, TaskStatus

        if not terminal_data:
            return None
        status = terminal_data.get("status")
        conclusion = terminal_data.get("conclusion")
        if not (status and conclusion):
            return None
        try:
            return TaskCompletion(status=TaskStatus(status), conclusion=conclusion)
        except ValueError:
            self.lg.warning(
                "invalid terminal status",
                extra={"status": status, "agent": self._agent_name},
            )
            return None

    # -------------------------------------------------------------------------
    # Simple completion helpers (when no SAIA)
    # -------------------------------------------------------------------------

    def _try_extract_structured_simple(
        self, completion_result: Any, output_schema: type
    ) -> tuple[Any, int] | TaskResult:
        """Try to extract structured output from simple completion."""
        if self.llm_trait is None:
            return TaskResult(
                success=False,
                content=completion_result.content,
                error="LLMTrait required for structured extraction",
            )

        from llm_gent.core.llm.backend import StructuredOutputError

        messages = self._build_extraction_messages(completion_result.content)

        try:
            result = self.llm_trait.complete(
                messages=messages,
                output_schema=output_schema,
            )
            return result.parsed, result.tokens_used
        except StructuredOutputError as e:
            return TaskResult(
                success=False,
                content=completion_result.content,
                tool_calls=[],
                iterations=1,
                tokens_used=completion_result.tokens_used,
                error=f"Structured output error: {e}",
            )

    def _build_extraction_messages(self, content: str) -> list[LLMMessage]:
        """Build messages for structured output extraction."""
        system_prompt = (
            "You are a data extraction assistant. "
            "Your task is to extract information and return it in the exact JSON format requested."
        )
        user_prompt = (
            "Based on the following information, extract the relevant data.\n\n"
            f"Information:\n{content}"
        )
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
