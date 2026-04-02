"""Conversation trait for maintaining context across agent interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_kelt.conversation import (
    Compactor,
    Config,
    Conversation,
    Message,
    Role,
    SlidingWindowCompactor,
)

from ..base import BaseTrait


if TYPE_CHECKING:
    from llm_gent.core.agent import Agent


@dataclass
class ConversationTraitConfig:
    """Configuration for conversation trait.

    Attributes:
        max_tokens: Maximum tokens before compaction required.
        compact_threshold: Trigger compaction at this fraction (0.0-1.0).
        preserve_system: Always preserve system message during compaction.
        min_recent_messages: Minimum recent messages to keep.
        compactor: Compaction strategy ("sliding_window" or "summarizing").
    """

    max_tokens: int = 32000
    compact_threshold: float = 0.8
    preserve_system: bool = True
    min_recent_messages: int = 4
    compactor: str = "sliding_window"


class ConversationTrait(BaseTrait):
    """Conversation management trait for agents.

    Adds conversation history and automatic compaction to any agent. When attached,
    the agent maintains context across run_once() or ask() calls, with automatic
    compaction when approaching token limits.

    Example:
        from llm_gent.agents.default import Agent
        from llm_gent.core.traits.builtin.conv import (
            ConversationTrait,
            ConversationTraitConfig,
        )

        agent = Agent(lg, identity, "You are a helpful assistant")
        agent.add_trait(SAIATrait(...))
        agent.add_trait(
            ConversationTrait(
                agent,
                ConversationTraitConfig(max_tokens=16000, compactor="sliding_window"),
            )
        )
        agent.start()

        # First interaction
        result = agent.ask("What is 2+2?")
        # Second interaction - has context from first
        result = agent.ask("What about multiplying that by 3?")

    Lifecycle:
        - on_start(): Initializes conversation with system prompt from agent
        - on_stop(): No-op (conversation state preserved in memory)
    """

    def __init__(self, agent: Agent, config: ConversationTraitConfig | None = None) -> None:
        """Initialize conversation trait.

        Args:
            agent: The agent this trait belongs to.
            config: Conversation configuration.
        """
        super().__init__(agent)
        self.config = config or ConversationTraitConfig()

        # Create conversation
        conv_config = Config(
            max_tokens=self.config.max_tokens,
            compact_threshold=self.config.compact_threshold,
            preserve_system=self.config.preserve_system,
            min_recent_messages=self.config.min_recent_messages,
        )
        self._conversation = Conversation(config=conv_config)

        # Create compactor
        self._compactor = self._create_compactor()

    def _create_compactor(self) -> Compactor:
        """Create compactor from config."""
        if self.config.compactor == "sliding_window":
            return SlidingWindowCompactor()
        elif self.config.compactor == "summarizing":
            # TODO: SummarizingCompactor requires LLM backend - add support later
            raise NotImplementedError("SummarizingCompactor not yet supported in ConversationTrait")
        else:
            raise ValueError(f"Unknown compactor: {self.config.compactor}")

    def on_start(self) -> None:
        """Initialize conversation with system prompt from agent."""
        # Get system prompt from SAIATrait if available
        from .saia import SAIATrait

        saia_trait = self.agent.get_trait(SAIATrait)
        if saia_trait is not None and saia_trait.config.system_prompt:
            self._conversation.add(saia_trait.config.system_prompt, Role.SYSTEM)

    def on_stop(self) -> None:
        """Stop trait (conversation state preserved in memory)."""
        pass

    @property
    def conversation(self) -> Conversation:
        """Access the conversation object."""
        return self._conversation

    def get_context(self) -> list[Message]:
        """Get conversation history for LLM context.

        Returns:
            List of messages to include in LLM prompt.
        """
        return list(self._conversation.messages)

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        """Add a conversation turn and compact if needed.

        Args:
            user_content: The user's message (task/question).
            assistant_content: The assistant's response.
        """
        self._conversation.add(user_content)
        self._conversation.add(assistant_content, Role.ASSISTANT)

        # Compact if needed
        if self._conversation.needs_compaction():
            self.agent.lg.debug(
                "compacting conversation",
                extra={
                    "agent": self.agent.name,
                    "tokens_before": self._conversation.token_count,
                },
            )
            self._compactor.compact(self._conversation)
            self.agent.lg.debug(
                "compaction complete",
                extra={
                    "agent": self.agent.name,
                    "tokens_after": self._conversation.token_count,
                    "messages": self._conversation.message_count,
                },
            )

    def reset(self) -> None:
        """Clear conversation history (keeps system message if preserve_system=True)."""
        self._conversation.clear()
        # Re-add system prompt if it was there
        if self.config.preserve_system:
            from .saia import SAIATrait

            saia_trait = self.agent.get_trait(SAIATrait)
            if saia_trait is not None and saia_trait.config.system_prompt:
                self._conversation.add(saia_trait.config.system_prompt, Role.SYSTEM)
