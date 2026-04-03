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

    Adds conversation history with automatic compaction to any agent.
    Compaction is handled by kelt's Conversation internally when token
    thresholds are exceeded.

    Example:
        agent = Agent(lg, config)
        agent.add_trait(SAIATrait(agent, backend=backend))
        agent.add_trait(ConversationTrait(agent))
        agent.start()

        # First interaction
        result = agent.ask("What is 2+2?")
        # Second interaction - has context from first
        result = agent.ask("What about multiplying that by 3?")
    """

    def __init__(self, agent: Agent, config: ConversationTraitConfig | None = None) -> None:
        """Initialize conversation trait.

        Args:
            agent: The agent this trait belongs to.
            config: Conversation configuration.
        """
        super().__init__(agent)
        self.config = config or ConversationTraitConfig()

        conv_config = Config(
            max_tokens=self.config.max_tokens,
            compact_threshold=self.config.compact_threshold,
            preserve_system=self.config.preserve_system,
            min_recent_messages=self.config.min_recent_messages,
        )
        compactor = self._create_compactor()
        self._conversation = Conversation(config=conv_config, compactor=compactor)

    def _create_compactor(self) -> Compactor:
        """Create compactor from config."""
        if self.config.compactor == "sliding_window":
            return SlidingWindowCompactor()
        elif self.config.compactor == "summarizing":
            raise NotImplementedError("SummarizingCompactor not yet supported in ConversationTrait")
        else:
            raise ValueError(f"Unknown compactor: {self.config.compactor}")

    def on_start(self) -> None:
        """Initialize conversation with system prompt from agent."""
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
        """Add a conversation turn (compaction is automatic).

        Args:
            user_content: The user's message (task/question).
            assistant_content: The assistant's response.
        """
        self._conversation.add(user_content)
        self._conversation.add(assistant_content, Role.ASSISTANT)

    def reset(self) -> None:
        """Clear conversation history and re-add system prompt if configured."""
        self._conversation.clear()
        if self.config.preserve_system:
            from .saia import SAIATrait

            saia_trait = self.agent.get_trait(SAIATrait)
            if saia_trait is not None and saia_trait.config.system_prompt:
                self._conversation.add(saia_trait.config.system_prompt, Role.SYSTEM)
