"""Conversation management for agents.

Re-exports from llm-kelt conversation module, plus the local ConversationRunner
which ties conversation to the agent execution lifecycle.
"""

from llm_kelt.conversation import (
    Compactor,
    Config,
    Conversation,
    Message,
    Role,
    SlidingWindowCompactor,
    SummarizingCompactor,
    estimate_message_tokens,
    estimate_tokens,
)

from .runner import ConversationRunner


ConversationConfig = Config

__all__ = [
    # Core (from llm-kelt)
    "Compactor",
    "Config",
    "ConversationConfig",
    "Conversation",
    "Message",
    "Role",
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    "estimate_tokens",
    "estimate_message_tokens",
    # Local
    "ConversationRunner",
]
