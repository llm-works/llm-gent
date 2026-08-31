# llm-gent Documentation

Agent framework with trait-based architecture and learning capabilities.

## Quick Links

- [README](../README.md) - Overview, installation, quick start
- [CHANGELOG](../CHANGELOG.md) - Release history

## Core Concepts

### Agents

An `Agent` is the central unit - a container for traits with lifecycle management. Each agent has:

- **Identity** - `name` plus optional `context_key` for namespacing
- **Config** - agent-config dict passed to `AgentFactory.from_config`
- **Traits** - pluggable capabilities

### Traits

Traits provide specific capabilities to agents:

| Trait | Purpose |
|-------|---------|
| `LLMTrait` | LLM completions with multi-backend routing |
| `DirectiveTrait` | System prompts and agent instructions |
| `StorageTrait` | PostgreSQL persistence with schema migrations |
| `RatingTrait` | Automated LLM-based content evaluation |
| `LearnTrait` | Training data collection (SFT/DPO) |
| `ToolsTrait` | Tool/function calling support |

### Lifecycle

```python
from llm_gent import AgentFactory, LLMTrait

agent = AgentFactory(lg).from_config(
    {
        "identity": {"name": "my-agent"},
        "llm": llm_config,
        "traits": {"required": ["llm"]},
    }
)
agent.start()  # Initialize all traits
llm = agent.require_trait(LLMTrait)
result = llm.complete([{"role": "user", "content": "Hello!"}])
agent.stop()  # Cleanup all traits
```

For cycle-driven agents (`agent.run_once()`), use `RunnableAgent` — set
`agent_class = RunnableAgent` on an `AgentFactory` subclass, or subclass
`RunnableAgent` directly.

## Related Projects

- [llm-infer](https://github.com/llm-works/llm-infer) - LLM inference server and client
- [llm-kelt](https://github.com/llm-works/llm-kelt) - Training infrastructure (SFT/DPO)
- [appinfra](https://github.com/llm-works/appinfra) - Application infrastructure utilities

---

Maintained by [LLM Works LLC](https://llm-works.ai) and contributors.
