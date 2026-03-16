# llm-gent Documentation

Agent framework with trait-based architecture and learning capabilities.

## Quick Links

- [README](../README.md) - Overview, installation, quick start
- [CHANGELOG](../CHANGELOG.md) - Release history

## Core Concepts

### Agents

An `Agent` is the central unit - a container for traits with lifecycle management. Each agent has:

- **Identity** - domain/workspace/name tuple for namespacing
- **Config** - agent-specific configuration
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
agent = Agent(lg, config)
agent.add_trait(LLMTrait(agent, llm_config))
agent.start()      # Initialize all traits
result = agent.run_once()  # Execute one cycle
agent.stop()       # Cleanup all traits
```

## Related Projects

- [llm-infer](https://github.com/llm-works/llm-infer) - LLM inference server and client
- [llm-kelt](https://github.com/llm-works/llm-kelt) - Training infrastructure (SFT/DPO)
- [appinfra](https://github.com/llm-works/appinfra) - Application infrastructure utilities
