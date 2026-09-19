# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow-substrate example agents.

Demonstrates the Flow substrate directly: typed state, ``.iterate``,
``.with_checkpointer`` pause/resume, and multi-role composition. See the
sibling ``llm_gent.examples`` modules (``quickstart``, ``external_agent``)
for the older ``AgentFactory`` + ``LLMTrait`` surface.

Modules:

- :mod:`resume` — halt-and-resume demo on a counter loop. Pure-Python
  verbs; exercises checkpoint mechanics without needing an LLM backend.
- :mod:`verifier` — multi-model consensus loop. Primary LLM answers a
  query, self-reviews, verifier LLM reviews, third LLM judges semantic
  agreement, loops until consensus or ``n=5`` rounds. Uses a scripted
  stub SAIA from :mod:`._infra` so the example runs deterministically
  without contacting a real backend.
"""
