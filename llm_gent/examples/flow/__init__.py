# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Flow-substrate example agents.

Demonstrates the Flow substrate directly: typed state, ``.iterate``,
checkpointed pause/resume, and multi-role composition. See the
sibling ``llm_gent.examples`` modules (``quickstart``, ``external_agent``)
for the older ``AgentFactory`` + ``LLMTrait`` surface.

Modules:

- :mod:`resume` — halt-and-resume demo on a counter loop. Pure-Python
  verbs; exercises checkpoint mechanics without needing an LLM backend.
- :mod:`durable_resume` — halts a real SAIA turn between a model tool
  call and its follow-up completion, exits, and resumes that turn from
  the checkpoint store in a second process. Runs against an
  OpenAI-compatible server or Anthropic; ``--smoke`` uses a scripted
  ``saia.Backend``. ``durable_resume.md`` walks the on-disk store.
- :mod:`verifier` — multi-model consensus loop. Primary LLM answers a
  query, self-reviews, verifier LLM reviews, third LLM judges semantic
  agreement, loops until consensus or ``n=5`` rounds. Uses a scripted
  stub SAIA from :mod:`._infra` so the example runs deterministically
  without contacting a real backend.
- :mod:`structured_agent` — structured-output agent that classifies and
  triages a bug report.
- :mod:`plan_execute` — plan-and-execute agent: planner + extractor +
  ``.branch`` + ``.iterate``.
- :mod:`subflow_research` — multi-topic research agent:
  ``.map(state=, merge=)`` + ``state.root()`` + synthesizer.
- :mod:`batch_grade` — batch grader: ``.map(strict=False)`` + a map over graders
  majority vote + ``.guard`` + ``.rescue``.
"""
