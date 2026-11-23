# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The execution of a flow: the runner, the run, the walk and what it carries.

A flow is structure; this package executes it. It imports the structure
(:mod:`llm_gent.flow`); the structure never imports it. :class:`Runner`
(exported as :class:`llm_gent.flow.FlowRunner`) starts runs, and
:mod:`.start` says how each one starts. :mod:`.scope` defines the inputs in
effect at one place in a run and how they derive from the enclosing place
(the fold, see ``docs/design/agents-at-scale.md``, "Scopes: the fold"); the
walk (``_chain``, ``_iterate``, ``_map``, ``_executor``), starting and
resuming (``_flow_run``, ``_resume``) and history writing
(``_checkpoint_ctx``) live beside it.
"""

from .runner import Runner
from .scope import Position, Scope
from .start import Fresh, Restart, Resume, Start


__all__ = ["Fresh", "Position", "Restart", "Resume", "Runner", "Scope", "Start"]
