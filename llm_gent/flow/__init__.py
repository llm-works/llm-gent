"""Agent flow framework — role-based verb dispatch on top of saia.

Public surface:

- :class:`Role` — pure config for a persona (name, backend, model, sampling)
- :class:`SAIAFactory` — protocol that turns a Role into a saia instance
- :func:`verb` — decorator marking an async function as a role-bound verb
- :class:`Context` — runtime environment injected into every verb
- :class:`State` — scope-aware wrapper around the user-owned payload on
  ``ctx.state`` (``.data`` reaches the payload; ``.root()`` walks to the
  outermost scope)
- :class:`Flow` — verb registry + role-routed dispatch + fluent composition
- :class:`Failure` — sentinel returned for a failed item in ``Flow.map(strict=False)``
- :class:`Panel` — fan-out N verbs in parallel + aggregate their results
- Archetype decorators: :func:`planner`, :func:`extractor`, :func:`grader`,
  :func:`synthesizer` — semantic tags for the standard agent shape

Aggregation helpers exposed via :mod:`llm_gent.flow.panel`: ``majority``,
``unanimous``, ``mean``, ``weighted``.

State (facts, KG, RAG, conversation) is not provided here — consumers
compose those from ``llm_kelt`` (default) or their own implementations, and
mount them via the existing trait system.
"""

from .archetypes import extractor, grader, planner, synthesizer
from .context import Context
from .factory import SAIAFactory
from .flow import Flow
from .nodes import Failure
from .panel import Panel
from .role import Role
from .state import State
from .verb import verb


__all__ = [
    "Context",
    "Failure",
    "Flow",
    "Panel",
    "Role",
    "SAIAFactory",
    "State",
    "extractor",
    "grader",
    "planner",
    "synthesizer",
    "verb",
]
