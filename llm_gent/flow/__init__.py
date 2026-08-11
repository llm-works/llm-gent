"""Agent flow framework — role-based verb dispatch on top of saia.

Public surface:

- :class:`Role` — pure config for a persona (name, backend, model, sampling)
- :class:`SAIAFactory` — protocol that turns a Role into a saia instance
- :func:`verb` — decorator marking an async function as a role-bound verb
- :class:`Context` — runtime environment injected into every verb
- :class:`Flow` — verb registry + role-routed dispatch

Later PRs add: parallel composition (``Panel``) and the four archetype base
classes (``Planner`` / ``Extractor`` / ``Grader`` / ``Synthesizer``).

State (facts, KG, RAG, conversation) is not provided here — consumers
compose those from ``llm_kelt`` (default) or their own implementations, and
mount them via the existing trait system.
"""

from .context import Context
from .factory import SAIAFactory
from .flow import Flow
from .role import Role
from .verb import verb


__all__ = [
    "Context",
    "Flow",
    "Role",
    "SAIAFactory",
    "verb",
]
