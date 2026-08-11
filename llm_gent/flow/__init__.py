"""Agent flow framework — role-based verb dispatch on top of saia.

Public surface (PR 1 — foundation only):

- :class:`Role` — pure config for a persona (name, backend, model, sampling)
- :class:`SAIAFactory` — protocol that turns a Role into a saia instance

Later PRs add: verb dispatch (the flow runner), parallel composition
(``Panel``), and the four archetype base classes (``Planner`` / ``Extractor``
/ ``Grader`` / ``Synthesizer``).

State (facts, KG, RAG, conversation) is not provided here — consumers
compose those from ``llm_kelt`` (default) or their own implementations, and
mount them via the existing trait system.
"""

from .factory import SAIAFactory
from .role import Role


__all__ = [
    "Role",
    "SAIAFactory",
]
