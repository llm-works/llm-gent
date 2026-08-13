"""SAIAFactory — plug point that constructs a role-bound saia instance.

A :class:`Role` is pure config. Turning a role into a runnable saia client
requires wiring together an llm-infer backend, saia's builder, tools, and a
system prompt. Any of those pieces can vary between deployments, so the
framework leaves construction to a user-supplied factory and provides only
the protocol.

A flow holds a single factory and typically caches saia instances per role
so verbs share bindings within one run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from .role import Role


class SAIAFactory(Protocol):
    """Constructs a role-bound saia instance.

    Implementations decide how a :class:`Role` maps to a backend + tools +
    system prompt + call options. The framework's only expectation is that
    :meth:`build` returns an object supporting saia's public surface — the
    verb calling code uses ``.verify(...)`` / ``.complete(...)`` / etc.

    Reference sketch (users typically write one specific to their yaml
    config)::

        class MySAIAFactory:
            def __init__(self, lg, llm_yaml, tools=None):
                self._lg = lg
                self._llm_yaml = llm_yaml
                self._tools = tools or []

            def build(self, role):
                backend = build_backend_from_config(self._llm_yaml, role)
                builder = SAIA.builder().backend(backend).logger(self._lg)
                if self._tools:
                    builder = builder.tools(self._tools, executor)
                if role.style:
                    builder = builder.system(role.style)
                return builder.build()
    """

    def build(self, role: Role) -> Any:
        """Return a saia instance configured for ``role``."""
        ...
