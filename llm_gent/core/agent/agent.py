# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Slim Agent container.

Agent = container + lifecycle + traits + identity. It is not itself a
:class:`Runnable`; the runtime-executable surface (``run_once`` / ``ask`` /
feedback) lives on :class:`RunnableAgent`. Consumers that only need the
container (traits, lifecycle, identity) can construct a plain Agent and
never see the runner abstracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from appinfra import DotDict
from appinfra.log import Logger

from ..errors import TraitNotFoundError
from ..traits.base import BaseTrait
from ..traits.registry import Registry as TraitRegistry


if TYPE_CHECKING:
    from .identity import Identity

TraitT = TypeVar("TraitT", bound=BaseTrait)


class Agent:
    """Trait container with lifecycle + identity.

    Directly instantiable: an ``Agent(lg, config)`` wires up the trait
    registry, resolves identity, and answers ``start`` / ``stop``. Consumers
    that need execution semantics (``run_once`` / ``ask``) subclass
    :class:`RunnableAgent` instead.

    Trait attachment is via :meth:`add_trait`. Lifecycle is via :meth:`start`
    / :meth:`stop`, or the ``async with agent`` context manager. Reaching
    into a specific trait's surface goes through :meth:`get_trait` /
    :meth:`require_trait`.

    Example:
        agent = Agent(lg, config=DotDict(identity={"name": "explorer"}))
        agent.add_trait(SAIATrait(agent, backend=backend))
        async with agent:
            saia = agent.require_trait(SAIATrait).saia
            ...
    """

    def __init__(self, lg: Logger, config: DotDict | dict[str, Any] | None = None) -> None:
        """Initialize the container.

        Args:
            lg: Logger instance.
            config: Agent configuration (converted to DotDict if dict, empty
                if None). Must contain ``identity.name`` if any config is
                provided.

        Raises:
            ConfigError: If ``identity.name`` is missing from config.
        """
        self._lg = lg
        if isinstance(config, DotDict):
            self._config = config
        elif config is not None:
            self._config = DotDict(**config)
        else:
            self._config = DotDict()

        self._identity: Identity = self._resolve_identity()
        self._traits = TraitRegistry(lg)
        self._started = False
        self._cycle_count = 0

    @property
    def name(self) -> str:
        """Agent identifier from identity."""
        return self._identity.name

    @property
    def identity(self) -> Identity:
        """Agent identity."""
        return self._identity

    @property
    def cycle_count(self) -> int:
        """Number of execution cycles completed.

        Base :class:`Agent` never increments this; runnable subclasses
        maintain ``_cycle_count`` themselves.
        """
        return self._cycle_count

    @property
    def lg(self) -> Logger:
        """Logger instance for this agent.

        Traits should read this via ``self.agent.lg`` rather than storing
        their own.
        """
        return self._lg

    @property
    def config(self) -> DotDict:
        """Agent configuration (empty DotDict if none was provided)."""
        return self._config

    @property
    def traits(self) -> TraitRegistry:
        """Trait registry for this agent.

        Introspection surface (``.all()`` / ``.count()`` / ``.types()``).
        For attachment, prefer :meth:`add_trait` so lifecycle stays in sync
        when the agent is already started.
        """
        return self._traits

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start attached traits. Idempotent — a second call is a no-op."""
        if self._started:
            return
        self._lg.trace(
            "starting agent...",
            extra={"agent": self.name, "trait_count": self._traits.count()},
        )
        self._start_traits()
        self._lg.debug("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop attached traits. Idempotent — a second call is a no-op."""
        if not self._started:
            return
        self._lg.trace(
            "stopping agent...",
            extra={"agent": self.name, "trait_count": self._traits.count()},
        )
        self._stop_traits()
        self._lg.debug("agent stopped", extra={"agent": self.name})

    async def __aenter__(self) -> Agent:
        """Async context-manager entry — calls :meth:`start`."""
        self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Async context-manager exit — calls :meth:`stop`."""
        self.stop()

    # =========================================================================
    # Trait Management
    # =========================================================================

    def add_trait(self, trait: BaseTrait) -> None:
        """Add a trait to this agent.

        Traits must be constructed with this agent as a parameter. If the
        agent is already started, the trait's ``on_start()`` is called
        automatically.

        Args:
            trait: The trait instance to add.

        Raises:
            DuplicateTraitError: If a trait of this type is already added.
        """
        self._traits.register(trait)
        self._lg.debug(
            "trait added",
            extra={
                "agent": self.name,
                "trait": type(trait).__name__,
                "started": self._started,
            },
        )

        if self._started:
            try:
                trait.on_start()
            except Exception as e:
                self._lg.warning(
                    "trait add rolled back",
                    extra={
                        "agent": self.name,
                        "trait": type(trait).__name__,
                        "exception": e,
                    },
                )
                self._traits.unregister(type(trait))
                raise

    def get_trait(self, trait_type: type[TraitT]) -> TraitT | None:
        """Get an attached trait by its type (or ``None``)."""
        return self._traits.get(trait_type)

    def has_trait(self, trait_type: type[BaseTrait]) -> bool:
        """Return whether a trait of this type is attached."""
        return self._traits.has(trait_type)

    def require_trait(self, trait_type: type[TraitT]) -> TraitT:
        """Get a required trait, raising if not attached.

        Raises:
            TraitNotFoundError: If the trait is not attached.
        """
        try:
            return self._traits.require(trait_type)
        except TraitNotFoundError as e:
            raise TraitNotFoundError(
                f"{trait_type.__name__} required but not attached - "
                f"add it with agent.add_trait({trait_type.__name__}(...))"
            ) from e

    def replace_trait(self, trait: BaseTrait) -> None:
        """Register ``trait``, replacing any existing entry of the same type.

        Registry membership implies agent-managed lifecycle: when the agent
        is started, the new trait's ``on_start`` is called and the ejected
        trait's ``on_stop`` is called automatically. If the new trait's
        ``on_start`` raises, the ejected trait is restored to the registry
        (and not stopped) before the exception propagates.

        Ordering is start-new-before-stop-old so a failing new-``on_start``
        can roll back cleanly without having already closed the old trait's
        resources.

        The caller's live references to the ejected trait become stale after
        this call — its ``on_stop`` has fired, and for ownership-carrying
        traits like :class:`LLMTrait` with ``owns_router=True`` its
        resources are closed. Re-fetch via :meth:`get_trait` or
        :meth:`require_trait` to obtain a usable reference.

        Typical pairing with ``LLMTrait.with_router``::

            agent.replace_trait(
                agent.require_trait(LLMTrait).with_router(other_router)
            )
            llm = agent.require_trait(LLMTrait)  # reload

        Args:
            trait: The trait instance to register or use as a replacement.
        """
        trait_type = type(trait)
        old = self._traits.get(trait_type)
        self._lg.trace(
            "replacing trait...",
            extra={
                "agent": self.name,
                "trait": trait_type.__name__,
                "had_previous": old is not None,
                "started": self._started,
            },
        )
        self._traits.replace(trait)

        if not self._started:
            self._log_trait_replaced(trait_type, ejected=False)
            return

        try:
            trait.on_start()
        except Exception as e:
            self._rollback_trait_replace(trait, old, trait_type, e)
            raise

        if old is not None:
            self._safe_stop_trait(old, "ejected trait on_stop failed")
        self._log_trait_replaced(trait_type, ejected=old is not None)

    def _rollback_trait_replace(
        self,
        trait: BaseTrait,
        old: BaseTrait | None,
        trait_type: type[BaseTrait],
        exception: Exception,
    ) -> None:
        """Roll back a failed replace_trait: stop the new trait, restore the old."""
        self._lg.warning(
            "trait replace rolled back",
            extra={
                "agent": self.name,
                "trait": trait_type.__name__,
                "exception": exception,
            },
        )
        self._safe_stop_trait(trait, "failed trait cleanup during rollback")
        if old is not None:
            self._traits.replace(old)
        else:
            self._traits.unregister(trait_type)

    def _log_trait_replaced(self, trait_type: type[BaseTrait], *, ejected: bool) -> None:
        """Debug-log the outcome of a successful replace_trait call."""
        self._lg.debug(
            "trait replaced",
            extra={"agent": self.name, "trait": trait_type.__name__, "ejected": ejected},
        )

    # =========================================================================
    # Trait Lifecycle Helpers
    # =========================================================================

    def _safe_stop_trait(self, trait: BaseTrait, context: str) -> None:
        """Attempt on_stop, logging but swallowing any exception."""
        try:
            trait.on_stop()
        except Exception as e:
            self._lg.warning(context, extra={"exception": e, "trait": type(trait).__name__})

    def _start_traits(self) -> None:
        """Start all attached traits; flip ``_started`` only on success."""
        for trait in self._traits.all():
            trait.on_start()
            self._lg.debug(
                "trait started",
                extra={"agent": self.name, "trait": type(trait).__name__},
            )
        self._started = True

    def _stop_traits(self) -> None:
        """Stop all attached traits; flip ``_started`` regardless of errors."""
        for trait in self._traits.all():
            try:
                trait.on_stop()
                self._lg.debug(
                    "trait stopped",
                    extra={"agent": self.name, "trait": type(trait).__name__},
                )
            except Exception as e:
                self._lg.warning(
                    "error stopping trait",
                    extra={"trait": type(trait).__name__, "exception": e},
                )
        self._started = False

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    def _resolve_identity(self) -> Identity:
        """Resolve :class:`Identity` from ``self.config``.

        Raises:
            ConfigError: If ``identity.name`` is missing.
        """
        from ..errors import ConfigError
        from .identity import Identity

        identity_config = self.config.get("identity", {})
        if not identity_config.get("name"):
            kelt_config = self.config.get("kelt", {})
            identity_config = kelt_config.get("identity", {})

        name = identity_config.get("name")
        if not name:
            raise ConfigError("identity.name is required in config")

        return Identity.from_config(identity_config, defaults=DotDict(name=name))
