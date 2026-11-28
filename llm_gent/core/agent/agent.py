# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Slim Agent container.

Agent = container + lifecycle + traits + identity, on a platform. It is not
itself a :class:`Runnable`; the runtime-executable surface (``run_once`` /
``ask`` / feedback) lives on :class:`RunnableAgent`. Consumers that only
need the container (traits, lifecycle, identity) can construct a plain Agent
and never see the runner abstracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

from appinfra import DotDict
from appinfra.log import Logger

from ..errors import ConfigError, DuplicateTraitError, TraitNotFoundError
from ..traits.base import Trait, TraitBinding
from ..traits.registry import Registry as TraitRegistry


if TYPE_CHECKING:
    # Circular at runtime: the platform package imports the builtin traits,
    # whose HTTP trait imports the runtime, which imports the agent.
    from ..platform import Platform
    from .identity import Identity

TraitT = TypeVar("TraitT", bound=TraitBinding)
BindingT = TypeVar("BindingT", bound=TraitBinding)


class Agent:
    """Holds an agent's identity, config and all of its data, on a platform.

    Directly instantiable: an ``Agent(lg, platform, config)`` resolves its
    identity and answers ``start`` / ``stop``. Consumers that need execution
    semantics (``run_once`` / ``ask``) subclass :class:`RunnableAgent`
    instead.

    The platform's traits (``platform.traits``) hold only shared resources
    and are always running; the platform starts and stops them. An agent
    holds a binding of each one it uses (:meth:`use`, :meth:`Trait.bind`):
    the binding holds all of the agent's data for that trait. It also holds
    the traits it builds itself (:meth:`add_trait`). The agent starts and
    stops what it holds with itself (:meth:`start` / :meth:`stop`, or
    ``async with agent``), never a platform trait.

    One lookup, :meth:`require` (or :meth:`get` / :meth:`has`), finds what
    the agent holds: ``agent.require(MemoryTrait)`` is the agent's
    ``MemoryBinding``; ``agent.require(SAIATrait)`` is its own SAIA trait.

    Example:
        agent = Agent(lg, platform, config=DotDict(identity={"name": "explorer"}))
        agent.add_trait(SAIATrait(agent, backend=backend))
        agent.use(MemoryTrait)
        async with agent:
            saia = agent.require(SAIATrait).saia
            memory = agent.require(MemoryTrait)  # the agent's MemoryBinding
            ...
    """

    def __init__(
        self, lg: Logger, platform: Platform, config: DotDict | dict[str, Any] | None = None
    ) -> None:
        """Initialize the container.

        Args:
            lg: Logger instance.
            platform: The platform the agent runs on; the agent binds the
                platform traits it uses.
            config: Agent configuration (converted to DotDict if dict, empty
                if None). Must contain ``identity.name`` if any config is
                provided.

        Raises:
            ConfigError: If ``identity.name`` is missing from config, or the
                config sets what is the platform's: an ``llm`` key or
                ``kelt.schema``.
        """
        self._lg = lg
        self._platform = platform
        if isinstance(config, DotDict):
            self._config = config
        elif config is not None:
            self._config = DotDict(**config)
        else:
            self._config = DotDict()

        self._reject_platform_settings()
        self._identity: Identity = self._resolve_identity()
        self._traits = TraitRegistry(lg)
        self._bindings: dict[type[Trait[Any]], TraitBinding] = {}
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
    def platform(self) -> Platform:
        """The platform the agent runs on."""
        return self._platform

    @property
    def traits(self) -> TraitRegistry:
        """The traits the agent built itself (its bindings of platform traits are not in it).

        Introspection surface (``.all()`` / ``.count()`` / ``.types()``).
        For attachment, prefer :meth:`add_trait` so lifecycle stays in sync
        when the agent is already started.
        """
        return self._traits

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start what the agent holds. Idempotent — a second call is a no-op."""
        if self._started:
            return
        self._lg.trace(
            "starting agent...",
            extra={"agent": self.name, "trait_count": self._traits.count()},
        )
        self._start_traits()
        self._lg.debug("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop what the agent holds. Idempotent — a second call is a no-op."""
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

    def add_trait(self, trait: TraitBinding) -> None:
        """Add a trait to this agent.

        Traits must be constructed with this agent as a parameter. If the
        agent is already started, the trait's ``on_start()`` is called
        automatically; if it raises, the trait is stopped and removed, and the
        exception propagates.

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
                self._safe_stop_trait(trait, "trait stop after a failed start failed")
                self._traits.unregister(type(trait))
                raise

    def use(self, trait_type: type[Trait[Any]]) -> None:
        """Bind the platform's ``trait_type`` to this agent and hold the binding.

        The binding starts now if the agent is started; otherwise with it. A
        binding that fails to start is stopped and dropped, and the exception
        propagates.

        Raises:
            DuplicateTraitError: The agent already holds a binding of ``trait_type``;
                it is kept as it is.
            TraitNotFoundError: The platform has no ``trait_type`` to bind.
        """
        if trait_type in self._bindings:
            raise DuplicateTraitError(f"{self.name} already uses {trait_type.__name__}")
        trait = self._platform.traits.get(trait_type)
        if not isinstance(trait, Trait):
            raise TraitNotFoundError(
                f"{self.name} uses {trait_type.__name__}, which the platform does not have"
            )
        binding = trait.bind(self)
        self._bindings[trait_type] = binding
        self._lg.debug("trait bound", extra={"agent": self.name, "trait": trait_type.__name__})
        if self._started:
            try:
                binding.on_start()
            except Exception:
                self._safe_stop_trait(binding, "binding stop after a failed start failed")
                del self._bindings[trait_type]
                raise

    @overload
    def get(self, trait_type: type[Trait[BindingT]]) -> BindingT | None:
        """A platform trait: the agent's binding of it."""

    @overload
    def get(self, trait_type: type[TraitT]) -> TraitT | None:
        """A trait the agent built itself."""

    def get(self, trait_type: type[Any]) -> Any:
        """What the agent holds for ``trait_type``, or None.

        For a platform trait, the agent's binding of it; otherwise the
        trait the agent built itself.
        """
        binding = self._bindings.get(trait_type)
        if binding is not None:
            return binding
        return self._traits.get(trait_type)

    def has(self, trait_type: type[Any]) -> bool:
        """Whether :meth:`get` finds ``trait_type``."""
        return trait_type in self._bindings or self._traits.has(trait_type)

    @overload
    def require(self, trait_type: type[Trait[BindingT]]) -> BindingT:
        """A platform trait: the agent's binding of it."""

    @overload
    def require(self, trait_type: type[TraitT]) -> TraitT:
        """A trait the agent built itself."""

    def require(self, trait_type: type[Any]) -> Any:
        """What the agent holds for ``trait_type``, as :meth:`get` finds it.

        Raises:
            TraitNotFoundError: The agent holds nothing for ``trait_type``.
        """
        held = self.get(trait_type)
        if held is None:
            raise TraitNotFoundError(
                f"{trait_type.__name__} required but not attached - add it with "
                f"agent.add_trait({trait_type.__name__}(...)), or agent.use() the platform's"
            )
        return held

    def replace_trait(self, trait: TraitBinding) -> None:
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
        traits like :class:`SAIATrait` with ``owns_saia=True`` it releases
        what it owns. Re-fetch via :meth:`get` or :meth:`require` to obtain a
        usable reference.

        Typical pairing with ``SAIATrait.with_saia``::

            agent.replace_trait(agent.require(SAIATrait).with_saia(other_saia))
            saia_trait = agent.require(SAIATrait)  # reload

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
        trait: TraitBinding,
        old: TraitBinding | None,
        trait_type: type[TraitBinding],
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

    def _log_trait_replaced(self, trait_type: type[TraitBinding], *, ejected: bool) -> None:
        """Debug-log the outcome of a successful replace_trait call."""
        self._lg.debug(
            "trait replaced",
            extra={"agent": self.name, "trait": trait_type.__name__, "ejected": ejected},
        )

    # =========================================================================
    # Trait Lifecycle Helpers
    # =========================================================================

    def _safe_stop_trait(self, trait: TraitBinding, context: str) -> None:
        """Attempt on_stop, logging but swallowing any exception."""
        try:
            trait.on_stop()
        except Exception as e:
            self._lg.warning(context, extra={"exception": e, "trait": type(trait).__name__})

    def _held(self) -> list[TraitBinding]:
        """What the agent holds: its bindings of platform traits, then its own traits."""
        return [*self._bindings.values(), *self._traits.all()]

    def _start_traits(self) -> None:
        """Start what the agent holds; flip ``_started`` only on success.

        When one fails to start, it is stopped (releasing what it acquired before
        failing), then the ones started before it in reverse, and the exception
        propagates: nothing is left running.
        """
        started: list[TraitBinding] = []
        for held in self._held():
            try:
                held.on_start()
            except Exception:
                for done in [held, *reversed(started)]:
                    self._safe_stop_trait(done, "trait stop after a failed start failed")
                raise
            started.append(held)
            self._lg.debug(
                "trait started",
                extra={"agent": self.name, "trait": type(held).__name__},
            )
        self._started = True

    def _stop_traits(self) -> None:
        """Stop what the agent holds, in reverse of the start; flip ``_started`` regardless.

        Its own traits stop last-added first, then its bindings: a trait that
        used a binding when it started stops while that binding still runs.
        """
        for held in reversed(self._held()):
            try:
                held.on_stop()
                self._lg.debug(
                    "trait stopped",
                    extra={"agent": self.name, "trait": type(held).__name__},
                )
            except Exception as e:
                self._lg.warning(
                    "error stopping trait",
                    extra={"trait": type(held).__name__, "exception": e},
                )
        self._started = False

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    def _reject_platform_settings(self) -> None:
        """Raise ``ConfigError`` for agent config that only the platform sets.

        Models and the memory schema are the platform's: an agent's ``llm`` key
        or ``kelt.schema`` would have no effect, so it is refused however the
        agent is built.
        """
        if "llm" in self.config:
            raise ConfigError(
                "agent config has an 'llm' key: every agent uses the platform's model router; "
                "configure backends in the platform config's llm section"
            )
        if (self.config.get("kelt") or {}).get("schema"):
            raise ConfigError(
                "agent config sets kelt.schema: the memory schema is the platform's; "
                "set memory.schema in the platform config, or partition with identity.context_key"
            )

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
