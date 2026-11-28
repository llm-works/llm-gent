# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Trait registry: type-safe access and discovery by type and by name.

The platform keeps its traits in one (``platform.traits``); an agent keeps
its bindings and its own traits in another. Both kinds have the
``on_start`` / ``on_stop`` lifecycle, so the type parameter is bound to the
:class:`TraitBinding` protocol rather than :class:`BaseTrait`: anything that
implements the protocol directly type-checks cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from appinfra.log import Logger

from ..errors import DuplicateTraitError, TraitNotFoundError
from .base import TraitBinding


if TYPE_CHECKING:
    from . import TraitName


T = TypeVar("T", bound=TraitBinding)


class Registry:
    """Registry of available platform traits.

    Manages the collection of configured traits that can be attached to agents.
    Provides type-safe access and trait discovery.

    Example:
        # An agent keeps its own traits in a registry; the platform keeps the
        # shared ones (platform.traits) in another.
        agent = MyAgent(lg, platform, config)

        # The platform's TraitFactory builds an agent's own traits on the
        # platform's shared clients.
        agent.add_trait(platform.trait_factory.create(TraitName.DIRECTIVE, agent))
        agent.use(MemoryTrait)  # the platform's, bound to the agent's context key

        # Access traits by type through agent
        directive_trait = agent.get(DirectiveTrait)
        if directive_trait:
            # Use the trait
            pass

        # Or require it (raises if missing)
        memory_trait = agent.require(MemoryTrait)
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize empty trait registry.

        Args:
            lg: Logger instance for registry operations.
        """
        self._lg = lg
        self._traits: dict[type[TraitBinding], TraitBinding] = {}
        self._by_name: dict[TraitName, TraitBinding] = {}

    def register(self, trait: TraitBinding) -> None:
        """Register a trait in the registry.

        Automatically registers by name if trait has trait_name attribute.

        Args:
            trait: Trait instance to register.

        Raises:
            DuplicateTraitError: If a trait of this type is already registered.
        """
        trait_type = type(trait)
        if trait_type in self._traits:
            raise DuplicateTraitError(
                f"Trait {trait_type.__name__} is already registered. "
                "Use replace() to update an existing trait."
            )

        # Check for name collision before registering
        if hasattr(trait, "trait_name"):
            name = trait.trait_name
            existing = self._by_name.get(name)
            if existing is not None and type(existing) is not trait_type:
                raise DuplicateTraitError(
                    f"Trait name '{name}' is already registered for {type(existing).__name__}."
                )

        self._traits[trait_type] = trait

        # Auto-register by name if trait declares one
        if hasattr(trait, "trait_name"):
            self._by_name[trait.trait_name] = trait

        self._lg.trace(
            "trait registered",
            extra={"trait": trait_type.__name__, "total": len(self._traits)},
        )

    def replace(self, trait: TraitBinding) -> None:
        """Replace an existing trait or register if not present.

        Args:
            trait: Trait instance to register/replace.
        """
        trait_type = type(trait)
        old = self._traits.get(trait_type)
        self._traits[trait_type] = trait

        # Remove old name mapping if it exists and differs from new name
        if old is not None and hasattr(old, "trait_name"):
            old_name = old.trait_name
            new_name = getattr(trait, "trait_name", None)
            if old_name != new_name:
                self._by_name.pop(old_name, None)

        # Set new name mapping if trait has one
        if hasattr(trait, "trait_name"):
            self._by_name[trait.trait_name] = trait

        self._lg.trace("trait replaced", extra={"trait": trait_type.__name__})

    def get(self, trait_type: type[T]) -> T | None:
        """Get a trait by type.

        Args:
            trait_type: The trait class to retrieve.

        Returns:
            Trait instance if registered, None otherwise.
        """
        trait = self._traits.get(trait_type)
        self._lg.trace(
            "trait lookup",
            extra={"trait": trait_type.__name__, "hit": trait is not None},
        )
        return trait  # type: ignore[return-value]

    def get_by_name(self, name: TraitName) -> TraitBinding | None:
        """Get a trait by name.

        Args:
            name: The trait name to retrieve.

        Returns:
            Trait instance if registered with this name, None otherwise.
        """
        return self._by_name.get(name)

    def require(self, trait_type: type[T]) -> T:
        """Get a required trait by type.

        Args:
            trait_type: The trait class to retrieve.

        Returns:
            Trait instance.

        Raises:
            TraitNotFoundError: If trait is not registered.
        """
        trait = self.get(trait_type)
        if trait is None:
            raise TraitNotFoundError(
                f"Trait {trait_type.__name__} is required but not registered. "
                "Check platform configuration."
            )
        return trait

    def require_by_name(self, name: TraitName) -> TraitBinding:
        """Get a required trait by name.

        Args:
            name: The trait name to retrieve.

        Returns:
            Trait instance.

        Raises:
            TraitNotFoundError: If trait with this name is not registered.
        """
        trait = self.get_by_name(name)
        if trait is None:
            raise TraitNotFoundError(
                f"Trait '{name.value}' is required but not registered. "
                "Check platform configuration."
            )
        return trait

    def has(self, trait_type: type[TraitBinding]) -> bool:
        """Check if a trait type is registered.

        Args:
            trait_type: The trait class to check.

        Returns:
            True if trait is registered, False otherwise.
        """
        return trait_type in self._traits

    def all(self) -> list[TraitBinding]:
        """Get all registered traits.

        Returns:
            List of all registered trait instances.
        """
        return list(self._traits.values())

    def types(self) -> list[type[TraitBinding]]:
        """Get all registered trait types.

        Returns:
            List of registered trait classes.
        """
        return list(self._traits.keys())

    def count(self) -> int:
        """Get number of registered traits.

        Returns:
            Count of registered traits.
        """
        return len(self._traits)

    def unregister(self, trait_type: type[TraitBinding]) -> None:
        """Unregister a trait by type.

        Args:
            trait_type: The trait class to unregister.

        Raises:
            TraitNotFoundError: If trait is not registered.
        """
        if trait_type not in self._traits:
            raise TraitNotFoundError(
                f"Trait {trait_type.__name__} is not registered and cannot be unregistered."
            )

        trait = self._traits[trait_type]
        del self._traits[trait_type]

        # Remove name mapping if it exists
        if hasattr(trait, "trait_name"):
            self._by_name.pop(trait.trait_name, None)

        self._lg.trace("trait unregistered", extra={"trait": trait_type.__name__})

    def clear(self) -> None:
        """Remove all traits from registry."""
        self._traits.clear()
        self._by_name.clear()
        self._lg.debug("trait registry cleared")

    def __repr__(self) -> str:
        """String representation of registry."""
        trait_names = [t.__name__ for t in self._traits]
        return f"Registry({', '.join(trait_names)})"
