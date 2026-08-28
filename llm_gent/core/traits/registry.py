# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Trait registry for managing available platform traits.

The registry is a container for all configured platform traits (LLM, Learn, HTTP, etc.)
that can be attached to agents. It provides type-safe access and discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from appinfra.log import Logger

from ..errors import DuplicateTraitError, TraitNotFoundError
from .base import BaseTrait


if TYPE_CHECKING:
    from . import TraitName


T = TypeVar("T", bound=BaseTrait)


class Registry:
    """Registry of available platform traits.

    Manages the collection of configured traits that can be attached to agents.
    Provides type-safe access and trait discovery.

    Example:
        # Agent uses registry internally to manage its traits
        agent = MyAgent(lg, config)

        # Traits are registered via agent.add_trait()
        agent.add_trait(LLMTrait(agent, llm_config))
        agent.add_trait(LearnTrait(agent, learn_config))

        # Access traits by type through agent
        llm_trait = agent.get_trait(LLMTrait)
        if llm_trait:
            # Use the trait
            pass

        # Or require it (raises if missing)
        learn_trait = agent.require_trait(LearnTrait)
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize empty trait registry.

        Args:
            lg: Logger instance for registry operations.
        """
        self._lg = lg
        self._traits: dict[type[BaseTrait], BaseTrait] = {}
        self._by_name: dict[TraitName, BaseTrait] = {}

    def register(self, trait: BaseTrait) -> None:
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

        self._lg.debug(
            "trait registered",
            extra={"trait": trait_type.__name__, "total": len(self._traits)},
        )

    def replace(self, trait: BaseTrait) -> None:
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

        self._lg.debug("trait replaced", extra={"trait": trait_type.__name__})

    def get(self, trait_type: type[T]) -> T | None:
        """Get a trait by type.

        Args:
            trait_type: The trait class to retrieve.

        Returns:
            Trait instance if registered, None otherwise.
        """
        return self._traits.get(trait_type)  # type: ignore[return-value]

    def get_by_name(self, name: TraitName) -> BaseTrait | None:
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

    def require_by_name(self, name: TraitName) -> BaseTrait:
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

    def has(self, trait_type: type[BaseTrait]) -> bool:
        """Check if a trait type is registered.

        Args:
            trait_type: The trait class to check.

        Returns:
            True if trait is registered, False otherwise.
        """
        return trait_type in self._traits

    def all(self) -> list[BaseTrait]:
        """Get all registered traits.

        Returns:
            List of all registered trait instances.
        """
        return list(self._traits.values())

    def types(self) -> list[type[BaseTrait]]:
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

    def unregister(self, trait_type: type[BaseTrait]) -> None:
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

        self._lg.debug("trait unregistered", extra={"trait": trait_type.__name__})

    def clear(self) -> None:
        """Remove all traits from registry."""
        self._traits.clear()
        self._by_name.clear()
        self._lg.debug("trait registry cleared")

    def __repr__(self) -> str:
        """String representation of registry."""
        trait_names = [t.__name__ for t in self._traits]
        return f"Registry({', '.join(trait_names)})"
