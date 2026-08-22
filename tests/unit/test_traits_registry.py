"""Tests for trait registry."""

from unittest.mock import MagicMock

import pytest

from llm_gent.core.errors import DuplicateTraitError, TraitNotFoundError
from llm_gent.core.traits import TraitName
from llm_gent.core.traits.base import BaseTrait
from llm_gent.core.traits.registry import Registry


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test trait classes
# ---------------------------------------------------------------------------


class FooTrait(BaseTrait):
    trait_name = TraitName.LLM  # reuse existing enum value for "llm"


class BarTrait(BaseTrait):
    trait_name = TraitName.MEMORY  # reuse existing enum value for "memory"


class NoNameTrait(BaseTrait):
    """Trait without trait_name attribute."""

    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lg():
    return MagicMock()


@pytest.fixture
def agent():
    return MagicMock()


@pytest.fixture
def registry(lg):
    return Registry(lg)


# ---------------------------------------------------------------------------
# register() — name collision
# ---------------------------------------------------------------------------


class TestRegisterNameCollision:
    """L80-83: same trait_name, different type raises DuplicateTraitError."""

    def test_same_name_different_type_raises(self, registry, agent):
        """Registering a different type with an already-taken name raises."""

        class ConflictTrait(BaseTrait):
            trait_name = TraitName.LLM  # same name as FooTrait

        registry.register(FooTrait(agent))

        with pytest.raises(DuplicateTraitError, match="already registered"):
            registry.register(ConflictTrait(agent))


# ---------------------------------------------------------------------------
# replace()
# ---------------------------------------------------------------------------


class TestReplace:
    """L102-117: replace existing trait or insert if absent."""

    def test_replace_existing_trait(self, registry, agent):
        foo1 = FooTrait(agent)
        foo2 = FooTrait(agent)
        registry.register(foo1)

        registry.replace(foo2)

        assert registry.get(FooTrait) is foo2
        assert registry.get_by_name(TraitName.LLM) is foo2

    def test_replace_with_name_change(self, registry, agent):
        """Old name mapping removed when replacement has a different name."""

        class RenamedFoo(FooTrait):
            trait_name = TraitName.HTTP  # different name

        foo = FooTrait(agent)
        registry.register(foo)
        assert registry.get_by_name(TraitName.LLM) is foo

        # Replace FooTrait-type with subclass that has a different name.
        # Because RenamedFoo is a *subclass*, its type differs — register a
        # plain FooTrait first, then replace it with a new FooTrait whose name
        # we override.
        renamed = FooTrait(agent)
        renamed.trait_name = TraitName.HTTP  # type: ignore[assignment]
        registry.replace(renamed)

        # Old name gone, new name mapped
        assert registry.get_by_name(TraitName.LLM) is None
        assert registry.get_by_name(TraitName.HTTP) is renamed

    def test_replace_when_not_present_inserts(self, registry, agent):
        """replace() on an unregistered type behaves like register()."""
        foo = FooTrait(agent)
        registry.replace(foo)

        assert registry.get(FooTrait) is foo
        assert registry.get_by_name(TraitName.LLM) is foo
        assert registry.count() == 1


# ---------------------------------------------------------------------------
# get_by_name()
# ---------------------------------------------------------------------------


class TestGetByName:
    """L139: lookup by TraitName."""

    def test_found(self, registry, agent):
        foo = FooTrait(agent)
        registry.register(foo)

        assert registry.get_by_name(TraitName.LLM) is foo

    def test_not_found_returns_none(self, registry):
        assert registry.get_by_name(TraitName.LLM) is None


# ---------------------------------------------------------------------------
# require_by_name()
# ---------------------------------------------------------------------------


class TestRequireByName:
    """L173-179: require_by_name raises when missing."""

    def test_found(self, registry, agent):
        foo = FooTrait(agent)
        registry.register(foo)

        assert registry.require_by_name(TraitName.LLM) is foo

    def test_not_found_raises(self, registry):
        with pytest.raises(TraitNotFoundError, match="required but not registered"):
            registry.require_by_name(TraitName.LLM)


# ---------------------------------------------------------------------------
# types() and count()
# ---------------------------------------------------------------------------


class TestTypesAndCount:
    """L206, L214."""

    def test_types_returns_registered_classes(self, registry, agent):
        registry.register(FooTrait(agent))
        registry.register(BarTrait(agent))

        result = registry.types()
        assert set(result) == {FooTrait, BarTrait}

    def test_count(self, registry, agent):
        assert registry.count() == 0
        registry.register(FooTrait(agent))
        assert registry.count() == 1
        registry.register(BarTrait(agent))
        assert registry.count() == 2


# ---------------------------------------------------------------------------
# unregister()
# ---------------------------------------------------------------------------


class TestUnregister:
    """L225-237: removes trait and name mapping, raises if not found."""

    def test_unregister_removes_trait_and_name(self, registry, agent):
        foo = FooTrait(agent)
        registry.register(foo)

        registry.unregister(FooTrait)

        assert registry.get(FooTrait) is None
        assert registry.get_by_name(TraitName.LLM) is None
        assert registry.count() == 0

    def test_unregister_trait_without_name(self, registry, agent):
        trait = NoNameTrait(agent)
        registry.register(trait)

        registry.unregister(NoNameTrait)

        assert registry.get(NoNameTrait) is None
        assert registry.count() == 0

    def test_unregister_not_found_raises(self, registry):
        with pytest.raises(TraitNotFoundError, match="not registered"):
            registry.unregister(FooTrait)


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    """L241-243: clears both dicts."""

    def test_clear_removes_all(self, registry, agent):
        registry.register(FooTrait(agent))
        registry.register(BarTrait(agent))

        registry.clear()

        assert registry.count() == 0
        assert registry.get_by_name(TraitName.LLM) is None
        assert registry.get_by_name(TraitName.MEMORY) is None
        assert registry.all() == []


# ---------------------------------------------------------------------------
# Protocol-only traits (Registry-mode adoption path)
# ---------------------------------------------------------------------------


class TestProtocolOnlyTrait:
    """Registry-mode adoption: a trait implementing the ``Trait`` protocol
    directly, without inheriting :class:`BaseTrait`, registers and is retrieved
    without a ``# type: ignore``. Guards against re-tightening the
    ``Registry.get`` TypeVar bound back to ``BaseTrait``.
    """

    def test_register_get_require(self, registry):
        class ProtoTrait:
            """Bare ``Trait`` protocol implementer — no BaseTrait, no Agent."""

            def on_start(self) -> None:
                pass

            def on_stop(self) -> None:
                pass

        trait = ProtoTrait()
        registry.register(trait)

        assert registry.get(ProtoTrait) is trait
        assert registry.require(ProtoTrait) is trait
        assert registry.has(ProtoTrait)
        assert ProtoTrait in registry.types()

    def test_unregister(self, registry):
        class ProtoTrait:
            def on_start(self) -> None:
                pass

            def on_stop(self) -> None:
                pass

        trait = ProtoTrait()
        registry.register(trait)
        registry.unregister(ProtoTrait)

        assert registry.get(ProtoTrait) is None
        assert not registry.has(ProtoTrait)
