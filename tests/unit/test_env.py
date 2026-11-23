# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The Environment: one SAIA per role, built on first use, and the traits."""

from __future__ import annotations

from typing import Any

import pytest

from llm_gent.core import Environment, Role
from llm_gent.core.traits import Registry as TraitRegistry
from tests.helpers import make_test_logger


pytestmark = pytest.mark.unit

LG = make_test_logger()


class _CountingFactory:
    """Builds a fresh object per call and records the roles it was asked for."""

    def __init__(self) -> None:
        self.built: list[Role] = []

    def build(self, role: Role) -> Any:
        self.built.append(role)
        return object()


class TestSaia:
    def test_a_role_is_built_once_and_kept(self) -> None:
        factory = _CountingFactory()
        env = Environment(saia_factory=factory)
        role = Role(name="planner")
        first = env.saia(role)
        assert env.saia(role) is first
        assert env.saia(Role(name="planner")) is first
        assert factory.built == [role]

    def test_each_role_gets_its_own_saia(self) -> None:
        env = Environment(saia_factory=_CountingFactory())
        assert env.saia(Role(name="planner")) is not env.saia(Role(name="worker"))

    def test_roles_differing_only_in_params_build_apart(self) -> None:
        factory = _CountingFactory()
        env = Environment(saia_factory=factory)
        role = Role(name="planner")
        varied = role.with_params(max_iterations=3)
        assert env.saia(role) is not env.saia(varied)
        assert factory.built == [role, varied]

    def test_environments_do_not_share_saias(self) -> None:
        factory = _CountingFactory()
        role = Role(name="planner")
        assert Environment(saia_factory=factory).saia(role) is not Environment(
            saia_factory=factory
        ).saia(role)

    def test_without_a_factory_an_ask_raises(self) -> None:
        env = Environment()
        assert env.saia_factory is None
        with pytest.raises(RuntimeError, match="no SAIAFactory .* role 'planner'"):
            env.saia(Role(name="planner"))


class TestTraits:
    def test_traits_are_the_registry_given(self) -> None:
        registry = TraitRegistry(LG)
        assert Environment(traits=registry).traits is registry

    def test_traits_are_none_without_a_registry(self) -> None:
        assert Environment().traits is None
