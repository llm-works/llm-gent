"""Tests for the hub unified agent registry."""

from datetime import UTC, datetime, timedelta

import pytest
from appinfra import DotDict

from llm_gent.bus.protocol import AgentStats
from llm_gent.hub.registry import AgentEntry, AgentHealth, AgentType, Registry


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# AgentEntry
# ---------------------------------------------------------------------------


class TestAgentEntryIsAlive:
    """Tests for AgentEntry.is_alive()."""

    def test_alive_within_default_timeout(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL)
        assert entry.is_alive() is True

    def test_dead_after_default_timeout(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, dead_timeout=10.0)
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=15)
        assert entry.is_alive() is False

    def test_custom_timeout_overrides_default(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, dead_timeout=10.0)
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=5)
        assert entry.is_alive(timeout_secs=3.0) is False

    def test_custom_timeout_keeps_alive(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, dead_timeout=2.0)
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=3)
        assert entry.is_alive(timeout_secs=10.0) is True


class TestAgentEntryHealth:
    """Tests for AgentEntry.health property."""

    def test_healthy(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, dead_timeout=60.0)
        assert entry.health == AgentHealth.HEALTHY

    def test_unhealthy(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, dead_timeout=10.0)
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=15)
        assert entry.health == AgentHealth.UNHEALTHY

    def test_dead(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, dead_timeout=10.0)
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=25)
        assert entry.health == AgentHealth.DEAD


class TestAgentEntryScheduleInterval:
    """Tests for AgentEntry.schedule_interval property."""

    def test_no_schedule_key(self):
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, config=DotDict())
        assert entry.schedule_interval is None

    def test_schedule_with_interval(self):
        cfg = DotDict({"schedule": {"interval": 300}})
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, config=cfg)
        assert entry.schedule_interval == 300.0

    def test_schedule_string_interval(self):
        cfg = DotDict({"schedule": {"interval": "120"}})
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, config=cfg)
        assert entry.schedule_interval == 120.0

    def test_schedule_non_numeric_interval(self):
        cfg = DotDict({"schedule": {"interval": "never"}})
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, config=cfg)
        assert entry.schedule_interval is None

    def test_schedule_none_interval(self):
        cfg = DotDict({"schedule": {"interval": None}})
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, config=cfg)
        assert entry.schedule_interval is None

    def test_schedule_not_dict(self):
        cfg = DotDict({"schedule": "daily"})
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, config=cfg)
        assert entry.schedule_interval is None

    def test_schedule_missing_interval_key(self):
        cfg = DotDict({"schedule": {"cron": "0 * * * *"}})
        entry = AgentEntry(id="a", agent_type=AgentType.EXTERNAL, config=cfg)
        assert entry.schedule_interval is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistryRegisterOrMerge:
    """Tests for register_or_merge, focusing on the INJECTED merge path."""

    def test_merge_into_injected(self):
        reg = Registry()
        cfg = DotDict({"key": "val"})
        reg.register("bot", AgentType.INJECTED, config=cfg)

        result = reg.register_or_merge(
            "bot",
            agent_type=AgentType.EXTERNAL,
            capabilities=["search"],
            metadata={"version": 2},
        )

        assert result.agent_type == AgentType.INJECTED
        assert result.capabilities == ["search"]
        assert result.metadata["version"] == 2
        assert result.config.get("key") == "val"

    def test_merge_updates_heartbeat(self):
        reg = Registry()
        reg.register("bot", AgentType.INJECTED)
        entry = reg.get("bot")
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=5)
        old_hb = entry.last_heartbeat
        reg.register_or_merge("bot")
        assert reg.get("bot").last_heartbeat > old_hb

    def test_merge_no_caps_preserves_existing(self):
        reg = Registry()
        reg.register("bot", AgentType.INJECTED, capabilities=["chat"])
        reg.register_or_merge("bot", metadata={"v": 1})
        assert reg.get("bot").capabilities == ["chat"]

    def test_new_external_when_not_injected(self):
        reg = Registry()
        result = reg.register_or_merge("new-bot", capabilities=["search"])
        assert result.agent_type == AgentType.EXTERNAL
        assert result.capabilities == ["search"]


class TestRegistryHeartbeat:
    """Tests for heartbeat() with stats updates."""

    def test_heartbeat_unknown_agent(self):
        reg = Registry()
        assert reg.heartbeat("ghost") is False

    def test_heartbeat_updates_timestamp(self):
        reg = Registry()
        reg.register("bot", AgentType.EXTERNAL)
        entry = reg.get("bot")
        entry.last_heartbeat = datetime.now(UTC) - timedelta(seconds=10)

        assert reg.heartbeat("bot") is True
        assert (datetime.now(UTC) - reg.get("bot").last_heartbeat).total_seconds() < 2

    def test_heartbeat_with_increased_ticks_sets_last_run(self):
        reg = Registry()
        reg.register("bot", AgentType.EXTERNAL)
        assert reg.get("bot").last_run is None

        stats = AgentStats(ticks=1)
        reg.heartbeat("bot", stats=stats)

        entry = reg.get("bot")
        assert entry.last_run is not None
        assert entry.stats.ticks == 1

    def test_heartbeat_with_same_ticks_no_last_run(self):
        reg = Registry()
        reg.register("bot", AgentType.EXTERNAL)

        stats = AgentStats(ticks=0)
        reg.heartbeat("bot", stats=stats)

        assert reg.get("bot").last_run is None

    def test_heartbeat_replaces_stats(self):
        reg = Registry()
        reg.register("bot", AgentType.EXTERNAL)

        stats = AgentStats(ticks=0, errors=5)
        reg.heartbeat("bot", stats=stats)

        assert reg.get("bot").stats.errors == 5


class TestRegistryListAndFilter:
    """Tests for list_agents, get_by_type, get_healthy, get_unhealthy, get_dead."""

    @pytest.fixture
    def populated_registry(self):
        reg = Registry(dead_timeout=10.0)
        reg.register("healthy", AgentType.EXTERNAL)
        reg.register("sick", AgentType.INJECTED)
        reg.register("gone", AgentType.EXTERNAL)

        reg.get("sick").last_heartbeat = datetime.now(UTC) - timedelta(seconds=15)
        reg.get("gone").last_heartbeat = datetime.now(UTC) - timedelta(seconds=25)
        return reg

    def test_list_agents(self, populated_registry):
        agents = populated_registry.list_agents()
        assert len(agents) == 3
        ids = {a.id for a in agents}
        assert ids == {"healthy", "sick", "gone"}

    def test_get_by_type(self, populated_registry):
        externals = populated_registry.get_by_type(AgentType.EXTERNAL)
        assert {a.id for a in externals} == {"healthy", "gone"}

        injected = populated_registry.get_by_type(AgentType.INJECTED)
        assert [a.id for a in injected] == ["sick"]

    def test_get_healthy(self, populated_registry):
        healthy = populated_registry.get_healthy()
        assert [a.id for a in healthy] == ["healthy"]

    def test_get_unhealthy(self, populated_registry):
        unhealthy = populated_registry.get_unhealthy()
        assert [a.id for a in unhealthy] == ["sick"]

    def test_get_dead(self, populated_registry):
        dead = populated_registry.get_dead()
        assert [a.id for a in dead] == ["gone"]


class TestRegistryIncrementRestart:
    """Tests for increment_restart()."""

    def test_increment_existing(self):
        reg = Registry()
        reg.register("bot", AgentType.EXTERNAL)

        assert reg.increment_restart("bot") == 1
        assert reg.increment_restart("bot") == 2
        assert reg.get("bot").restart_count == 2

    def test_increment_unknown(self):
        reg = Registry()
        assert reg.increment_restart("ghost") == 0


class TestRegistryHealthyCount:
    """Tests for healthy_count property."""

    def test_healthy_count(self):
        reg = Registry(dead_timeout=10.0)
        reg.register("a", AgentType.EXTERNAL)
        reg.register("b", AgentType.EXTERNAL)
        reg.register("c", AgentType.EXTERNAL)

        reg.get("c").last_heartbeat = datetime.now(UTC) - timedelta(seconds=15)

        assert reg.count == 3
        assert reg.healthy_count == 2
