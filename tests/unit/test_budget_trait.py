# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for BudgetTrait: tracker accessor + lifecycle hooks."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra.log import quick_console_logger

from llm_gent.core.budget import PricingConfig, Tracker
from llm_gent.core.traits import BudgetTrait


pytestmark = pytest.mark.unit


def _tracker() -> Tracker:
    lg = quick_console_logger("test", config={"level": "error"})
    return Tracker(lg, PricingConfig(), budget=1.0)


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.name = "test-agent"
    agent.lg = MagicMock()
    return agent


class TestBudgetTrait:
    """Thin wrapper: exposes tracker; logs on lifecycle hooks."""

    def test_tracker_accessor(self) -> None:
        tracker = _tracker()
        trait = BudgetTrait(_agent(), tracker=tracker)
        assert trait.tracker is tracker

    def test_on_start_logs(self) -> None:
        agent = _agent()
        tracker = _tracker()
        trait = BudgetTrait(agent, tracker=tracker)
        trait.on_start()
        agent.lg.trace.assert_called_once()
        call = agent.lg.trace.call_args
        assert call.kwargs["extra"]["budget"] == pytest.approx(1.0)

    def test_on_stop_logs_summary(self) -> None:
        agent = _agent()
        tracker = _tracker()
        trait = BudgetTrait(agent, tracker=tracker)
        trait.on_stop()
        agent.lg.info.assert_called_once()
        extra = agent.lg.info.call_args.kwargs["extra"]
        assert extra["budget"] == pytest.approx(1.0)
        assert extra["spent"] == 0.0
        assert extra["exceeded"] is False
