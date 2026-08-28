# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Tests for CLI tools."""

from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import pytest

from llm_gent.cli.tools.ask import AskTool
from llm_gent.cli.tools.feedback import FeedbackTool
from llm_gent.cli.tools.start import StartTool
from llm_gent.cli.tools.stop import StopTool


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MagicMock()


class TestAskTool:
    """Tests for AskTool."""

    def test_ask_agent_http_status_error(self, mock_logger):
        """HTTPStatusError is handled gracefully."""
        error_response = MagicMock()
        error_response.status_code = 500

        with patch.object(AskTool, "lg", new_callable=PropertyMock) as mock_lg:
            mock_lg.return_value = mock_logger
            tool = AskTool()

            with patch.object(httpx, "post") as mock_post:
                mock_post.return_value.status_code = 500
                mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Internal Server Error", request=MagicMock(), response=error_response
                )
                result = tool._ask_agent("http://localhost:8080", "test", "question")

        assert result is None
        mock_logger.error.assert_called_once()


class TestStartTool:
    """Tests for StartTool."""

    def test_start_agent_http_status_error(self, mock_logger):
        """HTTPStatusError is handled gracefully."""
        error_response = MagicMock()
        error_response.status_code = 500

        with patch.object(StartTool, "lg", new_callable=PropertyMock) as mock_lg:
            mock_lg.return_value = mock_logger
            tool = StartTool()

            with patch.object(httpx, "post") as mock_post:
                mock_post.return_value.status_code = 500
                mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Internal Server Error", request=MagicMock(), response=error_response
                )
                result = tool._start_agent("http://localhost:8080", "test")

        assert result is None
        mock_logger.error.assert_called_once()


class TestStopTool:
    """Tests for StopTool."""

    def test_stop_agent_http_status_error(self, mock_logger):
        """HTTPStatusError is handled gracefully."""
        error_response = MagicMock()
        error_response.status_code = 500

        with patch.object(StopTool, "lg", new_callable=PropertyMock) as mock_lg:
            mock_lg.return_value = mock_logger
            tool = StopTool()

            with patch.object(httpx, "post") as mock_post:
                mock_post.return_value.status_code = 500
                mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Internal Server Error", request=MagicMock(), response=error_response
                )
                result = tool._stop_agent("http://localhost:8080", "test")

        assert result is None
        mock_logger.error.assert_called_once()


class TestFeedbackTool:
    """Tests for FeedbackTool."""

    def test_send_feedback_http_status_error(self, mock_logger):
        """HTTPStatusError is handled gracefully."""
        error_response = MagicMock()
        error_response.status_code = 500

        with patch.object(FeedbackTool, "lg", new_callable=PropertyMock) as mock_lg:
            mock_lg.return_value = mock_logger
            tool = FeedbackTool()

            with patch.object(httpx, "post") as mock_post:
                mock_post.return_value.status_code = 500
                mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Internal Server Error", request=MagicMock(), response=error_response
                )
                result = tool._send_feedback("http://localhost:8080", "test", "message")

        assert result is None
        mock_logger.error.assert_called_once()
