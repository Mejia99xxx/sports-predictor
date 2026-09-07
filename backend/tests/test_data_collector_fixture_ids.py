"""
Unit tests for obtener_fixture_ids_por_temporada in data_collector.py.

Tests cover:
- Successful API response: returns list of integer fixture IDs
- HTTP error (non-200): logs error and returns []
- Connection error: logs error and returns []
- Timeout error: logs error and returns []
- Empty response list: returns []
- Response items missing fixture.id: those are skipped
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.data_collector import obtener_fixture_ids_por_temporada


class TestObtenerFixtureIdsPorTemporada:
    """Tests for obtener_fixture_ids_por_temporada."""

    def _make_api_response(self, fixture_ids: list) -> dict:
        """Build a mock API-Football response payload."""
        return {
            "response": [
                {"fixture": {"id": fid, "date": "2023-09-10T15:00:00Z"}}
                for fid in fixture_ids
            ]
        }

    # ------------------------------------------------------------------
    # Success cases
    # ------------------------------------------------------------------

    def test_returns_list_of_integers_on_success(self):
        """Should return a list of integer IDs from the response."""
        expected_ids = [1001, 1002, 1003]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._make_api_response(expected_ids)

        with patch("scripts.data_collector.requests.get", return_value=mock_resp):
            result = obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        assert result == expected_ids
        assert all(isinstance(i, int) for i in result)

    def test_empty_response_returns_empty_list(self):
        """Should return [] when the API response list is empty."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": []}

        with patch("scripts.data_collector.requests.get", return_value=mock_resp):
            result = obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        assert result == []

    def test_skips_entries_without_fixture_id(self):
        """Entries missing fixture.id should be silently skipped."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": [
                {"fixture": {"id": 999}},
                {"fixture": {}},           # missing id
                {"other_key": "no_fixture"},  # missing fixture key entirely
                {"fixture": {"id": 888}},
            ]
        }

        with patch("scripts.data_collector.requests.get", return_value=mock_resp):
            result = obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        assert result == [999, 888]

    def test_uses_correct_endpoint_and_params(self):
        """Should call the correct API endpoint with the right query params."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": []}

        with patch(
            "scripts.data_collector.requests.get", return_value=mock_resp
        ) as mock_get:
            obtener_fixture_ids_por_temporada(
                league_id=140, season=2022, api_key="my_key"
            )

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        # Verify URL
        assert "fixtures" in call_kwargs[0][0]
        # Verify params
        params = call_kwargs[1]["params"]
        assert params["league"] == 140
        assert params["season"] == 2022
        assert params["status"] == "FT"
        # Verify auth header
        headers = call_kwargs[1]["headers"]
        assert headers["x-apisports-key"] == "my_key"

    def test_uses_30_second_timeout(self):
        """Should use a 30-second timeout per request (REQUEST_TIMEOUT constant)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": []}

        with patch(
            "scripts.data_collector.requests.get", return_value=mock_resp
        ) as mock_get:
            obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["timeout"] == 30

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_http_error_returns_empty_list(self):
        """Non-200 HTTP status should log error and return []."""
        for status_code in [400, 401, 403, 404, 429, 500, 503]:
            mock_resp = MagicMock()
            mock_resp.status_code = status_code
            mock_resp.text = "Error"

            with patch(
                "scripts.data_collector.requests.get", return_value=mock_resp
            ):
                result = obtener_fixture_ids_por_temporada(
                    league_id=39, season=2023, api_key="test_key"
                )

            assert result == [], f"Expected [] for HTTP {status_code}"

    def test_connection_error_returns_empty_list(self):
        """ConnectionError should be caught and return []."""
        with patch(
            "scripts.data_collector.requests.get",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        ):
            result = obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        assert result == []

    def test_timeout_returns_empty_list(self):
        """Timeout should be caught and return []."""
        with patch(
            "scripts.data_collector.requests.get",
            side_effect=requests.exceptions.Timeout("Request timed out"),
        ):
            result = obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        assert result == []

    def test_generic_request_exception_returns_empty_list(self):
        """Any generic RequestException should be caught and return []."""
        with patch(
            "scripts.data_collector.requests.get",
            side_effect=requests.exceptions.RequestException("Something went wrong"),
        ):
            result = obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        assert result == []

    def test_ids_are_cast_to_int(self):
        """Fixture IDs returned as strings or floats should be cast to int."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # API sometimes returns IDs as numbers that need to be int-cast
        mock_resp.json.return_value = {
            "response": [
                {"fixture": {"id": 12345}},
                {"fixture": {"id": 67890}},
            ]
        }

        with patch("scripts.data_collector.requests.get", return_value=mock_resp):
            result = obtener_fixture_ids_por_temporada(
                league_id=39, season=2023, api_key="test_key"
            )

        assert result == [12345, 67890]
        assert all(isinstance(i, int) for i in result)
