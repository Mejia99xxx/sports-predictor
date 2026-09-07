"""
Unit tests for services.py — task 1.1
Covers: obtener_ultimo_partido_stats
Requirements: 2.1, 2.2, 2.3, 2.4
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import requests as requests_lib

# Add backend dir to path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import obtener_ultimo_partido_stats


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_fixture(fixture_id: int, date: str) -> dict:
    """Build a minimal fixture dict matching the API-Football response shape."""
    return {
        "fixture": {"id": fixture_id, "date": date},
        "goals": {"home": 1, "away": 0},
        "teams": {
            "home": {"id": 33, "name": "Manchester United"},
            "away": {"id": 34, "name": "Newcastle"},
        },
    }


def _make_stats_response() -> list:
    """Return a minimal statistics list (what obtener_estadisticas_partido returns)."""
    return [
        {
            "team": {"id": 33, "name": "Manchester United"},
            "statistics": [
                {"type": "Ball Possession", "value": "55%"},
                {"type": "Total Shots", "value": 10},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Test: empty fixture list returns []
# ---------------------------------------------------------------------------

class TestObtenerUltimoPartidoStatsEmptyFixtures:
    def test_empty_response_returns_empty_list(self):
        """When obtener_partidos_por_equipo returns no fixtures, return []."""
        with patch("services.obtener_partidos_por_equipo") as mock_partidos:
            mock_partidos.return_value = {"response": []}
            result = obtener_ultimo_partido_stats(33)
        assert result == []

    def test_missing_response_key_returns_empty_list(self):
        """When the API response dict has no 'response' key, return []."""
        with patch("services.obtener_partidos_por_equipo") as mock_partidos:
            mock_partidos.return_value = {}
            result = obtener_ultimo_partido_stats(33)
        assert result == []

    def test_non_dict_api_response_returns_empty_list(self):
        """When obtener_partidos_por_equipo returns a non-dict (e.g. {}), return []."""
        with patch("services.obtener_partidos_por_equipo") as mock_partidos:
            mock_partidos.return_value = {}
            result = obtener_ultimo_partido_stats(99)
        assert result == []


# ---------------------------------------------------------------------------
# Test: most recent fixture is selected
# ---------------------------------------------------------------------------

class TestObtenerUltimoPartidoStatsMostRecent:
    def test_selects_most_recent_fixture_from_multiple(self):
        """
        When multiple fixtures are returned, the one with the most recent
        fixture.date must be used to fetch statistics.
        """
        fixtures = [
            _make_fixture(fixture_id=100, date="2024-01-15T20:00:00+00:00"),
            _make_fixture(fixture_id=200, date="2024-03-10T18:00:00+00:00"),  # most recent
            _make_fixture(fixture_id=150, date="2024-02-20T21:00:00+00:00"),
        ]
        expected_stats = _make_stats_response()

        with patch("services.obtener_partidos_por_equipo") as mock_partidos, \
             patch("services.obtener_estadisticas_partido") as mock_stats:
            mock_partidos.return_value = {"response": fixtures}
            mock_stats.return_value = expected_stats

            result = obtener_ultimo_partido_stats(33)

            # Must have called obtener_estadisticas_partido with the MOST RECENT fixture id
            mock_stats.assert_called_once_with(200)
            assert result == expected_stats

    def test_single_fixture_is_used(self):
        """When only one fixture is present, it is used regardless of date."""
        fixture = _make_fixture(fixture_id=500, date="2024-01-01T12:00:00+00:00")
        expected_stats = _make_stats_response()

        with patch("services.obtener_partidos_por_equipo") as mock_partidos, \
             patch("services.obtener_estadisticas_partido") as mock_stats:
            mock_partidos.return_value = {"response": [fixture]}
            mock_stats.return_value = expected_stats

            result = obtener_ultimo_partido_stats(33)

            mock_stats.assert_called_once_with(500)
            assert result == expected_stats

    def test_returns_statistics_list_from_most_recent_fixture(self):
        """The raw statistics list returned by obtener_estadisticas_partido is passed through."""
        fixtures = [
            _make_fixture(fixture_id=1, date="2023-11-01T00:00:00+00:00"),
            _make_fixture(fixture_id=2, date="2024-04-01T00:00:00+00:00"),  # most recent
        ]
        expected = [{"team": {"id": 42}, "statistics": []}]

        with patch("services.obtener_partidos_por_equipo") as mock_partidos, \
             patch("services.obtener_estadisticas_partido") as mock_stats:
            mock_partidos.return_value = {"response": fixtures}
            mock_stats.return_value = expected

            result = obtener_ultimo_partido_stats(42)

        assert result == expected


# ---------------------------------------------------------------------------
# Test: API connection error returns []
# ---------------------------------------------------------------------------

class TestObtenerUltimoPartidoStatsConnectionError:
    def test_requests_connection_error_returns_empty_list(self):
        """When obtener_partidos_por_equipo raises a ConnectionError, return []."""
        with patch("services.obtener_partidos_por_equipo") as mock_partidos:
            mock_partidos.side_effect = requests_lib.exceptions.ConnectionError("timeout")
            result = obtener_ultimo_partido_stats(33)
        assert result == []

    def test_requests_timeout_returns_empty_list(self):
        """When obtener_partidos_por_equipo raises a Timeout, return []."""
        with patch("services.obtener_partidos_por_equipo") as mock_partidos:
            mock_partidos.side_effect = requests_lib.exceptions.Timeout("timed out")
            result = obtener_ultimo_partido_stats(33)
        assert result == []

    def test_unexpected_exception_returns_empty_list(self):
        """Any unexpected exception in the pipeline is caught and [] is returned."""
        with patch("services.obtener_partidos_por_equipo") as mock_partidos:
            mock_partidos.side_effect = RuntimeError("unexpected")
            result = obtener_ultimo_partido_stats(33)
        assert result == []

    def test_stats_call_error_returns_empty_list(self):
        """When obtener_estadisticas_partido raises an error, return []."""
        fixture = _make_fixture(fixture_id=99, date="2024-01-01T00:00:00+00:00")
        with patch("services.obtener_partidos_por_equipo") as mock_partidos, \
             patch("services.obtener_estadisticas_partido") as mock_stats:
            mock_partidos.return_value = {"response": [fixture]}
            mock_stats.side_effect = requests_lib.exceptions.ConnectionError("down")

            result = obtener_ultimo_partido_stats(33)

        assert result == []
