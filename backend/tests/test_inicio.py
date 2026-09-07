"""
Unit tests for GET /inicio — Task 7.1
Tests that the /inicio route renders index.html with correct league selector,
disabled team dropdowns, and link to /partidos.

Requirements: 1.1, 1.3, 1.4, 5.5, 5.6
"""
import os
import sys

# Add backend/ to path so imports resolve when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch
import app as app_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a Flask test client with testing mode enabled."""
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    with app_module.app.test_client() as client:
        yield client


def _make_mock_ligas():
    """Return a minimal ligas response matching the API-Football shape."""
    return {
        "response": [
            {
                "league": {"id": 39, "name": "Premier League"},
                "country": {"name": "England"},
            },
            {
                "league": {"id": 140, "name": "La Liga"},
                "country": {"name": "Spain"},
            },
        ]
    }


# ---------------------------------------------------------------------------
# Test 1: Response contains league selector element (id="liga_id")
# Validates: Requirements 1.1, 5.5
# ---------------------------------------------------------------------------

class TestLeagueSelectorPresent:
    def test_response_contains_liga_id_select(self, client):
        """GET /inicio must render a <select> element with id='liga_id'.

        Validates: Requirements 1.1
        """
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        assert response.status_code == 200
        body = response.data.decode("utf-8")
        assert 'id="liga_id"' in body

    def test_league_selector_populated_with_options(self, client):
        """League selector options must be populated from the mocked API response.

        Validates: Requirements 1.1
        """
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert "Premier League" in body
        assert "La Liga" in body


# ---------------------------------------------------------------------------
# Test 2: Response contains two disabled team selector dropdowns
# Validates: Requirements 1.3, 1.4
# ---------------------------------------------------------------------------

class TestDisabledTeamDropdowns:
    def test_local_team_dropdown_present(self, client):
        """GET /inicio must render a dropdown with id='local_team_id'.

        Validates: Requirements 1.4
        """
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert 'id="local_team_id"' in body

    def test_visitante_team_dropdown_present(self, client):
        """GET /inicio must render a dropdown with id='visitante_team_id'.

        Validates: Requirements 1.4
        """
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert 'id="visitante_team_id"' in body

    def test_local_team_dropdown_is_disabled(self, client):
        """The local_team_id dropdown must have the disabled attribute on page load.

        Validates: Requirements 1.3
        """
        import re
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        select_match = re.search(
            r'<select[^>]+id="local_team_id"[^>]*>', body
        )
        assert select_match is not None, "local_team_id select element not found"
        assert "disabled" in select_match.group(0), (
            "local_team_id select must have 'disabled' attribute before a league is selected"
        )

    def test_visitante_team_dropdown_is_disabled(self, client):
        """The visitante_team_id dropdown must have the disabled attribute on page load.

        Validates: Requirements 1.3
        """
        import re
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        select_match = re.search(
            r'<select[^>]+id="visitante_team_id"[^>]*>', body
        )
        assert select_match is not None, "visitante_team_id select element not found"
        assert "disabled" in select_match.group(0), (
            "visitante_team_id select must have 'disabled' attribute before a league is selected"
        )


# ---------------------------------------------------------------------------
# Test 3: Response contains a link to /partidos
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------

class TestLinkToPartidos:
    def test_response_contains_link_to_partidos(self, client):
        """GET /inicio must render an anchor element pointing to /partidos.

        Validates: Requirements 5.6
        """
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert 'href="/partidos"' in body


# ---------------------------------------------------------------------------
# Test 4: Graceful degradation — API failure passes empty list
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

class TestLigasApiFails:
    def test_inicio_returns_200_when_api_raises_exception(self, client):
        """GET /inicio must return HTTP 200 even when obtener_ligas raises an exception.

        Validates: Requirements 1.5
        """
        with patch("app.obtener_ligas", side_effect=Exception("API error")):
            response = client.get("/inicio")
        assert response.status_code == 200

    def test_inicio_still_renders_liga_id_when_api_fails(self, client):
        """The league selector must still be present even when obtener_ligas fails.

        Validates: Requirements 1.5
        """
        with patch("app.obtener_ligas", side_effect=Exception("API error")):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert 'id="liga_id"' in body

    def test_inicio_returns_200_when_ligas_is_empty(self, client):
        """GET /inicio must return HTTP 200 when obtener_ligas returns an empty response list.

        Validates: Requirements 1.5
        """
        with patch("app.obtener_ligas", return_value={"response": []}):
            response = client.get("/inicio")
        assert response.status_code == 200
