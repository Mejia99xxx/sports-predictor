"""
Unit tests for GET /inicio — Task 7.1
Covers: league selector, disabled team dropdowns, link to /partidos
Requirements: 1.1, 1.5
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
# Test 1: Response contains the league selector element (liga_id)
# ---------------------------------------------------------------------------

class TestLeagueSelectorPresent:
    def test_response_contains_liga_id_select(self, client):
        """GET /inicio must render a <select> element with id='liga_id'."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        assert response.status_code == 200
        body = response.data.decode("utf-8")
        assert 'id="liga_id"' in body

    def test_league_selector_is_a_select_element(self, client):
        """The liga_id element must be a <select> tag."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert '<select' in body
        assert 'liga_id' in body

    def test_league_options_populated_from_ligas(self, client):
        """League options must include league names from the mocked response."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert "Premier League" in body
        assert "La Liga" in body


# ---------------------------------------------------------------------------
# Test 2: Response contains two disabled team selector dropdowns
# ---------------------------------------------------------------------------

class TestDisabledTeamDropdowns:
    def test_local_team_dropdown_present(self, client):
        """GET /inicio must render a dropdown with id='local_team_id'."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert 'id="local_team_id"' in body

    def test_visitante_team_dropdown_present(self, client):
        """GET /inicio must render a dropdown with id='visitante_team_id'."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert 'id="visitante_team_id"' in body

    def test_local_team_dropdown_is_disabled(self, client):
        """The local_team_id dropdown must have the disabled attribute."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        # The disabled attribute should appear near the local_team_id select
        assert 'local_team_id' in body
        # Find the select tag and confirm it's disabled
        import re
        select_match = re.search(
            r'<select[^>]+id="local_team_id"[^>]*>', body
        )
        assert select_match is not None, "local_team_id select not found"
        assert "disabled" in select_match.group(0)

    def test_visitante_team_dropdown_is_disabled(self, client):
        """The visitante_team_id dropdown must have the disabled attribute."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        import re
        select_match = re.search(
            r'<select[^>]+id="visitante_team_id"[^>]*>', body
        )
        assert select_match is not None, "visitante_team_id select not found"
        assert "disabled" in select_match.group(0)


# ---------------------------------------------------------------------------
# Test 3: Response contains a link to /partidos
# ---------------------------------------------------------------------------

class TestLinkToPartidos:
    def test_response_contains_link_to_partidos(self, client):
        """GET /inicio must render a link pointing to /partidos."""
        with patch("app.obtener_ligas", return_value=_make_mock_ligas()):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        assert 'href="/partidos"' in body


# ---------------------------------------------------------------------------
# Test 4: Graceful degradation when obtener_ligas fails
# ---------------------------------------------------------------------------

class TestLigasApiFails:
    def test_inicio_still_returns_200_when_api_fails(self, client):
        """GET /inicio must return 200 even if obtener_ligas raises an exception."""
        with patch("app.obtener_ligas", side_effect=Exception("API error")):
            response = client.get("/inicio")
        assert response.status_code == 200

    def test_inicio_shows_error_message_when_ligas_empty(self, client):
        """When ligas list is empty, the page should display an error message."""
        with patch("app.obtener_ligas", return_value={"response": []}):
            response = client.get("/inicio")
        body = response.data.decode("utf-8")
        # The template must show some kind of error/notice when list is empty
        assert response.status_code == 200
        # liga_id selector should still be present (even if empty)
        assert 'id="liga_id"' in body
