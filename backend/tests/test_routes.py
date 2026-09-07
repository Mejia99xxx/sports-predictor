"""
Unit tests for Flask routes — Task 4.1
Covers: POST /predecir/equipos
Requirements: 4.1, 4.2, 4.4, 4.5, 3.3
"""
import os
import sys

# Add backend/ to path so imports resolve when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
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


def _make_mock_stats():
    """Return a minimal non-empty stats list that satisfies ensamblar_vector_prediccion."""
    return [
        {
            "team": {"id": 1, "name": "Team A"},
            "statistics": [
                {"type": "Ball Possession", "value": 55},
                {"type": "Total Shots", "value": 10},
                {"type": "Fouls", "value": 12},
                {"type": "Yellow Cards", "value": 2},
                {"type": "Goals", "value": 2},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Test 1: Missing local_team_id → HTTP 400 with parameter name in message
# ---------------------------------------------------------------------------

class TestMissingLocalTeamId:
    def test_missing_local_team_id_returns_400(self, client):
        """POST without local_team_id must return HTTP 400."""
        response = client.post(
            "/predecir/equipos",
            data={"visitante_team_id": "42"},
        )
        assert response.status_code == 400

    def test_missing_local_team_id_message_identifies_param(self, client):
        """Response body must mention 'local_team_id' so the caller knows which param is missing."""
        response = client.post(
            "/predecir/equipos",
            data={"visitante_team_id": "42"},
        )
        assert b"local_team_id" in response.data


# ---------------------------------------------------------------------------
# Test 2: Missing visitante_team_id → HTTP 400 with parameter name in message
# ---------------------------------------------------------------------------

class TestMissingVisitanteTeamId:
    def test_missing_visitante_team_id_returns_400(self, client):
        """POST without visitante_team_id must return HTTP 400."""
        response = client.post(
            "/predecir/equipos",
            data={"local_team_id": "33"},
        )
        assert response.status_code == 400

    def test_missing_visitante_team_id_message_identifies_param(self, client):
        """Response body must mention 'visitante_team_id'."""
        response = client.post(
            "/predecir/equipos",
            data={"local_team_id": "33"},
        )
        assert b"visitante_team_id" in response.data


# ---------------------------------------------------------------------------
# Test 3: Same team ID on both sides → HTTP 400
# ---------------------------------------------------------------------------

class TestSameTeamId:
    def test_same_team_id_returns_400(self, client):
        """When local and visitante IDs are identical, return HTTP 400."""
        response = client.post(
            "/predecir/equipos",
            data={"local_team_id": "33", "visitante_team_id": "33"},
        )
        assert response.status_code == 400

    def test_same_team_id_message_content(self, client):
        """Response body should indicate teams must be different."""
        response = client.post(
            "/predecir/equipos",
            data={"local_team_id": "33", "visitante_team_id": "33"},
        )
        assert "diferentes" in response.data.decode("utf-8")


# ---------------------------------------------------------------------------
# Test 4: Model unavailable (modelo = None) → HTTP 503
# ---------------------------------------------------------------------------

class TestModelUnavailable:
    def test_model_none_returns_503(self, client):
        """When modelo is None the endpoint must return HTTP 503."""
        original_modelo = app_module.modelo
        original_scaler = app_module.scaler
        try:
            app_module.modelo = None
            app_module.scaler = None
            response = client.post(
                "/predecir/equipos",
                data={"local_team_id": "33", "visitante_team_id": "42"},
            )
            assert response.status_code == 503
        finally:
            app_module.modelo = original_modelo
            app_module.scaler = original_scaler

    def test_model_none_503_message(self, client):
        """503 response must mention the model is unavailable."""
        original_modelo = app_module.modelo
        original_scaler = app_module.scaler
        try:
            app_module.modelo = None
            app_module.scaler = None
            response = client.post(
                "/predecir/equipos",
                data={"local_team_id": "33", "visitante_team_id": "42"},
            )
            assert "modelo" in response.data.decode("utf-8").lower()
        finally:
            app_module.modelo = original_modelo
            app_module.scaler = original_scaler


# ---------------------------------------------------------------------------
# Test 5: obtener_ultimo_partido_stats returns [] for local team → HTTP 400
# ---------------------------------------------------------------------------

class TestNoStatsForLocalTeam:
    def test_empty_stats_for_local_team_returns_400(self, client):
        """When the local team has no recent stats, return HTTP 400."""
        # Ensure a model is available so we don't hit the 503 branch first
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test stats error branch")

        with patch("app.obtener_ultimo_partido_stats") as mock_stats:
            mock_stats.return_value = []  # both calls return []
            response = client.post(
                "/predecir/equipos",
                data={
                    "local_team_id": "33",
                    "visitante_team_id": "42",
                    "local_team_name": "Manchester United",
                    "visitante_team_name": "Arsenal",
                },
            )
        assert response.status_code == 400

    def test_empty_stats_for_local_team_message_includes_team_name(self, client):
        """400 message for missing local stats must include the team name."""
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test stats error branch")

        with patch("app.obtener_ultimo_partido_stats") as mock_stats:
            mock_stats.return_value = []
            response = client.post(
                "/predecir/equipos",
                data={
                    "local_team_id": "33",
                    "visitante_team_id": "42",
                    "local_team_name": "Manchester United",
                    "visitante_team_name": "Arsenal",
                },
            )
        assert "Manchester United" in response.data.decode("utf-8")


# ---------------------------------------------------------------------------
# Test 6: Full happy path → HTTP 200 + resultado.html rendered
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_happy_path_returns_200(self, client):
        """A fully valid request with mocked services must return HTTP 200."""
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test happy path")

        mock_stats = _make_mock_stats()

        with patch("app.obtener_ultimo_partido_stats", return_value=mock_stats), \
             patch("app.ensamblar_vector_prediccion", return_value=[55, 45, 10, 8, 12, 14, 2, 1, 1.0]), \
             patch("app.predecir_modelo", return_value="Gana equipo local"):
            response = client.post(
                "/predecir/equipos",
                data={
                    "local_team_id": "33",
                    "visitante_team_id": "42",
                    "local_team_name": "Manchester United",
                    "visitante_team_name": "Arsenal",
                },
            )

        assert response.status_code == 200

    def test_happy_path_renders_resultado_template(self, client):
        """resultado.html must be rendered — check for the result heading."""
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test happy path")

        mock_stats = _make_mock_stats()

        with patch("app.obtener_ultimo_partido_stats", return_value=mock_stats), \
             patch("app.ensamblar_vector_prediccion", return_value=[55, 45, 10, 8, 12, 14, 2, 1, 1.0]), \
             patch("app.predecir_modelo", return_value="Gana equipo local"):
            response = client.post(
                "/predecir/equipos",
                data={
                    "local_team_id": "33",
                    "visitante_team_id": "42",
                    "local_team_name": "Manchester United",
                    "visitante_team_name": "Arsenal",
                },
            )

        body = response.data.decode("utf-8")
        # resultado.html contains a heading with the result
        assert "Resultado del Pronóstico" in body
        assert "Gana equipo local" in body

    def test_happy_path_includes_team_names_in_response(self, client):
        """Both team names must appear in the rendered response."""
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test happy path")

        mock_stats = _make_mock_stats()

        with patch("app.obtener_ultimo_partido_stats", return_value=mock_stats), \
             patch("app.ensamblar_vector_prediccion", return_value=[55, 45, 10, 8, 12, 14, 2, 1, 1.0]), \
             patch("app.predecir_modelo", return_value="Empate"):
            response = client.post(
                "/predecir/equipos",
                data={
                    "local_team_id": "33",
                    "visitante_team_id": "42",
                    "local_team_name": "Manchester United",
                    "visitante_team_name": "Arsenal",
                },
            )

        body = response.data.decode("utf-8")
        assert "Manchester United" in body
        assert "Arsenal" in body


# ---------------------------------------------------------------------------
# Test 9: Regression smoke test for existing /predecir manual flow
# ---------------------------------------------------------------------------

class TestPredecirManualRegression:
    """
    Regression smoke test confirming that the existing /predecir?modo=manual
    flow still returns HTTP 200 with a result after the team-based feature
    was added.

    Validates: Requirements 4.3
    """

    _VALID_MANUAL_DATA = {
        "modo": "manual",
        "posesion_local": "55",
        "posesion_visitante": "45",
        "tiros_local": "10",
        "tiros_visitante": "8",
        "faltas_local": "12",
        "faltas_visitante": "14",
        "tarjetas_local": "2",
        "tarjetas_visitante": "1",
    }

    def test_predecir_manual_returns_200(self, client):
        """POST /predecir with modo=manual and valid numeric fields must return HTTP 200."""
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test manual prediction flow")

        response = client.post("/predecir", data=self._VALID_MANUAL_DATA)
        assert response.status_code == 200

    def test_predecir_manual_returns_result_in_body(self, client):
        """Response body must contain a prediction result string."""
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test manual prediction flow")

        response = client.post("/predecir", data=self._VALID_MANUAL_DATA)
        body = response.data.decode("utf-8")
        valid_outcomes = ["Gana equipo local", "Gana equipo visitante", "Empate"]
        assert any(outcome in body for outcome in valid_outcomes), (
            f"Expected one of {valid_outcomes} in response body, got: {body[:200]}"
        )

    def test_predecir_manual_renders_resultado_template(self, client):
        """resultado.html must be rendered — the result heading must appear."""
        if app_module.modelo is None:
            pytest.skip("modelo not loaded — cannot test manual prediction flow")

        response = client.post("/predecir", data=self._VALID_MANUAL_DATA)
        body = response.data.decode("utf-8")
        assert "Resultado del Pronóstico" in body
