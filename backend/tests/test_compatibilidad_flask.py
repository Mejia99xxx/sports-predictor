"""
Smoke tests for Flask/ml_model compatibility — task 9.1
Verifies that the ml_model public API contract is intact and usable from Flask.

Requirements: 8.1, 8.3
"""
import sys
import os
import inspect
import pytest

# Make the backend package importable regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelo.pkl")
MODEL_AVAILABLE = os.path.exists(MODEL_PATH)


def _valid_stats_list():
    """Return a minimal stats list accepted by preparar_datos_para_modelo."""
    return [
        {
            "team": {"name": "local"},
            "statistics": [
                {"type": "Ball Possession", "value": 55},
                {"type": "Total Shots", "value": 10},
                {"type": "Fouls", "value": 12},
                {"type": "Yellow Cards", "value": 2},
                {"type": "Goals", "value": 1},
            ],
        },
        {
            "team": {"name": "visitante"},
            "statistics": [
                {"type": "Ball Possession", "value": 45},
                {"type": "Total Shots", "value": 8},
                {"type": "Fouls", "value": 14},
                {"type": "Yellow Cards", "value": 1},
                {"type": "Goals", "value": 0},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 1. cargar_modelo — existence and signature
# ---------------------------------------------------------------------------

class TestCargarModeloContract:
    def test_cargar_modelo_exists(self):
        """cargar_modelo debe existir en ml_model."""
        assert hasattr(ml_model, "cargar_modelo"), "ml_model.cargar_modelo no existe"

    def test_cargar_modelo_is_callable(self):
        """cargar_modelo debe ser llamable."""
        assert callable(ml_model.cargar_modelo)

    def test_cargar_modelo_accepts_ruta_argument(self):
        """cargar_modelo debe aceptar al menos un argumento posicional (ruta)."""
        sig = inspect.signature(ml_model.cargar_modelo)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "cargar_modelo debe aceptar al menos un argumento (ruta)"
        # The first parameter should be named 'ruta' or be positional-compatible
        assert "ruta" in params, f"El primer parámetro de cargar_modelo debe llamarse 'ruta', se encontró: {params}"


# ---------------------------------------------------------------------------
# 2. guardar_modelo — existence and signature
# ---------------------------------------------------------------------------

class TestGuardarModeloContract:
    def test_guardar_modelo_exists(self):
        """guardar_modelo debe existir en ml_model."""
        assert hasattr(ml_model, "guardar_modelo"), "ml_model.guardar_modelo no existe"

    def test_guardar_modelo_is_callable(self):
        """guardar_modelo debe ser llamable."""
        assert callable(ml_model.guardar_modelo)

    def test_guardar_modelo_accepts_modelo_scaler_ruta(self):
        """guardar_modelo debe aceptar (modelo, scaler, ruta)."""
        sig = inspect.signature(ml_model.guardar_modelo)
        params = list(sig.parameters.keys())
        assert len(params) >= 3, (
            f"guardar_modelo debe aceptar al menos 3 parámetros (modelo, scaler, ruta), "
            f"se encontraron: {params}"
        )
        assert "modelo" in params, f"Falta el parámetro 'modelo': {params}"
        assert "scaler" in params, f"Falta el parámetro 'scaler': {params}"
        assert "ruta" in params, f"Falta el parámetro 'ruta': {params}"


# ---------------------------------------------------------------------------
# 3. predecir — existence and signature
# ---------------------------------------------------------------------------

class TestPredecirContract:
    def test_predecir_exists(self):
        """predecir debe existir en ml_model."""
        assert hasattr(ml_model, "predecir"), "ml_model.predecir no existe"

    def test_predecir_is_callable(self):
        """predecir debe ser llamable."""
        assert callable(ml_model.predecir)

    def test_predecir_accepts_modelo_scaler_entrada(self):
        """predecir debe aceptar (modelo, scaler, entrada) con entrada de 9 números."""
        sig = inspect.signature(ml_model.predecir)
        params = list(sig.parameters.keys())
        assert len(params) >= 3, (
            f"predecir debe aceptar al menos 3 parámetros (modelo, scaler, entrada), "
            f"se encontraron: {params}"
        )
        assert "modelo" in params, f"Falta el parámetro 'modelo': {params}"
        assert "scaler" in params, f"Falta el parámetro 'scaler': {params}"
        assert "entrada" in params, f"Falta el parámetro 'entrada': {params}"


# ---------------------------------------------------------------------------
# 4. preparar_datos_para_modelo — existence and signature
# ---------------------------------------------------------------------------

class TestPrepararDatosContract:
    def test_preparar_datos_para_modelo_exists(self):
        """preparar_datos_para_modelo debe existir en ml_model."""
        assert hasattr(ml_model, "preparar_datos_para_modelo"), (
            "ml_model.preparar_datos_para_modelo no existe"
        )

    def test_preparar_datos_para_modelo_is_callable(self):
        """preparar_datos_para_modelo debe ser llamable."""
        assert callable(ml_model.preparar_datos_para_modelo)

    def test_preparar_datos_para_modelo_accepts_stats(self):
        """preparar_datos_para_modelo debe aceptar un parámetro (stats)."""
        sig = inspect.signature(ml_model.preparar_datos_para_modelo)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, (
            "preparar_datos_para_modelo debe aceptar al menos un parámetro (stats)"
        )

    def test_preparar_datos_para_modelo_returns_list_of_9(self):
        """preparar_datos_para_modelo con stats válido retorna lista de 9 valores numéricos."""
        stats = _valid_stats_list()
        resultado = ml_model.preparar_datos_para_modelo(stats)
        assert isinstance(resultado, list), (
            f"preparar_datos_para_modelo debe retornar una lista, retornó: {type(resultado).__name__}"
        )
        assert len(resultado) == 9, (
            f"preparar_datos_para_modelo debe retornar exactamente 9 valores, retornó: {len(resultado)}"
        )
        for i, val in enumerate(resultado):
            assert isinstance(val, (int, float)), (
                f"El elemento {i} de la lista no es numérico: {val!r} ({type(val).__name__})"
            )


# ---------------------------------------------------------------------------
# 5. cargar_modelo runtime — returns (modelo, scaler) with correct methods
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_AVAILABLE, reason="modelo.pkl no disponible — omitiendo prueba de carga")
class TestCargarModeloRuntime:
    def test_returns_tuple_of_two_elements(self):
        """cargar_modelo(ruta) debe retornar una tupla de exactamente 2 elementos."""
        resultado = ml_model.cargar_modelo(MODEL_PATH)
        assert isinstance(resultado, tuple), (
            f"cargar_modelo debe retornar una tupla, retornó: {type(resultado).__name__}"
        )
        assert len(resultado) == 2, (
            f"La tupla retornada debe tener exactamente 2 elementos, tiene: {len(resultado)}"
        )

    def test_modelo_has_predict_method(self):
        """El primer elemento (modelo) debe tener método .predict()."""
        modelo, _ = ml_model.cargar_modelo(MODEL_PATH)
        assert hasattr(modelo, "predict") and callable(getattr(modelo, "predict")), (
            "El modelo cargado no tiene el método .predict()"
        )

    def test_scaler_has_transform_method(self):
        """El segundo elemento (scaler) debe tener método .transform()."""
        _, scaler = ml_model.cargar_modelo(MODEL_PATH)
        assert hasattr(scaler, "transform") and callable(getattr(scaler, "transform")), (
            "El scaler cargado no tiene el método .transform()"
        )


# ---------------------------------------------------------------------------
# 6. predecir runtime — returns valid prediction string
# ---------------------------------------------------------------------------

VALID_PREDICTIONS = {"Gana equipo local", "Gana equipo visitante", "Empate"}


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="modelo.pkl no disponible — omitiendo prueba de predicción")
class TestPredecirRuntime:
    def test_predecir_returns_valid_string(self):
        """predecir con entrada conocida debe retornar uno de los tres strings válidos."""
        modelo, scaler = ml_model.cargar_modelo(MODEL_PATH)
        entrada = [55, 45, 10, 8, 12, 14, 2, 1, 1.0]
        resultado = ml_model.predecir(modelo, scaler, entrada)
        assert resultado in VALID_PREDICTIONS, (
            f"predecir retornó '{resultado}', se esperaba uno de: {VALID_PREDICTIONS}"
        )

    def test_predecir_returns_string_type(self):
        """predecir debe retornar un str, no int u otro tipo."""
        modelo, scaler = ml_model.cargar_modelo(MODEL_PATH)
        entrada = [55, 45, 10, 8, 12, 14, 2, 1, 1.0]
        resultado = ml_model.predecir(modelo, scaler, entrada)
        assert isinstance(resultado, str), (
            f"predecir debe retornar str, retornó: {type(resultado).__name__}"
        )


# ---------------------------------------------------------------------------
# 7. preparar_datos_para_modelo — runtime output shape and types
# ---------------------------------------------------------------------------

class TestPrepararDatosRuntime:
    def test_returns_exactly_9_numeric_values(self):
        """preparar_datos_para_modelo con stats válido retorna lista de exactamente 9 numéricos."""
        stats = _valid_stats_list()
        resultado = ml_model.preparar_datos_para_modelo(stats)
        assert isinstance(resultado, list), (
            f"Se esperaba lista, se obtuvo: {type(resultado).__name__}"
        )
        assert len(resultado) == 9, (
            f"Se esperaban 9 valores, se obtuvieron: {len(resultado)}"
        )
        for i, val in enumerate(resultado):
            assert isinstance(val, (int, float)), (
                f"El valor en posición {i} no es numérico: {val!r}"
            )
