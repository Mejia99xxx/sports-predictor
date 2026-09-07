"""
Unit tests for ml_model.py — task 2.2
Covers: cargar_modelo (FileNotFoundError), predecir (ValueError + return values)
Requirements: 3.4, 3.5, 3.6
"""
import os
import pytest
import numpy as np
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Add backend dir to path so imports resolve
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model import cargar_modelo, predecir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trained_model():
    """Return a minimal trained (modelo, scaler) pair for testing predecir."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 100, size=(90, 9)).tolist()
    # Three balanced classes: 1, -1, 0
    y = [1, -1, 0] * 30

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = RandomForestClassifier(n_estimators=10, random_state=42)
    modelo.fit(X_scaled, y)
    return modelo, scaler


# ---------------------------------------------------------------------------
# cargar_modelo — FileNotFoundError
# ---------------------------------------------------------------------------

class TestCargarModelo:
    def test_raises_file_not_found_when_file_missing(self, tmp_path):
        """cargar_modelo debe lanzar FileNotFoundError si el .pkl no existe."""
        ruta_inexistente = str(tmp_path / "no_existe.pkl")
        with pytest.raises(FileNotFoundError) as exc_info:
            cargar_modelo(ruta=ruta_inexistente)
        # El mensaje debe incluir la ruta
        assert ruta_inexistente in str(exc_info.value)

    def test_error_message_is_descriptive(self, tmp_path):
        """El mensaje de FileNotFoundError debe ser descriptivo."""
        ruta = str(tmp_path / "missing_model.pkl")
        with pytest.raises(FileNotFoundError) as exc_info:
            cargar_modelo(ruta=ruta)
        msg = str(exc_info.value)
        assert len(msg) > 20  # mensaje no vacío ni trivial

    def test_loads_successfully_when_file_exists(self, tmp_path):
        """cargar_modelo retorna tupla (modelo, scaler) si el archivo existe."""
        import pickle
        modelo, scaler = _make_trained_model()
        ruta = str(tmp_path / "modelo_test.pkl")
        with open(ruta, "wb") as f:
            pickle.dump((modelo, scaler), f)

        resultado = cargar_modelo(ruta=ruta)
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2


# ---------------------------------------------------------------------------
# predecir — ValueError on bad input
# ---------------------------------------------------------------------------

class TestPredecir:
    def setup_method(self):
        self.modelo, self.scaler = _make_trained_model()

    def test_raises_value_error_when_entrada_is_not_list_or_tuple(self):
        """predecir debe lanzar ValueError si entrada no es list/tuple."""
        with pytest.raises(ValueError):
            predecir(self.modelo, self.scaler, "cadena_invalida")

    def test_raises_value_error_when_entrada_has_wrong_length_short(self):
        """predecir debe lanzar ValueError si entrada tiene menos de 9 elementos."""
        with pytest.raises(ValueError) as exc_info:
            predecir(self.modelo, self.scaler, [1.0, 2.0, 3.0])
        assert "9" in str(exc_info.value)

    def test_raises_value_error_when_entrada_has_wrong_length_long(self):
        """predecir debe lanzar ValueError si entrada tiene más de 9 elementos."""
        with pytest.raises(ValueError) as exc_info:
            predecir(self.modelo, self.scaler, list(range(10)))
        assert "9" in str(exc_info.value)

    def test_raises_value_error_when_entrada_is_none(self):
        """predecir debe lanzar ValueError si entrada es None."""
        with pytest.raises((ValueError, TypeError)):
            predecir(self.modelo, self.scaler, None)

    def test_error_message_contains_received_type_and_length(self):
        """El mensaje de ValueError debe mencionar el tipo y longitud recibidos."""
        entrada_corta = [1.0, 2.0]
        with pytest.raises(ValueError) as exc_info:
            predecir(self.modelo, self.scaler, entrada_corta)
        msg = str(exc_info.value)
        assert "2" in msg  # longitud recibida

    # --- Return value tests ---

    def test_returns_one_of_three_valid_strings_with_list(self):
        """predecir con lista de 9 floats retorna uno de los tres strings válidos."""
        entrada = [50.0, 50.0, 5.0, 5.0, 10.0, 10.0, 1.0, 1.0, 0.0]
        resultado = predecir(self.modelo, self.scaler, entrada)
        assert resultado in ("Gana equipo local", "Gana equipo visitante", "Empate")

    def test_returns_one_of_three_valid_strings_with_tuple(self):
        """predecir acepta tupla de 9 elementos."""
        entrada = tuple([50.0, 50.0, 5.0, 5.0, 10.0, 10.0, 1.0, 1.0, 0.0])
        resultado = predecir(self.modelo, self.scaler, entrada)
        assert resultado in ("Gana equipo local", "Gana equipo visitante", "Empate")

    def test_returns_string_not_integer(self):
        """predecir retorna str, no int ni otro tipo."""
        entrada = [60.0, 40.0, 8.0, 4.0, 12.0, 8.0, 2.0, 1.0, 1.0]
        resultado = predecir(self.modelo, self.scaler, entrada)
        assert isinstance(resultado, str)

    def test_accepts_zero_vector(self):
        """predecir acepta vector de nueve ceros sin error."""
        entrada = [0.0] * 9
        resultado = predecir(self.modelo, self.scaler, entrada)
        assert resultado in ("Gana equipo local", "Gana equipo visitante", "Empate")
